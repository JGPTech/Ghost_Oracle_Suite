// =============================================================================
// MEGAKERNELS 2D — TRIG-HOISTED, 32x32 TILED
// =============================================================================
// Appended to ghost_kernel.cu at compile time by the v2 benchmark scripts.
//
// All three kernels share the same load pattern:
//   - Each block handles a 32-query x 32-key tile.
//   - Phase-lift + cos are fused INTO the global->shared transfer:
//       cos(theta(x)) = cos((pi/2)(1 + tanh(x/3)))
//     stored in shared memory as cosines, not angles, not raw embeddings.
//   - Inner loop reads cosines directly. No __cosf, no __sinf in the d-loop.
//
// The half-angle identity that makes the proj fuse safe (verified at machine
// precision):
//
//   p = sin^2(theta/2) = (1 - cos(theta)) / 2
//
// So clipped_log_pair_from_cos(cos_theta) recovers p without sinf:
//
//   p = (1 - cos_theta) * 0.5
//   clip to [EPS, 1-EPS]
//   logs as usual
//
// 2D tiling materializes (N_chunk, M) score matrices. The host wraps with
// query chunking (like ghost_oracle_ai_retrieval_probe_v1.py) to bound VRAM.
//
// All three kernels live here. The streaming variants in the canonical
// ghost_kernel.cu remain for verification probes; production now uses these.
// =============================================================================

#define MK_TILE 32
#define MK_PI_HALF 1.57079632679f

// -----------------------------------------------------------------------------
// Fused phase-lift + cos.
//   theta(x)   = (pi/2) * (1 + tanh(x/3))
//   cos_theta  = cos(theta)
//
// At x=0:   theta=pi/2, cos_theta=0
// At x->inf: theta->pi, cos_theta->-1
// At x->-inf:theta->0,  cos_theta->+1
// -----------------------------------------------------------------------------
__device__ inline float lift_and_cos(float x)
{
    return __cosf(MK_PI_HALF * (1.0f + tanhf(x / 3.0f)));
}

// -----------------------------------------------------------------------------
// clipped_log_pair_from_cos:
//   p = (1 - cos_theta) * 0.5
//   clip to [EPS, 1-EPS]
//   return (log p, log(1-p))
//
// Equivalent to clipped_log_pair(theta) but never computes sin or theta itself.
// -----------------------------------------------------------------------------
__device__ inline void clipped_log_pair_from_cos(float cos_theta,
                                                 float *log_p, float *log_1mp)
{
    float p = (1.0f - cos_theta) * 0.5f;
    if (p < EPS)        p = EPS;
    if (p > 1.0f - EPS) p = 1.0f - EPS;
    *log_p   = __logf(p);
    *log_1mp = __logf(1.0f - p);
}

// -----------------------------------------------------------------------------
// Hoisted projection invariants, computed once per block from origin angles.
// -----------------------------------------------------------------------------
struct MKProjInv {
    float log_pa_o, log_1mpa_o, log_pb_o, log_1mpb_o;
};

__device__ inline void mk_proj_compute_inv(float orig_a, float orig_b,
                                           MKProjInv *inv)
{
    // Origin angles ARE angles here, not cosines (host hands them in directly).
    // So we use the original clipped_log_pair from Section 1.
    clipped_log_pair(orig_a, &inv->log_pa_o, &inv->log_1mpa_o);
    clipped_log_pair(orig_b, &inv->log_pb_o, &inv->log_1mpb_o);
}

// -----------------------------------------------------------------------------
// projection_from_cos: computes G_M_proj(cos_a, cos_b) given hoisted inv
// and pre-computed cosines. Same 9-cell weighted bucket sum as
// projection_channel in Section 1, but with the trig hoist applied to the
// per-(q, k, dim) work.
// -----------------------------------------------------------------------------
__device__ inline float projection_from_cos(
    const int *__restrict__ counts18, const MKProjInv &inv,
    float cos_a, float cos_b, float alpha_norm, int mask_bits)
{
    float log_pa_n, log_1mpa_n, log_pb_n, log_1mpb_n;
    clipped_log_pair_from_cos(cos_a, &log_pa_n, &log_1mpa_n);
    clipped_log_pair_from_cos(cos_b, &log_pb_n, &log_1mpb_n);

    float a_base  = log_1mpa_n - inv.log_1mpa_o;
    float a_slope = (log_pa_n - inv.log_pa_o) - a_base;
    float b_base  = log_1mpb_n - inv.log_1mpb_o;
    float b_slope = (log_pb_n - inv.log_pb_o) - b_base;

    float w_sum = 0.0f, w0_sum = 0.0f;

    #pragma unroll
    for (int a_b = 0; a_b < 3; a_b++)
    {
        float fa = 0.5f * (float)a_b;
        float lw_a = a_base + fa * a_slope;
        #pragma unroll
        for (int b_b = 0; b_b < 3; b_b++)
        {
            int idx = a_b * 3 + b_b;
            if (!((mask_bits >> idx) & 1)) continue;
            float fb = 0.5f * (float)b_b;
            float lw = lw_a + b_base + fb * b_slope;
            // Mask-aware path uses wider clip (matches isolated_kernels.cu).
            if (lw > 18.0f)  lw = 18.0f;
            if (lw < -18.0f) lw = -18.0f;
            float w = __expf(lw);
            int off = idx * 2;
            w_sum  += w * (float)(counts18[off] + counts18[off + 1]);
            w0_sum += w * (float)counts18[off];
        }
    }

    float p0  = (w_sum > 1e-12f) ? (w0_sum / w_sum) : 0.5f;
    float raw = fmaxf(2.0f * p0 - 1.0f, 0.0f);
    return fminf(sqrtf(raw) / alpha_norm, 1.0f);
}

// =============================================================================
// GEO: pure geometry, 2D-tiled, trig-hoisted.
//
// Each block: (32, 32) threads = 1024 threads per block, handling a
// 32-query x 32-key tile. The d dimension is streamed in MK_TILE=32 chunks.
//
// Shared memory per block:
//   s_Q[32][32]   = cos(theta(raw_Q chunk))      4 KB
//   s_K[32][32]   = cos(theta(raw_K chunk))      4 KB
// Total: 8 KB shared, well under the 48 KB default limit.
//
// Writes a full (N_chunk, M) score matrix; host argmax does retrieval.
// =============================================================================
extern "C" __global__ void geo_megakernel_2d(
    const float *__restrict__ raw_Q,
    const float *__restrict__ raw_K,
    float *__restrict__ score_out,
    int N, int M, int d, float alpha_norm)
{
    __shared__ float s_Q[MK_TILE][MK_TILE];
    __shared__ float s_K[MK_TILE][MK_TILE];

    int i = blockIdx.y * blockDim.y + threadIdx.y;   // query row
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // key col

    float sum = 0.0f;

    for (int k_off = 0; k_off < d; k_off += MK_TILE)
    {
        // Each thread (ty, tx) cooperatively loads ONE element of s_Q and ONE
        // element of s_K. The (ty, tx) -> dim mapping is chosen so adjacent
        // threads in a warp hit adjacent memory addresses.

        // s_Q[ty][tx] holds cos(theta(raw_Q[i, k_off + tx])).
        // s_K[tx][ty] holds cos(theta(raw_K[j, k_off + ty])).
        //
        // Why the swapped indexing on s_K: the inner dot loop reads
        // s_Q[ty][k] and s_K[tx][k]; we want each thread's (ty, tx) pair to
        // index naturally. So s_K is "transposed" at load time so the
        // inner loop reads s_K[tx][k] which means s_K's first index is the
        // key column, second is the dim.

        if (i < N && (k_off + threadIdx.x) < d)
            s_Q[threadIdx.y][threadIdx.x] = lift_and_cos(raw_Q[i * d + k_off + threadIdx.x]);
        else
            s_Q[threadIdx.y][threadIdx.x] = 0.0f;

        if (j < M && (k_off + threadIdx.y) < d)
            s_K[threadIdx.x][threadIdx.y] = lift_and_cos(raw_K[j * d + k_off + threadIdx.y]);
        else
            s_K[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        if (i < N && j < M)
        {
            int k_max = (d - k_off < MK_TILE) ? (d - k_off) : MK_TILE;
            #pragma unroll
            for (int k = 0; k < MK_TILE; ++k)
            {
                if (k >= k_max) break;
                // G_M(a, b) per dim = sqrt((1 + cos a cos b)/2)
                float val = 0.5f + 0.5f * s_Q[threadIdx.y][k] * s_K[threadIdx.x][k];
                sum += sqrtf(val > 0.0f ? val : 0.0f);
            }
        }
        __syncthreads();
    }

    if (i < N && j < M)
    {
        float mean = sum / (float)d;
        float final_score = mean / alpha_norm;
        if (final_score > 1.0f) final_score = 1.0f;
        score_out[i * M + j] = final_score;
    }
}

// =============================================================================
// PROJ: projection-only, 2D-tiled, trig-hoisted.
//
// Same 32x32 block geometry as GEO. Shared memory adds counts18 (72 B) and
// MKProjInv (16 B) on top of the two 32x32 cosine tiles. Still ~8 KB total.
//
// The inner loop calls projection_from_cos which does the 9-cell weighted
// bucket sum given cosines. The bucket sum's only theta-dependence is via
// clipped_log_pair_from_cos, which is two adds and two clips per arg.
//
// Threshold handling: kernel writes RAW scores. The host does threshold
// shifting + argmax in a second small pass. Keeps kernel simple and lets
// the same materialized scores serve top-1, top-K, recall@K, and MRR.
// =============================================================================
extern "C" __global__ void proj_megakernel_2d(
    const int   *__restrict__ counts18,
    float orig_a, float orig_b,
    const float *__restrict__ raw_Q,
    const float *__restrict__ raw_K,
    float *__restrict__ score_out,
    int N, int M, int d,
    float alpha_norm, int mask_bits)
{
    __shared__ int       s_counts[18];
    __shared__ MKProjInv s_inv;
    __shared__ float     s_Q[MK_TILE][MK_TILE];
    __shared__ float     s_K[MK_TILE][MK_TILE];

    int tid_flat = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid_flat < 18) s_counts[tid_flat] = counts18[tid_flat];
    if (tid_flat == 0) mk_proj_compute_inv(orig_a, orig_b, &s_inv);
    __syncthreads();

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int k_off = 0; k_off < d; k_off += MK_TILE)
    {
        if (i < N && (k_off + threadIdx.x) < d)
            s_Q[threadIdx.y][threadIdx.x] = lift_and_cos(raw_Q[i * d + k_off + threadIdx.x]);
        else
            s_Q[threadIdx.y][threadIdx.x] = 0.0f;

        if (j < M && (k_off + threadIdx.y) < d)
            s_K[threadIdx.x][threadIdx.y] = lift_and_cos(raw_K[j * d + k_off + threadIdx.y]);
        else
            s_K[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        if (i < N && j < M)
        {
            int k_max = (d - k_off < MK_TILE) ? (d - k_off) : MK_TILE;
            for (int k = 0; k < MK_TILE; ++k)
            {
                if (k >= k_max) break;
                float cos_a = s_Q[threadIdx.y][k];
                float cos_b = s_K[threadIdx.x][k];
                sum += projection_from_cos(s_counts, s_inv,
                                           cos_a, cos_b,
                                           alpha_norm, mask_bits);
            }
        }
        __syncthreads();
    }

    if (i < N && j < M)
    {
        float mean = sum / (float)d;
        if (mean > 1.0f) mean = 1.0f;
        score_out[i * M + j] = mean;
    }
}

// =============================================================================
// TIED: both channels + agreement, 2D-tiled, trig-hoisted.
//
// For verification probes. Writes geom, proj, and per-query agreement.
// Same load pattern; inner loop does both channels and accumulates |g - p|.
// =============================================================================
extern "C" __global__ void tied_megakernel_2d(
    const int   *__restrict__ counts18,
    float orig_a, float orig_b,
    const float *__restrict__ raw_Q,
    const float *__restrict__ raw_K,
    float *__restrict__ geom_out,
    float *__restrict__ proj_out,
    float *__restrict__ agree_partial_out,  // (N, num_M_blocks) partial sums
    int N, int M, int d,
    float alpha_norm, int mask_bits)
{
    __shared__ int       s_counts[18];
    __shared__ MKProjInv s_inv;
    __shared__ float     s_Q[MK_TILE][MK_TILE];
    __shared__ float     s_K[MK_TILE][MK_TILE];

    int tid_flat = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid_flat < 18) s_counts[tid_flat] = counts18[tid_flat];
    if (tid_flat == 0) mk_proj_compute_inv(orig_a, orig_b, &s_inv);
    __syncthreads();

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float g_sum = 0.0f, p_sum = 0.0f;

    for (int k_off = 0; k_off < d; k_off += MK_TILE)
    {
        if (i < N && (k_off + threadIdx.x) < d)
            s_Q[threadIdx.y][threadIdx.x] = lift_and_cos(raw_Q[i * d + k_off + threadIdx.x]);
        else
            s_Q[threadIdx.y][threadIdx.x] = 0.0f;

        if (j < M && (k_off + threadIdx.y) < d)
            s_K[threadIdx.x][threadIdx.y] = lift_and_cos(raw_K[j * d + k_off + threadIdx.y]);
        else
            s_K[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        if (i < N && j < M)
        {
            int k_max = (d - k_off < MK_TILE) ? (d - k_off) : MK_TILE;
            for (int k = 0; k < MK_TILE; ++k)
            {
                if (k >= k_max) break;
                float cos_a = s_Q[threadIdx.y][k];
                float cos_b = s_K[threadIdx.x][k];
                float val = 0.5f + 0.5f * cos_a * cos_b;
                g_sum += sqrtf(val > 0.0f ? val : 0.0f);
                p_sum += projection_from_cos(s_counts, s_inv,
                                             cos_a, cos_b,
                                             alpha_norm, mask_bits);
            }
        }
        __syncthreads();
    }

    if (i < N && j < M)
    {
        float inv_d = 1.0f / (float)d;
        float g_mean = g_sum * inv_d / alpha_norm;
        float p_mean = p_sum * inv_d;
        if (g_mean > 1.0f) g_mean = 1.0f;
        if (p_mean > 1.0f) p_mean = 1.0f;
        geom_out[i * M + j] = g_mean;
        proj_out[i * M + j] = p_mean;
        // Per-block partial agreement sum, reduced host-side.
        // One slot per (query, m_block) so atomics aren't needed.
        atomicAdd(&agree_partial_out[i * gridDim.x + blockIdx.x],
                  fabsf(g_mean - p_mean));
    }
}
