// =============================================================================
// GHOST ORACLE SUITE — UNIFIED CUDA KERNEL FILE
// =============================================================================
// This file consolidates every CUDA kernel used across the Ghost Oracle Suite.
// Single source of truth — gpu.py and projection_benchmark.py load this one .cu
// via cupy.RawModule.
//
// Sections:
//   1. Shared constants and device helpers
//   2. Rank-K trigonometric matmul (T1 target)
//        ghost_rank_k_matmul
//        ghost_rank_k_matmul_batch
//   3. Bucketed projection (G_M from QPU/GPU base)
//        ghost_projection
//   4. Tied-channel attention (G_M operator, the headline benchmark)
//        tied_streaming_perdim
//        tied_materialize_perdim
//   5. GHZ closed-form sampler (replaces the Python statevector loop in gpu.py)
//        ghost_ghz_sample
//
// Operator legend (referenced throughout this file):
//   T1  = |cos(a - b)|                                 (standard cosine kernel)
//   T3  = mixed-state target produced by the ghost-CNOT'd Hadamard test
//   G_M = sqrt((1 + cos(a) cos(b)) / 2) / alpha        (the headline operator;
//         the noiseless limit of what T3 computes, after the GHZ-block reduction)
// See docs/math.md for derivations.
//
// Notes on the unified design:
//   - clipped_log_pair was duplicated between ghost_projection.cu and the
//     embedded benchmark kernel; here it appears once.
//   - projection_channel and geometry_channel are shared device functions used
//     by both the standalone projection kernel and the tied-channel kernels.
//   - The legacy "projection_4x4_legacy" name from the benchmark is dropped;
//     ghost_projection is the canonical N-matrix projection entry point and
//     handles the 4x4 case as N=1, matrix_size=4.
// =============================================================================

#include <curand_kernel.h>

// =============================================================================
// SECTION 1 — SHARED CONSTANTS AND DEVICE HELPERS
// =============================================================================

#define EPS 0.05f
#define CLIP_LOG_W 3.0f
#define NUM_BUCKETS 3 // a_fire and b_fire each in {0.0, 0.5, 1.0}

// Clipped log of P(measure=1) and P(measure=0) at angle theta.
// p = sin^2(theta/2), clipped to [EPS, 1-EPS] for numerical stability.
__device__ inline void clipped_log_pair(float theta, float *log_p, float *log_1mp)
{
    float s = __sinf(0.5f * theta);
    float p = s * s;
    if (p < EPS)
        p = EPS;
    if (p > 1.0f - EPS)
        p = 1.0f - EPS;
    *log_p = __logf(p);
    *log_1mp = __logf(1.0f - p);
}

// Importance-weighted reweighting of the bucketed shot counts (counts18 = 18
// ints in a_bucket-major order: counts[a_b, b_b, ctrl]) under the change-of-
// angle (orig_a, orig_b) -> (na, nb). Returns G_M(na, nb) = sqrt((1+...)/2)/α.
__device__ inline float projection_channel(
    const int *__restrict__ counts18,
    float orig_a, float orig_b,
    float na, float nb,
    float alpha_norm)
{
    float log_pa_o, log_1mpa_o, log_pb_o, log_1mpb_o;
    float log_pa_n, log_1mpa_n, log_pb_n, log_1mpb_n;
    clipped_log_pair(orig_a, &log_pa_o, &log_1mpa_o);
    clipped_log_pair(orig_b, &log_pb_o, &log_1mpb_o);
    clipped_log_pair(na, &log_pa_n, &log_1mpa_n);
    clipped_log_pair(nb, &log_pb_n, &log_1mpb_n);

    // Per-shot log-weight for firing value f in {0, 0.5, 1}:
    //   lw(f) = f * (log_pn - log_po) + (1-f) * (log_1mpn - log_1mpo)
    //         = base + f * slope
    float a_base = log_1mpa_n - log_1mpa_o;
    float a_slope = (log_pa_n - log_pa_o) - a_base;
    float b_base = log_1mpb_n - log_1mpb_o;
    float b_slope = (log_pb_n - log_pb_o) - b_base;

    float w_sum = 0.0f, w0_sum = 0.0f;

#pragma unroll
    for (int a_b = 0; a_b < NUM_BUCKETS; a_b++)
    {
        float fa = 0.5f * (float)a_b; // {0.0, 0.5, 1.0}
        float lw_a = a_base + fa * a_slope;
#pragma unroll
        for (int b_b = 0; b_b < NUM_BUCKETS; b_b++)
        {
            float fb = 0.5f * (float)b_b;
            float lw = lw_a + b_base + fb * b_slope;
            if (lw > CLIP_LOG_W)
                lw = CLIP_LOG_W;
            if (lw < -CLIP_LOG_W)
                lw = -CLIP_LOG_W;
            float w = __expf(lw);
            int off = (a_b * NUM_BUCKETS + b_b) * 2;
            int n_zero = counts18[off + 0];
            int n_one = counts18[off + 1];
            w_sum += w * (float)(n_zero + n_one);
            w0_sum += w * (float)n_zero;
        }
    }

    float p0 = (w_sum > 1e-12f) ? (w0_sum / w_sum) : 0.5f;
    float raw = 2.0f * p0 - 1.0f;
    if (raw < 0.0f)
        raw = 0.0f;
    float val = sqrtf(raw) / alpha_norm;
    if (val > 1.0f)
        val = 1.0f;
    return val;
}

// Analytical closed form of G_M(na, nb) = sqrt((1 + cos(na)*cos(nb))/2) / α.
// This is what the QPU is computing in the noiseless limit; the projection
// channel above estimates it from physical shot counts.
__device__ inline float geometry_channel(float na, float nb, float alpha_norm)
{
    float cosab = __cosf(na) * __cosf(nb);
    float raw = (1.0f + cosab) * 0.5f;
    if (raw < 0.0f)
        raw = 0.0f;
    float val = sqrtf(raw) / alpha_norm;
    if (val > 1.0f)
        val = 1.0f;
    return val;
}

// =============================================================================
// SECTION 2 — RANK-K TRIGONOMETRIC MATMUL (T1 TARGET)
// =============================================================================
// Computes C[r, c] = | sum_{k=0..K-1} cos(theta_a[r, k] - theta_b[c, k]) |
// At K=1 this is T1(a, b) = |cos(a-b)|, what standard cuBLAS produces from
// the [cos | sin] lifted representation via cos(a-b) = cos a cos b + sin a sin b.
// =============================================================================

extern "C" __global__ void ghost_rank_k_matmul(
    float *__restrict__ out,
    const float *__restrict__ theta_a,
    const float *__restrict__ theta_b,
    int N,
    int K)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < N && c < N)
    {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
        {
            float a = theta_a[r * K + k];
            float b = theta_b[c * K + k];
            acc += __cosf(a - b);
        }
        out[r * N + c] = acc;
    }
}

// Batched variant: one block per matrix in the batch; each block computes
// its own NxN output. theta_a is (n_matrices, N, K), theta_b similarly.
extern "C" __global__ void ghost_rank_k_matmul_batch(
    float *__restrict__ out,           // (n_matrices, N, N)
    const float *__restrict__ theta_a, // (n_matrices, N, K)
    const float *__restrict__ theta_b, // (n_matrices, N, K)
    int N,
    int K,
    int n_matrices)
{
    int matrix_idx = blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    if (matrix_idx >= n_matrices || r >= N || c >= N)
        return;

    int a_base = matrix_idx * N * K;
    int b_base = matrix_idx * N * K;
    int o_base = matrix_idx * N * N;

    float acc = 0.0f;
    for (int k = 0; k < K; k++)
    {
        float a = theta_a[a_base + r * K + k];
        float b = theta_b[b_base + c * K + k];
        acc += __cosf(a - b);
    }
    out[o_base + r * N + c] = acc;
}

// =============================================================================
// SECTION 3 — BUCKETED PROJECTION (G_M FROM SHOT-COUNT BASE)
// =============================================================================
// Exploits the fact that per-shot weights depend only on (a_fire, b_fire),
// both of which take values in {0.0, 0.5, 1.0} (averages of two binary
// ancilla bits). So per tile there are only 9 (a_bucket, b_bucket)
// combinations × 2 ctrl values = 18 distinct counts. The base is collapsed
// to 18 ints per tile once when loaded, and this kernel never sees the
// shot-level data.
//
// Bucketing avoids one global-memory read per shot: per-tile reads scale with
// the 18 buckets instead of the shot count, which is the dominant savings at
// 4096 shots.
//
// Inputs:
//   counts        : (num_tiles, 3, 3, 2) int32
//   tile_orig_a   : (num_tiles,)        float32   prepared a angle per tile
//   tile_orig_b   : (num_tiles,)        float32   prepared b angle per tile
//   tile_r, tile_c: (num_tiles,)        int32     (r, c) index per tile
//   new_a         : (n_matrices, matrix_size) float32
//   new_b         : (n_matrices, matrix_size) float32
//
// Output:
//   out           : (n_matrices, matrix_size, matrix_size) float32
//
// Grid : gridDim.x = num_tiles, gridDim.y = n_matrices
// Block: (1, 1) — one thread per (matrix, tile); work is small and independent.
// =============================================================================

extern "C" __global__ void ghost_projection(
    const int *__restrict__ counts_all,    // (T, 3, 3, 2) — all tiles, full bucket counts
    const float *__restrict__ tile_orig_a, // (T,)
    const float *__restrict__ tile_orig_b, // (T,)
    const int *__restrict__ tile_r,        // (T,)
    const int *__restrict__ tile_c,        // (T,)
    const float *__restrict__ new_a,       // (N, matrix_size)
    const float *__restrict__ new_b,       // (N, matrix_size)
    float *__restrict__ out,               // (N, matrix_size, matrix_size)
    int num_tiles,
    int n_matrices,
    int matrix_size,
    float alpha_norm)
{
    int tile_idx = blockIdx.x;
    int matrix_idx = blockIdx.y;
    if (matrix_idx >= n_matrices || tile_idx >= num_tiles)
        return;

    int r = tile_r[tile_idx];
    int c = tile_c[tile_idx];

    float oa = tile_orig_a[tile_idx];
    float ob = tile_orig_b[tile_idx];
    float na = new_a[matrix_idx * matrix_size + r];
    float nb = new_b[matrix_idx * matrix_size + c];

    int counts_base = tile_idx * NUM_BUCKETS * NUM_BUCKETS * 2;
    float val = projection_channel(counts_all + counts_base, oa, ob, na, nb, alpha_norm);

    out[matrix_idx * matrix_size * matrix_size + r * matrix_size + c] = val;
}

// =============================================================================
// SECTION 4 — TIED-CHANNEL ATTENTION (G_M OPERATOR, HEADLINE BENCHMARK)
// =============================================================================
// Per-dim G_M aggregation under two parallel channels.
// Updated with FlashAttention-style 2D shared memory tiling to eliminate VRAM
// bandwidth bottlenecks and stack overflows at massive dimension sizes.

#define TILE_M 32
#define TILE_D 64

extern "C" __global__ void tied_streaming_perdim(
    const int *__restrict__ counts18,
    float orig_a, float orig_b,
    const float *__restrict__ theta_Q,
    const float *__restrict__ theta_K,
    int *__restrict__ out_idx,
    float *__restrict__ out_score,
    float *__restrict__ out_proj_score,
    float *__restrict__ out_agreement,
    int N, int M, int d,
    float alpha_norm)
{
    __shared__ int s_counts[18];
    __shared__ float s_K[TILE_M][TILE_D]; // 8 KB chunk

    int tid = threadIdx.x;
    if (tid < 18)
    {
        s_counts[tid] = counts18[tid];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float best_geom = -1.0f;
    int best_j = -1;
    float best_proj = 0.0f;
    float agreement_sum = 0.0f;
    float inv_d = 1.0f / (float)d;

    // Small bounded register array for the query slice
    float local_q[TILE_D];

    int num_m_chunks = (M + TILE_M - 1) / TILE_M;
    int num_d_chunks = (d + TILE_D - 1) / TILE_D;

    // Outer loop: Stream through Keys in blocks of TILE_M
    for (int m_c = 0; m_c < num_m_chunks; ++m_c)
    {
        int j_start = m_c * TILE_M;
        int valid_keys = (m_c == num_m_chunks - 1) ? (M - j_start) : TILE_M;

        // Accumulate partial sums for the current block of Keys
        float geom_sum_arr[TILE_M] = {0.0f};
        float proj_sum_arr[TILE_M] = {0.0f};
        float diff_sum_arr[TILE_M] = {0.0f};

        // Inner loop: Stream through Dimensions in blocks of TILE_D
        for (int d_c = 0; d_c < num_d_chunks; ++d_c)
        {
            int k_start = d_c * TILE_D;
            int valid_dims = (d_c == num_d_chunks - 1) ? (d - k_start) : TILE_D;

            // 1. Thread 'i' loads its own Query slice into fast registers
            if (i < N)
            {
                for (int k = 0; k < valid_dims; ++k)
                {
                    local_q[k] = theta_Q[i * d + k_start + k];
                }
            }

            // 2. Collaboratively load the Key block slice into Shared Memory
            int elements_to_load = valid_keys * valid_dims;
            for (int idx = threadIdx.x; idx < elements_to_load; idx += blockDim.x)
            {
                int row = idx / valid_dims;
                int col = idx % valid_dims;
                s_K[row][col] = theta_K[(j_start + row) * d + (k_start + col)];
            }
            __syncthreads();

            // 3. Compute partial geometry and projection scores
            if (i < N)
            {
                for (int kj = 0; kj < valid_keys; ++kj)
                {
                    for (int k = 0; k < valid_dims; ++k)
                    {
                        float na = local_q[k];
                        float nb = s_K[kj][k];

                        float g = geometry_channel(na, nb, alpha_norm);
                        float p = projection_channel(s_counts, orig_a, orig_b, na, nb, alpha_norm);

                        geom_sum_arr[kj] += g;
                        proj_sum_arr[kj] += p;
                        diff_sum_arr[kj] += fabsf(g - p);
                    }
                }
            }
            __syncthreads();
        }

        // 4. Resolve the best match within this chunk of M keys
        if (i < N)
        {
            for (int kj = 0; kj < valid_keys; ++kj)
            {
                float g_agg = geom_sum_arr[kj] * inv_d;
                float p_agg = proj_sum_arr[kj] * inv_d;
                float d_agg = diff_sum_arr[kj] * inv_d;

                agreement_sum += d_agg;

                if (g_agg > best_geom)
                {
                    best_geom = g_agg;
                    best_j = j_start + kj;
                    best_proj = p_agg;
                }
            }
        }
    }

    // Write final top-1 retrieval
    if (i < N)
    {
        out_idx[i] = best_j;
        out_score[i] = best_geom;
        out_proj_score[i] = best_proj;
        out_agreement[i] = agreement_sum / (float)M;
    }
}

extern "C" __global__ void tied_materialize_perdim(
    const int *__restrict__ counts18,
    float orig_a, float orig_b,
    const float *__restrict__ theta_Q,
    const float *__restrict__ theta_K,
    float *__restrict__ out_proj,
    float *__restrict__ out_geom,
    int N, int M, int d,
    float alpha_norm)
{
    __shared__ int s_counts[18];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < 18)
    {
        s_counts[tid] = counts18[tid];
    }
    __syncthreads();

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= M)
        return;

    const float *q_row = theta_Q + i * d;
    const float *k_row = theta_K + j * d;

    float geom_sum = 0.0f, proj_sum = 0.0f;
    for (int k = 0; k < d; ++k)
    {
        float na = q_row[k];
        float nb = k_row[k];
        geom_sum += geometry_channel(na, nb, alpha_norm);
        proj_sum += projection_channel(s_counts, orig_a, orig_b, na, nb, alpha_norm);
    }
    float inv_d = 1.0f / (float)d;
    out_geom[i * M + j] = geom_sum * inv_d;
    out_proj[i * M + j] = proj_sum * inv_d;
}

// =============================================================================
// SECTION 5 — GHZ CLOSED-FORM SAMPLER (NOISELESS BASE BUILDER)
// =============================================================================
// Generates shot data for the per-tile circuit defined in qpu.py and gpu.py:
//
//   Per-tile circuit (qubits: a1, v1, a2, ctrl, b1, v2, b2):
//     1. Ry(theta_a) on v1; Ry(theta_b) on v2
//     2. CNOT(v1->a1); CNOT(v1->a2); CNOT(v2->b1); CNOT(v2->b2)
//     3. H on ctrl; CSWAP(ctrl, v1, v2); H on ctrl
//     4. Measure {ctrl, a1, a2, b1, b2}
//
// After the ghost CNOTs, {v1, a1, a2} is in the GHZ state
//   cos(a/2) |000> + sin(a/2) |111>
// and {v2, b1, b2} similarly. Consequence: a1 == a2 and b1 == b2 always.
// So the 32-bin joint distribution P(ctrl, a1, a2, b1, b2) has only 8
// nonzero bins, indexed by (ctrl, a, b) where a in {0,1}, b in {0,1}.
//
// Closed-form probabilities (derived in Probe 4):
//   ca = cos^2(theta_a/2)     sa = sin^2(theta_a/2)
//   cb = cos^2(theta_b/2)     sb = sin^2(theta_b/2)
//
//   <SWAP_v1v2>   = ca*cb + sa*sb
//   P(ctrl=0)     = 0.5 * (1 + <SWAP>)
//   P(ctrl=1)     = 0.5 * (1 - <SWAP>)
//
//   The (a, b) marginals factor independently from ctrl in this circuit
//   because the ghost ancillas commute with the CSWAP+H sequence:
//   P(a=1 | ctrl) = sa,    P(b=1 | ctrl) = sb
//
// So per-shot sampling: draw ctrl from a Bernoulli(P(ctrl=1)), draw
// a from Bernoulli(sa), draw b from Bernoulli(sb), then set
// (a1, a2) = (a, a) and (b1, b2) = (b, b).
//
// One block per tile, blockDim.x threads sample shots in parallel (strided
// over the shot range so n_shots can exceed blockDim.x).
//
// Inputs:
//   angles_a    : (num_tiles,) float32
//   angles_b    : (num_tiles,) float32
//   seed        : uint64  (cuRAND seed; per-shot subsequence drawn from
//                          (tile_idx * n_shots + shot_idx) so output is
//                          reproducible across grid configurations)
//   n_shots     : int     shots per tile
//   num_tiles   : int
//
// Outputs:
//   ctrl_out    : (num_tiles, n_shots)        uint8
//   ghost_out   : (num_tiles, n_shots, 4)     uint8  columns [a1, a2, b1, b2]
//
// Grid : gridDim.x = num_tiles
// Block: blockDim.x = up to 1024; choose based on n_shots
// =============================================================================

extern "C" __global__ void ghost_ghz_sample(
    const float *__restrict__ angles_a,
    const float *__restrict__ angles_b,
    unsigned long long seed,
    int n_shots,
    int num_tiles,
    unsigned char *__restrict__ ctrl_out,
    unsigned char *__restrict__ ghost_out)
{
    int tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles)
        return;

    // Closed-form probabilities for this tile — compute once per block.
    __shared__ float s_p_ctrl1;
    __shared__ float s_p_a1;
    __shared__ float s_p_b1;

    if (threadIdx.x == 0)
    {
        float theta_a = angles_a[tile_idx];
        float theta_b = angles_b[tile_idx];
        float sa = __sinf(0.5f * theta_a);
        sa = sa * sa;
        float sb = __sinf(0.5f * theta_b);
        sb = sb * sb;
        float ca = 1.0f - sa;
        float cb = 1.0f - sb;
        float swap_exp = ca * cb + sa * sb;
        s_p_ctrl1 = 0.5f * (1.0f - swap_exp);
        s_p_a1 = sa;
        s_p_b1 = sb;
    }
    __syncthreads();

    float p_ctrl1 = s_p_ctrl1;
    float p_a1 = s_p_a1;
    float p_b1 = s_p_b1;

    // Each thread handles a strided range of shots in this tile.
    int tile_offset = tile_idx * n_shots;
    int ghost_tile_offset = tile_idx * n_shots * 4;

    for (int shot = threadIdx.x; shot < n_shots; shot += blockDim.x)
    {
        // Per-(tile, shot) cuRAND subsequence: deterministic given (seed, tile, shot).
        curandStatePhilox4_32_10_t state;
        unsigned long long subseq = (unsigned long long)tile_idx * (unsigned long long)n_shots + (unsigned long long)shot;
        curand_init(seed, subseq, 0, &state);

        // Three uniform draws.
        float u_ctrl = curand_uniform(&state);
        float u_a = curand_uniform(&state);
        float u_b = curand_uniform(&state);

        unsigned char ctrl = (u_ctrl < p_ctrl1) ? 1 : 0;
        unsigned char a = (u_a < p_a1) ? 1 : 0;
        unsigned char b = (u_b < p_b1) ? 1 : 0;

        ctrl_out[tile_offset + shot] = ctrl;

        // GHZ: a1 == a2 == a, b1 == b2 == b. Column order [a1, a2, b1, b2].
        int g = ghost_tile_offset + shot * 4;
        ghost_out[g + 0] = a;
        ghost_out[g + 1] = a;
        ghost_out[g + 2] = b;
        ghost_out[g + 3] = b;
    }
}
