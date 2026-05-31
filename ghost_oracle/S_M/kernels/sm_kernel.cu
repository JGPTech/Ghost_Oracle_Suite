// =============================================================================
// S_M KERNEL - SYNDROME METRIC FIELD FEATURE REDUCTIONS
// =============================================================================
// Single CUDA source for the S_M optimized benchmark path.
//
// Purpose
// -------
// This file accelerates the expensive S_M benchmark inner loop:
//
//   data[shots, d]
//   synd[shots, rounds, d-1]
//        ->
//   windowed S_M features
//
// It computes the same core S_M feature families used by s_m_benchmark.py:
//
//   raw_rates
//   detection_rates
//   agreement_profiles
//   sm_field
//   sm_all
//
// Boundary
// --------
// This is the S_M operator kernel only. It intentionally does NOT compute the
// T_S stress tensor. Stress tensor features belong to the separate T_S operator.
//
// Input layout
// ------------
// data:
//   uint8, shape (shots, d), row-major
//
// synd:
//   uint8, shape (shots, rounds, edges), row-major
//   where edges = d - 1
//
// Output layout
// -------------
// One output row per non-overlapping window.
//
// For each window w, this kernel writes:
//
//   raw_out[w, :]
//       data mean per data bit                         length d
//       syndrome mean per (round, edge)                length rounds * edges
//
//   det_out[w, :]
//       detection-event mean per (round-1, edge)       length (rounds-1) * edges
//
//   agree_prof_out[w, :]
//       agreement mean per edge                        length edges
//       agreement mean per round                       length rounds
//
//   sm_field_out[w, :]
//       agreement field mean per (round, edge)         length rounds * edges
//       detection field mean per (round-1, edge)       length (rounds-1) * edges
//
// Agreement definition
// --------------------
//   E_i      = D_i XOR D_{i+1}
//   A[t, i]  = 1 - (S[t, i] XOR E_i)
//
// Detection definition
// --------------------
//   X[t, i] = S[t+1, i] XOR S[t, i]
//
// Kernel strategy
// ---------------
// One CUDA block computes one window.
// Threads stride through feature coordinates and reduce over the shots inside
// that window. This keeps the implementation simple, deterministic, and easy to
// validate against the reference NumPy benchmark.
//
// Grid
// ----
//   gridDim.x  = n_windows
//   blockDim.x = usually 128 or 256
//
// Notes
// -----
// - Inputs are expected to contain bits, but values are masked with & 1.
// - Windows are non-overlapping: start = window_id * window_size.
// - If shots is not divisible by window_size, trailing shots are ignored.
// - All outputs are float32 rates in [0,1].
// =============================================================================

extern "C" __global__ void sm_window_features_kernel(
    const unsigned char *__restrict__ data,
    const unsigned char *__restrict__ synd,
    int shots,
    int d,
    int rounds,
    int window_size,
    float *__restrict__ raw_out,
    float *__restrict__ det_out,
    float *__restrict__ agree_prof_out,
    float *__restrict__ sm_field_out)
{
    const int w = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int edges = d - 1;
    const int det_rounds = rounds - 1;
    const int start = w * window_size;

    if (edges <= 0 || rounds <= 0 || window_size <= 0)
        return;
    if (start + window_size > shots)
        return;

    const float inv_window = 1.0f / (float)window_size;

    const int raw_width = d + rounds * edges;
    const int det_width = (det_rounds > 0) ? det_rounds * edges : 0;
    const int agree_width = edges + rounds;
    const int field_agree_width = rounds * edges;
    const int field_det_width = det_width;
    const int sm_field_width = field_agree_width + field_det_width;

    float *raw_row = raw_out + (size_t)w * (size_t)raw_width;
    float *det_row = det_out + (size_t)w * (size_t)det_width;
    float *agree_row = agree_prof_out + (size_t)w * (size_t)agree_width;
    float *field_row = sm_field_out + (size_t)w * (size_t)sm_field_width;

    // -------------------------------------------------------------------------
    // raw_rates: data bit means.
    // -------------------------------------------------------------------------
    for (int coord = tid; coord < d; coord += nthreads)
    {
        int acc = 0;
        for (int s = 0; s < window_size; ++s)
        {
            int sh = start + s;
            acc += (int)(data[(size_t)sh * (size_t)d + (size_t)coord] & 1u);
        }
        raw_row[coord] = (float)acc * inv_window;
    }

    // -------------------------------------------------------------------------
    // raw_rates: syndrome means per (round, edge).
    // raw offset is d.
    // -------------------------------------------------------------------------
    const int synd_coords = rounds * edges;
    for (int coord = tid; coord < synd_coords; coord += nthreads)
    {
        int r = coord / edges;
        int e = coord - r * edges;

        int acc = 0;
        for (int s = 0; s < window_size; ++s)
        {
            int sh = start + s;
            size_t idx = ((size_t)sh * (size_t)rounds + (size_t)r) * (size_t)edges + (size_t)e;
            acc += (int)(synd[idx] & 1u);
        }
        raw_row[d + coord] = (float)acc * inv_window;
    }

    // -------------------------------------------------------------------------
    // detection_rates and sm_field detection half:
    // X[t,e] = synd[t+1,e] XOR synd[t,e].
    // -------------------------------------------------------------------------
    for (int coord = tid; coord < det_width; coord += nthreads)
    {
        int r = coord / edges;
        int e = coord - r * edges;

        int acc = 0;
        for (int s = 0; s < window_size; ++s)
        {
            int sh = start + s;
            size_t idx0 = ((size_t)sh * (size_t)rounds + (size_t)r) * (size_t)edges + (size_t)e;
            size_t idx1 = ((size_t)sh * (size_t)rounds + (size_t)(r + 1)) * (size_t)edges + (size_t)e;
            unsigned char x = (synd[idx0] ^ synd[idx1]) & 1u;
            acc += (int)x;
        }

        float mean_x = (float)acc * inv_window;
        det_row[coord] = mean_x;
        field_row[field_agree_width + coord] = mean_x;
    }

    // -------------------------------------------------------------------------
    // agreement field mean per (round, edge), written into sm_field_out first.
    // A[t,e] = 1 - (S[t,e] XOR (D[e] XOR D[e+1])).
    // -------------------------------------------------------------------------
    for (int coord = tid; coord < field_agree_width; coord += nthreads)
    {
        int r = coord / edges;
        int e = coord - r * edges;

        int acc = 0;
        for (int s = 0; s < window_size; ++s)
        {
            int sh = start + s;
            unsigned char de0 = data[(size_t)sh * (size_t)d + (size_t)e] & 1u;
            unsigned char de1 = data[(size_t)sh * (size_t)d + (size_t)(e + 1)] & 1u;
            unsigned char edge = (de0 ^ de1) & 1u;

            size_t sidx = ((size_t)sh * (size_t)rounds + (size_t)r) * (size_t)edges + (size_t)e;
            unsigned char sv = synd[sidx] & 1u;
            unsigned char agree = (unsigned char)(1u - ((sv ^ edge) & 1u));
            acc += (int)agree;
        }

        field_row[coord] = (float)acc * inv_window;
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // agreement_profiles: edge profile.
    // mean over shots and rounds for each edge.
    // Uses field_row values already reduced over shots.
    // -------------------------------------------------------------------------
    for (int e = tid; e < edges; e += nthreads)
    {
        float acc = 0.0f;
        for (int r = 0; r < rounds; ++r)
        {
            acc += field_row[r * edges + e];
        }
        agree_row[e] = acc / (float)rounds;
    }

    // -------------------------------------------------------------------------
    // agreement_profiles: time profile.
    // mean over shots and edges for each round.
    // Written after edge profile.
    // -------------------------------------------------------------------------
    for (int r = tid; r < rounds; r += nthreads)
    {
        float acc = 0.0f;
        for (int e = 0; e < edges; ++e)
        {
            acc += field_row[r * edges + e];
        }
        agree_row[edges + r] = acc / (float)edges;
    }
}

// =============================================================================
// Optional compact S_M collapse kernel.
// =============================================================================
// Computes a small diagnostic vector per window:
//
//   compact_out[w, 0] = mean data rate
//   compact_out[w, 1] = mean syndrome rate
//   compact_out[w, 2] = mean detection-event rate
//   compact_out[w, 3] = mean agreement rate
//
// This is useful for quick sanity checks and substrate agreement dashboards.
// The full benchmark should use sm_window_features_kernel above.
// =============================================================================

extern "C" __global__ void sm_window_compact_kernel(
    const unsigned char *__restrict__ data,
    const unsigned char *__restrict__ synd,
    int shots,
    int d,
    int rounds,
    int window_size,
    float *__restrict__ compact_out)
{
    const int w = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int edges = d - 1;
    const int start = w * window_size;

    if (edges <= 0 || rounds <= 0 || window_size <= 0)
        return;
    if (start + window_size > shots)
        return;

    __shared__ float shared[4 * 256];

    float data_sum = 0.0f;
    float synd_sum = 0.0f;
    float det_sum = 0.0f;
    float agree_sum = 0.0f;

    for (int s = tid; s < window_size; s += nthreads)
    {
        int sh = start + s;

        for (int j = 0; j < d; ++j)
        {
            data_sum += (float)(data[(size_t)sh * (size_t)d + (size_t)j] & 1u);
        }

        for (int r = 0; r < rounds; ++r)
        {
            for (int e = 0; e < edges; ++e)
            {
                size_t idx = ((size_t)sh * (size_t)rounds + (size_t)r) * (size_t)edges + (size_t)e;
                synd_sum += (float)(synd[idx] & 1u);

                unsigned char de0 = data[(size_t)sh * (size_t)d + (size_t)e] & 1u;
                unsigned char de1 = data[(size_t)sh * (size_t)d + (size_t)(e + 1)] & 1u;
                unsigned char edge = (de0 ^ de1) & 1u;
                unsigned char agree = (unsigned char)(1u - (((synd[idx] & 1u) ^ edge) & 1u));
                agree_sum += (float)agree;
            }
        }

        for (int r = 0; r + 1 < rounds; ++r)
        {
            for (int e = 0; e < edges; ++e)
            {
                size_t idx0 = ((size_t)sh * (size_t)rounds + (size_t)r) * (size_t)edges + (size_t)e;
                size_t idx1 = ((size_t)sh * (size_t)rounds + (size_t)(r + 1)) * (size_t)edges + (size_t)e;
                det_sum += (float)((synd[idx0] ^ synd[idx1]) & 1u);
            }
        }
    }

    // This compact kernel assumes blockDim.x <= 256.
    if (tid < 256)
    {
        shared[tid] = data_sum;
        shared[256 + tid] = synd_sum;
        shared[512 + tid] = det_sum;
        shared[768 + tid] = agree_sum;
    }
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
            shared[256 + tid] += shared[256 + tid + stride];
            shared[512 + tid] += shared[512 + tid + stride];
            shared[768 + tid] += shared[768 + tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        float denom_data = (float)(window_size * d);
        float denom_synd = (float)(window_size * rounds * edges);
        float denom_det = (rounds > 1) ? (float)(window_size * (rounds - 1) * edges) : 1.0f;

        compact_out[(size_t)w * 4u + 0u] = shared[0] / denom_data;
        compact_out[(size_t)w * 4u + 1u] = shared[256] / denom_synd;
        compact_out[(size_t)w * 4u + 2u] = shared[512] / denom_det;
        compact_out[(size_t)w * 4u + 3u] = shared[768] / denom_synd;
    }
}
