
/*
==============================================================================
GHOST ORACLE SUITE - D_M PROJECTOR CUDA KERNEL
==============================================================================

Purpose
-------
Optimized QPROJ/GPROJ projection kernel for D_M.

D_M current interpretation:
    Dimensional Entanglement Projection / Bell-Witness Manifold Operator

Record schema:
    pair[tile, shot, 2]              uint8
    tile_rung_index[tile]            int32
    tile_witness_index[tile]         int32    0=XY, 1=YZ, 2=ZY, 3=YX
    tile_base_delay_dt[tile]         int32
    tile_offset_dt[tile]             int32
    tile_total_delay_dt[tile]        int32

Current discovered witness:
    YZ = primary witness dimension
    ZY = reciprocal / inverted witness dimension
    XY / YX = comparison dimensions

Projection coordinates:
    C(P0,P1) = <P0 P1> - <P0><P1>
    Y        = C(YZ)
    Z        = C(ZY)
    R        = -Z
    E        = sqrt(Y^2 + R^2)
    S        = E - sqrt(C(XY)^2 + C(YX)^2)
    phi      = atan2(R, Y) mod pi

Kernels
-------
1. dm_tile_correlator_kernel_u8
   Compress pair[tile, shot, 2] -> tile_stats[tile, metric].

2. dm_independent_bit_shuffle_tile_kernel_u8
   Destructive control: preserve q0/q1 marginals but break same-shot pairing.

3. dm_rung_projection_kernel_f32
   Compress tile_stats -> rung_stats[rung, metric].

4. dm_projection_summary_kernel_f32
   Compress rung_stats -> summary[metric].

This is the final repo kernel for QPROJ/GPROJ plus backwards-compatible legacy GEO and the Probe-25 exact GEO reference path.

PATCH (GPU-only benchmark):
    dm_geo_rung_projection_kernel_f32 now takes an explicit geo_phase_scale
    argument instead of a hardcoded 0.37f, matching the per-candidate `ps`
    parameter already used by dm_geo_sweep_summary_kernel_f32. This keeps the
    single-manifold geo path and the batched sweep path in sync, and lets the
    Python --geo-phase-scale flag move off its default without desyncing.
==============================================================================
*/

extern "C"
{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef DM_MAX_THREADS
#define DM_MAX_THREADS 1024
#endif

#ifndef DM_SIGN_EPS
#define DM_SIGN_EPS 1.0e-6f
#endif

    // -----------------------------------------------------------------------------
    // Metric indices
    // -----------------------------------------------------------------------------

    enum DMTileMetric
    {
        DM_TILE_N_SHOTS = 0,
        DM_TILE_MEAN_Q0 = 1,
        DM_TILE_MEAN_Q1 = 2,
        DM_TILE_CORR = 3,
        DM_TILE_CONNECTED = 4,
        DM_TILE_P00 = 5,
        DM_TILE_P01 = 6,
        DM_TILE_P10 = 7,
        DM_TILE_P11 = 8,
        DM_TILE_Q0_ONE_RATE = 9,
        DM_TILE_Q1_ONE_RATE = 10,
        DM_TILE_ABS_CONNECTED = 11,
        DM_TILE_N_METRICS = 12
    };

    enum DMRungMetric
    {
        DM_RUNG_XY = 0,
        DM_RUNG_YZ = 1,
        DM_RUNG_ZY = 2,
        DM_RUNG_YX = 3,
        DM_RUNG_YZ_PRIMARY = 4,
        DM_RUNG_ZY_RETURN = 5,
        DM_RUNG_YZZY_ENERGY = 6,
        DM_RUNG_COMPARISON_ENERGY = 7,
        DM_RUNG_DIRECTIONAL_SPECIFICITY = 8,
        DM_RUNG_DIRECTIONAL_GAP = 9,
        DM_RUNG_INVERSION = 10,
        DM_RUNG_PI_PHASE = 11,
        DM_RUNG_PI_COS2 = 12,
        DM_RUNG_PI_SIN2 = 13,
        DM_RUNG_BASE_DELAY = 14,
        DM_RUNG_OFFSET = 15,
        DM_RUNG_TOTAL_DELAY = 16,
        DM_RUNG_COUNT_ALL = 17,
        DM_RUNG_COUNT_YZZY = 18,
        DM_RUNG_N_METRICS = 19
    };

    enum DMSummaryMetric
    {
        DM_SUMMARY_N_RUNGS = 0,
        DM_SUMMARY_YZ_MEAN = 1,
        DM_SUMMARY_YZ_POS_FRAC = 2,
        DM_SUMMARY_ZY_MEAN = 3,
        DM_SUMMARY_ZY_INVERTED_FRAC = 4,
        DM_SUMMARY_YZZY_ENERGY_MEAN = 5,
        DM_SUMMARY_YZZY_ENERGY_MAX = 6,
        DM_SUMMARY_SPECIFICITY_MEAN = 7,
        DM_SUMMARY_SPECIFICITY_MAX = 8,
        DM_SUMMARY_PI_PERIODIC_SCORE = 9,
        DM_SUMMARY_PI_PERIODIC_MODE = 10, // 0=linear, 1=log1p
        DM_SUMMARY_ENERGY_TRACKING_R = 11,
        DM_SUMMARY_SPECIFICITY_TRACKING_R = 12,
        DM_SUMMARY_PHASE_VELOCITY_R = 13,
        DM_SUMMARY_PHASE_SPAN_PI_UNITS = 14,
        DM_SUMMARY_PROJECTION_SCORE = 15,
        DM_SUMMARY_N_METRICS = 16
    };

    // -----------------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------------

    __device__ __forceinline__ int dm_pair_idx(int tile, int shot, int bit, int shots)
    {
        return (tile * shots + shot) * 2 + bit;
    }

    __device__ __forceinline__ int dm_tile_idx(int tile, int metric)
    {
        return tile * DM_TILE_N_METRICS + metric;
    }

    __device__ __forceinline__ int dm_rung_idx(int rung, int metric)
    {
        return rung * DM_RUNG_N_METRICS + metric;
    }

    __device__ __forceinline__ int dm_spin(unsigned char b)
    {
        return b ? -1 : 1; // bit 0 -> +1, bit 1 -> -1
    }

    __device__ __forceinline__ float dm_wrap_pi(float x)
    {
        float y = fmodf(x, (float)M_PI);
        if (y < 0.0f)
            y += (float)M_PI;
        return y;
    }

    __device__ __forceinline__ float dm_wrap_pi_delta(float d)
    {
        float y = fmodf(d + 0.5f * (float)M_PI, (float)M_PI);
        if (y < 0.0f)
            y += (float)M_PI;
        return y - 0.5f * (float)M_PI;
    }

    __device__ __forceinline__ int dm_perm_shot(int s, int shots, int a, int b)
    {
        return (int)(((long long)a * (long long)s + (long long)b) % (long long)shots);
    }

    __device__ __forceinline__ float dm_corr_small(const float *x, const float *y, int n)
    {
        if (n < 3)
            return 0.0f;

        float sx = 0.0f, sy = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            sx += x[i];
            sy += y[i];
        }

        float mx = sx / (float)n;
        float my = sy / (float)n;

        float vx = 0.0f, vy = 0.0f, c = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            float dx = x[i] - mx;
            float dy = y[i] - my;
            vx += dx * dx;
            vy += dy * dy;
            c += dx * dy;
        }

        if (vx <= 1.0e-12f || vy <= 1.0e-12f)
            return 0.0f;
        return c / sqrtf(vx * vy);
    }

    __device__ __forceinline__ void dm_norm_x_small(const float *raw, int n, int mode, float *out)
    {
        float mn = 0.0f, mx = 0.0f;

        for (int i = 0; i < n; ++i)
        {
            float v = raw[i];
            if (mode == 1)
                v = log1pf(fmaxf(0.0f, v));
            out[i] = v;

            if (i == 0)
            {
                mn = v;
                mx = v;
            }
            else
            {
                mn = fminf(mn, v);
                mx = fmaxf(mx, v);
            }
        }

        float span = mx - mn;
        if (fabsf(span) <= 1.0e-12f)
        {
            for (int i = 0; i < n; ++i)
                out[i] = 0.0f;
            return;
        }

        for (int i = 0; i < n; ++i)
            out[i] = (out[i] - mn) / span;
    }

    __device__ __forceinline__ float dm_pi_score_small(const float *x_raw, const float *phase, int n, int mode)
    {
        if (n < 3 || n > 64)
            return 0.0f;

        float x[64], c2[64], s2[64];
        dm_norm_x_small(x_raw, n, mode, x);

        for (int i = 0; i < n; ++i)
        {
            c2[i] = cosf(2.0f * phase[i]);
            s2[i] = sinf(2.0f * phase[i]);
        }

        float rc = dm_corr_small(x, c2, n);
        float rs = dm_corr_small(x, s2, n);
        float score = sqrtf(rc * rc + rs * rs);
        if (!isfinite(score))
            return 0.0f;
        return fminf(1.0f, score);
    }

    // -----------------------------------------------------------------------------
    // Shared reduction writer for normal or shuffled tile reads
    // -----------------------------------------------------------------------------

    __device__ void dm_write_tile_stats_from_sums(
        int tile,
        int shots,
        int sum_q0,
        int sum_q1,
        int sum_prod,
        int p00,
        int p01,
        int p10,
        int p11,
        float *tile_stats)
    {
        float n = (float)shots;
        float m0 = (float)sum_q0 / n;
        float m1 = (float)sum_q1 / n;
        float corr = (float)sum_prod / n;
        float conn = corr - m0 * m1;

        tile_stats[dm_tile_idx(tile, DM_TILE_N_SHOTS)] = n;
        tile_stats[dm_tile_idx(tile, DM_TILE_MEAN_Q0)] = m0;
        tile_stats[dm_tile_idx(tile, DM_TILE_MEAN_Q1)] = m1;
        tile_stats[dm_tile_idx(tile, DM_TILE_CORR)] = corr;
        tile_stats[dm_tile_idx(tile, DM_TILE_CONNECTED)] = conn;
        tile_stats[dm_tile_idx(tile, DM_TILE_P00)] = (float)p00 / n;
        tile_stats[dm_tile_idx(tile, DM_TILE_P01)] = (float)p01 / n;
        tile_stats[dm_tile_idx(tile, DM_TILE_P10)] = (float)p10 / n;
        tile_stats[dm_tile_idx(tile, DM_TILE_P11)] = (float)p11 / n;
        tile_stats[dm_tile_idx(tile, DM_TILE_Q0_ONE_RATE)] = (float)(p10 + p11) / n;
        tile_stats[dm_tile_idx(tile, DM_TILE_Q1_ONE_RATE)] = (float)(p01 + p11) / n;
        tile_stats[dm_tile_idx(tile, DM_TILE_ABS_CONNECTED)] = fabsf(conn);
    }

    __device__ __forceinline__ int dm_witness_shift_device(int witness_index)
    {
        int wi = witness_index & 3;
        if (wi == 0)
            return 5; // XY
        if (wi == 3)
            return 11; // YX
        return 0;      // YZ / ZY
    }

    __global__ void dm_make_projected_pair_from_bits_kernel_u8(
        const unsigned char *bit_bank,
        const int nbits,
        const unsigned char *base_pair,
        const int *tile_rung_index,
        const int *tile_witness_index,
        const int *tile_total_delay_dt,
        const int tiles,
        const int shots,
        const int data_stride,
        const int delay_scale,
        unsigned char *projected_pair)
    {
        int idx_global = blockIdx.x * blockDim.x + threadIdx.x;
        int total = tiles * shots;

        if (idx_global >= total || nbits <= 0)
            return;

        int tile = idx_global / shots;
        int shot = idx_global - tile * shots;

        int rung = tile_rung_index[tile];
        int wi = tile_witness_index[tile];
        int total_delay_dt = tile_total_delay_dt[tile];

        int dscale = delay_scale > 0 ? delay_scale : 1;
        int stride = data_stride > 0 ? data_stride : 1;

        int delay_bits = abs(total_delay_dt) / dscale;
        if (total_delay_dt == 0)
            delay_bits = 0;

        int tile_phase =
            (1009 * (tile + 1) + 9176 * (rung + 1) + 7919 * (wi + 1) + 13 * abs(total_delay_dt)) % nbits;

        int shift = dm_witness_shift_device(wi);
        int idx = (tile_phase + shot * stride) % nbits;

        int idx0 = idx;
        int idx1 = idx;

        if (wi == 1) // YZ
        {
            idx0 = idx;
            idx1 = (idx + delay_bits) % nbits;
        }
        else if (wi == 2) // ZY
        {
            idx0 = (idx + delay_bits) % nbits;
            idx1 = idx;
        }
        else if (wi == 0) // XY
        {
            idx0 = idx;
            idx1 = (idx + delay_bits + shift) % nbits;
        }
        else if (wi == 3) // YX
        {
            idx0 = (idx + delay_bits + shift) % nbits;
            idx1 = idx;
        }

        int out0 = (tile * shots + shot) * 2;
        int out1 = out0 + 1;

        projected_pair[out0] = bit_bank[idx0] ^ base_pair[out0];
        projected_pair[out1] = bit_bank[idx1] ^ base_pair[out1];
    }
    // -----------------------------------------------------------------------------
    // 1. Fast tile correlator compression
    // -----------------------------------------------------------------------------

    __global__ void dm_tile_correlator_kernel_u8(
        const unsigned char *pair,
        const int tiles,
        const int shots,
        float *tile_stats)
    {
        int tile = blockIdx.x;
        int tid = threadIdx.x;
        int nt = blockDim.x;

        if (tile >= tiles || nt > DM_MAX_THREADS)
            return;

        __shared__ int sh0[DM_MAX_THREADS], sh1[DM_MAX_THREADS], shp[DM_MAX_THREADS];
        __shared__ int sh00[DM_MAX_THREADS], sh01[DM_MAX_THREADS], sh10[DM_MAX_THREADS], sh11[DM_MAX_THREADS];

        int s0 = 0, s1 = 0, sp = 0;
        int p00 = 0, p01 = 0, p10 = 0, p11 = 0;

        for (int s = tid; s < shots; s += nt)
        {
            unsigned char b0 = pair[dm_pair_idx(tile, s, 0, shots)];
            unsigned char b1 = pair[dm_pair_idx(tile, s, 1, shots)];

            int q0 = dm_spin(b0);
            int q1 = dm_spin(b1);

            s0 += q0;
            s1 += q1;
            sp += q0 * q1;

            if (b0 == 0 && b1 == 0)
                ++p00;
            else if (b0 == 0 && b1 == 1)
                ++p01;
            else if (b0 == 1 && b1 == 0)
                ++p10;
            else
                ++p11;
        }

        sh0[tid] = s0;
        sh1[tid] = s1;
        shp[tid] = sp;
        sh00[tid] = p00;
        sh01[tid] = p01;
        sh10[tid] = p10;
        sh11[tid] = p11;
        __syncthreads();

        for (int stride = nt >> 1; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                sh0[tid] += sh0[tid + stride];
                sh1[tid] += sh1[tid + stride];
                shp[tid] += shp[tid + stride];
                sh00[tid] += sh00[tid + stride];
                sh01[tid] += sh01[tid + stride];
                sh10[tid] += sh10[tid + stride];
                sh11[tid] += sh11[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            dm_write_tile_stats_from_sums(
                tile, shots,
                sh0[0], sh1[0], shp[0],
                sh00[0], sh01[0], sh10[0], sh11[0],
                tile_stats);
        }
    }

    // -----------------------------------------------------------------------------
    // 2. Independent-bit-shuffle destructive control
    // -----------------------------------------------------------------------------

    __global__ void dm_independent_bit_shuffle_tile_kernel_u8(
        const unsigned char *pair,
        const int tiles,
        const int shots,
        const int seed,
        float *tile_stats)
    {
        int tile = blockIdx.x;
        int tid = threadIdx.x;
        int nt = blockDim.x;

        if (tile >= tiles || nt > DM_MAX_THREADS)
            return;

        __shared__ int sh0[DM_MAX_THREADS], sh1[DM_MAX_THREADS], shp[DM_MAX_THREADS];
        __shared__ int sh00[DM_MAX_THREADS], sh01[DM_MAX_THREADS], sh10[DM_MAX_THREADS], sh11[DM_MAX_THREADS];

        int a = 1664525 | 1;
        int b = (seed * 1013904223 + tile * 9176 + 1337) & 0x7fffffff;

        int s0 = 0, s1 = 0, sp = 0;
        int p00 = 0, p01 = 0, p10 = 0, p11 = 0;

        for (int s = tid; s < shots; s += nt)
        {
            int s_perm = dm_perm_shot(s, shots, a, b);

            unsigned char b0 = pair[dm_pair_idx(tile, s, 0, shots)];
            unsigned char b1 = pair[dm_pair_idx(tile, s_perm, 1, shots)];

            int q0 = dm_spin(b0);
            int q1 = dm_spin(b1);

            s0 += q0;
            s1 += q1;
            sp += q0 * q1;

            if (b0 == 0 && b1 == 0)
                ++p00;
            else if (b0 == 0 && b1 == 1)
                ++p01;
            else if (b0 == 1 && b1 == 0)
                ++p10;
            else
                ++p11;
        }

        sh0[tid] = s0;
        sh1[tid] = s1;
        shp[tid] = sp;
        sh00[tid] = p00;
        sh01[tid] = p01;
        sh10[tid] = p10;
        sh11[tid] = p11;
        __syncthreads();

        for (int stride = nt >> 1; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                sh0[tid] += sh0[tid + stride];
                sh1[tid] += sh1[tid + stride];
                shp[tid] += shp[tid + stride];
                sh00[tid] += sh00[tid + stride];
                sh01[tid] += sh01[tid + stride];
                sh10[tid] += sh10[tid + stride];
                sh11[tid] += sh11[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            dm_write_tile_stats_from_sums(
                tile, shots,
                sh0[0], sh1[0], shp[0],
                sh00[0], sh01[0], sh10[0], sh11[0],
                tile_stats);
        }
    }

    // -----------------------------------------------------------------------------
    // 3. Rung-level D_M projection
    // -----------------------------------------------------------------------------

    __global__ void dm_rung_projection_kernel_f32(
        const float *tile_stats,
        const int *tile_rung_index,
        const int *tile_witness_index,
        const int *tile_base_delay_dt,
        const int *tile_offset_dt,
        const int *tile_total_delay_dt,
        const int tiles,
        const int n_rungs,
        float *rung_stats)
    {
        int rung = blockIdx.x;
        if (rung >= n_rungs || threadIdx.x != 0)
            return;

        float w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int seen[4] = {0, 0, 0, 0};

        float bsum = 0.0f, osum = 0.0f, tsum = 0.0f;
        int count = 0;

        for (int t = 0; t < tiles; ++t)
        {
            if (tile_rung_index[t] != rung)
                continue;

            int wi = tile_witness_index[t];
            if (wi >= 0 && wi < 4)
            {
                w[wi] = tile_stats[dm_tile_idx(t, DM_TILE_CONNECTED)];
                seen[wi] = 1;
            }

            bsum += (float)tile_base_delay_dt[t];
            osum += (float)tile_offset_dt[t];
            tsum += (float)tile_total_delay_dt[t];
            ++count;
        }

        float xy = w[0];
        float yz = w[1];
        float zy = w[2];
        float yx = w[3];

        float ret = -zy;
        float energy = sqrtf(yz * yz + ret * ret);
        float comp = sqrtf(xy * xy + yx * yx);
        float spec = energy - comp;
        float gap = yz - zy;
        float inversion = -yz * zy;

        float phase = dm_wrap_pi(atan2f(ret, yz));
        float c2 = cosf(2.0f * phase);
        float s2 = sinf(2.0f * phase);

        float denom = count > 0 ? (float)count : 1.0f;

        rung_stats[dm_rung_idx(rung, DM_RUNG_XY)] = xy;
        rung_stats[dm_rung_idx(rung, DM_RUNG_YZ)] = yz;
        rung_stats[dm_rung_idx(rung, DM_RUNG_ZY)] = zy;
        rung_stats[dm_rung_idx(rung, DM_RUNG_YX)] = yx;
        rung_stats[dm_rung_idx(rung, DM_RUNG_YZ_PRIMARY)] = yz;
        rung_stats[dm_rung_idx(rung, DM_RUNG_ZY_RETURN)] = ret;
        rung_stats[dm_rung_idx(rung, DM_RUNG_YZZY_ENERGY)] = energy;
        rung_stats[dm_rung_idx(rung, DM_RUNG_COMPARISON_ENERGY)] = comp;
        rung_stats[dm_rung_idx(rung, DM_RUNG_DIRECTIONAL_SPECIFICITY)] = spec;
        rung_stats[dm_rung_idx(rung, DM_RUNG_DIRECTIONAL_GAP)] = gap;
        rung_stats[dm_rung_idx(rung, DM_RUNG_INVERSION)] = inversion;
        rung_stats[dm_rung_idx(rung, DM_RUNG_PI_PHASE)] = phase;
        rung_stats[dm_rung_idx(rung, DM_RUNG_PI_COS2)] = c2;
        rung_stats[dm_rung_idx(rung, DM_RUNG_PI_SIN2)] = s2;
        rung_stats[dm_rung_idx(rung, DM_RUNG_BASE_DELAY)] = bsum / denom;
        rung_stats[dm_rung_idx(rung, DM_RUNG_OFFSET)] = osum / denom;
        rung_stats[dm_rung_idx(rung, DM_RUNG_TOTAL_DELAY)] = tsum / denom;
        rung_stats[dm_rung_idx(rung, DM_RUNG_COUNT_ALL)] = (float)count;
        rung_stats[dm_rung_idx(rung, DM_RUNG_COUNT_YZZY)] = (float)(seen[1] && seen[2]);
    }

    // -----------------------------------------------------------------------------
    // 4. Base-level projection summary
    // -----------------------------------------------------------------------------

    __global__ void dm_projection_summary_kernel_f32(
        const float *rung_stats,
        const int n_rungs,
        float *summary)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0)
            return;

        if (n_rungs <= 0 || n_rungs > 64)
        {
            for (int m = 0; m < DM_SUMMARY_N_METRICS; ++m)
                summary[m] = 0.0f;
            return;
        }

        float yz[64], zy[64], energy[64], spec[64], phase[64], x_total[64];
        int n = 0;

        float yz_sum = 0.0f, zy_sum = 0.0f, e_sum = 0.0f, s_sum = 0.0f;
        float e_max = 0.0f, s_max = -1.0e20f;
        int yz_pos = 0, zy_inv = 0;

        for (int r = 0; r < n_rungs; ++r)
        {
            if (rung_stats[dm_rung_idx(r, DM_RUNG_COUNT_YZZY)] <= 0.0f)
                continue;

            float yv = rung_stats[dm_rung_idx(r, DM_RUNG_YZ_PRIMARY)];
            float zv = rung_stats[dm_rung_idx(r, DM_RUNG_ZY)];
            float ev = rung_stats[dm_rung_idx(r, DM_RUNG_YZZY_ENERGY)];
            float sv = rung_stats[dm_rung_idx(r, DM_RUNG_DIRECTIONAL_SPECIFICITY)];
            float ph = rung_stats[dm_rung_idx(r, DM_RUNG_PI_PHASE)];
            float xt = rung_stats[dm_rung_idx(r, DM_RUNG_TOTAL_DELAY)];

            yz[n] = yv;
            zy[n] = zv;
            energy[n] = ev;
            spec[n] = sv;
            phase[n] = ph;
            x_total[n] = xt;

            yz_sum += yv;
            zy_sum += zv;
            e_sum += ev;
            s_sum += sv;

            e_max = n == 0 ? ev : fmaxf(e_max, ev);
            s_max = n == 0 ? sv : fmaxf(s_max, sv);

            if (yv > DM_SIGN_EPS)
                ++yz_pos;
            if (yv * zv < -DM_SIGN_EPS)
                ++zy_inv;

            ++n;
        }

        if (n <= 0)
        {
            for (int m = 0; m < DM_SUMMARY_N_METRICS; ++m)
                summary[m] = 0.0f;
            return;
        }

        float invn = 1.0f / (float)n;
        float yz_mean = yz_sum * invn;
        float zy_mean = zy_sum * invn;
        float e_mean = e_sum * invn;
        float s_mean = s_sum * invn;
        float yz_pos_frac = (float)yz_pos * invn;
        float zy_inv_frac = (float)zy_inv * invn;

        float x_lin[64], x_log[64];
        dm_norm_x_small(x_total, n, 0, x_lin);
        dm_norm_x_small(x_total, n, 1, x_log);

        float e_r_lin = dm_corr_small(x_lin, energy, n);
        float e_r_log = dm_corr_small(x_log, energy, n);
        float s_r_lin = dm_corr_small(x_lin, spec, n);
        float s_r_log = dm_corr_small(x_log, spec, n);

        float e_r = fabsf(e_r_log) > fabsf(e_r_lin) ? e_r_log : e_r_lin;
        float s_r = fabsf(s_r_log) > fabsf(s_r_lin) ? s_r_log : s_r_lin;

        float pi_lin = dm_pi_score_small(x_total, phase, n, 0);
        float pi_log = dm_pi_score_small(x_total, phase, n, 1);
        float pi_score = pi_log > pi_lin ? pi_log : pi_lin;
        float pi_mode = pi_log > pi_lin ? 1.0f : 0.0f;

        float phase_vel_r = 0.0f;
        float phase_span = 0.0f;

        if (n >= 3)
        {
            float mid_x[64], vel[64], mid_norm[64];
            int nv = 0;

            float acc = phase[0], pmin = acc, pmax = acc;

            for (int i = 1; i < n; ++i)
            {
                float dx = x_total[i] - x_total[i - 1];
                float dp = dm_wrap_pi_delta(phase[i] - phase[i - 1]);
                acc += dp;
                pmin = fminf(pmin, acc);
                pmax = fmaxf(pmax, acc);

                if (fabsf(dx) > 1.0e-12f)
                {
                    mid_x[nv] = 0.5f * (x_total[i] + x_total[i - 1]);
                    vel[nv] = dp / dx;
                    ++nv;
                }
            }

            phase_span = fabsf(pmax - pmin) / (float)M_PI;

            if (nv >= 3)
            {
                dm_norm_x_small(mid_x, nv, 1, mid_norm);
                phase_vel_r = dm_corr_small(mid_norm, vel, nv);
            }
        }

        // Bounded projector score for benchmark ranking, not entanglement certification.
        // Non-normalized physical projection score.
        //
        // This intentionally removes empirical calibration divisors such as /0.35
        // and /0.25. The score lives on the natural connected-correlator scale:
        //
        //     connected correlator C in [-1, 1]
        //     YZ/ZY energy sqrt(YZ^2 + (-ZY)^2)
        //     specificity = YZ/ZY energy - comparison energy
        //
        // The pi score remains a shape/fit diagnostic, but the load-bearing pi term is
        // physical pi witness strength:
        //
        //     pi_witness_strength = energy_mean * pi_score
        //
        // That keeps pi influence naturally bounded by the measured YZ/ZY witness
        // energy instead of letting an idealized phase fit dominate low-energy residue.
        float e_term = fmaxf(0.0f, e_mean);
        float s_term = fmaxf(0.0f, s_mean);
        float y_term = fmaxf(0.0f, yz_mean);
        float pi_witness_strength = fmaxf(0.0f, e_mean) * pi_score;
        float p_term = pi_witness_strength;
        float t_term = 0.5f * (fabsf(e_r) + fabsf(s_r));

        float projection =
            0.35f * e_term +
            0.25f * s_term +
            0.15f * y_term +
            0.15f * p_term +
            0.10f * t_term;

        summary[DM_SUMMARY_N_RUNGS] = (float)n;
        summary[DM_SUMMARY_YZ_MEAN] = yz_mean;
        summary[DM_SUMMARY_YZ_POS_FRAC] = yz_pos_frac;
        summary[DM_SUMMARY_ZY_MEAN] = zy_mean;
        summary[DM_SUMMARY_ZY_INVERTED_FRAC] = zy_inv_frac;
        summary[DM_SUMMARY_YZZY_ENERGY_MEAN] = e_mean;
        summary[DM_SUMMARY_YZZY_ENERGY_MAX] = e_max;
        summary[DM_SUMMARY_SPECIFICITY_MEAN] = s_mean;
        summary[DM_SUMMARY_SPECIFICITY_MAX] = s_max;
        summary[DM_SUMMARY_PI_PERIODIC_SCORE] = pi_score;
        summary[DM_SUMMARY_PI_PERIODIC_MODE] = pi_mode;
        summary[DM_SUMMARY_ENERGY_TRACKING_R] = e_r;
        summary[DM_SUMMARY_SPECIFICITY_TRACKING_R] = s_r;
        summary[DM_SUMMARY_PHASE_VELOCITY_R] = phase_vel_r;
        summary[DM_SUMMARY_PHASE_SPAN_PI_UNITS] = phase_span;
        summary[DM_SUMMARY_PROJECTION_SCORE] = projection;
    }

    // -----------------------------------------------------------------------------
    // GEO ANALYTIC PATH
    // -----------------------------------------------------------------------------

    /*
    ------------------------------------------------------------------------------
    dm_geo_rung_projection_kernel_f32

    Analytic GEO path for D_M.

    This computes the same rung_stats[rung, metric] object as the qproj/gproj
    record path, but directly from condition metadata and analytic parameters.

    Grid:
        blockIdx.x = rung

    condition_kind:
        0 = null
        1 = base_delay
        2 = offset_deformed

    Inputs:
        base_delays_dt[rung]
        offset_dt
        geo_base_energy
        geo_energy_gain
        geo_comparison_scale
        geo_offset_deform
        geo_phase_scale          // PATCHED: was hardcoded 0.37f

    Output:
        rung_stats[rung, metric]

    Notes:
        GEO has no shots. It is an analytic manifold, not sampled records.
    ------------------------------------------------------------------------------
    */
    __global__ void dm_geo_rung_projection_kernel_f32(
        const int condition_kind,
        const int *base_delays_dt,
        const int n_rungs,
        const int offset_dt,
        const float geo_base_energy,
        const float geo_energy_gain,
        const float geo_comparison_scale,
        const float geo_offset_deform,
        const float geo_phase_scale,
        float *rung_stats)
    {
        int r = blockIdx.x;

        if (r >= n_rungs)
            return;

        if (threadIdx.x != 0)
            return;

        // Compute total-delay normalization over rung-average total delay.
        float total_vals[64];
        float lin_min = 0.0f, lin_max = 0.0f;
        float log_min = 0.0f, log_max = 0.0f;

        if (n_rungs > 64)
            return;

        for (int i = 0; i < n_rungs; ++i)
        {
            float off_mean_i = ((float)(4 * i) + 1.5f) * (float)offset_dt;
            float total_i = (float)base_delays_dt[i] + off_mean_i;
            float log_i = log1pf(fmaxf(0.0f, total_i));

            total_vals[i] = total_i;

            if (i == 0)
            {
                lin_min = total_i;
                lin_max = total_i;
                log_min = log_i;
                log_max = log_i;
            }
            else
            {
                lin_min = fminf(lin_min, total_i);
                lin_max = fmaxf(lin_max, total_i);
                log_min = fminf(log_min, log_i);
                log_max = fmaxf(log_max, log_i);
            }
        }

        float base_delay = (float)base_delays_dt[r];
        float offset_mean = ((float)(4 * r) + 1.5f) * (float)offset_dt;
        float total_delay = base_delay + offset_mean;

        float x_lin = 0.0f;
        if (fabsf(lin_max - lin_min) > 1.0e-12f)
            x_lin = (total_delay - lin_min) / (lin_max - lin_min);

        float log_total = log1pf(fmaxf(0.0f, total_delay));
        float x_log = 0.0f;
        if (fabsf(log_max - log_min) > 1.0e-12f)
            x_log = (log_total - log_min) / (log_max - log_min);

        float xy = 0.0f;
        float yz = 0.0f;
        float zy = 0.0f;
        float yx = 0.0f;

        if (condition_kind == 0)
        {
            // Weak deterministic null residue.
            yz = 0.010f * sinf(1.7f * (float)r + 0.2f);
            zy = 0.010f * cosf(1.3f * (float)r + 0.5f);
            xy = 0.010f * cosf(0.9f * (float)r);
            yx = 0.010f * sinf(0.8f * (float)r + 0.4f);
        }
        else
        {
            float amp = geo_base_energy + geo_energy_gain * (0.20f + 0.80f * powf(x_log, 1.10f));
            float phase = (float)M_PI * (0.05f + geo_phase_scale * x_log);

            if (condition_kind == 2)
            {
                phase += geo_offset_deform * sinf(2.0f * (float)M_PI * x_lin + 0.35f);
                amp *= 0.90f + 0.18f * cosf(2.0f * (float)M_PI * x_lin + 0.17f);
            }

            phase = dm_wrap_pi(phase);

            yz = amp * cosf(phase);
            float ret = amp * sinf(phase);
            zy = -ret;

            if (yz < 0.0f)
                yz = -0.35f * yz;

            xy = geo_comparison_scale * cosf(1.2f * (float)r + 0.3f);
            yx = geo_comparison_scale * sinf(1.1f * (float)r + 0.9f);
        }

        float ret = -zy;
        float energy = sqrtf(yz * yz + ret * ret);
        float comp = sqrtf(xy * xy + yx * yx);
        float spec = energy - comp;
        float gap = yz - zy;
        float inversion = -yz * zy;
        float phase2 = dm_wrap_pi(atan2f(ret, yz));

        rung_stats[dm_rung_idx(r, DM_RUNG_XY)] = xy;
        rung_stats[dm_rung_idx(r, DM_RUNG_YZ)] = yz;
        rung_stats[dm_rung_idx(r, DM_RUNG_ZY)] = zy;
        rung_stats[dm_rung_idx(r, DM_RUNG_YX)] = yx;
        rung_stats[dm_rung_idx(r, DM_RUNG_YZ_PRIMARY)] = yz;
        rung_stats[dm_rung_idx(r, DM_RUNG_ZY_RETURN)] = ret;
        rung_stats[dm_rung_idx(r, DM_RUNG_YZZY_ENERGY)] = energy;
        rung_stats[dm_rung_idx(r, DM_RUNG_COMPARISON_ENERGY)] = comp;
        rung_stats[dm_rung_idx(r, DM_RUNG_DIRECTIONAL_SPECIFICITY)] = spec;
        rung_stats[dm_rung_idx(r, DM_RUNG_DIRECTIONAL_GAP)] = gap;
        rung_stats[dm_rung_idx(r, DM_RUNG_INVERSION)] = inversion;
        rung_stats[dm_rung_idx(r, DM_RUNG_PI_PHASE)] = phase2;
        rung_stats[dm_rung_idx(r, DM_RUNG_PI_COS2)] = cosf(2.0f * phase2);
        rung_stats[dm_rung_idx(r, DM_RUNG_PI_SIN2)] = sinf(2.0f * phase2);
        rung_stats[dm_rung_idx(r, DM_RUNG_BASE_DELAY)] = base_delay;
        rung_stats[dm_rung_idx(r, DM_RUNG_OFFSET)] = offset_mean;
        rung_stats[dm_rung_idx(r, DM_RUNG_TOTAL_DELAY)] = total_delay;
        rung_stats[dm_rung_idx(r, DM_RUNG_COUNT_ALL)] = 4.0f;
        rung_stats[dm_rung_idx(r, DM_RUNG_COUNT_YZZY)] = 1.0f;
    }

    /*
    ------------------------------------------------------------------------------
    dm_geo_sweep_summary_kernel_f32

    Batched analytic GEO sweep for D_M.

    This is the actual optimized GEO path. It does not sample shots and it does not
    launch one tiny 20-point manifold at a time. Instead, it evaluates many analytic
    candidate manifolds in parallel.

    Grid:
        blockIdx.x = candidate
        blockIdx.y = condition

    condition:
        0 = null
        1 = base_delay
        2 = offset_deformed

    Each block writes:
        sweep_summary[(candidate, condition, metric)]

    Candidate parameter arrays:
        geo_base_energy[candidate]
        geo_energy_gain[candidate]
        geo_comparison_scale[candidate]
        geo_offset_deform[candidate]
        geo_phase_scale[candidate]
        geo_phase_shift[candidate]

    Rate semantics:
        analytic_points_per_second = candidates * conditions * n_rungs * 4 / seconds

    This is the minimized geometric path: metadata/parameters -> D_M summary vector,
    with no shot-record reduction.
    ------------------------------------------------------------------------------
    */
    __global__ void dm_geo_sweep_summary_kernel_f32(
        const int *base_delays_dt,
        const int n_rungs,
        const int offset_dt,
        const int n_candidates,
        const float *geo_base_energy,
        const float *geo_energy_gain,
        const float *geo_comparison_scale,
        const float *geo_offset_deform,
        const float *geo_phase_scale,
        const float *geo_phase_shift,
        float *sweep_summary)
    {
        int cand = blockIdx.x;
        int condition_kind = blockIdx.y;

        if (cand >= n_candidates || condition_kind >= 3)
            return;

        if (threadIdx.x != 0)
            return;

        if (n_rungs <= 0 || n_rungs > 64)
            return;

        float be = geo_base_energy[cand];
        float eg = geo_energy_gain[cand];
        float cs = geo_comparison_scale[cand];
        float od = geo_offset_deform[cand];
        float ps = geo_phase_scale[cand];
        float pshift = geo_phase_shift[cand];

        float total_vals[64];
        float lin_min = 0.0f, lin_max = 0.0f;
        float log_min = 0.0f, log_max = 0.0f;

        for (int i = 0; i < n_rungs; ++i)
        {
            int use_offset = (condition_kind == 2) ? offset_dt : 0;
            int use_base = (condition_kind == 0) ? 0 : base_delays_dt[i];

            float off_mean_i = ((float)(4 * i) + 1.5f) * (float)use_offset;
            float total_i = (float)use_base + off_mean_i;
            float log_i = log1pf(fmaxf(0.0f, total_i));

            total_vals[i] = total_i;

            if (i == 0)
            {
                lin_min = total_i;
                lin_max = total_i;
                log_min = log_i;
                log_max = log_i;
            }
            else
            {
                lin_min = fminf(lin_min, total_i);
                lin_max = fmaxf(lin_max, total_i);
                log_min = fminf(log_min, log_i);
                log_max = fmaxf(log_max, log_i);
            }
        }

        float yz[64], zy[64], energy[64], spec[64], phase_arr[64], x_total[64];

        float yz_sum = 0.0f;
        float zy_sum = 0.0f;
        float e_sum = 0.0f;
        float s_sum = 0.0f;
        float e_max = 0.0f;
        float s_max = -1.0e20f;
        int yz_pos = 0;
        int zy_inv = 0;

        for (int r = 0; r < n_rungs; ++r)
        {
            int use_offset = (condition_kind == 2) ? offset_dt : 0;
            int use_base = (condition_kind == 0) ? 0 : base_delays_dt[r];

            float base_delay = (float)use_base;
            float offset_mean = ((float)(4 * r) + 1.5f) * (float)use_offset;
            float total_delay = base_delay + offset_mean;

            float x_lin = 0.0f;
            if (fabsf(lin_max - lin_min) > 1.0e-12f)
                x_lin = (total_delay - lin_min) / (lin_max - lin_min);

            float log_total = log1pf(fmaxf(0.0f, total_delay));
            float x_log = 0.0f;
            if (fabsf(log_max - log_min) > 1.0e-12f)
                x_log = (log_total - log_min) / (log_max - log_min);

            float xy = 0.0f;
            float yv = 0.0f;
            float zv = 0.0f;
            float yx = 0.0f;

            if (condition_kind == 0)
            {
                yv = 0.010f * sinf(1.7f * (float)r + 0.2f + pshift);
                zv = 0.010f * cosf(1.3f * (float)r + 0.5f + pshift);
                xy = 0.010f * cosf(0.9f * (float)r + pshift);
                yx = 0.010f * sinf(0.8f * (float)r + 0.4f + pshift);
            }
            else
            {
                float amp = be + eg * (0.20f + 0.80f * powf(x_log, 1.10f));
                float ph = (float)M_PI * (0.05f + ps * x_log) + pshift;

                if (condition_kind == 2)
                {
                    ph += od * sinf(2.0f * (float)M_PI * x_lin + 0.35f);
                    amp *= 0.90f + 0.18f * cosf(2.0f * (float)M_PI * x_lin + 0.17f);
                }

                ph = dm_wrap_pi(ph);

                yv = amp * cosf(ph);
                float ret0 = amp * sinf(ph);
                zv = -ret0;

                if (yv < 0.0f)
                    yv = -0.35f * yv;

                xy = cs * cosf(1.2f * (float)r + 0.3f);
                yx = cs * sinf(1.1f * (float)r + 0.9f);
            }

            float ret = -zv;
            float ev = sqrtf(yv * yv + ret * ret);
            float comp = sqrtf(xy * xy + yx * yx);
            float sv = ev - comp;
            float ph2 = dm_wrap_pi(atan2f(ret, yv));

            yz[r] = yv;
            zy[r] = zv;
            energy[r] = ev;
            spec[r] = sv;
            phase_arr[r] = ph2;
            x_total[r] = total_delay;

            yz_sum += yv;
            zy_sum += zv;
            e_sum += ev;
            s_sum += sv;

            e_max = r == 0 ? ev : fmaxf(e_max, ev);
            s_max = r == 0 ? sv : fmaxf(s_max, sv);

            if (yv > DM_SIGN_EPS)
                yz_pos += 1;
            if (yv * zv < -DM_SIGN_EPS)
                zy_inv += 1;
        }

        float invn = 1.0f / (float)n_rungs;
        float yz_mean = yz_sum * invn;
        float zy_mean = zy_sum * invn;
        float e_mean = e_sum * invn;
        float s_mean = s_sum * invn;
        float yz_pos_frac = (float)yz_pos * invn;
        float zy_inv_frac = (float)zy_inv * invn;

        float x_lin_arr[64], x_log_arr[64];
        dm_norm_x_small(x_total, n_rungs, 0, x_lin_arr);
        dm_norm_x_small(x_total, n_rungs, 1, x_log_arr);

        float e_r_lin = dm_corr_small(x_lin_arr, energy, n_rungs);
        float e_r_log = dm_corr_small(x_log_arr, energy, n_rungs);
        float s_r_lin = dm_corr_small(x_lin_arr, spec, n_rungs);
        float s_r_log = dm_corr_small(x_log_arr, spec, n_rungs);

        float e_r = fabsf(e_r_log) > fabsf(e_r_lin) ? e_r_log : e_r_lin;
        float s_r = fabsf(s_r_log) > fabsf(s_r_lin) ? s_r_log : s_r_lin;

        float pi_lin = dm_pi_score_small(x_total, phase_arr, n_rungs, 0);
        float pi_log = dm_pi_score_small(x_total, phase_arr, n_rungs, 1);
        float pi_score = pi_log > pi_lin ? pi_log : pi_lin;
        float pi_mode = pi_log > pi_lin ? 1.0f : 0.0f;

        float phase_vel_r = 0.0f;
        float phase_span = 0.0f;

        if (n_rungs >= 3)
        {
            float mid_x[64], vel[64], mid_norm[64];
            int nv = 0;

            float acc = phase_arr[0];
            float pmin = acc;
            float pmax = acc;

            for (int i = 1; i < n_rungs; ++i)
            {
                float dx = x_total[i] - x_total[i - 1];
                float dp = dm_wrap_pi_delta(phase_arr[i] - phase_arr[i - 1]);
                acc += dp;
                pmin = fminf(pmin, acc);
                pmax = fmaxf(pmax, acc);

                if (fabsf(dx) > 1.0e-12f)
                {
                    mid_x[nv] = 0.5f * (x_total[i] + x_total[i - 1]);
                    vel[nv] = dp / dx;
                    ++nv;
                }
            }

            phase_span = fabsf(pmax - pmin) / (float)M_PI;

            if (nv >= 3)
            {
                dm_norm_x_small(mid_x, nv, 1, mid_norm);
                phase_vel_r = dm_corr_small(mid_norm, vel, nv);
            }
        }

        // Non-normalized physical projection score.
        //
        // This intentionally removes empirical calibration divisors such as /0.35
        // and /0.25. The score lives on the natural connected-correlator scale:
        //
        //     connected correlator C in [-1, 1]
        //     YZ/ZY energy sqrt(YZ^2 + (-ZY)^2)
        //     specificity = YZ/ZY energy - comparison energy
        //
        // The pi score remains a shape/fit diagnostic, but the load-bearing pi term is
        // physical pi witness strength:
        //
        //     pi_witness_strength = energy_mean * pi_score
        //
        // That keeps pi influence naturally bounded by the measured YZ/ZY witness
        // energy instead of letting an idealized phase fit dominate low-energy residue.
        float e_term = fmaxf(0.0f, e_mean);
        float s_term = fmaxf(0.0f, s_mean);
        float y_term = fmaxf(0.0f, yz_mean);
        float pi_witness_strength = fmaxf(0.0f, e_mean) * pi_score;
        float p_term = pi_witness_strength;
        float t_term = 0.5f * (fabsf(e_r) + fabsf(s_r));

        float projection =
            0.35f * e_term +
            0.25f * s_term +
            0.15f * y_term +
            0.15f * p_term +
            0.10f * t_term;

        int base_idx = (cand * 3 + condition_kind) * DM_SUMMARY_N_METRICS;

        sweep_summary[base_idx + DM_SUMMARY_N_RUNGS] = (float)n_rungs;
        sweep_summary[base_idx + DM_SUMMARY_YZ_MEAN] = yz_mean;
        sweep_summary[base_idx + DM_SUMMARY_YZ_POS_FRAC] = yz_pos_frac;
        sweep_summary[base_idx + DM_SUMMARY_ZY_MEAN] = zy_mean;
        sweep_summary[base_idx + DM_SUMMARY_ZY_INVERTED_FRAC] = zy_inv_frac;
        sweep_summary[base_idx + DM_SUMMARY_YZZY_ENERGY_MEAN] = e_mean;
        sweep_summary[base_idx + DM_SUMMARY_YZZY_ENERGY_MAX] = e_max;
        sweep_summary[base_idx + DM_SUMMARY_SPECIFICITY_MEAN] = s_mean;
        sweep_summary[base_idx + DM_SUMMARY_SPECIFICITY_MAX] = s_max;
        sweep_summary[base_idx + DM_SUMMARY_PI_PERIODIC_SCORE] = pi_score;
        sweep_summary[base_idx + DM_SUMMARY_PI_PERIODIC_MODE] = pi_mode;
        sweep_summary[base_idx + DM_SUMMARY_ENERGY_TRACKING_R] = e_r;
        sweep_summary[base_idx + DM_SUMMARY_SPECIFICITY_TRACKING_R] = s_r;
        sweep_summary[base_idx + DM_SUMMARY_PHASE_VELOCITY_R] = phase_vel_r;
        sweep_summary[base_idx + DM_SUMMARY_PHASE_SPAN_PI_UNITS] = phase_span;
        sweep_summary[base_idx + DM_SUMMARY_PROJECTION_SCORE] = projection;
    }

    // -----------------------------------------------------------------------------
    // GEO EXACT PATH (Probe 25 closed-form classical reference)
    // -----------------------------------------------------------------------------

    __device__ __forceinline__ void dm_geo_exact_compute_norms(
        const int condition_kind,
        const int *base_delays_dt,
        const int n_rungs,
        const int offset_dt,
        float *space_norm,
        float *time_norm)
    {
        float s_min = 0.0f, s_max = 0.0f;
        float t_min = 0.0f, t_max = 0.0f;

        for (int i = 0; i < n_rungs; ++i)
        {
            int use_base = (condition_kind == 0) ? 0 : base_delays_dt[i];
            int use_offset = (condition_kind == 2) ? offset_dt : 0;
            float base_delay = (float)use_base;
            float offset_mean = ((float)(4 * i) + 1.5f) * (float)use_offset;
            float s = log1pf(fmaxf(0.0f, base_delay));
            float t = log1pf(fmaxf(0.0f, base_delay + offset_mean));

            space_norm[i] = s;
            time_norm[i] = t;

            if (i == 0)
            {
                s_min = s_max = s;
                t_min = t_max = t;
            }
            else
            {
                s_min = fminf(s_min, s);
                s_max = fmaxf(s_max, s);
                t_min = fminf(t_min, t);
                t_max = fmaxf(t_max, t);
            }
        }

        float s_span = s_max - s_min;
        float t_span = t_max - t_min;
        for (int i = 0; i < n_rungs; ++i)
        {
            space_norm[i] = (fabsf(s_span) > 1.0e-12f) ? (space_norm[i] - s_min) / s_span : 0.0f;
            time_norm[i] = (fabsf(t_span) > 1.0e-12f) ? (time_norm[i] - t_min) / t_span : 0.0f;
        }
    }

    __device__ __forceinline__ void dm_write_rung_from_witnesses(
        int r,
        float xy,
        float yz,
        float zy,
        float yx,
        float base_delay,
        float offset_mean,
        float total_delay,
        float count_yzzy,
        float *rung_stats)
    {
        float ret = -zy;
        float energy = sqrtf(yz * yz + ret * ret);
        float comp = sqrtf(xy * xy + yx * yx);
        float spec = energy - comp;
        float gap = yz - zy;
        float inversion = -yz * zy;
        float phase = dm_wrap_pi(atan2f(ret, yz));

        rung_stats[dm_rung_idx(r, DM_RUNG_XY)] = xy;
        rung_stats[dm_rung_idx(r, DM_RUNG_YZ)] = yz;
        rung_stats[dm_rung_idx(r, DM_RUNG_ZY)] = zy;
        rung_stats[dm_rung_idx(r, DM_RUNG_YX)] = yx;
        rung_stats[dm_rung_idx(r, DM_RUNG_YZ_PRIMARY)] = yz;
        rung_stats[dm_rung_idx(r, DM_RUNG_ZY_RETURN)] = ret;
        rung_stats[dm_rung_idx(r, DM_RUNG_YZZY_ENERGY)] = energy;
        rung_stats[dm_rung_idx(r, DM_RUNG_COMPARISON_ENERGY)] = comp;
        rung_stats[dm_rung_idx(r, DM_RUNG_DIRECTIONAL_SPECIFICITY)] = spec;
        rung_stats[dm_rung_idx(r, DM_RUNG_DIRECTIONAL_GAP)] = gap;
        rung_stats[dm_rung_idx(r, DM_RUNG_INVERSION)] = inversion;
        rung_stats[dm_rung_idx(r, DM_RUNG_PI_PHASE)] = phase;
        rung_stats[dm_rung_idx(r, DM_RUNG_PI_COS2)] = cosf(2.0f * phase);
        rung_stats[dm_rung_idx(r, DM_RUNG_PI_SIN2)] = sinf(2.0f * phase);
        rung_stats[dm_rung_idx(r, DM_RUNG_BASE_DELAY)] = base_delay;
        rung_stats[dm_rung_idx(r, DM_RUNG_OFFSET)] = offset_mean;
        rung_stats[dm_rung_idx(r, DM_RUNG_TOTAL_DELAY)] = total_delay;
        rung_stats[dm_rung_idx(r, DM_RUNG_COUNT_ALL)] = 4.0f;
        rung_stats[dm_rung_idx(r, DM_RUNG_COUNT_YZZY)] = count_yzzy;
    }

    /*
    ------------------------------------------------------------------------------
    dm_geo_exact_rung_projection_kernel_f32

    Probe-25 / final exact GEO reference path.

    condition_kind:
        0 = null                 -> exact zero manifold; count_yzzy = 0
        1 = base_delay           -> active closed-form manifold
        2 = offset_deformed      -> active closed-form manifold with offset axis

    GEO rule:
        x_space = normalize(log1p(base_delay))
        x_time  = normalize(log1p(base_delay + mean_offset))
        x_dm    = sqrt((w_space*x_space^2 + w_time*x_time^2)/(w_space+w_time))
        cos(2phi) = 2*x_time - 1
        YZ      = E*cos(phi)
        ZY      = -E*sin(phi)
        E       = energy_floor + energy_scale*x_dm^energy_gamma

    The output is the same rung_stats[rung, metric] object used by qproj/gproj
    and legacy GEO. Run dm_projection_summary_kernel_f32 after this kernel.
    ------------------------------------------------------------------------------
    */
    __global__ void dm_geo_exact_rung_projection_kernel_f32(
        const int condition_kind,
        const int *base_delays_dt,
        const int n_rungs,
        const int offset_dt,
        const float energy_floor,
        const float energy_scale,
        const float energy_gamma,
        const float weight_space,
        const float weight_time,
        float *rung_stats)
    {
        int r = blockIdx.x;
        if (r >= n_rungs || threadIdx.x != 0 || n_rungs <= 0 || n_rungs > 64)
            return;

        int use_base = (condition_kind == 0) ? 0 : base_delays_dt[r];
        int use_offset = (condition_kind == 2) ? offset_dt : 0;
        float base_delay = (float)use_base;
        float offset_mean = ((float)(4 * r) + 1.5f) * (float)use_offset;
        float total_delay = base_delay + offset_mean;

        if (condition_kind == 0)
        {
            dm_write_rung_from_witnesses(r, 0.0f, 0.0f, 0.0f, 0.0f, base_delay, offset_mean, total_delay, 0.0f, rung_stats);
            return;
        }

        float xs[64], xt[64];
        dm_geo_exact_compute_norms(condition_kind, base_delays_dt, n_rungs, offset_dt, xs, xt);

        float ws = fmaxf(0.0f, weight_space);
        float wt = fmaxf(0.0f, weight_time);
        float denom = fmaxf(ws + wt, 1.0e-12f);
        float x_dm = sqrtf((ws * xs[r] * xs[r] + wt * xt[r] * xt[r]) / denom);

        float gamma = fmaxf(energy_gamma, 1.0e-6f);
        float energy = energy_floor + energy_scale * powf(fmaxf(0.0f, x_dm), gamma);
        energy = fmaxf(0.0f, energy);

        float cos2phi = fminf(1.0f, fmaxf(-1.0f, 2.0f * xt[r] - 1.0f));
        float phi = 0.5f * acosf(cos2phi);
        float yz = energy * cosf(phi);
        float zy = -energy * sinf(phi);

        dm_write_rung_from_witnesses(r, 0.0f, yz, zy, 0.0f, base_delay, offset_mean, total_delay, 1.0f, rung_stats);
    }

    /*
    ------------------------------------------------------------------------------
    dm_geo_exact_sweep_summary_kernel_f32

    Batched exact GEO reference sweep. It is optional for benchmarks; it exists so
    GEO can be stress-tested without reintroducing the old aperture/tuning path.

    Grid:
        blockIdx.x = candidate
        blockIdx.y = condition_kind (0=null, 1=base_delay, 2=offset_deformed)

    Output:
        sweep_summary[(candidate, condition, metric)]
    ------------------------------------------------------------------------------
    */
    __global__ void dm_geo_exact_sweep_summary_kernel_f32(
        const int *base_delays_dt,
        const int n_rungs,
        const int offset_dt,
        const int n_candidates,
        const float *energy_floor,
        const float *energy_scale,
        const float *energy_gamma,
        const float *weight_space,
        const float *weight_time,
        float *sweep_summary)
    {
        int cand = blockIdx.x;
        int condition_kind = blockIdx.y;
        if (cand >= n_candidates || condition_kind >= 3 || threadIdx.x != 0 || n_rungs <= 0 || n_rungs > 64)
            return;

        float xs[64], xt[64];
        dm_geo_exact_compute_norms(condition_kind, base_delays_dt, n_rungs, offset_dt, xs, xt);

        float yz[64], zy[64], energy[64], spec[64], phase[64], x_total[64];
        float yz_sum = 0.0f, zy_sum = 0.0f, e_sum = 0.0f, s_sum = 0.0f;
        float e_max = 0.0f, s_max = -1.0e20f;
        int yz_pos = 0, zy_inv = 0;

        float ef = energy_floor[cand];
        float es = energy_scale[cand];
        float eg = fmaxf(energy_gamma[cand], 1.0e-6f);
        float ws = fmaxf(0.0f, weight_space[cand]);
        float wt = fmaxf(0.0f, weight_time[cand]);
        float wden = fmaxf(ws + wt, 1.0e-12f);

        for (int r = 0; r < n_rungs; ++r)
        {
            int use_base = (condition_kind == 0) ? 0 : base_delays_dt[r];
            int use_offset = (condition_kind == 2) ? offset_dt : 0;
            float base_delay = (float)use_base;
            float offset_mean = ((float)(4 * r) + 1.5f) * (float)use_offset;
            float total_delay = base_delay + offset_mean;

            float yv = 0.0f, zv = 0.0f, ev = 0.0f, ph = 0.0f;
            if (condition_kind != 0)
            {
                float x_dm = sqrtf((ws * xs[r] * xs[r] + wt * xt[r] * xt[r]) / wden);
                ev = fmaxf(0.0f, ef + es * powf(fmaxf(0.0f, x_dm), eg));
                float cos2phi = fminf(1.0f, fmaxf(-1.0f, 2.0f * xt[r] - 1.0f));
                ph = 0.5f * acosf(cos2phi);
                yv = ev * cosf(ph);
                zv = -ev * sinf(ph);
            }

            yz[r] = yv;
            zy[r] = zv;
            energy[r] = ev;
            spec[r] = ev; // comparison energy is exact zero in Probe-25 GEO.
            phase[r] = dm_wrap_pi(atan2f(-zv, yv));
            x_total[r] = total_delay;

            if (condition_kind != 0)
            {
                yz_sum += yv;
                zy_sum += zv;
                e_sum += ev;
                s_sum += ev;
                e_max = (r == 0) ? ev : fmaxf(e_max, ev);
                s_max = (r == 0) ? ev : fmaxf(s_max, ev);
                if (yv > DM_SIGN_EPS)
                    yz_pos++;
                if (yv * zv < -DM_SIGN_EPS)
                    zy_inv++;
            }
        }

        int n = (condition_kind == 0) ? 0 : n_rungs;
        int base_idx = (cand * 3 + condition_kind) * DM_SUMMARY_N_METRICS;
        if (n <= 0)
        {
            for (int m = 0; m < DM_SUMMARY_N_METRICS; ++m)
                sweep_summary[base_idx + m] = 0.0f;
            return;
        }

        float invn = 1.0f / (float)n;
        float yz_mean = yz_sum * invn;
        float zy_mean = zy_sum * invn;
        float e_mean = e_sum * invn;
        float s_mean = s_sum * invn;

        float x_lin[64], x_log[64];
        dm_norm_x_small(x_total, n, 0, x_lin);
        dm_norm_x_small(x_total, n, 1, x_log);
        float e_r_lin = dm_corr_small(x_lin, energy, n);
        float e_r_log = dm_corr_small(x_log, energy, n);
        float s_r_lin = dm_corr_small(x_lin, spec, n);
        float s_r_log = dm_corr_small(x_log, spec, n);
        float e_r = fabsf(e_r_log) > fabsf(e_r_lin) ? e_r_log : e_r_lin;
        float s_r = fabsf(s_r_log) > fabsf(s_r_lin) ? s_r_log : s_r_lin;
        float pi_lin = dm_pi_score_small(x_total, phase, n, 0);
        float pi_log = dm_pi_score_small(x_total, phase, n, 1);
        float pi_score = pi_log > pi_lin ? pi_log : pi_lin;
        float pi_mode = pi_log > pi_lin ? 1.0f : 0.0f;

        float phase_vel_r = 0.0f;
        float phase_span = 0.0f;
        if (n >= 3)
        {
            float mid_x[64], vel[64], mid_norm[64];
            int nv = 0;
            float acc = phase[0], pmin = acc, pmax = acc;
            for (int i = 1; i < n; ++i)
            {
                float dx = x_total[i] - x_total[i - 1];
                float dp = dm_wrap_pi_delta(phase[i] - phase[i - 1]);
                acc += dp;
                pmin = fminf(pmin, acc);
                pmax = fmaxf(pmax, acc);
                if (fabsf(dx) > 1.0e-12f)
                {
                    mid_x[nv] = 0.5f * (x_total[i] + x_total[i - 1]);
                    vel[nv] = dp / dx;
                    ++nv;
                }
            }
            phase_span = fabsf(pmax - pmin) / (float)M_PI;
            if (nv >= 3)
            {
                dm_norm_x_small(mid_x, nv, 1, mid_norm);
                phase_vel_r = dm_corr_small(mid_norm, vel, nv);
            }
        }

        float projection =
            0.35f * fmaxf(0.0f, e_mean) +
            0.25f * fmaxf(0.0f, s_mean) +
            0.15f * fmaxf(0.0f, yz_mean) +
            0.15f * (fmaxf(0.0f, e_mean) * pi_score) +
            0.10f * (0.5f * (fabsf(e_r) + fabsf(s_r)));

        sweep_summary[base_idx + DM_SUMMARY_N_RUNGS] = (float)n;
        sweep_summary[base_idx + DM_SUMMARY_YZ_MEAN] = yz_mean;
        sweep_summary[base_idx + DM_SUMMARY_YZ_POS_FRAC] = (float)yz_pos * invn;
        sweep_summary[base_idx + DM_SUMMARY_ZY_MEAN] = zy_mean;
        sweep_summary[base_idx + DM_SUMMARY_ZY_INVERTED_FRAC] = (float)zy_inv * invn;
        sweep_summary[base_idx + DM_SUMMARY_YZZY_ENERGY_MEAN] = e_mean;
        sweep_summary[base_idx + DM_SUMMARY_YZZY_ENERGY_MAX] = e_max;
        sweep_summary[base_idx + DM_SUMMARY_SPECIFICITY_MEAN] = s_mean;
        sweep_summary[base_idx + DM_SUMMARY_SPECIFICITY_MAX] = s_max;
        sweep_summary[base_idx + DM_SUMMARY_PI_PERIODIC_SCORE] = pi_score;
        sweep_summary[base_idx + DM_SUMMARY_PI_PERIODIC_MODE] = pi_mode;
        sweep_summary[base_idx + DM_SUMMARY_ENERGY_TRACKING_R] = e_r;
        sweep_summary[base_idx + DM_SUMMARY_SPECIFICITY_TRACKING_R] = s_r;
        sweep_summary[base_idx + DM_SUMMARY_PHASE_VELOCITY_R] = phase_vel_r;
        sweep_summary[base_idx + DM_SUMMARY_PHASE_SPAN_PI_UNITS] = phase_span;
        sweep_summary[base_idx + DM_SUMMARY_PROJECTION_SCORE] = projection;
    }

    // -----------------------------------------------------------------------------
    // 5. DER QUERY / RETRIEVAL MEGAKERNEL
    // -----------------------------------------------------------------------------

#ifndef DM_DER_MAX_TOPK
#define DM_DER_MAX_TOPK 16
#endif

#ifndef DM_DER_MAX_THREADS
#define DM_DER_MAX_THREADS 256
#endif

    enum DMDERBackend
    {
        DM_DER_ENERGY = 0,
        DM_DER_PI_FIT = 1,
        DM_DER_FLAT_COSINE = 2,
        DM_DER_MANIFOLD = 3
    };

    __device__ __forceinline__ void dm_der_insert_topk(
        float score,
        int idx,
        float *scores,
        int *indices,
        int top_k)
    {
        if (score <= scores[top_k - 1])
            return;

        int pos = top_k - 1;
        while (pos > 0 && score > scores[pos - 1])
        {
            scores[pos] = scores[pos - 1];
            indices[pos] = indices[pos - 1];
            --pos;
        }

        scores[pos] = score;
        indices[pos] = idx;
    }

    __device__ __forceinline__ float dm_der_score_candidate(
        const int q,
        const int k,
        const int backend_kind,
        const int rungs,
        const float *q_yz,
        const float *q_ret,
        const float *q_energy,
        const float *q_spec,
        const float *q_cos2,
        const float *q_sin2,
        const float *k_yz,
        const float *k_ret,
        const float *k_energy,
        const float *k_spec,
        const float *k_cos2,
        const float *k_sin2)
    {
        float s_dir = 0.0f;
        float s_pi = 0.0f;
        float s_e = 0.0f;
        float s_s = 0.0f;

        float dot = 0.0f;
        float qn = 0.0f;
        float kn = 0.0f;

        for (int r = 0; r < rungs; ++r)
        {
            int qi = q * rungs + r;
            int ki = k * rungs + r;

            float qyz = q_yz[qi];
            float qrt = q_ret[qi];
            float qen = q_energy[qi];
            float qsp = q_spec[qi];
            float qco = q_cos2[qi];
            float qsi = q_sin2[qi];

            float kyz = k_yz[ki];
            float krt = k_ret[ki];
            float ken = k_energy[ki];
            float ksp = k_spec[ki];
            float kco = k_cos2[ki];
            float ksi = k_sin2[ki];

            if (backend_kind == DM_DER_FLAT_COSINE)
            {
                dot += qyz * kyz + qrt * krt + qen * ken + qsp * ksp + qco * kco + qsi * ksi;
                qn += qyz * qyz + qrt * qrt + qen * qen + qsp * qsp + qco * qco + qsi * qsi;
                kn += kyz * kyz + krt * krt + ken * ken + ksp * ksp + kco * kco + ksi * ksi;
            }
            else
            {
                float dyz = qyz - kyz;
                float drt = qrt - krt;
                float de = qen - ken;
                float ds = qsp - ksp;
                float dc = qco - kco;
                float dsn = qsi - ksi;

                s_dir += -(dyz * dyz + drt * drt);
                s_pi += -(dc * dc + dsn * dsn);
                s_e += -(de * de);
                s_s += -(ds * ds);
            }
        }

        if (backend_kind == DM_DER_ENERGY)
            return s_e;

        if (backend_kind == DM_DER_PI_FIT)
            return s_pi;

        if (backend_kind == DM_DER_FLAT_COSINE)
        {
            float denom = sqrtf(fmaxf(qn, 1.0e-20f) * fmaxf(kn, 1.0e-20f));
            return dot / denom;
        }

        // D_M paired manifold distance:
        //   YZ/return directional mismatch
        //   pi-phase mismatch
        //   energy mismatch
        //   specificity mismatch
        return 0.46f * s_dir + 0.24f * s_pi + 0.18f * s_e + 0.12f * s_s;
    }

    /*
    ------------------------------------------------------------------------------
    dm_der_topk_kernel_f32

    Fused D_M DER query/top-k kernel.

    This kernel covers both classical DER tasks:

        is_local = 1:
            matched local hard-negative set.
            Candidates are:
                truth[q] + 0 ... local_width-1

        is_local = 0:
            global bank stress.
            Candidates are:
                0 ... n_keys-1

    backend_kind:
        0 = energy profile distance
        1 = pi-fit profile distance
        2 = flat full-feature cosine
        3 = D_M paired manifold distance

    Grid:
        blockIdx.x = query

    Block:
        threads cooperatively scan candidates, each thread keeps a local top-k,
        then thread 0 merges the block-local candidates.

    Output:
        out_indices[query, top_k]
        out_scores[query, top_k]
    ------------------------------------------------------------------------------
    */
    __global__ void dm_der_topk_kernel_f32(
        const float *q_yz,
        const float *q_ret,
        const float *q_energy,
        const float *q_spec,
        const float *q_cos2,
        const float *q_sin2,
        const float *k_yz,
        const float *k_ret,
        const float *k_energy,
        const float *k_spec,
        const float *k_cos2,
        const float *k_sin2,
        const long long *truth,
        const int n_queries,
        const int n_keys,
        const int rungs,
        const int backend_kind,
        const int is_local,
        const int local_width,
        const int top_k,
        int *out_indices,
        float *out_scores)
    {
        int q = blockIdx.x;
        int tid = threadIdx.x;
        int nt = blockDim.x;

        if (q >= n_queries || nt > DM_DER_MAX_THREADS)
            return;

        if (top_k <= 0 || top_k > DM_DER_MAX_TOPK)
            return;

        __shared__ float sh_scores[DM_DER_MAX_THREADS * DM_DER_MAX_TOPK];
        __shared__ int sh_indices[DM_DER_MAX_THREADS * DM_DER_MAX_TOPK];

        float local_scores[DM_DER_MAX_TOPK];
        int local_indices[DM_DER_MAX_TOPK];

        for (int i = 0; i < top_k; ++i)
        {
            local_scores[i] = -3.402823466e+38f;
            local_indices[i] = -1;
        }

        int start = 0;
        int count = n_keys;

        if (is_local)
        {
            start = (int)truth[q];
            count = local_width;
        }

        for (int j = tid; j < count; j += nt)
        {
            int cand = start + j;

            if (cand < 0 || cand >= n_keys)
                continue;

            float score = dm_der_score_candidate(
                q,
                cand,
                backend_kind,
                rungs,
                q_yz,
                q_ret,
                q_energy,
                q_spec,
                q_cos2,
                q_sin2,
                k_yz,
                k_ret,
                k_energy,
                k_spec,
                k_cos2,
                k_sin2);

            dm_der_insert_topk(score, cand, local_scores, local_indices, top_k);
        }

        int base = tid * top_k;
        for (int i = 0; i < top_k; ++i)
        {
            sh_scores[base + i] = local_scores[i];
            sh_indices[base + i] = local_indices[i];
        }

        __syncthreads();

        if (tid == 0)
        {
            float best_scores[DM_DER_MAX_TOPK];
            int best_indices[DM_DER_MAX_TOPK];

            for (int i = 0; i < top_k; ++i)
            {
                best_scores[i] = -3.402823466e+38f;
                best_indices[i] = -1;
            }

            for (int t = 0; t < nt; ++t)
            {
                int b = t * top_k;
                for (int i = 0; i < top_k; ++i)
                {
                    int idx = sh_indices[b + i];
                    float score = sh_scores[b + i];

                    if (idx >= 0)
                        dm_der_insert_topk(score, idx, best_scores, best_indices, top_k);
                }
            }

            int out_base = q * top_k;
            for (int i = 0; i < top_k; ++i)
            {
                out_indices[out_base + i] = best_indices[i];
                out_scores[out_base + i] = best_scores[i];
            }
        }
    }

} // extern "C"
