/*
==============================================================================
GHOST ORACLE SUITE — F_M PROJECTOR CUDA KERNEL
==============================================================================

Purpose
-------
CUDA projector kernel for F_M.

This file now supports three F_M substrate paths:

    qproj:
        raw QPU records:
            g[tile, shot, bit]
            em[tile, shot, bit]

    gproj:
        GPU-generated synthetic/emulated records with the same schema:
            g[tile, shot, bit]
            em[tile, shot, bit]

    geo:
        optimized classical analytic response path:
            tile_delay_dt[tile]
            tile_scale_level[tile]
            tile_theta[tile]
            mode_id[tile]
        -> analytic response curves
        -> wave metrics

Core discovered F_M signature
-----------------------------
Current discovered F_M projector signature:

    primary:
        xor_delta / bit_diff / delay

    runners-up:
        xor_delta / bit1_mean / delay
        delta     / transition / delay
        delta     / bit_diff / delay

The geo path is designed to compute these response curves directly, without
sampling shots.

Kernels
-------
1. fm_response_kernel_u8

    Computes response curves from g/em records.

    field_kind:
        0 = delta      signed em - g
        1 = xor_delta  binary em XOR g
        2 = g          raw g bit values
        3 = em         raw em bit values

    response_kind:
        0 = mean
        1 = energy
        2 = transition
        3 = imbalance
        4 = bit0_mean
        5 = bit1_mean
        6 = bit_diff

    Output:
        out[field_index, response_index, tile]

2. fm_path_pair_break_response_kernel_u8

    Deterministic path-pair destruction control for qproj/gproj records.

3. fm_geo_curve_kernel_f32

    Computes analytic F_M geo curves directly from tile metadata and formula
    parameters. This is the optimized classical path.

    Fixed geo curve indices:
        0 = xor_delta / bit_diff
        1 = xor_delta / bit1_mean
        2 = xor_delta / transition
        3 = xor_delta / energy
        4 = delta     / bit_diff
        5 = delta     / bit1_mean
        6 = delta     / transition
        7 = delta     / energy

    Output:
        curves[curve_index, ordered_point]

4. fm_wave_metric_kernel_f32

    Computes wave metrics for each curve:

        metric 0 = wave_score
        metric 1 = peak_ratio
        metric 2 = spectral_entropy
        metric 3 = best_r2
        metric 4 = best_freq
        metric 5 = best_amp
        metric 6 = best_phase
        metric 7 = low_high_ratio

Notes
-----
This kernel is built for operator probing and optimized projector evaluation.
The response kernels are correctness-first for the current small-tile qproj
base. The geo path is the first minimal classical F_M operator path.

==============================================================================
*/

extern "C"
{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

    // =========================================================================
    // SHARED INDEX HELPERS
    // =========================================================================

    __device__ __forceinline__ int fm_idx3(
        int tile,
        int shot,
        int bit,
        int shots,
        int bits)
    {
        return (tile * shots + shot) * bits + bit;
    }

    __device__ __forceinline__ int fm_idx3_field(
        int field,
        int response,
        int tile,
        int n_responses,
        int tiles)
    {
        return (field * n_responses + response) * tiles + tile;
    }

    __device__ __forceinline__ int fm_curve_idx(
        int curve,
        int point,
        int n_points)
    {
        return curve * n_points + point;
    }

    // =========================================================================
    // QPROJ / GPROJ RESPONSE PATH
    // =========================================================================

    __device__ __forceinline__ float fm_value_at(
        const unsigned char *g,
        const unsigned char *em,
        int field_kind,
        int tile,
        int shot,
        int bit,
        int shots,
        int bits)
    {
        int idx = fm_idx3(tile, shot, bit, shots, bits);
        unsigned char gv = g[idx];
        unsigned char ev = em[idx];

        if (field_kind == 0)
        {
            return (float)((int)ev - (int)gv); // delta
        }
        if (field_kind == 1)
        {
            return (float)((unsigned char)(ev ^ gv)); // xor_delta
        }
        if (field_kind == 2)
        {
            return (float)gv; // g
        }
        return (float)ev; // em
    }

    __device__ __forceinline__ float fm_response_value(
        const unsigned char *g,
        const unsigned char *em,
        int field_kind,
        int response_kind,
        int tile,
        int shots,
        int bits)
    {
        float sum = 0.0f;
        float sum2 = 0.0f;
        float pos = 0.0f;
        float neg = 0.0f;
        float transitions = 0.0f;

        if (response_kind == 4)
        {
            int bit = 0;
            for (int s = 0; s < shots; ++s)
            {
                sum += fm_value_at(g, em, field_kind, tile, s, bit, shots, bits);
            }
            return sum / (float)shots;
        }

        if (response_kind == 5)
        {
            int bit = bits > 1 ? 1 : 0;
            for (int s = 0; s < shots; ++s)
            {
                sum += fm_value_at(g, em, field_kind, tile, s, bit, shots, bits);
            }
            return sum / (float)shots;
        }

        if (response_kind == 6)
        {
            if (bits < 2)
                return 0.0f;

            float s0 = 0.0f;
            float s1 = 0.0f;

            for (int s = 0; s < shots; ++s)
            {
                s0 += fm_value_at(g, em, field_kind, tile, s, 0, shots, bits);
                s1 += fm_value_at(g, em, field_kind, tile, s, 1, shots, bits);
            }

            return (s1 - s0) / (float)shots;
        }

        float prev = 0.0f;
        bool have_prev = false;
        int n = shots * bits;

        for (int s = 0; s < shots; ++s)
        {
            for (int b = 0; b < bits; ++b)
            {
                float v = fm_value_at(g, em, field_kind, tile, s, b, shots, bits);

                sum += v;
                sum2 += v * v;

                if (v > 0.0f)
                    pos += 1.0f;
                if (v < 0.0f)
                    neg += 1.0f;

                if (have_prev && v != prev)
                    transitions += 1.0f;

                prev = v;
                have_prev = true;
            }
        }

        if (response_kind == 0)
            return sum / (float)n;

        if (response_kind == 1)
            return sum2 / (float)n;

        if (response_kind == 2)
            return transitions / (float)(n > 1 ? n - 1 : 1);

        if (response_kind == 3)
            return (pos - neg) / (float)n;

        return 0.0f;
    }

    /*
    ------------------------------------------------------------------------------
    fm_response_kernel_u8

    Grid:
        blockIdx.x = tile
        blockIdx.y = response index
        blockIdx.z = field index

    Output:
        out[field_index, response_index, tile]
    ------------------------------------------------------------------------------
    */
    __global__ void fm_response_kernel_u8(
        const unsigned char *g,
        const unsigned char *em,
        const int tiles,
        const int shots,
        const int bits,
        const int *field_kinds,
        const int n_fields,
        const int *response_kinds,
        const int n_responses,
        float *out)
    {
        int tile = blockIdx.x;
        int response_i = blockIdx.y;
        int field_i = blockIdx.z;

        if (tile >= tiles || response_i >= n_responses || field_i >= n_fields)
            return;

        if (threadIdx.x == 0)
        {
            int field_kind = field_kinds[field_i];
            int response_kind = response_kinds[response_i];

            float val = fm_response_value(
                g,
                em,
                field_kind,
                response_kind,
                tile,
                shots,
                bits);

            int out_idx = fm_idx3_field(field_i, response_i, tile, n_responses, tiles);
            out[out_idx] = val;
        }
    }

    // =========================================================================
    // PATH-PAIR BREAK CONTROL PATH
    // =========================================================================

    __device__ __forceinline__ int fm_perm_shot(
        int s,
        int shots,
        int a,
        int b)
    {
        return (int)(((long long)a * (long long)s + (long long)b) % (long long)shots);
    }

    __device__ __forceinline__ float fm_broken_value_at(
        const unsigned char *g,
        const unsigned char *em,
        int field_kind,
        int tile,
        int shot,
        int bit,
        int shots,
        int bits,
        int seed)
    {
        int ag = 1103515245 | 1;
        int ae = 1664525 | 1;

        int bg = (seed * 1013904223 + tile * 9176 + bit * 131) & 0x7fffffff;
        int be = (seed * 1664525 + tile * 49157 + bit * 257) & 0x7fffffff;

        int sg = fm_perm_shot(shot, shots, ag, bg);
        int se = fm_perm_shot(shot, shots, ae, be);

        int idx_g = fm_idx3(tile, sg, bit, shots, bits);
        int idx_e = fm_idx3(tile, se, bit, shots, bits);

        unsigned char gv = g[idx_g];
        unsigned char ev = em[idx_e];

        if (field_kind == 0)
            return (float)((int)ev - (int)gv);

        if (field_kind == 1)
            return (float)((unsigned char)(ev ^ gv));

        if (field_kind == 2)
            return (float)gv;

        return (float)ev;
    }

    __device__ __forceinline__ float fm_broken_response_value(
        const unsigned char *g,
        const unsigned char *em,
        int field_kind,
        int response_kind,
        int tile,
        int shots,
        int bits,
        int seed)
    {
        float sum = 0.0f;
        float sum2 = 0.0f;
        float pos = 0.0f;
        float neg = 0.0f;
        float transitions = 0.0f;

        if (response_kind == 4)
        {
            int bit = 0;
            for (int s = 0; s < shots; ++s)
            {
                sum += fm_broken_value_at(g, em, field_kind, tile, s, bit, shots, bits, seed);
            }
            return sum / (float)shots;
        }

        if (response_kind == 5)
        {
            int bit = bits > 1 ? 1 : 0;
            for (int s = 0; s < shots; ++s)
            {
                sum += fm_broken_value_at(g, em, field_kind, tile, s, bit, shots, bits, seed);
            }
            return sum / (float)shots;
        }

        if (response_kind == 6)
        {
            if (bits < 2)
                return 0.0f;

            float s0 = 0.0f;
            float s1 = 0.0f;

            for (int s = 0; s < shots; ++s)
            {
                s0 += fm_broken_value_at(g, em, field_kind, tile, s, 0, shots, bits, seed);
                s1 += fm_broken_value_at(g, em, field_kind, tile, s, 1, shots, bits, seed);
            }

            return (s1 - s0) / (float)shots;
        }

        float prev = 0.0f;
        bool have_prev = false;
        int n = shots * bits;

        for (int s = 0; s < shots; ++s)
        {
            for (int b = 0; b < bits; ++b)
            {
                float v = fm_broken_value_at(g, em, field_kind, tile, s, b, shots, bits, seed);

                sum += v;
                sum2 += v * v;

                if (v > 0.0f)
                    pos += 1.0f;
                if (v < 0.0f)
                    neg += 1.0f;

                if (have_prev && v != prev)
                    transitions += 1.0f;

                prev = v;
                have_prev = true;
            }
        }

        if (response_kind == 0)
            return sum / (float)n;

        if (response_kind == 1)
            return sum2 / (float)n;

        if (response_kind == 2)
            return transitions / (float)(n > 1 ? n - 1 : 1);

        if (response_kind == 3)
            return (pos - neg) / (float)n;

        return 0.0f;
    }

    /*
    ------------------------------------------------------------------------------
    fm_path_pair_break_response_kernel_u8

    Grid:
        blockIdx.x = tile
        blockIdx.y = response index
        blockIdx.z = field index
    ------------------------------------------------------------------------------
    */
    __global__ void fm_path_pair_break_response_kernel_u8(
        const unsigned char *g,
        const unsigned char *em,
        const int tiles,
        const int shots,
        const int bits,
        const int *field_kinds,
        const int n_fields,
        const int *response_kinds,
        const int n_responses,
        const int seed,
        float *out)
    {
        int tile = blockIdx.x;
        int response_i = blockIdx.y;
        int field_i = blockIdx.z;

        if (tile >= tiles || response_i >= n_responses || field_i >= n_fields)
            return;

        if (threadIdx.x == 0)
        {
            int field_kind = field_kinds[field_i];
            int response_kind = response_kinds[response_i];

            float val = fm_broken_response_value(
                g,
                em,
                field_kind,
                response_kind,
                tile,
                shots,
                bits,
                seed);

            int out_idx = fm_idx3_field(field_i, response_i, tile, n_responses, tiles);
            out[out_idx] = val;
        }
    }

    // =========================================================================
    // GEO PATH
    // =========================================================================

    __device__ __forceinline__ float fm_mode_phase(int mode_id)
    {
        if (mode_id == 1)
            return 0.37f;
        if (mode_id == 2)
            return 0.74f;
        return 0.0f;
    }

    __device__ __forceinline__ float fm_delay_norm_from_metadata(
        float delay,
        float max_delay)
    {
        if (fabsf(max_delay) > 1.0e-12f)
            return delay / max_delay;
        return 0.0f;
    }

    __device__ __forceinline__ float fm_geo_curve_value(
        int curve_kind,
        float delay,
        float scale_level,
        float theta,
        int mode_id,
        float max_delay,
        float wave_freq,
        float phase0,
        float bitdiff_amp,
        float bit1_amp,
        float transition_amp,
        float energy_amp,
        float scale_phase,
        float theta_phase,
        float base_xor,
        float base_delta)
    {
        float delay_norm = fm_delay_norm_from_metadata(delay, max_delay);
        float scale_term = scale_phase * log2f(fmaxf(1.0f, scale_level) + 1.0f);
        float theta_term = theta_phase * theta;
        float mode_term = fm_mode_phase(mode_id);

        float phi =
            2.0f * (float)M_PI * wave_freq * delay_norm + phase0 + scale_term + theta_term + mode_term;

        float phi2 =
            2.0f * (float)M_PI * (wave_freq + 1.1f) * delay_norm + 0.5f * phase0 + scale_term + 0.5f * mode_term;

        // Fixed curve indices:
        // 0 = xor_delta / bit_diff
        // 1 = xor_delta / bit1_mean
        // 2 = xor_delta / transition
        // 3 = xor_delta / energy
        // 4 = delta     / bit_diff
        // 5 = delta     / bit1_mean
        // 6 = delta     / transition
        // 7 = delta     / energy

        if (curve_kind == 0)
        {
            return base_xor + bitdiff_amp * sinf(phi);
        }

        if (curve_kind == 1)
        {
            return 0.11f + bit1_amp * sinf(phi + 0.19f);
        }

        if (curve_kind == 2)
        {
            return 0.50f + transition_amp * sinf(phi2);
        }

        if (curve_kind == 3)
        {
            return 0.12f + energy_amp * sinf(phi2 + 0.33f);
        }

        if (curve_kind == 4)
        {
            return base_delta + 0.78f * bitdiff_amp * sinf(phi + 0.34f);
        }

        if (curve_kind == 5)
        {
            return 0.02f + 0.70f * bit1_amp * sinf(phi + 0.52f);
        }

        if (curve_kind == 6)
        {
            return 0.50f + transition_amp * sinf(phi2 + 0.06f);
        }

        return 0.12f + energy_amp * sinf(phi2 + 0.33f);
    }

    /*
    ------------------------------------------------------------------------------
    fm_geo_curve_kernel_f32

    Computes analytic F_M geo response curves.

    Grid:
        blockIdx.x = curve kind
        threadIdx.x strides ordered points

    Inputs:
        delays[tile]         float32
        scales[tile]         float32
        theta[tile]          float32
        mode_id[tile]        int32
        order_indices[point] int32, maps ordered point -> tile index
        max_delay            float32

    Output:
        curves[curve_kind, point]

    Fixed curve kinds:
        0 = xor_delta / bit_diff
        1 = xor_delta / bit1_mean
        2 = xor_delta / transition
        3 = xor_delta / energy
        4 = delta     / bit_diff
        5 = delta     / bit1_mean
        6 = delta     / transition
        7 = delta     / energy

    This is the minimal classical F_M path. It avoids shot sampling and computes
    the same projector-facing response curves directly from metadata.
    ------------------------------------------------------------------------------
    */
    __global__ void fm_geo_curve_kernel_f32(
        const float *delays,
        const float *scales,
        const float *theta,
        const int *mode_id,
        const int *order_indices,
        const int n_points,
        const float max_delay,
        const float wave_freq,
        const float phase0,
        const float bitdiff_amp,
        const float bit1_amp,
        const float transition_amp,
        const float energy_amp,
        const float scale_phase,
        const float theta_phase,
        const float base_xor,
        const float base_delta,
        float *curves)
    {
        int curve_kind = blockIdx.x;

        if (curve_kind >= 8)
            return;

        for (int p = threadIdx.x; p < n_points; p += blockDim.x)
        {
            int tile = order_indices[p];

            float delay = delays[tile];
            float scale_level = scales[tile];
            float th = theta[tile];
            int mode = mode_id[tile];

            float val = fm_geo_curve_value(
                curve_kind,
                delay,
                scale_level,
                th,
                mode,
                max_delay,
                wave_freq,
                phase0,
                bitdiff_amp,
                bit1_amp,
                transition_amp,
                energy_amp,
                scale_phase,
                theta_phase,
                base_xor,
                base_delta);

            curves[fm_curve_idx(curve_kind, p, n_points)] = val;
        }
    }

    /*
    ------------------------------------------------------------------------------
    fm_geo_sweep_kernel_f32

    Optimized parameter sweep for the geo path.

    Each block evaluates one parameter candidate. Each block writes:

        sweep_metrics[candidate, curve, metric]

    Candidate parameter arrays:
        wave_freqs[candidate]
        phase0s[candidate]
        bitdiff_amps[candidate]
        bit1_amps[candidate]
        transition_amps[candidate]
        energy_amps[candidate]
        scale_phases[candidate]
        theta_phases[candidate]
        base_xors[candidate]
        base_deltas[candidate]

    This kernel computes curves and then calls the same metric logic inline for
    each of the 8 fixed geo curves.

    For simplicity and robustness, each block uses thread 0 for the small-N
    metric work. This is still fast for large sweeps because the work per
    candidate is tiny and avoids Python-side parameter loops.
    ------------------------------------------------------------------------------
    */

    // Forward declarations for metric helper functions used below.
    __device__ __forceinline__ float fm_curve_mean_inline(const float *y, int n_points)
    {
        float s = 0.0f;
        for (int i = 0; i < n_points; ++i)
            s += y[i];
        return s / (float)n_points;
    }

    __device__ __forceinline__ float fm_curve_var_inline(const float *y, int n_points, float mean)
    {
        float s = 0.0f;
        for (int i = 0; i < n_points; ++i)
        {
            float d = y[i] - mean;
            s += d * d;
        }
        return s / (float)n_points;
    }

    __device__ void fm_wave_metrics_inline(
        const float *y,
        const float *xs,
        const int n_points,
        float *out_metrics)
    {
        const int n_metrics = 8;

        float mean = fm_curve_mean_inline(y, n_points);
        float var = fm_curve_var_inline(y, n_points, mean);

        if (n_points < 3 || var < 1.0e-12f)
        {
            for (int m = 0; m < n_metrics; ++m)
                out_metrics[m] = 0.0f;
            return;
        }

        int n_freq = n_points / 2 + 1;
        float total_power = 0.0f;
        float peak_power = -1.0f;
        int peak_idx = 0;
        float peak_re = 0.0f;
        float peak_im = 0.0f;

        for (int k = 1; k < n_freq; ++k)
        {
            float re = 0.0f;
            float im = 0.0f;

            for (int i = 0; i < n_points; ++i)
            {
                float yy = y[i] - mean;
                float ang = -2.0f * (float)M_PI * (float)k * (float)i / (float)n_points;
                re += yy * cosf(ang);
                im += yy * sinf(ang);
            }

            float p = re * re + im * im;
            total_power += p;

            if (p > peak_power)
            {
                peak_power = p;
                peak_idx = k;
                peak_re = re;
                peak_im = im;
            }
        }

        total_power += 1.0e-12f;
        float peak_ratio = peak_power / total_power;

        float spectral_entropy = 0.0f;
        int bins = max(1, n_freq - 1);

        for (int k = 1; k < n_freq; ++k)
        {
            float re = 0.0f;
            float im = 0.0f;

            for (int i = 0; i < n_points; ++i)
            {
                float yy = y[i] - mean;
                float ang = -2.0f * (float)M_PI * (float)k * (float)i / (float)n_points;
                re += yy * cosf(ang);
                im += yy * sinf(ang);
            }

            float p = (re * re + im * im) / total_power;
            if (p > 1.0e-12f)
                spectral_entropy += -p * logf(p);
        }

        if (bins > 1)
            spectral_entropy /= logf((float)bins);

        float low = 0.0f;
        float high = 0.0f;
        int half = max(1, bins / 2);

        for (int k = 1; k < n_freq; ++k)
        {
            float re = 0.0f;
            float im = 0.0f;

            for (int i = 0; i < n_points; ++i)
            {
                float yy = y[i] - mean;
                float ang = -2.0f * (float)M_PI * (float)k * (float)i / (float)n_points;
                re += yy * cosf(ang);
                im += yy * sinf(ang);
            }

            float p = re * re + im * im;
            if ((k - 1) < half)
                low += p;
            else
                high += p;
        }

        float low_high_ratio = low / (high + 1.0e-12f);
        float dominant_phase = atan2f(peak_im, peak_re);

        float best_r2 = -1.0e20f;
        float best_freq = 0.0f;
        float best_amp = 0.0f;
        float best_phase = 0.0f;

        float ss_tot = 0.0f;
        for (int i = 0; i < n_points; ++i)
        {
            float d = y[i] - mean;
            ss_tot += d * d;
        }
        ss_tot += 1.0e-12f;

        float xmin = xs[0];
        float xmax = xs[0];

        for (int i = 1; i < n_points; ++i)
        {
            xmin = fminf(xmin, xs[i]);
            xmax = fmaxf(xmax, xs[i]);
        }

        for (int step = 0; step < 26; ++step)
        {
            float freq = 0.5f + (3.0f - 0.5f) * ((float)step / 25.0f);
            float w = 2.0f * (float)M_PI * freq;

            float s_s = 0.0f, s_c = 0.0f, s_1 = (float)n_points;
            float ss = 0.0f, cc = 0.0f, sc = 0.0f;
            float sy = 0.0f, cy = 0.0f, yy_sum = 0.0f;

            for (int i = 0; i < n_points; ++i)
            {
                float x = xs[i];

                if (xmax > xmin)
                    x = (x - xmin) / (xmax - xmin);
                else
                    x = (float)i / (float)max(1, n_points - 1);

                float sv = sinf(w * x);
                float cv = cosf(w * x);
                float yv = y[i];

                s_s += sv;
                s_c += cv;
                ss += sv * sv;
                cc += cv * cv;
                sc += sv * cv;
                sy += sv * yv;
                cy += cv * yv;
                yy_sum += yv;
            }

            float A[3][4];

            A[0][0] = ss;
            A[0][1] = sc;
            A[0][2] = s_s;
            A[0][3] = sy;

            A[1][0] = sc;
            A[1][1] = cc;
            A[1][2] = s_c;
            A[1][3] = cy;

            A[2][0] = s_s;
            A[2][1] = s_c;
            A[2][2] = s_1;
            A[2][3] = yy_sum;

            bool ok = true;

            for (int col = 0; col < 3; ++col)
            {
                int piv = col;
                float best = fabsf(A[col][col]);

                for (int r = col + 1; r < 3; ++r)
                {
                    float v = fabsf(A[r][col]);
                    if (v > best)
                    {
                        best = v;
                        piv = r;
                    }
                }

                if (best < 1.0e-12f)
                {
                    ok = false;
                    break;
                }

                if (piv != col)
                {
                    for (int c = col; c < 4; ++c)
                    {
                        float tmp = A[col][c];
                        A[col][c] = A[piv][c];
                        A[piv][c] = tmp;
                    }
                }

                float div = A[col][col];

                for (int c = col; c < 4; ++c)
                    A[col][c] /= div;

                for (int r = 0; r < 3; ++r)
                {
                    if (r == col)
                        continue;

                    float f = A[r][col];

                    for (int c = col; c < 4; ++c)
                        A[r][c] -= f * A[col][c];
                }
            }

            if (!ok)
                continue;

            float a = A[0][3];
            float b = A[1][3];
            float c0 = A[2][3];

            float ss_res = 0.0f;

            for (int i = 0; i < n_points; ++i)
            {
                float x = xs[i];

                if (xmax > xmin)
                    x = (x - xmin) / (xmax - xmin);
                else
                    x = (float)i / (float)max(1, n_points - 1);

                float pred = a * sinf(w * x) + b * cosf(w * x) + c0;
                float err = y[i] - pred;
                ss_res += err * err;
            }

            float r2 = 1.0f - ss_res / ss_tot;
            float amp = sqrtf(a * a + b * b);
            float phase = atan2f(b, a);

            if (!isfinite(amp) || amp > 10.0f || !isfinite(r2))
                continue;

            if (r2 > best_r2)
            {
                best_r2 = r2;
                best_freq = freq;
                best_amp = amp;
                best_phase = phase;
            }
        }

        if (!isfinite(best_r2) || best_r2 < 0.0f)
        {
            best_r2 = 0.0f;
            best_freq = 0.0f;
            best_amp = 0.0f;
            best_phase = 0.0f;
        }

        float score =
            0.40f * peak_ratio + 0.25f * best_r2 + 0.20f * (1.0f - spectral_entropy) + 0.15f * fminf(1.0f, fabsf(low_high_ratio) / 10.0f);

        out_metrics[0] = score;
        out_metrics[1] = peak_ratio;
        out_metrics[2] = spectral_entropy;
        out_metrics[3] = best_r2;
        out_metrics[4] = best_freq;
        out_metrics[5] = best_amp;
        out_metrics[6] = best_phase;
        out_metrics[7] = low_high_ratio;
    }

    __global__ void fm_geo_sweep_kernel_f32(
        const float *delays,
        const float *scales,
        const float *theta,
        const int *mode_id,
        const int *order_indices,
        const float *xs,
        const int n_points,
        const float max_delay,
        const float *wave_freqs,
        const float *phase0s,
        const float *bitdiff_amps,
        const float *bit1_amps,
        const float *transition_amps,
        const float *energy_amps,
        const float *scale_phases,
        const float *theta_phases,
        const float *base_xors,
        const float *base_deltas,
        const int n_candidates,
        float *sweep_metrics)
    {
        int cand = blockIdx.x;

        if (cand >= n_candidates)
            return;

        if (threadIdx.x != 0)
            return;

        const int n_curves = 8;
        const int n_metrics = 8;

        float wf = wave_freqs[cand];
        float ph = phase0s[cand];
        float bda = bitdiff_amps[cand];
        float b1a = bit1_amps[cand];
        float tra = transition_amps[cand];
        float ena = energy_amps[cand];
        float scp = scale_phases[cand];
        float thp = theta_phases[cand];
        float bx = base_xors[cand];
        float bd = base_deltas[cand];

        // The current F_M bases use <= 64 ordered points. Keep a simple stack
        // buffer for fast small-N curve metrics.
        float ybuf[64];

        if (n_points > 64)
            return;

        for (int curve = 0; curve < n_curves; ++curve)
        {
            for (int p = 0; p < n_points; ++p)
            {
                int tile = order_indices[p];

                ybuf[p] = fm_geo_curve_value(
                    curve,
                    delays[tile],
                    scales[tile],
                    theta[tile],
                    mode_id[tile],
                    max_delay,
                    wf,
                    ph,
                    bda,
                    b1a,
                    tra,
                    ena,
                    scp,
                    thp,
                    bx,
                    bd);
            }

            float metrics_local[8];
            fm_wave_metrics_inline(ybuf, xs, n_points, metrics_local);

            for (int m = 0; m < n_metrics; ++m)
            {
                int out_idx = (cand * n_curves + curve) * n_metrics + m;
                sweep_metrics[out_idx] = metrics_local[m];
            }
        }
    }

    // =========================================================================
    // WAVE METRIC KERNEL FOR PRECOMPUTED CURVES
    // =========================================================================

    __device__ __forceinline__ float fm_curve_mean(
        const float *curves,
        int curve,
        int n_points)
    {
        float s = 0.0f;

        for (int i = 0; i < n_points; ++i)
            s += curves[curve * n_points + i];

        return s / (float)n_points;
    }

    __device__ __forceinline__ float fm_curve_var(
        const float *curves,
        int curve,
        int n_points,
        float mean)
    {
        float s = 0.0f;

        for (int i = 0; i < n_points; ++i)
        {
            float d = curves[curve * n_points + i] - mean;
            s += d * d;
        }

        return s / (float)n_points;
    }

    __global__ void fm_wave_metric_kernel_f32(
        const float *curves,
        const float *xs,
        const int n_curves,
        const int n_points,
        float *metrics)
    {
        int curve = blockIdx.x;

        if (curve >= n_curves)
            return;

        if (threadIdx.x != 0)
            return;

        const int n_metrics = 8;

        float mean = fm_curve_mean(curves, curve, n_points);
        float var = fm_curve_var(curves, curve, n_points, mean);

        if (n_points < 3 || var < 1.0e-12f)
        {
            for (int m = 0; m < n_metrics; ++m)
                metrics[curve * n_metrics + m] = 0.0f;

            return;
        }

        int n_freq = n_points / 2 + 1;
        float total_power = 0.0f;
        float peak_power = -1.0f;
        int peak_idx = 0;
        float peak_re = 0.0f;
        float peak_im = 0.0f;

        for (int k = 1; k < n_freq; ++k)
        {
            float re = 0.0f;
            float im = 0.0f;

            for (int i = 0; i < n_points; ++i)
            {
                float y = curves[curve * n_points + i] - mean;
                float ang = -2.0f * (float)M_PI * (float)k * (float)i / (float)n_points;

                re += y * cosf(ang);
                im += y * sinf(ang);
            }

            float p = re * re + im * im;
            total_power += p;

            if (p > peak_power)
            {
                peak_power = p;
                peak_idx = k;
                peak_re = re;
                peak_im = im;
            }
        }

        total_power += 1.0e-12f;
        float peak_ratio = peak_power / total_power;

        float spectral_entropy = 0.0f;
        int bins = max(1, n_freq - 1);

        for (int k = 1; k < n_freq; ++k)
        {
            float re = 0.0f;
            float im = 0.0f;

            for (int i = 0; i < n_points; ++i)
            {
                float y = curves[curve * n_points + i] - mean;
                float ang = -2.0f * (float)M_PI * (float)k * (float)i / (float)n_points;

                re += y * cosf(ang);
                im += y * sinf(ang);
            }

            float p = (re * re + im * im) / total_power;

            if (p > 1.0e-12f)
                spectral_entropy += -p * logf(p);
        }

        if (bins > 1)
            spectral_entropy /= logf((float)bins);

        float low = 0.0f;
        float high = 0.0f;
        int half = max(1, bins / 2);

        for (int k = 1; k < n_freq; ++k)
        {
            float re = 0.0f;
            float im = 0.0f;

            for (int i = 0; i < n_points; ++i)
            {
                float y = curves[curve * n_points + i] - mean;
                float ang = -2.0f * (float)M_PI * (float)k * (float)i / (float)n_points;

                re += y * cosf(ang);
                im += y * sinf(ang);
            }

            float p = re * re + im * im;

            if ((k - 1) < half)
                low += p;
            else
                high += p;
        }

        float low_high_ratio = low / (high + 1.0e-12f);
        float dominant_phase = atan2f(peak_im, peak_re);

        float best_r2 = -1.0e20f;
        float best_freq = 0.0f;
        float best_amp = 0.0f;
        float best_phase = 0.0f;

        float ss_tot = 0.0f;

        for (int i = 0; i < n_points; ++i)
        {
            float d = curves[curve * n_points + i] - mean;
            ss_tot += d * d;
        }

        ss_tot += 1.0e-12f;

        float xmin = xs[0];
        float xmax = xs[0];

        for (int i = 1; i < n_points; ++i)
        {
            xmin = fminf(xmin, xs[i]);
            xmax = fmaxf(xmax, xs[i]);
        }

        for (int step = 0; step < 26; ++step)
        {
            float freq = 0.5f + (3.0f - 0.5f) * ((float)step / 25.0f);
            float w = 2.0f * (float)M_PI * freq;

            float s_s = 0.0f, s_c = 0.0f, s_1 = (float)n_points;
            float ss = 0.0f, cc = 0.0f, sc = 0.0f;
            float sy = 0.0f, cy = 0.0f, yy = 0.0f;

            for (int i = 0; i < n_points; ++i)
            {
                float x = xs[i];

                if (xmax > xmin)
                    x = (x - xmin) / (xmax - xmin);
                else
                    x = (float)i / (float)max(1, n_points - 1);

                float sv = sinf(w * x);
                float cv = cosf(w * x);
                float yv = curves[curve * n_points + i];

                s_s += sv;
                s_c += cv;
                ss += sv * sv;
                cc += cv * cv;
                sc += sv * cv;
                sy += sv * yv;
                cy += cv * yv;
                yy += yv;
            }

            float A[3][4];

            A[0][0] = ss;
            A[0][1] = sc;
            A[0][2] = s_s;
            A[0][3] = sy;

            A[1][0] = sc;
            A[1][1] = cc;
            A[1][2] = s_c;
            A[1][3] = cy;

            A[2][0] = s_s;
            A[2][1] = s_c;
            A[2][2] = s_1;
            A[2][3] = yy;

            bool ok = true;

            for (int col = 0; col < 3; ++col)
            {
                int piv = col;
                float best = fabsf(A[col][col]);

                for (int r = col + 1; r < 3; ++r)
                {
                    float v = fabsf(A[r][col]);

                    if (v > best)
                    {
                        best = v;
                        piv = r;
                    }
                }

                if (best < 1.0e-12f)
                {
                    ok = false;
                    break;
                }

                if (piv != col)
                {
                    for (int c = col; c < 4; ++c)
                    {
                        float tmp = A[col][c];
                        A[col][c] = A[piv][c];
                        A[piv][c] = tmp;
                    }
                }

                float div = A[col][col];

                for (int c = col; c < 4; ++c)
                    A[col][c] /= div;

                for (int r = 0; r < 3; ++r)
                {
                    if (r == col)
                        continue;

                    float f = A[r][col];

                    for (int c = col; c < 4; ++c)
                        A[r][c] -= f * A[col][c];
                }
            }

            if (!ok)
                continue;

            float a = A[0][3];
            float b = A[1][3];
            float c0 = A[2][3];

            float ss_res = 0.0f;

            for (int i = 0; i < n_points; ++i)
            {
                float x = xs[i];

                if (xmax > xmin)
                    x = (x - xmin) / (xmax - xmin);
                else
                    x = (float)i / (float)max(1, n_points - 1);

                float pred = a * sinf(w * x) + b * cosf(w * x) + c0;
                float err = curves[curve * n_points + i] - pred;

                ss_res += err * err;
            }

            float r2 = 1.0f - ss_res / ss_tot;
            float amp = sqrtf(a * a + b * b);
            float phase = atan2f(b, a);

            if (!isfinite(amp) || amp > 10.0f || !isfinite(r2))
                continue;

            if (r2 > best_r2)
            {
                best_r2 = r2;
                best_freq = freq;
                best_amp = amp;
                best_phase = phase;
            }
        }

        if (!isfinite(best_r2) || best_r2 < 0.0f)
        {
            best_r2 = 0.0f;
            best_freq = 0.0f;
            best_amp = 0.0f;
            best_phase = 0.0f;
        }

        float score =
            0.40f * peak_ratio + 0.25f * best_r2 + 0.20f * (1.0f - spectral_entropy) + 0.15f * fminf(1.0f, fabsf(low_high_ratio) / 10.0f);

        metrics[curve * n_metrics + 0] = score;
        metrics[curve * n_metrics + 1] = peak_ratio;
        metrics[curve * n_metrics + 2] = spectral_entropy;
        metrics[curve * n_metrics + 3] = best_r2;
        metrics[curve * n_metrics + 4] = best_freq;
        metrics[curve * n_metrics + 5] = best_amp;
        metrics[curve * n_metrics + 6] = best_phase;
        metrics[curve * n_metrics + 7] = low_high_ratio;
    }

} // extern "C"