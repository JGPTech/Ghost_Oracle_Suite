// =================================================================================================
// T_S GEO KERNEL — RAW ROUTES + RAW DAMAGE SIGNATURE SUPPORT
// =================================================================================================
// Single shared CUDA file for T_S geo and qproj raw-damage work.
//
// Compatibility:
//   - Keeps the original public entry point:
//
//         ts_raw_geo_monotonic_kernel
//
//     so Probe 04 / existing callers do not break.
//
// Additions:
//   - route vector output kernel
//   - raw route damage kernel
//   - packed real-vs-ablated route damage kernel
//   - index-wise aggregate damage kernel
//
// Methodology:
//   - RAW ONLY.
//   - NO normalization.
//   - NO cosine.
//   - NO classifier logic.
//   - NO base/projection assumptions inside this file.
//   - This is arithmetic support for the QPU/GPU probe code.
//
// Geometry:
//   node = (delay_cell, round_cell, edge_cell)
//
// Monotonic forward route:
//   source = (0,0,0)
//   target = (A-1,R-1,E-1)
//
// DP recurrence:
//   dp[a,r,e] = min(
//       dp[a-1,r,e] + cost_tau(a,r,e),
//       dp[a,r-1,e] + cost_round(a,r,e),
//       dp[a,r,e-1] + cost_edge(a,r,e)
//   )
//
// Movement costs:
//   cost_tau   = Ttau_tau + c*(Ttau_r + Ttau_x)
//   cost_round = Trr      + c*(Ttau_r + Trx)
//   cost_edge  = Txx      + c*(Ttau_x + Trx)
//
// Raw damage:
//   damage =
//       abs(full_ablated  - full_real)
//     + abs(delay_ablated - delay_real)
//     + abs(edge_ablated  - edge_real)
//     + max(0, avoidance_real - avoidance_ablated)
//
// Route vector layout:
//   route[0] = full_cost
//   route[1] = delay_cost
//   route[2] = edge_cost
//   route[3] = delay_to_full
//   route[4] = edge_to_full
//   route[5] = trace_mean
//   route[6] = stress_avoidance_proxy
//   route[7] = path_trace_mean_proxy
//
// Damage vector layout:
//   damage[0] = total_damage
//   damage[1] = full_delta
//   damage[2] = delay_delta
//   damage[3] = edge_delta
//   damage[4] = avoidance_loss
//   damage[5] = abs_full_delta
//   damage[6] = abs_delay_delta
//   damage[7] = abs_edge_delta
//
// Notes:
//   - This is a specialized structured-grid route engine, not a generic graph library.
//   - Generic SSSP/Dijkstra baselines are intentionally more general.
//   - This file is designed to remain the single T_S geo kernel file as projector support grows.
// =================================================================================================

#ifndef TS_GEO_KERNEL_CU
#define TS_GEO_KERNEL_CU

#define TS_ROUTE_STRIDE 8
#define TS_DAMAGE_STRIDE 8

extern "C" __device__ __forceinline__
float ts_absf(float x) {
    return x < 0.0f ? -x : x;
}

extern "C" __device__ __forceinline__
float ts_safe_div(float num, float den) {
    const float eps = 1.0e-12f;
    float d = ts_absf(den) + eps;
    return num / d;
}

extern "C" __device__ __forceinline__
float ts_clip_cost(float c, float min_cost, float max_cost) {
    return fminf(fmaxf(c, min_cost), max_cost);
}

extern "C" __device__ __forceinline__
int ts_idx3(int a, int r, int e, int R, int E) {
    return (a * R + r) * E + e;
}

extern "C" __device__ __forceinline__
float ts_geo_cost_tau(
    const float* __restrict__ tau_tau,
    const float* __restrict__ tau_r,
    const float* __restrict__ tau_x,
    int idx,
    float coupling_weight,
    float min_cost,
    float max_cost
) {
    float c = tau_tau[idx] + coupling_weight * (tau_r[idx] + tau_x[idx]);
    return ts_clip_cost(c, min_cost, max_cost);
}

extern "C" __device__ __forceinline__
float ts_geo_cost_round(
    const float* __restrict__ rr,
    const float* __restrict__ tau_r,
    const float* __restrict__ r_x,
    int idx,
    float coupling_weight,
    float min_cost,
    float max_cost
) {
    float c = rr[idx] + coupling_weight * (tau_r[idx] + r_x[idx]);
    return ts_clip_cost(c, min_cost, max_cost);
}

extern "C" __device__ __forceinline__
float ts_geo_cost_edge(
    const float* __restrict__ xx,
    const float* __restrict__ tau_x,
    const float* __restrict__ r_x,
    int idx,
    float coupling_weight,
    float min_cost,
    float max_cost
) {
    float c = xx[idx] + coupling_weight * (tau_x[idx] + r_x[idx]);
    return ts_clip_cost(c, min_cost, max_cost);
}

extern "C" __device__ __forceinline__
float ts_trace_at(
    const float* __restrict__ tau_tau,
    const float* __restrict__ rr,
    const float* __restrict__ xx,
    int idx
) {
    return tau_tau[idx] + rr[idx] + xx[idx];
}

// -------------------------------------------------------------------------------------------------
// Device route computation.
// One thread owns one instance.
// dp points at that instance's DP scratch of length N.
// Component pointers already point at the start of that instance.
// -------------------------------------------------------------------------------------------------

extern "C" __device__ __forceinline__
void ts_compute_raw_routes_device(
    const float* __restrict__ tau_tau,
    const float* __restrict__ rr,
    const float* __restrict__ xx,
    const float* __restrict__ tau_r,
    const float* __restrict__ tau_x,
    const float* __restrict__ r_x,
    float* __restrict__ dp,
    int A,
    int R,
    int E,
    float coupling_weight,
    float min_cost,
    float max_cost,
    float* __restrict__ route_out
) {
    const float INF = 3.402823466e+38F;
    int N = A * R * E;

    // Full monotonic DP.
    for (int a = 0; a < A; ++a) {
        for (int r = 0; r < R; ++r) {
            for (int e = 0; e < E; ++e) {
                int local = ts_idx3(a, r, e, R, E);

                if (a == 0 && r == 0 && e == 0) {
                    dp[local] = 0.0f;
                    continue;
                }

                float best = INF;

                if (a > 0) {
                    int prev = ts_idx3(a - 1, r, e, R, E);
                    float c = ts_geo_cost_tau(tau_tau, tau_r, tau_x, local, coupling_weight, min_cost, max_cost);
                    best = fminf(best, dp[prev] + c);
                }

                if (r > 0) {
                    int prev = ts_idx3(a, r - 1, e, R, E);
                    float c = ts_geo_cost_round(rr, tau_r, r_x, local, coupling_weight, min_cost, max_cost);
                    best = fminf(best, dp[prev] + c);
                }

                if (e > 0) {
                    int prev = ts_idx3(a, r, e - 1, R, E);
                    float c = ts_geo_cost_edge(xx, tau_x, r_x, local, coupling_weight, min_cost, max_cost);
                    best = fminf(best, dp[prev] + c);
                }

                dp[local] = best;
            }
        }
    }

    int target = ts_idx3(A - 1, R - 1, E - 1, R, E);
    float full_cost = dp[target];

    // Delay-only route through center round/edge.
    int cr = R / 2;
    int ce = E / 2;
    float delay_cost = 0.0f;
    for (int a = 1; a < A; ++a) {
        int idx = ts_idx3(a, cr, ce, R, E);
        delay_cost += ts_geo_cost_tau(tau_tau, tau_r, tau_x, idx, coupling_weight, min_cost, max_cost);
    }

    // Edge-only route through center delay/round.
    int ca = A / 2;
    float edge_cost = 0.0f;
    for (int e = 1; e < E; ++e) {
        int idx = ts_idx3(ca, cr, e, R, E);
        edge_cost += ts_geo_cost_edge(xx, tau_x, r_x, idx, coupling_weight, min_cost, max_cost);
    }

    // Raw trace mean.
    float trace_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        trace_sum += ts_trace_at(tau_tau, rr, xx, i);
    }
    float trace_mean = trace_sum / (float)N;

    // Lightweight raw path trace proxy: diagonal-ish route through the stress grid.
    int steps = A;
    if (R > steps) steps = R;
    if (E > steps) steps = E;

    float path_trace_sum = 0.0f;
    for (int k = 0; k < steps; ++k) {
        int a = 0;
        int r = 0;
        int e = 0;

        if (steps > 1) {
            a = (int)floorf(((float)k * (float)(A - 1) / (float)(steps - 1)) + 0.5f);
            r = (int)floorf(((float)k * (float)(R - 1) / (float)(steps - 1)) + 0.5f);
            e = (int)floorf(((float)k * (float)(E - 1) / (float)(steps - 1)) + 0.5f);
        }

        if (a < 0) a = 0;
        if (a >= A) a = A - 1;
        if (r < 0) r = 0;
        if (r >= R) r = R - 1;
        if (e < 0) e = 0;
        if (e >= E) e = E - 1;

        int idx = ts_idx3(a, r, e, R, E);
        path_trace_sum += ts_trace_at(tau_tau, rr, xx, idx);
    }

    float path_trace_mean = path_trace_sum / (float)steps;
    float stress_avoidance = trace_mean - path_trace_mean;

    route_out[0] = full_cost;
    route_out[1] = delay_cost;
    route_out[2] = edge_cost;
    route_out[3] = ts_safe_div(delay_cost, full_cost);
    route_out[4] = ts_safe_div(edge_cost, full_cost);
    route_out[5] = trace_mean;
    route_out[6] = stress_avoidance;
    route_out[7] = path_trace_mean;
}

extern "C" __device__ __forceinline__
void ts_compute_raw_damage_device(
    const float* __restrict__ real_route,
    const float* __restrict__ ablated_route,
    float* __restrict__ damage_out
) {
    float full_delta = ablated_route[0] - real_route[0];
    float delay_delta = ablated_route[1] - real_route[1];
    float edge_delta = ablated_route[2] - real_route[2];

    float avoidance_loss = real_route[6] - ablated_route[6];
    if (avoidance_loss < 0.0f) avoidance_loss = 0.0f;

    float abs_full = ts_absf(full_delta);
    float abs_delay = ts_absf(delay_delta);
    float abs_edge = ts_absf(edge_delta);

    float total = abs_full + abs_delay + abs_edge + avoidance_loss;

    damage_out[0] = total;
    damage_out[1] = full_delta;
    damage_out[2] = delay_delta;
    damage_out[3] = edge_delta;
    damage_out[4] = avoidance_loss;
    damage_out[5] = abs_full;
    damage_out[6] = abs_delay;
    damage_out[7] = abs_edge;
}

// -------------------------------------------------------------------------------------------------
// ORIGINAL COMPATIBILITY KERNEL.
// Keep this name and signature stable.
// -------------------------------------------------------------------------------------------------

extern "C" __global__
void ts_raw_geo_monotonic_kernel(
    const float* __restrict__ tau_tau,   // [B,N]
    const float* __restrict__ rr,        // [B,N]
    const float* __restrict__ xx,        // [B,N]
    const float* __restrict__ tau_r,     // [B,N]
    const float* __restrict__ tau_x,     // [B,N]
    const float* __restrict__ r_x,       // [B,N]
    float* __restrict__ dp,              // [B,N] scratch/output DP table
    float* __restrict__ out_full,        // [B]
    float* __restrict__ out_delay,       // [B]
    float* __restrict__ out_edge,        // [B]
    int B,
    int A,
    int R,
    int E,
    float coupling_weight,
    float min_cost,
    float max_cost
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int N = A * R * E;
    int base = b * N;

    float route[TS_ROUTE_STRIDE];

    ts_compute_raw_routes_device(
        tau_tau + base,
        rr + base,
        xx + base,
        tau_r + base,
        tau_x + base,
        r_x + base,
        dp + base,
        A,
        R,
        E,
        coupling_weight,
        min_cost,
        max_cost,
        route
    );

    out_full[b] = route[0];
    out_delay[b] = route[1];
    out_edge[b] = route[2];
}

// -------------------------------------------------------------------------------------------------
// ENHANCED ROUTE VECTOR KERNEL.
// Outputs all 8 raw route components per instance.
// -------------------------------------------------------------------------------------------------

extern "C" __global__
void ts_raw_geo_route_vector_kernel(
    const float* __restrict__ tau_tau,   // [B,N]
    const float* __restrict__ rr,        // [B,N]
    const float* __restrict__ xx,        // [B,N]
    const float* __restrict__ tau_r,     // [B,N]
    const float* __restrict__ tau_x,     // [B,N]
    const float* __restrict__ r_x,       // [B,N]
    float* __restrict__ dp,              // [B,N]
    float* __restrict__ out_routes,      // [B,8]
    int B,
    int A,
    int R,
    int E,
    float coupling_weight,
    float min_cost,
    float max_cost
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int N = A * R * E;
    int base = b * N;
    int route_base = b * TS_ROUTE_STRIDE;

    ts_compute_raw_routes_device(
        tau_tau + base,
        rr + base,
        xx + base,
        tau_r + base,
        tau_x + base,
        r_x + base,
        dp + base,
        A,
        R,
        E,
        coupling_weight,
        min_cost,
        max_cost,
        out_routes + route_base
    );
}

// -------------------------------------------------------------------------------------------------
// RAW DAMAGE KERNEL.
// Compares precomputed real and ablated route vectors.
// -------------------------------------------------------------------------------------------------

extern "C" __global__
void ts_raw_geo_damage_kernel(
    const float* __restrict__ real_routes,     // [B,8]
    const float* __restrict__ ablated_routes,  // [B,8]
    float* __restrict__ out_damage,            // [B,8]
    int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int rb = b * TS_ROUTE_STRIDE;
    int db = b * TS_DAMAGE_STRIDE;

    ts_compute_raw_damage_device(
        real_routes + rb,
        ablated_routes + rb,
        out_damage + db
    );
}

// -------------------------------------------------------------------------------------------------
// PACKED REAL-VS-ABLATED RAW DAMAGE KERNEL.
// Useful for Probe 06 optimized pass.
//
// Input layout:
//   real components    [B,N]
//   ablated components [K,B,N] flattened as [K*B,N]
//
// Output:
//   real_routes        [B,8]
//   ablated_routes     [K,B,8]
//   damage             [K,B,8]
//
// One thread handles one (k,b) ablated instance.
// k = ablation/control index, b = block/sample index.
// -------------------------------------------------------------------------------------------------

extern "C" __global__
void ts_raw_geo_packed_damage_kernel(
    const float* __restrict__ real_tau_tau,      // [B,N]
    const float* __restrict__ real_rr,
    const float* __restrict__ real_xx,
    const float* __restrict__ real_tau_r,
    const float* __restrict__ real_tau_x,
    const float* __restrict__ real_r_x,

    const float* __restrict__ ab_tau_tau,        // [K,B,N]
    const float* __restrict__ ab_rr,
    const float* __restrict__ ab_xx,
    const float* __restrict__ ab_tau_r,
    const float* __restrict__ ab_tau_x,
    const float* __restrict__ ab_r_x,

    float* __restrict__ dp_real,                 // [B,N]
    float* __restrict__ dp_ab,                   // [K,B,N]

    float* __restrict__ real_routes,             // [B,8]
    float* __restrict__ ablated_routes,          // [K,B,8]
    float* __restrict__ damage,                  // [K,B,8]

    int B,
    int K,
    int A,
    int R,
    int E,
    float coupling_weight,
    float min_cost,
    float max_cost
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * K;
    if (t >= total) return;

    int k = t / B;
    int b = t - k * B;

    int N = A * R * E;
    int real_base = b * N;
    int ab_base = (k * B + b) * N;

    int real_route_base = b * TS_ROUTE_STRIDE;
    int ab_route_base = (k * B + b) * TS_ROUTE_STRIDE;
    int damage_base = (k * B + b) * TS_DAMAGE_STRIDE;

    // Computing real routes redundantly per K is acceptable for small K and keeps
    // launch orchestration simple. For large K, call ts_raw_geo_route_vector_kernel
    // first and use ts_raw_geo_damage_kernel or a dedicated no-real-recompute variant.
    ts_compute_raw_routes_device(
        real_tau_tau + real_base,
        real_rr + real_base,
        real_xx + real_base,
        real_tau_r + real_base,
        real_tau_x + real_base,
        real_r_x + real_base,
        dp_real + real_base,
        A,
        R,
        E,
        coupling_weight,
        min_cost,
        max_cost,
        real_routes + real_route_base
    );

    ts_compute_raw_routes_device(
        ab_tau_tau + ab_base,
        ab_rr + ab_base,
        ab_xx + ab_base,
        ab_tau_r + ab_base,
        ab_tau_x + ab_base,
        ab_r_x + ab_base,
        dp_ab + ab_base,
        A,
        R,
        E,
        coupling_weight,
        min_cost,
        max_cost,
        ablated_routes + ab_route_base
    );

    ts_compute_raw_damage_device(
        real_routes + real_route_base,
        ablated_routes + ab_route_base,
        damage + damage_base
    );
}

// -------------------------------------------------------------------------------------------------
// DAMAGE AGGREGATE KERNEL.
// Reduces damage [K,B,8] into mean damage by K over B.
//
// Output:
//   out_mean_damage[K,8]
//
// This is a simple one-block-per-K reduction. It is designed for small/medium B.
// For very large B, use a two-pass reduction.
// -------------------------------------------------------------------------------------------------

extern "C" __global__
void ts_raw_geo_damage_mean_by_k_kernel(
    const float* __restrict__ damage,       // [K,B,8]
    float* __restrict__ out_mean_damage,    // [K,8]
    int B,
    int K
) {
    int k = blockIdx.x;
    int component = threadIdx.x;

    if (k >= K || component >= TS_DAMAGE_STRIDE) return;

    float sum = 0.0f;
    for (int b = 0; b < B; ++b) {
        int idx = (k * B + b) * TS_DAMAGE_STRIDE + component;
        sum += damage[idx];
    }

    out_mean_damage[k * TS_DAMAGE_STRIDE + component] = sum / (float)B;
}

#endif // TS_GEO_KERNEL_CU
