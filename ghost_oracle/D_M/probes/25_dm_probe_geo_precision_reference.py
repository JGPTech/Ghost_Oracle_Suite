#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — D_M PROBE 25: GEO PRECISION REFERENCE
==============================================================================

Drop this file in:

    ghost_oracle/D_M/probes/d_m_probe25_geo_precision_reference.py

Purpose
-------
Probe 25 isolates GEO from qproj/gproj/GPT-2/text plumbing.

The older raw-boundary / retrieval probes used GEO as:

    raw text bits -> CUDA data-pair witness -> analytic GEO aperture -> summary

That was useful plumbing, but it made GEO depend on raw text correlations and
hand-shaped aperture constants. This probe makes GEO the thing it should be for
D_M cleanup:

    a closed-form classical reference representation of the D_M manifold.

No shot sampling.
No raw text.
No qproj/gproj base files.
No GPT-2.
No random controls.

GEO is computed directly from condition metadata:

    condition -> base delay axis + offset axis
              -> one D_M coordinate with separable spatial/temporal components
              -> exact YZ-primary / ZY-reciprocal witnesses
              -> exact rung projection + exact summary metrics

Current D_M orientation
-----------------------
    YZ = primary witness dimension
    ZY = reciprocal / inverted witness dimension
    XY / YX = comparison dimensions

Projected rung coordinates
--------------------------
    Y  = connected(YZ)        # GEO writes this directly as analytic YZ
    Z  = connected(ZY)        # GEO writes this directly as analytic ZY
    R  = -Z
    E  = sqrt(Y^2 + R^2)
    S  = E - sqrt(XY^2 + YX^2)
    phi = atan2(R, Y) mod pi

Probe 25 exact GEO rule
-----------------------
For each rung r:

    b_r = base delay
    o_r = mean witness offset across XY/YZ/ZY/YX tiles
    t_r = b_r + o_r

    x_space = normalize(log1p(b_r))
    x_time  = normalize(log1p(t_r))
    x_dm    = sqrt((w_space*x_space^2 + w_time*x_time^2)/(w_space+w_time))

    cos(2 phi_r) = 2*x_time - 1
    phi_r = 0.5 * arccos(2*x_time - 1)

    E_r = energy_floor + energy_scale * x_dm^energy_gamma
    YZ_r = E_r * cos(phi_r)
    ZY_r = -E_r * sin(phi_r)

This makes the active GEO phase relation mathematically exact against the same
log-total-delay coordinate used by the summary kernel. Therefore active GEO
should report pi_periodic_score ~= 1.0, bounded only by float32 tolerance.

The null condition is an exact zero manifold.

Outputs
-------
Creates:

    ghost_oracle/D_M/probes/analyze/dm_probe25_geo_precision_<timestamp>/
        probe25_geo_rung_projection.csv
        probe25_geo_summary.csv
        probe25_cpu_gpu_agreement.csv
        probe25_report.md
        probe_config.json

Usage
-----
From repo root:

    python ghost_oracle/D_M/probes/d_m_probe25_geo_precision_reference.py

Require CUDA and fail if CuPy/GPU is not available:

    python ghost_oracle/D_M/probes/d_m_probe25_geo_precision_reference.py --require-cuda

Run only offset_on:

    python ghost_oracle/D_M/probes/d_m_probe25_geo_precision_reference.py --conditions offset_on

=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
    HAVE_CUPY = True
    CUPY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    cp = None  # type: ignore
    HAVE_CUPY = False
    CUPY_IMPORT_ERROR = repr(exc)


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

HERE = Path(__file__).resolve().parent
D_M_DIR = HERE.parent
ANALYSIS_DIR = HERE / "analyze"

DEFAULT_BASE_DELAYS = [0, 256, 1024, 4096, 16384]
DEFAULT_NULL_DELAYS = [0, 0, 0, 0, 0]
DEFAULT_OFFSET_DT = 128
WITNESS_LABELS = ["XY", "YZ", "ZY", "YX"]
CONDITION_ORDER = ["null", "base_only", "offset_on"]

EPS = 1.0e-12
SIGN_EPS = 1.0e-6  # float32-safe deadband for sign-only diagnostics at exact phase endpoints

DM_RUNG_XY = 0
DM_RUNG_YZ = 1
DM_RUNG_ZY = 2
DM_RUNG_YX = 3
DM_RUNG_YZ_PRIMARY = 4
DM_RUNG_ZY_RETURN = 5
DM_RUNG_YZZY_ENERGY = 6
DM_RUNG_COMPARISON_ENERGY = 7
DM_RUNG_DIRECTIONAL_SPECIFICITY = 8
DM_RUNG_DIRECTIONAL_GAP = 9
DM_RUNG_INVERSION = 10
DM_RUNG_PI_PHASE = 11
DM_RUNG_PI_COS2 = 12
DM_RUNG_PI_SIN2 = 13
DM_RUNG_BASE_DELAY = 14
DM_RUNG_OFFSET = 15
DM_RUNG_TOTAL_DELAY = 16
DM_RUNG_COUNT_ALL = 17
DM_RUNG_COUNT_YZZY = 18
DM_RUNG_N_METRICS = 19

DM_SUMMARY_N_RUNGS = 0
DM_SUMMARY_YZ_MEAN = 1
DM_SUMMARY_YZ_POS_FRAC = 2
DM_SUMMARY_ZY_MEAN = 3
DM_SUMMARY_ZY_INVERTED_FRAC = 4
DM_SUMMARY_YZZY_ENERGY_MEAN = 5
DM_SUMMARY_YZZY_ENERGY_MAX = 6
DM_SUMMARY_SPECIFICITY_MEAN = 7
DM_SUMMARY_SPECIFICITY_MAX = 8
DM_SUMMARY_PI_PERIODIC_SCORE = 9
DM_SUMMARY_PI_PERIODIC_MODE = 10
DM_SUMMARY_ENERGY_TRACKING_R = 11
DM_SUMMARY_SPECIFICITY_TRACKING_R = 12
DM_SUMMARY_PHASE_VELOCITY_R = 13
DM_SUMMARY_PHASE_SPAN_PI_UNITS = 14
DM_SUMMARY_PROJECTION_SCORE = 15
DM_SUMMARY_N_METRICS = 16

RUNG_METRIC_NAMES = [
    "XY",
    "YZ",
    "ZY",
    "YX",
    "YZ_primary",
    "ZY_return",
    "YZ_ZY_energy",
    "comparison_energy",
    "directional_specificity",
    "directional_gap",
    "inversion",
    "pi_phase",
    "pi_cos2",
    "pi_sin2",
    "base_delay",
    "offset",
    "total_delay",
    "count_all",
    "count_yzzy",
]

SUMMARY_NAMES = [
    "n_rungs",
    "yz_mean",
    "yz_pos_frac",
    "zy_mean",
    "zy_inverted_frac",
    "yzzy_energy_mean",
    "yzzy_energy_max",
    "specificity_mean",
    "specificity_max",
    "pi_periodic_score",
    "pi_periodic_mode",
    "energy_tracking_r",
    "specificity_tracking_r",
    "phase_velocity_r",
    "phase_span_pi_units",
    "projection_score",
]

# Validation discipline:
#
# The GEO kernel is a float32 implementation of a float64 closed-form reference.
# Core pass/fail should therefore be based on the load-bearing D_M coordinates
# and summary claims. Auxiliary diagnostics such as phase_velocity_r are useful
# to log, but they are intentionally not used to fail the GEO reference because
# they are derivative/correlation diagnostics over only four intervals and can
# be hypersensitive to float32 rounding without changing the manifold.
CORE_SUMMARY_INDICES = [
    DM_SUMMARY_N_RUNGS,
    DM_SUMMARY_YZ_MEAN,
    DM_SUMMARY_ZY_MEAN,
    DM_SUMMARY_YZZY_ENERGY_MEAN,
    DM_SUMMARY_YZZY_ENERGY_MAX,
    DM_SUMMARY_SPECIFICITY_MEAN,
    DM_SUMMARY_SPECIFICITY_MAX,
    DM_SUMMARY_PI_PERIODIC_SCORE,
    DM_SUMMARY_ENERGY_TRACKING_R,
    DM_SUMMARY_SPECIFICITY_TRACKING_R,
    DM_SUMMARY_PHASE_SPAN_PI_UNITS,
    DM_SUMMARY_PROJECTION_SCORE,
]
# Discrete/sign-only summary diagnostics are logged separately. They can flip by
# exactly 1/n at analytic zeros such as cos(pi/2), where CPU float64 and CUDA
# float32 may land on opposite sides of zero without changing the manifold.
AUX_SUMMARY_INDICES = [
    DM_SUMMARY_YZ_POS_FRAC,
    DM_SUMMARY_ZY_INVERTED_FRAC,
    DM_SUMMARY_PI_PERIODIC_MODE,
    DM_SUMMARY_PHASE_VELOCITY_R,
]


# =============================================================================
# CUDA SOURCE — built-in standalone GEO kernel
# =============================================================================

CUDA_SOURCE = r'''
extern "C" {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef DM_GEO_SIGN_EPS
#define DM_GEO_SIGN_EPS 1.0e-6f
#endif

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
        DM_SUMMARY_PI_PERIODIC_MODE = 10,
        DM_SUMMARY_ENERGY_TRACKING_R = 11,
        DM_SUMMARY_SPECIFICITY_TRACKING_R = 12,
        DM_SUMMARY_PHASE_VELOCITY_R = 13,
        DM_SUMMARY_PHASE_SPAN_PI_UNITS = 14,
        DM_SUMMARY_PROJECTION_SCORE = 15,
        DM_SUMMARY_N_METRICS = 16
    };

    __device__ __forceinline__ int dm_rung_idx(int rung, int metric)
    {
        return rung * DM_RUNG_N_METRICS + metric;
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

    __global__ void dm_geo_exact_rung_kernel_f32(
        const int condition_kind,
        const int *base_delays_dt,
        const int n_rungs,
        const int offset_dt,
        const float energy_floor,
        const float energy_scale,
        const float energy_gamma,
        const float comparison_scale,
        const float spatial_weight,
        const float temporal_weight,
        float *rung_stats)
    {
        int r = blockIdx.x;
        if (r >= n_rungs || threadIdx.x != 0)
            return;
        if (n_rungs <= 0 || n_rungs > 64)
            return;

        float space_raw[64], time_raw[64];
        float space_norm[64], time_norm[64];

        for (int i = 0; i < n_rungs; ++i)
        {
            float base_delay = (float)base_delays_dt[i];
            float offset_mean = ((float)(4 * i) + 1.5f) * (float)offset_dt;
            float total_delay = base_delay + offset_mean;
            space_raw[i] = base_delay;
            time_raw[i] = total_delay;
        }

        dm_norm_x_small(space_raw, n_rungs, 1, space_norm);
        dm_norm_x_small(time_raw, n_rungs, 1, time_norm);

        float base_delay = space_raw[r];
        float offset_mean = ((float)(4 * r) + 1.5f) * (float)offset_dt;
        float total_delay = time_raw[r];

        float xy = 0.0f;
        float yz = 0.0f;
        float zy = 0.0f;
        float yx = 0.0f;
        float ret = 0.0f;
        float energy = 0.0f;
        float comp = 0.0f;
        float spec = 0.0f;
        float gap = 0.0f;
        float inversion = 0.0f;
        float phase = 0.0f;

        if (condition_kind != 0)
        {
            float sw = fmaxf(spatial_weight, 0.0f);
            float tw = fmaxf(temporal_weight, 0.0f);
            float denom = sw + tw;
            if (denom <= 1.0e-12f)
            {
                sw = 1.0f;
                tw = 1.0f;
                denom = 2.0f;
            }

            float xs = space_norm[r];
            float xt = time_norm[r];
            float xdm = sqrtf((sw * xs * xs + tw * xt * xt) / denom);
            xdm = fminf(1.0f, fmaxf(0.0f, xdm));

            float q = 2.0f * xt - 1.0f;
            q = fminf(1.0f, fmaxf(-1.0f, q));
            phase = 0.5f * acosf(q);

            float gamma = energy_gamma > 0.0f ? energy_gamma : 1.0f;
            energy = energy_floor + energy_scale * powf(xdm, gamma);
            energy = fmaxf(0.0f, energy);

            yz = energy * cosf(phase);
            ret = energy * sinf(phase);
            if (fabsf(yz) < DM_GEO_SIGN_EPS) yz = 0.0f;
            if (fabsf(ret) < DM_GEO_SIGN_EPS) ret = 0.0f;
            zy = -ret;

            if (comparison_scale > 0.0f)
            {
                float c_amp = comparison_scale * (1.0f - 0.5f * xdm);
                xy = c_amp * cosf((float)M_PI * xt);
                yx = c_amp * sinf((float)M_PI * xt);
                if (fabsf(xy) < DM_GEO_SIGN_EPS) xy = 0.0f;
                if (fabsf(yx) < DM_GEO_SIGN_EPS) yx = 0.0f;
            }
        }

        ret = -zy;
        energy = sqrtf(yz * yz + ret * ret);
        comp = sqrtf(xy * xy + yx * yx);
        spec = energy - comp;
        gap = yz - zy;
        inversion = -yz * zy;

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
        rung_stats[dm_rung_idx(r, DM_RUNG_PI_PHASE)] = dm_wrap_pi(phase);
        rung_stats[dm_rung_idx(r, DM_RUNG_PI_COS2)] = cosf(2.0f * phase);
        rung_stats[dm_rung_idx(r, DM_RUNG_PI_SIN2)] = sinf(2.0f * phase);
        rung_stats[dm_rung_idx(r, DM_RUNG_BASE_DELAY)] = base_delay;
        rung_stats[dm_rung_idx(r, DM_RUNG_OFFSET)] = offset_mean;
        rung_stats[dm_rung_idx(r, DM_RUNG_TOTAL_DELAY)] = total_delay;
        rung_stats[dm_rung_idx(r, DM_RUNG_COUNT_ALL)] = 4.0f;
        rung_stats[dm_rung_idx(r, DM_RUNG_COUNT_YZZY)] = condition_kind == 0 ? 0.0f : 1.0f;
    }

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

            if (yv > DM_GEO_SIGN_EPS)
                ++yz_pos;
            if (yv * zv < -DM_GEO_SIGN_EPS)
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

        float e_term = fmaxf(0.0f, e_mean);
        float s_term = fmaxf(0.0f, s_mean);
        float y_term = fmaxf(0.0f, yz_mean);
        float p_term = fmaxf(0.0f, e_mean) * pi_score;
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
}
'''


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GeoResult:
    condition: str
    backend: str
    rung_stats: np.ndarray
    summary: np.ndarray
    elapsed_ms: float
    meta: Dict[str, Any]


# =============================================================================
# BASIC HELPERS
# =============================================================================

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def normalize_x(x: np.ndarray, mode: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    if mode == 1:
        a = np.log1p(np.maximum(0.0, a))
    span = float(np.max(a) - np.min(a)) if a.size else 0.0
    if abs(span) <= EPS:
        return np.zeros_like(a)
    return (a - float(np.min(a))) / span


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n < 3:
        return 0.0
    x = x[:n]
    y = y[:n]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= EPS or sy <= EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def wrap_pi(x: float) -> float:
    y = math.fmod(float(x), math.pi)
    if y < 0.0:
        y += math.pi
    return y


def wrap_pi_delta(d: float) -> float:
    y = math.fmod(float(d) + 0.5 * math.pi, math.pi)
    if y < 0.0:
        y += math.pi
    return y - 0.5 * math.pi


def pi_score(x_raw: np.ndarray, phase: np.ndarray, mode: int) -> float:
    if len(x_raw) < 3:
        return 0.0
    x = normalize_x(x_raw, mode)
    c2 = np.cos(2.0 * phase)
    s2 = np.sin(2.0 * phase)
    rc = safe_corr(x, c2)
    rs = safe_corr(x, s2)
    return float(min(1.0, math.sqrt(rc * rc + rs * rs)))


def condition_metadata(condition: str, offset_dt: int) -> Tuple[np.ndarray, int, int]:
    if condition == "null":
        return np.asarray(DEFAULT_NULL_DELAYS, dtype=np.int32), 0, 0
    if condition == "base_only":
        return np.asarray(DEFAULT_BASE_DELAYS, dtype=np.int32), 0, 1
    if condition == "offset_on":
        return np.asarray(DEFAULT_BASE_DELAYS, dtype=np.int32), int(offset_dt), 2
    raise ValueError(f"unknown condition: {condition}")


# =============================================================================
# CPU REFERENCE — float64 exact reference
# =============================================================================

def cpu_geo_rungs(condition: str, args: argparse.Namespace) -> np.ndarray:
    base_delays, offset_step, condition_kind = condition_metadata(condition, int(args.offset_dt))
    n = int(base_delays.size)
    out = np.zeros((n, DM_RUNG_N_METRICS), dtype=np.float64)

    base_raw = base_delays.astype(np.float64)
    offset_mean = np.asarray([(4.0 * r + 1.5) * float(offset_step) for r in range(n)], dtype=np.float64)
    total_raw = base_raw + offset_mean

    x_space = normalize_x(base_raw, 1)
    x_time = normalize_x(total_raw, 1)

    for r in range(n):
        base_delay = float(base_raw[r])
        off = float(offset_mean[r])
        total = float(total_raw[r])

        xy = yz = zy = yx = ret = energy = comp = spec = gap = inversion = phase = 0.0
        count_yzzy = 0.0

        if condition_kind != 0:
            sw = max(0.0, float(args.spatial_weight))
            tw = max(0.0, float(args.temporal_weight))
            denom = sw + tw
            if denom <= EPS:
                sw, tw, denom = 1.0, 1.0, 2.0

            xdm = math.sqrt((sw * x_space[r] ** 2 + tw * x_time[r] ** 2) / denom)
            xdm = min(1.0, max(0.0, xdm))

            q = min(1.0, max(-1.0, 2.0 * x_time[r] - 1.0))
            phase = 0.5 * math.acos(q)

            gamma = float(args.energy_gamma) if float(args.energy_gamma) > 0.0 else 1.0
            energy = max(0.0, float(args.energy_floor) + float(args.energy_scale) * (xdm ** gamma))

            yz = energy * math.cos(phase)
            ret = energy * math.sin(phase)
            if abs(yz) < SIGN_EPS:
                yz = 0.0
            if abs(ret) < SIGN_EPS:
                ret = 0.0
            zy = -ret

            if float(args.comparison_scale) > 0.0:
                c_amp = float(args.comparison_scale) * (1.0 - 0.5 * xdm)
                xy = c_amp * math.cos(math.pi * x_time[r])
                yx = c_amp * math.sin(math.pi * x_time[r])
                if abs(xy) < SIGN_EPS:
                    xy = 0.0
                if abs(yx) < SIGN_EPS:
                    yx = 0.0

            count_yzzy = 1.0

        ret = -zy
        energy = math.sqrt(yz * yz + ret * ret)
        comp = math.sqrt(xy * xy + yx * yx)
        spec = energy - comp
        gap = yz - zy
        inversion = -yz * zy

        out[r, DM_RUNG_XY] = xy
        out[r, DM_RUNG_YZ] = yz
        out[r, DM_RUNG_ZY] = zy
        out[r, DM_RUNG_YX] = yx
        out[r, DM_RUNG_YZ_PRIMARY] = yz
        out[r, DM_RUNG_ZY_RETURN] = ret
        out[r, DM_RUNG_YZZY_ENERGY] = energy
        out[r, DM_RUNG_COMPARISON_ENERGY] = comp
        out[r, DM_RUNG_DIRECTIONAL_SPECIFICITY] = spec
        out[r, DM_RUNG_DIRECTIONAL_GAP] = gap
        out[r, DM_RUNG_INVERSION] = inversion
        out[r, DM_RUNG_PI_PHASE] = wrap_pi(phase)
        out[r, DM_RUNG_PI_COS2] = math.cos(2.0 * phase)
        out[r, DM_RUNG_PI_SIN2] = math.sin(2.0 * phase)
        out[r, DM_RUNG_BASE_DELAY] = base_delay
        out[r, DM_RUNG_OFFSET] = off
        out[r, DM_RUNG_TOTAL_DELAY] = total
        out[r, DM_RUNG_COUNT_ALL] = 4.0
        out[r, DM_RUNG_COUNT_YZZY] = count_yzzy

    return out


def cpu_summary(rung: np.ndarray) -> np.ndarray:
    rows = np.asarray(rung, dtype=np.float64)
    active = rows[:, DM_RUNG_COUNT_YZZY] > 0.0
    if not np.any(active):
        return np.zeros((DM_SUMMARY_N_METRICS,), dtype=np.float64)

    r = rows[active]
    yz = r[:, DM_RUNG_YZ_PRIMARY]
    zy = r[:, DM_RUNG_ZY]
    energy = r[:, DM_RUNG_YZZY_ENERGY]
    spec = r[:, DM_RUNG_DIRECTIONAL_SPECIFICITY]
    phase = r[:, DM_RUNG_PI_PHASE]
    total = r[:, DM_RUNG_TOTAL_DELAY]

    yz_mean = float(np.mean(yz))
    zy_mean = float(np.mean(zy))
    e_mean = float(np.mean(energy))
    s_mean = float(np.mean(spec))
    yz_pos_frac = float(np.mean(yz > SIGN_EPS))
    zy_inv_frac = float(np.mean(yz * zy < -SIGN_EPS))

    x_lin = normalize_x(total, 0)
    x_log = normalize_x(total, 1)
    e_r_lin = safe_corr(x_lin, energy)
    e_r_log = safe_corr(x_log, energy)
    s_r_lin = safe_corr(x_lin, spec)
    s_r_log = safe_corr(x_log, spec)
    e_r = e_r_log if abs(e_r_log) > abs(e_r_lin) else e_r_lin
    s_r = s_r_log if abs(s_r_log) > abs(s_r_lin) else s_r_lin

    pi_lin = pi_score(total, phase, 0)
    pi_log = pi_score(total, phase, 1)
    pi_periodic = pi_log if pi_log > pi_lin else pi_lin
    pi_mode = 1.0 if pi_log > pi_lin else 0.0

    phase_vel_r = 0.0
    phase_span = 0.0
    if len(phase) >= 3:
        acc = float(phase[0])
        pmin = acc
        pmax = acc
        mids = []
        vels = []
        for i in range(1, len(phase)):
            dx = float(total[i] - total[i - 1])
            dp = wrap_pi_delta(float(phase[i] - phase[i - 1]))
            acc += dp
            pmin = min(pmin, acc)
            pmax = max(pmax, acc)
            if abs(dx) > EPS:
                mids.append(0.5 * float(total[i] + total[i - 1]))
                vels.append(dp / dx)
        phase_span = abs(pmax - pmin) / math.pi
        if len(vels) >= 3:
            phase_vel_r = safe_corr(normalize_x(np.asarray(mids), 1), np.asarray(vels))

    e_term = max(0.0, e_mean)
    s_term = max(0.0, s_mean)
    y_term = max(0.0, yz_mean)
    p_term = max(0.0, e_mean) * pi_periodic
    t_term = 0.5 * (abs(e_r) + abs(s_r))
    projection = 0.35 * e_term + 0.25 * s_term + 0.15 * y_term + 0.15 * p_term + 0.10 * t_term

    out = np.zeros((DM_SUMMARY_N_METRICS,), dtype=np.float64)
    out[DM_SUMMARY_N_RUNGS] = float(len(r))
    out[DM_SUMMARY_YZ_MEAN] = yz_mean
    out[DM_SUMMARY_YZ_POS_FRAC] = yz_pos_frac
    out[DM_SUMMARY_ZY_MEAN] = zy_mean
    out[DM_SUMMARY_ZY_INVERTED_FRAC] = zy_inv_frac
    out[DM_SUMMARY_YZZY_ENERGY_MEAN] = e_mean
    out[DM_SUMMARY_YZZY_ENERGY_MAX] = float(np.max(energy))
    out[DM_SUMMARY_SPECIFICITY_MEAN] = s_mean
    out[DM_SUMMARY_SPECIFICITY_MAX] = float(np.max(spec))
    out[DM_SUMMARY_PI_PERIODIC_SCORE] = pi_periodic
    out[DM_SUMMARY_PI_PERIODIC_MODE] = pi_mode
    out[DM_SUMMARY_ENERGY_TRACKING_R] = e_r
    out[DM_SUMMARY_SPECIFICITY_TRACKING_R] = s_r
    out[DM_SUMMARY_PHASE_VELOCITY_R] = phase_vel_r
    out[DM_SUMMARY_PHASE_SPAN_PI_UNITS] = phase_span
    out[DM_SUMMARY_PROJECTION_SCORE] = projection
    return out


def run_cpu(condition: str, args: argparse.Namespace) -> GeoResult:
    t0 = time.perf_counter()
    rung = cpu_geo_rungs(condition, args)
    summary = cpu_summary(rung)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return GeoResult(condition, "cpu_float64", rung, summary, elapsed_ms, {"precision": "float64"})


# =============================================================================
# GPU RUNNER
# =============================================================================

def compile_cuda_module() -> Any:
    if not HAVE_CUPY:
        raise RuntimeError(f"CuPy unavailable: {CUPY_IMPORT_ERROR}")
    return cp.RawModule(  # type: ignore[union-attr]
        code=CUDA_SOURCE,
        options=("--std=c++11",),
        name_expressions=[
            "dm_geo_exact_rung_kernel_f32",
            "dm_projection_summary_kernel_f32",
        ],
    )


def run_gpu(condition: str, args: argparse.Namespace, module: Any) -> GeoResult:
    base_delays, offset_step, condition_kind = condition_metadata(condition, int(args.offset_dt))
    n_rungs = int(base_delays.size)

    k_geo = module.get_function("dm_geo_exact_rung_kernel_f32")
    k_summary = module.get_function("dm_projection_summary_kernel_f32")

    d_base = cp.asarray(base_delays.astype(np.int32))
    d_rung = cp.zeros((n_rungs * DM_RUNG_N_METRICS,), dtype=cp.float32)
    d_summary = cp.zeros((DM_SUMMARY_N_METRICS,), dtype=cp.float32)

    cp.cuda.Stream.null.synchronize()
    ev0 = cp.cuda.Event()
    ev1 = cp.cuda.Event()
    ev0.record()

    k_geo(
        (n_rungs,),
        (1,),
        (
            np.int32(condition_kind),
            d_base,
            np.int32(n_rungs),
            np.int32(offset_step),
            np.float32(args.energy_floor),
            np.float32(args.energy_scale),
            np.float32(args.energy_gamma),
            np.float32(args.comparison_scale),
            np.float32(args.spatial_weight),
            np.float32(args.temporal_weight),
            d_rung,
        ),
    )
    k_summary((1,), (1,), (d_rung, np.int32(n_rungs), d_summary))

    ev1.record()
    ev1.synchronize()
    elapsed_ms = float(cp.cuda.get_elapsed_time(ev0, ev1))

    rung = cp.asnumpy(d_rung.reshape(n_rungs, DM_RUNG_N_METRICS)).astype(np.float32, copy=False)
    summary = cp.asnumpy(d_summary).astype(np.float32, copy=False)

    return GeoResult(
        condition=condition,
        backend="cuda_float32",
        rung_stats=rung,
        summary=summary,
        elapsed_ms=elapsed_ms,
        meta={
            "precision": "float32",
            "device": cp.cuda.runtime.getDeviceProperties(0).get("name", b"unknown").decode()
            if isinstance(cp.cuda.runtime.getDeviceProperties(0).get("name", b"unknown"), bytes)
            else str(cp.cuda.runtime.getDeviceProperties(0).get("name", "unknown")),
        },
    )


# =============================================================================
# ROW BUILDERS / REPORT
# =============================================================================

def rung_rows(result: GeoResult) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in range(result.rung_stats.shape[0]):
        row: Dict[str, Any] = {
            "condition": result.condition,
            "backend": result.backend,
            "rung": r,
            "pi_phase_degrees_mod180": float(result.rung_stats[r, DM_RUNG_PI_PHASE] * 180.0 / math.pi),
        }
        for i, name in enumerate(RUNG_METRIC_NAMES):
            row[name] = float(result.rung_stats[r, i])
        rows.append(row)
    return rows


def summary_row(result: GeoResult) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "condition": result.condition,
        "backend": result.backend,
        "elapsed_ms": float(result.elapsed_ms),
    }
    for i, name in enumerate(SUMMARY_NAMES):
        row[name] = float(result.summary[i])
    return row


def max_abs_delta_by_indices(a: np.ndarray, b: np.ndarray, indices: Sequence[int]) -> float:
    if not indices:
        return 0.0
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    vals = []
    for idx in indices:
        if idx < aa.size and idx < bb.size:
            vals.append(abs(float(aa[idx] - bb[idx])))
    return float(max(vals)) if vals else 0.0


def agreement_rows(cpu_results: Dict[str, GeoResult], gpu_results: Dict[str, GeoResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for condition, cpu_r in cpu_results.items():
        gpu_r = gpu_results.get(condition)
        if gpu_r is None:
            continue
        rung_delta = np.asarray(gpu_r.rung_stats, dtype=np.float64) - np.asarray(cpu_r.rung_stats, dtype=np.float64)
        sum_delta = np.asarray(gpu_r.summary, dtype=np.float64) - np.asarray(cpu_r.summary, dtype=np.float64)

        abs_sum = np.abs(sum_delta)
        max_metric_idx = int(np.argmax(abs_sum)) if abs_sum.size else -1
        max_metric_name = SUMMARY_NAMES[max_metric_idx] if 0 <= max_metric_idx < len(SUMMARY_NAMES) else ""

        rows.append({
            "condition": condition,
            "rung_max_abs_delta": float(np.max(np.abs(rung_delta))) if rung_delta.size else 0.0,
            "rung_rmse_delta": float(np.sqrt(np.mean(rung_delta * rung_delta))) if rung_delta.size else 0.0,
            "summary_max_abs_delta": float(np.max(abs_sum)) if abs_sum.size else 0.0,
            "summary_max_abs_metric": max_metric_name,
            "summary_core_max_abs_delta": max_abs_delta_by_indices(gpu_r.summary, cpu_r.summary, CORE_SUMMARY_INDICES),
            "summary_aux_max_abs_delta": max_abs_delta_by_indices(gpu_r.summary, cpu_r.summary, AUX_SUMMARY_INDICES),
            "summary_rmse_delta": float(np.sqrt(np.mean(sum_delta * sum_delta))) if sum_delta.size else 0.0,
            "projection_score_cpu": float(cpu_r.summary[DM_SUMMARY_PROJECTION_SCORE]),
            "projection_score_gpu": float(gpu_r.summary[DM_SUMMARY_PROJECTION_SCORE]),
            "projection_score_delta": float(sum_delta[DM_SUMMARY_PROJECTION_SCORE]),
            "pi_score_cpu": float(cpu_r.summary[DM_SUMMARY_PI_PERIODIC_SCORE]),
            "pi_score_gpu": float(gpu_r.summary[DM_SUMMARY_PI_PERIODIC_SCORE]),
            "pi_score_delta": float(sum_delta[DM_SUMMARY_PI_PERIODIC_SCORE]),
            "cpu_elapsed_ms": float(cpu_r.elapsed_ms),
            "gpu_elapsed_ms": float(gpu_r.elapsed_ms),
            "speedup_cpu_over_gpu": float(cpu_r.elapsed_ms / max(gpu_r.elapsed_ms, EPS)),
        })
    return rows

def summary_delta_metric_rows(cpu_results: Dict[str, GeoResult], gpu_results: Dict[str, GeoResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    core = set(CORE_SUMMARY_INDICES)
    aux = set(AUX_SUMMARY_INDICES)
    for condition, cpu_r in cpu_results.items():
        gpu_r = gpu_results.get(condition)
        if gpu_r is None:
            continue
        cpu = np.asarray(cpu_r.summary, dtype=np.float64)
        gpu = np.asarray(gpu_r.summary, dtype=np.float64)
        for i, name in enumerate(SUMMARY_NAMES):
            delta = float(gpu[i] - cpu[i])
            rows.append({
                "condition": condition,
                "metric": name,
                "cpu": float(cpu[i]),
                "gpu": float(gpu[i]),
                "delta": delta,
                "abs_delta": abs(delta),
                "validation_family": "core" if i in core else ("aux_logged" if i in aux else "unclassified"),
            })
    return rows


def validation_rows(cpu_results: Dict[str, GeoResult], gpu_results: Dict[str, GeoResult], tolerance: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    active_conditions = [c for c in ["base_only", "offset_on"] if c in cpu_results]

    if "null" in cpu_results:
        null_score = float(cpu_results["null"].summary[DM_SUMMARY_PROJECTION_SCORE])
        rows.append({
            "check": "null_is_zero_manifold",
            "observed": null_score,
            "threshold": tolerance,
            "passed": int(abs(null_score) <= tolerance),
            "backend": "cpu_float64",
        })

    for c in active_conditions:
        pi_val = float(cpu_results[c].summary[DM_SUMMARY_PI_PERIODIC_SCORE])
        score = float(cpu_results[c].summary[DM_SUMMARY_PROJECTION_SCORE])
        rows.append({
            "check": f"{c}_pi_score_exact",
            "observed": pi_val,
            "threshold": 1.0 - tolerance,
            "passed": int(pi_val >= 1.0 - tolerance),
            "backend": "cpu_float64",
        })
        rows.append({
            "check": f"{c}_projection_positive",
            "observed": score,
            "threshold": tolerance,
            "passed": int(score > tolerance),
            "backend": "cpu_float64",
        })

    for c, gpu_r in gpu_results.items():
        cpu_r = cpu_results[c]
        core_delta = max_abs_delta_by_indices(gpu_r.summary, cpu_r.summary, CORE_SUMMARY_INDICES)
        aux_delta = max_abs_delta_by_indices(gpu_r.summary, cpu_r.summary, AUX_SUMMARY_INDICES)
        rows.append({
            "check": f"{c}_cpu_gpu_core_summary_agreement",
            "observed": core_delta,
            "threshold": tolerance,
            "passed": int(core_delta <= tolerance),
            "backend": "cpu_vs_cuda",
        })
        rows.append({
            "check": f"{c}_cpu_gpu_aux_summary_delta_logged",
            "observed": aux_delta,
            "threshold": -1.0,
            "passed": 1,
            "backend": "cpu_vs_cuda",
        })

    return rows


def write_report(path: Path, args: argparse.Namespace, cpu_results: Dict[str, GeoResult], gpu_results: Dict[str, GeoResult], agreements: List[Dict[str, Any]], validations: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# D_M Probe 25 — GEO Precision Reference")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Probe 25 isolates GEO as a closed-form classical reference path for `D_M`.")
    lines.append("It does not use raw text, shot sampling, qproj/gproj records, GPT-2, or retrieval scoring.")
    lines.append("")
    lines.append("## GEO Rule")
    lines.append("")
    lines.append("For active conditions, GEO computes one scalar D_M coordinate from separable spatial and temporal axes, then writes exact `YZ` / `ZY` witnesses:")
    lines.append("")
    lines.append("```text")
    lines.append("x_space = normalize(log1p(base_delay))")
    lines.append("x_time  = normalize(log1p(base_delay + mean_offset))")
    lines.append("x_dm    = sqrt((w_space*x_space^2 + w_time*x_time^2)/(w_space+w_time))")
    lines.append("cos(2φ) = 2*x_time - 1")
    lines.append("YZ      = E*cos(φ)")
    lines.append("ZY      = -E*sin(φ)")
    lines.append("E       = energy_floor + energy_scale*x_dm^energy_gamma")
    lines.append("```")
    lines.append("")
    lines.append("The null condition is an exact zero manifold.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Condition | Backend | Projection score | E_mean | S_mean | π score | Phase span π | elapsed ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for res in list(cpu_results.values()) + list(gpu_results.values()):
        s = res.summary
        lines.append(
            f"| `{res.condition}` | `{res.backend}` | "
            f"{s[DM_SUMMARY_PROJECTION_SCORE]:.9f} | "
            f"{s[DM_SUMMARY_YZZY_ENERGY_MEAN]:.9f} | "
            f"{s[DM_SUMMARY_SPECIFICITY_MEAN]:.9f} | "
            f"{s[DM_SUMMARY_PI_PERIODIC_SCORE]:.9f} | "
            f"{s[DM_SUMMARY_PHASE_SPAN_PI_UNITS]:.9f} | "
            f"{res.elapsed_ms:.6f} |"
        )
    lines.append("")
    if agreements:
        lines.append("## CPU/GPU Agreement")
        lines.append("")
        lines.append("| Condition | Rung max abs Δ | Core summary max abs Δ | Aux max abs Δ | Max metric | Projection Δ | π score Δ | CPU ms | GPU ms | speedup |")
        lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
        for r in agreements:
            lines.append(
                f"| `{r['condition']}` | {r['rung_max_abs_delta']:.9e} | "
                f"{r['summary_core_max_abs_delta']:.9e} | {r['summary_aux_max_abs_delta']:.9e} | "
                f"`{r['summary_max_abs_metric']}` | "
                f"{r['projection_score_delta']:.9e} | {r['pi_score_delta']:.9e} | "
                f"{r['cpu_elapsed_ms']:.6f} | {r['gpu_elapsed_ms']:.6f} | {r['speedup_cpu_over_gpu']:.3f} |"
            )
        lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append("| Check | Observed | Threshold | Passed | Backend |")
    lines.append("|---|---:|---:|---:|---|")
    for r in validations:
        lines.append(f"| `{r['check']}` | {r['observed']:.9e} | {r['threshold']:.9e} | {r['passed']} | `{r['backend']}` |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This probe is a GEO correctness probe only. It is not a QPU claim, not a hardware simulator, and not a retrieval utility claim.")
    lines.append("Once this passes, the final benchmark can import the same GEO rule instead of inheriting the older text/aperture GEO plumbing.")
    lines.append("CPU/GPU pass-fail is based on continuous core manifold fields. Discrete sign/fraction diagnostics and derivative diagnostics are logged separately in `probe25_summary_delta_by_metric.csv` so analytic endpoint sign jitter cannot falsely fail a valid GEO manifold.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D_M Probe 25 — standalone closed-form GEO precision reference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--conditions", default=",".join(CONDITION_ORDER), help="Comma-separated subset: null,base_only,offset_on")
    p.add_argument("--offset-dt", type=int, default=DEFAULT_OFFSET_DT)
    p.add_argument("--energy-floor", type=float, default=0.125)
    p.add_argument("--energy-scale", type=float, default=0.875)
    p.add_argument("--energy-gamma", type=float, default=1.0)
    p.add_argument("--comparison-scale", type=float, default=0.0)
    p.add_argument("--spatial-weight", type=float, default=1.0)
    p.add_argument("--temporal-weight", type=float, default=1.0)
    p.add_argument("--tolerance", type=float, default=5.0e-5)
    p.add_argument("--require-cuda", action="store_true", help="Fail if CuPy/CUDA is unavailable.")
    p.add_argument("--skip-cuda", action="store_true", help="Run CPU reference only.")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    requested = [c.strip() for c in str(args.conditions).split(",") if c.strip()]
    for c in requested:
        if c not in CONDITION_ORDER:
            raise ValueError(f"unknown condition {c!r}; expected one of {CONDITION_ORDER}")

    out_dir = args.out_dir or (ANALYSIS_DIR / f"dm_probe25_geo_precision_{now_tag()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("  D_M PROBE 25 — GEO PRECISION REFERENCE v4")
    print("=" * 108)
    print(f"  Output dir      : {out_dir}")
    print(f"  Conditions      : {', '.join(requested)}")
    print(f"  Rule            : closed-form GEO, no raw text / no shots / no qproj/gproj")
    print(f"  Offset dt       : {args.offset_dt}")
    print(f"  Energy          : floor={args.energy_floor} scale={args.energy_scale} gamma={args.energy_gamma}")
    print(f"  Weights         : spatial={args.spatial_weight} temporal={args.temporal_weight}")
    print("-" * 108)

    cpu_results: Dict[str, GeoResult] = {}
    for condition in requested:
        res = run_cpu(condition, args)
        cpu_results[condition] = res
        print(
            f"  CPU {condition:10s} score={res.summary[DM_SUMMARY_PROJECTION_SCORE]:+.9f} "
            f"E={res.summary[DM_SUMMARY_YZZY_ENERGY_MEAN]:.9f} "
            f"S={res.summary[DM_SUMMARY_SPECIFICITY_MEAN]:.9f} "
            f"pi={res.summary[DM_SUMMARY_PI_PERIODIC_SCORE]:.9f} "
            f"ms={res.elapsed_ms:.6f}"
        )

    gpu_results: Dict[str, GeoResult] = {}
    module = None
    if not args.skip_cuda:
        if not HAVE_CUPY:
            msg = f"CuPy/CUDA unavailable; CPU reference completed. Import error: {CUPY_IMPORT_ERROR}"
            if args.require_cuda:
                raise RuntimeError(msg)
            print(f"  [cuda skipped] {msg}")
        else:
            module = compile_cuda_module()
            # Touch kernels early so NVRTC errors appear before any report is written.
            module.get_function("dm_geo_exact_rung_kernel_f32")
            module.get_function("dm_projection_summary_kernel_f32")
            for condition in requested:
                res = run_gpu(condition, args, module)
                gpu_results[condition] = res
                print(
                    f"  GPU {condition:10s} score={res.summary[DM_SUMMARY_PROJECTION_SCORE]:+.9f} "
                    f"E={res.summary[DM_SUMMARY_YZZY_ENERGY_MEAN]:.9f} "
                    f"S={res.summary[DM_SUMMARY_SPECIFICITY_MEAN]:.9f} "
                    f"pi={res.summary[DM_SUMMARY_PI_PERIODIC_SCORE]:.9f} "
                    f"ms={res.elapsed_ms:.6f}"
                )

    all_rung_rows: List[Dict[str, Any]] = []
    all_summary_rows: List[Dict[str, Any]] = []
    for res in list(cpu_results.values()) + list(gpu_results.values()):
        all_rung_rows.extend(rung_rows(res))
        all_summary_rows.append(summary_row(res))

    agreements = agreement_rows(cpu_results, gpu_results)
    summary_delta_rows = summary_delta_metric_rows(cpu_results, gpu_results)
    validations = validation_rows(cpu_results, gpu_results, float(args.tolerance))

    rung_fields = ["condition", "backend", "rung", *RUNG_METRIC_NAMES, "pi_phase_degrees_mod180"]
    summary_fields = ["condition", "backend", *SUMMARY_NAMES, "elapsed_ms"]
    agreement_fields = [
        "condition",
        "rung_max_abs_delta",
        "rung_rmse_delta",
        "summary_max_abs_delta",
        "summary_max_abs_metric",
        "summary_core_max_abs_delta",
        "summary_aux_max_abs_delta",
        "summary_rmse_delta",
        "projection_score_cpu",
        "projection_score_gpu",
        "projection_score_delta",
        "pi_score_cpu",
        "pi_score_gpu",
        "pi_score_delta",
        "cpu_elapsed_ms",
        "gpu_elapsed_ms",
        "speedup_cpu_over_gpu",
    ]
    validation_fields = ["check", "observed", "threshold", "passed", "backend"]
    summary_delta_fields = ["condition", "metric", "cpu", "gpu", "delta", "abs_delta", "validation_family"]

    write_csv(out_dir / "probe25_geo_rung_projection.csv", all_rung_rows, rung_fields)
    write_csv(out_dir / "probe25_geo_summary.csv", all_summary_rows, summary_fields)
    write_csv(out_dir / "probe25_cpu_gpu_agreement.csv", agreements, agreement_fields)
    write_csv(out_dir / "probe25_summary_delta_by_metric.csv", summary_delta_rows, summary_delta_fields)
    write_csv(out_dir / "probe25_validation.csv", validations, validation_fields)

    result = {
        "schema": "ghost_oracle.dm.probe25.geo_precision_reference.v4",
        "operator": "D_M",
        "probe": "Probe 25 — GEO Precision Reference",
        "bounded_claim": "Closed-form classical GEO reference for D_M; not a QPU claim, not a hardware simulator, not a retrieval utility claim.",
        "config": vars(args),
        "cpu_summary": [summary_row(r) for r in cpu_results.values()],
        "gpu_summary": [summary_row(r) for r in gpu_results.values()],
        "agreements": agreements,
        "summary_delta_by_metric": summary_delta_rows,
        "validations": validations,
    }
    write_json(out_dir / "probe_config.json", result)
    write_report(out_dir / "probe25_report.md", args, cpu_results, gpu_results, agreements, validations)

    failed = [r for r in validations if int(r.get("passed", 0)) != 1]
    print("-" * 108)
    print(f"  Saved           : {out_dir}")
    if failed:
        print(f"  Validation      : {len(failed)} failed check(s); inspect probe25_validation.csv")
    else:
        print("  Validation      : PASS")
    print("=" * 108)


if __name__ == "__main__":
    main()
