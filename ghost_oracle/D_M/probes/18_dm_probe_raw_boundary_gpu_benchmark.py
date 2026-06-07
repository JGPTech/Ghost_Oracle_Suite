#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
D_M RAW-BOUNDARY GPU-ONLY PROJECTION BENCHMARK
==============================================================================

Drop this file in:

    ghost_oracle/D_M/probes/d_m_raw_boundary_gpu_benchmark.py

Purpose
-------
GPU-only benchmark for the corrected D_M / GPT-2 comparison architecture.

Hard rule
---------
GPT-2 is never D_M input.

The same raw text is sent to independent products:

    1. qproj
        raw text bits -> CUDA fused D_M pair projection -> CUDA D_M kernels

    2. gproj
        raw text bits -> CUDA fused D_M pair projection -> CUDA D_M kernels

    3. geo
        raw text bits -> CUDA raw-data witness -> CUDA analytic GEO aperture

    4. GPT-2
        raw text -> GPT-2 native pre-softmax QK products on torch CUDA

No CPU fallback is provided. If CUDA/CuPy/Torch CUDA are unavailable, this script
raises immediately. CPU is used only for file I/O, deterministic text bit-bank
construction, and final CSV/printing after GPU results have been reduced.

D_M record projection
---------------------
For qproj/gproj, the existing base file provides the operator boundary:

    pair[tile, shot, 2]
    tile_rung_index[tile]
    tile_witness_index[tile]     0=XY, 1=YZ, 2=ZY, 3=YX
    tile_base_delay_dt[tile]
    tile_offset_dt[tile]
    tile_total_delay_dt[tile]

CUDA builds the raw data pair and applies the base in one fused kernel:

    projected_pair = data_pair XOR base_pair

Then all D_M correlator/projection/summary reductions are CUDA kernels.

GPT-2 path
----------
The GPT-2 path extracts only native pre-softmax QK products:

    L_ij = Q_i · K_j / sqrt(d)

No softmax. No outputs.attentions. No attention output path.

The QK matmul and witness reductions stay on torch CUDA. Only final tiny witness
matrices are copied to CPU for CSV output.

Outputs
-------
Creates:

    ghost_oracle/D_M/analysis/d_m_raw_boundary_gpu_benchmark_<timestamp>/

with:

    d_m_projected_manifolds.csv
    d_m_summary_stats.csv
    d_m_active_margin_scores.csv
    gpt2_free_qk_manifolds.csv
    gpt2_vs_dm_compatibility.csv
    benchmark_timers.csv
    probe_config.json

Example
-------
From repo root:

    python ghost_oracle/D_M/probes/d_m_raw_boundary_gpu_benchmark.py ^
      --text-file ghost_oracle/D_M/probes/d_m_probe_corpus.txt ^
      --split-lines ^
      --max-texts 256 ^
      --max-length 128 ^
      --gpt2-batch-size 16

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

PROBE_DIR = Path(__file__).resolve().parent
D_M_DIR = PROBE_DIR.parent
DATA_DIR = D_M_DIR / "data"
ANALYSIS_DIR = PROBE_DIR / "analyze"
KERNEL_DIR = D_M_DIR / "kernels"


# =============================================================================
# DEFAULT BASE FILES
# =============================================================================

DEFAULT_QPROJ_NULL = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fm4ihvjngc73aq3ccg.npz"
DEFAULT_QPROJ_BASE = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8flk2jo3njc73f0g560.npz"
DEFAULT_QPROJ_OFFSET = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fl82bo3njc73f0fgd0.npz"

DEFAULT_GPROJ_NULL = DATA_DIR / "dm_gpu_data_null_4096shots_seed9031229662612491082.npz"
DEFAULT_GPROJ_BASE = DATA_DIR / "dm_gpu_data_base_delay_4096shots_seed2669559634056472362.npz"
DEFAULT_GPROJ_OFFSET = DATA_DIR / "dm_gpu_data_offset_deformed_4096shots_seed6727069190982977623.npz"


# =============================================================================
# CONSTANTS MIRRORED FROM THE CUDA KERNEL
# =============================================================================

WITNESS_LABELS = ["XY", "YZ", "ZY", "YX"]
WITNESS_TO_INDEX = {x: i for i, x in enumerate(WITNESS_LABELS)}

DEFAULT_BASE_DELAYS = [0, 256, 1024, 4096, 16384]
DEFAULT_NULL_DELAYS = [0, 0, 0, 0, 0]
DEFAULT_OFFSET_DT = 128

EPS = 1.0e-12

DM_TILE_N_METRICS = 12
DM_RUNG_N_METRICS = 19
DM_SUMMARY_N_METRICS = 16

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

DM_SUMMARY_NAMES = [
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


# =============================================================================
# CUDA SOURCE
# =============================================================================

CUDA_SOURCE = r'''
extern "C" {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef DM_MAX_THREADS
#define DM_MAX_THREADS 1024
#endif

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
        DM_SUMMARY_PI_PERIODIC_MODE = 10,
        DM_SUMMARY_ENERGY_TRACKING_R = 11,
        DM_SUMMARY_SPECIFICITY_TRACKING_R = 12,
        DM_SUMMARY_PHASE_VELOCITY_R = 13,
        DM_SUMMARY_PHASE_SPAN_PI_UNITS = 14,
        DM_SUMMARY_PROJECTION_SCORE = 15,
        DM_SUMMARY_N_METRICS = 16
    };

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
        return b ? -1 : 1;
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

    __device__ __forceinline__ int dm_witness_shift(int witness_index)
    {
        int wi = witness_index & 3;
        if (wi == 0) return 5;
        if (wi == 3) return 11;
        return 0;
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
            (1009 * (tile + 1)
           + 9176 * (rung + 1)
           + 7919 * (wi + 1)
           + 13 * abs(total_delay_dt)) % nbits;

        int shift = dm_witness_shift(wi);
        int idx = (tile_phase + shot * stride) % nbits;

        int idx0 = idx;
        int idx1 = idx;

        if (wi == 1)
        {
            idx0 = idx;
            idx1 = (idx + delay_bits) % nbits;
        }
        else if (wi == 2)
        {
            idx0 = (idx + delay_bits) % nbits;
            idx1 = idx;
        }
        else if (wi == 0)
        {
            idx0 = idx;
            idx1 = (idx + delay_bits + shift) % nbits;
        }
        else if (wi == 3)
        {
            idx0 = (idx + delay_bits + shift) % nbits;
            idx1 = idx;
        }

        int out0 = (tile * shots + shot) * 2;
        int out1 = out0 + 1;

        projected_pair[out0] = bit_bank[idx0] ^ base_pair[out0];
        projected_pair[out1] = bit_bank[idx1] ^ base_pair[out1];
    }

    __global__ void dm_make_data_pair_from_bits_kernel_u8(
        const unsigned char *bit_bank,
        const int nbits,
        const int *tile_rung_index,
        const int *tile_witness_index,
        const int *tile_total_delay_dt,
        const int tiles,
        const int shots,
        const int data_stride,
        const int delay_scale,
        unsigned char *data_pair)
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
            (1009 * (tile + 1)
           + 9176 * (rung + 1)
           + 7919 * (wi + 1)
           + 13 * abs(total_delay_dt)) % nbits;

        int shift = dm_witness_shift(wi);
        int idx = (tile_phase + shot * stride) % nbits;

        int idx0 = idx;
        int idx1 = idx;

        if (wi == 1)
        {
            idx0 = idx;
            idx1 = (idx + delay_bits) % nbits;
        }
        else if (wi == 2)
        {
            idx0 = (idx + delay_bits) % nbits;
            idx1 = idx;
        }
        else if (wi == 0)
        {
            idx0 = idx;
            idx1 = (idx + delay_bits + shift) % nbits;
        }
        else if (wi == 3)
        {
            idx0 = (idx + delay_bits + shift) % nbits;
            idx1 = idx;
        }

        int out0 = (tile * shots + shot) * 2;
        int out1 = out0 + 1;

        data_pair[out0] = bit_bank[idx0];
        data_pair[out1] = bit_bank[idx1];
    }

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

            if (yv > 0.0f)
                ++yz_pos;
            if (yv * zv < 0.0f)
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

        if (r >= n_rungs || threadIdx.x != 0)
            return;

        if (n_rungs > 64)
            return;

        float lin_min = 0.0f, lin_max = 0.0f;
        float log_min = 0.0f, log_max = 0.0f;

        for (int i = 0; i < n_rungs; ++i)
        {
            float off_mean_i = ((float)(4 * i) + 1.5f) * (float)offset_dt;
            float total_i = (float)base_delays_dt[i] + off_mean_i;
            float log_i = log1pf(fmaxf(0.0f, total_i));

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

    __global__ void dm_apply_geo_witness_to_tile_stats_kernel_f32(
        float *tile_stats,
        const float *geo_rung_stats,
        const int *tile_rung_index,
        const int *tile_witness_index,
        const int tiles)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= tiles)
            return;

        int r = tile_rung_index[t];
        int wi = tile_witness_index[t];
        if (r < 0 || wi < 0 || wi > 3)
            return;

        float geo_w = geo_rung_stats[dm_rung_idx(r, wi)];
        tile_stats[dm_tile_idx(t, DM_TILE_CONNECTED)] *= geo_w;
    }
}
'''


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DMRecordBase:
    substrate: str
    condition: str
    source: str
    pair: np.ndarray
    tile_rung: np.ndarray
    tile_witness: np.ndarray
    tile_base_delay: np.ndarray
    tile_offset: np.ndarray
    tile_total_delay: np.ndarray
    meta: Dict[str, Any]


@dataclass
class DMGpuResult:
    substrate: str
    condition: str
    source: str
    projection_kind: str
    n_rungs: int
    rung_stats: np.ndarray
    summary: np.ndarray
    gpu_ms: float
    wall_seconds: float
    work_items: int
    meta: Dict[str, Any]

    @property
    def witnesses(self) -> np.ndarray:
        return self.rung_stats[:, [DM_RUNG_XY, DM_RUNG_YZ, DM_RUNG_ZY, DM_RUNG_YX]]

    @property
    def energy(self) -> np.ndarray:
        return self.rung_stats[:, DM_RUNG_YZZY_ENERGY]

    @property
    def phase(self) -> np.ndarray:
        return self.rung_stats[:, DM_RUNG_PI_PHASE]

    @property
    def specificity(self) -> np.ndarray:
        return self.rung_stats[:, DM_RUNG_DIRECTIONAL_SPECIFICITY]

    @property
    def projection_score(self) -> float:
        return float(self.summary[15]) if self.summary.size >= 16 else 0.0


@dataclass
class GPT2FreeManifold:
    model_name: str
    layer: int
    head: int
    center_mode: str
    n_rungs: int
    witnesses: np.ndarray
    energy: np.ndarray
    phase: np.ndarray
    specificity: np.ndarray
    meta: Dict[str, Any]


# =============================================================================
# GENERAL HELPERS
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def decode_str_array(arr: Any) -> List[str]:
    a = np.asarray(arr)
    out: List[str] = []
    for x in a.reshape(-1):
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or str(s).strip() == "":
        return None
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n <= 0:
        return 0.0
    x = x[:n]
    y = y[:n]
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < EPS or ny < EPS:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def phase_distance_pi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    d = np.abs(aa - bb)
    d = np.minimum(d, math.pi - d)
    return d / math.pi


def summary_dict(summary: np.ndarray) -> Dict[str, float]:
    return {name: float(summary[i]) for i, name in enumerate(DM_SUMMARY_NAMES)}


def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


# =============================================================================
# CUDA / TORCH REQUIREMENTS
# =============================================================================

def require_cupy_cuda():
    try:
        import cupy as cp
    except Exception as e:
        raise RuntimeError("This benchmark requires CuPy with CUDA. No CPU fallback exists.") from e

    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        raise RuntimeError("CuPy could not access a CUDA device. No CPU fallback exists.") from e

    if count <= 0:
        raise RuntimeError("No CUDA devices found for CuPy. No CPU fallback exists.")

    return cp


def require_torch_cuda(no_gpt2: bool):
    if no_gpt2:
        return None, None, None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError("GPT-2 benchmark requires torch and transformers. No CPU fallback exists.") from e

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false. No CPU fallback exists.")

    return torch, AutoModelForCausalLM, AutoTokenizer


def compile_cuda_module(cp: Any):
    return cp.RawModule(
        code=CUDA_SOURCE,
        options=("--std=c++11",),
        name_expressions=[
            "dm_make_projected_pair_from_bits_kernel_u8",
            "dm_make_data_pair_from_bits_kernel_u8",
            "dm_tile_correlator_kernel_u8",
            "dm_rung_projection_kernel_f32",
            "dm_projection_summary_kernel_f32",
            "dm_geo_rung_projection_kernel_f32",
            "dm_apply_geo_witness_to_tile_stats_kernel_f32",
        ],
    )


# =============================================================================
# TEXT INPUT AND BIT BANK
# =============================================================================

def load_texts(args: argparse.Namespace) -> List[str]:
    if args.text_file:
        p = Path(args.text_file)
        if not p.exists():
            raise FileNotFoundError(f"text file not found: {p}")
        raw = p.read_text(encoding="utf-8", errors="replace")
        if args.split_lines:
            texts = [line.strip() for line in raw.splitlines() if line.strip()]
        else:
            chunks = [x.strip() for x in raw.split("\n\n") if x.strip()]
            texts = chunks if chunks else [raw.strip()]
    elif args.text:
        texts = [args.text]
    else:
        texts = [
            "D_M projects raw data through bounded qproj, gproj, and geo operator bases.",
            "GPT-2 is a separate free-running reference path and is not used as D_M input.",
            "The same input data can produce different products under different projection boundaries.",
        ]

    if args.max_texts and args.max_texts > 0:
        texts = texts[: int(args.max_texts)]

    if not texts:
        raise ValueError("no input texts loaded")

    return texts


def make_text_bit_bank(texts: Sequence[str], min_bits: int, salt: str = "D_M_RAW_BOUNDARY_GPU_ONLY") -> np.ndarray:
    text = "\n".join(texts)
    raw_bytes = text.encode("utf-8", errors="replace") or b"empty"

    raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bits = np.unpackbits(raw_arr, bitorder="little").astype(np.uint8)

    chunks = [bits]
    total = int(bits.size)
    counter = 0
    seed = hashlib.sha256(salt.encode("utf-8") + b"\0" + raw_bytes).digest()

    while total < min_bits:
        h = hashlib.sha256(seed + counter.to_bytes(8, "little") + raw_bytes).digest()
        h_arr = np.frombuffer(h, dtype=np.uint8)
        h_bits = np.unpackbits(h_arr, bitorder="little").astype(np.uint8)
        chunks.append(h_bits)
        total += int(h_bits.size)
        counter += 1

    return np.ascontiguousarray(np.concatenate(chunks)[:max(min_bits, total)].astype(np.uint8))


# =============================================================================
# LOAD QPROJ/GPROJ RECORD BASES
# =============================================================================

def optional_int_array(z: Any, name: str, tiles: int, default: int = 0) -> np.ndarray:
    if name in z.files:
        arr = np.asarray(z[name], dtype=np.int32)
        if arr.shape[0] == tiles:
            return np.ascontiguousarray(arr)
    return np.full((tiles,), default, dtype=np.int32)


def load_record_base(path: Path, substrate: str, condition_hint: str) -> DMRecordBase:
    if not path.exists():
        raise FileNotFoundError(f"Missing D_M base file: {path}")

    z = np.load(path, allow_pickle=True)

    if "pair" in z.files:
        pair = np.asarray(z["pair"], dtype=np.uint8)
    else:
        pair_keys = sorted(
            [k for k in z.files if k.startswith("pair_tile")],
            key=lambda k: int(k.replace("pair_tile", "")),
        )
        if not pair_keys:
            raise KeyError(f"{path} has no pair or pair_tile* arrays.")
        pair = np.stack([np.asarray(z[k], dtype=np.uint8) for k in pair_keys], axis=0)

    if pair.ndim != 3 or pair.shape[2] != 2:
        raise ValueError(f"{path} pair must have shape (tiles, shots, 2), got {pair.shape}")

    pair = np.ascontiguousarray(pair.astype(np.uint8, copy=False))
    tiles = int(pair.shape[0])
    shots = int(pair.shape[1])

    if "tile_rung_index" in z.files:
        tile_rung = np.asarray(z["tile_rung_index"], dtype=np.int32)
    else:
        tile_rung = (np.arange(tiles) // 4).astype(np.int32)

    if "tile_witness_index" in z.files:
        tile_witness = np.asarray(z["tile_witness_index"], dtype=np.int32)
    elif "tile_witness_label" in z.files:
        labels = decode_str_array(z["tile_witness_label"])
        tile_witness = np.asarray(
            [WITNESS_TO_INDEX.get(x, i % 4) for i, x in enumerate(labels[:tiles])],
            dtype=np.int32,
        )
    else:
        tile_witness = (np.arange(tiles) % 4).astype(np.int32)

    if tile_rung.shape[0] != tiles or tile_witness.shape[0] != tiles:
        raise ValueError(f"{path} tile metadata length does not match tiles={tiles}")

    tile_base = optional_int_array(z, "tile_base_delay_dt", tiles, 0)
    tile_off = optional_int_array(z, "tile_offset_dt", tiles, 0)
    tile_total = optional_int_array(z, "tile_total_delay_dt", tiles, 0)

    return DMRecordBase(
        substrate=substrate,
        condition=condition_hint,
        source=str(path),
        pair=pair,
        tile_rung=np.ascontiguousarray(tile_rung.astype(np.int32, copy=False)),
        tile_witness=np.ascontiguousarray(tile_witness.astype(np.int32, copy=False)),
        tile_base_delay=np.ascontiguousarray(tile_base.astype(np.int32, copy=False)),
        tile_offset=np.ascontiguousarray(tile_off.astype(np.int32, copy=False)),
        tile_total_delay=np.ascontiguousarray(tile_total.astype(np.int32, copy=False)),
        meta={
            "path": str(path),
            "substrate": substrate,
            "condition": condition_hint,
            "tiles": tiles,
            "shots": shots,
            "n_rungs": int(np.max(tile_rung)) + 1 if tiles else 0,
        },
    )


# =============================================================================
# GPU D_M RUNNERS
# =============================================================================

def run_record_base_gpu(
    cp: Any,
    module: Any,
    bit_bank_gpu: Any,
    base: DMRecordBase,
    args: argparse.Namespace,
) -> DMGpuResult:
    make_projected = module.get_function("dm_make_projected_pair_from_bits_kernel_u8")
    tile_corr = module.get_function("dm_tile_correlator_kernel_u8")
    rung_proj = module.get_function("dm_rung_projection_kernel_f32")
    summary_kernel = module.get_function("dm_projection_summary_kernel_f32")

    tiles = int(base.pair.shape[0])
    shots = int(base.pair.shape[1])
    n_rungs = int(np.max(base.tile_rung)) + 1 if tiles else 0

    base_pair_gpu = cp.asarray(base.pair.reshape(-1), dtype=cp.uint8)
    tile_rung_gpu = cp.asarray(base.tile_rung, dtype=cp.int32)
    tile_witness_gpu = cp.asarray(base.tile_witness, dtype=cp.int32)
    tile_base_gpu = cp.asarray(base.tile_base_delay, dtype=cp.int32)
    tile_offset_gpu = cp.asarray(base.tile_offset, dtype=cp.int32)
    tile_total_gpu = cp.asarray(base.tile_total_delay, dtype=cp.int32)

    projected_pair_gpu = cp.empty((tiles * shots * 2,), dtype=cp.uint8)
    tile_stats_gpu = cp.empty((tiles * DM_TILE_N_METRICS,), dtype=cp.float32)
    rung_stats_gpu = cp.empty((n_rungs * DM_RUNG_N_METRICS,), dtype=cp.float32)
    summary_gpu = cp.empty((DM_SUMMARY_N_METRICS,), dtype=cp.float32)

    threads_linear = int(args.dm_linear_threads)
    blocks_linear = (tiles * shots + threads_linear - 1) // threads_linear
    tile_threads = int(args.dm_tile_threads)

    cp.cuda.Stream.null.synchronize()
    ev_start = cp.cuda.Event()
    ev_end = cp.cuda.Event()
    wall_start = time.perf_counter()
    ev_start.record()

    make_projected(
        (blocks_linear,),
        (threads_linear,),
        (
            bit_bank_gpu,
            np.int32(bit_bank_gpu.size),
            base_pair_gpu,
            tile_rung_gpu,
            tile_witness_gpu,
            tile_total_gpu,
            np.int32(tiles),
            np.int32(shots),
            np.int32(args.data_stride),
            np.int32(args.delay_scale),
            projected_pair_gpu,
        ),
    )

    tile_corr(
        (tiles,),
        (tile_threads,),
        (projected_pair_gpu, np.int32(tiles), np.int32(shots), tile_stats_gpu),
    )

    rung_proj(
        (n_rungs,),
        (1,),
        (
            tile_stats_gpu,
            tile_rung_gpu,
            tile_witness_gpu,
            tile_base_gpu,
            tile_offset_gpu,
            tile_total_gpu,
            np.int32(tiles),
            np.int32(n_rungs),
            rung_stats_gpu,
        ),
    )

    summary_kernel((1,), (1,), (rung_stats_gpu, np.int32(n_rungs), summary_gpu))

    ev_end.record()
    ev_end.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    gpu_ms = float(cp.cuda.get_elapsed_time(ev_start, ev_end))

    rung_stats = cp.asnumpy(rung_stats_gpu.reshape(n_rungs, DM_RUNG_N_METRICS))
    summary = cp.asnumpy(summary_gpu)

    return DMGpuResult(
        substrate=base.substrate,
        condition=base.condition,
        source=base.source,
        projection_kind="cuda_raw_text_xor_record_base",
        n_rungs=n_rungs,
        rung_stats=rung_stats,
        summary=summary,
        gpu_ms=gpu_ms,
        wall_seconds=wall_seconds,
        work_items=tiles * shots,
        meta={
            "tiles": tiles,
            "shots": shots,
            "data_stride": int(args.data_stride),
            "delay_scale": int(args.delay_scale),
        },
    )


def make_geo_metadata(condition: str, offset_dt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    if condition == "null":
        base_delays = np.asarray(DEFAULT_NULL_DELAYS, dtype=np.int32)
        offset_step = 0
        condition_kind = 0
    elif condition == "base_only":
        base_delays = np.asarray(DEFAULT_BASE_DELAYS, dtype=np.int32)
        offset_step = 0
        condition_kind = 1
    elif condition == "offset_on":
        base_delays = np.asarray(DEFAULT_BASE_DELAYS, dtype=np.int32)
        offset_step = int(offset_dt)
        condition_kind = 2
    else:
        raise ValueError(f"unknown geo condition: {condition}")

    tile_rung: List[int] = []
    tile_witness: List[int] = []
    tile_base: List[int] = []
    tile_offset: List[int] = []
    tile_total: List[int] = []

    for r, bd in enumerate(base_delays.tolist()):
        for wi in range(4):
            off = (4 * r + wi) * offset_step
            tile_rung.append(r)
            tile_witness.append(wi)
            tile_base.append(int(bd))
            tile_offset.append(int(off))
            tile_total.append(int(bd) + int(off))

    return (
        base_delays,
        np.ascontiguousarray(np.asarray(tile_rung, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(tile_witness, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(tile_base, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(tile_offset, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(tile_total, dtype=np.int32)),
        condition_kind,
    )


def run_geo_gpu(
    cp: Any,
    module: Any,
    bit_bank_gpu: Any,
    condition: str,
    args: argparse.Namespace,
) -> DMGpuResult:
    make_data = module.get_function("dm_make_data_pair_from_bits_kernel_u8")
    tile_corr = module.get_function("dm_tile_correlator_kernel_u8")
    geo_rung_kernel = module.get_function("dm_geo_rung_projection_kernel_f32")
    apply_geo = module.get_function("dm_apply_geo_witness_to_tile_stats_kernel_f32")
    rung_proj = module.get_function("dm_rung_projection_kernel_f32")
    summary_kernel = module.get_function("dm_projection_summary_kernel_f32")

    (
        base_delays,
        tile_rung,
        tile_witness,
        tile_base,
        tile_offset,
        tile_total,
        condition_kind,
    ) = make_geo_metadata(condition, args.geo_offset_dt)

    n_rungs = int(base_delays.shape[0])
    tiles = int(tile_rung.shape[0])
    shots = int(args.geo_shots)

    base_delays_gpu = cp.asarray(base_delays, dtype=cp.int32)
    tile_rung_gpu = cp.asarray(tile_rung, dtype=cp.int32)
    tile_witness_gpu = cp.asarray(tile_witness, dtype=cp.int32)
    tile_base_gpu = cp.asarray(tile_base, dtype=cp.int32)
    tile_offset_gpu = cp.asarray(tile_offset, dtype=cp.int32)
    tile_total_gpu = cp.asarray(tile_total, dtype=cp.int32)

    data_pair_gpu = cp.empty((tiles * shots * 2,), dtype=cp.uint8)
    tile_stats_gpu = cp.empty((tiles * DM_TILE_N_METRICS,), dtype=cp.float32)
    analytic_geo_rung_gpu = cp.empty((n_rungs * DM_RUNG_N_METRICS,), dtype=cp.float32)
    projected_rung_gpu = cp.empty((n_rungs * DM_RUNG_N_METRICS,), dtype=cp.float32)
    summary_gpu = cp.empty((DM_SUMMARY_N_METRICS,), dtype=cp.float32)

    threads_linear = int(args.dm_linear_threads)
    blocks_linear = (tiles * shots + threads_linear - 1) // threads_linear
    tile_threads = int(args.dm_tile_threads)
    blocks_tiles = (tiles + threads_linear - 1) // threads_linear

    cp.cuda.Stream.null.synchronize()
    ev_start = cp.cuda.Event()
    ev_end = cp.cuda.Event()
    wall_start = time.perf_counter()
    ev_start.record()

    make_data(
        (blocks_linear,),
        (threads_linear,),
        (
            bit_bank_gpu,
            np.int32(bit_bank_gpu.size),
            tile_rung_gpu,
            tile_witness_gpu,
            tile_total_gpu,
            np.int32(tiles),
            np.int32(shots),
            np.int32(args.data_stride),
            np.int32(args.delay_scale),
            data_pair_gpu,
        ),
    )

    tile_corr((tiles,), (tile_threads,), (data_pair_gpu, np.int32(tiles), np.int32(shots), tile_stats_gpu))

    geo_rung_kernel(
        (n_rungs,),
        (1,),
        (
            np.int32(condition_kind),
            base_delays_gpu,
            np.int32(n_rungs),
            np.int32(args.geo_offset_dt if condition == "offset_on" else 0),
            np.float32(args.geo_base_energy),
            np.float32(args.geo_energy_gain),
            np.float32(args.geo_comparison_scale),
            np.float32(args.geo_offset_deform),
            np.float32(args.geo_phase_scale),
            analytic_geo_rung_gpu,
        ),
    )

    apply_geo((blocks_tiles,), (threads_linear,), (tile_stats_gpu, analytic_geo_rung_gpu, tile_rung_gpu, tile_witness_gpu, np.int32(tiles)))

    rung_proj(
        (n_rungs,),
        (1,),
        (
            tile_stats_gpu,
            tile_rung_gpu,
            tile_witness_gpu,
            tile_base_gpu,
            tile_offset_gpu,
            tile_total_gpu,
            np.int32(tiles),
            np.int32(n_rungs),
            projected_rung_gpu,
        ),
    )

    summary_kernel((1,), (1,), (projected_rung_gpu, np.int32(n_rungs), summary_gpu))

    ev_end.record()
    ev_end.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    gpu_ms = float(cp.cuda.get_elapsed_time(ev_start, ev_end))

    rung_stats = cp.asnumpy(projected_rung_gpu.reshape(n_rungs, DM_RUNG_N_METRICS))
    summary = cp.asnumpy(summary_gpu)

    return DMGpuResult(
        substrate="geo",
        condition=condition,
        source=f"cuda_analytic_geo_{condition}",
        projection_kind="cuda_raw_text_times_analytic_geo_aperture",
        n_rungs=n_rungs,
        rung_stats=rung_stats,
        summary=summary,
        gpu_ms=gpu_ms,
        wall_seconds=wall_seconds,
        work_items=tiles * shots,
        meta={
            "tiles": tiles,
            "shots": shots,
            "data_stride": int(args.data_stride),
            "delay_scale": int(args.delay_scale),
            "geo_offset_dt": int(args.geo_offset_dt),
            "geo_base_energy": float(args.geo_base_energy),
            "geo_energy_gain": float(args.geo_energy_gain),
            "geo_comparison_scale": float(args.geo_comparison_scale),
            "geo_offset_deform": float(args.geo_offset_deform),
            "geo_phase_scale": float(args.geo_phase_scale),
        },
    )


# =============================================================================
# GPT-2 GPU-ONLY QK PATH
# =============================================================================

def torch_dtype_from_arg(torch: Any, dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"unknown GPT-2 dtype: {dtype_name}")


def reduce_qk_logits_to_witness_vectors_gpu(torch: Any, logits: Any, center_mode: str) -> Tuple[Any, int]:
    # logits shape: [batch, heads, seq, seq]
    L = logits.float()

    if center_mode == "none":
        pass
    elif center_mode == "row":
        L = L - L.mean(dim=-1, keepdim=True)
    elif center_mode == "global":
        L = L - L.mean(dim=(-1, -2), keepdim=True)
    elif center_mode == "zscore":
        mu = L.mean(dim=(-1, -2), keepdim=True)
        sd = L.std(dim=(-1, -2), keepdim=True).clamp_min(1.0e-12)
        L = (L - mu) / sd
    else:
        raise ValueError(f"unknown center mode: {center_mode}")

    batch, heads, seq_len, _ = L.shape
    I, J = torch.tril_indices(seq_len, seq_len, offset=-1, device=L.device)
    if I.numel() <= 0:
        return torch.zeros((heads, 4), device=L.device, dtype=torch.float32), 0

    j_cmp = (J + 1) % seq_len
    j_cmp = torch.where(j_cmp == I, (j_cmp + 1) % seq_len, j_cmp)

    i_cmp = (I - 1) % seq_len
    i_cmp = torch.where(i_cmp == J, (i_cmp - 1) % seq_len, i_cmp)

    yz = L[:, :, I, J].mean(dim=(0, 2))
    zy = L[:, :, J, I].mean(dim=(0, 2))
    xy = L[:, :, I, j_cmp].mean(dim=(0, 2))
    yx = L[:, :, J, i_cmp].mean(dim=(0, 2))

    vectors = torch.stack([xy, yz, zy, yx], dim=-1).contiguous()
    return vectors, int(batch * I.numel())


def manifold_from_witness_vector_np(vector: np.ndarray, n_rungs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    witnesses = np.tile(np.asarray(vector, dtype=np.float64).reshape(1, 4), (n_rungs, 1))
    xy = witnesses[:, 0]
    yz = witnesses[:, 1]
    zy = witnesses[:, 2]
    yx = witnesses[:, 3]
    reciprocal = -zy
    energy = np.sqrt(yz * yz + reciprocal * reciprocal)
    comp = np.sqrt(xy * xy + yx * yx)
    specificity = energy - comp
    phase = np.mod(np.arctan2(reciprocal, yz), math.pi)
    return witnesses, energy, phase, specificity


def load_gpt2_free_manifolds_gpu(
    torch: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    texts: Sequence[str],
    args: argparse.Namespace,
    n_rungs: int,
) -> Tuple[List[GPT2FreeManifold], Dict[str, Any], List[Dict[str, Any]]]:
    center_modes = [x.strip() for x in args.center_modes.split(",") if x.strip()]
    if not center_modes:
        center_modes = ["row"]

    dtype = torch_dtype_from_arg(torch, args.gpt2_dtype)

    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    try:
        torch.set_float32_matmul_precision(args.matmul_precision)
    except Exception:
        pass

    print(f"  loading tokenizer/model : {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval().cuda()
    model_load_seconds = time.perf_counter() - model_load_start

    n_layer = len(model.transformer.h)
    n_head = int(model.config.n_head)
    n_embd = int(model.config.n_embd)
    head_dim = n_embd // n_head

    layers_arg = parse_int_list(args.layers)
    heads_arg = parse_int_list(args.heads)
    layer_ids = list(range(n_layer)) if layers_arg is None else [x for x in layers_arg if 0 <= x < n_layer]
    head_ids = list(range(n_head)) if heads_arg is None else [x for x in heads_arg if 0 <= x < n_head]

    if not layer_ids:
        raise ValueError("No valid GPT-2 layers selected.")
    if not head_ids:
        raise ValueError("No valid GPT-2 heads selected.")

    # Accumulator structure: (layer, center_mode) -> sum[heads,4], count scalar.
    accum: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for layer in layer_ids:
        for center_mode in center_modes:
            accum[(layer, center_mode)] = {
                "sum": torch.zeros((n_head, 4), device="cuda", dtype=torch.float64),
                "count": 0,
            }

    total_qk_products = 0
    total_reduction_pairs = 0
    qk_gpu_ms = 0.0
    reduction_gpu_ms = 0.0
    n_batches = int(math.ceil(len(texts) / max(1, int(args.gpt2_batch_size))))

    batch_rows: List[Dict[str, Any]] = []

    for batch_index in range(n_batches):
        lo = batch_index * int(args.gpt2_batch_size)
        hi = min(len(texts), lo + int(args.gpt2_batch_size))
        batch_texts = list(texts[lo:hi])
        print(f"  GPT-2 batch {batch_index + 1}/{n_batches}: texts {lo}:{hi}")

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=int(args.max_length),
        )
        input_ids = enc["input_ids"].cuda(non_blocking=True)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.cuda(non_blocking=True)

        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev2 = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                out = model.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            hidden_states = out.hidden_states

            ev0.record()
            for layer in layer_ids:
                block = model.transformer.h[layer]
                x = hidden_states[layer]

                with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                    qkv = block.attn.c_attn(x)
                    q, k, _v = qkv.split(n_embd, dim=2)
                    batch, seq_len, _ = q.shape
                    q = q.view(batch, seq_len, n_head, head_dim).permute(0, 2, 1, 3).contiguous()
                    k = k.view(batch, seq_len, n_head, head_dim).permute(0, 2, 1, 3).contiguous()
                    logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(head_dim))

                ev1.record()
                torch.cuda.synchronize()
                qk_gpu_ms += float(ev0.elapsed_time(ev1))

                ev_reduce_start = torch.cuda.Event(enable_timing=True)
                ev_reduce_end = torch.cuda.Event(enable_timing=True)
                ev_reduce_start.record()
                for center_mode in center_modes:
                    vectors, pairs = reduce_qk_logits_to_witness_vectors_gpu(torch, logits, center_mode)
                    accum[(layer, center_mode)]["sum"] += vectors.double() * float(pairs)
                    accum[(layer, center_mode)]["count"] += int(pairs)
                    total_reduction_pairs += int(pairs)
                ev_reduce_end.record()
                ev_reduce_end.synchronize()
                reduction_gpu_ms += float(ev_reduce_start.elapsed_time(ev_reduce_end))

                total_qk_products += int(batch * n_head * seq_len * seq_len)

                # Restart QK timer baseline for the next selected layer.
                ev0.record()

        batch_rows.append({
            "batch_index": batch_index,
            "texts_lo": lo,
            "texts_hi": hi,
            "seq_len": int(input_ids.shape[1]),
            "selected_layers": len(layer_ids),
            "selected_heads": len(head_ids),
        })

        del out, hidden_states, input_ids, attention_mask
        torch.cuda.empty_cache()

    manifolds: List[GPT2FreeManifold] = []
    for layer in layer_ids:
        for center_mode in center_modes:
            item = accum[(layer, center_mode)]
            count = int(item["count"])
            if count <= 0:
                continue
            mean_vectors_gpu = item["sum"] / float(count)
            mean_vectors_np = mean_vectors_gpu.detach().float().cpu().numpy()

            for head in head_ids:
                witnesses, energy, phase, specificity = manifold_from_witness_vector_np(mean_vectors_np[head], n_rungs)
                manifolds.append(
                    GPT2FreeManifold(
                        model_name=args.model,
                        layer=layer,
                        head=head,
                        center_mode=center_mode,
                        n_rungs=n_rungs,
                        witnesses=witnesses,
                        energy=energy,
                        phase=phase,
                        specificity=specificity,
                        meta={
                            "dtype": args.gpt2_dtype,
                            "allow_tf32": bool(args.allow_tf32),
                            "matmul_precision": args.matmul_precision,
                            "counted_pairs": count,
                            "note": "Free GPT-2 raw QK product reference. Not D_M input.",
                        },
                    )
                )

    timing = {
        "model_load_seconds": model_load_seconds,
        "qk_gpu_ms": qk_gpu_ms,
        "reduction_gpu_ms": reduction_gpu_ms,
        "total_gpu_ms": qk_gpu_ms + reduction_gpu_ms,
        "total_qk_products": total_qk_products,
        "total_reduction_pairs": total_reduction_pairs,
        "layers": layer_ids,
        "heads": head_ids,
        "center_modes": center_modes,
        "dtype": args.gpt2_dtype,
        "allow_tf32": bool(args.allow_tf32),
        "matmul_precision": args.matmul_precision,
    }

    return manifolds, timing, batch_rows


# =============================================================================
# ROW BUILDERS AND SCORING
# =============================================================================

def dm_rows(results: Sequence[DMGpuResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in results:
        for r in range(m.n_rungs):
            rows.append({
                "substrate": m.substrate,
                "condition": m.condition,
                "source": m.source,
                "projection_kind": m.projection_kind,
                "rung": r,
                "XY": float(m.rung_stats[r, DM_RUNG_XY]),
                "YZ": float(m.rung_stats[r, DM_RUNG_YZ]),
                "ZY": float(m.rung_stats[r, DM_RUNG_ZY]),
                "YX": float(m.rung_stats[r, DM_RUNG_YX]),
                "energy": float(m.rung_stats[r, DM_RUNG_YZZY_ENERGY]),
                "comparison_energy": float(m.rung_stats[r, DM_RUNG_COMPARISON_ENERGY]),
                "specificity": float(m.rung_stats[r, DM_RUNG_DIRECTIONAL_SPECIFICITY]),
                "phase": float(m.rung_stats[r, DM_RUNG_PI_PHASE]),
                "phase_pi": float(m.rung_stats[r, DM_RUNG_PI_PHASE] / math.pi),
                "base_delay": float(m.rung_stats[r, DM_RUNG_BASE_DELAY]),
                "offset": float(m.rung_stats[r, DM_RUNG_OFFSET]),
                "total_delay": float(m.rung_stats[r, DM_RUNG_TOTAL_DELAY]),
                "summary_projection_score": m.projection_score,
                "gpu_ms_for_method_condition": m.gpu_ms,
            })
    return rows


def dm_summary_rows(results: Sequence[DMGpuResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in results:
        s = summary_dict(m.summary)
        row: Dict[str, Any] = {
            "substrate": m.substrate,
            "condition": m.condition,
            "source": m.source,
            "projection_kind": m.projection_kind,
            "gpu_ms": m.gpu_ms,
            "wall_seconds": m.wall_seconds,
            "work_items": m.work_items,
            "work_items_per_gpu_second": float(m.work_items) / max(m.gpu_ms / 1000.0, EPS),
        }
        row.update(s)
        rows.append(row)
    return sorted(rows, key=lambda r: (str(r["substrate"]), -float(r["projection_score"])))


def gpt2_rows(manifolds: Sequence[GPT2FreeManifold]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for h in manifolds:
        for r in range(h.n_rungs):
            rows.append({
                "model_name": h.model_name,
                "layer": h.layer,
                "head": h.head,
                "center_mode": h.center_mode,
                "rung": r,
                "XY": float(h.witnesses[r, 0]),
                "YZ": float(h.witnesses[r, 1]),
                "ZY": float(h.witnesses[r, 2]),
                "YX": float(h.witnesses[r, 3]),
                "energy": float(h.energy[r]),
                "phase": float(h.phase[r]),
                "phase_pi": float(h.phase[r] / math.pi),
                "specificity": float(h.specificity[r]),
                "note": "Free GPT-2 raw QK product reference. Not D_M input.",
            })
    return rows


def active_margin_rows(results: Sequence[DMGpuResult]) -> List[Dict[str, Any]]:
    by_sub: Dict[str, Dict[str, DMGpuResult]] = {}
    for r in results:
        by_sub.setdefault(r.substrate, {})[r.condition] = r

    rows: List[Dict[str, Any]] = []
    for substrate, d in by_sub.items():
        null = d.get("null")
        if null is None:
            continue
        for condition in ["base_only", "offset_on"]:
            active = d.get(condition)
            if active is None:
                continue
            rows.append({
                "substrate": substrate,
                "active_condition": condition,
                "active_score": active.projection_score,
                "null_score": null.projection_score,
                "score_margin": active.projection_score - null.projection_score,
                "active_energy_mean": float(active.summary[5]),
                "null_energy_mean": float(null.summary[5]),
                "energy_margin": float(active.summary[5] - null.summary[5]),
                "active_specificity_mean": float(active.summary[7]),
                "null_specificity_mean": float(null.summary[7]),
                "specificity_margin": float(active.summary[7] - null.summary[7]),
                "active_gpu_ms": active.gpu_ms,
                "null_gpu_ms": null.gpu_ms,
            })
    return sorted(rows, key=lambda r: (str(r["substrate"]), -float(r["score_margin"])))


def compatibility_rows(dm_results: Sequence[DMGpuResult], gpt2_manifolds: Sequence[GPT2FreeManifold]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in dm_results:
        d_vec = d.witnesses.reshape(-1)
        d_energy = d.energy
        d_spec = d.specificity
        d_phase = d.phase
        for g in gpt2_manifolds:
            g_vec = g.witnesses.reshape(-1)
            wc = cosine(d_vec, g_vec)
            ec = cosine(d_energy, g.energy)
            sc = cosine(d_spec, g.specificity)
            pd = float(np.mean(phase_distance_pi(d_phase, g.phase)))
            score = 0.40 * wc + 0.25 * ec + 0.20 * sc + 0.15 * (1.0 - pd)
            rows.append({
                "dm_substrate": d.substrate,
                "dm_condition": d.condition,
                "dm_projection_score": d.projection_score,
                "gpt2_model": g.model_name,
                "gpt2_layer": g.layer,
                "gpt2_head": g.head,
                "gpt2_center_mode": g.center_mode,
                "witness_cosine": wc,
                "energy_cosine": ec,
                "specificity_cosine": sc,
                "phase_distance_pi_mean": pd,
                "compatibility_score": score,
                "note": "Compatibility only. GPT-2 was not D_M input.",
            })
    return sorted(rows, key=lambda r: r["compatibility_score"], reverse=True)


def timer_rows(dm_results: Sequence[DMGpuResult], gpt2_timing: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in dm_results:
        rows.append({
            "family": "D_M",
            "method": r.substrate,
            "condition": r.condition,
            "gpu_ms": r.gpu_ms,
            "wall_seconds": r.wall_seconds,
            "work_items": r.work_items,
            "items_per_gpu_second": float(r.work_items) / max(r.gpu_ms / 1000.0, EPS),
            "note": r.projection_kind,
        })
    if gpt2_timing:
        rows.append({
            "family": "GPT-2",
            "method": "gpt2_raw_qk",
            "condition": "all_selected_layers_heads",
            "gpu_ms": float(gpt2_timing.get("total_gpu_ms", 0.0)),
            "wall_seconds": "",
            "work_items": int(gpt2_timing.get("total_qk_products", 0)),
            "items_per_gpu_second": float(gpt2_timing.get("total_qk_products", 0)) / max(float(gpt2_timing.get("total_gpu_ms", 0.0)) / 1000.0, EPS),
            "note": f"QK matmul + GPU witness reductions; dtype={gpt2_timing.get('dtype')}",
        })
    return rows


# =============================================================================
# PRINTING
# =============================================================================

def print_method_leaders(results: Sequence[DMGpuResult], top: int) -> None:
    print()
    print("[TOP D_M SUMMARY STATS BY METHOD]")
    for method in ["geo", "qproj", "gproj"]:
        group = [r for r in results if r.substrate == method]
        if not group:
            continue
        group = sorted(group, key=lambda r: r.projection_score, reverse=True)
        print(f"\n  {method.upper()}")
        print("  " + "-" * 104)
        print("  condition   score       E_mean      E_max       S_mean      S_max       YZ_mean     ZY_mean     pi_score   gpu_ms")
        for r in group[:top]:
            s = summary_dict(r.summary)
            print(
                f"  {r.condition:10s} "
                f"{s['projection_score']:+.6f}  "
                f"{s['yzzy_energy_mean']:+.6f}  "
                f"{s['yzzy_energy_max']:+.6f}  "
                f"{s['specificity_mean']:+.6f}  "
                f"{s['specificity_max']:+.6f}  "
                f"{s['yz_mean']:+.6f}  "
                f"{s['zy_mean']:+.6f}  "
                f"{s['pi_periodic_score']:+.6f}  "
                f"{r.gpu_ms:8.3f}"
            )


def print_top_rungs_by_method(results: Sequence[DMGpuResult], top: int) -> None:
    print()
    print("[TOP D_M RUNG STATS BY METHOD]")
    for method in ["geo", "qproj", "gproj"]:
        rows: List[Tuple[float, DMGpuResult, int]] = []
        for r in results:
            if r.substrate != method:
                continue
            for rung in range(r.n_rungs):
                rows.append((float(r.rung_stats[rung, DM_RUNG_YZZY_ENERGY]), r, rung))
        if not rows:
            continue
        rows.sort(key=lambda x: x[0], reverse=True)
        print(f"\n  {method.upper()}")
        print("  " + "-" * 104)
        print("  condition   rung  energy      specificity  YZ          ZY          phase/pi    total_delay")
        for _, r, rung in rows[:top]:
            rs = r.rung_stats[rung]
            print(
                f"  {r.condition:10s} "
                f"{rung:4d}  "
                f"{rs[DM_RUNG_YZZY_ENERGY]:+.6f}  "
                f"{rs[DM_RUNG_DIRECTIONAL_SPECIFICITY]:+.6f}  "
                f"{rs[DM_RUNG_YZ]:+.6f}  "
                f"{rs[DM_RUNG_ZY]:+.6f}  "
                f"{rs[DM_RUNG_PI_PHASE] / math.pi:+.6f}  "
                f"{rs[DM_RUNG_TOTAL_DELAY]:+.1f}"
            )


def print_timer_benchmark(dm_results: Sequence[DMGpuResult], gpt2_timing: Optional[Dict[str, Any]]) -> None:
    print()
    print("[GPU TIMER BENCHMARK]")
    print("  " + "-" * 104)
    by_method: Dict[str, List[DMGpuResult]] = {}
    for r in dm_results:
        by_method.setdefault(r.substrate, []).append(r)
    for method in ["geo", "qproj", "gproj"]:
        group = by_method.get(method, [])
        if not group:
            continue
        total_ms = sum(r.gpu_ms for r in group)
        total_items = sum(r.work_items for r in group)
        rate = float(total_items) / max(total_ms / 1000.0, EPS)
        print(f"  D_M {method:5s}: {total_ms:10.3f} ms GPU | items={total_items:,} | items/s={rate:,.0f}")
    d_total_ms = sum(r.gpu_ms for r in dm_results)
    d_total_items = sum(r.work_items for r in dm_results)
    d_rate = float(d_total_items) / max(d_total_ms / 1000.0, EPS)
    print(f"  D_M total: {d_total_ms:10.3f} ms GPU | items={d_total_items:,} | items/s={d_rate:,.0f}")

    if gpt2_timing:
        g_ms = float(gpt2_timing.get("total_gpu_ms", 0.0))
        g_items = int(gpt2_timing.get("total_qk_products", 0))
        g_rate = float(g_items) / max(g_ms / 1000.0, EPS)
        print(f"  GPT-2 QK : {g_ms:10.3f} ms GPU | QK products={g_items:,} | products/s={g_rate:,.0f}")
        if d_total_ms > 0:
            print(f"  GPT/D_M time ratio: {g_ms / d_total_ms:.3f}x")


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPU-only D_M raw-boundary projection benchmark with GPT-2 CUDA QK comparison.")

    p.add_argument("--qproj-null", type=Path, default=DEFAULT_QPROJ_NULL)
    p.add_argument("--qproj-base", type=Path, default=DEFAULT_QPROJ_BASE)
    p.add_argument("--qproj-offset", type=Path, default=DEFAULT_QPROJ_OFFSET)
    p.add_argument("--gproj-null", type=Path, default=DEFAULT_GPROJ_NULL)
    p.add_argument("--gproj-base", type=Path, default=DEFAULT_GPROJ_BASE)
    p.add_argument("--gproj-offset", type=Path, default=DEFAULT_GPROJ_OFFSET)

    p.add_argument("--no-geo", action="store_true")
    p.add_argument("--no-gpt2", action="store_true")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--gpt2-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--layers", default="", help="Comma-separated GPT-2 layer ids. Default: all.")
    p.add_argument("--heads", default="", help="Comma-separated GPT-2 head ids. Default: all.")
    p.add_argument("--center-modes", default="row", help="Comma-separated GPT-2 centering modes: none,row,global,zscore.")
    p.add_argument("--gpt2-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--matmul-precision", default="high", choices=["highest", "high", "medium"])

    p.add_argument("--text", default="")
    p.add_argument("--text-file", default="")
    p.add_argument("--split-lines", action="store_true")
    p.add_argument("--max-texts", type=int, default=0)

    p.add_argument("--data-stride", type=int, default=17, help="Stride through raw text bit bank for D_M data signal.")
    p.add_argument("--delay-scale", type=int, default=1, help="Convert D_M dt to raw bit offset by delay_bits=dt//scale.")
    p.add_argument("--min-bit-multiplier", type=int, default=8, help="Minimum bit bank multiplier over largest base records.")

    p.add_argument("--dm-linear-threads", type=int, default=256)
    p.add_argument("--dm-tile-threads", type=int, default=256, help="Power-of-two CUDA threads per tile correlator block.")

    p.add_argument("--geo-shots", type=int, default=4096)
    p.add_argument("--geo-offset-dt", type=int, default=DEFAULT_OFFSET_DT)
    p.add_argument("--geo-base-energy", type=float, default=0.04)
    p.add_argument("--geo-energy-gain", type=float, default=0.28)
    p.add_argument("--geo-comparison-scale", type=float, default=0.010)
    p.add_argument("--geo-offset-deform", type=float, default=0.18)
    p.add_argument("--geo-phase-scale", type=float, default=0.37)

    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--top", type=int, default=3, help="Top rows to print per method.")

    return p


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = build_arg_parser().parse_args()

    if not is_power_of_two(int(args.dm_tile_threads)) or int(args.dm_tile_threads) > 1024:
        raise ValueError("--dm-tile-threads must be a power of two <= 1024")
    if int(args.dm_linear_threads) <= 0:
        raise ValueError("--dm-linear-threads must be positive")

    cp = require_cupy_cuda()
    torch, AutoModelForCausalLM, AutoTokenizer = require_torch_cuda(args.no_gpt2)

    tag = now_tag()
    out_dir = args.out_dir or (ANALYSIS_DIR / f"dm_probe_18_raw_boundary_gpu_benchmark_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 112)
    print("  D_M RAW-BOUNDARY GPU-ONLY PROJECTION BENCHMARK")
    print("=" * 112)
    print(f"  D_M dir       : {D_M_DIR}")
    print(f"  Output dir    : {out_dir}")
    print(f"  CUDA device   : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode() if isinstance(cp.cuda.runtime.getDeviceProperties(0)['name'], bytes) else cp.cuda.runtime.getDeviceProperties(0)['name']}")
    print(f"  Rule          : GPT-2 QK is comparison only; D_M gets raw text bits")
    print(f"  CPU fallback  : removed / forbidden")

    print()
    print("[COMPILE] CUDA D_M kernels")
    module = compile_cuda_module(cp)
    # Touch one function so NVRTC errors appear early.
    module.get_function("dm_tile_correlator_kernel_u8")
    print("  compiled      : yes")

    print()
    print("[LOAD] input texts")
    texts = load_texts(args)
    print(f"  text sequences: {len(texts)}")

    print()
    print("[LOAD] qproj/gproj record bases")
    base_specs = [
        ("qproj", "null", args.qproj_null),
        ("qproj", "base_only", args.qproj_base),
        ("qproj", "offset_on", args.qproj_offset),
        ("gproj", "null", args.gproj_null),
        ("gproj", "base_only", args.gproj_base),
        ("gproj", "offset_on", args.gproj_offset),
    ]

    record_bases: List[DMRecordBase] = []
    for substrate, condition, path in base_specs:
        b = load_record_base(path, substrate, condition)
        record_bases.append(b)
        print(f"  {substrate:5s} {condition:9s} tiles={b.pair.shape[0]:3d} shots={b.pair.shape[1]:6d} rungs={b.meta['n_rungs']:3d} {path}")

    n_rungs = min(int(b.meta["n_rungs"]) for b in record_bases)
    max_records = max(int(np.prod(b.pair.shape[:2])) for b in record_bases)
    max_delay = max(int(np.max(np.abs(b.tile_total_delay))) for b in record_bases)
    min_bits = max(8192, int(args.min_bit_multiplier) * max_records + max_delay + 1024)

    print()
    print("[BUILD] raw text bit bank")
    bit_bank = make_text_bit_bank(texts, min_bits=min_bits)
    bit_bank_gpu = cp.asarray(bit_bank, dtype=cp.uint8)
    cp.cuda.Stream.null.synchronize()
    print(f"  bits          : {bit_bank.size:,}")
    print(f"  data stride   : {args.data_stride}")
    print(f"  delay scale   : {args.delay_scale}")

    print()
    print("[PROJECT] qproj/gproj via fused CUDA projection + CUDA D_M kernels")
    dm_results: List[DMGpuResult] = []
    for b in record_bases:
        r = run_record_base_gpu(cp, module, bit_bank_gpu, b, args)
        dm_results.append(r)
        s = summary_dict(r.summary)
        print(
            f"  {r.substrate:5s} {r.condition:9s} "
            f"score={s['projection_score']:+.6f} "
            f"E_mean={s['yzzy_energy_mean']:+.6f} "
            f"S_mean={s['specificity_mean']:+.6f} "
            f"gpu_ms={r.gpu_ms:.3f}"
        )

    if not args.no_geo:
        print()
        print("[PROJECT] geo via CUDA raw-data witness + CUDA analytic aperture")
        for condition in ["null", "base_only", "offset_on"]:
            r = run_geo_gpu(cp, module, bit_bank_gpu, condition, args)
            dm_results.append(r)
            s = summary_dict(r.summary)
            print(
                f"  {r.substrate:5s} {r.condition:9s} "
                f"score={s['projection_score']:+.6f} "
                f"E_mean={s['yzzy_energy_mean']:+.6f} "
                f"S_mean={s['specificity_mean']:+.6f} "
                f"gpu_ms={r.gpu_ms:.3f}"
            )

    gpt2_manifolds: List[GPT2FreeManifold] = []
    gpt2_timing: Optional[Dict[str, Any]] = None
    gpt2_batch_rows: List[Dict[str, Any]] = []

    if not args.no_gpt2:
        print()
        print("[GPT-2] raw QK GPU benchmark path")
        assert torch is not None and AutoModelForCausalLM is not None and AutoTokenizer is not None
        gpt2_manifolds, gpt2_timing, gpt2_batch_rows = load_gpt2_free_manifolds_gpu(
            torch,
            AutoModelForCausalLM,
            AutoTokenizer,
            texts,
            args,
            n_rungs=n_rungs,
        )
        print(
            f"  GPT-2 manifolds: {len(gpt2_manifolds)} | "
            f"QK gpu_ms={float(gpt2_timing['qk_gpu_ms']):.3f} | "
            f"reduce gpu_ms={float(gpt2_timing['reduction_gpu_ms']):.3f} | "
            f"total gpu_ms={float(gpt2_timing['total_gpu_ms']):.3f}"
        )

    print_method_leaders(dm_results, top=int(args.top))
    print_top_rungs_by_method(dm_results, top=int(args.top))
    print_timer_benchmark(dm_results, gpt2_timing)

    print()
    print("[WRITE] CSV / JSON outputs")

    dm_manifest_rows = dm_rows(dm_results)
    write_csv(
        out_dir / "d_m_projected_manifolds.csv",
        dm_manifest_rows,
        [
            "substrate", "condition", "source", "projection_kind", "rung",
            "XY", "YZ", "ZY", "YX", "energy", "comparison_energy", "specificity",
            "phase", "phase_pi", "base_delay", "offset", "total_delay",
            "summary_projection_score", "gpu_ms_for_method_condition",
        ],
    )

    dms_rows = dm_summary_rows(dm_results)
    write_csv(
        out_dir / "d_m_summary_stats.csv",
        dms_rows,
        [
            "substrate", "condition", "source", "projection_kind", "gpu_ms", "wall_seconds", "work_items", "work_items_per_gpu_second",
            *DM_SUMMARY_NAMES,
        ],
    )

    margin_rows = active_margin_rows(dm_results)
    write_csv(
        out_dir / "d_m_active_margin_scores.csv",
        margin_rows,
        [
            "substrate", "active_condition", "active_score", "null_score", "score_margin",
            "active_energy_mean", "null_energy_mean", "energy_margin",
            "active_specificity_mean", "null_specificity_mean", "specificity_margin",
            "active_gpu_ms", "null_gpu_ms",
        ],
    )

    g_rows = gpt2_rows(gpt2_manifolds)
    write_csv(
        out_dir / "gpt2_free_qk_manifolds.csv",
        g_rows,
        [
            "model_name", "layer", "head", "center_mode", "rung",
            "XY", "YZ", "ZY", "YX", "energy", "phase", "phase_pi", "specificity", "note",
        ],
    )

    comp_rows = compatibility_rows(dm_results, gpt2_manifolds) if gpt2_manifolds else []
    write_csv(
        out_dir / "gpt2_vs_dm_compatibility.csv",
        comp_rows,
        [
            "dm_substrate", "dm_condition", "dm_projection_score",
            "gpt2_model", "gpt2_layer", "gpt2_head", "gpt2_center_mode",
            "witness_cosine", "energy_cosine", "specificity_cosine", "phase_distance_pi_mean", "compatibility_score", "note",
        ],
    )

    timers = timer_rows(dm_results, gpt2_timing)
    write_csv(
        out_dir / "benchmark_timers.csv",
        timers,
        ["family", "method", "condition", "gpu_ms", "wall_seconds", "work_items", "items_per_gpu_second", "note"],
    )

    if gpt2_batch_rows:
        write_csv(
            out_dir / "gpt2_batch_log.csv",
            gpt2_batch_rows,
            ["batch_index", "texts_lo", "texts_hi", "seq_len", "selected_layers", "selected_heads"],
        )

    write_json(
        out_dir / "probe_config.json",
        {
            "script": "d_m_raw_boundary_gpu_benchmark.py",
            "timestamp": tag,
            "gpu_only": True,
            "cpu_fallback": False,
            "d_m_dir": D_M_DIR,
            "output_dir": out_dir,
            "texts": {
                "count": len(texts),
                "text_file": args.text_file,
                "split_lines": bool(args.split_lines),
                "max_texts": int(args.max_texts),
                "bit_bank_bits": int(bit_bank.size),
            },
            "dm_args": {
                "data_stride": int(args.data_stride),
                "delay_scale": int(args.delay_scale),
                "dm_linear_threads": int(args.dm_linear_threads),
                "dm_tile_threads": int(args.dm_tile_threads),
                "geo_shots": int(args.geo_shots),
                "geo_offset_dt": int(args.geo_offset_dt),
                "geo_base_energy": float(args.geo_base_energy),
                "geo_energy_gain": float(args.geo_energy_gain),
                "geo_comparison_scale": float(args.geo_comparison_scale),
                "geo_offset_deform": float(args.geo_offset_deform),
                "geo_phase_scale": float(args.geo_phase_scale),
            },
            "gpt2": gpt2_timing,
            "record_bases": [b.meta for b in record_bases],
            "timers": timers,
        },
    )

    print(f"  wrote: {out_dir / 'd_m_projected_manifolds.csv'}")
    print(f"  wrote: {out_dir / 'd_m_summary_stats.csv'}")
    print(f"  wrote: {out_dir / 'd_m_active_margin_scores.csv'}")
    print(f"  wrote: {out_dir / 'gpt2_free_qk_manifolds.csv'}")
    print(f"  wrote: {out_dir / 'gpt2_vs_dm_compatibility.csv'}")
    print(f"  wrote: {out_dir / 'benchmark_timers.csv'}")
    print(f"  wrote: {out_dir / 'probe_config.json'}")

    print()
    print("[DONE] GPU-only D_M / GPT-2 benchmark complete.")


if __name__ == "__main__":
    main()
