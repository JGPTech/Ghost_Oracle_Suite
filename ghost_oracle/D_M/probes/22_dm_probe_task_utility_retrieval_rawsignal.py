#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
D_M TASK UTILITY RETRIEVAL GPU BENCHMARK
==============================================================================

Drop this file in:

    ghost_oracle/D_M/probes/d_m_task_utility_retrieval_gpu_benchmark_rawsignal.py

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

EPS = 1.0e-16

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
    """
    Convert raw input text into a deterministic bit bank without destroying
    local similarity.

    IMPORTANT FOR D_M RETRIEVAL
    ---------------------------
    D_M is being tested as a raw-boundary structural projector. The qproj/gproj
    base pair and geo aperture already bound the signal. The bit bank must
    therefore preserve the raw text structure.

    The previous retrieval script expanded short texts with SHA-256 chunks seeded
    by the whole text and, worse, salted the call with the row index. That made
    even an exact query/candidate pair encode differently, and light character
    noise avalanched across almost the entire bank. GPT-2 did not have this
    problem, so the comparison was invalid.

    This version uses raw-repeat expansion:

        UTF-8 bytes -> little-endian bits -> tile/repeat to min_bits

    No per-row salt. No hash avalanche. No normalization.
    The raw text remains the signal; the D_M base remains the boundary.
    "salt" is kept only for API compatibility and is intentionally ignored.
    """
    text = "\n".join(str(t) for t in texts)
    raw_bytes = text.encode("utf-8", errors="replace") or b"empty"

    raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bits = np.unpackbits(raw_arr, bitorder="little").astype(np.uint8)

    if bits.size <= 0:
        bits = np.zeros((8,), dtype=np.uint8)

    target = max(int(min_bits), int(bits.size))
    reps = int(math.ceil(float(target) / float(bits.size)))
    expanded = np.tile(bits, reps)[:target]
    return np.ascontiguousarray(expanded.astype(np.uint8, copy=False))


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
# RETRIEVAL TASK BUILDERS
# =============================================================================

@dataclass
class RetrievalTask:
    query_texts: List[str]
    candidate_texts: List[str]
    true_candidate_indices: List[int]
    candidate_rows: List[Dict[str, Any]]


def stable_rng(seed_text: str) -> np.random.Generator:
    h = hashlib.sha256(seed_text.encode("utf-8", errors="replace")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False) & 0x7fffffffffffffff
    return np.random.default_rng(seed)


def normalize_ws(text: str) -> str:
    return " ".join(str(text).split())


def shuffle_words(text: str, seed_text: str) -> str:
    words = normalize_ws(text).split()
    if len(words) <= 2:
        return text[::-1]
    rng = stable_rng(seed_text)
    idx = np.arange(len(words))
    rng.shuffle(idx)
    shuffled = [words[int(i)] for i in idx]
    if shuffled == words:
        shuffled = shuffled[1:] + shuffled[:1]
    return " ".join(shuffled)


def light_char_noise(text: str, seed_text: str, frac: float) -> str:
    s = list(str(text))
    if not s:
        return text
    rng = stable_rng(seed_text)
    n = max(1, int(round(len(s) * max(0.0, min(0.25, frac)))))
    positions = rng.choice(len(s), size=min(n, len(s)), replace=False)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    for p in positions:
        old = s[int(p)]
        if old.isspace():
            continue
        s[int(p)] = alphabet[int(rng.integers(0, len(alphabet)))]
    return "".join(s)


def rotate_text(text: str, seed_text: str) -> str:
    s = str(text)
    if len(s) <= 3:
        return s[::-1]
    rng = stable_rng(seed_text)
    k = int(rng.integers(1, max(2, len(s) - 1)))
    return s[k:] + s[:k]


def make_positive_text(text: str, idx: int, kind: str, noise_frac: float) -> str:
    if kind == "exact":
        return str(text)
    if kind == "whitespace":
        return normalize_ws(text)
    if kind == "light_noise":
        return light_char_noise(text, f"positive:{idx}:{text}", noise_frac)
    if kind == "prefix_suffix":
        return f"BEGIN {normalize_ws(text)} END"
    raise ValueError(f"unknown --positive-kind: {kind}")


def make_decoy_text(text: str, idx: int, kind: str, noise_frac: float) -> str:
    if kind == "shuffle":
        return shuffle_words(text, f"decoy-shuffle:{idx}:{text}")
    if kind == "light_noise":
        return light_char_noise(text, f"decoy-noise:{idx}:{text}", max(noise_frac, 0.035))
    if kind == "rotate":
        return rotate_text(text, f"decoy-rotate:{idx}:{text}")
    if kind == "reverse":
        return str(text)[::-1]
    if kind == "prefix_suffix":
        return f"DECOY {normalize_ws(text)} DECOY"
    raise ValueError(f"unknown decoy kind: {kind}")


def build_retrieval_task(texts: Sequence[str], args: argparse.Namespace) -> RetrievalTask:
    base_texts = [normalize_ws(t) for t in texts if normalize_ws(t)]
    if args.max_queries and args.max_queries > 0:
        base_texts = base_texts[: int(args.max_queries)]
    if len(base_texts) < 2:
        raise ValueError("Retrieval task needs at least 2 non-empty text sequences.")

    decoy_kinds = [x.strip() for x in str(args.decoy_kinds).split(",") if x.strip()]
    query_texts: List[str] = []
    candidate_texts: List[str] = []
    true_candidate_indices: List[int] = []
    candidate_rows: List[Dict[str, Any]] = []

    for i, text in enumerate(base_texts):
        query_texts.append(text)
        true_idx = len(candidate_texts)
        true_candidate_indices.append(true_idx)
        positive = make_positive_text(text, i, args.positive_kind, float(args.noise_frac))
        candidate_texts.append(positive)
        candidate_rows.append({
            "candidate_index": true_idx,
            "source_query_index": i,
            "candidate_kind": f"positive_{args.positive_kind}",
            "is_positive": 1,
            "text_preview": positive[:200],
        })

        for kind in decoy_kinds:
            d = make_decoy_text(text, i, kind, float(args.noise_frac))
            cidx = len(candidate_texts)
            candidate_texts.append(d)
            candidate_rows.append({
                "candidate_index": cidx,
                "source_query_index": i,
                "candidate_kind": f"same_source_decoy_{kind}",
                "is_positive": 0,
                "text_preview": d[:200],
            })

    return RetrievalTask(
        query_texts=query_texts,
        candidate_texts=candidate_texts,
        true_candidate_indices=true_candidate_indices,
        candidate_rows=candidate_rows,
    )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

# D_M retrieval features must preserve raw bounded operator amplitude.
# The qproj/gproj base pair and geo aperture already bound the signal.
# Phase-only/correlation-only diagnostics are useful in the manifold report, but
# they can wash out retrieval utility by replacing amplitude evidence with
# normalized shape evidence.
DM_FEATURE_RUNG_COLS_BOUNDED_CORE = [
    DM_RUNG_XY,
    DM_RUNG_YZ,
    DM_RUNG_ZY,
    DM_RUNG_YX,
    DM_RUNG_YZ_PRIMARY,
    DM_RUNG_ZY_RETURN,
    DM_RUNG_YZZY_ENERGY,
    DM_RUNG_COMPARISON_ENERGY,
    DM_RUNG_DIRECTIONAL_SPECIFICITY,
    DM_RUNG_DIRECTIONAL_GAP,
    DM_RUNG_INVERSION,
]

DM_FEATURE_RUNG_COLS_FULL = [
    DM_RUNG_XY,
    DM_RUNG_YZ,
    DM_RUNG_ZY,
    DM_RUNG_YX,
    DM_RUNG_YZ_PRIMARY,
    DM_RUNG_ZY_RETURN,
    DM_RUNG_YZZY_ENERGY,
    DM_RUNG_COMPARISON_ENERGY,
    DM_RUNG_DIRECTIONAL_SPECIFICITY,
    DM_RUNG_DIRECTIONAL_GAP,
    DM_RUNG_INVERSION,
    DM_RUNG_PI_PHASE,
    DM_RUNG_PI_COS2,
    DM_RUNG_PI_SIN2,
]

# Summary indices that remain on the raw bounded D_M evidence scale.
# Excludes n_rungs, positive fractions, pi-fit score, tracking correlations,
# phase velocity, and phase span because those are normalized diagnostics.
DM_FEATURE_SUMMARY_COLS_BOUNDED_CORE = [
    1,   # yz_mean
    3,   # zy_mean
    5,   # yzzy_energy_mean
    6,   # yzzy_energy_max
    7,   # specificity_mean
    8,   # specificity_max
    15,  # projection_score: raw bounded energy/specificity/YZ-derived score
]


def dm_feature_from_result(result: DMGpuResult, feature_mode: str = "bounded_core") -> np.ndarray:
    mode = str(feature_mode).strip().lower()
    if mode == "bounded_core":
        rung = np.asarray(result.rung_stats[:, DM_FEATURE_RUNG_COLS_BOUNDED_CORE], dtype=np.float32).reshape(-1)
        summary = np.asarray(result.summary[DM_FEATURE_SUMMARY_COLS_BOUNDED_CORE], dtype=np.float32).reshape(-1)
    elif mode == "full":
        rung = np.asarray(result.rung_stats[:, DM_FEATURE_RUNG_COLS_FULL], dtype=np.float32).reshape(-1)
        summary = np.asarray(result.summary, dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f"unknown --dm-feature-mode: {feature_mode}")

    feat = np.concatenate([rung, summary], axis=0).astype(np.float32, copy=False)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def parse_dm_methods(methods: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for raw in str(methods).split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("D_M method specs must be like geo:offset_on or qproj:base_only")
        substrate, condition = [x.strip() for x in item.split(":", 1)]
        if substrate not in {"geo", "qproj", "gproj"}:
            raise ValueError(f"unknown D_M substrate in --dm-methods: {substrate}")
        if condition not in {"null", "base_only", "offset_on"}:
            raise ValueError(f"unknown D_M condition in --dm-methods: {condition}")
        out.append((substrate, condition))
    if not out:
        raise ValueError("No D_M methods selected.")
    return out


def method_label(substrate: str, condition: str) -> str:
    return f"{substrate}_{condition}"


def compute_dm_features_for_texts(
    cp: Any,
    module: Any,
    texts: Sequence[str],
    method: Tuple[str, str],
    bases_by_key: Dict[Tuple[str, str], DMRecordBase],
    args: argparse.Namespace,
    min_bits: int,
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    substrate, condition = method
    label = method_label(substrate, condition)
    features: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    total_gpu_ms = 0.0
    total_wall = 0.0
    total_items = 0

    for i, text in enumerate(texts):
        bits = make_text_bit_bank([text], min_bits=min_bits)
        bit_bank_gpu = cp.asarray(bits, dtype=cp.uint8)
        cp.cuda.Stream.null.synchronize()

        if substrate == "geo":
            result = run_geo_gpu(cp, module, bit_bank_gpu, condition, args)
        else:
            base = bases_by_key[(substrate, condition)]
            result = run_record_base_gpu(cp, module, bit_bank_gpu, base, args)

        feat = dm_feature_from_result(result, args.dm_feature_mode)
        features.append(feat)
        total_gpu_ms += float(result.gpu_ms)
        total_wall += float(result.wall_seconds)
        total_items += int(result.work_items)
        s = summary_dict(result.summary)
        rows.append({
            "method": label,
            "substrate": substrate,
            "condition": condition,
            "text_index": i,
            "projection_score": s.get("projection_score", 0.0),
            "energy_mean": s.get("yzzy_energy_mean", 0.0),
            "specificity_mean": s.get("specificity_mean", 0.0),
            "yz_mean": s.get("yz_mean", 0.0),
            "zy_mean": s.get("zy_mean", 0.0),
            "pi_score": s.get("pi_periodic_score", 0.0),
            "gpu_ms": result.gpu_ms,
            "feature_dim": int(feat.size),
        })

    mat = np.vstack(features).astype(np.float32, copy=False)
    timing = {
        "method": label,
        "family": "D_M",
        "substrate": substrate,
        "condition": condition,
        "gpu_ms": total_gpu_ms,
        "wall_seconds": total_wall,
        "work_items": total_items,
        "items_per_gpu_second": float(total_items) / max(total_gpu_ms / 1000.0, EPS),
        "feature_dim": int(mat.shape[1]),
        "texts_encoded": int(mat.shape[0]),
    }
    return mat, timing, rows


def torch_dtype_from_arg(torch: Any, dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"unknown GPT-2 dtype: {dtype_name}")


def reduce_qk_logits_to_text_features_gpu(torch: Any, logits: Any, center_mode: str, head_ids: Sequence[int]) -> Any:
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
        return torch.zeros((batch, len(head_ids) * 4), device=L.device, dtype=torch.float32)

    j_cmp = (J + 1) % seq_len
    j_cmp = torch.where(j_cmp == I, (j_cmp + 1) % seq_len, j_cmp)

    i_cmp = (I - 1) % seq_len
    i_cmp = torch.where(i_cmp == J, (i_cmp - 1) % seq_len, i_cmp)

    yz = L[:, :, I, J].mean(dim=2)
    zy = L[:, :, J, I].mean(dim=2)
    xy = L[:, :, I, j_cmp].mean(dim=2)
    yx = L[:, :, J, i_cmp].mean(dim=2)
    vec = torch.stack([xy, yz, zy, yx], dim=-1)  # [B,H,4]
    hidx = torch.tensor(list(head_ids), device=L.device, dtype=torch.long)
    vec = vec.index_select(dim=1, index=hidx).contiguous().view(batch, -1)
    return torch.nan_to_num(vec.float(), nan=0.0, posinf=0.0, neginf=0.0)


def compute_gpt2_feature_matrices_gpu(
    torch: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    texts: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    dtype = torch_dtype_from_arg(torch, args.gpt2_dtype)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    try:
        torch.set_float32_matmul_precision(args.matmul_precision)
    except Exception:
        pass

    feature_modes = [x.strip() for x in str(args.gpt2_feature_modes).split(",") if x.strip()]
    if not feature_modes:
        feature_modes = ["qk", "hidden"]
    for m in feature_modes:
        if m not in {"qk", "hidden"}:
            raise ValueError(f"unknown GPT-2 feature mode: {m}")

    center_modes = [x.strip() for x in str(args.center_modes).split(",") if x.strip()] or ["row"]

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

    hidden_chunks: List[Any] = []
    qk_chunks: List[Any] = []
    total_gpu_ms = 0.0
    total_qk_products = 0
    batch_size = max(1, int(args.gpt2_batch_size))
    n_batches = int(math.ceil(len(texts) / batch_size))

    for batch_index in range(n_batches):
        lo = batch_index * batch_size
        hi = min(len(texts), lo + batch_size)
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
        else:
            attention_mask = torch.ones_like(input_ids, device="cuda")

        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                out = model.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            hidden_states = out.hidden_states
            batch = int(input_ids.shape[0])
            seq_len = int(input_ids.shape[1])

            if "hidden" in feature_modes:
                final_hidden = hidden_states[-1].float()
                mask = attention_mask.float().unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                pooled = (final_hidden * mask).sum(dim=1) / denom
                hidden_chunks.append(torch.nan_to_num(pooled.float(), nan=0.0, posinf=0.0, neginf=0.0).detach())

            if "qk" in feature_modes:
                per_batch_qk_parts: List[Any] = []
                for layer in layer_ids:
                    block = model.transformer.h[layer]
                    x = hidden_states[layer]
                    with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                        qkv = block.attn.c_attn(x)
                        q, k, _v = qkv.split(n_embd, dim=2)
                        q = q.view(batch, seq_len, n_head, head_dim).permute(0, 2, 1, 3).contiguous()
                        k = k.view(batch, seq_len, n_head, head_dim).permute(0, 2, 1, 3).contiguous()
                        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(head_dim))
                    total_qk_products += int(batch * n_head * seq_len * seq_len)
                    for center_mode in center_modes:
                        per_batch_qk_parts.append(reduce_qk_logits_to_text_features_gpu(torch, logits, center_mode, head_ids))
                qk_feat = torch.cat(per_batch_qk_parts, dim=1) if per_batch_qk_parts else torch.empty((batch, 0), device="cuda")
                qk_chunks.append(torch.nan_to_num(qk_feat.float(), nan=0.0, posinf=0.0, neginf=0.0).detach())

        ev1.record()
        ev1.synchronize()
        total_gpu_ms += float(ev0.elapsed_time(ev1))

        del out, hidden_states, input_ids, attention_mask
        torch.cuda.empty_cache()

    matrices: Dict[str, np.ndarray] = {}
    timings: List[Dict[str, Any]] = []

    if "hidden" in feature_modes:
        hidden_gpu = torch.cat(hidden_chunks, dim=0).contiguous()
        matrices["gpt2_hidden"] = hidden_gpu.detach().float().cpu().numpy().astype(np.float32, copy=False)
        timings.append({
            "family": "GPT-2",
            "method": "gpt2_hidden",
            "gpu_ms": total_gpu_ms,
            "work_items": int(len(texts) * int(args.max_length) * n_embd),
            "items_per_gpu_second": float(len(texts) * int(args.max_length) * n_embd) / max(total_gpu_ms / 1000.0, EPS),
            "feature_dim": int(matrices["gpt2_hidden"].shape[1]),
            "texts_encoded": int(matrices["gpt2_hidden"].shape[0]),
            "model_load_seconds": model_load_seconds,
            "dtype": args.gpt2_dtype,
            "note": "Mean-pooled final hidden state from GPT-2 transformer on torch CUDA.",
        })

    if "qk" in feature_modes:
        qk_gpu = torch.cat(qk_chunks, dim=0).contiguous()
        matrices["gpt2_qk"] = qk_gpu.detach().float().cpu().numpy().astype(np.float32, copy=False)
        timings.append({
            "family": "GPT-2",
            "method": "gpt2_qk",
            "gpu_ms": total_gpu_ms,
            "work_items": int(total_qk_products),
            "items_per_gpu_second": float(total_qk_products) / max(total_gpu_ms / 1000.0, EPS),
            "feature_dim": int(matrices["gpt2_qk"].shape[1]),
            "texts_encoded": int(matrices["gpt2_qk"].shape[0]),
            "model_load_seconds": model_load_seconds,
            "dtype": args.gpt2_dtype,
            "note": "Raw pre-softmax QK structural features from selected layers/heads on torch CUDA.",
        })

    return matrices, timings


# =============================================================================
# GPU RETRIEVAL SCORING
# =============================================================================

def split_query_candidate_features(mat: np.ndarray, n_queries: int) -> Tuple[np.ndarray, np.ndarray]:
    if mat.shape[0] < n_queries:
        raise ValueError("Feature matrix has fewer rows than queries.")
    return mat[:n_queries], mat[n_queries:]


def gpu_similarity(cp: Any, q: np.ndarray, c: np.ndarray, mode: str) -> Tuple[np.ndarray, float]:
    """
    GPU retrieval scoring.

    D_M rule:
        D_M signatures are already bounded by the qproj/gproj/geo base.
        Cosine-normalizing D_M removes amplitude-bearing operator evidence.

    Modes:
        cosine      : unit-vector cosine similarity, useful for GPT-style embeddings.
        raw_dot     : unnormalized q @ c.T, preserves amplitude but favors high-energy candidates.
        raw_neg_l2  : -sum((q-c)^2), preserves amplitude and rewards raw bounded closeness.
        raw_neg_mse : -mean((q-c)^2), same ranking as raw_neg_l2 but easier to read.
    """
    score_mode = str(mode).strip().lower()
    q_gpu = cp.asarray(np.nan_to_num(q.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0))
    c_gpu = cp.asarray(np.nan_to_num(c.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0))
    cp.cuda.Stream.null.synchronize()
    ev0 = cp.cuda.Event()
    ev1 = cp.cuda.Event()
    ev0.record()

    if score_mode == "cosine":
        q_norm = cp.linalg.norm(q_gpu, axis=1, keepdims=True)
        c_norm = cp.linalg.norm(c_gpu, axis=1, keepdims=True)
        qn = q_gpu / cp.maximum(q_norm, np.float32(EPS))
        cn = c_gpu / cp.maximum(c_norm, np.float32(EPS))
        sim = qn @ cn.T
    elif score_mode == "raw_dot":
        sim = q_gpu @ c_gpu.T
    elif score_mode in {"raw_neg_l2", "raw_neg_mse"}:
        q2 = cp.sum(q_gpu * q_gpu, axis=1, keepdims=True)
        c2 = cp.sum(c_gpu * c_gpu, axis=1, keepdims=True).T
        dist2 = cp.maximum(q2 + c2 - 2.0 * (q_gpu @ c_gpu.T), np.float32(0.0))
        if score_mode == "raw_neg_mse":
            dist2 = dist2 / np.float32(max(1, int(q_gpu.shape[1])))
        sim = -dist2
    else:
        raise ValueError(f"unknown score mode: {mode}")

    sim = cp.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    ev1.record()
    ev1.synchronize()
    score_ms = float(cp.cuda.get_elapsed_time(ev0, ev1))
    return cp.asnumpy(sim), score_ms


def evaluate_similarity_matrix(
    method: str,
    family: str,
    sim: np.ndarray,
    true_indices: Sequence[int],
    candidate_rows: Sequence[Dict[str, Any]],
    top_k: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n_queries = int(sim.shape[0])
    top_k = max(1, int(top_k))
    ranks: List[int] = []
    margins: List[float] = []
    per_query: List[Dict[str, Any]] = []

    for qi in range(n_queries):
        scores = np.asarray(sim[qi], dtype=np.float64)
        true_idx = int(true_indices[qi])
        order = np.argsort(-scores, kind="stable")
        rank = int(np.where(order == true_idx)[0][0]) + 1
        ranks.append(rank)
        true_score = float(scores[true_idx])
        decoy_scores = np.delete(scores, true_idx)
        best_decoy = float(np.max(decoy_scores)) if decoy_scores.size else 0.0
        margins.append(true_score - best_decoy)
        best_idx = int(order[0])
        best_row = candidate_rows[best_idx]
        per_query.append({
            "method": method,
            "family": family,
            "query_index": qi,
            "true_candidate_index": true_idx,
            "rank": rank,
            "top1_correct": int(rank == 1),
            f"top{top_k}_correct": int(rank <= top_k),
            "true_score": true_score,
            "best_candidate_index": best_idx,
            "best_candidate_kind": best_row.get("candidate_kind", ""),
            "best_candidate_source_query_index": best_row.get("source_query_index", ""),
            "best_score": float(scores[best_idx]),
            "best_decoy_score": best_decoy,
            "margin_true_minus_best_decoy": true_score - best_decoy,
        })

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    margins_arr = np.asarray(margins, dtype=np.float64)
    summary = {
        "method": method,
        "family": family,
        "queries": n_queries,
        "candidates": int(sim.shape[1]),
        "top1": float(np.mean(ranks_arr <= 1.0)),
        f"top{top_k}": float(np.mean(ranks_arr <= float(top_k))),
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "mean_rank": float(np.mean(ranks_arr)),
        "median_rank": float(np.median(ranks_arr)),
        "mean_margin": float(np.mean(margins_arr)),
        "median_margin": float(np.median(margins_arr)),
        "min_margin": float(np.min(margins_arr)),
    }
    return summary, per_query


def print_retrieval_leaderboard(rows: Sequence[Dict[str, Any]], top_k: int) -> None:
    print()
    print("[RETRIEVAL UTILITY LEADERBOARD]")
    print("  " + "-" * 120)
    hdr = f"  {'method':22s} {'family':8s} {'score_mode':>12s} {'top1':>8s} {('top' + str(top_k)):>8s} {'MRR':>8s} {'mean_rank':>10s} {'margin':>10s} {'encode_ms':>11s} {'score_ms':>10s} {'dim':>6s}"
    print(hdr)
    for r in sorted(rows, key=lambda x: (float(x.get('top1', 0.0)), float(x.get('mrr', 0.0)), float(x.get('mean_margin', 0.0))), reverse=True):
        print(
            f"  {str(r['method']):22s} {str(r['family']):8s} {str(r.get('score_mode', '')):>12s} "
            f"{100.0 * float(r['top1']):7.2f}% "
            f"{100.0 * float(r.get('top' + str(top_k), 0.0)):7.2f}% "
            f"{float(r['mrr']):8.4f} "
            f"{float(r['mean_rank']):10.2f} "
            f"{float(r['mean_margin']):+10.5f} "
            f"{float(r.get('encode_gpu_ms', 0.0)):11.3f} "
            f"{float(r.get('score_gpu_ms', 0.0)):10.3f} "
            f"{int(r.get('feature_dim', 0)):6d}"
        )


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPU-only D_M task utility retrieval benchmark against GPT-2 representations.")

    p.add_argument("--qproj-null", type=Path, default=DEFAULT_QPROJ_NULL)
    p.add_argument("--qproj-base", type=Path, default=DEFAULT_QPROJ_BASE)
    p.add_argument("--qproj-offset", type=Path, default=DEFAULT_QPROJ_OFFSET)
    p.add_argument("--gproj-null", type=Path, default=DEFAULT_GPROJ_NULL)
    p.add_argument("--gproj-base", type=Path, default=DEFAULT_GPROJ_BASE)
    p.add_argument("--gproj-offset", type=Path, default=DEFAULT_GPROJ_OFFSET)

    p.add_argument("--dm-methods", default="geo:offset_on,qproj:base_only,qproj:offset_on,gproj:base_only,gproj:offset_on,geo:null,qproj:null,gproj:null")
    p.add_argument("--dm-feature-mode", default="bounded_core", choices=["bounded_core", "full"],
                   help="D_M feature vector. bounded_core preserves raw base-bounded amplitude and removes normalized diagnostic washout fields.")
    p.add_argument("--dm-score-mode", default="raw_neg_mse", choices=["raw_neg_mse", "raw_neg_l2", "raw_dot", "cosine"],
                   help="D_M retrieval score. Default preserves raw bounded closeness; cosine is diagnostic only.")
    p.add_argument("--gpt-score-mode", default="cosine", choices=["cosine", "raw_dot", "raw_neg_l2", "raw_neg_mse"],
                   help="GPT-2 retrieval score. Cosine remains the default for neural representations.")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--gpt2-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--layers", default="", help="Comma-separated GPT-2 layer ids. Default: all.")
    p.add_argument("--heads", default="", help="Comma-separated GPT-2 head ids. Default: all.")
    p.add_argument("--center-modes", default="row", help="Comma-separated GPT-2 centering modes: none,row,global,zscore.")
    p.add_argument("--gpt2-feature-modes", default="qk,hidden", help="Comma-separated: qk,hidden")
    p.add_argument("--gpt2-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--matmul-precision", default="high", choices=["highest", "high", "medium"])

    p.add_argument("--text", default="")
    p.add_argument("--text-file", default="")
    p.add_argument("--split-lines", action="store_true")
    p.add_argument("--max-texts", type=int, default=0)
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--positive-kind", default="exact", choices=["exact", "whitespace", "light_noise", "prefix_suffix"])
    p.add_argument("--decoy-kinds", default="shuffle,light_noise,rotate")
    p.add_argument("--noise-frac", type=float, default=0.02)
    p.add_argument("--top-k", type=int, default=5)

    p.add_argument("--data-stride", type=int, default=17, help="Stride through raw text bit bank for D_M data signal.")
    p.add_argument("--delay-scale", type=int, default=1, help="Convert D_M dt to raw bit offset by delay_bits=dt//scale.")
    p.add_argument("--min-bit-multiplier", type=int, default=8, help="Minimum bit bank multiplier over largest base records.")
    p.add_argument("--min-bits", type=int, default=131072, help="Minimum per-text D_M bit bank size.")

    p.add_argument("--dm-linear-threads", type=int, default=512)
    p.add_argument("--dm-tile-threads", type=int, default=1024, help="Power-of-two CUDA threads per tile correlator block.")

    p.add_argument("--geo-shots", type=int, default=4096)
    p.add_argument("--geo-offset-dt", type=int, default=DEFAULT_OFFSET_DT)
    p.add_argument("--geo-base-energy", type=float, default=0.04)
    p.add_argument("--geo-energy-gain", type=float, default=0.28)
    p.add_argument("--geo-comparison-scale", type=float, default=0.010)
    p.add_argument("--geo-offset-deform", type=float, default=0.18)
    p.add_argument("--geo-phase-scale", type=float, default=0.37)

    p.add_argument("--out-dir", type=Path, default=None)

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
    torch, AutoModelForCausalLM, AutoTokenizer = require_torch_cuda(False)

    tag = now_tag()
    out_dir = args.out_dir or (ANALYSIS_DIR / f"dm_probe_22_task_utility_retrieval_gpu_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]

    print()
    print("=" * 112)
    print("  D_M TASK UTILITY RETRIEVAL BENCHMARK — GPU ONLY")
    print("=" * 112)
    print(f"  D_M dir       : {D_M_DIR}")
    print(f"  Output dir    : {out_dir}")
    print(f"  CUDA device   : {name}")
    print("  Rule          : GPT-2 is comparison only; D_M gets raw text bits")
    print("  CPU fallback  : removed / forbidden")

    print()
    print("[COMPILE] CUDA D_M kernels")
    module = compile_cuda_module(cp)
    module.get_function("dm_tile_correlator_kernel_u8")
    print("  compiled      : yes")

    print()
    print("[LOAD] input texts")
    texts = load_texts(args)
    print(f"  loaded texts  : {len(texts)}")

    print()
    print("[BUILD] retrieval task")
    task = build_retrieval_task(texts, args)
    all_task_texts = list(task.query_texts) + list(task.candidate_texts)
    n_queries = len(task.query_texts)
    n_candidates = len(task.candidate_texts)
    print(f"  queries       : {n_queries}")
    print(f"  candidates    : {n_candidates}")
    print(f"  positive kind : {args.positive_kind}")
    print(f"  decoys/source : {args.decoy_kinds}")

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
    bases_by_key: Dict[Tuple[str, str], DMRecordBase] = {}
    record_bases: List[DMRecordBase] = []
    for substrate, condition, path in base_specs:
        b = load_record_base(path, substrate, condition)
        bases_by_key[(substrate, condition)] = b
        record_bases.append(b)
        print(f"  {substrate:5s} {condition:9s} tiles={b.pair.shape[0]:3d} shots={b.pair.shape[1]:6d} rungs={b.meta['n_rungs']:3d} {path}")

    max_records = max(int(np.prod(b.pair.shape[:2])) for b in record_bases)
    max_delay = max(int(np.max(np.abs(b.tile_total_delay))) for b in record_bases)
    min_bits = max(int(args.min_bits), int(args.min_bit_multiplier) * max_records + max_delay + 1024)
    print()
    print("[D_M] per-text GPU signatures")
    print(f"  texts encoded : {len(all_task_texts)}")
    print(f"  min bits/text : {min_bits:,}")

    method_specs = parse_dm_methods(args.dm_methods)
    feature_matrices: Dict[str, np.ndarray] = {}
    timing_rows: List[Dict[str, Any]] = []
    dm_signature_rows: List[Dict[str, Any]] = []

    for method in method_specs:
        label = method_label(*method)
        print(f"  encoding {label:18s} ...")
        mat, timing, rows = compute_dm_features_for_texts(cp, module, all_task_texts, method, bases_by_key, args, min_bits)
        feature_matrices[label] = mat
        timing_rows.append(timing)
        dm_signature_rows.extend(rows)
        print(f"    done dim={mat.shape[1]} gpu_ms={timing['gpu_ms']:.3f}")

    print()
    print("[GPT-2] per-text GPU signatures")
    assert torch is not None and AutoModelForCausalLM is not None and AutoTokenizer is not None
    gpt_mats, gpt_timings = compute_gpt2_feature_matrices_gpu(torch, AutoModelForCausalLM, AutoTokenizer, all_task_texts, args)
    for k, v in gpt_mats.items():
        feature_matrices[k] = v
    timing_rows.extend(gpt_timings)
    for t in gpt_timings:
        print(f"  {t['method']:12s} dim={t['feature_dim']} gpu_ms={float(t['gpu_ms']):.3f}")

    print()
    print("[SCORE] GPU retrieval")
    summary_rows: List[Dict[str, Any]] = []
    per_query_rows: List[Dict[str, Any]] = []

    timing_by_method = {str(r["method"]): r for r in timing_rows}

    # Warm up CuPy/cuBLAS scoring paths so the first reported method does not
    # absorb library initialization latency. This is still GPU-only; it is not a
    # CPU fallback or CPU scoring path.
    warm_q = np.zeros((2, 4), dtype=np.float32)
    warm_c = np.zeros((3, 4), dtype=np.float32)
    for warm_mode in sorted({str(args.dm_score_mode), str(args.gpt_score_mode), "cosine", "raw_neg_mse"}):
        _warm_sim, _warm_ms = gpu_similarity(cp, warm_q, warm_c, warm_mode)
    cp.cuda.Stream.null.synchronize()

    for method, mat in feature_matrices.items():
        q_feat, c_feat = split_query_candidate_features(mat, n_queries)
        family = "GPT-2" if method.startswith("gpt2") else "D_M"
        score_mode = str(args.gpt_score_mode if family == "GPT-2" else args.dm_score_mode)
        sim, score_ms = gpu_similarity(cp, q_feat, c_feat, score_mode)
        summary, perq = evaluate_similarity_matrix(
            method=method,
            family=family,
            sim=sim,
            true_indices=task.true_candidate_indices,
            candidate_rows=task.candidate_rows,
            top_k=int(args.top_k),
        )
        timing = timing_by_method.get(method, {})
        summary["encode_gpu_ms"] = float(timing.get("gpu_ms", 0.0))
        summary["score_gpu_ms"] = float(score_ms)
        summary["total_gpu_ms"] = float(summary["encode_gpu_ms"] + score_ms)
        summary["feature_dim"] = int(mat.shape[1])
        summary["work_items"] = int(timing.get("work_items", 0))
        summary["score_mode"] = score_mode
        summary_rows.append(summary)
        for row in perq:
            row["score_gpu_ms_for_method"] = score_ms
            row["score_mode"] = score_mode
            per_query_rows.append(row)
        print(
            f"  {method:22s} top1={100*summary['top1']:6.2f}% "
            f"top{args.top_k}={100*summary['top' + str(args.top_k)]:6.2f}% "
            f"MRR={summary['mrr']:.4f} margin={summary['mean_margin']:+.5f} "
            f"mode={score_mode} score_ms={score_ms:.3f}"
        )

    print_retrieval_leaderboard(summary_rows, int(args.top_k))

    print()
    print("[WRITE] CSV / JSON outputs")
    topk_name = "top" + str(int(args.top_k))
    write_csv(
        out_dir / "retrieval_summary.csv",
        sorted(summary_rows, key=lambda r: (float(r["top1"]), float(r["mrr"]), float(r["mean_margin"])), reverse=True),
        [
            "method", "family", "queries", "candidates", "top1", topk_name, "mrr", "mean_rank", "median_rank",
            "mean_margin", "median_margin", "min_margin", "encode_gpu_ms", "score_gpu_ms", "total_gpu_ms", "score_mode", "feature_dim", "work_items",
        ],
    )
    write_csv(
        out_dir / "retrieval_per_query.csv",
        per_query_rows,
        [
            "method", "family", "query_index", "true_candidate_index", "rank", "top1_correct", topk_name + "_correct",
            "true_score", "best_candidate_index", "best_candidate_kind", "best_candidate_source_query_index", "best_score",
            "best_decoy_score", "margin_true_minus_best_decoy", "score_mode", "score_gpu_ms_for_method",
        ],
    )
    write_csv(
        out_dir / "retrieval_candidates.csv",
        task.candidate_rows,
        ["candidate_index", "source_query_index", "candidate_kind", "is_positive", "text_preview"],
    )
    write_csv(
        out_dir / "dm_signature_stats.csv",
        dm_signature_rows,
        [
            "method", "substrate", "condition", "text_index", "projection_score", "energy_mean", "specificity_mean",
            "yz_mean", "zy_mean", "pi_score", "gpu_ms", "feature_dim",
        ],
    )
    write_csv(
        out_dir / "benchmark_timers.csv",
        timing_rows,
        [
            "family", "method", "substrate", "condition", "gpu_ms", "wall_seconds", "work_items", "items_per_gpu_second",
            "feature_dim", "texts_encoded", "model_load_seconds", "dtype", "note",
        ],
    )
    write_json(
        out_dir / "probe_config.json",
        {
            "script": "d_m_task_utility_retrieval_gpu_benchmark_rawsignal.py",
            "timestamp": tag,
            "gpu_only": True,
            "cpu_fallback": False,
            "d_m_dir": D_M_DIR,
            "output_dir": out_dir,
            "retrieval_task": {
                "queries": n_queries,
                "candidates": n_candidates,
                "positive_kind": args.positive_kind,
                "decoy_kinds": args.decoy_kinds,
                "noise_frac": float(args.noise_frac),
                "top_k": int(args.top_k),
            },
            "dm": {
                "methods": args.dm_methods,
                "feature_mode": args.dm_feature_mode,
                "score_mode": args.dm_score_mode,
                "min_bits_per_text": int(min_bits),
                "data_stride": int(args.data_stride),
                "delay_scale": int(args.delay_scale),
                "geo_shots": int(args.geo_shots),
                "geo_offset_dt": int(args.geo_offset_dt),
                "geo_base_energy": float(args.geo_base_energy),
                "geo_energy_gain": float(args.geo_energy_gain),
                "geo_comparison_scale": float(args.geo_comparison_scale),
                "geo_offset_deform": float(args.geo_offset_deform),
                "geo_phase_scale": float(args.geo_phase_scale),
            },
            "gpt2": {
                "model": args.model,
                "score_mode": args.gpt_score_mode,
                "feature_modes": args.gpt2_feature_modes,
                "layers": args.layers,
                "heads": args.heads,
                "center_modes": args.center_modes,
                "dtype": args.gpt2_dtype,
                "allow_tf32": bool(args.allow_tf32),
                "matmul_precision": args.matmul_precision,
                "batch_size": int(args.gpt2_batch_size),
                "max_length": int(args.max_length),
            },
            "record_bases": [b.meta for b in record_bases],
        },
    )

    print(f"  wrote: {out_dir / 'retrieval_summary.csv'}")
    print(f"  wrote: {out_dir / 'retrieval_per_query.csv'}")
    print(f"  wrote: {out_dir / 'retrieval_candidates.csv'}")
    print(f"  wrote: {out_dir / 'dm_signature_stats.csv'}")
    print(f"  wrote: {out_dir / 'benchmark_timers.csv'}")
    print(f"  wrote: {out_dir / 'probe_config.json'}")
    print()
    print("[DONE] GPU-only D_M / GPT-2 task utility retrieval benchmark complete.")


if __name__ == "__main__":
    main()
