# F_M Math

Mathematical definitions, derivations, and benchmark framing for the **F_M** operator family in the Ghost Oracle Suite.

This file intentionally avoids GitHub/KaTeX equation rendering. All formulas are written in fenced plain-text blocks so they cannot produce LaTeX parse errors.

`F_M` is the paired-path differential wave operator discovered from the F_M QPU circuit and then implemented across three substrate paths:

- **QPROJ** — real QPU shot record.
- **GPROJ** — GPU-generated paired-path record with the same schema.
- **GEO** — optimized classical analytic path.

The process record explains how the operator was discovered. The architecture document explains how the math compiles into CUDA kernels. This document is the math standing alone.

---

## Conventions

A **tile** is one paired-path circuit instance with fixed metadata:

```text
tile = (theta, d_t, s, m)
```

where:

- `theta` is the tile angle parameter.
- `d_t` is the delay step.
- `s` is the scale level.
- `m` is the mode label.

The first frozen F_M QPU base used:

```text
d_t = [0, 1, 2, 4, 8, 16, 0]
s   = [1, 1, 1, 1, 1, 1, 2]
```

Each tile is sampled for:

```text
N_shots = 4096
```

shots by default.

Each shot records two paired paths:

```text
g[t, n, b]
e[t, n, b]
```

where:

- `t` indexes tile.
- `n` indexes shot.
- `b` indexes bit.
- `g` is the first path.
- `e` is the second path, called `em` in code.

The code names are:

```text
g
em
delta
xor_delta
```

The mathematical notation used here is:

```text
g := g
e := em
```

---

## The paired-path fields

The two raw paths are not the final operator. The discovered F_M signal lives in the differential fields.

### Signed differential field

```text
Delta[t, n, b] = e[t, n, b] - g[t, n, b]
```

Since `e` and `g` are binary:

```text
e, g in {0, 1}

Delta[t, n, b] in {-1, 0, 1}
```

This is the signed path difference.

In code:

```text
delta = em - g
```

### Binary differential field

```text
X[t, n, b] = e[t, n, b] XOR g[t, n, b]
```

Since `e` and `g` are binary:

```text
X[t, n, b] in {0, 1}
```

This is the unsigned disagreement field.

In code:

```text
xor_delta = em XOR g
```

### Primary discovered field

The final locked F_M signature is carried primarily by:

```text
X = xor_delta
```

with response:

```text
bit_diff
```

ordered by:

```text
delay d_t
```

So the primary operator signature is:

```text
xor_delta / bit_diff / delay
```

---

## Response curves

The CUDA projector does not operate directly on every raw shot for the final metric. It first compresses each tile into response curves.

For any field:

```text
Y[t, n, b]
```

where `Y` may be `g`, `em`, `delta`, or `xor_delta`, define the following tile responses.

### Mean response

```text
R_mean(t)
=
(1 / (N_shots * B))
*
sum over n=1..N_shots
sum over b=1..B
Y[t, n, b]
```

where `B` is the number of bits.

### Energy response

```text
R_energy(t)
=
(1 / (N_shots * B))
*
sum over n=1..N_shots
sum over b=1..B
Y[t, n, b]^2
```

For binary `X`, this equals the mean.

For signed `Delta`, this measures nonzero disagreement energy.

### Transition response

Flatten the tile record into a one-dimensional sequence:

```text
y[1], y[2], ..., y[L]
```

Then:

```text
R_transition(t)
=
(1 / (L - 1))
*
sum over i=1..L-1 of indicator(y[i+1] != y[i])
```

This measures local alternation in the tile record.

### Imbalance response

For signed fields:

```text
R_imbalance(t)
=
(N_pos(Y) - N_neg(Y)) / (N_shots * B)
```

where:

```text
N_pos(Y) = number of positive entries
N_neg(Y) = number of negative entries
```

For binary fields, this reduces to a positive-fraction style statistic.

### Bit-specific responses

For bit `b = 0`:

```text
R_bit0(t)
=
(1 / N_shots)
*
sum over n=1..N_shots
Y[t, n, 0]
```

For bit `b = 1`:

```text
R_bit1(t)
=
(1 / N_shots)
*
sum over n=1..N_shots
Y[t, n, 1]
```

### Bit-difference response

The primary F_M response is:

```text
R_bitdiff(t) = R_bit1(t) - R_bit0(t)
```

For the locked field `X = xor_delta`:

```text
R_FM(t)
=
(1 / N_shots) * sum over n of X[t, n, 1]
-
(1 / N_shots) * sum over n of X[t, n, 0]
```

This is the core qproj/gproj response curve used by the final benchmark.

---

## Delay ordering

The F_M wave signature is not just a scalar score. It is an ordered response over tile delay.

Let:

```text
d_t = delay metadata for tile t
```

The delay-ordered response curve is:

```text
C[i] = R_FM(t_i)
```

where the tiles are sorted so:

```text
d_{t_1} <= d_{t_2} <= ... <= d_{t_N}
```

with tile index used as a tie-breaker.

For the first locked base:

```text
d_t = [0, 1, 2, 4, 8, 16, 0]
```

so the delay-ordered curve is the tile response sorted by this ladder.

The primary qproj signature is therefore:

```text
C_primary_qproj(d) = R_bitdiff^X(d)

where:

X = e XOR g
```

---

## Wave metrics

The CUDA projector computes wave metrics from each ordered response curve.

Let:

```text
C = [C[0], C[1], ..., C[N-1]]
```

Define the demeaned curve:

```text
C_tilde[i] = C[i] - mean(C)
```

---

## Spectral peak ratio

Compute the real Fourier transform:

```text
C_hat[k]
=
sum over j=0..N-1 of
C_tilde[j] * exp(-2*pi*i*j*k/N)
```

Power is:

```text
P[k] = abs(C_hat[k])^2
```

Ignoring DC, the total non-DC power is:

```text
P_tot = sum over k=1..K of P[k]
```

The peak ratio is:

```text
rho_peak = max(P[k] for k >= 1) / (P_tot + epsilon)
```

This measures how concentrated the curve is around a dominant frequency.

---

## Spectral entropy

Define normalized non-DC spectral power:

```text
q[k] = P[k] / (P_tot + epsilon)
```

Then:

```text
H_spec
=
-(1 / log(K))
*
sum over k=1..K of q[k] * log(q[k])
```

where `K` is the number of non-DC frequency bins.

Low entropy means the spectral power is concentrated.

High entropy means power is diffuse.

---

## Sinusoid fit

The projector also fits a small sinusoidal model over a fixed frequency grid.

For normalized x-coordinate:

```text
x[i] in [0, 1]
```

the model is:

```text
C[i]
approximately =
a * sin(2*pi*f*x[i])
+
b * cos(2*pi*f*x[i])
+
c
```

For each candidate frequency `f`, the coefficients `(a, b, c)` are solved by least squares.

The amplitude is:

```text
A_f = sqrt(a^2 + b^2)
```

The phase is:

```text
phi_f = atan2(b, a)
```

The fit score is:

```text
R2_f
=
1
-
sum over i of (C[i] - C_hat[i])^2
/
(sum over i of (C[i] - mean(C))^2 + epsilon)
```

The best sinusoid fit is:

```text
R2_best = max over f of R2_f
```

with corresponding:

```text
f_best
A_best
phi_best
```

The implementation rejects ill-conditioned short-curve fits where:

```text
A_best > 10
```

or where `A` or `R2` is not finite.

This guard is necessary because early seven-point curves can produce unstable least-squares amplitudes on nearly singular design matrices.

---

## Low/high spectral ratio

The projector also computes a coarse low-frequency to high-frequency ratio:

```text
rho_low_high = P_low / (P_high + epsilon)
```

This is not the main discovery metric. It contributes to the composite wave score as a coarse shape term.

---

## Composite F_M wave score

The final wave score used by the projector is:

```text
S_wave
=
0.40 * rho_peak
+
0.25 * max(0, R2_best)
+
0.20 * (1 - H_spec)
+
0.15 * min(1, abs(rho_low_high) / 10)
```

This score is deliberately simple.

It rewards:

- concentrated spectral power,
- sinusoidal fit,
- low spectral entropy,
- coarse low/high structure.

The primary qproj value was:

```text
S_wave = 0.6571
```

for:

```text
xor_delta / bit_diff / delay
```

---

## Controls

F_M relies heavily on controls. The important question is not just whether the wave score is high. The question is what destroys it.

### Delay shuffle

Delay shuffle preserves the curve values but destroys their delay order.

If:

```text
C = [C[0], ..., C[N-1]]
```

then:

```text
C_shuffle = [C[pi(0)], ..., C[pi(N-1)]]
```

for random permutation `pi`.

A delay-dependent wave should weaken under delay shuffle.

### Delay reverse

Delay reverse tests whether the curve has order structure but not necessarily orientation-stable phase:

```text
C_reverse[i] = C[N - 1 - i]
```

### Phase scramble

Phase scramble preserves Fourier magnitudes but randomizes phase.

If:

```text
C_hat[k] = A[k] * exp(i * phi[k])
```

then phase scramble uses:

```text
C_hat_prime[k] = A[k] * exp(i * phi_prime[k])
```

with random `phi_prime[k]`.

This tests whether coherent phase matters.

### Circular shift

Circular shift preserves wave shape but moves phase:

```text
C_prime[i] = C[(i - s) mod N]
```

A pure phase-shifted wave should survive circular shift better than phase scramble.

### Tile shuffle

Tile shuffle destroys tile/address order.

### Path-pair break

This is the most important F_M control.

For each tile, independently shuffle shots in the two paths:

```text
g_prime[t, n, b] = g[t, pi_g(n), b]

e_prime[t, n, b] = e[t, pi_e(n), b]
```

Then recompute:

```text
Delta_prime = e_prime - g_prime

X_prime = e_prime XOR g_prime
```

If F_M depends on the paired relationship between `g` and `e`, path-pair breaking should reduce the signal.

In the qproj wave probe:

```text
xor_delta / bit_diff / delay
vs path_pair_break

effect = 0.3052
auc    = 1.000
z      = 24.40
```

This was the key evidence that the signal is carried by paired-path structure, not by independent path marginals.

---

## QPROJ path

The qproj path is the real QPU record.

It starts with measured arrays:

```text
g[t, n, b]
e[t, n, b]
```

from the QPU dump.

The CUDA projector computes:

```text
X[t, n, b] = e[t, n, b] XOR g[t, n, b]
```

then:

```text
R_bitdiff^X(t)
=
(1 / N_shots) * sum over n of X[t, n, 1]
-
(1 / N_shots) * sum over n of X[t, n, 0]
```

then delay-orders the tile response:

```text
C_qproj[i] = R_bitdiff^X(t_i)
```

and finally computes:

```text
S_wave(C_qproj)
```

The locked qproj primary signature:

```text
QPROJ  xor_delta / bit_diff / delay
score = 0.6571
peak  = 0.769
R2    = 0.819
freq  = 1.30
amp   = 0.05800
```

---

## GPROJ path

The gproj path is a GPU-generated base with the same analysis-facing schema as qproj.

It generates raw paired paths:

```text
g[t, n, b]
e[t, n, b]
```

so the same CUDA projector can consume it.

The generator samples a controllable binary differential field:

```text
X[t, n, b] ~ Bernoulli(p_X(t, b, n))
```

then sets:

```text
e[t, n, b] = g[t, n, b] XOR X[t, n, b]
```

The probability model contains a delay-wave term:

```text
p_X(t, b, n)
=
p_0
+
A_X
*
sin(
    2*pi*f*d_t/d_max
    +
    phi_0
    +
    phi_b
    +
    phi_s
    +
    phi_m
    +
    phi_n
)
```

where:

- `p_0` is the xor base probability.
- `A_X` is the xor wave amplitude.
- `f` is the wave frequency.
- `d_t / d_max` is normalized delay.
- `phi_b` is a bit phase.
- `phi_s` is a scale phase.
- `phi_m` is a mode phase.
- `phi_n` is a small shot-local phase.

The generated path is not meant to be a microscopic hardware simulation. It is meant to preserve the projector-relevant qproj signature family while keeping schema compatibility.

The locked gproj primary signature:

```text
GPROJ  xor_delta / bit_diff / delay
score = 0.6796
peak  = 0.772
R2    = 0.986
freq  = 0.90
amp   = 0.03837
```

---

## GEO path

The geo path is the optimized classical analytic form.

It does not generate shots.

It computes the F_M response curves directly from tile metadata.

Let:

```text
d_t = tile delay

d_max = max over t of abs(d_t)
```

Define normalized delay:

```text
u_t = d_t / d_max
```

Define the scale phase:

```text
psi_s(t) = lambda_s * log2(s_t + 1)
```

Define the theta phase:

```text
psi_theta(t) = lambda_theta * theta_t
```

Define an optional mode phase:

```text
psi_m(t)
```

Then the primary geo phase is:

```text
Phi_t
=
2*pi*f*u_t
+
phi_0
+
psi_s(t)
+
psi_theta(t)
+
psi_m(t)
```

The secondary transition phase is:

```text
Phi2_t
=
2*pi*(f + 1.1)*u_t
+
0.5*phi_0
+
psi_s(t)
+
0.5*psi_m(t)
```

### GEO primary curve

The optimized geo approximation for the primary curve is:

```text
C_geo_X_bitdiff(t)
=
b_X
+
A_bitdiff * sin(Phi_t)
```

where:

```text
X = xor_delta
```

### GEO bit1 curve

```text
C_geo_X_bit1(t)
=
0.11
+
A_bit1 * sin(Phi_t + 0.19)
```

### GEO transition curve

```text
C_geo_X_transition(t)
=
0.50
+
A_transition * sin(Phi2_t)
```

### GEO delta transition curve

```text
C_geo_Delta_transition(t)
=
0.50
+
A_transition * sin(Phi2_t + 0.06)
```

### GEO delta bitdiff curve

```text
C_geo_Delta_bitdiff(t)
=
b_Delta
+
0.78 * A_bitdiff * sin(Phi_t + 0.34)
```

The CUDA kernel computes eight fixed geo curves:

```text
0 = xor_delta / bit_diff
1 = xor_delta / bit1_mean
2 = xor_delta / transition
3 = xor_delta / energy
4 = delta     / bit_diff
5 = delta     / bit1_mean
6 = delta     / transition
7 = delta     / energy
```

The final optimized geo primary signature:

```text
GEO  xor_delta / bit_diff / delay
score = 0.7356
peak  = 0.812
R2    = 0.988
freq  = 1.10
amp   = 0.04813
```

---

## GEO parameter sweep

The CUDA geo sweep searches parameter candidates:

```text
Theta =
(
    f,
    phi_0,
    A_bitdiff,
    A_bit1,
    A_transition,
    A_energy,
    lambda_s,
    lambda_theta,
    b_X,
    b_Delta
)
```

For each candidate, it computes the eight geo curves and wave metrics.

The loss compares selected target rows against qproj-discovered values.

For each target row `r`:

```text
L_r
=
w_r
*
[
    2.00 * (S_r - S_r_target)^2
    +
    1.00 * (rho_r - rho_r_target)^2
    +
    0.60 * (R2_r - R2_r_target)^2
    +
    0.15 * (f_r - f_r_target)^2
    +
    0.50 * (A_r - A_r_target)^2
]
```

The total loss is:

```text
L = sum over r of L_r
```

The current target rows are:

```text
xor_delta / bit_diff / delay
xor_delta / bit1_mean / delay
xor_delta / transition / delay
delta     / transition / delay
```

The sweep is fast because each candidate evaluates only:

```text
8 curves x about 7 ordered points
```

on GPU.

Final measured sweep speed:

```text
250,000 candidates in 290.061 ms
```

or about:

```text
0.86 million candidates/sec
```

---

## Final benchmark comparison

The final benchmark compares:

| Path | Meaning |
|---|---|
| `QPROJ` | QPU record through response + wave metric kernels. |
| `GPROJ` | GPU-generated record through the same kernels. |
| `GEO` | Analytic metadata path through geo curve + wave metric kernels. |
| `FFT_GPU` | Adjacent spectral reader on the primary curve. |
| `DCT_GPU` | Adjacent DCT-style energy reader on the primary curve. |
| `AUTOCORR_GPU` | Adjacent autocorrelation reader on the primary curve. |
| `SINFIT_GPU` | Adjacent sinusoidal least-squares reader on the primary curve. |

The key distinction:

```text
F_M GEO = metadata -> operator curve -> wave metric
```

while:

```text
FFT/SinFit = already-built curve -> reader metric
```

The classical baselines are not substrate paths. They are adjacent curve readers.

---

## Final benchmark values

Primary substrate comparison:

| Substrate | Field / response / order | Score | Peak | R2 | Freq | Amp |
|---|---|---:|---:|---:|---:|---:|
| QPROJ | `xor_delta / bit_diff / delay` | 0.6571 | 0.769 | 0.819 | 1.30 | 0.05800 |
| GPROJ | `xor_delta / bit_diff / delay` | 0.6796 | 0.772 | 0.986 | 0.90 | 0.03837 |
| GEO | `xor_delta / bit_diff / delay` | 0.7356 | 0.812 | 0.988 | 1.10 | 0.04813 |

Speed comparison:

| Path | Operation | Time |
|---|---|---:|
| QPROJ | response + metric | 1.045625 ms |
| GPROJ | response + metric | 1.059980 ms |
| GEO | geo curve + metric | 0.382735 ms |
| GEO | 250k candidate sweep | 290.061 ms |

Approximate speedup:

```text
1.045625 / 0.382735 ~= 2.73
```

so GEO is about:

```text
2.7x faster than QPROJ projector evaluation
```

and:

```text
1.059980 / 0.382735 ~= 2.77
```

so GEO is about:

```text
2.8x faster than GPROJ projector evaluation
```

Classical adjacent readers on the primary curve:

| Baseline | Score | Time |
|---|---:|---:|
| FFT_GPU | 0.7688 | 0.459 ms |
| DCT_GPU | 0.6047 | 0.477 ms |
| AUTOCORR_GPU | 0.4519 | 1.010 ms |
| SINFIT_GPU | 0.8193 | 44.191 ms |

Interpretation:

- FFT is a strong and fast reader of the primary wave curve.
- SinFit scores highest because the curve is sinusoidal, but is much slower in this implementation.
- F_M GEO is the optimized operator path, not merely a reader.
- QPROJ discovers the signature.
- GPROJ reproduces the signature family.
- GEO computes the signature directly.

---

## What the current math supports

The current mathematical and benchmark-supported F_M claim is:

```text
F_M = paired-path differential wave operator
```

with primary signature:

```text
X = e XOR g
```

```text
R(t)
=
(1 / N) * sum over n of X[t, n, 1]
-
(1 / N) * sum over n of X[t, n, 0]
```

ordered by delay:

```text
d_t
```

and scored by the wave metric:

```text
S_wave
```

The evidence supports:

- the useful signal is differential, not raw-path marginal;
- `xor_delta / bit_diff / delay` is the locked primary signature;
- path-pair breaking weakens the signal;
- delay shuffling weakens the signal;
- the qproj signature can be reproduced by gproj;
- the same signature family can be computed directly by geo;
- the optimized geo path is faster than record-based qproj/gproj projector evaluation.

The evidence does not support:

- claiming the physical cavity model is proven as literal hardware gravity/electromagnetic dynamics;
- treating raw z-score magnitude as the main proof;
- treating FFT or SinFit as full replacements for F_M;
- claiming qproj/gproj/geo are numerically identical;
- claiming a universal frequency/amplitude across all future QPU jobs.

The correct bounded framing is:

```text
F_M = a substrate-linked paired-path differential wave operator
```

---

## Known mathematical cautions

### Seven-point wave curves are short

The first base has only seven tiles. This is enough to discover and benchmark the signature, but it is not enough for high-resolution frequency analysis.

Future larger delay ladders should improve:

- frequency stability,
- phase stability,
- sinusoid fit conditioning,
- spectral entropy estimates.

### GEO is an optimized analytic approximation

GEO is not a perfect reconstruction of qproj. It is the minimal classical operator path that captures the locked signature family.

### Controls matter more than raw z-score

Early probes produced large z-values when null variance was tiny. Later analysis therefore emphasizes:

- effect size,
- AUC/rank,
- control collapse,
- repeatability of signature ordering,
- speed.

### Path-pair structure is load-bearing

Any future F_M variant should include a path-pair destruction control.

If a proposed F_M variant does not weaken under path-pair break, it is probably not measuring the same operator.

---

## Pointers

- **`f_m_qpu_generate.py`** — QPU circuit submission / dump path for F_M.
- **`f_m_gpu_generate.py`** — GPU-compatible gproj base generator.
- **`kernels/fm_projector_kernel.cu`** — response, path-pair-break, wave metric, geo curve, and geo sweep kernels.
- **`probes/f_m_probe01_*`** — early field/family/control scan.
- **`probes/f_m_probe02_delta_address.py`** — tile/metadata address localization.
- **`probes/f_m_probe03_wave_nature.py`** — wave nature and control-collapse test.
- **`probes/f_m_probe04_qproj_kernel_finalizer.py`** — CUDA projector signature finalizer.
- **`probes/f_m_probe05_geo_numpy.py`** — NumPy geo formula discovery.
- **`probes/f_m_probe06_geo_cuda_finalizer.py`** — optimized CUDA geo finalizer.
- **`F_M_final_benchmark.py`** — final capstone benchmark.
- **`process_record.md`** — chronological discovery and engineering record.

---

## Final read

The F_M math says:

```text
two paired paths
-> differential field
-> delay-ordered response curve
-> wave score
```

The primary field is:

```text
X = e XOR g
```

The primary response is:

```text
R_bitdiff(t)
=
E[X[t, :, 1]]
-
E[X[t, :, 0]]
```

The primary ordering is:

```text
d_t
```

The primary signature is:

```text
xor_delta / bit_diff / delay
```

The benchmark says:

```text
QPROJ discovers it.
GPROJ reproduces it.
GEO computes it directly.
```

That is the F_M claim to defend.
