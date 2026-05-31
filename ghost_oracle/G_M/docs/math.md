# Math

Derivations of the four operators the suite touches — $T_1$, $T_2$, $T_3$, and $G_M$ — plus the projection-channel identity that turns physical shot counts into operator estimates, and the substrate-comparison framing the current benchmark verifies.

`G_M` is the **Generalized Metric**, formerly called the Ghost Metric.

The architecture document explains how this math gets compiled into a kernel. The probes record how it was discovered. This document is the math standing alone.

GitHub renders LaTeX inside `$...$` and `$$...$$` delimiters. Everything here is written as raw Markdown and should be pasted as source, not copied from a rendered preview.

Conventions used throughout:

- $a,b \in [0,\pi]$ are rotation angles on single qubits, after the `data_to_angles` scaling.
- $\alpha = \mathtt{ALPHA\_NORM} = 0.9127$ is the suite-wide normalization constant.
- A **tile** is one execution of the seven-qubit per-tile circuit at a fixed $(a,b)$.
- A **base** is a `.npz` file of per-shot measurements for many tiles at fixed angles, sampled $N_{\mathrm{shots}}=4096$ times each by default.
- A **substrate** is one faithful implementation of the projection circuit:
  - analytical closed form,
  - classical noiseless GPU projection base,
  - real QPU shot-count projection base from IBM Runtime.

---

## The four operators

Naming follows the rest of the suite. The numbers attached to $T_1$, $T_2$, and $T_3$ are chronological in the trajectory, not algebraic.

$$
T_1(a,b)=\lvert \cos(a-b) \rvert
$$

Standard rank-1 cosine kernel. This is what a lifted trigonometric dot product computes when fed the representation $[\cos\theta \mid \sin\theta]$, using:

$$
\cos(a-b)=\cos a \cos b+\sin a \sin b
$$

This was the first obvious comparison target, but it is not what the seven-qubit QPU circuit was computing.

---

$$
T_2(a,b)=\left|\cos\!\left(\frac{a-b}{2}\right)\right|
$$

Half-angle Hadamard form. This is the textbook expectation for a Hadamard/swap-style test on product states:

$$
|\psi_a\rangle = R_y(a)|0\rangle,
\qquad
|\psi_b\rangle = R_y(b)|0\rangle
$$

The original framing of the project assumed the QPU circuit was computing this. The probes falsified that assumption.

---

$$
T_3(a,b)
=
\frac{1}{2}
\left(
1
+
\cos^2\!\left(\frac{a}{2}\right)
\cos^2\!\left(\frac{b}{2}\right)
+
\sin^2\!\left(\frac{a}{2}\right)
\sin^2\!\left(\frac{b}{2}\right)
\right)
=
\frac{3}{4}
+
\frac{1}{4}\cos a \cos b
$$

This is the actual target the seven-qubit per-tile circuit implements in the noiseless limit, in $P(\mathrm{ctrl}=0)$-space.

It appears because the ghost CNOTs turn the intended single-qubit product-state test into a GHZ-correlated mixed-state target.

---

$$
\boxed{
G_M(a,b)
=
\frac{1}{\alpha}
\sqrt{
\frac{1+\cos a \cos b}{2}
}
}
$$

This is the operator the current suite is built on.

$T_3$ is the circuit probability-space form. $G_M$ is the normalized matrix-entry form used by the benchmark.

The current framing is:

$$
G_M
=
\text{bounded projection-channel / geometry-channel generalized similarity operator}
$$

Three structural properties matter.

### Bounded output

Since:

$$
\cos a \cos b \in [-1,1]
$$

then:

$$
\frac{1+\cos a \cos b}{2} \in [0,1]
$$

and therefore:

$$
\sqrt{\frac{1+\cos a \cos b}{2}} \in [0,1]
$$

The normalization by $\alpha$ maps the expected operating range into the suite's score range. Production code clamps where appropriate.

This boundedness is the core difference from dot-product attention. No single dimension can become unbounded inside $G_M$.

### Low-rank-like, but not just rank-1

The argument:

$$
1+\cos a \cos b
$$

is a constant plus a rank-1 outer-product structure in cosine-space. The square root bends that structure. So $G_M$ behaves like a low-rank generalized similarity, but not a plain dot product.

That square-root curvature is part of the operator's useful behavior.

### Not a normal PSD kernel

$G_M$ should not be treated as a standard Mercer / RKHS kernel. It is better understood as an indefinite generalized metric / projection-style similarity operator.

That is why the suite treats it as a calibrated scoring operator rather than as a drop-in kernel replacement.

---

## Why $T_2$ is wrong: the seven-qubit circuit

The per-tile circuit operates on seven qubits:

$$
\{a_1,v_1,a_2,\mathrm{ctrl},b_1,v_2,b_2\}
$$

The circuit structure is:

1. Apply $R_y(a)$ on $v_1$ and $R_y(b)$ on $v_2$.
2. Apply ghost CNOTs:
   $$
   \mathrm{CNOT}(v_1 \to a_1),
   \qquad
   \mathrm{CNOT}(v_1 \to a_2),
   $$
   $$
   \mathrm{CNOT}(v_2 \to b_1),
   \qquad
   \mathrm{CNOT}(v_2 \to b_2).
   $$
3. Apply $H$ on $\mathrm{ctrl}$.
4. Apply $\mathrm{CSWAP}(\mathrm{ctrl};v_1,v_2)$.
5. Apply $H$ on $\mathrm{ctrl}$.
6. Measure $\{\mathrm{ctrl},a_1,a_2,b_1,b_2\}$.

The textbook swap-test analysis assumes the swap operates on product states:

$$
|\psi_a\rangle \otimes |\psi_b\rangle
$$

In that case:

$$
P(\mathrm{ctrl}=0)
=
\frac{1+\left|\langle \psi_a|\psi_b\rangle\right|^2}{2}
$$

and the matrix-entry conversion:

$$
\sqrt{2P_0-1}
$$

recovers:

$$
\left|\langle \psi_a|\psi_b\rangle\right|
=
\left|\cos\!\left(\frac{a-b}{2}\right)\right|
=
T_2(a,b)
$$

But the ghost CNOTs break the product-state assumption.

After the ghost CNOTs, each side becomes a GHZ-correlated block:

$$
|v_1 a_1 a_2\rangle
=
\cos\!\left(\frac{a}{2}\right)|000\rangle
+
\sin\!\left(\frac{a}{2}\right)|111\rangle
$$

$$
|v_2 b_1 b_2\rangle
=
\cos\!\left(\frac{b}{2}\right)|000\rangle
+
\sin\!\left(\frac{b}{2}\right)|111\rangle
$$

The full six-qubit state before the control test is:

$$
|\Psi\rangle
=
\left(c_a|000\rangle+s_a|111\rangle\right)_{v_1a_1a_2}
\otimes
\left(c_b|000\rangle+s_b|111\rangle\right)_{v_2b_1b_2}
$$

where:

$$
c_a=\cos\!\left(\frac{a}{2}\right),
\qquad
s_a=\sin\!\left(\frac{a}{2}\right)
$$

and likewise for $b$.

The swap acts on $(v_1,v_2)$ inside an entangled GHZ context, not on isolated product states. That is the entire reason $T_2$ was the wrong target.

---

## Deriving $T_3$ from the ghost-CNOT circuit

The swap expectation conditional on a basis state of $(v_1,v_2)$ is:

$$
\langle \mathrm{SWAP}\rangle_{v_1v_2}
=
\begin{cases}
1, & v_1=v_2,\\
0, & v_1\ne v_2.
\end{cases}
$$

From the GHZ product state, the four basis configurations of $(v_1,v_2)$ occur with probabilities:

$$
c_a^2c_b^2,
\qquad
c_a^2s_b^2,
\qquad
s_a^2c_b^2,
\qquad
s_a^2s_b^2
$$

The equal states $|00\rangle$ and $|11\rangle$ contribute to the swap expectation. The cross states $|01\rangle$ and $|10\rangle$ do not.

Therefore:

$$
\langle \mathrm{SWAP}\rangle
=
c_a^2c_b^2+s_a^2s_b^2
$$

or:

$$
\langle \mathrm{SWAP}\rangle
=
\cos^2\!\left(\frac{a}{2}\right)
\cos^2\!\left(\frac{b}{2}\right)
+
\sin^2\!\left(\frac{a}{2}\right)
\sin^2\!\left(\frac{b}{2}\right)
$$

The Hadamard test then gives:

$$
P(\mathrm{ctrl}=0)
=
\frac{1+\langle \mathrm{SWAP}\rangle}{2}
$$

so:

$$
T_3(a,b)
=
\frac{1}{2}
\left(
1
+
\cos^2\!\left(\frac{a}{2}\right)
\cos^2\!\left(\frac{b}{2}\right)
+
\sin^2\!\left(\frac{a}{2}\right)
\sin^2\!\left(\frac{b}{2}\right)
\right)
$$

This is the noiseless probability-space target of the seven-qubit circuit.

---

## Simplifying $T_3$

Use the half-angle identities:

$$
\cos^2\!\left(\frac{x}{2}\right)=\frac{1+\cos x}{2}
$$

$$
\sin^2\!\left(\frac{x}{2}\right)=\frac{1-\cos x}{2}
$$

Then:

$$
\cos^2\!\left(\frac{a}{2}\right)
\cos^2\!\left(\frac{b}{2}\right)
+
\sin^2\!\left(\frac{a}{2}\right)
\sin^2\!\left(\frac{b}{2}\right)
$$

becomes:

$$
\frac{(1+\cos a)(1+\cos b)}{4}
+
\frac{(1-\cos a)(1-\cos b)}{4}
$$

Expand:

$$
\frac{
1+\cos a+\cos b+\cos a\cos b
+
1-\cos a-\cos b+\cos a\cos b
}{4}
$$

Cancel terms:

$$
\frac{2+2\cos a\cos b}{4}
=
\frac{1+\cos a\cos b}{2}
$$

So:

$$
T_3(a,b)
=
\frac{1}{2}
\left(
1+\frac{1+\cos a\cos b}{2}
\right)
$$

and therefore:

$$
T_3(a,b)
=
\frac{3}{4}
+
\frac{1}{4}\cos a\cos b
$$

This is the compact form.

---

## From $T_3$ to $G_M$

The suite uses the swap-test-style matrix-entry conversion:

$$
\sqrt{2P_0-1}
$$

Apply it to $T_3$:

$$
2T_3(a,b)-1
=
2\left(
\frac{3}{4}
+
\frac{1}{4}\cos a\cos b
\right)
-1
$$

$$
=
\frac{3}{2}
+
\frac{1}{2}\cos a\cos b
-
1
$$

$$
=
\frac{1+\cos a\cos b}{2}
$$

Then normalize by $\alpha$:

$$
G_M(a,b)
=
\frac{1}{\alpha}
\sqrt{2T_3(a,b)-1}
$$

so:

$$
G_M(a,b)
=
\frac{1}{\alpha}
\sqrt{
\frac{1+\cos a\cos b}{2}
}
$$

That is the Generalized Metric.

The identity between forms is:

$$
G_M^2(a,b)
=
\frac{2T_3(a,b)-1}{\alpha^2}
$$

and equivalently:

$$
T_3(a,b)
=
\frac{1}{2}
\left(
1+\alpha^2G_M^2(a,b)
\right)
$$

The product-to-sum view is also useful:

$$
\cos a\cos b
=
\frac{1}{2}\cos(a-b)
+
\frac{1}{2}\cos(a+b)
$$

Therefore:

$$
G_M(a,b)
=
\frac{1}{\alpha}
\sqrt{
\frac{1}{2}
+
\frac{1}{4}\cos(a-b)
+
\frac{1}{4}\cos(a+b)
}
$$

So $G_M$ contains a similarity-like $(a-b)$ coupling, an $(a+b)$ coupling, a constant term, and a square-root curvature. This is why it is related to cosine structure without being cosine similarity.

---

## The projection-channel identity

The projection channel estimates $G_M(a,b)$ at a new angle pair $(a,b)$ from physical or simulated shot counts collected at a fixed base angle pair $(a_o,b_o)$.

The identity behind it is importance reweighting.

The shot-level random variables are:

$$
f_a=\frac{a_1+a_2}{2}
$$

$$
f_b=\frac{b_1+b_2}{2}
$$

with:

$$
f_a,f_b \in \{0,0.5,1\}
$$

and the control bit:

$$
c \in \{0,1\}
$$

In the noiseless GHZ limit:

$$
a_1=a_2,
\qquad
b_1=b_2
$$

so $f_a,f_b\in\{0,1\}$. The middle bucket $0.5$ appears when hardware noise, decoherence, or measurement effects open the GHZ block. This is useful: the projection channel can absorb imperfect hardware structure without pretending it is noiseless.

Define:

$$
p_a=\sin^2\!\left(\frac{a}{2}\right)
$$

$$
p_b=\sin^2\!\left(\frac{b}{2}\right)
$$

The log-likelihood of a firing pair $(f_a,f_b)$ at angles $(a,b)$ is:

$$
\log L(f_a,f_b \mid a,b)
=
f_a\log p_a
+
(1-f_a)\log(1-p_a)
+
f_b\log p_b
+
(1-f_b)\log(1-p_b)
$$

The importance weight from base angles $(a_o,b_o)$ to target angles $(a,b)$ is:

$$
w(f_a,f_b)
=
\exp\left(
\log L(f_a,f_b \mid a,b)
-
\log L(f_a,f_b \mid a_o,b_o)
\right)
$$

The reweighted estimate of $P(\mathrm{ctrl}=0)$ is:

$$
\hat P_0(a,b)
=
\frac{
\sum_s
w(f_a^{(s)},f_b^{(s)})
\mathbf{1}\!\left[c^{(s)}=0\right]
}{
\sum_s
w(f_a^{(s)},f_b^{(s)})
}
$$

where $s$ indexes shots from the base.

The projection-channel estimate of $G_M$ is then:

$$
\hat G_M(a,b)
=
\frac{1}{\alpha}
\sqrt{
\max\left(
0,
2\hat P_0(a,b)-1
\right)
}
$$

The $\max(0,\cdot)$ term protects against shot-noise excursions where the reweighted estimate dips below $1/2$.

This identity is substrate-agnostic. It applies to:

- analytical / reference distributions,
- noiseless GPU-generated projection bases,
- real QPU shot-count bases.

The arithmetic is the same. Only the bucket-count noise changes.

---

## Bucket compression

The projection channel does not need shot ordering. It only needs counts over:

$$
(f_a,f_b,c)
\in
\{0,0.5,1\}^2
\times
\{0,1\}
$$

That gives:

$$
3\times 3\times 2 = 18
$$

integer buckets per tile.

So the per-shot list compresses into an 18-int histogram. From that point forward, every projection consumer reads 18 integers per tile rather than $N_{\mathrm{shots}}\times 5$ raw measured bits.

Let $n_c(i,j)$ be the count of shots with:

$$
(f_a,f_b,\mathrm{ctrl})=(i,j,c)
$$

Then the reweighted probability becomes:

$$
\hat P_0(a,b)
=
\frac{
\sum_{i,j\in\{0,0.5,1\}}
w(i,j)n_0(i,j)
}{
\sum_{i,j\in\{0,0.5,1\}}
w(i,j)
\left(
n_0(i,j)+n_1(i,j)
\right)
}
$$

This is the constant-size projection identity used by the CUDA path.

---

## Log-space evaluation

The projection weights are evaluated in log-space for fp32 stability.

The log-weight decomposes into base and slope terms:

$$
\log w(f_a,f_b)
=
B_a+f_aS_a+B_b+f_bS_b
$$

where:

$$
B_a
=
\log(1-p_a)
-
\log(1-p_{a_o})
$$

and:

$$
S_a
=
\log p_a
-
\log p_{a_o}
-
B_a
$$

with analogous definitions for $B_b$ and $S_b$.

The kernel clips probabilities before logarithms:

$$
p \leftarrow \mathrm{clip}(p,\varepsilon,1-\varepsilon)
$$

with:

$$
\varepsilon=0.05
$$

and clips assembled log-weights:

$$
\log w \leftarrow \mathrm{clip}(\log w,-C,C)
$$

with:

$$
C=3.0
$$

This keeps the projection estimate stable in fp32 while preserving the calibrated bucket structure.

---

## Phase lift

To use $G_M$ on real-valued embeddings, each component $x\in\mathbb{R}$ must be mapped to an angle $\theta\in[0,\pi]$.

The production runtime uses a bounded saturating phase lift:

$$
\theta(x)
=
\frac{\pi}{2}
\left(
1+\tanh\!\left(\frac{x}{3}\right)
\right)
$$

This lift has three useful properties.

### 1. Bounded

$$
\theta(x)\in[0,\pi]
$$

so:

$$
\cos\theta(x)\in[-1,1]
$$

and each $G_M$ term stays bounded.

### 2. Smooth around the origin

For $|x|\ll 1$:

$$
\tanh\!\left(\frac{x}{3}\right)\approx\frac{x}{3}
$$

so:

$$
\theta(x)
\approx
\frac{\pi}{2}
+
\frac{\pi x}{6}
$$

Small component changes therefore produce small angle changes.

### 3. Saturating at fixed endpoints

As $x\to-\infty$:

$$
\theta(x)\to 0,
\qquad
\cos\theta(x)\to 1
$$

As $x\to+\infty$:

$$
\theta(x)\to\pi,
\qquad
\cos\theta(x)\to -1
$$

Large spikes therefore produce fixed endpoint behavior instead of unbounded score behavior.

The earlier affine lift:

$$
\theta=\frac{\pi}{2}(1+x)
$$

was useful for early probes but wrong for unbounded inputs, because large $|x|$ wraps cosine around its period. The saturating lift fixes that.

---

## Per-dimension generalized similarity

For embeddings $X_Q,X_K\in\mathbb{R}^d$, the geometry-channel similarity is:

$$
\mathrm{sim}(X_Q,X_K)
=
\frac{1}{d\alpha}
\sum_{k=1}^{d}
\sqrt{
\frac{
1+
\cos\theta\!\left(X_Q^{(k)}\right)
\cos\theta\!\left(X_K^{(k)}\right)
}{2}
}
$$

Each term is bounded. Therefore a single dimension contributes at most:

$$
\frac{1}{d\alpha}
$$

to the aggregate score.

This is the key outlier-resistance mechanism.

In dot-product attention, one large shared coordinate can dominate:

$$
Q\cdot K
$$

and softmax can amplify that domination.

In $G_M$, a large coordinate saturates before aggregation. It can influence the score, but it cannot explode.

This is the architectural payoff:

$$
\text{saturating phase lift}
+
\text{bounded per-dimension generalized metric}
+
\text{mean aggregation}
$$

gives a similarity path whose worst-case single-dimension influence is bounded by construction.

---

## Geometry channel and projection channel

The current benchmark treats $G_M$ through two linked channels.

### Geometry channel

The geometry channel evaluates the closed-form expression directly:

$$
G_M^{\mathrm{geom}}(a,b)
=
\frac{1}{\alpha}
\sqrt{
\frac{1+\cos a\cos b}{2}
}
$$

This is the clean analytical reference.

### Projection channel

The projection channel estimates the same generalized metric from calibrated bucket counts:

$$
\hat G_M^{\mathrm{proj}}(a,b)
=
\frac{1}{\alpha}
\sqrt{
\max\left(0,2\hat P_0(a,b)-1\right)
}
$$

This is the substrate-backed estimate.

The projection channel can be driven by:

- real QPU shot buckets,
- noiseless GPU-generated shot buckets,
- analytical/reference buckets.

---

## Agreement metric

The tied-channel architecture reports the mean absolute difference between projection and geometry:

$$
A_i
=
\frac{1}{M}
\sum_{j=1}^{M}
\left|
\hat G_M^{\mathrm{proj}}(Q_i,K_j)
-
G_M^{\mathrm{geom}}(Q_i,K_j)
\right|
$$

where $i$ indexes a query and $j$ indexes keys.

The benchmark then reports the average agreement over queries.

This is not a speed metric. It is a substrate-quality metric.

It asks:

> How far does this projection substrate drift from the closed-form geometry channel on the same task?

The latest capstone run reports example agreement values:

| Base type | Observed agreement range |
|---|---:|
| QPU bases | 0.0049–0.0888 |
| GPU bases | 0.0104–0.1049 |

These values should not be interpreted as a simple "QPU is always noisier" or "GPU is always cleaner" rule. Agreement depends on the selected tile, mask, threshold, shot bucket structure, and task distribution.

The correct read is:

$$
\text{agreement}
=
\text{projection-vs-geometry substrate comparison}
$$

not:

$$
\text{agreement}
=
\text{quantum advantage}
$$

and not:

$$
\text{agreement}
=
\text{dense attention throughput}
$$

---

## Current capstone benchmark

The current canonical runner is:

```bash
python g_m_benchmark.py
```

Full benchmark modes:

```bash
python g_m_benchmark.py --sweep ALL
python g_m_benchmark.py --probe
python g_m_benchmark.py --skip-verify --sweep ALL --probe
```

This runner combines the older five-way verification path and the older Auto Oracle semantic retrieval path into one capstone script.

Legacy examples remain useful for continuity:

```bash
python examples/final_benchmark_5way.py
python examples/auto_oracle.py
python examples/projection_benchmark.py
```

but the current benchmark claim should be taken from `g_m_benchmark.py`.

The current capstone has three major stages:

1. five-way verification,
2. semantic scale sweep,
3. negative control probes.

---

## Calibration in the current benchmark

The current benchmark uses two different calibration objectives because the verification and semantic retrieval tasks measure different things.

### Verification calibration

The five-way verification path uses:

```text
calibrate_candidates()
```

It scores candidate $(\mathrm{tile},\mathrm{mask},\mathrm{threshold})$ choices using the projection megakernel and the Flash-Squelch certificate:

$$
\mathrm{clean}
:=
\mathrm{median\ spike\ fraction}
\le
\mathtt{SPIKE\_TOLERANCE}
$$

Then it ranks:

```text
clean first, then signal fraction descending
```

This produces a calibrated projection component for the coherent same-dimension attack benchmark.

### Sweep / probe calibration

The semantic sweep and probe path uses:

```text
calibrate_recall1()
```

It ranks $(\mathrm{tile},\mathrm{mask})$ components by the same argmax recall@1 retrieval objective used by the sweep.

This prevents a mismatch where a component looks clean under the Flash-Squelch certificate but fails as a semantic retriever.

A base whose best component does not clear the recall@1 floor is excluded from sweep/probe evaluation rather than silently included.

The current floor is:

$$
\mathtt{CALIB\_MIN\_R1}=0.50
$$

---

## Five-way verification math

The five-way benchmark compares:

| Path | Meaning |
|---|---|
| `CUBLAS` | Standard dot-product attention control on raw embeddings. |
| `TIED` | Dual-channel geometry + projection kernel with agreement reporting. |
| `GEO` | Closed-form $G_M$ geometry channel. |
| `QPROJ` | Projection channel driven by real QPU shot-count buckets. |
| `GPROJ` | Projection channel driven by noiseless GPU shot-count buckets. |

Current verification configuration:

```text
N = 4096
d = 64
jitter = 0.3
attack fraction = 0.05
attack magnitude = 200.0
squelch power = 256.0
base files = 10
```

The attack is a coherent same-dimension spike. It is designed to expose the difference between unbounded dot-product scoring and bounded per-dimension generalized metric scoring.

Current five-way summary:

| Path | Top-1 | Signal | Spike fraction | Time |
|---|---:|---:|---:|---:|
| `CUBLAS` | 57.9% | 57.9% | 0.426528 | 0.74 ms |
| `GEO` mean across bases | 100.0% | — | — | — |
| `QPROJ` mean across QPU bases | 100.0% | 100.0% | 0.0498 | ~22–26 ms per base |
| `GPROJ` mean across GPU bases | 99.7% | 99.6% | 0.0502 | ~22–30 ms per base |

The mathematical read is:

$$
\text{dot-product attention is vulnerable to the coherent shared-dimension spike}
$$

because one large dimension can dominate the dot product.

Meanwhile, $G_M$ preserves retrieval because each per-dimension contribution is bounded before aggregation.

The benchmark does not claim QPU projection is faster than cuBLAS. cuBLAS is much faster as a dense GEMM primitive.

The benchmark claim is narrower:

$$
\text{bounded generalized metric scoring resists this coherent same-dimension attack}
$$

and:

$$
\text{the projection channel reproduces this behavior from calibrated shot-count substrates}
$$

---

## Semantic scale sweep

The semantic sweep uses clustered semantic embeddings with coherent outliers and evaluates Recall@1, Recall@5, Recall@10, and MRR.

Current sweep settings:

```text
d = 1024
TOP_K = [1, 5, 10]
strategy = mixed projection component(s)
```

One GPU base was excluded from sweep/probe evaluation because no component cleared the 50% recall@1 calibration floor.

### SMALL sweep

| Backend | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| cosine | 99.12% | 99.12% | 99.12% | 0.991 |
| `GEO` | 100.00% | 100.00% | 100.00% | 1.000 |
| QPU projection bases | 99.90–100.00% | 100.00% | 100.00% | ~1.000 |
| usable GPU projection bases | 97.27–100.00% | 99.61–100.00% | 99.90–100.00% | 0.983–1.000 |

### MEDIUM sweep

| Backend | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| cosine | 96.88% | 96.88% | 96.88% | 0.969 |
| `GEO` | 100.00% | 100.00% | 100.00% | 1.000 |
| QPU projection bases | 99.41–100.00% | 99.80–100.00% | 100.00% | 0.995–1.000 |
| usable GPU projection bases | 40.33–100.00% | 58.98–100.00% | 68.16–100.00% | 0.485–1.000 |

### LARGE sweep

| Backend | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| cosine | 95.31% | 95.31% | 95.31% | 0.953 |
| `GEO` | 100.00% | 100.00% | 100.00% | 1.000 |
| QPU projection bases | 91.99–100.00% | 95.41–100.00% | 96.78–100.00% | 0.935–1.000 |
| usable GPU projection bases | 1.46–100.00% | 2.73–100.00% | 3.22–100.00% | 0.019–1.000 |

The geometry channel is the mathematical ceiling in this sweep. Projection performance depends on calibrated bucket structure, selected tile/mask components, and distribution match.

This sweep should be read as a substrate/component quality diagnostic, not as a universal guarantee.

---

## Probe A: load-bearing shot-count control

Probe A asks whether the projection path is actually using calibrated shot structure.

It reuses each calibrated projection component and changes only the count structure.

| Count condition | Meaning |
|---|---|
| real calibrated counts | Actual calibrated bucket structure. |
| permuted counts | Same total count mass, bucket structure destroyed. |
| uniformized counts | Live buckets forced uniform. |

Current representative result:

| Base | Component | Real R@1 | Permuted R@1 | Uniform R@1 |
|---|---:|---:|---:|---:|
| `job_d8c4q5r8ch0s738uaq30.npz` | `t1/M4` | 99.41% | 0.00% | 0.00% |
| `job_d8c4qjr8ch0s738uaqk0.npz` | `t1/M4` | 100.00% | 0.00% | 0.00% |
| `job_d8c4qmr8amns73bj0b0g.npz` | `t1/M4` | 100.00% | 0.00% | 0.00% |
| `job_d8dod1i4gq0s73aqj3m0.npz` | `t0/M1` | 100.00% | 0.00% | 0.00% |
| `job_d8e4fmpvjngc73ansgug.npz` | `t2/M5` | 100.00% | 0.00% | 0.00% |
| usable GPU base | `t6/M6` | 100.00% | 0.00% | 0.00% |
| usable GPU base | `t12/M1` | 100.00% | 0.00% | 0.00% |
| usable GPU base | `t5/M6` | 40.33% | 0.00% | 0.00% |
| usable GPU base | `t0/M1` | 96.78% | 0.00% | 0.00% |

The important inequalities are:

$$
\text{real counts} \gg \text{permuted counts}
$$

and:

$$
\text{real counts} \gg \text{uniformized counts}
$$

This means the calibrated bucket structure is load-bearing.

The projection path is not silently collapsing to the geometry channel.

---

## Probe B: separation spectrum

Probe B asks whether the task is too easy.

It compares cosine, $G_M^{\mathrm{geom}}$, and $\hat G_M^{\mathrm{proj}}$ across controlled attack and noise axes.

Selected base/component:

```text
base = job_d8c4q5r8ch0s738uaq30.npz
component = t1/M4
d = 1024
```

### B1: outlier magnitude varies, noise fixed

Noise is fixed at $0.10$.

| Outlier magnitude | Noise | Cosine | GEO | PROJ |
|---:|---:|---:|---:|---:|
| 0.0 | 0.10 | 100.00% | 100.00% | 100.00% |
| 5.0 | 0.10 | 99.61% | 100.00% | 99.32% |
| 20.0 | 0.10 | 94.63% | 100.00% | 99.32% |
| 40.0 | 0.10 | 94.63% | 100.00% | 99.32% |
| 60.0 | 0.10 | 94.63% | 100.00% | 99.32% |
| 100.0 | 0.10 | 94.63% | 100.00% | 99.32% |

At outlier magnitude $0$, all methods are competitive. This checks that $G_M$ is not winning only because the attack was artificially added.

As coherent outlier magnitude rises, cosine drops while geometry and calibrated projection hold.

### B2: noise varies, outlier magnitude fixed

Outlier magnitude is fixed at $60.0$.

| Noise | Outlier magnitude | Cosine | GEO | PROJ |
|---:|---:|---:|---:|---:|
| 0.05 | 60.0 | 94.63% | 100.00% | 99.51% |
| 0.10 | 60.0 | 94.63% | 100.00% | 99.32% |
| 0.15 | 60.0 | 94.63% | 100.00% | 98.14% |
| 0.20 | 60.0 | 94.63% | 100.00% | 95.02% |
| 0.25 | 60.0 | 94.63% | 100.00% | 88.18% |
| 0.30 | 60.0 | 94.63% | 100.00% | 74.22% |

The projection component was recall@1-calibrated near $\mathrm{noise}=0.10$.

So degradation as noise rises is expected. That is distribution shift against the calibrated projection estimator, not evidence that the base is unused.

The geometry channel stays at the closed-form ceiling because it does not depend on bucket counts.

---

## Bucket-mask calibration

The projection identity assumes all nine $(f_a,f_b)$ buckets contribute:

$$
(f_a,f_b)\in\{0,0.5,1\}^2
$$

In practice, the benchmark allows masks that zero selected buckets before reweighting.

This is not a modification to $G_M$.

It is calibration of the projection-channel estimator $\hat G_M^{\mathrm{proj}}$.

The mask controls which bucket subset gives the cleanest projection estimate on a given base and task.

The current mask set includes:

```text
M1: Baseline all buckets
M2: Drop buckets 4-8
M3: Anti-pillars
M4: Drop (0,1)(1,0)
M5: Drop (1,2)(2,1)
M6: Drop pillars
M7: Pure core
M8: Mirror core
```

The latest benchmark separates mask selection by task:

- verification calibration selects `(tile, mask, threshold)` by Flash-Squelch certificate;
- semantic sweep/probe calibration selects `(tile, mask)` by recall@1.

This distinction matters.

A mask that is good for attack squelching may not be good for semantic retrieval. The current runner handles that by calibrating against the metric it actually reports.

---

## What the current math supports

The current mathematical and benchmark-supported claim is that $G_M$ is a bounded generalized similarity operator with both:

1. a closed-form geometry channel,
2. a projection channel reconstructible from calibrated shot-count buckets.

It can be expressed across:

1. analytical closed form,
2. noiseless GPU projection bases,
3. real QPU shot-count bases.

The current evidence supports:

- the seven-qubit circuit does not compute the originally assumed $T_2$ product-state target;
- the noiseless circuit target is $T_3$;
- the normalized matrix-entry form is $G_M$;
- $G_M$ is bounded and therefore structurally resists single-dimension coherent outlier domination;
- calibrated projection bases can reproduce retrieval behavior from shot-count structure;
- destroying the bucket structure destroys projection retrieval;
- agreement measures projection-vs-geometry substrate quality;
- cuBLAS remains the correct dense GEMM throughput baseline.

The current evidence does **not** support:

- a universal replacement claim for dot-product attention;
- a QPU speedup claim over cuBLAS;
- a quantum advantage claim;
- a claim that one mask is universal;
- a claim that projection is distribution-shift-proof without recalibration.

The correct bounded framing is:

$$
G_M
=
\text{bounded, calibrated, substrate-comparable generalized metric}
$$

---

## Pointers

- **Probe 4** — Direct statevector derivation of $T_3$ and the seven-qubit circuit.
- **Probe 6** — Three-target table: $T_1$, $T_2$, $T_3$.
- **Probe 7** — Physical GHZ parity test on QPU shots.
- **Probe 9** — Identity $T_3 \leftrightarrow G_M$ and classification of $G_M$.
- **Probe 10.1** — Per-dimension aggregation and same-dimension coherent attack benchmark.
- **Probe 11–11.2** — Projection-channel range and float32 noise-floor investigations.
- **Probes 13–18** — Bucket-mask ablations across tiles and calibration runs.
- **Probe 12 corrected** — Substrate-equivalence verification with corrected classical sampler.
- **Probe 20** — Auto-calibrating production kernel with per-base mask/threshold selection.
- **`g_m_benchmark.py`** — Current canonical benchmark runner.
- **`g_m_gpu_generate.py`** — Noiseless GPU projection-base generator.
- **`g_m_qpu_generate.py`** — QPU projection-base generator.
- **`docs/architecture.md`** — How the math compiles into kernels.
- **`docs/known_issues.md`** — Where the math meets imperfect software.

---

## Final read

The process is the process.

The project started with the wrong target, falsified it, found the circuit's actual target, derived the normalized generalized metric, built the projection identity, corrected the classical sampler, added calibration, added controls, and now reports bounded claims.

The math says:

$$
T_2 \text{ was the wrong target}
$$

$$
T_3 \text{ is the circuit probability-space target}
$$

$$
G_M \text{ is the normalized generalized metric}
$$

The benchmark says:

$$
\text{geometry retrieves}
$$

$$
\text{calibrated projection retrieves}
$$

$$
\text{destroying bucket structure destroys projection}
$$

$$
\text{cuBLAS remains the dense throughput control}
$$

That is the claim to defend.
