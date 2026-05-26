# Math

Derivations of the four operators the suite touches — $T_1$, $T_2$, $T_3$, and $G_M$ — plus the projection-channel identity that turns physical shot counts into operator estimates, and the substrate-comparison framing the final benchmark verifies. The architecture document explains how this math gets compiled into a kernel; the probes record how it was discovered. This is the math standing alone.

GitHub renders LaTeX inside `$...$` and `$$...$$` delimiters. Everything here is checkable on paper.

Conventions used throughout:

- $a, b \in [0, \pi]$ are rotation angles on single qubits, after the `data_to_angles` scaling (the suite-wide `ANGLE_SCALE = 1.05` keeps the half-angles $a/2, b/2$ inside $[0, \pi/2 \cdot 1.05]$ where the half-angle Hadamard form is smooth).
- $\alpha = \mathtt{ALPHA\_NORM} = 0.9127$ is a suite-wide normalization so $G_M$ peaks at $1$ over the expected angle range.
- A "tile" is one execution of the seven-qubit per-tile circuit at a fixed $(a, b)$. A base is a `.npz` of per-shot measurements for many tiles at fixed angles, sampled $N_{\text{shots}} = 4096$ times each by default.
- A **substrate** is one of three faithful implementations of the projection circuit: mathematical (FP64 numpy reference), classical (noiseless GPU sampler in `gpu.py`), or quantum (real QPU shots from IBM Runtime).

---

## The four operators

Naming follows the rest of the suite. The numbers attached to $T_1, T_2, T_3$ are chronological in the trajectory, not algebraic.

$$T_1(a, b) = |\cos(a - b)|$$

Standard rank-1 cosine kernel. What cuBLAS computes when fed the lifted representation $[\cos\theta \mid \sin\theta]$ and asked for the inner product, via the identity $\cos(a - b) = \cos a \cos b + \sin a \sin b$. Also what `ghost_rank_k_matmul` in `ghost_kernel.cu` computes at $K = 1$.

$$T_2(a, b) = \left|\cos\!\left(\frac{a - b}{2}\right)\right|$$

Half-angle Hadamard form. The textbook expectation for a Hadamard test on **product states** $|\psi_a\rangle = R_y(a)|0\rangle$ and $|\psi_b\rangle = R_y(b)|0\rangle$. The original framing of the project assumed the QPU was computing this. Probe 1 falsified that assumption, Probe 4 explained why.

$$T_3(a, b) = \tfrac{1}{2}\bigl(1 + \cos^2(a/2)\cos^2(b/2) + \sin^2(a/2)\sin^2(b/2)\bigr) = \tfrac{3}{4} + \tfrac{1}{4}\cos a \cos b$$

The actual target the seven-qubit per-tile circuit implements in the noiseless limit, in $P(\text{ctrl}{=}0)$-space. Derived in Probe 4 by direct statevector simulation; both forms above are equivalent and the second is verified to machine precision against the first.

$$\boxed{\;G_M(a, b) = \frac{1}{\alpha}\sqrt{\frac{1 + \cos a \cos b}{2}}\;}$$

The operator the rest of the suite is built on. It's the matrix-entry form of $T_3$ — i.e. $T_3$ is $G_M$ expressed in $P(\text{ctrl}{=}0)$-space, $G_M$ is $T_3$ expressed in the normalized matrix-entry space the benchmark uses. Derivation below.

Three structural properties of $G_M$ are worth pinning down up front, because they explain the design choices in the kernel.

**Bounded output.** $\cos a \cos b \in [-1, 1]$, so $(1 + \cos a \cos b)/2 \in [0, 1]$ and the un-normalized $\sqrt{(1 + \cos a \cos b)/2} \in [0, 1]$. Dividing by $\alpha < 1$ pushes the peak just above 1; the production code clamps to $[0, 1]$ in software. The bound is what makes per-dimension aggregation safe: no single dimension can dominate a mean.

**Rank-1 in $\cos$-space, but the $\sqrt{\cdot}$ breaks that.** The argument $1 + \cos a \cos b$ is a rank-1 outer product (in $\cos$-space) plus a constant. Numerical SVD of a $32 \times 32$ Gram matrix shows the top singular value carries 97.2%, the next 2.8%, the rest below 0.1% — so $G_M$ is "low-but-not-rank-1," effectively rank 1 to 2. Empirically, $\text{corr}(G_M, \cos a \cdot \cos b)$ on a $32 \times 32$ random-angle grid is $+0.9992$. The 0.0008 gap is the sqrt curvature, and is exactly where the structural robustness of Probe 10.1 lives.

**Not positive semidefinite.** Tested on 50 random Gram matrices of size $32 \times 32$ at random angles in $[0, \pi/2]$: 0 of 50 were PSD. Mean minimum eigenvalue $-0.0074$, worst $-0.0117$. $G_M$ is an *indefinite* kernel — Krein-space minimizers apply, RKHS guarantees do not. This is what closed the indefinite-kernel SVM angle in Probe 9.1 Stage 3.

---

## Why $T_2$ is wrong: the seven-qubit circuit

The per-tile circuit operates on seven qubits, labeled $\{a_1, v_1, a_2, \mathrm{ctrl}, b_1, v_2, b_2\}$:

1. $R_y(a)$ on $v_1$ and $R_y(b)$ on $v_2$.
2. $\mathrm{CNOT}(v_1 \to a_1)$, $\mathrm{CNOT}(v_1 \to a_2)$, $\mathrm{CNOT}(v_2 \to b_1)$, $\mathrm{CNOT}(v_2 \to b_2)$. **These are the ghost CNOTs.**
3. $H$ on $\mathrm{ctrl}$.
4. $\mathrm{CSWAP}(\mathrm{ctrl}; v_1, v_2)$.
5. $H$ on $\mathrm{ctrl}$.
6. Measure $\{\mathrm{ctrl}, a_1, a_2, b_1, b_2\}$.

The textbook swap-test analysis assumes the swap operates on product states $|\psi_a\rangle \otimes |\psi_b\rangle$, in which case $P(\mathrm{ctrl}{=}0) = (1 + |\langle \psi_a | \psi_b \rangle|^2)/2$ and the matrix-entry conversion $\sqrt{2 P_0 - 1}$ recovers $|\langle \psi_a | \psi_b \rangle| = |\cos((a - b)/2)| = T_2$.

The ghost CNOTs break this. After step 2, the registers $\{v_1, a_1, a_2\}$ and $\{v_2, b_1, b_2\}$ are each in a GHZ-correlated state:

$$|v_1 a_1 a_2\rangle = \cos(a/2)\,|000\rangle + \sin(a/2)\,|111\rangle$$

$$|v_2 b_1 b_2\rangle = \cos(b/2)\,|000\rangle + \sin(b/2)\,|111\rangle$$

These are no longer simple single-qubit states. The full six-qubit pre-Hadamard-test state is a product of two GHZ blocks:

$$|\Psi\rangle = \bigl(c_a |000\rangle + s_a |111\rangle\bigr)_{v_1 a_1 a_2} \otimes \bigl(c_b |000\rangle + s_b |111\rangle\bigr)_{v_2 b_1 b_2}$$

where $c_a = \cos(a/2)$, $s_a = \sin(a/2)$, and likewise for $b$. The swap test acts on $(v_1, v_2)$ inside this entangled context, not on isolated $|\psi_a\rangle, |\psi_b\rangle$ — and that's what changes the answer.

---

## Deriving $T_3$ from the ghost-CNOT circuit

The swap-test expectation conditional on a given basis state of $(v_1, v_2)$ is:

$$\langle \mathrm{SWAP}\rangle_{v_1 v_2} = \begin{cases} 1 & v_1 = v_2 \\ 0 & v_1 \neq v_2 \end{cases}$$

Reading off $|\Psi\rangle$, the four basis states of $(v_1, v_2)$ occur with the squared-amplitude probabilities $c_a^2 c_b^2,\ c_a^2 s_b^2,\ s_a^2 c_b^2,\ s_a^2 s_b^2$. Of these, $|00\rangle$ and $|11\rangle$ have $v_1 = v_2$ (contribute $+1$ to the swap expectation), and the cross terms $|01\rangle, |10\rangle$ have $v_1 \neq v_2$ (contribute $0$). So:

$$\langle \mathrm{SWAP}\rangle = c_a^2 c_b^2 + s_a^2 s_b^2 = \cos^2(a/2)\cos^2(b/2) + \sin^2(a/2)\sin^2(b/2)$$

The Hadamard test on $\mathrm{ctrl}$ then converts the swap expectation to a measurement probability:

$$P(\mathrm{ctrl}{=}0) = \frac{1 + \langle \mathrm{SWAP}\rangle}{2} = \frac{1}{2}\bigl(1 + \cos^2(a/2)\cos^2(b/2) + \sin^2(a/2)\sin^2(b/2)\bigr)$$

This is $T_3$. Compare to $T_2$: the textbook product-state derivation gives $\langle \mathrm{SWAP}\rangle = |\langle\psi_a|\psi_b\rangle|^2 = \cos^2((a-b)/2)$, whereas the ghost-CNOT'd circuit gives the mixed-cosine form above. The functional difference (as Probe 6 reports it):

$$\mathrm{MAE}(T_2, T_3) = 0.1010$$

over the $4 \times 4$ angle grid used by the suite — not small. This is the gap Probe 1 measured against $T_2$ and reported as $\approx 0.19$ MAE on the QPU, which is the gap-to-$T_2$ plus the QPU's residual against $T_3$.

The marginals on the ancillas are even simpler. Because the ghost CNOTs perfectly correlate $\{v_1, a_1, a_2\}$ and $\{v_2, b_1, b_2\}$ within each GHZ block, the ancilla bits satisfy $a_1 = a_2$ and $b_1 = b_2$ deterministically in the noiseless limit, with single-qubit marginals:

$$P(a = 1) = \sin^2(a/2), \qquad P(b = 1) = \sin^2(b/2)$$

The full 32-bin joint distribution $P(\mathrm{ctrl}, a_1, a_2, b_1, b_2)$ collapses to 8 non-zero bins (one per choice of $\mathrm{ctrl}, a, b$ with $a_1 = a_2 = a$, $b_1 = b_2 = b$); the remaining 24 are exactly zero.

The classical noiseless sampler in `gpu.py` implements this distribution faithfully: $a_1 = a_2$ drawn from $\mathrm{Bernoulli}(\sin^2(a/2))$, $b_1 = b_2$ from $\mathrm{Bernoulli}(\sin^2(b/2))$, and the control bit constrained so that $a = b$ produces $\mathrm{ctrl} = 0$ deterministically and $a \neq b$ produces a fair coin flip. This is what "noiseless classical implementation of the projection circuit" means — same algorithm, no hardware noise layer.

The Probe 7 parity test directly measures the GHZ correlation on physical hardware: if the ancillas were independent, $P(a_1 \neq a_2)$ would equal the independence null $2 P(a_1)(1 - P(a_2))$, around $0.30$ over the suite's angle range. Observed mean on a representative QPU job: $0.15$. Not zero (decoherence opens the GHZ block), but well below independence — direct hardware confirmation that the GHZ correlation is physically present.

---

## From $T_3$ to $G_M$

$T_3$ as derived is a function of half-angles. The first simplification: $\cos^2(x/2) = (1 + \cos x)/2$ and $\sin^2(x/2) = (1 - \cos x)/2$. Substituting,

$$\cos^2(a/2)\cos^2(b/2) + \sin^2(a/2)\sin^2(b/2) = \frac{(1+\cos a)(1+\cos b)}{4} + \frac{(1-\cos a)(1-\cos b)}{4}$$

Expanding both numerators,

$$= \frac{(1 + \cos a + \cos b + \cos a \cos b) + (1 - \cos a - \cos b + \cos a \cos b)}{4} = \frac{2 + 2\cos a \cos b}{4} = \frac{1 + \cos a \cos b}{2}$$

So $T_3$ in full-angle form is:

$$T_3(a, b) = \frac{1}{2}\!\left(1 + \frac{1 + \cos a \cos b}{2}\right) = \frac{3}{4} + \frac{1}{4}\cos a \cos b$$

A low-order trigonometric polynomial. The $4 \times 4$ angle grid evaluation matches both forms of $T_3$ to machine precision (verified in Probe 9 Stage 1).

The second step is the normalization the suite uses to extract a matrix entry. The QPU manifold is defined as $\sqrt{2 P_0 - 1} / \alpha$ — the textbook way to convert a swap-test $P_0$ measurement into a similarity score. Apply this to $T_3$:

$$2\,T_3(a, b) - 1 = 2\!\left(\frac{3}{4} + \frac{1}{4}\cos a \cos b\right) - 1 = \frac{1}{2} + \frac{1}{2}\cos a \cos b = \frac{1 + \cos a \cos b}{2}$$

Taking the square root and dividing by $\alpha$:

$$G_M(a, b) = \frac{\sqrt{2\,T_3(a, b) - 1}}{\alpha} = \frac{1}{\alpha}\sqrt{\frac{1 + \cos a \cos b}{2}}$$

This is the operator. The matrix-entry form is $G_M$; the $P(\text{ctrl}{=}0)$-space form is $T_3$. The identity is checkable inline:

$$G_M(a, b) = \frac{1}{\alpha}\sqrt{\frac{1 + \cos a \cos b}{2}} \iff T_3(a, b) = \tfrac{3}{4} + \tfrac{1}{4}\cos a \cos b$$

via $G_M^2 = (2 T_3 - 1)/\alpha^2$ in both directions.

A useful product-to-sum form for thinking about how $G_M$ relates to $T_1$: using $\cos a \cos b = \tfrac{1}{2}\cos(a - b) + \tfrac{1}{2}\cos(a + b)$,

$$G_M(a, b) = \frac{1}{\alpha}\sqrt{\frac{1}{2} + \frac{\cos(a - b)}{4} + \frac{\cos(a + b)}{4}}$$

So $G_M$ lives between a $T_1$-style $(a - b)$ coupling and an $(a + b)$ coupling, plus a constant, all under a square root. Both halves matter — the $(a-b)$ component gives the similarity-like behavior, the $(a+b)$ component gives the "angle product" structure that makes $G_M$ anti-correlated with standard matmul on inputs in $[0, \pi/2]$ ($\text{corr}(G_M, \text{matmul}) = -0.75$ on random uniform draws over this range). Without the $(a+b)$ term we'd be back to $T_1$.

---

## The projection-channel identity

The projection channel is how the production runtime estimates $G_M(a, b)$ at a new angle pair $(a, b)$ from physical shot counts that were collected at a fixed pair $(a_o, b_o)$. The identity behind it is importance reweighting.

The shot-level random variables are the ancilla parities $f_a = (a_1 + a_2)/2 \in \{0, 0.5, 1\}$ and $f_b = (b_1 + b_2)/2 \in \{0, 0.5, 1\}$, plus the control bit $c \in \{0, 1\}$. Under noiseless GHZ correlation $a_1 = a_2$ and $b_1 = b_2$, so $f_a$ and $f_b$ are Bernoulli (in $\{0, 1\}$); the third bucket value $0.5$ appears only when decoherence opens the GHZ block, which is how the projection channel quietly accommodates the QPU residual without modeling it.

Treating $f_a, f_b$ as approximately independent Bernoulli variables with success probabilities $p_a = \sin^2(a/2)$ and $p_b = \sin^2(b/2)$, the log-likelihood of a single shot at angles $(\theta_a, \theta_b)$ given firing values $(f_a, f_b)$ is:

$$\log L(f_a, f_b \mid \theta_a, \theta_b) = f_a \log p_a + (1 - f_a) \log(1 - p_a) + f_b \log p_b + (1 - f_b) \log(1 - p_b)$$

The importance weight for reweighting a shot collected at $(a_o, b_o)$ to behave as if it had been collected at $(a, b)$ is the likelihood ratio:

$$w(f_a, f_b) = \exp\Bigl(\log L(f_a, f_b \mid a, b) - \log L(f_a, f_b \mid a_o, b_o)\Bigr)$$

The reweighted estimate of $P(\text{ctrl}{=}0)$ at the new angles is then:

$$\hat P_0(a, b) = \frac{\sum_s w(f_a^{(s)}, f_b^{(s)}) \cdot \mathbf{1}[c^{(s)} = 0]}{\sum_s w(f_a^{(s)}, f_b^{(s)})}$$

where the sum runs over all shots $s$ at the base $(a_o, b_o)$. Plugging this into the matrix-entry conversion gives the projection-channel estimate of $G_M$:

$$\hat G_M(a, b) = \frac{1}{\alpha}\sqrt{\max\bigl(0,\ 2\,\hat P_0(a, b) - 1\bigr)}$$

with the $\max(0, \cdot)$ guarding against shot-noise excursions where $\hat P_0$ briefly dips below $1/2$.

This identity is substrate-agnostic. It applies equally to:
- Bucket counts produced by FP64 numpy reference samples.
- Bucket counts produced by the classical noiseless GPU sampler.
- Bucket counts produced by real QPU shots.

The arithmetic is the same; only the noise on the input bucket counts differs.

Two implementation details that earn their keep:

**Bucket compression.** Both $f_a$ and $f_b$ take only three values, and the projection channel doesn't care about shot ordering. So the entire per-tile shot list compresses losslessly into an $18$-int histogram over $(f_a, f_b, c) \in \{0, 0.5, 1\}^2 \times \{0, 1\}$. From compression onward, every consumer reads 18 ints per tile rather than $N_{\text{shots}} \times 5$ raw bits. The importance-weight sum becomes a constant-size 9-cell weighted dot product:

$$\hat P_0(a, b) = \frac{\sum_{i, j \in \{0, 0.5, 1\}} w(i, j) \cdot n_0(i, j)}{\sum_{i, j} w(i, j) \cdot \bigl(n_0(i, j) + n_1(i, j)\bigr)}$$

where $n_c(i, j)$ is the count of shots with $(f_a, f_b, \mathrm{ctrl}) = (i, j, c)$.

**Log-space evaluation with clipping.** The per-shot log-weight decomposes as a sum of base and slope terms:

$$\log w(f_a, f_b) = \bigl[\log(1 - p_a) - \log(1 - p_{a_o})\bigr] + f_a\bigl[\log p_a - \log p_{a_o} - \bigl(\log(1 - p_a) - \log(1 - p_{a_o})\bigr)\bigr] + (\ldots b \ldots)$$

which lets the kernel compute base and slope once per tile and assemble the nine $\log w$ values by addition only. The $p$ values are clipped to $[\varepsilon, 1 - \varepsilon]$ with $\varepsilon = 0.05$ before any logarithm, and the assembled $\log w$ is clipped to $[-C, +C]$ with $C = 3.0$. These constants correspond to the `EPS` and `CLIP_LOG_W` `#define`s at the top of `ghost_kernel.cu` Section 1, and are calibrated for fp32 stability over the suite's angle range.

The agreement metric in the tied-channel architecture is the per-row mean of $|\hat G_M^{\text{proj}} - G_M^{\text{geom}}|$. On a noiseless GPU base, this measures shot noise on $N_{\text{shots}} = 4096$ per bucket, around 0.01 to 0.06 depending on angle. On a physical QPU base, it measures shot noise plus the hardware residual, around 0.10 to 0.20 — the same range Probes 7 and 8 characterized directly.

---

## The phase-lift

To use $G_M$ as a similarity operator on real-valued embeddings, the embedding components $x \in \mathbb{R}$ need to be mapped onto angles $\theta \in [0, \pi]$. The production runtime uses a bounded saturating phase-lift:

$$\theta(x) = \frac{\pi}{2}\bigl(1 + \tanh(x / 3)\bigr)$$

Three properties this lift was chosen for:

1. **Bounded.** $\theta(x) \in [0, \pi]$, so $\cos\theta \in [-1, 1]$ and $G_M(\theta_Q, \theta_K)$ is bounded in $[0, 1/\alpha]$ regardless of the input magnitudes.
2. **Smooth and near-identity at small inputs.** $\theta(x) \approx \pi/2 + (\pi x)/6$ for $|x| \ll 1$, so small differences in the original embedding translate to small differences in the angle.
3. **Saturating, with non-trivial cosine values at the saturation extremes.** $\theta(x) \to 0$ as $x \to -\infty$ and $\theta(x) \to \pi$ as $x \to +\infty$, giving $\cos\theta \to 1$ and $\cos\theta \to -1$ respectively. Crucially, *both extremes are well-defined fixed points*, so a large spike in any input dimension produces a fixed contribution rather than an unbounded one.

The earlier affine lift $\theta = (\pi/2)(1 + x)$ used in Probe 10 fails property 3: large $|x|$ wraps the cosine around its full period, accidentally cancelling spike contributions and suppressing the very mechanism the experiment was trying to test. The $\tanh$ version is what Probe 10.1 used to expose the per-dim aggregation result.

The per-dimension aggregated similarity on $d$-dimensional embeddings $X_Q, X_K \in \mathbb{R}^d$ is:

$$\mathrm{sim}(X_Q, X_K) = \frac{1}{d \alpha} \sum_{k=1}^{d} \sqrt{\frac{1 + \cos\theta(X_Q^{(k)})\cos\theta(X_K^{(k)})}{2}}$$

with each term bounded in $[0, 1/\alpha]$. The bound on each term divided by the mean factor $1/d$ caps any single dimension's contribution to the aggregate at $1/(d\alpha)$ — which is what makes coherent same-dimension outlier attacks structurally impotent against this operator. A single big dimension cannot bias the score beyond $1/(d\alpha)$, whereas in dot-product attention a single big dimension can dominate via $e^{Q \cdot K / \sqrt{d}}$ with no upper bound.

This is the architectural payoff: $G_M$'s saturation + per-dim averaging gives an attention-shaped similarity whose worst-case behavior under outlier injection is *bounded by construction*.

---

## What the agreement metric actually measures

The agreement metric in the tied-channel architecture is the per-row mean of $|\hat G_M^{\text{proj}} - G_M^{\text{geom}}|$:

$$A_i = \frac{1}{M}\sum_{j=1}^{M} \bigl|\hat G_M^{\text{proj}}(Q_i, K_j) - G_M^{\text{geom}}(Q_i, K_j)\bigr|$$

per query $i$, averaged over all $M$ keys it was compared against. The geometry channel is the closed-form $G_M$ evaluated analytically; the projection channel is the importance-reweighted estimate from physical or simulated shot counts.

What this metric actually measures, per substrate:

- **Mathematical reference (FP64 numpy on the analytical $T_3$ distribution):** $A_i$ is fp32 quantization noise, $\sim 10^{-5}$. The projection identity is correct by derivation.
- **Classical noiseless sampler (`gpu.py`):** $A_i \approx 0.01$ to $0.03$. This is pure shot noise at $N_{\text{shots}} = 4096$ per bucket. The final benchmark measures this directly on three independent GPU bases (mean $A \approx 0.02$).
- **Quantum hardware shots:** $A_i \approx 0.07$ to $0.13$ across the QPU bases the suite ships with. This is shot noise plus a hardware-noise floor — decoherence on the ghost CNOTs, gate-fidelity errors, and calibration drift smearing the bucket counts away from their analytical predictions.

The ratio of QPU to GPU agreement is roughly **5×**. This is the **quantitative hardware-noise readout** the substrate-comparison architecture was built to produce. The same algorithm running on three substrates, with the projection channel agreeing with geometry by a substrate-dependent amount — and that amount being the platform-specific story.

The crucial correction from the trajectory: **the agreement metric does not certify a "quantum advantage."** An earlier version of this document (corresponding to Probe 12 as originally reported) framed the projection channel as preserving signal on QPU that classical hardware destroyed. Probe 12 was subsequently re-run with a corrected classical sampler — `gpu.py` had been emitting bucket counts that didn't faithfully implement the GHZ correlations the noiseless circuit produces. With the corrected sampler, the **classical** projection gap ($+0.121$) is larger than the QPU projection gap ($+0.062$), and both substrates retrieve cleanly under the same operating conditions. The earlier "GPU below the noise floor" result was measuring the sampler bug, not a physics property.

What the final benchmark actually shows is the substrate-equivalence the project set out to verify: the same projection-channel attention algorithm works on all three substrates, with measurable noise attenuation when run on real hardware. Same physics, three platforms.

What this metric does *not* prove:

- It doesn't prove the geometry kernel is the right operator for the application; that's an empirical question that the attention robustness result (Probe 10.1, `final_benchmark_5way.py`) answers separately.
- It doesn't bound a per-element error in retrieval rank; it's a per-row mean over the score matrix. The argmax could still pick the wrong key even with small mean agreement (in practice it doesn't at d=256 with calibrated squelch, because the agreement is much smaller than the inter-key score gap).
- It does not characterize quantum-specific advantages or limitations. The QPU is one of three substrates that faithfully implement the projection circuit; its differences from the classical noiseless reference are purely hardware-noise effects.

The agreement metric is best understood as a **continuous integrity check** that says: "the projection-channel kernel, running on this substrate at this calibration, is producing $G_M$ to within $A$ of its analytical value, per query." For shipping a kernel that runs on physical hardware, that bound is exactly what you want.

---

## Bucket-mask calibration

The projection identity above assumes all 9 buckets $(f_a, f_b) \in \{0, 0.5, 1\}^2$ contribute to the importance-weighted sum. In practice, per-base calibration sometimes finds it advantageous to **mask out** specific buckets — zero their counts before reweighting — to improve signal recovery under attack.

This is not a modification to the operator $G_M$. It's a calibration of the projection-channel **estimator**: which subsets of the bucket histogram give the cleanest estimate of $\hat G_M$ on a given base. Probes 13 through 18 systematically explored which masks help on which bases, with the finding that mask selection is **calibration-dependent rather than physics-dependent** — different QPU runs of the same algorithm (and even different GPU sampler runs) sometimes prefer different masks, and the "Golden Mask" naming from the earlier trajectory (Probe 16) was specific to one tile of one calibration rather than a universal property of the projection circuit.

The final production kernel (`final_benchmark_5way.py`) includes a per-base pre-flight that searches over candidate masks and Flash-Squelch thresholds on a small calibration set, picks the best (mask, threshold) for that base, and locks it for inference. At the production operating point ($d = 256$, power $= 256$), the baseline mask (no buckets dropped) wins on all observed bases — meaning the calibration-dependent mask story is real but **doesn't affect retrieval at the production operating point**, because the per-dim averaging at $d = 256$ already gives enough SNR for the unmodified projection channel to lock cleanly.

Mask calibration is preserved in the pipeline as a defensive measure: at narrower $d$, lower $N_{\text{shots}}$, or harder attack profiles, the right mask might no longer be the identity. The infrastructure to detect that and pick correctly is in place.

---

## Pointers

- **Probe 4** — Direct statevector derivation of $T_3$ and the seven-qubit circuit, with the GHZ block analysis spelled out.
- **Probe 6** — The three-target table ($T_1$, $T_2$, $T_3$), backend-vs-target convergence demonstration.
- **Probe 7** — Direct physical test of GHZ parity ($a_1 = a_2$, $b_1 = b_2$) on QPU shots.
- **Probe 9** — Identity $T_3 \leftrightarrow G_M$ verified to machine precision; classification of $G_M$ (indefinite, low rank, anti-correlated with matmul).
- **Probe 10.1** — Per-dim aggregation and phase-lift design; same-dim coherent attack benchmark.
- **Probe 11–11.2** — Projection-channel range and float32 noise-floor investigations.
- **Probes 13–18** — Bucket-mask ablations across tiles and calibration runs; established that mask selection is calibration-dependent.
- **Probe 12 (corrected)** — Substrate-equivalence verification with the fixed classical sampler.
- **Probe 20** — Auto-calibrating production kernel with per-base mask/threshold selection.
- **`docs/architecture.md`** — How the math compiles into a kernel and a benchmark.
- **`docs/known_issues.md`** — Where the math meets the imperfect software.
