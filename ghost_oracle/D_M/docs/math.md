# D_M Math

This document gives the mathematical formulation for the **D_M**
(Dimensional Entanglement Projection) operator.

It is written as GitHub-flavored Markdown with GitHub-compatible LaTeX math.
Equations are kept in display math blocks and custom LaTeX macros are avoided
so the file renders cleanly on GitHub.

---

## 1. Scope

`D_M` is a dimensional witness-manifold projection operator.

It has three substrate paths:

1. `qproj`: real QPU Bell-listener shot records.
2. `gproj`: GPU-generated controlled Bell-witness records.
3. `geo`: exact closed-form classical reference.

The bounded mathematical claim is:

$$
D_M:
\text{record or metadata}
\longmapsto
\text{YZ-primary / ZY-reciprocal witness manifold}.
$$

The operator is used to test whether active delay/offset conditions separate
from null conditions and whether same-shot pairing, reciprocal structure, and
delay order are load-bearing.

`D_M` does **not** claim to certify Bell nonlocality, reconstruct density
matrices, prove prepared Bell states, or demonstrate quantum advantage.

---

## 2. Record schema

A qproj/gproj D_M base stores a two-bit shot record:

$$
P \in \{0,1\}^{T \times S \times 2},
$$

where:

- $T$ is the number of tiles,
- $S$ is the number of shots per tile,
- the last axis stores the two measured bits of the tile.

Each tile has metadata:

$$
r(t) \in \{0,\ldots,R-1\},
$$

$$
w(t) \in \{0,1,2,3\},
$$

with witness labels:

$$
0 = XY,\qquad 1 = YZ,\qquad 2 = ZY,\qquad 3 = YX.
$$

Each tile also carries delay metadata:

$$
b(t) = \text{base delay},
$$

$$
o(t) = \text{offset delay},
$$

$$
d(t) = b(t) + o(t).
$$

---

## 3. Bit-to-spin map

Each measured bit is mapped to a spin value:

$$
\sigma(x) =
\begin{cases}
+1, & x = 0,\\
-1, & x = 1.
\end{cases}
$$

For tile $t$ and shot $s$:

$$
q_{0,t,s} = \sigma(P_{t,s,0}),
$$

$$
q_{1,t,s} = \sigma(P_{t,s,1}).
$$

---

## 4. Tile-level connected correlator

For each tile, D_M computes three tile moments:

$$
\mu_{0,t}
=
\frac{1}{S}\sum_{s=0}^{S-1} q_{0,t,s},
$$

$$
\mu_{1,t}
=
\frac{1}{S}\sum_{s=0}^{S-1} q_{1,t,s},
$$

$$
\Gamma_t
=
\frac{1}{S}\sum_{s=0}^{S-1} q_{0,t,s}q_{1,t,s}.
$$

The connected two-bit correlator is:

$$
C_t
=
\Gamma_t - \mu_{0,t}\mu_{1,t}.
$$

This is the load-bearing tile statistic used by the qproj/gproj record path.

---

## 5. Rung-level witness projection

For each rung $r$, the four witness components are gathered from the tiles
assigned to that rung:

$$
X_r = C_t \quad \text{where } r(t)=r,\ w(t)=XY,
$$

$$
Y_r = C_t \quad \text{where } r(t)=r,\ w(t)=YZ,
$$

$$
Z_r = C_t \quad \text{where } r(t)=r,\ w(t)=ZY,
$$

$$
U_r = C_t \quad \text{where } r(t)=r,\ w(t)=YX.
$$

Equivalently:

$$
\mathbf{W}_r =
\begin{bmatrix}
X_r\\
Y_r\\
Z_r\\
U_r
\end{bmatrix}
=
\begin{bmatrix}
C(XY)_r\\
C(YZ)_r\\
C(ZY)_r\\
C(YX)_r
\end{bmatrix}.
$$

The canonical D_M coordinate frame is:

$$
Y_r = C(YZ)_r,
$$

$$
R_r = -C(ZY)_r = -Z_r.
$$

Here $Y_r$ is the primary witness coordinate and $R_r$ is the reciprocal return
coordinate.

The comparison coordinate is:

$$
B_r =
\sqrt{X_r^2 + U_r^2}.
$$

The YZ/ZY witness energy is:

$$
E_r =
\sqrt{Y_r^2 + R_r^2}.
$$

The directional specificity is:

$$
S_r =
E_r - B_r.
$$

The directional gap is:

$$
G_r =
Y_r - Z_r.
$$

The reciprocal inversion term is:

$$
I_r =
-Y_r Z_r.
$$

---

## 6. Pi-phase coordinate

The YZ/ZY pair defines a phase-space coordinate:

$$
\phi_r
=
\operatorname{atan2}(R_r,Y_r)\bmod \pi.
$$

The pi-periodic phase features are:

$$
c_r =
\cos(2\phi_r),
$$

$$
s_r =
\sin(2\phi_r).
$$

The factor of $2$ makes the phase representation $\pi$-periodic.

The rung-level D_M vector is therefore:

$$
\mathbf{m}_r =
\begin{bmatrix}
X_r\\
Y_r\\
Z_r\\
U_r\\
Y_r\\
R_r\\
E_r\\
B_r\\
S_r\\
G_r\\
I_r\\
\phi_r\\
c_r\\
s_r\\
b_r\\
o_r\\
d_r
\end{bmatrix}.
$$

---

## 7. Delay normalization

For a sequence $x_r$, D_M uses min-max normalization:

$$
\mathcal{N}(x_r)
=
\frac{x_r - \min_j x_j}{\max_j x_j - \min_j x_j}.
$$

If the denominator is zero, the normalized sequence is defined as zero:

$$
\mathcal{N}(x_r)=0
\quad
\text{when}
\quad
\max_j x_j = \min_j x_j.
$$

Two delay coordinates are used:

$$
x^{\mathrm{lin}}_r = \mathcal{N}(d_r),
$$

$$
x^{\log}_r = \mathcal{N}(\log(1+d_r)).
$$

The active tracking score chooses the stronger absolute correlation between
linear-delay and log-delay tracking.

For a feature sequence $a_r$, define:

$$
\rho_{\mathrm{lin}}(a)
=
\operatorname{corr}(x^{\mathrm{lin}},a),
$$

$$
\rho_{\log}(a)
=
\operatorname{corr}(x^{\log},a).
$$

Then:

$$
\rho_E
=
\begin{cases}
\rho_{\log}(E), & |\rho_{\log}(E)| > |\rho_{\mathrm{lin}}(E)|,\\
\rho_{\mathrm{lin}}(E), & \text{otherwise},
\end{cases}
$$

and:

$$
\rho_S
=
\begin{cases}
\rho_{\log}(S), & |\rho_{\log}(S)| > |\rho_{\mathrm{lin}}(S)|,\\
\rho_{\mathrm{lin}}(S), & \text{otherwise}.
\end{cases}
$$

---

## 8. Pi-periodic fit score

The pi-periodic fit asks whether the phase trajectory tracks delay through
the $\cos(2\phi)$ and $\sin(2\phi)$ coordinates.

For a chosen normalized delay coordinate $x_r$, define:

$$
\Pi(x,\phi)
=
\sqrt{
\operatorname{corr}(x,\cos(2\phi))^2
+
\operatorname{corr}(x,\sin(2\phi))^2
}.
$$

The final pi score is:

$$
\Pi_D
=
\max
\left(
\Pi(x^{\mathrm{lin}},\phi),
\Pi(x^{\log},\phi)
\right).
$$

The selected mode is:

$$
\operatorname{mode}_{\Pi}
=
\begin{cases}
0, & \Pi(x^{\mathrm{lin}},\phi) \ge \Pi(x^{\log},\phi),\\
1, & \Pi(x^{\log},\phi) > \Pi(x^{\mathrm{lin}},\phi).
\end{cases}
$$

---

## 9. Phase velocity

Phase differences are wrapped into the interval $[-\pi/2,\pi/2]$:

$$
\Delta_{\pi}(\alpha)
=
\left((\alpha+\pi/2)\bmod \pi\right)-\pi/2.
$$

The local phase velocity is:

$$
v_i
=
\frac{\Delta_{\pi}(\phi_i-\phi_{i-1})}{d_i-d_{i-1}},
\qquad i=1,\ldots,R-1,
$$

whenever $d_i \ne d_{i-1}$.

The phase velocity tracking score is:

$$
\rho_v
=
\operatorname{corr}
\left(
\mathcal{N}\left(\log(1+\tfrac{d_i+d_{i-1}}{2})\right),
v_i
\right).
$$

The unwrapped phase span in units of $\pi$ is:

$$
L_{\phi}
=
\frac{\max_i \Phi_i-\min_i \Phi_i}{\pi},
$$

where:

$$
\Phi_0=\phi_0,
$$

$$
\Phi_i
=
\Phi_{i-1}
+
\Delta_{\pi}(\phi_i-\phi_{i-1}).
$$

---

## 10. Summary vector

The base-level D_M summary vector is:

$$
\mathbf{s}
=
\begin{bmatrix}
R\\
\overline{Y}\\
f_Y\\
\overline{Z}\\
f_I\\
\overline{E}\\
E_{\max}\\
\overline{S}\\
S_{\max}\\
\Pi_D\\
\operatorname{mode}_{\Pi}\\
\rho_E\\
\rho_S\\
\rho_v\\
L_{\phi}\\
P_D
\end{bmatrix}.
$$

The mean terms are:

$$
\overline{Y}
=
\frac{1}{R}\sum_{r=0}^{R-1}Y_r,
$$

$$
\overline{Z}
=
\frac{1}{R}\sum_{r=0}^{R-1}Z_r,
$$

$$
\overline{E}
=
\frac{1}{R}\sum_{r=0}^{R-1}E_r,
$$

$$
\overline{S}
=
\frac{1}{R}\sum_{r=0}^{R-1}S_r.
$$

The maximum terms are:

$$
E_{\max}
=
\max_r E_r,
$$

$$
S_{\max}
=
\max_r S_r.
$$

The sign fractions use a small numerical deadband:

$$
\epsilon_{\mathrm{sign}} = 10^{-6}.
$$

The YZ-positive fraction is:

$$
f_Y
=
\frac{1}{R}
\sum_{r=0}^{R-1}
\mathbf{1}\left[Y_r > \epsilon_{\mathrm{sign}}\right].
$$

The reciprocal-inversion fraction is:

$$
f_I
=
\frac{1}{R}
\sum_{r=0}^{R-1}
\mathbf{1}\left[Y_r Z_r < -\epsilon_{\mathrm{sign}}\right].
$$

The deadband prevents analytic-zero rungs from being counted differently by
float64 CPU and float32 CUDA paths.

---

## 11. Projection score

The scalar projection score is a bounded benchmark score, not an entanglement
certificate.

Define positive parts:

$$
E_+ = \max(0,\overline{E}),
$$

$$
S_+ = \max(0,\overline{S}),
$$

$$
Y_+ = \max(0,\overline{Y}).
$$

Define pi witness strength:

$$
W_{\pi}
=
E_+ \Pi_D.
$$

Define tracking strength:

$$
T_D
=
\frac{1}{2}\left(|\rho_E|+|\rho_S|\right).
$$

The D_M projection score is:

$$
P_D
=
0.35E_+
+
0.25S_+
+
0.15Y_+
+
0.15W_{\pi}
+
0.10T_D.
$$

This score lives on the natural connected-correlator scale. It intentionally
does not use empirical divisors that would allow a high phase fit to dominate
when the measured YZ/ZY witness energy is small.

---

## 12. Record-path destructive control

The primary destructive control is independent bit shuffle.

It preserves the two single-qubit marginal sequences but breaks same-shot
pairing.

For each tile $t$, choose a permutation $\pi_t$ over shots and define:

$$
P'_{t,s,0}=P_{t,s,0},
$$

$$
P'_{t,s,1}=P_{t,\pi_t(s),1}.
$$

The control connected correlator is:

$$
C'_t
=
\frac{1}{S}\sum_{s=0}^{S-1}
\sigma(P'_{t,s,0})\sigma(P'_{t,s,1})
-
\mu'_{0,t}\mu'_{1,t}.
$$

A clean D_M record path should satisfy:

$$
P_D(\text{active}) > P_D(\text{shuffle(active)}),
$$

with a stronger drop on active conditions than on null conditions.

The fractional collapse is:

$$
D_{\mathrm{shuffle}}
=
\frac{P_D-P'_D}{P_D},
$$

for $P_D \ne 0$.

---

## 13. Exact GEO reference

The exact GEO path is a closed-form classical reference. It does not sample
shots and does not use qproj/gproj records.

For each rung $r$, define base delay $b_r$ and mean offset:

$$
\bar{o}_r
=
(4r+\tfrac{3}{2})\,o,
$$

where $o$ is the condition offset step.

The active delay coordinates are:

$$
x^{\mathrm{space}}_r
=
\mathcal{N}(\log(1+b_r)),
$$

$$
x^{\mathrm{time}}_r
=
\mathcal{N}(\log(1+b_r+\bar{o}_r)).
$$

The combined D_M coordinate is:

$$
x^{D_M}_r
=
\sqrt{
\frac{
w_{\mathrm{space}}(x^{\mathrm{space}}_r)^2
+
w_{\mathrm{time}}(x^{\mathrm{time}}_r)^2
}{
w_{\mathrm{space}}+w_{\mathrm{time}}
}
}.
$$

The active GEO energy is:

$$
E^{\mathrm{geo}}_r
=
E_{\mathrm{floor}}
+
E_{\mathrm{scale}}
\left(x^{D_M}_r\right)^{\gamma}.
$$

The active GEO phase is defined by:

$$
\cos(2\phi^{\mathrm{geo}}_r)
=
2x^{\mathrm{time}}_r - 1.
$$

Equivalently:

$$
\phi^{\mathrm{geo}}_r
=
\frac{1}{2}
\arccos(2x^{\mathrm{time}}_r-1).
$$

The exact GEO witnesses are:

$$
Y^{\mathrm{geo}}_r
=
E^{\mathrm{geo}}_r\cos(\phi^{\mathrm{geo}}_r),
$$

$$
Z^{\mathrm{geo}}_r
=
-
E^{\mathrm{geo}}_r\sin(\phi^{\mathrm{geo}}_r).
$$

The comparison channels are zero in the exact reference:

$$
X^{\mathrm{geo}}_r = 0,
$$

$$
U^{\mathrm{geo}}_r = 0.
$$

The null GEO condition is the exact zero manifold:

$$
X^{\mathrm{geo}}_r
=
Y^{\mathrm{geo}}_r
=
Z^{\mathrm{geo}}_r
=
U^{\mathrm{geo}}_r
=
0.
$$

Thus:

$$
P_D(\mathrm{geo},\mathrm{null})=0.
$$

---

## 14. Conditions

The locked benchmark uses three canonical conditions.

Null:

$$
b_r = 0,
\qquad
o=0.
$$

Base-only:

$$
b_r \in \{0,256,1024,4096,16384\},
\qquad
o=0.
$$

Offset-on:

$$
b_r \in \{0,256,1024,4096,16384\},
\qquad
o=128.
$$

The expected ordering is:

$$
P_D(\mathrm{active}) > P_D(\mathrm{null}),
$$

for qproj, gproj, and geo.

The base-only and offset-on conditions are both active. They are not required
to separate strongly from each other at rung level.

---

## 15. Allowed channel re-description

The canonical YZ/ZY frame is a reporting coordinate. It is not the only valid
description of the dimensional manifold.

Let a rung witness vector be:

$$
\mathbf{W}_r =
(X_r,Y_r,Z_r,U_r).
$$

Allowed channel re-descriptions are transformations:

$$
A:\mathbb{R}^4 \to \mathbb{R}^4
$$

that preserve paired reciprocal structure. The final benchmark keeps these
as allowed transformations rather than falsification controls.

The dimensional-invariant score is:

$$
P_D^{\mathrm{inv}}(\mathbf{W})
=
\max_{A \in \mathcal{A}}
P_D(A\mathbf{W}),
$$

where $\mathcal{A}$ is the allowed family of channel re-descriptions.

A clean allowed transform should satisfy:

$$
P_D^{\mathrm{inv}}(A\mathbf{W})
\approx
P_D^{\mathrm{inv}}(\mathbf{W}).
$$

This is why witness-label shuffle is not treated as the primary destructive
control in the final claim. The primary destructive controls are same-shot
pairing break, reciprocal break, delay permutation, and compound corruption.

---

## 16. Forbidden single-fault controls

Let the forbidden fault family be:

$$
\mathcal{F}
=
\{
F_{\mathrm{shuffle}},
F_{\mathrm{reciprocal}},
F_{\mathrm{delay}},
F_{\mathrm{basis}},
F_{\mathrm{non\text{-}equiv}}
\}.
$$

Representative examples:

- $F_{\mathrm{shuffle}}$: independent bit shuffle.
- $F_{\mathrm{reciprocal}}$: reciprocal-return break.
- $F_{\mathrm{delay}}$: cross-rung delay permutation.
- $F_{\mathrm{basis}}$: same label with wrong delay.
- $F_{\mathrm{non\text{-}equiv}}$: non-equivalent channel corruption.

For a single fault $F$, define retention:

$$
\eta(F)
=
\frac{
P_D^{\mathrm{inv}}(F\mathbf{W})
}{
P_D^{\mathrm{inv}}(\mathbf{W})
},
$$

when $P_D^{\mathrm{inv}}(\mathbf{W}) \ne 0$.

Probe 23 showed that many single-fault retentions remain high. The final
interpretation is that D_M has a limited error-correcting character: a single
structural violation can often be repaired by the remaining dimensional
agreement.

---

## 17. Compound corruption boundary

Let a depth-$k$ corruption be a composition of $k$ independent forbidden
faults:

$$
F^{(k)}
=
F_k \circ F_{k-1} \circ \cdots \circ F_1,
\qquad
F_i \in \mathcal{F}.
$$

The depth-$k$ survival is:

$$
\eta_k
=
\frac{
P_D^{\mathrm{inv}}(F^{(k)}\mathbf{W})
}{
P_D^{\mathrm{inv}}(\mathbf{W})
}.
$$

Across trials, D_M reports:

$$
\overline{\eta}_k,
\qquad
\operatorname{median}(\eta_k),
\qquad
\Pr(\eta_k < \theta_c),
$$

with collapse threshold:

$$
\theta_c = 0.5.
$$

The collapse depth is:

$$
k_c
=
\min
\left\{
k:
\operatorname{median}(\eta_k)<\theta_c
\right\}.
$$

The active D_M manifolds cross the collapse boundary around:

$$
k_c \approx 2 \text{ to } 3.
$$

The null manifolds have no comparable active structure to collapse, so they do
not show the same collapse profile.

This is the strongest D_M control result.

---

## 18. Substrate agreement

For each substrate $s$ and condition $c$, the benchmark computes:

$$
\mathbf{s}_{s,c}
=
D_M(s,c).
$$

The active-vs-null separation for a substrate is:

$$
\Delta_s(c)
=
P_D(s,c)-P_D(s,\mathrm{null}),
\qquad
c \in \{\mathrm{base\_only},\mathrm{offset\_on}\}.
$$

A clean substrate should satisfy:

$$
\Delta_s(c)>0.
$$

Substrate agreement is evaluated by comparing summary vectors and rung profiles:

$$
\operatorname{corr}(\mathbf{s}_{s_1,c},\mathbf{s}_{s_2,c}),
$$

$$
\left\|
\mathbf{s}_{s_1,c}
-
\mathbf{s}_{s_2,c}
\right\|_2,
$$

and rung-level correlations such as:

$$
\operatorname{corr}(E_{s_1,c,r},E_{s_2,c,r}).
$$

The substrates are not expected to be numerically identical. The expected
relationship is:

$$
\text{same condition ordering}
+
\text{same control behavior}
+
\text{compatible witness geometry}.
$$

---

## 19. Raw-signal projection path

Probe 22 introduced the mature raw-signal qproj/gproj CUDA path.

Given a raw bit bank:

$$
B_i \in \{0,1\},
$$

and a frozen base pair record:

$$
P_{t,s,j},
$$

the projected pair is:

$$
\widehat{P}_{t,s,j}
=
D_{t,s,j} \oplus P_{t,s,j},
$$

where $D_{t,s,j}$ is a deterministic raw-data pair derived from the bit bank,
tile index, witness index, and delay metadata.

The projected pair then uses the same record path:

$$
\widehat{P}
\longmapsto
C_t
\longmapsto
\mathbf{W}_r
\longmapsto
\mathbf{s}.
$$

This path is retained for compatibility and future task probes, but the final
capstone claim does not rely on the saturated retrieval task from Probes 19/22.

---

## 20. DER appendix score

The DER retrieval kernel is retained as an appendix/probe tool.

For query $q$ and candidate $k$, define per-rung differences:

$$
\delta_Y = Y^q_r-Y^k_r,
$$

$$
\delta_R = R^q_r-R^k_r,
$$

$$
\delta_E = E^q_r-E^k_r,
$$

$$
\delta_S = S^q_r-S^k_r,
$$

$$
\delta_c = c^q_r-c^k_r,
$$

$$
\delta_s = s^q_r-s^k_r.
$$

The directional distance score is:

$$
D_{\mathrm{dir}}
=
-\sum_r
\left(
\delta_Y^2+\delta_R^2
\right).
$$

The pi-phase distance score is:

$$
D_{\pi}
=
-\sum_r
\left(
\delta_c^2+\delta_s^2
\right).
$$

The energy and specificity scores are:

$$
D_E
=
-\sum_r \delta_E^2,
$$

$$
D_S
=
-\sum_r \delta_S^2.
$$

The D_M DER score is:

$$
D_{\mathrm{DER}}
=
0.46D_{\mathrm{dir}}
+
0.24D_{\pi}
+
0.18D_E
+
0.12D_S.
$$

This score demonstrated useful control behavior, but the retrieval tasks
saturated against simple baselines. It is therefore not part of the default
final claim.

---

## 21. Non-claims

The following are explicitly out of scope:

$$
D_M \ne \text{Bell nonlocality certificate}.
$$

$$
D_M \ne \text{density-matrix reconstruction}.
$$

$$
D_M \ne \text{proof of prepared Bell states}.
$$

$$
D_M \ne \text{QPU speedup claim}.
$$

$$
D_M \ne \text{quantum advantage claim}.
$$

The final claim is narrower:

$$
D_M
=
\text{controlled dimensional witness-manifold projection}
$$

with qproj/gproj/geo substrate linkage and control-tested collapse behavior.

---

## 22. Implementation correspondence

The mathematical objects map directly to the CUDA kernel outputs.

Tile metrics:

$$
(\mu_0,\mu_1,\Gamma,C,p_{00},p_{01},p_{10},p_{11})
\longleftrightarrow
\texttt{dm\_tile\_correlator\_kernel\_u8}.
$$

Rung metrics:

$$
(X,Y,Z,U,Y,R,E,B,S,G,I,\phi,c,s,b,o,d)
\longleftrightarrow
\texttt{dm\_rung\_projection\_kernel\_f32}.
$$

Summary metrics:

$$
(R,\overline{Y},f_Y,\overline{Z},f_I,\overline{E},E_{\max},
\overline{S},S_{\max},\Pi_D,\rho_E,\rho_S,\rho_v,L_{\phi},P_D)
\longleftrightarrow
\texttt{dm\_projection\_summary\_kernel\_f32}.
$$

Exact GEO metrics:

$$
(b_r,\bar{o}_r)
\longmapsto
(Y^{\mathrm{geo}}_r,Z^{\mathrm{geo}}_r,E^{\mathrm{geo}}_r,\phi^{\mathrm{geo}}_r)
\longleftrightarrow
\texttt{dm\_geo\_exact\_rung\_projection\_kernel\_f32}.
$$

---

## 23. Final mathematical summary

The D_M operator is:

$$
D_M(P,\mathcal{M})
=
\mathbf{s},
$$

where $P$ is either a record substrate or a closed-form GEO metadata substrate,
$\mathcal{M}$ is the tile/rung/witness/delay metadata, and $\mathbf{s}$ is
the D_M summary vector.

The load-bearing path is:

$$
P
\longmapsto
C_t
\longmapsto
(X_r,Y_r,Z_r,U_r)
\longmapsto
(Y_r,R_r,E_r,S_r,\phi_r)
\longmapsto
P_D.
$$

The core validation inequalities are:

$$
P_D(\mathrm{active}) > P_D(\mathrm{null}),
$$

$$
P_D(\mathrm{active})
>
P_D(\mathrm{independent\ bit\ shuffle(active)}),
$$

$$
P_D^{\mathrm{inv}}(A\mathbf{W})
\approx
P_D^{\mathrm{inv}}(\mathbf{W})
\quad
\text{for allowed } A,
$$

and:

$$
\operatorname{median}(\eta_k)<0.5
\quad
\text{for active manifolds at } k\approx 2\text{ to }3.
$$

That is the mathematical claim to defend.
