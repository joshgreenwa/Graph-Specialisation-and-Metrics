# Functional carriage against the Bamberger range measure

A CPU-only replication of Figure 3 of Bamberger, Gutteridge, le Roux, Bronstein & Dong,
*On Measuring Long-Range Interactions in Graph Neural Networks* (ICML 2025,
[arXiv:2506.05971](https://arxiv.org/abs/2506.05971)), measured with this repository's canonical
donor-swap **Functional carriage** instead of their Jacobian.

The whole sequence runs in about twenty minutes on a laptop CPU, most of it in the
2,000-replicate bootstraps rather than the measurements:

```bash
./run_all.sh
```

## The two measures

**Theirs (tangent).** For a differentiable map `F`, the influence distribution at output node `u`
is `I_u(v) ∝ Σ_{a,b} |∂F^a_u / ∂x^b_v|` and the normalised range is its expected distance,
`ρ̂_u = E_{v~I_u}[d(u,v)]`, averaged over nodes and then graphs.

**Ours (secant).** A semantic donor-swap (methodology §2.1) replaces the entire payload of one
**source** node `s` with a real donor node's payload drawn under the graph-balanced,
minimum-degree-gap law, and the response is read at every **carrier** `i` as
`F_sens[i,s] = mean_k ||q_{s,k,i}||₂` with `q = ⟨g_out_i, Δh_i⟩` (methodology §5). Here the node
outputs *are* the carrier states and the readout is the identity, so `g_out` is the identity and
`||q_i||₂ = ||Δh_i||₂`.

**The bridge.** Carrier `i` ↔ their output node `u`; source `s` ↔ their input node `v`; the
distance is the same shortest-path distance on the pristine graph. The carriage analogue of `ρ̂_u`
is the expected distance of the carriage field **at fixed carrier, normalised over sources**:

```
ρ̂^F_i = Σ_s F̃[i,s] d(i,s) / Σ_s F̃[i,s]
```

This transposes the normalisation used by the methodology's §4.3 distance decomposition, which
fixes the source and distributes over carriers. Both read the same saved `(graph, i, s, distance,
F_sens)` artifact — the bridge is post-processing, not a new measurement.

**Event normalisation.** A donor swap moves the payload by `δ_{s,k} = x_s − x^donor_{s,k}`, whose
norm is the `dose` already recorded in the event manifest. For a linear operator the raw field is
`F_sens[i,s] = |L_{is}| · mean_k ||δ_{s,k}||₂`, so the per-source donor scale survives the ratio and
biases it. Dividing each event by its own dose,

```
F̃[i,s] = mean_k ( ||q_{s,k,i}||₂ / ||δ_{s,k}||₂ )
```

removes it exactly, and `F_sens` itself is unchanged. **Every carriage curve in these figures is
the derived, event-normalised field unless the legend says otherwise.** The registered `F_sens`
range is reported alongside it in `fig2_exactness.png` and in the step/oscillation sweeps of
`fig4_divergence.png`, and is stored in `carriage_raw` in every result JSON. The two are not
interchangeable where the response scale varies with the perturbation: in the oscillation sweep at
`ω = 32` they read `0.499` and `0.190`.

## What the replication establishes

Over all 32 settings (4 task variants × `k = 1…8`) on a 16×16 grid, 12 base graphs, 6 donor
graphs, 8 donor events per source, every node a source:

| quantity | measured |
|---|---|
| `max ‖F̃[i,s] − |L_is|‖∞` | `1.7e-16` |
| `max |ρ̂^Jacobian − ρ̂^carriage|` | `8.9e-15` |
| `max |published − ρ̂^carriage|` (24 settings) | `2.2e-06` |
| un-normalised `F_sens` range, deviation from Jacobian | `−0.03%` to `+0.17%` |

1. **Exact recovery of the influence matrix.** For a linear operator with identity readout, every
   *single* donor event satisfies `||q_{s,k,i}||₂ / ||δ_{s,k}||₂ = |L_{is}|` — there is no Monte
   Carlo error at all, at any `K ≥ 1`. The `1.7e-16` in the table is measured on the
   donor-averaged field; the stronger per-event identity is asserted directly in
   `test_event_normalised_carriage_recovers_influence_exactly`. The donor sweep confirms it: the event-normalised range is
   bit-identical (`2.8101897409`) at `K = 1, 2, 4, …, 64`, while the un-normalised one wanders at
   the `1e-3` level and does not converge to the Jacobian value. Asserted in `test_rangelib.py`,
   not just plotted.
2. **The two ranges agree to machine precision** across every task and every `k`, and the
   event-normalised carriage range and the Jacobian range both match the values the paper itself
   publishes (`data/plotting/grid_task_range_*.csv`) to `2.2e-6` — consistent with the float32
   rounding in their logged CSV, whose own relative resolution at these magnitudes is `~6e-7`.
   (The un-normalised `F_sens` range is the looser one, at `≤ 0.17%`; see point 3.)
3. **The donor-scaling bias is real but small here.** It equals `Cov_w(c_s, d)/E_w[c_s]` — a
   covariance between the per-source donor scale and distance — so it vanishes whenever payload
   magnitude is uncorrelated with distance, which is nearly true for i.i.d. features. Measured at
   `≤ 0.17%` across the 32 replication settings. Within the donor sweep's own controlled
   comparison, deliberately heteroscedastic payloads (an 8× ramp in per-node standard deviation,
   also in `results/donor_sweep.json`) raise the bias from `~0.04%` to `~0.11%` at `K = 1`, settling
   near `~0.07%` for `K ≥ 2` — still small, and still removed exactly by event normalisation.
4. **Two separate channel conditions, easy to conflate.** Carriage recovers a donor-direction-free
   magnitude, `F_sens[i,s] = |L_{is}| c_s`, iff every Jacobian block is conformal,
   `M_{is}ᵀM_{is} = L_{is}²I`. Agreement with *their* range needs strictly more: their weight is the
   entrywise `Σ_{a,b}|M^{ab}_{is}|`, which for `M_{is} = L_{is}O_{is}` equals `|L_{is}|·‖O_{is}‖₁,₁`
   — so the orthogonal factor must be the *same* `O` for every block, or its `L1` norm must not vary
   with `s`, for the extra factor to cancel in the ratio. The three synthetic operators act
   identically on all channels, so both conditions hold and agreement is exact; a channel-mixing
   operator breaks the second one and the two measures separate (Figure 4d).

## Two definitional findings about the paper

* **`k-Power` in Figure 3 uses self-loops — and Section 5.2 says both things.** The paper defines
  `Ã` as "the symmetric normalized adjacency with self loops" when introducing the topology
  experiment, and as "the symmetric normalized adjacency matrix without self-loops" fifty lines
  later in the k-Power task definition. The code follows the first: the Figure 3 block of
  `scripts/synthetic_exps.sh` selects `adjacency_self_loop_power_sym_`, and the shipped
  `grid_task_range_power.csv` names that distance function in its own `Name` column. We read the
  task-list bullet as a typo, not a code/text divergence.

  The discriminator is the **level, not the shape**. On the paper's own 16×16 grid the no-self-loop
  curve is also smooth and sublinear (`1.000, 1.457, 1.816, 2.094, 2.346, 2.558, 2.760, 2.937`) —
  it just sits above the published one, and at `k = 1` it is exactly `1.000` against a published
  `0.786`, which settles it on its own. The parity effect that a bipartite `Ã^k` without self-loops
  implies is visible only on a path, where the range becomes a non-monotone staircase dipping at
  every even `k` (33-node path: `1.000, 0.965, 1.472, 1.435, 1.818, 1.781, 2.101, 2.064`). Both
  variants are measured here on both topologies.
* **Figure 3(b) is a hand-drawn schematic, not a plot of measured data.** The paper's code does
  build the influence matrix — it has to, to compute the range — but it never plots one: the
  panel's three shapes are hard-coded constants in the plotting notebook
  (`power = 0.65·(1 − (|x|/2.15)^2.8)`, a cosmetic exponent), on stripped axes with unrelated
  vertical scales. Panel (b) of `fig1_replication.png` plots the actual influence distributions the
  schematic stands in for, by both methods, which coincide; the measured shapes do match the
  qualitative claim (spike at `k`, flat over the ball, decaying). Note the panel is the 16×16-grid
  analogue and plots the per-node mean within each distance shell; the schematic is drawn on a line
  graph, whose direct counterpart is the supplementary `fig1s_replication_path.png`.

## Where the two measures genuinely part company

`fig4_divergence.png`. All on a path graph. Panels (a)–(c) use a single feature channel, so the
`L1`-over-channel-pairs and `L2`-over-outputs aggregations coincide exactly and only the
tangent/secant difference is in play; panel (d) is the one that varies the channel count, and it
isolates the aggregation difference instead.

* **(a) The Jacobian range is the `α → 0` limit of carriage — on this task family.** Interpolating
  the donor payload from the clean value to the registered full swap moves the carriage range
  continuously from the Jacobian value to the finite-perturbation value.

  **The identity is not general.** The `α → 0` limit of event-normalised carriage is
  `E_δ̂ ‖J_{is} δ̂‖₂`, a directionally-averaged L2 operator response, whereas the paper's influence
  is the entrywise L1 norm `Σ_{a,b}|J^{ab}_{is}|`. These are different functionals of the same
  Jacobian block, and they agree only when every block is a scalar multiple of one *fixed*
  orthogonal matrix — in particular the single-channel case used in this panel, or an operator
  acting identically on all channels. Per-block conformality alone is not enough: a different
  orthogonal factor per block already separates them. Panel (d) is the standing counterexample.
* **(b) Conditionally-active long-range terms: the Jacobian under-reports.** With a far term
  `σ(x_v/τ)` (threshold at the payload mean), as `τ → 0` the tangent at almost every clean input
  is flat, so the Jacobian range collapses toward zero — while a realistic node replacement still
  flips the term and the carriage range does not move.
* **(c) High-frequency far terms: the Jacobian over-reports.** With `sin(ωx_v)/ω` the derivative
  stays `O(1)` for every `ω` while the realised response decays like `1/ω`. The Jacobian keeps
  reporting a long range for a dependence that no realistic input change can exercise.
* **(d) Channel aggregation.** Their `Σ_{a,b}|J^{ab}|` counts an entrywise-diffuse block `√d` times
  more than a conformal block of the same Frobenius energy; carriage contracts the donor
  displacement through the block first and sees only its singular values. On a task whose far block
  is rank-one and diffuse, the two ranges separate as `d` grows.

None of these makes one measure right and the other wrong: they answer different questions
("how far does an infinitesimal change propagate *at this input*" versus "how far does a realistic
node replacement propagate *under this donor law*").

**About the gradient-free probe in (b) and (c).** It uses no derivative, but it is *not*
independent of carriage: it replaces one node's payload, takes the output-change norm at the read
node, and weights each distance by its shell total — the same construction both ranges use. In this
experiment it even draws its replacement from the same standard-normal law the donor pool is built
from. It differs by dropping the dose normalisation (so it is the sibling of the un-normalised
`F_sens`, which it tracks to within ~2%) and by perturbing one node per shell rather than every
source. Read it as a robustness check on those two choices, not as an impartial arbiter of tangent
versus secant — it lives entirely on the secant side. On a linear control whose range is known
analytically (`8/3`) it returns `2.63`, so it carries a percent-level Monte-Carlo bias of its own.

## Learned approximators

`fig3_learned.png`. A small message-passing network (`relu(W_s h + W_n Ã h)`, `k+2` layers) is
trained to regress each operator on a 25-node path graph, and both measures are applied to the
trained map. This experiment uses the plain operator `M`, not the Figure-3 transpose `Mᵀ`; it is a
self-contained model-versus-target comparison, so the convention only has to be consistent within
it. Three restarts are run and the best `R²` is kept, so the reported `R²` is a
selected-maximum, not an unbiased fit statistic — it is there to order the models, not to be a
headline number. This is the setting the paper's own Section 6 uses, and the setting
where the map is no longer linear so neither exactness argument applies.

The two measures stay within `~0.05` of each other on every model that fits (`R² ≥ 0.96`), and both
under-report relative to the operator as the fit degrades — the measured range is a property of the
*model*, not of the task it was trained on.

`k-Dirac` at `k = 6` is reported as **non-estimable**, not as a number. The network has eight
layers, so the six-hop shell is well inside its reachability, but it collapses to a near-constant
map (`R² ≈ 0`) under every restart and seed we tried; the best any seed reached was `R² ≈ 0.25`.
A constant map has no influence mass anywhere, so both measures correctly refuse to report a range
rather than returning `0/0`. That is the methodology's non-estimable rule doing its job, and it is
also a small demonstration of the paper's own premise: reachability is not learnability.

## Graph-level tasks without a Hessian

`fig5_graphlevel.png`. For a pooled scalar output the Jacobian is a vector and carries no pairwise
structure, so the paper switches to the Hessian `η̂_u = Σ_v |∂²y/∂x_u∂x_v| d(u,v) / Σ_v |·|`. A donor
swap already supplies one index as a finite intervention (the source) and the other as a
pre-pooling carrier state, so Functional carriage reads a pairwise field from first-order
information alone — which is what the paper's own Table 1 footnote anticipates for the Jacobian
route on graph-level tasks.

On `y(X) = Σ_u |N_≤k(u)|⁻¹ Σ_{v∈N_≤k(u)} (x_u − x_v)²` the two agree on support exactly (mass on
`d ≤ k`, nothing beyond) and track each other linearly in `k`, with carriage running about 20%
longer because it weights within the ball differently. They are **not** the same object — theirs
pairs two inputs, ours pairs an input with a carrier state — so this is a tracking result, not an
identity, and the 20% gap is the price of the substitution rather than an error in either.

## Does it transfer? A trained model on a task we did not design

`fig6_realmodel.png`, `run_realmodel.py`. Every divergence above is on a task built to break a
first-order expansion, which is an existence proof and nothing more. This runs the same comparison
on a GraphGPS-style model (depth 3, 156k parameters, local GINE branch + biased self-attention)
trained on **MarkedTreePath**, a node-level task authored two months before this study: predict
whether each node lies on the path between two marked endpoints. It reaches 0.979 node accuracy
with a mean path length of 6.9 hops, so the task has genuine long-range structure. Both measures
are based at `h0 = token_emb(x) + pe_proj(pe)`, because the raw input is a discrete mark through an
embedding and has no Jacobian.

**The gap transfers.** The dose ladder is not flat: the carriage range falls from `3.125` at
`α = 0.001` to `1.991` at a full real-donor replacement — a **36% drop**, well outside the
intervals. Read the ladder's own endpoints rather than carriage-against-Jacobian: with 64 mixing
channels the conformality condition of point 4 fails, so a fixed-α comparison also contains the
`L1`-versus-`L2` aggregation difference, whereas both ends of the ladder share an aggregation and
isolate secant-versus-tangent cleanly.

**But the direction is the opposite of what we predicted, and the proposed mechanism is not
supported.** We expected the gate mechanism of panel (b) — inactive units hiding long-range
structure, so the Jacobian would *under*-report. It over-reports instead. The per-node gap
correlates with local gate inactivity in inconsistent directions (Spearman `−0.28` against dead
message-gates, `+0.64` against dead pointwise units, `−0.34` against attention support), which is
not the signature of a single gating story.

**What is actually happening** is visible in the distance-resolved attenuation, panel (b): the
ratio of finite to tangent influence is `2.06` at the swapped node itself and falls monotonically
to `0.87` by seven hops. A realistic swap is *super*-linear locally and *sub*-linear at range —
plausibly because a long path crosses more ReLU boundaries, so signed contributions partially
cancel. The response profile is more locally concentrated than the tangent's, which shortens the
measured range. The tangent measure therefore overstates this model's effective range by about
1.6×.

**The task-derived reference mildly favours carriage**: Spearman `+0.77` against the task's own
required range `(d(v,S) + d(v,T))/2` versus `+0.69` for the Jacobian. With 16 graphs and that
margin, treat it as suggestive.

The honest summary is that the first synthetic result (a gap exists at realistic perturbation
sizes) replicates on a model and task neither of which was chosen to produce it, while the second
(which direction, and why) did not. One model, one seed, one task — this is a single replication,
not a survey.

## Beneficial carriage, and a ground-truth test that had to be rebuilt

`fig7_beneficial.png`, `run_beneficial.py`. Every measure above is **label-free** — the Jacobian
range, `F_sens` and the shell probe all answer "did the output move". Beneficial carriage answers
"did the answer get worse", so it brings the one thing the study lacked: an external criterion.
Carriers are the pre-head states `h^L`, the readout is the exact nonlinear head, and the loss is
the training objective (BCE with the same capped positive weight). All paths converged with a
worst completeness residual of `7.7e-5`, and `B`'s per-source mass reproduces the directly measured
loss increase to `3e-5` — the section 6 identity holds.

`B` is signed, so it cannot go through the expected-distance formula. The reportable objects are
§7's accumulations, `S_B(b)` and `B_far(r)`.

**Where the task loss is carried.** In distribution, `8.1%` of the signed loss mass sits beyond four
hops; out of distribution (64–96 nodes, where graph-exact collapses to 0.004) that rises to
**`30.6%`**, and the `8+` bin goes from `0.008` to `1.795`. The model's task-relevant dependence
genuinely reaches further on the larger graphs it fails on.

These sums are **inverse-probability weighted**, and they have to be. The source cap force-includes
both marks, which carry roughly 8× the per-source signed mass of an ordinary node — and it bites
asymmetrically, capping 2/12 graphs in distribution against 12/12 out of it. Unweighted, the same
run reports `26%` rather than `30.6%` and overstates the mean event loss increase by 44%. AUROC is
unaffected (only negatives are sampled); the accumulations are not.

**A ground-truth test that did not survive contact.** The intended test was to score each measure
by how well it ranks the two marked endpoints above ordinary nodes as intervention sources. It
gave a clean-looking ordering — raw `F_sens` 0.975, Beneficial 0.935, event-normalised `F_sens`
0.171, Jacobian 0.019 — and it is **confounded**. A mark's `h0` row is 6.1× further from a typical
donor row than an ordinary node's, so **donor dose alone scores AUROC 1.000**. Any measure that
tracks perturbation magnitude scores high for free, and the ranking is not evidence of
faithfulness. It is reported here because the failure is instructive, not because it stands.

**The rebuilt test.** Restricting to ordinary nodes — which all share the same token embedding, so
the payload confound is gone — and asking which measure predicts the *actual* loss increase from
swapping that node:

| measure | Spearman with true loss increase |
|---|---|
| Functional `F_sens` (un-normalised) | **+0.801** |
| donor dose alone (model-free baseline) | +0.588 |
| Functional (event-normalised) | +0.328 |
| Jacobian influence | **−0.011** |

Raw `F_sens` beats the model-free baseline by a clear margin, so it carries genuine model-dependent
information about which node replacements will hurt. The Jacobian carries **none** — its
correlation with realised task damage is indistinguishable from zero. Beneficial carriage is
excluded from this table on purpose: by the completeness identity `B` *is* the loss change, so it
would score 1.000 by definition. `B` does not predict task damage; it localises it.

**Why event normalisation hurts here.** Dividing by dose is exactly right when the goal is to
recover an operator's influence profile (that is what makes the linear replication exact). It is
wrong when the perturbation magnitude is itself part of what makes a node important, which is the
case whenever payload classes differ — as they do for a mark versus an ordinary node. The two
variants answer different questions and neither is the default.

## Head-to-head on a quantised dependence

`fig8_quantised.png`, `run_quantised.py`. One operator family, a known answer, all three methods
used as their own specifications intend.

On a cycle of 24 nodes with scalar i.i.d. features, `F(X)_v = a·x_v + b·g_τ(x_{v+k})` with
`a = b = 1`, `k = 4`, `g_τ(z) = tanh(z/τ)`. Mass sits at distance 0 and `k` and nowhere else; `τ` is
the only knob, from linear to a sign step whose derivative is zero almost everywhere while the
functional dependence is undiminished. The cycle removes boundary corrections and scalar features
remove the `L1`-versus-`L2` channel confound.

**The reference.** A first-order spread decomposition of the operator by Monte Carlo, from the
operator definition alone: `w_0 = |a|·spread(x)`, `w_k = |b|·spread(g_τ(x))`,
`range = k·w_k/(w_0+w_k)`. Any spread that is positively homogeneous of degree 1 and
translation-invariant reduces this **exactly** to the paper's `ρ̂` for linear `g`, so the reference
agrees with the tangent measure precisely where the tangent measure is provably right — the linear
control puts truth and every arm at `2.000`. **Three** such conventions are reported (standard
deviation, mean absolute deviation, Gini mean difference), because each is some measure's own
sufficient statistic and picking one would decide the winner by definition.

**Two axes, not one — and this is where the first version of this experiment was wrong.** The
paper prescribes a mean of *per-node ratios* (Table 1); carriage averages donor events *before*
taking the ratio, and that inner average is what suppresses a Jensen collapse. Separating them at
`τ = 0.05` (95% graph-bootstrap intervals, 60 graphs, truth `1.88`–`2.20`):

| | ratio per node | ratio pooled / donor-averaged |
|---|---|---|
| **tangent** | **0.287** [0.228, 0.345] | 1.382 [1.013, 1.734] |
| **finite** | 1.257 [1.225, 1.288] | 1.727 [1.685, 1.767] |

Both axes contribute about equally: aggregation moves the tangent by **4.8×**, and switching
tangent→finite at matched aggregation moves it by **4.4×**. Averaging the *derivative* over
donor-perturbed inputs does not help (`0.388`) — it is the derivative and the per-node ratio
compounding, not a shortage of input samples.

**Mean absolute error over the sweep** (hops, including the linear control):

| arm | sd | mad | gmd |
|---|---|---|---|
| `√S_B` (Beneficial) | *0.047* | **0.087** | 0.080 |
| `F_sens` raw | 0.051 | 0.136 | **0.030** |
| `F_sens` event-normalised | 0.095 | 0.201 | 0.057 |
| tangent, pooled | 0.207 | 0.338 | 0.179 |
| **tangent, per node (as specified)** | **0.687** | **0.818** | **0.659** |

**What is established.** The paper's prescribed estimator is last under all three conventions by
5–20×, and every finite-perturbation variant beats it — that is robust to the convention and the
gap is far outside the intervals. **What is not:** any ordering *among* the finite variants. `√S_B`
wins under sd, but that column is circular — `√S_B` is algebraically *identical* to the sd
reference here, not merely similar — and raw `F_sens` wins under gmd, which is its own sufficient
statistic. Each measure wins under the convention shaped like itself.

**Three further caveats, all from the audit.** The tangent sweep is **not** monotone: `τ = 2` sits
below `τ = 1` because the far derivative is capped at `1/τ`, and the truth drops with it. The
reported completeness residual (`4e-16`) and convergence are **vacuous here** — an MSE loss on an
affine path is a degree-1 integrand, Gauss–Kronrod is exact, and the adaptive bisection never ran
(`intervals = 1` on every path), so this experiment says nothing about the integrator. And because
the targets are the operator's own outputs plus mean-zero noise, every swap hurts: `B` carries no
sign information here and degenerates to a squared functional-carriage readout, so it is not
independent evidence on this task.

## A far gate controlling a near value

`fig9_gate.png`, `run_gate.py`. The most informative test in the study, because the nonlinearity is
realistic rather than adversarial and the failure it exposes is structural.

`F(X)_v = a·x_v + b·σ(x_{v+j}/τ)·x_{v+k}` on a cycle, with the **gate at distance 8** and the value
it gates **at distance 3**. Multiplicative gating is the mechanism behind attention and gated
message passing, so this is a mainstream computation, not a constructed cliff; and the nonlinearity
is an *interaction*, which is a blind spot for any first-order method by definition rather than by
design. `τ → ∞` collapses the gate to the constant ½ and the map becomes linear — the anchor, where
every method returns `1.500`.

A far node deciding *whether* near information is used is exactly the "is this task long-range?"
question, and the two ground truths disagree violently about it: the gate's **first-order Sobol
index is essentially zero** (its mean effect vanishes because the value it gates is mean-zero)
while its **total-effect index is ~0.29**. A strictly first-order view puts the range at `1.71`;
the true total-effect range is `3.54`.

**Seeing the gate at `τ = 0.05`** (share of mass placed at distance 8; truth `0.260`–`0.287`
depending on convention, 95% graph-bootstrap intervals):

| method | gate share |
|---|---|
| `√S_B` | 0.280 [0.270, 0.290] |
| `F_sens` event-normalised | 0.274 [0.261, 0.283] |
| `F_sens` raw | 0.254 [0.243, 0.266] |
| **Jacobian, pooled** | **0.189 [0.137, 0.238]** |

The Jacobian's interval **excludes both total-effect truths**; all three carriage variants reach
them. It under-detects the far interaction node by roughly 30%. Note it does far better than
first-order theory predicts (0.033) because it picks the gate up through the gate's *own*
derivative — so this is not a claim that the Jacobian is blind to interactions, only that it
systematically under-weights them.

**Mean absolute range error over the sweep** (hops), under three references:

| method | total (variance) | total (absolute) | first-order |
|---|---|---|---|
| `√S_B` | *0.035* | 0.160 | 1.035 |
| `F_sens` event-normalised | 0.150 | 0.045 | 0.920 |
| `F_sens` raw | 0.201 | **0.006** | 0.870 |
| Jacobian, pooled | 0.308 | 0.114 | **0.762** |

**What is established.** Raw `F_sens` beats the pooled Jacobian under *both* total-effect
conventions (`0.201` vs `0.308`, `0.006` vs `0.114`), and the Jacobian's gate share is the only one
whose interval misses the truth. **What is not:** that `√S_B` is best. It wins the variance column
by construction — a squared-error loss allocation is algebraically what a variance-based total
index measures — and it *loses to the Jacobian* under the absolute convention (`0.160` vs `0.114`).
Each measure wins under the reference shaped like itself, which is why three are reported.

**And the Jacobian wins the first-order column**, as a first-order method should. That column is
the honest statement of what is really at stake: if "range" means the first-order decomposition,
the tangent is the right tool and everything else is biased; if it means total effect — which is
what the long-range question actually asks, since a gate that is invisible to a mean effect still
decides whether distant information is used — the tangent under-reports and the finite measures do
not. The experiment does not settle which definition is correct; it shows the choice is
consequential and quantifies it.

**Caveat carried over.** The path integrator is still not exercised (`intervals = 1`, residual
`2.7e-15`): an MSE loss on an affine path is a degree-1 integrand and Gauss–Kronrod is exact, so
`√S_B` here is a closed-form quantity, not a test of the quadrature.

## Saturated long-range pathway with local counterflow

`fig10_counterflow.png`, `run_counterflow.py`. A path graph with one source `s`, a near carrier at
`d=1` and a far carrier at `d=D`. The source feature is `b ∈ {−1,+1}` and

```
h_1 = −γ·b                    local pathway, linear
h_D = (1+γ)·φ_κ(b)            far pathway, saturating
φ_κ(b) = tanh(κb)/tanh(κ)     φ_κ(±1) = ±1 exactly, for every κ
ŷ = h_1 + h_D                 sum pooling
```

Because `φ_κ(±1) = ±1` **exactly for all κ**, the map on its data is `ŷ = b` and a donor swap flips
it to `−b` regardless of `κ`. The finite response is invariant to saturation while the far
pathway's derivative `(1+γ)φ'_κ(b)` vanishes: `κ` changes neither the function's values on the data
nor the task, only the tangent. With `±1` payloads, §2.1 eligibility rule 2 *guarantees* the flip —
the production donor law delivers exactly the required intervention, verified on every event.

**The exact reference.** The input is binary, so the donor swap is the *only* possible change to
it: the finite response profile is the complete functional dependence and its expected distance is
the exact range, with nothing left for a derivative to add. (It is computed from `F_sens`, so it
cannot be used to score `F_sens` — it is the reference for the tangent arms.)

**Both orientations of the Jacobian are reported**, because they behave completely differently and
showing only one would be unfair:

At `κ = 8`, `D = 10`, varying the pathway balance `γ`:

| γ | exact range | source-anchored `ρ` | **carrier-anchored `ρ̂` (the paper's own)** |
|---|---|---|---|
| 0.25 | 8.500 | 1.000 | **5.50** |
| 1.0 | 7.000 | 1.000 | **5.50** |
| 3.0 | 6.143 | 1.000 | **5.50** |

The source-anchored reading — the orientation in which `F_sens` and `B` are natively defined, and
the only one in which the three measures are comparable — collapses to `1.000` (precisely
`1 + 6.5e-5`, not exactly 1; it reaches 1 only as `κ → ∞`).

**The paper's own carrier-anchored `ρ̂` is not fooled by saturation — but only because it is
uninformative here.** Every estimable output node has exactly one input, so its per-node ratio
collapses to that distance and the graph mean is `(1 + D)/2` for **every** `γ` and **every** `κ`.
It returns `5.50` while the true range moves from `8.50` to `6.14`. That is the more interesting
finding than the collapse: on this construction their estimator is constant by construction, so it
cannot be wrong about saturation and equally cannot be right about anything else.

**What the finite measures do.** `F_sens` reports `F(1) = 2γ`, `F(D) = 2(1+γ)` — both pathways,
correct dominance, invariant to `κ`. `B` additionally separates them by *sign*: at `γ=1` the far
pathway is beneficial (`+4`) while the local pathway actively **counteracts** it (`−2`), and their
sum `+2` is exactly the donor-swap loss increase, as §6's completeness identity requires. All 315
cells match their closed forms: worst deviation `3.3e-14` for the Jacobian (traced to `tanh`
backward cancelling `1 − tanh²` at `κ = 8`), `0.0` for `F_sens`, `2.6e-7` for `B` at the
tolerance-limited off-dyadic cells.

**The target-alignment control** (`y = τb`) is the cleanest label-dependence result here. Both
Jacobian orientations and `F_sens` are numerically unchanged across `τ`; `B` scales exactly
linearly with it, vanishes at `τ = 0` — functionally active, zero task benefit — and inverts at
`τ < 0`.

**It also exercises the path integrator, but only at off-dyadic `τ`, and that distinction is the
point.** The L1 kink sits at `α = (1+τ)/2`. A kink landing on a Gauss–Kronrod panel centre is
odd-symmetric about it, so both embedded rules cancel to zero, the error estimate is exactly zero
and the scheme reports convergence **without refining**: `τ ∈ {1, 0.5, 0, −0.5}` use 1–2 intervals
even though `τ = 0`'s kink is strictly interior. Off-dyadic `τ ∈ {±0.3, 0.7}` use **22–23
intervals** with residual `~5e-8` — genuine refinement. This is the only result in the study that
is evidence about the quadrature rather than about a closed form, and it also shows the interval
count is not a reliable kink-localisation diagnostic on its own.

## Files

| file | what it does |
|---|---|
| `rangelib.py` | topologies, the three operators, both measures, the registered bootstrap |
| `run_replication.py` | Figure 3: range vs `k`, four task variants, two measures |
| `run_donor_sweep.py` | how the two carriage variants behave in `K`: the event-normalised one is `K`-independent, the raw one converges to a donor-scale-biased limit |
| `run_learned.py` | the same comparison on trained approximators |
| `run_divergence.py` | dose interpolation, step, oscillation and channel-count sweeps |
| `run_graphlevel.py` | graph-level task: Hessian range against first-order carriage |
| `run_counterflow.py` | saturated far pathway with local counterflow: closed-form check of all three measures, plus the target-alignment control |
| `run_gate.py` | the far-gate interaction test: Sobol ground truth, pooled Jacobian, `F_sens`, registered `beneficial_carriage` |
| `run_quantised.py` | the head-to-head on a quantised dependence, against a Monte-Carlo ground truth |
| `run_beneficial.py` | Beneficial carriage on both splits: `S_B`, `B_far`, and the source-ranking tests |
| `run_realmodel.py` | the follow-up on a trained MarkedTreePath model: dose ladder, mechanism test, task-derived reference |
| `figures.py` | all figures |
| `verify_reference.py` | checks the hard-coded published values against the vendored CSVs in `reference/` |
| `test_rangelib.py` | exactness, readout equivalence, bootstrap-path and policy checks |

Production code is imported, not reimplemented: `functional_carriage_events`,
`build_channel_events`, `SemanticDonorPool`, `semantic_donor_swap`, `shortest_path_distances`,
`trimmed_mean` and `BootstrapPolicy` all come from
`src/graph_specialisation_metrics/methodology/`. A `TinyData` duck type supplies the `x` /
`edge_index` / `clone()` surface those paths need without pulling in PyG or GRIT.

## Protocol notes and caveats

* **Bootstrap levels.** Every node of every graph is used as a source, so the source level is
  exhaustively enumerated and is held fixed — the methodology §7 exception, which is sufficient on
  its own: the estimand is conditional on the enumerated sources. Graphs and donors are resampled
  (donors independently *within* each source, matching the production nesting); 2,000 percentile
  replicates; 20% trimmed mean across graphs.

  There is a second, task-dependent reason to prefer the exception here, worth stating because it
  is easy to over-generalise: the range is a ratio over sources, and on the k-Power family —
  whose per-carrier mass concentrates on few sources — resampling them shifts the interval clear of
  the point estimate entirely. On k-Dirac and k-Rectangle it does not. `test_rangelib.py` records
  both the shift and the counterexamples, so it reads as a diagnostic rather than a law.
* **Graph pooling.** Carrier-uniform (`mean_i ρ̂^F_i`), matching the paper's mean over nodes. The
  mass-weighted alternative `Σ_{i,s} F d / Σ_{i,s} F` overweights high-throughput carriers and is a
  different quantity.
* **`d = 0` is included** in both measures, as in the paper. It is typically the largest single
  term, so any cross-paper number must state whether it is in.
* **Readout filtering.** With a non-identity readout, `g_out_i` can annihilate source-specific
  response components, and `F_sens` then measures *output-relevant* range — a readout-filtered `ρ̂`.
  That is intended, but it means carriage on a real model is not numerically comparable to a raw
  Jacobian range unless the readout is stated.
* **Donor dependence.** Carriage is defined relative to a donor law. A different pool, dose or
  matching tier is a different measurement; the paper's measure has no such parameter (and, by
  panel (a), corresponds to the `α → 0` corner of ours). The `α < 1` points of that panel are a
  deliberately **off-protocol dose diagnostic**: the interpolated payload is synthetic, supplied by
  no real donor node, and its recorded dose is rescaled locally. Only `α = 1` is a registered
  semantic donor-swap event.
* **Zero-width intervals on the linear and graph-level panels.** For a linear operator the
  event-normalised carriage field is exactly `|L_is|`, independent of the payloads and of the donor
  draw, so both measures return the same number on every graph and the §7 bootstrap collapses to
  zero width by construction. The same holds for the Hessian side of the graph-level task, whose
  second derivative is input-independent on a single shared topology. The bands in `fig1_replication.png` therefore carry no information; the intervals
  that do are in `fig3_learned.png` and `fig4_divergence.png`, where the map is nonlinear.
* **The two bands are not width-comparable.** The Jacobian and Hessian ranges have no donor level,
  so their intervals resample graphs only, while the carriage interval resamples graphs and donors.
  Read each band against its own estimator, not against the other.
* **Published-value provenance.** `reference/` vendors the three CSVs the comparison is made
  against, with their source and checksums; `verify_reference.py` asserts the values hard-coded in
  `run_replication.PUBLISHED_GRID` still match them (worst transcription gap `4.8e-7`).
* The topology, operators and `k` range follow the paper's own experiment scripts: `16 × 16` grid
  (256 nodes), `k = 1…8`, SPD, and the implemented map `F(X) = MᵀX` (their task reduces over the
  first axis), which is what reproduces their published k-Rectangle values.
