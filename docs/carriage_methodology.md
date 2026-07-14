# Carriage Methodology

This document is the source of truth for the **carriage** measurement: what it
estimates, the two estimators we have implemented (finite **swap** and
**integrated gradients**), the samples and assumptions each relies on, and the
decision about which is the core method. It is written to be readable by a new
reader; every symbol is defined where it first appears.

The reference implementation is Step 7 of
`grit_intervention_procedure.py` (`carriage_swap`, the IG carriage routine, and
`symbolic_structural_carriage_rows`), mirrored in the standalone synthetic
`synthetic_bottleneck_retrieval.py` (`_swap_carriage`, `_ig_carriage`,
`_structural_ig_joint`).

---

## 1. What we are estimating

### 1.1 The object: carriage

We have a trained GRIT-style graph transformer \(f\). It reads a graph \(G\) with
\(n\) nodes. Each node \(i\) carries two kinds of input:

- **Content** \(x_i\) — the symbolic features of the node (in the retrieval
  synthetic, the content-address / value; in a molecule, the atom type). This is
  the *symbolic* substrate.
- **Structure** — the random-walk positional encoding (RRWP). We split this into
  two objects:
  - **node-RRWP** \(p_i \in \mathbb{R}^{K}\): the diagonal of the RRWP tensor, a
    vector over walk lengths \(r = 1,\dots,K\). It is node \(i\)'s *own* global
    positional fingerprint — "where does this node sit in the graph at every
    scale." This is the structural signal that short-range message passing
    truncates.
  - **pair-RRWP** \(p_{ij} \in \mathbb{R}^{K}\): the off-diagonal entries, the
    *relative* structural encoding between a source \(j\) and another node. It is
    "how is \(j\) positioned relative to \(i\)."

The model encodes these, runs \(L\) attention layers, and produces final node
states \(h^L_i \in \mathbb{R}^{D}\), which a readout head \(\phi\) maps to a
prediction. We fix one **focal / carrier** node \(i\) (in the synthetic, the
single query node \(q\)) and a scalar **readout summary** \(S\) of its
prediction (defined in §1.2).

> **Carriage** \(C[i, j]\) is the contribution that source node \(j\)'s substrate
> — its content, its node-RRWP, or its pair-RRWP — makes to the readout summary
> \(S\) at the focal node \(i\).

The name is literal: it measures how much task-relevant information is *carried*
from a distant source \(j\), across the graph, to the focal node \(i\). Because
\(f\) is nonlinear and the layers mix information, \(j\)'s substrate can only
affect \(S\) by propagating through the attention stack; carriage quantifies the
size of that propagated effect.

We always stratify carriage by:

- **distance** \(d(i,j)\): the geodesic hop distance from focal to source. The
  scientific interest is the **far band** (large \(d\)), where a sparse (e.g.
  1-hop) bottleneck must compress many hops into few and where dense global
  attention should have an advantage.
- **factor**: content vs node-RRWP vs pair-RRWP — the symbolic/structural
  decomposition.
- **scale** \(r\) (structure only): which random-walk length carries the effect.
  High-\(r\) mass is exactly the global structure that local RRWP truncation
  discards.

### 1.2 The two readout summaries: functional and beneficial

We measure carriage under two definitions of \(S\), because "the model uses this
information" and "this information helps the model get the answer right" are
different questions.

- **Functional** \(S_{\text{fun}}\): the correct-class logit at the focal node,
  \(S_{\text{fun}} = z_{y}(i)\), where \(z\) is the logit vector and \(y\) the
  true label. This measures what the model actually *computes* at \(i\),
  regardless of whether it is correct — the mechanistic footprint of the source.
- **Beneficial** \(S_{\text{ben}}\): the cross-entropy loss at the focal node,
  \(S_{\text{ben}} = \mathrm{CE}\big(z(i), y\big)\), signed so that a **negative**
  carriage is **loss-reducing** = genuinely helpful. This measures what improves
  the prediction — the information whose carriage the task actually rewards.

Beneficial carriage is the headline for the sparsification story: the claim is
that *far-range beneficial* carriage survives under dense attention and collapses
under a 1-hop bottleneck. Functional carriage is the mechanistic control.

### 1.3 The samples

The unit of estimation is a **(focal node, source node)** pair within a single
graph. For each probe graph we compute carriage for the focal node against every
admissible source \(j\) (distance \(\ge\) `min_distance`, \(j \ne i\)), for each
factor and mode. We repeat over many probe graphs (`carriage_graphs_count`) and,
for the swap estimator, over multiple donor draws per source
(`carriage_donor_samples`). The reported quantity is a **mean finite effect** per
`(model, graph type, distance band, factor, mode)` cell, with the spread across
graphs and donors giving an honest sampling uncertainty. We compute the identical
suite for each model under comparison (dense GRIT, masked 1-hop, expander,
V-node) so that the numbers are directly comparable — the whole design is a
*contrast across models and graph types*, not an absolute number.

### 1.4 OOD risk (why this is delicate)

Every carriage estimator must construct a **counterfactual**: to measure \(j\)'s
contribution we must evaluate \(f\) on an input where \(j\)'s substrate is
*altered*. The danger is that the altered input is **out of distribution (OOD)**
— unlike anything in the training data — so the model's response reflects
extrapolation artefacts rather than the mechanism we care about. A carriage
number computed on an OOD input is only as trustworthy as the model's behaviour
off-manifold, which is exactly where trained networks are least reliable.

The two estimators take on OOD risk in different places and in different amounts;
this is the crux of the comparison in §3. Note one structural subtlety up front:
the full RRWP tensor is the random-walk profile of one *actual* adjacency matrix,
so it is highly constrained (symmetry, the walk recursions, a spectral/PSD
structure). Any edit to *part* of it — one node's long-range channels — produces
a tensor that is **not the walk profile of any graph**. Structural counterfactuals
are therefore intrinsically harder to keep on-manifold than content ones.

---

## 2. The two estimators

Both run the true nonlinear model — neither linearises the message passing. They
differ only in (a) how they define the counterfactual reference and (b) how they
aggregate over it.

### 2.1 Estimator (1): finite swap

**Idea.** Replace source \(j\)'s substrate with a **donor** node's real substrate,
run the model, and read off the shift in the focal node's readout, projected onto
the frozen readout direction.

Let \(g = \partial S / \partial h^L\big|_{\text{clean}}\) be the **readout
gradient** frozen at the clean state — a linearised readout that is nonzero at the
carrier. Let \(h^L(\text{clean})\) be the clean final states and
\(h^L(j\!\leftarrow\!\text{donor})\) the final states after swapping \(j\)'s
substrate for a donor's. The swap carriage is

\[
C^{\text{swap}}[i,j] \;=\; \big\langle\, g,\; h^L(\text{clean}) - h^L(j\!\leftarrow\!\text{donor}) \,\big\rangle,
\]

averaged over several donors (and, in the core, restricted to different-type
donors so the swap is a meaningful contrast). For the **content** factor the donor
is another node's encoded content. For **structure**, we replace only the
long-range channels (\(r \ge\) `channel_start`) of \(j\)'s node-RRWP diagonal, or
of the pair-RRWP entries incident to \(j\), preserving the local channels.

- **What it estimates:** the finite, contrastive effect on \(S\) of substituting
  \(j\)'s *actual* substrate for a typical alternative — "how much does this
  node's real content/structure matter versus a plausible replacement." This is a
  discrete do-style intervention with an approximately on-manifold replacement.
- **Frozen \(g\):** the estimator is first-order in the *final-layer* shift (it
  misses curvature of the head for large shifts) but **exact** in the propagation
  from the perturbed substrate through all \(L\) layers to \(h^L\).
- **Cost:** one forward pass per (source × donor). No backward pass. Embarrassingly
  parallel across sources, donors, and graphs.

### 2.2 Estimator (2): integrated gradients (IG)

**Idea.** Fade \(j\)'s substrate in from a featureless **baseline** to its clean
value along a straight path, accumulating the gradient of \(S\) along the way.

Let \(z_j\) be \(j\)'s substrate (encoded content, or its RRWP channels), with
baseline \(z_j^{\text{base}}\) = the graph-mean of that substrate (mean encoded
content over nodes; per-channel mean RRWP over the active node/pair slots). Along
the path \(z(\alpha) = z^{\text{base}} + \alpha\,(z^{\text{clean}} - z^{\text{base}})\),

\[
C^{\text{IG}}[i,j] \;=\; \int_0^1 \Big\langle\, \frac{\partial S}{\partial z_j}\big(z(\alpha)\big),\; z_j^{\text{clean}} - z_j^{\text{base}} \,\Big\rangle \; d\alpha,
\]

discretised into `ig_steps` Riemann steps. Its defining property is
**completeness**:

\[
\sum_j C^{\text{IG}}[i,j] \;=\; S(\text{clean}) - S(\text{base}),
\]

validated numerically to \(\sim 10^{-5}\) on the synthetic. The structural variant
(`_structural_ig_joint`) integrates over *all* walk-length channels and returns a
`joint[source, scale]` object that telescopes exactly; its by-distance and
by-scale marginals are the two axis-sums.

- **What it estimates:** each source's **additive share** of the total readout
  change relative to the featureless baseline — an attribution/decomposition, not
  a contrast against a real alternative.
- **Cost:** `ig_steps` forward+backward passes. Deterministic (no sampling
  variance), but one number per source per graph.

---

## 3. Theoretical comparison

We compare on six axes. The recurring theme: swap answers the causal-contrast
question the science asks and stays closer to the data manifold; IG answers a
budgeting question one step removed and pays for it with a fully off-manifold
path.

### 3.1 Which question does it answer?

The scientific claim is causal and finite: *does altering a far source's
information change the focal prediction, and does the 1-hop bottleneck destroy
that effect while dense attention preserves it?* The swap estimator **is** that
contrast — a finite intervention with a real alternative. IG answers "what share
of the total readout, measured from a featureless average, does this source
explain?" — a decomposition that is only indirectly about the bottleneck. **Edge:
swap.**

### 3.2 OOD / manifold risk (the crux)

- **Swap, content:** the donor's content is a real node's, so the content marginal
  stays on-manifold. The mild violation is a content–structure *mismatch* (real
  content in \(j\)'s structural slot). Multi-donor averaging and different-type
  partners keep the contrast representative. **Low OOD.**
- **Swap, structure:** editing only \(j\)'s long-range RRWP channels breaks
  walk-profile consistency — the perturbed tensor is not the RRWP of any graph.
  Preserving local channels and using donor values bounds the damage, but the
  combination is off-manifold. **Moderate OOD, intrinsic to structural edits.**
- **IG, any factor:** the graph-mean baseline corresponds to no real node, and the
  *entire interpolation path* lies off-manifold. Completeness is a mathematical
  identity that holds regardless, but the per-source attribution integrates
  gradient behaviour through a region the model never saw in training; for RRWP
  every intermediate point is an invalid walk profile. **High OOD, spread along
  the whole path.**

Both are compromised for structure (§1.4), but the *shape* of the compromise
differs decisively: swap's OOD is a **single, bounded, interpretable** point ("vs
a real donor"); IG's is a **path integral through the featureless middle** with no
physical meaning. **Edge: swap**, strongly for content and modestly for structure.

### 3.3 Additivity / completeness

IG's one clear win. \(\sum_j C^{\text{IG}} = S(\text{clean}) - S(\text{base})\)
lets you make budget statements — "far sources account for \(X\%\) of the readout"
— and the scale/joint decomposition inherits the exact telescoping. Swap has **no
conservation law**: each swap is an independent counterfactual and they do not
compose into a single total. You can rank and compare swap effects but not budget
them. **Edge: IG.**

### 3.4 Statistical power / estimation across many samples

Swaps are a natural **sampling estimator**: draw many donors × source–focal pairs
× graphs, and the mean finite effect is the estimand while the spread is honest
uncertainty. This is exactly the design in §1.3 and scales cheaply (forward-only,
parallel). IG produces one deterministic number per source per graph; to obtain a
*distribution* you must vary baselines, which reintroduces off-manifold choices
and is far less natural. For a claim that lives or dies on tight, comparable error
bars across many models and graph types, **edge: swap.**

### 3.5 Cost

Swap is forward-only, one pass per (source, donor); IG is `ig_steps`
forward+backward passes per source. At the sweep scale (many graphs × many models
× two modes × three factors) swap is roughly an order of magnitude cheaper, which
directly buys more samples and tighter estimates. **Edge: swap.**

### 3.6 Bias/variance and diagnostics

- Swap: variance from donor sampling (controlled by averaging); bias from frozen
  \(g\) (curvature of the head under large shifts). No built-in self-check.
- IG: near-zero variance; bias from finite steps, *measured for free* by the
  completeness residual — a genuine built-in diagnostic.

**Edge: IG on self-diagnosis**, but note the completeness residual only certifies
the integral was computed accurately, **not** that the off-manifold attribution is
meaningful. It is a numerical check, not a validity check.

---

## 4. Decision

**The finite swap is the core carriage methodology. IG is retained as a
validation cross-check and as the tool for completeness/budget claims only.**

Rationale, in one line: the scientific claim is a finite, comparative, causal
contrast measured across many samples and models — the swap estimator *is* that
contrast, it stays closest to the data manifold (cleanly for content, with a
bounded caveat for structure), it carries honest sampling uncertainty, and it is
cheap enough to run at the scale the contrast needs. IG's completeness is elegant
but answers a decomposition question one step removed from the claim, and pays for
it with a fully off-manifold interpolation path that is especially ill-defined for
RRWP.

Concretely:

- **Report** swap carriage (functional and beneficial, all three factors) as the
  primary result. The headline is far-band **beneficial** carriage by graph type
  (dense vs 1-hop vs expander).
- **Cross-check** with IG on an agreement scatter (already emitted). Where swap and
  IG agree we have converging evidence from two different OOD assumptions; where
  they disagree, prefer swap and flag the source as a candidate off-manifold
  artefact.
- **Use IG only** where the swap literally cannot answer the question — an exact
  budget ("far sources explain \(X\%\)") or the exact walk-length (scale) telescoped
  decomposition.

### 4.1 Symbolic vs structural: which structural object

Node-RRWP is a node's own global fingerprint (what local message passing
truncates); pair-RRWP is relative source–focal structure. In the content-addressed
retrieval task, pair-RRWP carriage is ~0 **by design** (the task is addressed by
content, not by relative position), and the structural niche appears as **high-\(r\)
node-RRWP scale mass**. So the structural headline should feature the **node-RRWP
scale marginal**, not pair-RRWP. Both are still emitted for completeness and as a
negative control.

### 4.2 Honest caveat and future work

For the **structural** factor neither estimator is fully on-manifold, because any
per-node RRWP edit breaks walk-profile consistency (§1.4). We therefore:

1. frame structural swap carriage as a **structural-sensitivity probe**, not a
   clean causal do-intervention, and lean on the (cleaner) content result for the
   central sparsification claim; and
2. flag the principled fix as future work: instead of editing RRWP channels
   directly, perform an actual **graph edit** (rewire / node relabel) at the source
   and *recompute* RRWP, so every counterfactual is a valid walk profile. This is
   more expensive and perturbs distances, but it is the only way to make structural
   carriage strictly on-manifold. A cheaper intermediate is a **whole-node** swap
   (donor's entire RRWP row/column rather than only long-range channels), which is
   less surgical but internally more consistent than a partial-channel edit.
