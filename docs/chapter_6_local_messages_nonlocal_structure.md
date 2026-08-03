# Chapter 6 direction: Local Messages, Non-local Structure

**Status:** proposed replacement for the current carriage-led Chapter 6  
**Working title:** *Where Does Global Information Enter Graph Transformers?*  
**Primary task:** ZINC  
**Primary endpoint:** held-out predictive performance  
**Supporting measurement:** semantic Functional carriage, used only to distinguish response propagation from precomputed structural context

## 1. Decision summary

The strongest remaining direction is to separate two notions that are currently conflated in sparse-versus-dense Graph Transformer comparisons:

1. **communication locality:** which nodes may exchange learned messages in one layer; and
2. **preprocessing locality:** how much of the graph can affect the structural encodings supplied to those messages.

A one-hop attention mask guarantees local message support. It does **not** guarantee local information processing when the model receives multi-hop Relative Random Walk Probabilities (RRWP). Even on a self-pair or molecular bond,

\[
\operatorname{RRWP}_{ij}=[I,P,P^2,\ldots,P^K]_{ij}
\]

contains statistics of walks extending away from that pair. A one-hop GRIT model can therefore perform bond-local communication conditioned on non-local structural information.

The central Chapter 6 question should be:

> When a one-hop GRIT matches dense GRIT, is direct all-pairs communication genuinely dispensable, or has multi-hop structural preprocessing already supplied the missing non-local information?

This is a stronger question than asking whether dense and sparse checkpoints happen to develop different carriage or head-score profiles. It directly manipulates the information available to the architecture and evaluates the consequence using test performance.

## 2. Why this direction survives the earlier failures

The previous Chapter 6 hypotheses tried to infer a stable architectural explanation from fitted-model internals. That inference is unreliable because performance-equivalent models can implement different solutions and distribute redundant computation differently.

The proposed experiment avoids that inference:

- it does not predict a particular head organisation;
- it does not require individual-head causal effects to be stable;
- it does not interpret post-hoc use as task necessity;
- it does not assume that dense and sparse models must converge to the same functional solution; and
- it uses an architectural intervention and held-out performance as the primary evidence.

The conclusion is limited to empirical architectural sufficiency or benefit. Failure of a restricted model may reflect representation, optimisation, or inductive bias; it must not be called proof of task necessity.

## 3. The existing observation and its revised interpretation

The current ZINC result is:

- one-hop GRIT with short/local RRWP and one-hop GRIT with multi-hop RRWP have very similar semantic and structural carriage profiles;
- nevertheless, the multi-hop-RRWP model performs substantially better.

This is not evidence that carriage has failed. Semantic carriage perturbs node content while keeping RRWP fixed. Multi-hop RRWP can improve how the model interprets and weights local semantic evidence without making the semantic perturbation travel farther.

The appropriate interpretation is therefore:

> Learned response reach and the structural context available to local computation are different properties.

Carriage measures the former. RRWP computation horizon controls the latter.

## 4. Why the naive sparse-versus-dense comparison is confounded

Official sparse and dense GRIT do not differ only in attention support:

- sparse GRIT exposes RRWP on self and attended local pairs;
- dense GRIT exposes RRWP on every node pair.

Changing from sparse to dense attention therefore adds both:

1. direct all-pairs content communication; and
2. distant pairwise structural coordinates.

A simple comparison between the existing one-hop checkpoint and official dense GRIT cannot attribute a performance difference to dense communication.

There are two additional language and implementation constraints:

- A ten-layer one-hop model has up to ten rounds of iterative bond-local propagation, and graph-level sum pooling is global. It should be described as using **iterative local propagation**, not as a wholly local model.
- RRWP up to order \(K\) is **multi-hop**, not automatically global. The analysis must report the graph-diameter distribution and the fraction of graphs for which \(K\) covers the diameter before using the term *global RRWP*.

## 5. Clean experimental design

### 5.1 Core factorial

The primary experiment crosses message support with precomputed RRWP horizon while holding the location of RRWP injection fixed.

| Model arm | Message support | RRWP horizon | RRWP pair support | Scientific role |
|---|---|---|---|---|
| A. Local messages, short RRWP | bonds + self | \(I,P\) | self/bond pairs | strictly local-context control |
| B. Local messages, multi-hop RRWP | bonds + self | \(I,\ldots,P^K\) | self/bond pairs | local communication with non-local structural context |
| C. Dense messages, short RRWP | all pairs | \(I,P\) | self/bond pairs; neutral token elsewhere | isolates direct all-pairs content exchange |
| D. Dense messages, multi-hop local-pair RRWP | all pairs | \(I,\ldots,P^K\) | self/bond pairs; neutral token elsewhere | clean dense counterpart to arm B |
| E. Official dense GRIT | all pairs | \(I,\ldots,P^K\) | all pairs | full-pair RRWP reference, outside the core factorial |

All nonlocal pairs in arms C and D must receive the same neutral initial pair representation. They may carry content messages, but they must not receive pair-specific RRWP coordinates. Arms A--D then isolate:

- direct all-pairs communication;
- multi-hop structural preprocessing on locally related pairs; and
- their interaction.

Arm E separately measures the additional value of supplying distant pairwise structural coordinates.

### 5.2 What the RRWP-horizon factor means

The existing short-RRWP control jointly removes higher-order:

- diagonal/node RRWP features; and
- RRWP features on local pairs.

The primary factor should therefore be named **precomputed RRWP horizon**, rather than being attributed specifically to node or pair conditioning.

If the effect is strong, add a ZINC-only decomposition:

1. multi-hop diagonal/node RRWP with short local-pair RRWP; and
2. short diagonal/node RRWP with multi-hop local-pair RRWP.

This decomposition is secondary and should not be run unless the main horizon effect is stable.

### 5.3 Controls that must be audited

Keep constant across the core arms:

- parameter count, hidden dimension, depth and number of heads;
- graph pooling and prediction head;
- optimiser, learning-rate schedule, epoch budget and checkpoint selection;
- train/validation/test examples and seeds;
- node and bond feature encoders; and
- the dimensionality of the RRWP encoder, including when higher-order channels are zeroed.

Two support-dependent implementation details require particular care:

1. **Attention dropout.** The same dropout probability removes a much larger fraction of the useful neighbourhood of a low-degree sparse node than of a dense node. The clean primary comparison should preferably use zero attention dropout. Official dropout can be restored as a sensitivity experiment.
2. **Pair-state normalisation.** Normalising over \(E+N\) sparse pair states versus \(N^2\) dense pair states can itself change optimisation. Use support-independent normalisation if practical, or explicitly audit and report the limitation.

## 6. Primary estimands and inference

Let \(L_{s,h}\) denote test loss for message support \(s\in\{\text{local},\text{dense}\}\) and RRWP horizon \(h\in\{\text{short},\text{multi-hop}\}\).

The primary interaction is

\[
I_{\mathrm{sub}}=
\left(L_{\text{local,short}}-L_{\text{local,multi-hop}}\right)
-
\left(L_{\text{dense,short}}-L_{\text{dense,multi-hop}}\right).
\]

A positive \(I_{\mathrm{sub}}\) means that multi-hop structural preprocessing benefits local message passing more than dense message passing, consistent with partial substitution between structural preprocessing and direct communication.

Also report:

- the dense-support effect at each RRWP horizon;
- the RRWP-horizon effect at each support level;
- the difference between arms D and E, isolating full-pair structural coordinates;
- training, validation and test curves; and
- parameter count, runtime and peak memory.

Use paired seeds where possible. Five seeds per ZINC arm is ideal; three is the minimum for a dissertation-level conclusion. Define an equivalence margin before examining final test results so that near-equal performance is tested rather than inferred from overlapping error bars.

Training curves provide a limited but useful distinction:

- a persistent training-loss gap shows that the restricted configuration was not equally fitted under the training protocol;
- similar training fit but worse validation/test performance is more consistent with an inductive-bias or generalisation difference.

Neither case proves representational necessity.

## 7. Lightweight synthetic control

Construct a connected graph with two marked degree-two candidate nodes whose \(L\)-hop neighbourhoods are identical. One candidate lies on a long cycle and the other on a long chain. Each candidate carries an independently sampled semantic value, and the target is the value attached to the candidate on the cycle.

The task cleanly separates:

- **local semantic evidence:** the two candidate values; and
- **non-local structural context:** which candidate belongs to the cycle.

The candidates must be sufficiently far from the cycle closure or distinguishing junction that arm A cannot distinguish them within its effective depth. Higher-order return probabilities can expose cycle membership to arm B. Dense content exchange provides a separate possible solution in arms C and D.

The synthetic has only two purposes:

1. verify that the proposed controls manipulate the intended capabilities; and
2. demonstrate concretely how local semantic evidence can be selected using non-local structural context.

It must not be used as evidence that the real ZINC models implement the same algorithm.

Before training this synthetic, include an even simpler deterministic check: alter an edge outside a selected pair's local neighbourhood and show that its \(I,P\) features remain fixed while at least one higher-order \(P^k\) feature changes. This directly demonstrates why mask locality does not imply RRWP-information locality.

## 8. Interpretation of possible outcomes

| Result | Defensible interpretation |
|---|---|
| B matches E while A is worse | Multi-hop structural preprocessing allows local messages to recover much of official dense-GRIT performance. |
| C rescues A | Direct all-pairs content access can compensate for short structural preprocessing. |
| Multi-hop RRWP helps local support much more than dense support | Precomputed structural context partially substitutes for direct all-pairs communication. |
| Dense support helps under both horizons | Direct all-pairs communication provides an independent empirical benefit. |
| E outperforms D | Distant pair-specific RRWP coordinates contribute beyond dense content communication alone. |
| Arms A--D are equivalent | The benchmark does not empirically distinguish a need for either mechanism under the tested training protocol. |
| RRWP helps both support levels equally | Multi-hop RRWP is useful, but there is no evidence of substitution with dense communication. |

The last two outcomes are scientifically valid but less likely to support a headline contribution unless they replicate across task families with tight equivalence bounds.

## 9. Role of the existing intervention methodology

### 9.1 Functional carriage

Carriage should no longer lead the chapter. Its narrow supporting question is:

> Does opening direct content communication change the distance over which semantic perturbations produce output-relevant representation changes?

One semantic-carriage figure across the four core arms may be useful. It must be interpreted only as learned response propagation, not task necessity or ground-truth information use.

The strongest combined observation would be:

> Performance changes with precomputed structural horizon even when semantic response reach changes little, demonstrating that response propagation and structural-context scope are distinct.

### 9.2 Specialisation scores

Aggregate semantic/structural score summaries may be reported descriptively after the performance result is established. They should not be used to predict which architecture must win, and no individual-head claims are required.

### 9.3 Analyses to remove from the main argument

- interaction carriage or output-modulation \(M\);
- large catalogues of one-hop, two-hop, virtual-node and dense checkpoints;
- per-head architecture comparisons;
- Beneficial carriage unless independently validated; and
- claims that additional carriage is necessary or helpful to the task.

The finite-carriage comparison with prior Jacobian range work can remain as a short methodological result or appendix, but it should not carry the chapter's scientific claim.

## 10. Staged execution and kill criteria

### Stage 0: implementation audit

1. Verify exactly which RRWP fields are visible on local and nonlocal pairs in every existing model.
2. Verify that nonlocal pair features in arms C and D are identical neutral tokens.
3. Check parameter counts and forward-pass equivalence at initialization where expected.
4. Plot graph diameter against RRWP horizon and model depth.
5. Run the remote-edge RRWP sensitivity demonstration.

### Stage 1: one-seed ZINC pilot

Train the missing dense-neutral-pair arms C and D. Reuse existing arms A, B and E only as exploratory references if their dropout and normalisation policies differ; do not treat that mixed comparison as final evidence.

Proceed if:

- the existing A-versus-B performance difference remains material;
- the new cells provide a coherent attribution rather than numerical instability; and
- model health and training curves show no obvious implementation failure.

### Stage 2: clean ZINC experiment

Run arms A--E under the controlled primary protocol for at least three paired seeds, preferably five. Predeclare the equivalence margin and primary interaction before evaluating the final test set.

Proceed to a second task only if the ZINC effect is stable.

### Stage 3: replication

Replicate the central contrast on QM9 HOMO--LUMO gap. A larger-diameter molecular benchmark such as Peptides-struct would provide a stronger test of preprocessing horizon if compute allows. Do not expand to many tasks before the ZINC mechanism is secure.

### Kill the chapter direction if

- the A-versus-B difference disappears across seeds;
- results depend on support-specific dropout or normalisation choices;
- the support-by-horizon interaction is unstable and no arm is equivalent within a useful bound;
- the only positive result is that adding more RRWP features improves one model; or
- interpretation requires returning to individual-head or necessity claims.

## 11. Proposed chapter structure

### 6.1 What does local mean in a Graph Transformer?

Define communication support, preprocessing horizon and global graph readout as distinct sources of information access.

### 6.2 RRWP makes local attention structurally non-local

Explain the dependency of \(P^k\), report graph-diameter coverage, and show the deterministic remote-edge example.

### 6.3 A controlled semantic-on-structure task

Use the lightweight synthetic to verify the distinction between local semantic evidence and non-local structural context.

### 6.4 Separating communication and structural preprocessing on ZINC

Present the clean factorial, official dense reference, performance, equivalence tests and training curves.

### 6.5 Learned reach is not structural-context horizon

Use at most one carriage analysis to show why similar response reach does not imply equivalent structural information.

### 6.6 Replication and implications

Report the second task and discuss what sparse-versus-dense benchmark comparisons do and do not establish.

## 12. Candidate contribution statement

> We distinguish communication locality from structural-preprocessing locality in Graph Transformers. Through a parameter-matched intervention on attention support, RRWP computation horizon and RRWP injection support, we test whether precomputed multi-hop structure substitutes for direct all-pairs communication. The results show which source of non-local information accounts for the performance of sparse and dense GRIT models, while intervention-based carriage separately characterises the distance over which fitted models propagate semantic responses.

The final sentence must be rewritten to match the observed result. In particular, the chapter must not claim that dense communication or multi-hop structure is necessary unless the experimental design genuinely establishes necessity, which it currently does not.

## 13. Position relative to prior work

The broad claims that positional encodings affect GT performance and that sparse GRIT can perform competitively are already established. Relevant references include:

- [Ma et al., *Graph Inductive Biases in Transformers without Message Passing*](https://proceedings.mlr.press/v202/ma23c.html), which introduces GRIT and demonstrates the expressive role of RRWP; and
- [Grötschla et al., *Benchmarking Positional Encodings for GNNs and Graph Transformers*](https://arxiv.org/abs/2411.12732), which benchmarks positional encodings across GNNs, dense GTs and sparsified GRIT.

The intended contribution is narrower:

> Existing sparse-attention comparisons do not isolate information locality because locally attended pairs may contain structural features computed from remote topology. The proposed experiment separates direct communication, preprocessing horizon and distant pairwise structural coordinates.

This narrower claim is both more defensible and more closely connected to the dissertation's semantic--structural framing.

## 14. Ideas considered but not selected

### Prospective shortcut auditing

Potentially high impact, but the current head scores measure division of labour rather than total channel reliance, and the registered structural donor swap leaves topology and attention support fixed. A clean shortcut study would require new calibration, new data generators and strong baselines, making it effectively a separate project.

### Two-source finite interaction

The exact two-source difference can separate pre-pooling non-additivity from readout curvature, but paired molecular donor swaps are questionable counterfactuals. Ten-layer local models also have overlapping receptive fields on most ZINC and QM9 molecules. A real-model distance law is unlikely to be stable enough for the chapter headline.

### Robustness to semantic or structural information loss

This would have a clear task-risk endpoint, but primarily studies robustness to engineered feature or PE masking rather than the dissertation's central question about dense attention and structural context.

### Underspecification and response algebra

These accurately describe performance-equivalent solutions, but risk making the final chapter a methodological limitation rather than a substantive graph-learning result.

### Intervention-diverse ensembles or benchmark coresets

These provide practical endpoints, but depend on uncertain incremental gains over simpler prediction-diversity or graph-stratification baselines and are less connected to the central scientific gap.

