# Chapter 6 plan: From Specialisation to Semantic and Structural Reach

## Narrative role

Chapter 6 should mark a deliberate change of scale:

> Chapters 4-5 ask **which heads process semantic and structural information**. Chapter 6 asks **how far the resulting semantic and structural computation reaches, and whether architectural reach is actually used**.

The conceptual progression is:

1. donor-swaps define the semantic and structural channels;
2. specialisation localises channel sensitivity across attention heads;
3. carriage localises the same channel sensitivity across graph distance; and
4. the GNN-GNN+-GT comparison tests how communication architecture changes learned use.

Carriage is therefore not a separate interpretability theme. It is the model-level spatial counterpart of head specialisation, built from the same interventions and output geometry.

## Proposed chapter structure

### 6.1 Research objectives

1. When do finite interventions reveal dependencies missed by local gradients?
2. Do classical GNNs, GNN+ models, and Graph Transformers learn different semantic and structural reach?
3. Do differences in architectural reach correspond to learned functional or task-beneficial reach?

### 6.2 Carriage as finite-intervention range

Briefly recap Functional carriage and relate it to Bamberger et al.'s Jacobian influence range.

The key distinctions are:

- carriage uses finite donor-swaps rather than infinitesimal gradients;
- it separates semantic and structural interventions;
- it retains a complete distance profile;
- its event-normalised first moment provides a Bamberger-compatible expected-distance summary; and
- Beneficial carriage can additionally test task alignment, if validated.

### 6.3 Finite-intervention range under nonlinear saturation

Use a compact controlled experiment to establish why the finite estimator is needed.

The experiment should show:

- agreement between Jacobian range and Functional carriage in the linear regime;
- divergence when an important long-range pathway becomes locally saturated;
- recovery of the finite long-range dependency by Functional carriage; and
- optionally, separation of beneficial and adverse pathways by Beneficial carriage.

Suggested title:

> **Finite-intervention range under nonlinear saturation**

The existing replication of Bamberger et al.'s Figure 3 is useful validation but can be placed in the appendix. The nonlinear experiment should be the main-text methodological result.

### 6.4 Architectural capacity versus learned reach

The headline comparison should include:

- **classical GNN:** local message passing;
- **GNN+:** local support with modern structural encodings and transformer-style engineering; and
- **Graph Transformer:** global communication support.

Report:

1. predictive performance;
2. semantic Functional-carriage profiles;
3. structural Functional-carriage profiles; and
4. an event-normalised expected-distance summary for comparison with prior range work.

Structural profiles should be described as the SPD-dependence of structural-payload sensitivity. They do not uniquely identify a message-passing path, because pairwise positional encodings may be directly available to distant receivers.

### 6.5 Implications for the GNN-GT debate

The chapter should distinguish architectural capacity from learned use. Possible findings have clear interpretations:

| Finding | Interpretation |
|---|---|
| GNN+ and GT match in performance and carriage | Dense communication capacity is available but not materially used for the task. |
| GNN+ and GT have similar total reach but different semantic/structural composition | Structural information may substitute for long-range semantic transport. |
| Classical GNN differs while GNN+ matches GT | Modern encodings or engineering, rather than dense attention, explain the improvement. |
| Only GT has additional far beneficial carriage and better performance | Evidence that useful global communication contributes to the performance gap. |
| GT has additional far Functional carriage but not additional Beneficial carriage | Dense attention creates long-range activity without additional demonstrated task benefit. |

The central question is not simply which architecture can communicate globally, but which architecture learns to use semantic and structural information at distance.

## Role of Beneficial carriage

Beneficial carriage should be a predeclared secondary analysis until its interpretation and empirical value are fully validated.

- Promote it into the headline comparison if it materially separates active reach from task-beneficial reach.
- If it closely mirrors Functional carriage, report that result briefly and place the complete analysis in the appendix.
- If it is not sufficiently validated, present it as future work rather than expanding the main methodology.
- If computed, retain and disclose the complete results even when they are not headline findings.

The strongest use would be to distinguish:

> additional long-range response from additional long-range task benefit.

## Paper-worthiness criteria

Carriage becomes a material contribution if the empirical results establish at least one of:

1. similar overall reach but different semantic/structural composition;
2. additional GT Functional carriage without additional Beneficial carriage; or
3. a performance gap associated with additional far Beneficial carriage.

Finite swaps alone are an incremental methodological change. The larger contribution is using finite, channel-resolved reach to alter or refine the interpretation of the GNN-GT comparison.

## Candidate contribution statement

> We introduce interventional, channel-resolved carriage as a finite-intervention measure of learned semantic and structural reach. Applying it across classical GNNs, GNN+ models, and Graph Transformers distinguishes architectural communication capacity from the distance-dependent computation actually learned for the task.

A stronger result-dependent version would be:

> Dense attention capacity does not imply useful long-range computation: performance-matched GNNs and Graph Transformers learn similar task-relevant reach, potentially through different semantic and structural pathways.

## Space-efficient presentation

The chapter can be driven primarily by two multi-panel figures:

1. **Synthetic validation:** linear agreement, nonlinear saturation, and finite-carriage recovery.
2. **Three-way architecture comparison:** performance, semantic reach, structural reach, and expected-distance summaries.

The surrounding text should focus on hypotheses, estimands, and interpretation rather than repeating information visible in the plots.
