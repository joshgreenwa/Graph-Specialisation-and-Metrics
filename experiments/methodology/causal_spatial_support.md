# Causal spatial support of specialised attention heads

## Question

The experiment asks:

> For a head already identified as semantically or structurally specialised,
> where in graph distance does its intervention-induced activity become relevant
> to the model output?

It separates three quantities that should not be conflated: access to nodes at a
distance, internal response at that distance, and finite output mediation through
those nodes.

## Experimental unit

For a held-out graph $G$, source node $s$, and canonical donor, we construct a
finite semantic or structural intervention $G'$. Distances are always measured
on the pristine graph:

\[
C_d(s)=\{i:d_G(s,i)=d\}.
\]

The pilot uses dense GRIT on ZINC with four held-out graphs, one source per graph,
one donor per source, and two heads per family. The full preset adds QM9 and uses
eight graphs, two sources, and three heads per family.

## Frozen head families

Families are selected once from the canonical discovery-split score cache:

- strongest semantic specialists;
- strongest structural specialists;
- high-activity, low-selectivity generalists; and
- registered same-layer central controls for each specialist family.

No output-mediation result is used to select these heads. Spatial mediation is
then evaluated on the independent causal split.

## Three spatial quantities

### 1. Direct attention access: \(A_x(h,d)\)

On the clean graph, we sum the attention sent directly from source $s$ to
receivers in $C_d(s)$, for head $h$. Family values average across selected
heads.

This answers **where the head can directly attend from the source**. Attention is
an access pattern, not evidence that the accessed information affects the output.

### 2. Internal response: \(S_x(h,d)\)

This is the canonical discovery-split, output-projected routed-response
contribution for intervention channel $x$, head $h$, and source--carrier
distance $d$. Family profiles sum the selected heads.

This answers **where intervention-responsive internal activity is observed**. It
does not establish that the activity survives downstream computation.

### 3. Finite output mediation: \(M_x(h,d)\)

Let $z_h^G(i)$ and $z_h^{G'}(i)$ be the native routed output of head $h$ at
carrier $i$ in the clean and intervened runs. For shell $C_d(s)$, we perform
two matched patches:

1. **Injection:** run $G$, replacing only $z_h^G(C_d)$ with
   $z_h^{G'}(C_d)$.
2. **Restoration:** run $G'$, replacing only $z_h^{G'}(C_d)$ with
   $z_h^G(C_d)$.

For the standardised scalar prediction $f$, event-level mediation is

\[
M_x(h,d)=\frac{1}{2}\left(
  \left|f(G;z_h(C_d)\leftarrow z'_h(C_d))-f(G)\right|
  +
  \left|f(G')-f(G';z'_h(C_d)\leftarrow z_h(C_d))\right|
\right).
\]

This is a symmetric, finite, intervention-specific measure of **realised output
mediation through a head-distance cell**. For a family, all selected heads are
patched jointly in the same shell.

## Aggregation

For each quantity, donors are averaged within source, sources within graph, and
then graphs are averaged. Each distance profile is normalised to unit mass so it
describes radial allocation rather than absolute effect magnitude. Confidence
intervals resample held-out graphs.

The headline summary also reports the proportion of mass beyond $d>2$. Because
the pilot has only four graphs, these intervals are directional rather than
confirmatory.

## Pathway concentration and overlap

Every selected head is also patched individually. For family $H$, the
distance-wise overlap diagnostic is

\[
R_x(H,d)=\frac{M_x(H,d)}{\sum_{h\in H}M_x(h,d)}.
\]

- $R\approx1$: approximately additive realised pathways;
- $R<1$: overlapping, saturating, or cancelling family pathways;
- $R>1$: synergistic joint mediation.

Ratios in shells with negligible mediation are numerically unstable and are not
substantive evidence. The effective number of long-range heads is the inverse
concentration of their raw mediated mass: $1/\sum_h p_h^2$.

## Interpretation

The central comparison is $A\rightarrow S\rightarrow M$:

- $A$ broad but $S$ narrow: available attention is not strongly used by the
  intervention response;
- $S$ broad but $M$ narrow: the head processes distant intervention effects,
  but much of that activity does not mediate the output;
- $S$ and $M$ aligned: the internal specialisation profile is spatially
  faithful to realised output computation;
- specialist $M$ differing from controls: specialisation predicts a distinct
  functional spatial role.

## What the experiment does not establish

This measures the realised mechanism of a trained model under specific finite
donor swaps. It is not a minimal task-necessity result: pathways may be redundant,
and the model is not retrained after removing them. It also does not treat
attention weights as explanations or equate internal response with task benefit.

## Outputs

The Colab frontend saves and displays:

1. radial $A$, $S$, and $M$ profiles plus long-range shares; and
2. joint-versus-individual mediation and effective long-range head counts.

Graph/channel measurement shards are cached on Drive. Setting `PHASE = "figures"`
regenerates figures from those caches without loading the model or dataset.
