# Model Fidelity Audit

This note fixes the implementation policy for the GraphBench algorithmic HPC
runs. The local runner is acceptable for dataset, PE-cache, SLURM, checkpoint,
and metric plumbing, but its dense model classes are not faithful enough to be
used as paper models under the names Graphormer, GraphGPS, GRIT, or GNN+.

## Source Repositories Checked

Pin official sources when building the training environment:

| Model family | Repository | Checked commit |
| --- | --- | --- |
| Graphormer | `https://github.com/microsoft/Graphormer.git` | `ac154fe4253d076a1c294f14be20dad0351cff3c` |
| GraphGPS | `https://github.com/rampasek/GraphGPS.git` | `28015707cbab7f8ad72bed0ee872d068ea59c94b` |
| GRIT | `https://github.com/LiamMa/GRIT.git` | `6c988ea600a606fbb49a2246c64a2d37396b3ab5` |
| GNN+ | `https://github.com/LUOyk1999/tunedGNN-G.git` | `0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28` |

## Verdict

Paper runs should use official-backed model implementations wherever possible.
This repository should own only the GraphBench-to-PyG conversion, PE
precompute/cache, algorithmic heads, loss/metric code, checkpointing, W&B, and
SLURM orchestration.

Do not report the current local dense classes in `bin/algoreas_hpc.py` as
faithful Graphormer, GraphGPS, GRIT, or GNN+ models. If they are kept at all,
label them as smoke-test or style implementations.

## Required Official Model Paths

### Graphormer

Use the official Graphormer modules:

- `GraphNodeFeature`
- `GraphAttnBias`
- `GraphormerGraphEncoder`
- `GraphormerGraphEncoderLayer`

The data adapter must reproduce the official preprocessing:

- `convert_to_single_emb`
- `attn_bias`
- `attn_edge_type`
- `spatial_pos`
- `edge_input` generated from shortest paths
- `in_degree` and `out_degree`
- graph token attention bias

Important detail: multi-hop edge bias is not just a direct edge-value projection.
Official Graphormer encodes shortest-path edge sequences with `edge_encoder`,
then applies `edge_dis_encoder` across hop distance before adding the result to
attention bias. A simplified SPD/direct-edge bias is only Graphormer-style.

### GraphGPS

Use the official GraphGPS `GPSLayer` and `GPSModel` path. The faithful
configuration for this project is:

- `gt.layer_type: CustomGatedGCN+Transformer`
- `gt.batch_norm: true`
- `gt.layer_norm: false`
- `gt.dropout: 0.1`
- `gt.attn_dropout: 0.1`
- local GatedGCN branch with edge attributes
- full global Transformer attention
- FFN expansion factor `2`, not `4`

Important detail: official GPS computes local and global branches independently
from the same input state, applies branch residuals and normalization, sums the
two branch outputs, then applies the FFN block.

RWSE should use the official kernel PE encoder semantics: raw optional
BatchNorm, MLP/linear PE encoder, and concatenation into the node embedding
budget. It should not be silently added as a projected residual unless this is
reported as an adaptation.

### GRIT

Use the official GRIT path:

- `GritTransformer`
- `GritTransformerLayer`
- `MultiHeadAttentionLayerGritSparse`
- `RRWPLinearNodeEncoder`
- `RRWPLinearEdgeEncoder`
- official RRWP transform fields `rrwp`, `rrwp_index`, `rrwp_val`, `deg`, and
  `log_deg`

Important detail: GRIT attention is not dot-product attention plus a pair bias.
Official GRIT uses sparse full-pair attention over RRWP-encoded pair edges:

- score starts from `K_h[src] + Q_h[dst]`
- edge state supplies multiplicative and additive terms through `E`
- score uses the signed square-root transform and learned `Aw`
- attention softmax is grouped by destination node
- edge state can be updated through `O_e`
- node update uses the learned degree scaler from `log_deg`
- normalization is BatchNorm-style by default

For degree, pass `deg` or `log_deg` exactly as expected by the official GRIT
layer. Do not substitute an arbitrary degree scalar node feature. The
`GraphormerDeg` encoder exists in the repo, but the official GRIT RRWP configs
do not use it by default; adding it would be an extra centrality feature.

### Static GRIT

Use the official GRIT layer path with the minimum ablation needed to isolate
pair-state evolution:

- keep RRWP node and edge encoders
- keep full-pair attention
- keep degree scaler
- keep GRIT additive/QK/edge scoring
- set `gt.update_e: false`

This is a static-edge-state GRIT ablation. A Graphormer-like GRIT ablation, for
example using GRIT's `graphormer_attn` option, should be a separate model label
because it changes the attention mechanism as well as pair-state evolution.

### GNN+

Use the official `custom_gnn` network from the GNN+ repository:

- `GCNConvLayer` for exact GCN+
- `GINEConvLayer` for GIN+
- `GatedGCNLayer` for GatedGCN+
- `GCNConvLayer` from `gcn_conv_layer_e.py` only if deliberately running an
  edge-aware `GCNE+` variant

Important details:

- official GNN+ layers use `BatchNorm1d`, not LayerNorm
- FFN expansion is `2`
- activation is the GraphGym configured activation, usually ReLU
- residual and FFN are enabled through `cfg.gnn.residual` and `cfg.gnn.ffn`
- official PyG `GINEConv` uses default epsilon behavior unless explicitly
  changed; do not call epsilon trainable unless the backend config sets it
- GatedGCN updates node and edge states with the official `A/B/C/D/E` maps and
  sigmoid gate

## Paper-Safe Implementation Boundary

Allowed local code:

- GraphBench task loading and deterministic subset selection
- conversion of GraphBench examples into PyG `Data`/`Batch`
- cached RWSE, RRWP, SPD, and Graphormer multi-hop preprocessing tensors
- algorithmic task heads and metrics
- training loop, checkpointing, W&B logging, and SLURM scripts

Not allowed for paper model claims:

- local reimplementations that alter attention equations
- replacing sparse PyG/GraphGym execution with dense approximations
- changing normalization style
- changing FFN expansion
- changing GIN epsilon behavior
- adding degree, SPD, RWSE, or RRWP to a model outside its declared policy

## Resulting PE Policy

Keep the existing fair PE policy, but attach each PE through the model-native
path:

- Graphormer: native degree/SPD/multi-hop edge attention bias
- GraphGPS: official RWSE node encoder
- GRIT/static GRIT: official RRWP node and edge encoders, no SPD
- GNN+: official RWSE node encoder when comparing against GraphGPS; label this
  as a controlled PE-enhanced GNN+ condition

