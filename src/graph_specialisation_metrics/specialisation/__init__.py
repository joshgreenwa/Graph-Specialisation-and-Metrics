"""Per-head semantic / structural specialisation scores for pretrained GRIT models.

The productionised, central version of the specialisation-score methodology documented in
``experiments/synthetic/specialisation/SPECIALISATION_SCORES.md`` -- lifted from the spec-lite
``Net`` onto the REAL GRIT transformer (dense ZINC + its 1-hop control). It reuses the carriage
package for model loading, the semantic donor swap (``carriage.content``) and the structural
transposition (``carriage.structural``); the new part is reading those interventions at the
per-head TRANSPORT site  o^{lh}_i = ``batch.wV``  (and, secondarily, the SELECTION site
``batch.attn``).

Notebooks clone this repo and call ``specialisation.run(...)``, so edits here propagate:

    from graph_specialisation_metrics.specialisation import run
    run(tasks=["zinc", "zinc_1hop"])

Layout:
    model.py         GritHeadModel: load a checkpoint (mirrors carriage.grit_runner) + per-head
                     capture/ablation hooks (wV transport, attn selection)
    scores.py        per-head S_sem / S_str (transport) + S_attn_sem (selection), + select_heads
    causal_mediation.py held-out channel-specific noising/denoising, controls and inference
    effective_transport.py exact routing/value reconstruction and effective transport treatments
    paper_figures.py two compact, failure-aware headline figures (PNG + PDF)
    ablation.py      causal head-ablation vs random-head null, per-graph, feature correlations
    attention_viz.py per-head attention maps across molecules
    figures.py       legacy appendix deliverables + cross-model scatter
    colab.py         default paper workflow plus explicit legacy reversion; collates on Drive
"""

from __future__ import annotations

__all__ = [
    "run",
    "score_model",
    "run_ablation",
    "select_heads",
    "collect_attention",
    "run_channel_causal_mediation",
    "run_effective_transport",
]


def run(*args, **kwargs):
    """Lazy proxy to specialisation.colab.run (keeps ``import specialisation`` torch/GRIT-free)."""
    from .colab import run as _run
    return _run(*args, **kwargs)


def score_model(*args, **kwargs):
    from .scores import score_model as _f
    return _f(*args, **kwargs)


def run_ablation(*args, **kwargs):
    from .ablation import run_ablation as _f
    return _f(*args, **kwargs)


def select_heads(*args, **kwargs):
    from .scores import select_heads as _f
    return _f(*args, **kwargs)


def collect_attention(*args, **kwargs):
    from .attention_viz import collect_attention as _f
    return _f(*args, **kwargs)


def run_channel_causal_mediation(*args, **kwargs):
    from .causal_mediation import run_channel_causal_mediation as _f
    return _f(*args, **kwargs)


def run_effective_transport(*args, **kwargs):
    from .effective_transport import run_effective_transport as _f
    return _f(*args, **kwargs)
