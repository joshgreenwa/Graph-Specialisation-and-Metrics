"""GritTaskSpec: everything that differs between GRIT-trained models.

The carriage methodology (semantic intervention -> C -> F(d)/B(d)/B_far(k)) is identical
across GRIT tasks. What differs is only: which GRIT config built the checkpoint, how many
parameters it should have, where its checkpoints live on Drive, what the node content is,
and what a "healthy" test metric looks like. A GritTaskSpec captures exactly that, so a
new task is a registry entry rather than a new script.

Carriage *correctness* preconditions (sum-pooling readout, RRWP invariant to content,
regression target) are NOT hard-coded per task -- they are checked on the loaded cfg at
run time by ``grit_runner.check_carriage_preconditions``, so they hold for any task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .content import ContentAdapter, FullNodeContentAdapter
from . import metrics

# Official GRIT (Ma et al.), pinned to the commit the training runners used.
OFFICIAL_GRIT_REPO = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"


@dataclass
class GritTaskSpec:
    """Describes one GRIT-trained model for carriage analysis.

    Attributes:
        name:            short id, used for the Drive collation subfolder (e.g. "zinc").
        title:           human label for figure captions.
        config_path:     GRIT-repo-relative YAML config (e.g. configs/GRIT/zinc-GRIT-RRWP.yaml).
                         Mutually exclusive with config_text.
        config_text:     inline YAML (for parameter-matched variants that aren't in the repo).
                         Written to <out>/task_config.yaml at run time.
        expected_params: exact param count to assert (None = skip; still logged).
        drive_dir:       default Drive dir holding results/ and datasets/ for this task
                         (the same --drive-dir the training runner used).
        dataset_dir:     optional dataset directory override. Use this when several training
                         runs share one dataset cache outside ``drive_dir/datasets``.
        paper_metric:    ("mae", 0.059)-style (name, value) for the load sanity check; None skips.
        metric_name:     display name for a recomputed metric when there is no paper reference.
        metric_fn:       (preds[N,T], trues[N,T]) -> float recomputed over the eval split.
                         Default: MAE. Peptides-func uses multilabel mean-AP.
        metric_higher_better: True for AP (abort if below threshold), False for MAE (abort if above).
        metric_abort:    threshold for the bad-load guard, in the metric's own units.
        content_adapter: how to read/write swappable node content (default: whole x row).
        env_hooks:       callables (repo_dir) applied after GRIT clone, before loaders build
                         (e.g. peptides RDKit + dataset/RRWP patches).
        grit_repo/commit: GRIT source to clone/pin. Default official.
        node_content_desc: label for logs ("atom type", ...).
    """

    name: str
    title: str
    config_path: Optional[str] = None
    config_text: Optional[str] = None
    expected_params: Optional[int] = None
    drive_dir: str = ""
    dataset_dir: Optional[str] = None
    paper_metric: Optional[tuple] = None
    metric_name: Optional[str] = None
    metric_fn: Callable = staticmethod(metrics.mae_metric)
    metric_higher_better: bool = False
    metric_abort: float = 0.15
    content_adapter: ContentAdapter = field(default_factory=FullNodeContentAdapter)
    env_hooks: tuple = ()
    grit_repo: str = OFFICIAL_GRIT_REPO
    grit_commit: str = OFFICIAL_GRIT_COMMIT
    grit_repo_dir: Optional[str] = None   # own clone dir when a task patches GRIT source
    node_content_desc: str = "node content"

    def __post_init__(self):
        if bool(self.config_path) == bool(self.config_text):
            raise ValueError(
                f"task {self.name!r}: set exactly one of config_path / config_text "
                f"(got path={self.config_path!r}, text set={bool(self.config_text)})."
            )


# ---------------------------------------------------------------------------------------
# Registry of known tasks. Add a new GRIT model here; nothing else needs to change.
# ---------------------------------------------------------------------------------------
TASKS: dict[str, GritTaskSpec] = {}


def register(spec: GritTaskSpec) -> GritTaskSpec:
    TASKS[spec.name] = spec
    return spec


def get_task(name: str) -> GritTaskSpec:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]


def resolve_dataset_dir(spec: GritTaskSpec, drive_dir: Optional[str] = None) -> str:
    """Resolve the dataset cache without assuming it lives beside task results."""
    if spec.dataset_dir:
        return str(Path(spec.dataset_dir))
    root = drive_dir or spec.drive_dir
    if not root:
        raise ValueError(f"task {spec.name!r} has no dataset_dir or drive_dir")
    return str(Path(root) / "datasets")


# Official dense GRIT+RRWP on ZINC-subset -- the reproduced reference (scalar regression).
register(GritTaskSpec(
    name="zinc",
    title="GRIT+RRWP ZINC-subset (dense)",
    config_path="configs/GRIT/zinc-GRIT-RRWP.yaml",
    expected_params=473_473,
    drive_dir="/content/drive/MyDrive/grit_zinc_official",
    paper_metric=("mae", 0.059),
    metric_fn=staticmethod(metrics.mae_metric),
    metric_higher_better=False,
    metric_abort=0.15,
    node_content_desc="atom type",
))


def _onehop_hooks():
    """Deferred so importing tasks stays torch/GRIT-free; called at run time."""
    from . import onehop_env

    def hook(repo_dir):
        onehop_env.apply_onehop_patch(repo_dir)

    return (hook,)


def _onehop_localrrwp_hooks():
    """Deferred exact reconstruction hook for the strictly local 1-hop control."""
    from . import onehop_env

    def hook(repo_dir):
        onehop_env.apply_onehop_localrrwp_patch(repo_dir)

    return (hook,)


def _khop_hooks(hops: int, global_vnode: bool):
    """Deferred hook that replays the exact k-hop (+ optional VNode) training patch.

    Reuses the packaged ``grit_patches.khop_zinc.apply_khop_patch`` function that trained these
    checkpoints, so the analysis architecture matches training exactly. Torch/GRIT-free at import
    time.
    """
    from . import khop_env

    return (khop_env.make_khop_hook(hops, global_vnode),)


def _peptides_hooks():
    """Deferred so importing tasks stays torch/GRIT-free; called at run time."""
    from . import peptides_env

    def hook(repo_dir):
        peptides_env.ensure_repo_root_on_path()
        peptides_env.install_peptides_deps()
        peptides_env.apply_peptides_patches(repo_dir)

    return (hook,)


def _qm9_hooks(
    attention: str,
    hops: int = 1,
    *,
    global_vnode: bool = False,
):
    """Deferred checkpoint-compatible reconstruction hook for a QM9 gap model."""
    from . import qm9_env

    return (
        qm9_env.make_qm9_hook(
            attention=attention,
            hops=hops,
            global_vnode=global_vnode,
        ),
    )


# Parameter-matched 1-hop GRIT+RRWP on ZINC-subset (sparse control; same scalar regression).
register(GritTaskSpec(
    name="zinc_1hop",
    title="GRIT+RRWP ZINC-subset (1-hop masked)",
    # Written into the GRIT clone by the 1-hop patch hook, then resolved from there.
    config_path="configs/GRIT/zinc-GRIT-RRWP-1hop.yaml",
    expected_params=473_473,       # parameter-matched to dense ZINC
    drive_dir="/content/drive/MyDrive/grit_zinc_1hop",
    paper_metric=None,             # a locality-restricted control; MAE is model-dependent
    metric_fn=staticmethod(metrics.mae_metric),
    metric_higher_better=False,
    metric_abort=0.6,              # catches an unloaded checkpoint (MAE ~ target std); a
                                   # trained 1-hop ZINC control is well under this
    env_hooks=_onehop_hooks(),
    grit_repo_dir="/content/GRIT_zinc_1hop",  # own clone: the patch edits GRIT source
    node_content_desc="atom type",
))


# ---------------------------------------------------------------------------------------
# QM9 HOMO-LUMO gap controls, reconstructed by the packaged training-compatible patch. Both models
# use the identical target/features/split contract and differ only in attention support.
# The training runs deliberately share one PyG QM9 cache outside either results directory.
# ---------------------------------------------------------------------------------------

register(GritTaskSpec(
    name="qm9_gap_dense",
    title="GRIT+RRWP QM9 HOMO-LUMO gap (dense)",
    config_path="configs/GRIT/qm9-gap-GRIT-RRWP.yaml",
    expected_params=472_769,
    drive_dir="/content/drive/MyDrive/grit_qm9_gap_dense",
    dataset_dir="/content/drive/MyDrive/grit_qm9_gap_data",
    paper_metric=None,
    metric_name="MAE (eV)",
    metric_fn=staticmethod(metrics.mae_metric),
    metric_higher_better=False,
    # A trained checkpoint is expected to be far below this; an uninitialised model is not.
    metric_abort=0.5,
    env_hooks=_qm9_hooks(attention="dense", hops=1),
    grit_repo_dir="/content/GRIT_qm9_gap_dense",
    node_content_desc="atomic number",
))


register(GritTaskSpec(
    name="qm9_gap_1hop",
    title="GRIT+RRWP QM9 HOMO-LUMO gap (1-hop masked)",
    config_path="configs/GRIT/qm9-gap-GRIT-RRWP.yaml",
    expected_params=472_769,
    drive_dir="/content/drive/MyDrive/grit_qm9_gap_1hop_real",
    dataset_dir="/content/drive/MyDrive/grit_qm9_gap_data",
    paper_metric=None,
    metric_name="MAE (eV)",
    metric_fn=staticmethod(metrics.mae_metric),
    metric_higher_better=False,
    metric_abort=0.5,
    env_hooks=_qm9_hooks(attention="khop", hops=1),
    grit_repo_dir="/content/GRIT_qm9_gap_1hop",
    node_content_desc="atomic number",
))


register(GritTaskSpec(
    name="qm9_gap_1hop_vnode",
    title="GRIT+RRWP QM9 HOMO-LUMO gap (1-hop masked + global VNode)",
    config_path="configs/GRIT/qm9-gap-GRIT-RRWP.yaml",
    expected_params=472_833,
    drive_dir="/content/drive/MyDrive/grit_qm9_gap_1hop_vnode",
    dataset_dir="/content/drive/MyDrive/grit_qm9_gap_data",
    paper_metric=None,
    metric_name="MAE (eV)",
    metric_fn=staticmethod(metrics.mae_metric),
    metric_higher_better=False,
    metric_abort=0.5,
    env_hooks=_qm9_hooks(
        attention="khop",
        hops=1,
        global_vnode=True,
    ),
    grit_repo_dir="/content/GRIT_qm9_gap_1hop_vnode",
    node_content_desc="atomic number",
))


# Parameter-matched 1-hop ZINC control with RRWP itself truncated to local information.
register(GritTaskSpec(
    name="zinc_1hop_local",
    title="GRIT+RRWP ZINC-subset (1-hop masked, local-only RRWP)",
    # Written into the GRIT clone by the same patch function used during training.
    config_path="configs/GRIT/zinc-GRIT-RRWP-1hop-localrrwp.yaml",
    expected_params=473_473,       # parameter-matched to dense and standard 1-hop ZINC
    drive_dir="/content/drive/MyDrive/grit_zinc_1hop_localrrwp",
    paper_metric=None,             # locality-restricted control; MAE is model-dependent
    metric_fn=staticmethod(metrics.mae_metric),
    metric_higher_better=False,
    metric_abort=0.6,
    env_hooks=_onehop_localrrwp_hooks(),
    # Separate clone: this source patch differs from both dense and standard 1-hop GRIT.
    grit_repo_dir="/content/GRIT_zinc_1hop_localrrwp",
    node_content_desc="atom type",
))

# Descriptive alias used by the reach experiment and new notebooks.  Keep the
# historical ``zinc_1hop_local`` registration above so existing cached carriage
# runs and command lines remain valid.
register(GritTaskSpec(
    name="zinc_1hop_localrrwp",
    title="GRIT+RRWP ZINC-subset (1-hop masked, local-only RRWP)",
    config_path="configs/GRIT/zinc-GRIT-RRWP-1hop-localrrwp.yaml",
    expected_params=473_473,
    drive_dir="/content/drive/MyDrive/grit_zinc_1hop_localrrwp",
    paper_metric=None,
    metric_fn=staticmethod(metrics.mae_metric),
    metric_higher_better=False,
    metric_abort=0.6,
    env_hooks=_onehop_localrrwp_hooks(),
    grit_repo_dir="/content/GRIT_zinc_1hop_localrrwp",
    node_content_desc="atom type",
))


# ---------------------------------------------------------------------------------------
# k-hop / virtual-node ZINC controls, reconstructed by the packaged training-compatible patch
# (sparsity=k_hop with an exact <=k shortest-path attention mask derived from RRWP walk
# channels, and an optional learned GlobalVNode excluded from pooling). Each replays the
# exact training patch (khop_env -> grit_patches.khop_zinc.apply_khop_patch) so checkpoints load.
# Each gets its own GRIT clone because the patch writes a hops/vnode-specific config into it.
# The checkpoints live at <drive_dir>/results/_recovery_checkpoints/seed0_<name_tag>/{best,latest}.ckpt
# (env.find_checkpoint has a recovery-checkpoint fallback for this layout).
# ---------------------------------------------------------------------------------------

# 2-hop masked control (no virtual node). name_tag "ColabDrive.2hop.GRITwRRWP".
register(GritTaskSpec(
    name="zinc_2hop",
    title="GRIT+RRWP ZINC-subset (2-hop masked)",
    # Written into the GRIT clone by the k-hop patch hook, then resolved from there.
    config_path="configs/GRIT/zinc-GRIT-RRWP-khop.yaml",
    expected_params=473_473,       # k-hop masking adds no parameters
    drive_dir="/content/drive/MyDrive/grit_zinc_2hop",
    paper_metric=None,             # a locality-restricted control; MAE is model-dependent
    metric_fn=staticmethod(metrics.mae_metric),
    metric_higher_better=False,
    metric_abort=0.6,
    env_hooks=_khop_hooks(hops=2, global_vnode=False),
    grit_repo_dir="/content/GRIT_zinc_2hop",
    node_content_desc="atom type",
))


# 1-hop masked control WITH a global virtual node. name_tag "ColabDrive.1hop.GRITwRRWP.VNode".
register(GritTaskSpec(
    name="zinc_1hop_vnode",
    title="GRIT+RRWP ZINC-subset (1-hop masked + global VNode)",
    config_path="configs/GRIT/zinc-GRIT-RRWP-khop.yaml",
    expected_params=473_537,       # 473,473 + 64 for the learned VNode embedding
    drive_dir="/content/drive/MyDrive/grit_zinc_1hop_vnode",
    paper_metric=None,
    metric_fn=staticmethod(metrics.mae_metric),
    metric_higher_better=False,
    metric_abort=0.6,
    env_hooks=_khop_hooks(hops=1, global_vnode=True),
    grit_repo_dir="/content/GRIT_zinc_1hop_vnode",
    node_content_desc="atom type",
))


# 2-hop masked control WITH a global virtual node. name_tag "ColabDrive.2hop.GRITwRRWP.VNode".
register(GritTaskSpec(
    name="zinc_2hop_vnode",
    title="GRIT+RRWP ZINC-subset (2-hop masked + global VNode)",
    config_path="configs/GRIT/zinc-GRIT-RRWP-khop.yaml",
    expected_params=473_537,       # 473,473 + 64 for the learned VNode embedding
    drive_dir="/content/drive/MyDrive/grit_zinc_2hop_vnode",
    paper_metric=None,
    metric_fn=staticmethod(metrics.mae_metric),
    metric_higher_better=False,
    metric_abort=0.6,
    env_hooks=_khop_hooks(hops=2, global_vnode=True),
    grit_repo_dir="/content/GRIT_zinc_2hop_vnode",
    node_content_desc="atom type",
))


# Official dense GRIT+RRWP on Peptides-func (10-way multilabel classification, metric AP).
register(GritTaskSpec(
    name="peptides_func",
    title="GRIT+RRWP Peptides-func (dense)",
    config_path="configs/GRIT/peptides-func-GRIT-RRWP.yaml",
    expected_params=None,  # not asserted; the AP recompute is the load check
    drive_dir="/content/drive/MyDrive/grit_peptides_func_official",
    paper_metric=("AP", 0.6988),
    metric_fn=staticmethod(metrics.multilabel_ap_metric),
    metric_higher_better=True,
    metric_abort=0.40,     # a correctly loaded model is ~0.65-0.70; below 0.40 => broken load
    env_hooks=_peptides_hooks(),
    node_content_desc="OGB atom features (9)",
))


# Official dense GRIT+RRWP on Peptides-struct (11-target regression, metric MAE).
register(GritTaskSpec(
    name="peptides_struct",
    title="GRIT+RRWP Peptides-struct (dense)",
    config_path="configs/GRIT/peptides-struct-GRIT-RRWP.yaml",
    expected_params=None,  # the MAE recompute is the load check
    drive_dir="/content/drive/MyDrive/grit_peptides_struct_official",
    paper_metric=("MAE", 0.2460),
    metric_fn=staticmethod(metrics.mae_metric),  # mean abs error over the 11 targets
    metric_higher_better=False,
    metric_abort=0.40,     # a correctly loaded model is ~0.246; above 0.40 => broken load
    env_hooks=_peptides_hooks(),
    node_content_desc="OGB atom features (9)",
))
