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
from typing import Optional

from .content import ContentAdapter, TypeDictContentAdapter

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
        paper_metric:    ("mae", 0.059)-style (name, value) for the load sanity check; None skips.
        metric_sanity_threshold: abort if the recomputed metric exceeds this (bad load guard).
        content_adapter: how to read/write swappable node content (default TypeDictNode).
        grit_repo/commit: GRIT source to clone/pin. Default official.
        node_content_desc: label for logs ("atom type", ...).
    """

    name: str
    title: str
    config_path: Optional[str] = None
    config_text: Optional[str] = None
    expected_params: Optional[int] = None
    drive_dir: str = ""
    paper_metric: Optional[tuple] = None
    metric_sanity_threshold: float = 0.15
    content_adapter: ContentAdapter = field(default_factory=TypeDictContentAdapter)
    grit_repo: str = OFFICIAL_GRIT_REPO
    grit_commit: str = OFFICIAL_GRIT_COMMIT
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


# Official dense GRIT+RRWP on ZINC-subset -- the reproduced reference.
register(GritTaskSpec(
    name="zinc",
    title="GRIT+RRWP ZINC-subset (dense)",
    config_path="configs/GRIT/zinc-GRIT-RRWP.yaml",
    expected_params=473_473,
    drive_dir="/content/drive/MyDrive/grit_zinc_official",
    paper_metric=("mae", 0.059),
    metric_sanity_threshold=0.15,
    node_content_desc="atom type",
))
