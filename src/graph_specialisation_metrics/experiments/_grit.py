"""Load the GRIT transformer layer."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import types
import warnings
from pathlib import Path

from . import ExperimentSetupError


def _install_scatter_fallback() -> None:
    try:
        import torch_scatter  # noqa: F401
    except ImportError:
        from torch_geometric.utils import scatter as pyg_scatter

        fallback = types.ModuleType("torch_scatter")

        def scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
            value = pyg_scatter(
                src,
                index,
                dim=dim,
                dim_size=dim_size,
                reduce="sum" if reduce == "add" else reduce,
            )
            if out is not None:
                out.copy_(value)
                return out
            return value

        fallback.scatter = scatter
        fallback.scatter_add = lambda src, index, dim=0, out=None, dim_size=None: scatter(
            src, index, dim, out, dim_size, "sum"
        )
        fallback.scatter_max = lambda src, index, dim=0, out=None, dim_size=None: (
            scatter(src, index, dim, out, dim_size, "max"),
            None,
        )
        sys.modules["torch_scatter"] = fallback


def _package_path() -> Path:
    distribution = importlib.metadata.distribution("graphgps")
    for file in distribution.files or ():
        if file.parts[:2] == ("grit", "__init__.py"):
            path = Path(distribution.locate_file(file)).parent
            if path.is_dir():
                return path
    fallback = Path(distribution.locate_file("grit"))
    if fallback.is_dir():
        return fallback
    raise importlib.metadata.PackageNotFoundError("the graphgps distribution has no GRIT package")


def grit_transformer_layer():
    """Load ``GritTransformerLayer`` without unused package imports."""

    _install_scatter_fallback()
    try:
        expected = str(_package_path())
        loaded = sys.modules.get("grit")
        if loaded is None or expected not in tuple(getattr(loaded, "__path__", ())):
            for name in [
                name for name in sys.modules if name == "grit" or name.startswith("grit.")
            ]:
                del sys.modules[name]
            package = types.ModuleType("grit")
            package.__path__ = [expected]
            sys.modules["grit"] = package
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="To use GraphGym, install")
            return importlib.import_module("grit.layer.grit_layer").GritTransformerLayer
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ExperimentSetupError(
            "GRIT is unavailable. Install the graphbench or mixed extra."
        ) from exc


def layer_config(*, signed_sqrt: bool):
    """Return the GRIT layer configuration used in the dissertation."""

    from yacs.config import CfgNode as CN

    return CN(
        {
            "update_e": True,
            "bn_momentum": 0.1,
            "bn_no_runner": False,
            "rezero": False,
            "attn": {
                "use": True,
                "deg_scaler": True,
                "use_bias": False,
                "clamp": 5.0,
                "act": "relu",
                "edge_enhance": True,
                "sqrt_relu": False,
                "signed_sqrt": bool(signed_sqrt),
                "scaled_attn": False,
                "no_qk": False,
                "graphormer_attn": False,
                "norm_e": True,
                "O_e": True,
            },
        }
    )
