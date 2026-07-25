"""Causal endpoint algebra, separate from GRIT hook execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .audit import audit_check


def _rows(value: Any) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value.reshape(value.shape[0], -1)


def clean_ablation(z_clean, z_ablated, loss_clean, loss_ablated) -> dict[str, np.ndarray]:
    clean, ablated = _rows(z_clean), _rows(z_ablated)
    if clean.shape != ablated.shape:
        raise ValueError("clean and ablated z must align")
    return {
        "prediction_movement": np.linalg.norm(clean - ablated, axis=-1),
        "loss_change": np.asarray(loss_ablated) - np.asarray(loss_clean),
    }


def donor_necessity(
    z_clean,
    z_event,
    z_clean_ablated,
    z_event_ablated,
    *,
    epsilon: float,
) -> dict[str, np.ndarray]:
    clean, event = _rows(z_clean), _rows(z_event)
    clean_a, event_a = _rows(z_clean_ablated), _rows(z_event_ablated)
    displacement = clean - event
    ablated_displacement = clean_a - event_a
    removed = displacement - ablated_displacement
    norm = np.linalg.norm(displacement, axis=-1)
    return {
        "aligned_necessity": np.sum(removed * displacement, axis=-1)
        / (norm + float(epsilon)),
        "gross_necessity": np.linalg.norm(removed, axis=-1),
        "event_effect": norm,
    }


@dataclass(frozen=True)
class PatchResponse:
    restoration_gross: np.ndarray
    injection_gross: np.ndarray
    restoration_aligned: np.ndarray
    injection_aligned: np.ndarray
    bidirectional_gross: np.ndarray
    bidirectional_aligned: np.ndarray
    event_effect: np.ndarray


def patch_response(
    z_clean,
    z_event,
    z_restored,
    z_injected,
    *,
    epsilon: float,
) -> PatchResponse:
    clean, event = _rows(z_clean), _rows(z_event)
    restored, injected = _rows(z_restored), _rows(z_injected)
    displacement = clean - event
    restore_move = restored - event
    inject_move = injected - clean
    norm = np.linalg.norm(displacement, axis=-1)
    restoration_gross = np.linalg.norm(restore_move, axis=-1)
    injection_gross = np.linalg.norm(inject_move, axis=-1)
    restoration_aligned = np.sum(restore_move * displacement, axis=-1) / (
        norm + float(epsilon)
    )
    injection_aligned = np.sum(inject_move * -displacement, axis=-1) / (
        norm + float(epsilon)
    )
    return PatchResponse(
        restoration_gross,
        injection_gross,
        restoration_aligned,
        injection_aligned,
        0.5 * (restoration_gross + injection_gross),
        0.5 * (restoration_aligned + injection_aligned),
        norm,
    )


def mismatch_adjusted_gross(
    matched: PatchResponse,
    mismatch: PatchResponse,
) -> np.ndarray:
    """G_c event contributions; aggregation remains donor->source->graph."""

    return matched.bidirectional_gross - mismatch.bidirectional_gross


def mismatch_adjusted_aligned(
    matched: PatchResponse,
    mismatch: PatchResponse,
    *,
    require_concordant: bool = True,
) -> np.ndarray:
    if require_concordant:
        valid = (
            (matched.restoration_aligned > 0)
            & (matched.injection_aligned > 0)
        )
        result = np.full_like(matched.bidirectional_aligned, np.nan)
        result[valid] = (
            matched.bidirectional_aligned[valid] - mismatch.bidirectional_aligned[valid]
        )
        return result
    return matched.bidirectional_aligned - mismatch.bidirectional_aligned


def reference_scale(values: Any, *, floor: float) -> float:
    """Positive unadjusted matched/gross reference mean, or ``nan`` when non-estimable."""

    value = float(np.mean(np.asarray(values, dtype=np.float64)))
    if not audit_check(
        bool(np.isfinite(value) and value > float(floor)),
        "causal.reference_scale",
        f"causal reference scale {value:.3e} is at or below the registered floor "
        f"{float(floor):.3e}; calibrated targets built on it are reported as non-estimable",
        observed=value,
        tolerance=float(floor),
    ):
        return float("nan")
    return value


def calibrated_targets(
    semantic_gross: Any,
    structural_gross: Any,
    semantic_necessity: Any,
    structural_necessity: Any,
    *,
    gross_scales: Mapping[str, float],
    necessity_scales: Mapping[str, float],
) -> dict[str, np.ndarray]:
    g_sem = np.asarray(semantic_gross) / float(gross_scales["semantic"])
    g_str = np.asarray(structural_gross) / float(gross_scales["structural"])
    n_sem = np.asarray(semantic_necessity) / float(necessity_scales["semantic"])
    n_str = np.asarray(structural_necessity) / float(necessity_scales["structural"])
    return {
        "g_semantic": g_sem,
        "g_structural": g_str,
        "gross_total_for_J": 0.5 * (g_sem + g_str),
        "gross_contrast_for_D_rel": g_sem - g_str,
        "n_semantic": n_sem,
        "n_structural": n_str,
        "necessity_total_for_J": 0.5 * (n_sem + n_str),
        "necessity_contrast_for_D_rel": n_sem - n_str,
    }
