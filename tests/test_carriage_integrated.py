from types import SimpleNamespace

import pytest
import torch

from graph_specialisation_metrics.carriage.core import integrated_loss_carriage
from graph_specialisation_metrics.carriage.grit_runner import (
    _integrated_failure_stats,
    _pooled_head_predictions,
    _retain_or_reject_unconverged_paths,
)


def _integrate(clean, swap, loss, *, pooling="add", **kwargs):
    return integrated_loss_carriage(
        torch.as_tensor(clean, dtype=torch.float64),
        torch.as_tensor(swap, dtype=torch.float64),
        loss,
        pooling=pooling,
        atol=kwargs.pop("atol", 1e-10),
        rtol=kwargs.pop("rtol", 1e-8),
        max_intervals=kwargs.pop("max_intervals", 64),
        **kwargs,
    )


def test_integrated_carriage_is_signed_complete_and_not_clipped():
    # swap pool=1 -> clean pool=0 under L=p^2.  The exact carrier terms cancel:
    # [-10,+9] sums to -1, so both entries legitimately exceed |dL|.
    swap = [[[10.0], [-9.0]]]
    clean = [[[0.0], [0.0]]]
    out = _integrate(clean, swap, lambda p: p[:, 0].square())

    torch.testing.assert_close(
        out["carriage"], torch.tensor([[-10.0, 9.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        out["loss_delta"], torch.tensor([-1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        out["carriage"].sum(dim=-1), out["loss_delta"], atol=1e-12, rtol=0
    )
    assert bool(out["converged"].all())


def test_integrated_carriage_averages_donor_paths_not_the_mean_endpoint():
    # Source-major order r=j*K+k.  Each source's two swapped endpoints average to
    # clean, so integrating from the mean endpoint would incorrectly return zero.
    clean = torch.zeros((4, 2, 1), dtype=torch.float64)
    swap = torch.tensor(
        [[[-1.0], [0.0]], [[1.0], [0.0]], [[0.0], [-2.0]], [[0.0], [2.0]]],
        dtype=torch.float64,
    )
    out = _integrate(clean, swap, lambda p: p[:, 0].square())
    B = out["carriage"].view(2, 2, 2).mean(dim=1).t()

    torch.testing.assert_close(
        B, torch.tensor([[-1.0, 0.0], [0.0, -4.0]], dtype=torch.float64)
    )


def test_integrated_carriage_mean_pooling_projects_back_to_carriers():
    clean = [[[0.0], [0.0]]]
    swap = [[[2.0], [0.0]]]
    out = _integrate(clean, swap, lambda p: p[:, 0].square(), pooling="mean")

    # mean pool moves 1 -> 0, and only the first carrier moved.
    torch.testing.assert_close(
        out["carriage"], torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(out["carriage"].sum(-1), out["loss_delta"])


def test_adaptive_quadrature_localises_an_l1_kink_without_rescaling():
    # H(alpha) crosses zero at alpha=1/3.  A fixed global midpoint rule converges
    # slowly here; local Gauss--Kronrod refinement should resolve the kink.
    clean = [[[2.0]]]
    swap = [[[-1.0]]]
    out = _integrate(clean, swap, lambda p: p[:, 0].abs())

    assert int(out["intervals"][0]) > 1
    assert bool(out["converged"][0])
    assert abs(float(out["completeness_residual"][0])) < 1e-8
    torch.testing.assert_close(
        out["carriage"], torch.tensor([[1.0]], dtype=torch.float64), atol=1e-8, rtol=0
    )

    unresolved = _integrate(
        clean,
        swap,
        lambda p: p[:, 0].abs(),
        max_intervals=1,
    )
    assert not bool(unresolved["converged"][0])
    # The raw estimate is exposed as-is: no forced closure and no clipping.
    assert abs(float(unresolved["completeness_residual"][0])) > 1e-3


def test_integrated_carriage_supports_smooth_multioutput_bce_loss():
    clean = torch.tensor(
        [[[0.2, -0.1], [0.3, 0.4]], [[-0.1, 0.2], [0.5, -0.3]]],
        dtype=torch.float64,
    )
    swap = clean + torch.tensor(
        [[[0.8, -0.2], [-0.1, 0.5]], [[-0.4, 0.1], [0.2, 0.7]]],
        dtype=torch.float64,
    )
    weight = torch.tensor([[1.0, -0.3], [-0.4, 0.8], [0.2, 0.5]], dtype=torch.float64)
    target = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)

    def bce(p):
        logits = p @ weight.t()
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target.expand_as(logits), reduction="none"
        ).mean(dim=-1)

    out = _integrate(clean, swap, bce)
    torch.testing.assert_close(
        out["carriage"].sum(dim=-1), out["loss_delta"], atol=1e-9, rtol=0
    )
    assert bool(out["converged"].all())


def test_pooled_head_adapter_preserves_replica_axis_for_scalar_output():
    class ScalarHead:
        def __call__(self, batch):
            return batch.x[:, 0], batch.y

    model = SimpleNamespace(model=SimpleNamespace(post_mp=ScalarHead()))
    pooled = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pred = _pooled_head_predictions(model, pooled, torch.zeros((1, 1)))

    assert pred.shape == (3, 1)
    torch.testing.assert_close(pred[:, 0], pooled[:, 0])


def test_rare_capped_path_is_retained_warned_and_reported(capsys):
    path = {
        "converged": torch.tensor([True, False, True]),
        "completeness_residual": torch.tensor([0.0, 2e-4, 0.0]),
        "quadrature_error": torch.tensor([1e-7, 3e-4, 1e-7]),
    }
    cfg = SimpleNamespace(
        integrated_unconverged_error_cap=5e-4,
        integrated_max_unconverged_fraction=1e-3,
    )
    mask = _retain_or_reject_unconverged_paths(path, cfg, "test")
    stats = _integrated_failure_stats(
        mask,
        path["completeness_residual"].numpy(),
        path["quadrature_error"].numpy(),
    )

    assert mask.tolist() == [True, False, True]  # failed path was retained, not dropped
    assert stats["integrated_unconverged_count"] == 1
    assert stats["integrated_path_count"] == 3
    assert stats["integrated_unconverged_completeness_residual_max"] == pytest.approx(2e-4)
    assert stats["integrated_unconverged_carrier_error_max"] == pytest.approx(3e-4)
    assert "retaining best estimates for 1/3 test paths" in capsys.readouterr().out


def test_capped_path_aborts_when_its_error_exceeds_policy_cap():
    path = {
        "converged": torch.tensor([False]),
        "completeness_residual": torch.tensor([6e-4]),
        "quadrature_error": torch.tensor([2e-4]),
    }
    cfg = SimpleNamespace(
        integrated_unconverged_error_cap=5e-4,
        integrated_max_unconverged_fraction=1e-3,
    )
    with pytest.raises(RuntimeError, match="exceeded integrated_unconverged_error_cap"):
        _retain_or_reject_unconverged_paths(path, cfg, "test")
