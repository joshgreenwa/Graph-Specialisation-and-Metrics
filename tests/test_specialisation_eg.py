from __future__ import annotations

import pytest


def test_eventwise_gross_avoids_opposite_donor_cancellation() -> None:
    torch = pytest.importorskip("torch")
    from graph_specialisation_metrics.specialisation.scores import (
        _eventwise_functional,
        _funcmag_contrib,
    )

    phi = torch.ones(1, 1, 1, 1)  # [T,N,H,D]
    delta = torch.tensor([1.0, -1.0]).reshape(2, 1, 1, 1)

    eg = _eventwise_functional(phi, delta).sum(dim=1).mean(dim=0)
    cg = _funcmag_contrib(phi, delta.mean(dim=0, keepdim=True))

    assert float(eg[0]) == pytest.approx(1.0)
    assert float(cg[0]) == pytest.approx(0.0)


def test_eventwise_functional_keeps_carriers_until_gross_sum() -> None:
    torch = pytest.importorskip("torch")
    from graph_specialisation_metrics.specialisation.scores import _eventwise_functional

    phi = torch.tensor([[[[1.0]], [[2.0]]]])  # [T=1,N=2,H=1,D=1]
    delta = torch.tensor([
        [[[2.0]], [[-3.0]]],
        [[[-1.0]], [[4.0]]],
    ])
    functional = _eventwise_functional(phi, delta)

    assert functional.shape == (2, 2, 1)
    torch.testing.assert_close(
        functional[:, :, 0], torch.tensor([[2.0, 6.0], [1.0, 8.0]])
    )
    assert float(functional.sum(dim=1).mean(dim=0)[0]) == pytest.approx(8.5)
