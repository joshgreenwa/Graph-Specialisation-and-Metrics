"""Bamberger-versus-Functional-carriage reach analysis for QM9 gap models.

This is the QM9 profile of the shared finite-intervention reach pipeline. It
compares dense, 1-hop, and 1-hop+VNode GRIT checkpoints on the HOMO--LUMO gap
target, including the semantic donor-fraction interpolation sweep.
"""

from __future__ import annotations

from typing import Any, Sequence

from .zinc_reach_analysis import QM9_PROFILE
from .zinc_reach_analysis import main as _shared_main


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return _shared_main(argv, profile=QM9_PROFILE)


if __name__ == "__main__":
    main()
