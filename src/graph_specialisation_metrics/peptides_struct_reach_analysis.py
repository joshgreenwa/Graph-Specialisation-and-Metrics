"""Finite-versus-Jacobian semantic reach analysis for Peptides-struct GRIT.

This profile reuses the ZINC/QM9 reach pipeline while representing all nine OGB
atom fields as differentiable one-hot inputs for the Bamberger Jacobian proxy.
"""

from __future__ import annotations

from typing import Any, Sequence

from .zinc_reach_analysis import PEPTIDES_STRUCT_PROFILE
from .zinc_reach_analysis import main as _shared_main


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return _shared_main(argv, profile=PEPTIDES_STRUCT_PROFILE)


if __name__ == "__main__":
    main()
