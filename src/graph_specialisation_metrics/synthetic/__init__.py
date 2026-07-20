"""Controlled synthetic experiments built around the repository methodologies."""


def run_nar_grit(*args, **kwargs):
    from .nar_grit_fixed import main

    return main(*args, **kwargs)


def run_legacy_mixed_nar_grit(*args, **kwargs):
    """Run the superseded dual-payload prototype retained for reproducibility."""
    from .nar_grit import main

    return main(*args, **kwargs)

__all__ = ["run_nar_grit", "run_legacy_mixed_nar_grit"]
