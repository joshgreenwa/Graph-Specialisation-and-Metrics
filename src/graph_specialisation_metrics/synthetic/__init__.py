"""Controlled synthetic experiments built around the repository methodologies."""


def run_nar_grit(*args, **kwargs):
    from .nar_grit_fixed import main

    return main(*args, **kwargs)


def run_nar_canonical_analysis(*args, **kwargs):
    from .nar_canonical_analysis import main

    return main(*args, **kwargs)


__all__ = ["run_nar_canonical_analysis", "run_nar_grit"]
