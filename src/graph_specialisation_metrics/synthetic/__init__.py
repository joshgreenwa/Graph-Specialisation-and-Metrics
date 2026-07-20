"""Controlled synthetic experiments built around the repository methodologies."""


def run_nar_grit(*args, **kwargs):
    from .nar_grit import main

    return main(*args, **kwargs)

__all__ = ["run_nar_grit"]
