"""Controlled synthetic experiments built around the repository methodologies."""


def run_nar_grit(*args, **kwargs):
    from .nar_grit_fixed import main

    return main(*args, **kwargs)


def run_nar_canonical_analysis(*args, **kwargs):
    from .nar_canonical_analysis import main

    return main(*args, **kwargs)


def run_nar_methodology_extension(*args, **kwargs):
    from .nar_methodology_extension import main

    return main(*args, **kwargs)


def run_nar_methodology_paper(*args, **kwargs):
    from .nar_methodology_paper import main

    return main(*args, **kwargs)


def run_nar_causal_transition(*args, **kwargs):
    from .nar_causal_transition import main

    return main(*args, **kwargs)


def run_nar_methodology_paper_v3(*args, **kwargs):
    from .nar_methodology_paper_v3 import main

    return main(*args, **kwargs)


def run_saturation_carriage(*args, **kwargs):
    from .saturation_carriage import main

    return main(*args, **kwargs)


def run_softmax_routing_carriage(*args, **kwargs):
    from .softmax_routing_carriage import main

    return main(*args, **kwargs)


def run_query_routing_carriage(*args, **kwargs):
    from .query_routing_carriage import main

    return main(*args, **kwargs)


__all__ = [
    "run_nar_causal_transition",
    "run_nar_canonical_analysis",
    "run_nar_grit",
    "run_nar_methodology_extension",
    "run_nar_methodology_paper",
    "run_nar_methodology_paper_v3",
    "run_query_routing_carriage",
    "run_saturation_carriage",
    "run_softmax_routing_carriage",
]
