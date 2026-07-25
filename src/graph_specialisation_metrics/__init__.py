"""Utilities for graph transformer specialisation metrics and visualisations."""

__all__ = ["MethodologyConfig", "PROTOCOL_VERSION", "main"]


def main(*args, **kwargs):
    """Lazily dispatch to the canonical public runner."""

    from .main import main as run

    return run(*args, **kwargs)


def __getattr__(name):
    """Keep the public API lazy so importing the package does not initialize a runner."""

    if name in {"MethodologyConfig", "PROTOCOL_VERSION"}:
        from .methodology.protocol import MethodologyConfig, PROTOCOL_VERSION

        return {
            "MethodologyConfig": MethodologyConfig,
            "PROTOCOL_VERSION": PROTOCOL_VERSION,
        }[name]
    raise AttributeError(name)
