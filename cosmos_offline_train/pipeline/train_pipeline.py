"""Public training pipeline surface during incremental migration."""

from __future__ import annotations


def run(config, args) -> None:
    """Run training through the stable legacy implementation."""
    from ..train import _run_training

    _run_training(config, args)


def prepare_splits(config) -> None:
    from ..train import _prepare_splits

    _prepare_splits(config)

