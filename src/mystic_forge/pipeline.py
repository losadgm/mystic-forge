from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd
from mystic_forge.logger import logger


@dataclass
class Context:
    """Shared state that flows through every pipeline stage."""
    raw: pd.DataFrame | None = None
    clean: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Stage(Protocol):
    name: str

    def run(self, ctx: Context) -> Context: ...


class Pipeline:
    def __init__(self, stages: list[Stage]) -> None:
        self._stages = stages

    def run(self) -> Context:
        ctx = Context()
        total = len(self._stages)
        logger.info("=== pipeline start ({} stages) ===", total)
        for i, stage in enumerate(self._stages, 1):
            logger.info("[{}/{}] stage '{}' starting", i, total, stage.name)
            try:
                ctx = stage.run(ctx)
            except Exception:
                logger.exception("[{}/{}] stage '{}' failed", i, total, stage.name)
                raise
            logger.info("[{}/{}] stage '{}' done", i, total, stage.name)
        logger.success("=== pipeline complete ===")
        return ctx
