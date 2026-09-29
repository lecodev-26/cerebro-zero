"""Single compatibility bridge to the historical V5 implementation.

New 1.x code must import V5 compatibility symbols from this module only.
The bridge is intentionally small so the historical dependency can be
replaced without changing the public 1.x surface.
"""

from v5.runtime import (
    CerebroZeroV5,
    LearningEngine,
    Planner,
    Plan,
    Action,
    ReasoningEngine,
    ToolRegistry,
    ToolSpec,
    PermissionPolicy,
)
from v5.security import ActionGuard
from v5.evaluation import Evaluator as V5Evaluator

__all__ = [
    "ActionGuard", "CerebroZeroV5", "LearningEngine", "PermissionPolicy",
    "Planner", "Plan", "Action", "ReasoningEngine", "ToolRegistry",
    "ToolSpec", "V5Evaluator",
]
