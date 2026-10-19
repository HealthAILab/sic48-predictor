"""Single SIC case-interpretation agent.

Architecture: observe current 12-feature model context -> plan evidence needs ->
retrieve local SIC knowledge -> synthesize a structured report -> preserve the
same case context for optional follow-up questions.
"""

from .controller import answer_case_followup, generate_case_report
from .schemas import AgentRunResult, InterpretationPlan

__all__ = [
    "AgentRunResult",
    "InterpretationPlan",
    "generate_case_report",
    "answer_case_followup",
]
