from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvidenceSource:
    source_id: str
    short: str
    citation: str


@dataclass
class EvidenceBundle:
    content: str = ""
    knowledge_modules: List[str] = field(default_factory=list)
    module_files: List[str] = field(default_factory=list)
    sources: List[EvidenceSource] = field(default_factory=list)


@dataclass
class InterpretationPlan:
    focus_summary: str = ""
    queries: List[str] = field(default_factory=list)
    monitoring_focus: List[str] = field(default_factory=list)


@dataclass
class AgentRunResult:
    answer: str
    elapsed_seconds: float = 0.0
    knowledge_modules: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    plan: InterpretationPlan = field(default_factory=InterpretationPlan)
    safety_flags: List[str] = field(default_factory=list)


@dataclass
class GuardrailDecision:
    level: str = "normal"
    flags: List[str] = field(default_factory=list)
    mandatory_directives: List[str] = field(default_factory=list)
    direct_response: Optional[str] = None
