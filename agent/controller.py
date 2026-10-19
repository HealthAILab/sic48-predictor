from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from sic_agent import _format_patient_context, chat_completion
from .guardrails import apply_safety_footer, evaluate_guardrails
from .prompts import (
    FOLLOWUP_SYSTEM_EN,
    FOLLOWUP_SYSTEM_ZH,
    PLANNER_SYSTEM_EN,
    PLANNER_SYSTEM_ZH,
    REPORT_SYSTEM_EN,
    REPORT_SYSTEM_ZH,
)
from .retrieval import retrieve_evidence, retrieve_for_question
from .quality_checks import build_repair_instruction, extract_source_ids, report_quality_flags
from .schemas import AgentRunResult, EvidenceBundle, InterpretationPlan

OUT_OF_SCOPE_TERMS = [
    "天气", "气温", "下雨", "股票", "股价", "基金", "比特币", "足球", "篮球", "电影", "明星", "旅游", "酒店", "机票",
    "weather", "stock price", "bitcoin", "football", "basketball", "movie", "celebrity", "hotel", "flight",
]
SIC_SCOPE_TERMS = [
    "sic", "dic", "脓毒症", "凝血", "coag", "sepsis", "shap", "xgboost", "模型", "风险", "sofa", "pt", "患者", "病例",
]


def _strip_json_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def _parse_plan(raw: str, case_context: Dict[str, Any], language: str) -> InterpretationPlan:
    value = _strip_json_fence(raw)
    data: Dict[str, Any] = {}
    try:
        data = json.loads(value)
    except Exception:
        match = re.search(r"\{.*\}", value, flags=re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = {}

    queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(queries, list):
        queries = []
    queries = [str(x).strip() for x in queries if str(x).strip()][:3]

    monitoring = data.get("monitoring_focus") if isinstance(data, dict) else None
    if not isinstance(monitoring, list):
        monitoring = []
    valid_features = set((case_context.get("feature_values") or {}).keys())
    monitoring = [str(x).strip() for x in monitoring if str(x).strip() and (str(x).strip() in valid_features or not valid_features)][:4]

    if not queries:
        shap_top = case_context.get("shap_top_features") or []
        top_names = [str(x.get("feature") or "") for x in shap_top[:3] if x.get("feature")]
        if language == "en":
            queries = [f"Clinical interpretation of {name} in sepsis-induced coagulopathy" for name in top_names]
            if not queries:
                queries = ["Interpretation of a 48-hour SIC machine-learning risk prediction"]
        else:
            queries = [f"{name} 与脓毒症诱导凝血病风险的临床解释" for name in top_names]
            if not queries:
                queries = ["48小时SIC机器学习风险预测的临床解释"]

    focus = str(data.get("focus_summary") or "").strip() if isinstance(data, dict) else ""
    if not focus:
        focus = (
            "Explain the prediction using the leading SHAP contributors and SIC evidence."
            if language == "en" else
            "结合主要SHAP贡献因素与SIC证据解释本次风险预测。"
        )
    return InterpretationPlan(focus_summary=focus, queries=queries, monitoring_focus=monitoring)


def _dedupe_sources(evidence: EvidenceBundle):
    """Return evidence sources once, preserving retrieval order."""
    seen = set()
    unique = []
    for source in evidence.sources or []:
        sid = str(source.source_id or "").strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        unique.append(source)
    return unique


def _authorized_sources_text(evidence: EvidenceBundle, language: str) -> str:
    sources = _dedupe_sources(evidence)
    if not sources:
        return "None" if language == "en" else "无"
    return "\n".join(f"[{s.source_id}] {s.citation}" for s in sources)


def _used_evidence_ids(answer: str, evidence: EvidenceBundle) -> List[str]:
    """Return only authorized source IDs actually cited in visible text."""
    allowed = [s.source_id for s in _dedupe_sources(evidence)]
    allowed_set = set(allowed)
    return [sid for sid in extract_source_ids(answer) if sid in allowed_set]


def _repair_report_if_needed(
    answer: str,
    *,
    language: str,
    evidence: EvidenceBundle,
    case_context: Dict[str, Any],
) -> str:
    """Run one targeted repair pass only when deterministic QA finds a material issue."""
    allowed_ids = [s.source_id for s in _dedupe_sources(evidence)]
    flags = report_quality_flags(answer, language=language, allowed_source_ids=allowed_ids)
    # Missing headings can result from harmless formatting drift and are normalized
    # separately. Repair only clinically material over-interpretation/source issues.
    critical = [f for f in flags if not f.startswith("missing_section:")]
    if not critical:
        return answer
    repair_system = REPORT_SYSTEM_EN if language == "en" else REPORT_SYSTEM_ZH
    repair_instruction = build_repair_instruction(critical, language=language)
    repaired = chat_completion(
        [
            {"role": "system", "content": repair_system},
            {
                "role": "user",
                "content": (
                    ("# Report requiring QA repair\n" if language == "en" else "# 需要质控修订的报告\n")
                    + answer
                    + "\n\n# "
                    + ("Original patient/model context\n" if language == "en" else "原始患者/模型上下文\n")
                    + _format_patient_context(case_context, language=language)
                    + "\n\n# "
                    + ("Repair instruction\n" if language == "en" else "修订要求\n")
                    + repair_instruction
                    + "\n\n# "
                    + ("Authorized evidence sources\n" if language == "en" else "允许引用的证据来源\n")
                    + _authorized_sources_text(evidence, language)
                ),
            },
        ],
        temperature=0.0,
        max_tokens=1100,
    )
    return repaired or answer


_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s{0,3}(?:#{1,6}\s*)?(?:📚\s*)?"
    r"(?:参考依据|参考文献|references|evidence)\s*[:：]?\s*$"
)


def _strip_generated_reference_section(answer: str) -> str:
    """Remove any LLM-generated bibliography so the UI owns source rendering.

    The model is asked to use inline source IDs such as [R1].  A deterministic
    bibliography is appended by Python, which prevents duplicated or invented
    reference lists across the report and follow-up answers.
    """
    text = (answer or "").strip()
    match = _REFERENCE_HEADING_RE.search(text)
    if match:
        text = text[:match.start()].rstrip()
    return text



def _trim_defensive_report_prose(answer: str, language: str) -> str:
    """Remove stock disclaimer sections/lines without touching clinical findings.

    Prompting + the targeted QA rewrite do the semantic editing. This final
    deterministic pass only handles obvious generic filler and the legacy
    fifth section; it never removes evidence or case-specific uncertainties.
    """
    text = (answer or "").strip()
    boundary_heading = re.compile(
        r"^\s{0,3}#{1,6}\s*(?:⚠️\s*)?"
        r"(?:解读边界|解释边界|局限性说明|Interpretation Boundaries|"
        r"Interpretation Limits|Disclaimer)\s*$", re.I
    )
    # Whole-line boilerplate only: do not modify paragraphs discussing, e.g.,
    # missing FiO2, unavailable coagulation parameters, or bleeding risk.
    generic_line = re.compile(
        r"^\s*(?:[-*]\s*)?(?:\*\*[^*]+\*\*[:：]\s*)?"
        r"(?:本(?:报告|系统|结果|预测)|该(?:报告|结果)|预测(?:结果|概率)?|"
        r"48\s*h\s*(?:SIC)?\s*风险预测|SHAP(?:结果|贡献)?|"
        r"This (?:report|tool|prediction|result)|"
        r"The (?:report|prediction)|Risk prediction|SHAP)"
        r"\s*(?:仅供|只(?:用于|是)|不(?:代表|等于|意味着|说明|替代)|"
        r"does not|is not|cannot|should not|is intended|is for)"
        r"[^\n]{0,190}$", re.I
    )
    lines = []
    skipping_boundary = False
    for line in text.splitlines():
        if boundary_heading.match(line):
            skipping_boundary = True
            continue
        if skipping_boundary:
            if re.match(r"^\s{0,3}#{1,6}\s+", line):
                skipping_boundary = False
            else:
                continue
        # Limit to standalone sentences that are only generic disclaimers.
        # Mixed clinical statements are deliberately preserved for the QA model.
        if generic_line.match(line.strip()):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _decorate_report_headings(answer: str, language: str) -> str:
    """Normalize the compact report headings and keep one icon per section."""
    text = (answer or "").strip()
    if language == "en":
        replacements = {
            r"(?im)^###\s*(?:🩺\s*)?(?:\d+\.\s*)?Final Risk Assessment\s*$": "### 🩺 Final Risk Assessment",
            r"(?im)^###\s*(?:🔎\s*)?(?:\d+\.\s*)?Key Variables\s*$": "### 🔎 Key Variables",
            r"(?im)^###\s*(?:🧠\s*)?(?:\d+\.\s*)?(?:Reasoning with Evidence|Evidence-grounded SIC Interpretation)\s*$": "### 🧠 Reasoning with Evidence",
            r"(?im)^###\s*(?:👁️?\s*)?(?:\d+\.\s*)?(?:Factors Needing Monitoring|Priorities Among Currently Available Variables)\s*$": "### 👁️ Factors Needing Monitoring",
            r"(?im)^###\s*(?:⚠️?\s*)?(?:\d+\.\s*)?Interpretation Boundaries\s*$": "### ⚠️ Interpretation Boundaries",
        }
    else:
        replacements = {
            r"(?im)^###\s*(?:🩺\s*)?(?:\d+\.\s*)?(?:最终风险评估|SIC风险概述)\s*$": "### 🩺 最终风险评估",
            r"(?im)^###\s*(?:🔎|📊)?\s*(?:\d+\.\s*)?(?:关键变量|主要模型贡献因素)\s*$": "### 🔎 关键变量",
            r"(?im)^###\s*(?:🧠|📚)?\s*(?:\d+\.\s*)?(?:循证推理|SIC相关循证解释)\s*$": "### 🧠 循证推理",
            r"(?im)^###\s*(?:👁️?\s*)?(?:\d+\.\s*)?(?:重点监测因素|当前已有变量中的重点关注项)\s*$": "### 👁️ 重点监测因素",
            r"(?im)^###\s*(?:⚠️?\s*)?(?:\d+\.\s*)?解读边界\s*$": "### ⚠️ 解读边界",
        }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text


def _append_report_evidence(answer: str, evidence: EvidenceBundle, language: str) -> str:
    """Append one deduplicated bibliography to the main case report only."""
    text = _strip_generated_reference_section(answer)
    sources = _dedupe_sources(evidence)
    if not sources:
        return text
    marker = "### 📚 Evidence" if language == "en" else "### 📚 参考依据"
    lines = [marker]
    for source in sources[:6]:
        lines.append(f"- [{source.source_id}] {source.citation}")
    return text + "\n\n" + "\n".join(lines)


def _append_followup_evidence_note(
    answer: str, evidence: EvidenceBundle, case_report: str, language: str
) -> str:
    """Use compact evidence IDs for follow-up answers instead of repeating citations.

    If a follow-up retrieves a source that was not listed in the main report, only
    that new source is shown once in compact form.
    """
    text = _strip_generated_reference_section(answer)
    sources = _dedupe_sources(evidence)
    if not sources:
        return text

    ids = [s.source_id for s in sources[:5]]
    report_ids = set(re.findall(r"\[([A-Za-z]+\d+)\]", case_report or ""))
    new_sources = [s for s in sources if s.source_id not in report_ids]

    if language == "en":
        note = "**📚 Evidence IDs for this answer:** " + ", ".join(f"[{sid}]" for sid in ids)
        if not new_sources:
            note += " (full citations are listed in the case report above)."
        elif new_sources:
            note += "\n\n**New source(s):**\n" + "\n".join(
                f"- [{s.source_id}] {s.citation}" for s in new_sources[:3]
            )
    else:
        note = "**📚 本次回答依据：** " + "、".join(f"[{sid}]" for sid in ids)
        if not new_sources:
            note += "（完整文献见病例报告下方“参考依据”）。"
        else:
            note += "\n\n**新增依据：**\n" + "\n".join(
                f"- [{s.source_id}] {s.citation}" for s in new_sources[:3]
            )
    return text + "\n\n" + note


def _plan_case(case_context: Dict[str, Any], language: str) -> InterpretationPlan:
    system = PLANNER_SYSTEM_EN if language == "en" else PLANNER_SYSTEM_ZH
    context = _format_patient_context(case_context, language=language)
    user = (
        f"Current case context:\n{context}\n\nReturn the planning JSON only."
        if language == "en" else
        f"当前病例上下文：\n{context}\n\n只返回规划JSON。"
    )
    raw = chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=500,
    )
    return _parse_plan(raw, case_context=case_context, language=language)


def generate_case_report(case_context: Dict[str, Any], language: str = "zh") -> AgentRunResult:
    """Run the single-agent P0 loop: observe -> plan -> retrieve -> synthesize."""
    t0 = time.perf_counter()
    if not case_context:
        raise ValueError("No current patient context is available.")

    plan = _plan_case(case_context, language=language)
    # Two case-specific queries plus deterministic feature hints are sufficient for
    # the P0 report and reduce prompt bloat for the reasoning model.
    evidence = retrieve_evidence(plan.queries[:2], case_context=case_context, language=language, max_modules=4)
    context = _format_patient_context(case_context, language=language)
    system = REPORT_SYSTEM_EN if language == "en" else REPORT_SYSTEM_ZH

    if language == "en":
        user = f"""# Current patient/model context
{context}

# Agent interpretation plan
Focus: {plan.focus_summary}
Evidence queries:
{chr(10).join('- ' + q for q in plan.queries)}

# Retrieved local SIC evidence
{evidence.content or 'No local evidence was retrieved.'}

# Authorized evidence sources
{_authorized_sources_text(evidence, language)}

Generate the structured case interpretation now. Do not reveal the planning or retrieval workflow."""
    else:
        user = f"""# 当前患者/模型上下文
{context}

# 智能体病例解释计划
解释重点：{plan.focus_summary}
循证检索问题：
{chr(10).join('- ' + q for q in plan.queries)}

# 检索到的本地SIC证据
{evidence.content or '本次未检索到本地证据。'}

# 允许引用的证据来源
{_authorized_sources_text(evidence, language)}

请生成结构化病例解读。不要向用户展示内部规划或检索流程。"""

    answer = chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.08,
        max_tokens=1300,
    )
    answer = _strip_generated_reference_section(answer)
    answer = _decorate_report_headings(answer, language)
    answer = _repair_report_if_needed(answer, language=language, evidence=evidence, case_context=case_context)
    answer = _strip_generated_reference_section(answer)
    answer = _trim_defensive_report_prose(answer, language)
    answer = _decorate_report_headings(answer, language)
    used_ids = _used_evidence_ids(answer, evidence)
    return AgentRunResult(
        answer=answer,
        elapsed_seconds=time.perf_counter() - t0,
        knowledge_modules=evidence.knowledge_modules,
        evidence_ids=used_ids,
        plan=plan,
    )


def _is_out_of_scope(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    has_bad = any(term.lower() in q for term in OUT_OF_SCOPE_TERMS)
    has_scope = any(term.lower() in q for term in SIC_SCOPE_TERMS)
    return has_bad and not has_scope


def answer_case_followup(
    question: str,
    case_context: Dict[str, Any],
    case_report: str,
    language: str = "zh",
    history: Optional[List[Dict[str, str]]] = None,
) -> AgentRunResult:
    """Answer a context-aware follow-up using the same single agent."""
    t0 = time.perf_counter()
    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if _is_out_of_scope(question):
        answer = (
            "该问题超出当前SIC病例解读智能体的范围。你可以继续询问本病例的风险预测、SHAP、SIC/DIC、凝血指标或相关监测含义。"
            if language == "zh" else
            "That question is outside the scope of this SIC case-interpretation agent. You can ask about this case's risk prediction, SHAP results, SIC/DIC, coagulation markers, or related monitoring implications."
        )
        return AgentRunResult(answer=answer, elapsed_seconds=time.perf_counter() - t0)

    guardrail = evaluate_guardrails(question, language=language)
    if guardrail.direct_response:
        return AgentRunResult(
            answer=guardrail.direct_response,
            elapsed_seconds=time.perf_counter() - t0,
            safety_flags=guardrail.flags,
        )

    evidence = retrieve_for_question(question, case_context=case_context, language=language, max_modules=2)
    context = _format_patient_context(case_context, language=language)
    system = FOLLOWUP_SYSTEM_EN if language == "en" else FOLLOWUP_SYSTEM_ZH

    recent = []
    for item in (history or [])[-4:]:
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            recent.append(f"{role}: {content[:1200]}")
    recent_text = "\n".join(recent) or ("None" if language == "en" else "无")

    if language == "en":
        user = f"""# Current patient/model context
{context}

# Existing case report
{case_report[:7000]}

# Recent case-specific conversation
{recent_text}

# Retrieved evidence for this question
{evidence.content or 'No local evidence was retrieved.'}

# Authorized evidence sources
{_authorized_sources_text(evidence, language)}

# User follow-up question
{question}

Answer the question directly and concisely."""
    else:
        user = f"""# 当前患者/模型上下文
{context}

# 已生成的病例解读报告
{case_report[:7000]}

# 最近的病例追问记录
{recent_text}

# 针对本题检索到的证据
{evidence.content or '本次未检索到本地证据。'}

# 允许引用的证据来源
{_authorized_sources_text(evidence, language)}

# 用户追问
{question}

请直接、简洁回答当前问题。"""

    answer = chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=700,
    )
    cited_ids = _used_evidence_ids(answer, evidence)
    if cited_ids:
        cited_set = set(cited_ids)
        cited_evidence = EvidenceBundle(
            content=evidence.content,
            knowledge_modules=evidence.knowledge_modules,
            module_files=evidence.module_files,
            sources=[s for s in evidence.sources if s.source_id in cited_set],
        )
        answer = _append_followup_evidence_note(answer, cited_evidence, case_report, language)
    answer = apply_safety_footer(answer, guardrail, language=language)
    return AgentRunResult(
        answer=answer,
        elapsed_seconds=time.perf_counter() - t0,
        knowledge_modules=evidence.knowledge_modules,
        evidence_ids=cited_ids,
        safety_flags=guardrail.flags,
    )
