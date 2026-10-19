from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set


# Deterministic report-level QA rules used as a safety net around the LLM.
# These are deliberately narrow: they target wording that would materially
# over-interpret the 12-variable model rather than attempting to grade prose.
_ZH_PATTERNS = [
    ("pao2_overinterpretation", re.compile(r"(?:氧合(?:良好|正常|保存良好)|无低氧|没有低氧|不提示(?:明显|严重)?呼吸衰竭|呼吸功能(?:正常|良好))")),
    ("overall_severity_overreach", re.compile(r"(?:总体|整体)(?:病情|状态).{0,4}(?:稳定|较轻|轻|良好|正常)")),
    ("low_sic_means_mild", re.compile(r"(?:低风险|风险较低).{0,12}(?:说明|提示|意味着).{0,8}(?:病情较轻|病情稳定|总体稳定|器官功能正常)")),
    ("shap_protective_claim", re.compile(r"(?<!不)(?<!不能)(?:是|属于|具有|存在).{0,4}(?:保护因素|保护作用)")),
]

_EN_PATTERNS = [
    ("pao2_overinterpretation", re.compile(r"\b(?:preserved|normal|good) oxygenation\b|\bno (?:evidence of )?hypoxia\b|\bdoes not suggest (?:severe )?respiratory failure\b|\bnormal respiratory function\b", re.I)),
    ("overall_severity_overreach", re.compile(r"\b(?:overall|global) (?:clinical )?(?:condition|status) (?:is |appears )?(?:stable|mild|good|normal)\b", re.I)),
    ("low_sic_means_mild", re.compile(r"\b(?:low|lower) (?:SIC )?risk\b.{0,80}\b(?:clinically stable|mild overall condition|overall condition is mild)\b", re.I | re.S)),
    ("shap_protective_claim", re.compile(r"\b(?:is|acts as|represents) (?:a )?protective (?:factor|effect)\b", re.I)),
]

_REQUIRED_HEADINGS = {
    "zh": ["最终风险评估", "关键变量", "循证推理", "重点监测因素", "解读边界"],
    "en": ["Final Risk Assessment", "Key Variables", "Reasoning with Evidence", "Factors Needing Monitoring", "Interpretation Boundaries"],
}


def extract_source_ids(text: str) -> List[str]:
    """Return citation IDs in first-appearance order (e.g., P1, R5)."""
    seen: Set[str] = set()
    out: List[str] = []
    for sid in re.findall(r"\[([A-Za-z]+\d+)\]", text or ""):
        sid = sid.strip()
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _allowed_source_ids(allowed: Iterable[str] | None) -> Set[str]:
    return {str(x).strip() for x in (allowed or []) if str(x).strip()}


def report_quality_flags(
    text: str,
    *,
    language: str = "zh",
    allowed_source_ids: Iterable[str] | None = None,
) -> List[str]:
    """Return deterministic quality/safety flags for a generated case report."""
    value = str(text or "")
    flags: List[str] = []
    patterns = _EN_PATTERNS if language == "en" else _ZH_PATTERNS
    for name, pattern in patterns:
        if pattern.search(value):
            flags.append(name)

    for heading in _REQUIRED_HEADINGS.get(language, _REQUIRED_HEADINGS["zh"]):
        if heading.lower() not in value.lower():
            flags.append(f"missing_section:{heading}")

    allowed = _allowed_source_ids(allowed_source_ids)
    if allowed:
        for sid in extract_source_ids(value):
            if sid not in allowed:
                flags.append(f"unauthorized_source:{sid}")

    # Direct diagnosis/treatment overclaim checks. These are intentionally
    # conservative and only flag definitive wording.
    if language == "en":
        if re.search(r"\b(?:this patient has|diagnosis is|confirmed) SIC\b", value, re.I):
            flags.append("prediction_as_diagnosis")
        if re.search(r"\b(?:start|initiate) anticoagulation\b", value, re.I):
            flags.append("direct_treatment_instruction")
    else:
        if re.search(r"(?:该患者(?:已经|已)?确诊SIC|当前就是SIC|已发生SIC)", value, re.I):
            flags.append("prediction_as_diagnosis")
        if re.search(r"(?:立即|应当|应该)(?:启动|开始).{0,4}抗凝", value):
            flags.append("direct_treatment_instruction")

    # Deduplicate in stable order.
    return list(dict.fromkeys(flags))


def build_repair_instruction(flags: List[str], language: str = "zh") -> str:
    """Build a compact instruction for one-pass report repair."""
    joined = ", ".join(flags) if flags else "none"
    if language == "en":
        return (
            "Rewrite the same five-section report, preserving the model probability, risk class, patient values, "
            "and valid inline source IDs. Correct only the following QA issues: " + joined + ". "
            "Do not add new patient facts. In particular: PaO2 max alone cannot establish preserved/normal oxygenation "
            "or exclude respiratory failure without contemporaneous FiO2/PF ratio/support; a lower-risk SHAP direction is "
            "a model contribution, not a protective effect; low SIC prediction must not be equated with mild or stable overall illness."
        )
    return (
        "请在不改变模型概率、风险分层、患者原始数值和有效[R#]/[P#]引用的前提下，重写同样的五段式报告，"
        "仅修正以下质控问题：" + joined + "。不得新增患者事实。特别注意：PaO₂最大值单独不能证明氧合良好/正常，"
        "也不能排除呼吸衰竭，缺少同步FiO₂/PF比值/呼吸支持信息时必须保留边界；SHAP向低风险方向移动不等于保护作用；"
        "SIC低风险预测不能推导为总体病情轻或稳定。"
    )


def qa_record(
    text: str,
    *,
    language: str,
    allowed_source_ids: Iterable[str] | None = None,
) -> Dict[str, Any]:
    flags = report_quality_flags(text, language=language, allowed_source_ids=allowed_source_ids)
    return {
        "flags": flags,
        "passed": not flags,
        "cited_source_ids": extract_source_ids(text),
    }
