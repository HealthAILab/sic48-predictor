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
    "zh": ["最终风险评估", "关键变量", "循证推理", "重点监测因素"],
    "en": ["Final Risk Assessment", "Key Variables", "Reasoning with Evidence", "Factors Needing Monitoring"],
}


# Flag boilerplate only when it is generic. Do not penalize case-specific
# uncertainty statements that actually affect the interpretation.
_ZH_BOILERPLATE = re.compile(
    r"(?:解读边界|免责声明|仅供(?:科研|研究|参考)|不能替代(?:医生|临床|医师)|"
    r"不(?:等于|代表)(?:临床)?(?:确诊|诊断|因果)|SHAP.{0,8}不(?:等于|代表)因果|"
    r"预测(?:结果|概率|风险)?不(?:等于|代表)(?:当前)?(?:临床)?(?:诊断|确诊))",
    re.I,
)
_EN_BOILERPLATE = re.compile(
    r"(?:Interpretation Boundaries|Disclaimer|for research (?:purposes|use) only|"
    r"does not (?:replace|substitute for) (?:clinical|medical|physician)|"
    r"prediction is not (?:a )?diagnosis|SHAP (?:is not|does not establish) causa|"
    r"not a confirmed diagnosis)", re.I,
)


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

    boilerplate = _EN_BOILERPLATE if language == "en" else _ZH_BOILERPLATE
    if boilerplate.search(value):
        flags.append("defensive_boilerplate")
    # Repeated negation-heavy caveats often indicate the LLM is writing for
    # anticipated criticism rather than explaining the case. Trigger a rewrite,
    # but retain specific factual limitations in the rewritten report.
    negative_phrases = (
        r"\b(?:does not (?:mean|imply|indicate)|cannot|not necessarily|should not be taken as)\b"
        if language == "en" else r"(?:不(?:代表|说明|意味着|等于)|不能(?:说明|证明|排除|替代))"
    )
    if len(re.findall(negative_phrases, value, flags=re.I)) >= 2:
        flags.append("repetitive_defensive_negation")

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
    """Repair misleading claims and eliminate empty defensive prose in one pass."""
    joined = ", ".join(flags) if flags else "none"
    if language == "en":
        return (
            "Rewrite the same four-section report, preserving the predicted 48-hour SIC probability, risk class, "
            "observed patient values, and authorized [R#] citations. Fix these QA flags: " + joined + ". "
            "Delete generic disclaimer paragraphs and self-defensive clauses rather than paraphrasing them. "
            "Do not add an Interpretation Boundaries section. State the result and SHAP direction positively. "
            "Avoid describing PaO2 max alone as normal oxygenation, lower-risk SHAP factors as protective, "
            "or low SIC risk as overall clinical stability. Keep only case-specific facts and evidence. "
            "Never invent data or clinical diagnoses."
        )
    return (
        "请重写同样的四部分病例报告，保留48小时SIC预测概率、风险分类、实际患者数值和有效[R#]引用；"
        "本次质控问题：" + joined + "。只修正问题，不补充患者事实。"
        "删掉用于自我辩护的句子和免责声明，不是换一种说法再写一遍。"
        "不要添加‘解读边界’章节；以‘48小时预测结果、变量上推或下拉、相关证据、具体监测重点’进行正面陈述。"
        "避免把单个PaO₂最大值解释成整体氧合正常、把SHAP下拉变量称为保护因素、"
        "把SIC低风险写成总体病情稳定。"
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
