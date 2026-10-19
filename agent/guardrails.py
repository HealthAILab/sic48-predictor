from __future__ import annotations

import re
from typing import Optional

from .schemas import GuardrailDecision


def evaluate_guardrails(question: str, language: str = "zh") -> GuardrailDecision:
    q = (question or "").strip().lower()
    flags = []
    directives = []

    anticoag = any(x in q for x in ["肝素", "抗凝", "lmwh", "ufh", "heparin", "anticoag"])
    dose = any(x in q for x in ["剂量", "多少单位", "多少mg", "多少 mg", "怎么打", "how many units", "what dose", "dosage"])
    active_bleeding = any(x in q for x in ["活动性出血", "正在出血", "active bleeding", "bleeding now"])
    model_drives_treatment = any(x in q for x in ["按模型", "根据模型", "风险高就", "概率高就", "based on the model", "based on the risk", "high risk so"])
    certainty = any(x in q for x in ["确定得", "确诊", "一定是", "肯定是", "definitely has", "confirmed sic", "certainly sic"])

    if anticoag and dose:
        flags.append("individualized_anticoagulant_dose")
        directives.append("Do not provide an individualized anticoagulant dose. Require a clear indication, bleeding-risk assessment, renal function, platelet count, and clinician-directed protocol.")
        direct = (
            "不能仅凭SIC风险、体重或网页结果给出个体化抗凝剂量。抗凝处方需要明确适应证，并结合活动性出血、血小板、肾功能、合并用药及当地方案由临床医生决定。"
            if language == "zh" else
            "An individualized anticoagulant dose cannot be determined from SIC risk, body weight, or the web result alone. Prescribing requires a clear indication plus assessment of active bleeding, platelet count, renal function, concomitant therapy, and a clinician-directed protocol."
        )
        return GuardrailDecision(level="block", flags=flags, mandatory_directives=directives, direct_response=direct)

    if active_bleeding and anticoag:
        flags.append("active_bleeding_anticoagulation")
        directives.append("Active bleeding requires urgent clinical assessment; do not recommend starting anticoagulation solely because of SIC or model risk.")
    if model_drives_treatment and anticoag:
        flags.append("prediction_not_treatment_indication")
        directives.append("The model predicts SIC risk and is not an anticoagulation indication.")
    if certainty:
        flags.append("prediction_not_diagnosis")
        directives.append("Do not convert a prediction probability into a confirmed current SIC diagnosis.")

    level = "caution" if flags else "normal"
    return GuardrailDecision(level=level, flags=flags, mandatory_directives=directives)


def apply_safety_footer(answer: str, decision: GuardrailDecision, language: str = "zh") -> str:
    text = (answer or "").strip()
    if decision.level != "caution" or not decision.flags:
        return text
    if language == "zh":
        footer = "**安全边界：** "
        statements = []
        if "active_bleeding_anticoagulation" in decision.flags:
            statements.append("活动性出血时应优先进行紧急临床评估，不能仅凭SIC或模型风险启动抗凝")
        if "prediction_not_treatment_indication" in decision.flags:
            statements.append("模型风险概率不是抗凝适应证")
        if "prediction_not_diagnosis" in decision.flags:
            statements.append("风险预测不等于当前SIC确诊")
    else:
        footer = "**Safety boundary:** "
        statements = []
        if "active_bleeding_anticoagulation" in decision.flags:
            statements.append("active bleeding requires urgent clinical assessment and anticoagulation should not be started solely because of SIC or model risk")
        if "prediction_not_treatment_indication" in decision.flags:
            statements.append("the model risk probability is not an anticoagulation indication")
        if "prediction_not_diagnosis" in decision.flags:
            statements.append("risk prediction is not a confirmed current SIC diagnosis")
    if not statements:
        return text
    footer_text = footer + ("；".join(statements) if language == "zh" else "; ".join(statements)) + "."
    if footer_text.lower() in text.lower():
        return text
    return text + "\n\n" + footer_text
