"""OpenAI-compatible LLM client for the SIC Streamlit application.

This module deliberately uses ``requests`` instead of the OpenAI SDK so it can
work with the user's OpenAI-compatible /v1/chat/completions endpoint on Python 3.9.
"""
from __future__ import annotations

import json
import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv
except Exception:  # keep the prediction app usable even if dotenv is absent
    load_dotenv = None

from sic_prompts import (
    PATIENT_INTERPRETATION_TASK_EN,
    PATIENT_INTERPRETATION_TASK_ZH,
    SYSTEM_PROMPT_EN,
    SYSTEM_PROMPT_ZH,
)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

if load_dotenv is not None:
    load_dotenv(dotenv_path=str(ENV_PATH), override=True)

DEFAULT_API_URL = "http://210.37.77.5:88/ai/llm/v1/chat/completions"
DEFAULT_MODEL = "openai/dsv4"

# V3-Expanded tuning: keep prompts compact so the private LLM endpoint has less
# context to process on every request.
QA_MAX_KNOWLEDGE_MODULES = 2
QA_MAX_HISTORY_MESSAGES = 4
QA_HISTORY_CHAR_LIMIT = 1200
QA_MAX_TOKENS = 700
PATIENT_MAX_TOKENS = 900

# Reuse TCP connections across Streamlit reruns in the same Python process.
HTTP_SESSION = requests.Session()


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return _clean(value)
    try:
        import streamlit as st
        return _clean(str(st.secrets.get(name, default)))
    except (FileNotFoundError, KeyError):
        return _clean(default)


def get_llm_config() -> Dict[str, str]:
    """Read LLM settings from environment variables.

    The key is intentionally never hard-coded in source code.
    """
    return {
        "api_url": _setting("SIC_LLM_API_URL", DEFAULT_API_URL),
        "api_key": _setting("SIC_LLM_API_KEY"),
        "model": _setting("SIC_LLM_MODEL", DEFAULT_MODEL),
    }


def get_llm_status(language: str = "zh") -> Tuple[bool, str]:
    cfg = get_llm_config()
    if not cfg["api_url"]:
        return False, ("SIC_LLM_API_URL is not configured" if language == "en" else "未配置 SIC_LLM_API_URL")
    if not cfg["model"]:
        return False, ("SIC_LLM_MODEL is not configured" if language == "en" else "未配置 SIC_LLM_MODEL")
    key = cfg["api_key"]
    if not key or key.upper().startswith("REPLACE_") or key.lower() in {"your-api-key", "changeme"}:
        return False, ("A valid SIC_LLM_API_KEY is not configured" if language == "en" else "未配置有效的 SIC_LLM_API_KEY")
    return True, ("LLM configured" if language == "en" else "LLM 已配置")


KNOWLEDGE_MODULES = [
    {
        "file": "01_definition.md",
        "title": "SIC定义与临床定位",
        "title_en": "SIC definition and clinical context",
        "keywords": ["什么是", "定义", "概念", "sic是什么", "what is", "definition", "临床定位", "early recognition"],
    },
    {
        "file": "02_pathophysiology.md",
        "title": "SIC病理生理与免疫血栓",
        "title_en": "SIC pathophysiology and immunothrombosis",
        "keywords": ["机制", "病理生理", "炎症", "凝血酶", "组织因子", "内皮", "net", "nets", "免疫血栓", "纤溶", "pai-1", "protein c", "antithrombin", "pathophysiology", "mechanism", "immunothrombosis"],
    },
    {
        "file": "03_diagnostic_criteria.md",
        "title": "ISTH SIC评分与诊断边界",
        "title_en": "ISTH SIC score and diagnostic boundaries",
        "keywords": ["评分", "诊断", "标准", "criteria", "score", "inr", "pt-inr", "血小板", "platelet", "sofa", "几分", "阈值", "四器官"],
    },
    {
        "file": "04_sic_to_dic.md",
        "title": "SIC与DIC进展关系",
        "title_en": "SIC-DIC continuum and progression",
        "keywords": ["dic", "进展", "发展", "恶化", "overt", "弥散性血管内凝血", "progress", "sic和dic", "sic与dic", "区别"],
    },
    {
        "file": "05_laboratory_markers.md",
        "title": "SIC实验室指标",
        "title_en": "SIC laboratory markers",
        "keywords": ["实验室", "指标", "血小板", "platelet", "pt", "inr", "d-dimer", "d二聚体", "fdp", "纤维蛋白原", "fibrinogen", "抗凝血酶", "protein c", "血红蛋白", "lab", "biomarker"],
    },
    {
        "file": "06_clinical_monitoring.md",
        "title": "临床监测与动态复评",
        "title_en": "Clinical monitoring and reassessment",
        "keywords": ["监测", "动态", "复查", "关注", "随访", "monitor", "trend", "观察", "多久查", "复评"],
    },
    {
        "file": "07_treatment_overview.md",
        "title": "SIC治疗总原则",
        "title_en": "General management principles",
        "keywords": ["治疗", "管理", "感染控制", "器官支持", "vte", "预防", "treatment", "management", "supportive", "therapy"],
    },
    {
        "file": "08_prognosis.md",
        "title": "预后与风险",
        "title_en": "Prognosis and risk",
        "keywords": ["预后", "死亡", "结局", "mortality", "outcome", "prognosis", "风险", "生存"],
    },
    {
        "file": "09_model_interpretation.md",
        "title": "本项目模型与SHAP解释",
        "title_en": "Study model and SHAP interpretation",
        "keywords": ["shap", "模型", "xgboost", "预测", "概率", "48h", "48小时", "特征", "threshold", "阈值", "model", "为什么风险"],
    },
    {
        "file": "10_differential_diagnosis.md",
        "title": "SIC鉴别诊断",
        "title_en": "Differential diagnosis of SIC",
        "keywords": ["鉴别", "肝病", "肝功能", "hit", "肝素诱导", "ttp", "hus", "tma", "微血管病", "溶血", "药物", "differential", "liver disease", "thrombocytopenia"],
    },
    {
        "file": "11_anticoagulation_evidence.md",
        "title": "抗凝治疗证据与边界",
        "title_en": "Anticoagulation evidence and boundaries",
        "keywords": ["抗凝", "肝素", "heparin", "lmwh", "ufh", "抗凝血酶", "thrombomodulin", "治疗剂量", "预防剂量", "anticoag", "anticoagulation"],
    },
    {
        "file": "12_supportive_management.md",
        "title": "支持治疗与血液成分治疗",
        "title_en": "Supportive and blood component management",
        "keywords": ["输血", "血浆", "血小板输注", "冷沉淀", "纤维蛋白原补充", "出血", "成分治疗", "transfusion", "plasma", "bleeding"],
    },
    {
        "file": "13_scores_and_frameworks.md",
        "title": "SIC/DIC评分框架比较",
        "title_en": "SIC and DIC scoring frameworks",
        "keywords": ["overt dic", "jaam", "评分比较", "两步", "two-step", "score comparison", "dic评分", "ist h dic", "isth dic"],
    },
    {
        "file": "14_common_questions.md",
        "title": "SIC常见问题与误区",
        "title_en": "Common SIC questions and misconceptions",
        "keywords": ["是不是", "能不能排除", "一定", "必须", "误区", "正常", "高就是", "低就是", "myth", "exclude", "always"],
    },
    {
        "file": "16_predictor_context.md",
        "title": "本项目12个预测变量的解释边界",
        "title_en": "Clinical context and interpretation boundaries for the 12 predictors",
        "keywords": ["pt max", "sofa", "po2", "bilirubin", "liver disease", "neutrophil", "rdw", "mchc", "chloride", "hemoglobin", "lactate", "预测变量", "模型变量", "predictor", "feature context"],
    },
]

def _read_knowledge_file(filename: str) -> str:
    path = KNOWLEDGE_DIR / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def get_relevant_knowledge_modules(query: str, max_modules: int = QA_MAX_KNOWLEDGE_MODULES) -> List[Dict[str, Any]]:
    """Select a small set of local SIC knowledge modules using transparent keyword scoring.

    This is deliberately lightweight: no vector database or embedding model is required,
    which keeps the Streamlit demo compatible with the existing Python environment.
    """
    q = (query or "").strip().lower()
    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, module in enumerate(KNOWLEDGE_MODULES):
        score = 0
        for kw in module["keywords"]:
            token = str(kw).lower()
            if token and token in q:
                # Longer, more specific keyword hits are slightly more informative.
                score += 2 if len(token) >= 4 else 1
        if score > 0:
            scored.append((score, -idx, module))

    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    selected = [item[2] for item in scored[:max_modules]]

    # Fallback for broad questions such as "介绍一下SIC".
    if not selected:
        selected = [KNOWLEDGE_MODULES[0], KNOWLEDGE_MODULES[2]]

    # A question about DIC benefits from both progression and mechanism modules.
    if "dic" in q or "弥散性血管内凝血" in q:
        wanted = {"04_sic_to_dic.md", "02_pathophysiology.md"}
        for module in KNOWLEDGE_MODULES:
            if module["file"] in wanted and module not in selected:
                selected.append(module)

    # If the user explicitly asks for references, ensure the reference registry is kept
    # even when the fast-mode module cap is small.
    if any(token in q for token in ["文献", "参考", "来源", "doi", "pmid", "reference", "citation"]):
        ref_module = {"file": "15_references.md", "title": "主要参考文献", "title_en": "Key references", "keywords": []}
        selected = [m for m in selected if m.get("file") != "15_references.md"]
        if max_modules <= 1:
            selected = [ref_module]
        else:
            selected = selected[: max_modules - 1] + [ref_module]

    # Deduplicate while preserving order and enforce the fast-mode cap.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for module in selected:
        if module["file"] not in seen:
            deduped.append(module)
            seen.add(module["file"])
    return deduped[:max_modules]


def _module_title(module: Dict[str, Any], language: str = "zh") -> str:
    if language == "en":
        return str(module.get("title_en") or module.get("title") or module.get("file", ""))
    return str(module.get("title") or module.get("file", ""))


def load_relevant_knowledge(query: str, max_modules: int = QA_MAX_KNOWLEDGE_MODULES, language: str = "zh") -> Tuple[str, List[str]]:
    modules = get_relevant_knowledge_modules(query, max_modules=max_modules)
    chunks: List[str] = []
    titles: List[str] = []
    for module in modules:
        text = _read_knowledge_file(module["file"])
        if text:
            chunks.append(text)
            titles.append(_module_title(module, language=language))
    return "\n\n---\n\n".join(chunks), titles


def load_patient_knowledge(language: str = "zh") -> Tuple[str, List[str]]:
    """Knowledge bundle used for patient-level model explanations."""
    # V3-Expanded: three focused modules are retained for legacy patient-level explanation calls.
    # Mechanistic background is intentionally not injected unless the user asks for it.
    filenames = [
        "09_model_interpretation.md",
        "03_diagnostic_criteria.md",
        "05_laboratory_markers.md",
    ]
    chunks: List[str] = []
    titles: List[str] = []
    lookup = {m["file"]: _module_title(m, language=language) for m in KNOWLEDGE_MODULES}
    for filename in filenames:
        text = _read_knowledge_file(filename)
        if text:
            chunks.append(text)
            titles.append(lookup.get(filename, filename))
    return "\n\n---\n\n".join(chunks), titles


def get_knowledge_status(language: str = "zh") -> Dict[str, Any]:
    available = []
    for module in KNOWLEDGE_MODULES:
        if (KNOWLEDGE_DIR / module["file"]).exists():
            available.append(_module_title(module, language=language))
    return {
        "version": "SIC-KB v1.1",
        "available_modules": available,
        "count": len(available),
    }


def _extract_content(data: Dict[str, Any]) -> str:
    """Parse common OpenAI-compatible chat completion response shapes."""
    try:
        choices = data.get("choices") or []
        if choices:
            choice0 = choices[0] or {}
            message = choice0.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str):
                            parts.append(text)
                joined = "\n".join(parts).strip()
                if joined:
                    return joined
            text = choice0.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    except Exception:
        pass

    # A few private gateways wrap the upstream response.
    for key in ("data", "result"):
        wrapped = data.get(key)
        if isinstance(wrapped, dict):
            parsed = _extract_content(wrapped)
            if parsed:
                return parsed
    return ""


def _response_meta(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return safe completion metadata without exposing reasoning_content."""
    choices = data.get("choices") or []
    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice0.get("message") or {}
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "finish_reason": choice0.get("finish_reason"),
        "has_reasoning_content": bool(message.get("reasoning_content")),
        "has_content": bool(message.get("content")),
    }


def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1400,
) -> str:
    """Call the OpenAI-compatible endpoint and return only final answer text.

    The current ``openai/dsv4`` gateway may return ``reasoning_content`` first.
    With a small token budget, the response can end with ``finish_reason=length``
    before any user-visible ``content`` is produced. In that specific case we
    retry once with a larger budget. Internal reasoning is never returned to the
    Streamlit UI.
    """
    cfg = get_llm_config()
    ok, status = get_llm_status()
    if not ok:
        raise RuntimeError(status)

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def _post(payload: Dict[str, Any]):
        try:
            return HTTP_SESSION.post(
                cfg["api_url"],
                headers=headers,
                json=payload,
                timeout=(10, 180),
            )
        except requests.Timeout as exc:
            raise RuntimeError("LLM 接口请求超时，请稍后重试。") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"无法连接 LLM 接口：{exc}") from exc

    # DSV4 is a reasoning-capable model. The gateway counts hidden reasoning
    # tokens inside ``max_tokens``. A budget that is perfectly adequate for the
    # visible report can therefore be exhausted before ``message.content`` is
    # emitted. Use a reasoning-aware completion budget while keeping the visible
    # response concise through the prompt. A larger ceiling does NOT mean the
    # model must consume all of those tokens; it only prevents premature cutoff.
    requested_budget = int(max_tokens)
    model_name = str(cfg.get("model") or "").lower()
    is_dsv4 = "dsv4" in model_name or "deepseek-v4" in model_name
    if is_dsv4:
        if requested_budget <= 550:       # planner / short structured JSON
            effective_budget = max(requested_budget, 2600)
        elif requested_budget <= 850:     # case-specific follow-up
            effective_budget = max(requested_budget, 3800)
        elif requested_budget <= 1800:    # structured case report
            effective_budget = max(requested_budget, 6800)
        else:
            effective_budget = requested_budget
    else:
        effective_budget = requested_budget

    payload: Dict[str, Any] = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(effective_budget),
        "stream": False,
    }
    response = _post(payload)

    # Some OpenAI-compatible gateways reject optional generation parameters.
    if response.status_code == 400:
        response = _post({
            "model": cfg["model"],
            "messages": messages,
            "stream": False,
        })

    if not response.ok:
        body = response.text.strip().replace("\n", " ")
        if len(body) > 400:
            body = body[:400] + "..."
        raise RuntimeError(f"LLM 接口返回 HTTP {response.status_code}: {body}")

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("LLM 接口返回的不是有效 JSON。") from exc

    content = _extract_content(data)
    if content:
        return content

    meta = _response_meta(data)
    # dsv4/vLLM can spend the whole small budget in hidden reasoning. Retry with
    # a larger completion budget and an explicit request for a concise final answer.
    if meta.get("finish_reason") == "length" and meta.get("has_reasoning_content"):
        retry_messages = [dict(m) for m in messages]
        if retry_messages:
            last = retry_messages[-1]
            if last.get("role") == "user":
                suffix = (
                    "\n\n重要：请控制内部推理长度，并务必在本次响应中输出简洁的最终回答。"
                    if any("中文" in str(m.get("content", "")) or "用户问题" in str(m.get("content", "")) for m in messages)
                    else "\n\nImportant: keep internal reasoning brief and make sure to produce a concise final answer in this response."
                )
                last["content"] = str(last.get("content", "")) + suffix
        # The previous P0 ceiling (4,200) was still too small for DSV4 on
        # evidence-rich case prompts: hidden reasoning could consume the entire
        # budget and leave ``content`` empty. Retry once with a substantially
        # larger ceiling.
        retry_budget = min(max(int(effective_budget * 1.5), int(requested_budget * 6), 7000), 12000)
        retry_payload = {
            "model": cfg["model"],
            "messages": retry_messages,
            "temperature": min(float(temperature), 0.1),
            "max_tokens": retry_budget,
            "stream": False,
        }
        retry_response = _post(retry_payload)
        if retry_response.status_code == 400:
            retry_response = _post({
                "model": cfg["model"],
                "messages": retry_messages,
                "stream": False,
            })
        if not retry_response.ok:
            body = retry_response.text.strip().replace("\n", " ")[:400]
            raise RuntimeError(f"LLM 重试返回 HTTP {retry_response.status_code}: {body}")
        try:
            retry_data = retry_response.json()
        except ValueError as exc:
            raise RuntimeError("LLM 重试返回的不是有效 JSON。") from exc
        retry_content = _extract_content(retry_data)
        if retry_content:
            return retry_content
        meta = _response_meta(retry_data)

    # Never expose reasoning_content, request IDs, or gateway internals to the
    # clinical UI. The user-facing message should be actionable and concise.
    if meta.get("finish_reason") == "length":
        raise RuntimeError(
            "LLM 本次推理达到长度上限，尚未生成最终回答。系统已自动扩展额度并重试，但仍未完成；请再次点击生成。"
        )
    raise RuntimeError("LLM 本次未返回可显示的最终回答，请稍后重试。")


def _strip_json_fence(text: str) -> str:
    """Remove a Markdown code fence around a JSON response if present."""
    value = (text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def translate_chat_messages(
    items: List[Dict[str, str]],
    target_language: str = "en",
) -> List[str]:
    """Translate conversation messages for bilingual history display.

    ``openai/dsv4`` is a reasoning model and is not fully reliable at returning a
    strict JSON array for batch translation.  Translate one message at a time
    instead.  This is slower only on the first language switch; translated copies
    are cached by Streamlit session state afterwards.
    """
    if target_language not in {"zh", "en"}:
        raise ValueError("target_language must be 'zh' or 'en'")
    if not items:
        return []

    target_name = "English" if target_language == "en" else "Simplified Chinese"
    system = (
        "You are a precise bilingual medical translator for a sepsis-induced "
        "coagulopathy (SIC) clinical decision-support interface. Translate the "
        f"provided text into {target_name}. Preserve Markdown headings, bullet lists, "
        "numbers, units, mathematical symbols, drug names, and medical abbreviations "
        "such as SIC, DIC, SOFA, SHAP and PT-INR. Do not summarize, answer, explain, "
        "or add any information. Return only the translated text, with no preface "
        "and no quotation marks."
    )

    results: List[str] = []
    for item in items:
        content = str(item.get("content", "")).strip()
        if not content:
            results.append("")
            continue

        role = str(item.get("role", "user"))
        user = (
            f"Message role: {role}\n"
            f"Target language: {target_name}\n\n"
            "Text to translate:\n"
            f"{content}"
        )

        # Allow enough room for Chinese -> English expansion. chat_completion()
        # already retries with a larger budget when dsv4 spends the first budget
        # entirely on hidden reasoning.
        translated = chat_completion(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=2200,
        ).strip()

        if not translated:
            raise RuntimeError("Conversation translation returned an empty response.")
        results.append(translated)

    return results


def _format_patient_context(case_context: Dict[str, Any], language: str = "zh") -> str:
    probability = case_context.get("deploy_prob")
    threshold = case_context.get("threshold")
    pred_code = case_context.get("pred_class_code")
    risk_code = case_context.get("risk_band_code")
    pred_class = case_context.get("pred_class")
    risk_band = case_context.get("risk_band")
    if pred_code is None and probability is not None and threshold is not None:
        pred_code = 1 if float(probability) >= float(threshold) else 0
    if not risk_code and probability is not None and threshold is not None:
        _p = float(probability)
        _th = float(threshold)
        risk_code = "low" if _p < 0.30 else ("intermediate" if _p < _th else "high")
    feature_values = case_context.get("feature_values") or {}
    shap_top = case_context.get("shap_top_features") or []

    if language == "en":
        pred_label = (
            "High 48-hour SIC risk" if pred_code == 1 else
            "Relatively lower 48-hour SIC risk" if pred_code == 0 else
            str(pred_class or "-")
        )
        risk_label = {"low": "Low risk", "intermediate": "Intermediate risk", "high": "High risk"}.get(
            str(risk_code or "").lower(), str(risk_band or "-")
        )
        lines = ["# Current patient model context"]
        if probability is not None:
            lines.append(f"- Calibrated 48-hour SIC risk probability: {float(probability):.4f} ({float(probability)*100:.2f}%)")
        if threshold is not None:
            lines.append(f"- Model decision threshold: {float(threshold):.4f}")
        lines.append(f"- Binary model result: {pred_label}")
        lines.append(f"- Risk stratum shown on the page: {risk_label}")
        lines.append("\n## Model inputs (12 predictors used by the research model)")
        for key, value in feature_values.items():
            try:
                missing = value is None or (isinstance(value, float) and np.isnan(value))
            except Exception:
                missing = value is None
            lines.append(f"- {key}: {'not available' if missing else value}")

        if shap_top:
            lines.append("\n## Leading SHAP contributors")
            for item in shap_top[:8]:
                try:
                    shap_value = float(item.get("shap_value", 0.0))
                    direction = "pushes the model prediction toward higher risk" if shap_value > 0 else "pushes the model prediction toward lower risk"
                    lines.append(
                        f"- {item.get('feature')}: value={item.get('value')}, "
                        f"SHAP={shap_value:+.4f} ({direction})"
                    )
                except Exception:
                    continue
        else:
            lines.append("\n## SHAP\n- No numeric SHAP summary is currently available.")

        lines.append(
            "\nImportant: this is machine-learning risk-prediction context, not an ISTH SIC diagnostic score. "
            "PT max in this model must not be treated as PT-INR. PaO2 max alone cannot establish normal/preserved oxygenation without contemporaneous FiO2/PF ratio/support. "
            "A lower-risk SHAP direction is not a protective effect, and lower predicted SIC risk does not imply mild or stable overall illness."
        )
        return "\n".join(lines)

    pred_label = (
        "48h SIC高风险" if pred_code == 1 else
        "48h SIC风险相对较低" if pred_code == 0 else
        str(pred_class or "-")
    )
    risk_label = {"low": "低风险", "intermediate": "中间风险", "high": "高风险"}.get(
        str(risk_code or "").lower(), str(risk_band or "-")
    )
    lines = ["# 当前患者模型上下文"]
    if probability is not None:
        lines.append(f"- 校准后 48h SIC 风险概率: {float(probability):.4f} ({float(probability)*100:.2f}%)")
    if threshold is not None:
        lines.append(f"- 模型判定阈值: {float(threshold):.4f}")
    lines.append(f"- 二分类结果: {pred_label}")
    lines.append(f"- 页面风险分层: {risk_label}")

    lines.append("\n## 模型输入（研究模型的 12 个预测特征）")
    for key, value in feature_values.items():
        try:
            missing = value is None or (isinstance(value, float) and np.isnan(value))
        except Exception:
            missing = value is None
        lines.append(f"- {key}: {'未提供' if missing else value}")

    if shap_top:
        lines.append("\n## SHAP 贡献度最高的特征")
        for item in shap_top[:8]:
            try:
                shap_value = float(item.get("shap_value", 0.0))
                direction = "推动模型风险升高" if shap_value > 0 else "推动模型风险降低"
                lines.append(
                    f"- {item.get('feature')}: value={item.get('value')}, "
                    f"SHAP={shap_value:+.4f}（{direction}）"
                )
            except Exception:
                continue
    else:
        lines.append("\n## SHAP\n- 当前没有可用的 SHAP 数值摘要。")

    lines.append(
        "\n注意：这是机器学习风险预测上下文，不是 ISTH SIC 诊断评分；"
        "模型输入中的 PT max 不能直接等同于 PT-INR。PaO₂ max 单独不能证明氧合正常/良好，缺少同步FiO₂/PF比值/呼吸支持信息时必须保留解释边界；"
        "SHAP向低风险方向移动不等于保护作用，SIC低风险预测也不等于总体病情轻或稳定。"
    )
    return "\n".join(lines)

def generate_patient_interpretation(case_context: Dict[str, Any], language: str = "zh") -> str:
    knowledge, module_titles = load_patient_knowledge(language=language)
    context_text = _format_patient_context(case_context, language=language)
    if language == "en":
        system = SYSTEM_PROMPT_EN
        task = PATIENT_INTERPRETATION_TASK_EN
    else:
        system = SYSTEM_PROMPT_ZH
        task = PATIENT_INTERPRETATION_TASK_ZH

    if language == "en":
        user_content = (
            f"{task}\n\n"
            f"{context_text}\n\n"
            "# Local SIC knowledge base (V3-Expanded; patient interpretation modules)\n"
            f"Loaded modules: {', '.join(module_titles) if module_titles else 'none'}\n\n"
            f"{knowledge if knowledge else 'No local SIC knowledge is currently loaded.'}\n\n"
            "Output requirement: answer entirely in English."
        )
    else:
        user_content = (
            f"{task}\n\n"
            f"{context_text}\n\n"
            "# SIC本地知识库（V3-Expanded，患者解释专用模块）\n"
            f"已载入模块：{', '.join(module_titles) if module_titles else '无'}\n\n"
            f"{knowledge if knowledge else '当前未加载本地SIC知识库。'}\n\n"
            "输出要求：全程使用中文。"
        )
    return chat_completion(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=PATIENT_MAX_TOKENS,
    )


def _question_needs_patient_context(question: str) -> bool:
    """Return True only when the question explicitly refers to the current case.

    Generic educational questions should not carry the 12-feature patient payload or
    SHAP table; that extra context slows generation and can distract from the question.
    """
    q = (question or "").strip().lower()
    patient_tokens = [
        "当前患者", "这个患者", "这位患者", "本患者", "本例", "这个病例",
        "当前病人", "这个病人", "结合患者", "结合当前", "我的结果",
        "当前预测", "这次预测", "模型结果", "shap结果", "为什么这么高",
        "为什么这么低", "current patient", "this patient", "this case",
        "my result", "current prediction", "this prediction", "shap result",
    ]
    return any(token in q for token in patient_tokens)


def answer_sic_question(
    question: str,
    case_context: Optional[Dict[str, Any]] = None,
    language: str = "zh",
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空。")

    system = SYSTEM_PROMPT_EN if language == "en" else SYSTEM_PROMPT_ZH
    knowledge, module_titles = load_relevant_knowledge(question, max_modules=QA_MAX_KNOWLEDGE_MODULES, language=language)
    include_patient = bool(case_context) and _question_needs_patient_context(question)
    patient_context = _format_patient_context(case_context, language=language) if include_patient else ""

    if language == "en":
        grounding = (
            "Use the retrieved local SIC knowledge below as the primary factual basis. "
            "Do not override diagnostic thresholds or treatment-evidence boundaries from the local knowledge with model memory. "
            "If the local knowledge is insufficient for a specific conclusion, state that the current local knowledge base does not provide enough evidence.\n\n"
            f"# Retrieved knowledge modules\n{', '.join(module_titles) if module_titles else 'none'}\n\n"
            f"# Local SIC knowledge base V3-Expanded\n{knowledge if knowledge else 'No local SIC knowledge is currently loaded.'}\n\n"
            + (f"# Current patient model context\n{patient_context}\n\n" if include_patient else "")
            + f"# User question\n{question}\n\n"
            + "Output requirement: answer entirely in English."
        )
    else:
        grounding = (
            "请优先根据下面检索到的本地SIC知识模块回答。不得用模型自身记忆覆盖或篡改本地知识库中的诊断阈值、治疗证据边界。"
            "若知识库不足以支持具体结论，请明确说明‘当前本地知识库未提供足够依据’。\n\n"
            f"# 本次检索到的知识模块\n{', '.join(module_titles) if module_titles else '无'}\n\n"
            f"# SIC本地知识库 V3-Expanded\n{knowledge if knowledge else '当前未加载本地SIC知识库。'}\n\n"
            + (f"# 当前患者模型上下文\n{patient_context}\n\n" if include_patient else "")
            + f"# 用户问题\n{question}\n\n"
            + "输出要求：全程使用中文。"
        )


    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    if history:
        # V3-Expanded: keep only a short recent context window and cap each message length.
        for item in history[-QA_MAX_HISTORY_MESSAGES:]:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                compact = content.strip()[:QA_HISTORY_CHAR_LIMIT]
                messages.append({"role": role, "content": compact})
    messages.append({"role": "user", "content": grounding})
    return chat_completion(messages, temperature=0.2, max_tokens=QA_MAX_TOKENS)


def question_uses_patient_context(question: str, case_context: Optional[Dict[str, Any]] = None) -> bool:
    """Public helper for UI diagnostics."""
    return bool(case_context) and _question_needs_patient_context(question)
