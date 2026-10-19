# /usr/bin/python3 -m streamlit run app.py
import os
import json
import warnings
import time
import hashlib
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

from sic_agent import get_llm_status, translate_chat_messages
from agent import answer_case_followup, generate_case_report

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="脓毒症诱导的凝血病（SIC）48h早期风险预测模型",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_META_PATH = os.path.join(BASE_DIR, "final_model_metadata.json")
RAW_MODEL_FALLBACK = os.path.join(BASE_DIR, "final_model_XGBoost_raw.pkl")
DEPLOY_MODEL_FALLBACK = os.path.join(BASE_DIR, "final_model_XGBoost_sigmoid.pkl")
LEGACY_DEPLOY_MODEL_FALLBACK = os.path.join(BASE_DIR, "final_model_XGBoost_isotonic.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "fa_sic_48h_mice_bayesridge_preprocessor.joblib")
PREPROCESS_META_PATH = os.path.join(BASE_DIR, "fa_sic_48h_mice_bayesridge_preprocess_meta.json")
BEST_REPORT_PATH = os.path.join(BASE_DIR, "best_model_report.txt")
EXAMPLE_CASES_PATH = os.path.join(BASE_DIR, "examples", "fah_hmuicu_example_cases.csv")
SOURCE_REGISTRY_PATH = os.path.join(BASE_DIR, "knowledge", "source_registry.json")
INPUT_STATE_SCHEMA_VERSION = "blank-initial-inputs-v9"


def _example_dataset_signature() -> str:
    """Stable fingerprint used to invalidate stale Streamlit session data after case-file updates."""
    try:
        with open(EXAMPLE_CASES_PATH, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "missing-example-dataset"


def _reset_stale_example_session_if_needed() -> None:
    """Clear old case/widget state when the bundled FAH example dataset changes.

    Streamlit hot-reload can preserve Session State across code/data replacements.
    Without this guard, a label such as FAH-01 could display values from an older
    example file even after the project bundle has been updated.
    """
    sig = _example_dataset_signature()
    state_key = "fah_example_dataset_signature"
    old_sig = st.session_state.get(state_key)
    schema_key = "sic_input_state_schema_version"
    old_schema = st.session_state.get(schema_key)
    schema_changed = old_schema != INPUT_STATE_SCHEMA_VERSION
    stale_case_present = any(
        key in st.session_state
        for key in ("loaded_fah_example", "sic_current_case", "sic_last_prediction", "sic_single_agent_report")
    )
    dataset_changed = old_sig != sig and (old_sig is not None or stale_case_present)
    if schema_changed or dataset_changed:
        exact_keys = {
            "delta_input_mode_selector", "delta_lactate_input", "delta_hemoglobin_input",
            "lactate_earlier", "lactate_later", "hemoglobin_earlier", "hemoglobin_later",
            "loaded_fah_example", "loaded_fah_outcome", "fah_example_selector",
            "sic_last_prediction", "sic_last_shap_result", "sic_current_case",
            "sic_single_agent_report", "sic_single_agent_history", "last_deploy_prob",
        }
        for key in list(st.session_state.keys()):
            if key.startswith("input_") or key in exact_keys or key.startswith("single_agent_pending_"):
                st.session_state.pop(key, None)
    st.session_state[state_key] = sig
    st.session_state[schema_key] = INPUT_STATE_SCHEMA_VERSION

FEATURE_META: Dict[str, Dict[str, Any]] = {
    "sofa_first_day": {
        "label": "SOFA score (first day)",
        "cn": "首日 SOFA 评分",
        "unit": "points",
        "hard_range": (0.0, 24.0),
        "typical_range": (0.0, 15.0),
        "default": 6.0,
        "step": 1.0,
        "format": "%.0f",
        "help_zh": "推荐输入首日 SOFA 总分。常见范围 0–15，极重症可更高。",
        "help_en": "Enter the first-day total SOFA score.",
    },
    "po2_max": {
        "label": "PaO₂ max",
        "cn": "氧分压最大值",
        "unit": "mmHg",
        "hard_range": (20.0, 600.0),
        "typical_range": (60.0, 300.0),
        "default": 110.0,
        "step": 1.0,
        "format": "%.1f",
        "help_zh": "动脉血氧分压最大值。吸氧/机械通气下可更高。",
        "help_en": "Maximum arterial oxygen partial pressure within 24h.",
    },
    "chloride_min": {
        "label": "Chloride min",
        "cn": "血氯最小值",
        "unit": "mmol/L",
        "hard_range": (70.0, 140.0),
        "typical_range": (95.0, 110.0),
        "default": 100.0,
        "step": 0.1,
        "format": "%.1f",
        "help_zh": "血氯最小值。正常多约 98–106 mmol/L。",
        "help_en": "Minimum chloride level within 24h.",
    },
    "hemoglobin_min": {
        "label": "Hemoglobin min",
        "cn": "血红蛋白最小值",
        "unit": "g/dL",
        "hard_range": (3.0, 20.0),
        "typical_range": (7.0, 16.0),
        "default": 10.0,
        "step": 0.1,
        "format": "%.1f",
        "help_zh": "血红蛋白最小值。重症患者常见贫血。",
        "help_en": "Minimum hemoglobin within 24h.",
    },
    "abs_neutrophils_max": {
        "label": "Absolute neutrophils max",
        "cn": "中性粒细胞绝对值最大值",
        "unit": "×10⁹/L",
        "hard_range": (0.0, 100.0),
        "typical_range": (1.5, 25.0),
        "default": 12.0,
        "step": 0.1,
        "format": "%.2f",
        "help_zh": "中性粒细胞绝对值最大值。感染/炎症时可显著升高。",
        "help_en": "Maximum absolute neutrophil count within 24h.",
    },
    "bilirubin_total_max": {
        "label": "Total bilirubin max",
        "cn": "总胆红素最大值",
        "unit": "mg/dL",
        "hard_range": (0.1, 80.0),
        "typical_range": (0.2, 10.0),
        "default": 1.2,
        "step": 0.1,
        "format": "%.1f",
        "help_zh": "总胆红素最大值。肝胆功能异常时升高。",
        "help_en": "Maximum total bilirubin within 24h.",
    },
    "pt_max": {
        "label": "PT max",
        "cn": "凝血酶原时间最大值",
        "unit": "s",
        "hard_range": (8.0, 150.0),
        "typical_range": (10.0, 30.0),
        "default": 14.0,
        "step": 0.1,
        "format": "%.1f",
        "help_zh": "凝血酶原时间最大值。SIC/凝血障碍时常延长。",
        "help_en": "Maximum prothrombin time within 24h.",
    },
    "delta_lactate": {
        "label": "Δ Lactate",
        "cn": "乳酸变化值",
        "unit": "mmol/L",
        "hard_range": (-10.0, 30.0),
        "typical_range": (-2.0, 8.0),
        "default": 1.0,
        "step": 0.1,
        "format": "%.1f",
        "help_zh": "按“后值 - 前值”自动计算。",
        "help_en": "Automatically computed as later value - earlier value.",
    },
    "delta_hemoglobin": {
        "label": "Δ Hemoglobin",
        "cn": "血红蛋白变化值",
        "unit": "g/dL",
        "hard_range": (-20.0, 10.0),
        "typical_range": (-5.0, 1.0),
        "default": -1.0,
        "step": 0.1,
        "format": "%.1f",
        "help_zh": "按“后值 - 前值”自动计算。",
        "help_en": "Automatically computed as later value - earlier value.",
    },
    "rdw": {
        "label": "RDW",
        "cn": "红细胞分布宽度",
        "unit": "%",
        "hard_range": (10.0, 30.0),
        "typical_range": (11.5, 18.0),
        "default": 15.0,
        "step": 0.1,
        "format": "%.1f",
        "help_zh": "RDW 常见参考约 11.5–14.5%。",
        "help_en": "Red cell distribution width.",
    },
    "mchc": {
        "label": "MCHC",
        "cn": "平均红细胞血红蛋白浓度",
        "unit": "g/dL",
        "hard_range": (25.0, 40.0),
        "typical_range": (30.0, 36.0),
        "default": 33.0,
        "step": 0.1,
        "format": "%.1f",
        "help_zh": "MCHC 常见参考约 32–36 g/dL。",
        "help_en": "Mean corpuscular hemoglobin concentration.",
    },
    "liver_disease": {
        "label": "Liver disease",
        "cn": "是否有肝脏疾病",
        "unit": "0/1",
        "hard_range": (0, 1),
        "typical_range": (0, 1),
        "default": 0,
        "step": 1,
        "format": "%d",
        "help_zh": "0=否，1=是。",
        "help_en": "0 = No, 1 = Yes.",
    },
}
FEATURE_ORDER = list(FEATURE_META.keys())
DEFAULT_THRESHOLD = 0.625

I18N = {
    "zh": {
        "title": "脓毒症诱导的凝血病（SIC）48h早期风险预测模型",
        "subtitle": "基于患者入 ICU 后 24 小时内的关键临床信息，对 48 小时内发生脓毒症诱导的凝血病（SIC）的风险进行早期评估。",
        "best_model": "模型类型",
        "deploy_threshold": "判定阈值",
        "test_auc_ap": "模型性能（AUC / AP）",
        "task_chip": "ICU 0–24h data → 48h SIC risk",
        "calibr_chip": "风险概率校准",
        "lang_label": "🌐 语言 / Language",
        "lang_zh": "中文",
        "lang_en": "English",
        "summary": "模型摘要",
        "summary_missing": "未找到模型摘要文件。",
        "usage": "📝 使用说明",
        "usage_md": "- 本页面基于 **12 个核心临床特征** 进行评估。\n- 建议录入 ICU 入科后 **0–24h** 内的代表性指标。\n- 输出结果用于 **48h 内 SIC 早期风险识别与辅助评估**，不能替代医师诊断。",
        "range_ref": "📚 临床输入范围参考",
        "input_title": "🧾 患者临床信息录入（入 ICU 后 0–24 小时）",
        "input_note": "请按临床类别填写关键指标。变化值特征可由前后两次检测结果自动计算。",
        "grp1": "凝血 / 血常规",
        "grp2": "代谢 / 肝功能",
        "grp3": "病情评估 / 氧合 / 炎症",
        "grp4": "动态变化",
        "delta_mode_pair": "前后测量值",
        "delta_mode_direct": "直接输入变化值",
        "delta_toggle_label": "变化值输入方式",
        "delta_direct_note": "直接输入：ΔLactate = {dl:.2f} mmol/L，ΔHemoglobin = {dh:.2f} g/dL",
        "lactate_earlier": "乳酸前值",
        "lactate_later": "乳酸后值",
        "hb_earlier": "血红蛋白前值",
        "hb_later": "血红蛋白后值",
        "auto_delta": "自动计算：ΔLactate = {dl:.2f} mmol/L，ΔHemoglobin = {dh:.2f} g/dL",
        "run_pred": "开始风险评估",
        "input_preview": "📋 本次录入信息预览",
        "prediction": "📌 风险评估结果",
        "pred_prob": "SIC 风险概率",
        "pred_class": "风险判定",
        "pred_high": "提示 48h 内 SIC 高风险",
        "pred_low": "提示 48h 内 SIC 风险相对较低",
        "low_risk": "低风险",
        "mid_risk": "中间风险",
        "high_risk": "高风险",
        "threshold_note": "当前判定阈值为 <b>{th:.3f}</b>。当预测概率大于等于该阈值时，提示患者在 <b>48 小时内发生 SIC 的风险较高</b>。",
        "shap": "🧠 个体解释（SHAP）",
        "waterfall": "Waterfall plot",
        "force": "Force plot",
        "waterfall_na": "Waterfall plot 当前不可用。",
        "force_na": "Force plot 当前不可用。",
        "methods": "使用说明",
        "methods_md": "- 本系统用于评估 ICU 入科后 48 小时内发生 SIC 的风险，输入信息来自 ICU 入科后前 24 小时。\n- 预测结果应结合患者临床表现、实验室检查及动态变化综合判断。\n- 本系统用于辅助风险识别与临床评估，不能替代医师诊断或治疗决策。",
        "err_range": "{label} 超出允许范围 [{lo}, {hi}]。",
        "err_missing": "{label} 当前为空，请重新确认该输入值。",
        "err_hbmin": "hemoglobin_min 不应大于前后两次血红蛋白中的较小值。",
        "err_model_none": "当前运行环境无法加载保存的模型，请尽量与训练环境保持一致后重试，尤其是 scikit-learn / xgboost 版本。",
        "pred_fail": "基于 12 特征直接预测失败：{detail}",
        "shap_na": "当前运行环境未安装 SHAP。",
        "shap_fail": "SHAP 解释失败：{detail}",
        "inference_path": "推理路径：{mode}",
        "raw_prob_note": "模型输出概率",
        "deploy_prob_note": "校准后风险概率",
        "prob_change_note": "较上次评估风险变化：{delta:+.6f}",
        "first_pred_note": "当前为本次会话首次评估",
        "recomputed_note": "结果基于当前录入信息计算",
        "current_threshold_note": "当前使用判定阈值 = {th:.3f}",
        "prob_source_label": "风险概率来源",
        "risk_note": "风险分层",
        "risk_band": "风险等级",
        "explainer_model": "个体解释模型：{model}",
        "yes": "是",
        "no": "否",
        "feature_col": "Feature",
        "meaning_col": "参考",
        "unit_col": "单位",
        "allow_col": "允许范围",
        "typical_col": "常见范围",
    },
    "en": {
        "title": "Sepsis-Induced Coagulopathy (SIC) 48h Early Risk Prediction Model",
        "subtitle": "Use key clinical information from the first 24 hours after ICU admission to estimate the patient-specific risk of developing SIC within 48 hours.",
        "best_model": "Model type",
        "deploy_threshold": "Decision threshold",
        "test_auc_ap": "Model performance (AUC / AP)",
        "task_chip": "ICU 0–24h data → 48h SIC risk",
        "calibr_chip": "Risk calibration",
        "lang_label": "🌐 Language",
        "lang_zh": "中文",
        "lang_en": "English",
        "summary": "Model summary",
        "summary_missing": "Model summary file not found.",
        "usage": "📝 Notes",
        "usage_md": "- This page evaluates risk using **12 core clinical features**.\n- Use representative values collected within the first **0–24h after ICU admission**.\n- The output supports **early SIC risk identification within 48h** and does not replace physician diagnosis.",
        "range_ref": "📚 Clinical input range reference",
        "input_title": "🧾 Patient clinical input (ICU 0–24h)",
        "input_note": "Enter key variables by clinical category. Delta features can be computed automatically from earlier and later measurements.",
        "grp1": "Coagulation / CBC",
        "grp2": "Metabolism / Liver function",
        "grp3": "Severity / Oxygenation / Inflammation",
        "grp4": "Dynamic changes",
        "delta_mode_pair": "Earlier/later measurements",
        "delta_mode_direct": "Enter change directly",
        "delta_toggle_label": "Delta input mode",
        "delta_direct_note": "Direct input: ΔLactate = {dl:.2f} mmol/L, ΔHemoglobin = {dh:.2f} g/dL",
        "lactate_earlier": "Lactate earlier",
        "lactate_later": "Lactate later",
        "hb_earlier": "Hemoglobin earlier",
        "hb_later": "Hemoglobin later",
        "auto_delta": "Auto-computed: ΔLactate = {dl:.2f} mmol/L, ΔHemoglobin = {dh:.2f} g/dL",
        "run_pred": "Start risk assessment",
        "input_preview": "📋 Input summary",
        "prediction": "📌 Risk assessment result",
        "pred_prob": "Estimated SIC risk probability",
        "pred_class": "Risk classification",
        "pred_high": "High risk of SIC within 48h",
        "pred_low": "Relatively lower risk of SIC within 48h",
        "low_risk": "Low risk",
        "mid_risk": "Intermediate risk",
        "high_risk": "High risk",
        "threshold_note": "Current decision threshold: <b>{th:.3f}</b>. When the predicted probability is greater than or equal to this threshold, the patient is flagged as <b>having a higher risk of SIC within 48 hours</b>.",
        "shap": "🧠 Individual explanation (SHAP)",
        "waterfall": "Waterfall plot",
        "force": "Force plot",
        "waterfall_na": "Waterfall plot is not available.",
        "force_na": "Force plot is not available.",
        "methods": "Use Notes",
        "methods_md": "- This system estimates the risk of SIC within 48 hours after ICU admission using information from the first 24 hours in the ICU.\n- Prediction results should be interpreted together with the patient's clinical presentation, laboratory findings, and temporal trends.\n- The system supports risk identification and clinical assessment and does not replace physician diagnosis or treatment decisions.",
        "err_range": "{label} is outside the allowed range [{lo}, {hi}].",
        "err_missing": "{label} is empty. Please confirm this input value.",
        "err_hbmin": "Hemoglobin min should not exceed the smaller of earlier and later hemoglobin.",
        "err_model_none": "The saved model could not be loaded in the current runtime. Align the deployment environment with the training environment, especially scikit-learn / xgboost versions.",
        "pred_fail": "Prediction failed on raw 12-feature input: {detail}",
        "shap_na": "SHAP package is not available in the current runtime.",
        "shap_fail": "SHAP explanation failed: {detail}",
        "inference_path": "Inference path: {mode}",
        "raw_prob_note": "Model probability",
        "deploy_prob_note": "Calibrated risk probability",
        "prob_change_note": "Change vs previous assessment: {delta:+.6f}",
        "first_pred_note": "This is the first assessment in the current session",
        "recomputed_note": "Computed from current inputs",
        "current_threshold_note": "Current decision threshold = {th:.3f}",
        "prob_source_label": "Risk probability source",
        "risk_note": "Risk stratum",
        "risk_band": "Risk level",
        "explainer_model": "Individual explanation model: {model}",
        "yes": "Yes",
        "no": "No",
        "feature_col": "Feature",
        "meaning_col": "Meaning",
        "unit_col": "Unit",
        "allow_col": "Allowed range",
        "typical_col": "Typical range",
    },
}

CUSTOM_CSS = """
<style>
/* ===== Overall page ===== */
.block-container {
    padding-top: 2.75rem;
    padding-bottom: 1.8rem;
    max-width: 1660px;
}
html, body, [class*="css"] {font-size: 15px;}
.stMarkdown p, .stMarkdown li {line-height: 1.62;}
.stMarkdown, .stCaption, label, button, [data-testid="stWidgetLabel"] {
    overflow-wrap: anywhere;
    word-break: break-word;
}

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background: #f8fafc;
    position: sticky !important;
    top: 0 !important;
    align-self: flex-start !important;
    height: 100vh !important;
    overflow: hidden !important;
    border-right: 1px solid #e5e7eb;
}
[data-testid="stSidebarContent"] {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow-y: auto !important;
    scrollbar-width: thin;
    padding-top: 0.75rem;
}
[data-testid="stSidebar"] button[kind="header"] {
    position: sticky !important;
    top: 0.35rem !important;
    z-index: 20 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {font-size: 0.88rem; line-height: 1.5;}

/* ===== Hero header ===== */
.main-card {
    background: linear-gradient(135deg, #f8fafc 0%, #eef4ff 56%, #fff8f1 100%);
    border: 1px solid #dbe5f1;
    border-radius: 24px;
    padding: 1.45rem 1.65rem;
    margin: 0.2rem 0 1.15rem 0;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.055);
}
.header-wrap {display:flex; justify-content:space-between; align-items:center; gap:1.2rem;}
.header-left {flex:1; min-width:0;}
.header-title {
    font-size: 2.22rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    color:#0f172a;
    line-height:1.24;
}
.header-sub {
    font-size:1rem;
    color:#64748b;
    margin-top:0.42rem;
    line-height:1.62;
    max-width: 980px;
}
.header-tags {margin-top:0.8rem; display:flex; flex-wrap:wrap; gap:0.5rem;}
.header-tag {
    display:inline-flex; align-items:center; gap:0.35rem;
    padding:0.42rem 0.72rem; border-radius:999px;
    background:rgba(255,255,255,0.88); border:1px solid #dbeafe;
    color:#334155; font-size:0.82rem; font-weight:700;
}
.header-right {
    width: 150px; min-width:150px; height:105px;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    background: linear-gradient(135deg, rgba(37,99,235,0.07), rgba(239,68,68,0.08));
    border-radius:20px; border:1px solid rgba(148,163,184,0.18);
}
.header-hero {font-size:2.9rem; line-height:1;}
.header-hero-text {font-size:0.8rem; color:#64748b; font-weight:750; margin-top:0.35rem; letter-spacing:0.03em;}

/* ===== Shared section rhythm ===== */
.section-title {
    font-size:1.08rem; font-weight:850; margin:0.45rem 0 0.42rem 0;
    color:#0f172a; line-height:1.35;
}
.small-note {color:#64748b; font-size:0.88rem; line-height:1.5; margin-bottom:0.55rem;}
.caption-box {
    background:#f8fafc; border-left:3px solid #2563eb;
    padding:0.72rem 0.85rem; border-radius:10px;
    color:#475569; font-size:0.86rem; line-height:1.55;
}
.small-chip-note {margin-top:0.45rem; font-size:0.81rem; color:#0f766e; font-weight:650;}
.fah-note {
    background:#fffaf0; border-left:3px solid #f59e0b;
    padding:0.6rem 0.75rem; border-radius:8px;
    color:#9a3412; font-size:0.82rem; margin:0.35rem 0 0.55rem 0; line-height:1.45;
}

/* ===== Inputs ===== */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextArea"] textarea {border-radius:10px !important;}
[data-testid="stWidgetLabel"] p {
    white-space: normal !important;
    line-height:1.28 !important;
    font-size:0.82rem !important;
}
div[data-testid="stHorizontalBlock"] {align-items:flex-start;}
.stButton button, .stFormSubmitButton button {
    white-space:normal !important; min-height:2.35rem; line-height:1.25;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color:#e5e7eb !important;
    border-radius:14px !important;
}

/* ===== Prediction ===== */
.risk-low {background:#ecfdf5; color:#065f46; padding:0.32rem 0.62rem; border-radius:999px; font-weight:750;}
.risk-mid {background:#fffbeb; color:#92400e; padding:0.32rem 0.62rem; border-radius:999px; font-weight:750;}
.risk-high {background:#fef2f2; color:#991b1b; padding:0.32rem 0.62rem; border-radius:999px; font-weight:750;}
.risk-summary-card {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:14px;
    padding:0.9rem 1rem; margin-top:0.3rem;
    height:8.4rem; box-sizing:border-box;
    display:flex; flex-direction:column; justify-content:center;
}
.risk-prob {font-size:1.72rem; font-weight:900; color:#0f172a; line-height:1.1;}
.risk-label {font-size:0.8rem; color:#64748b; margin-bottom:0.35rem;}
.risk-copy {font-size:0.86rem; color:#475569; line-height:1.5; margin-top:0.55rem;}

/* ===== Agent panel ===== */
.agent-hero-card {
    display:flex; align-items:center; gap:0.72rem;
    background:linear-gradient(135deg,#f8fbff 0%,#ffffff 100%);
    border:1px solid #dbeafe; border-radius:15px;
    padding:0.9rem 1rem; margin:0 0 0.72rem 0;
}
.agent-hero-icon {
    width:2.35rem; height:2.35rem; border-radius:11px;
    display:flex; align-items:center; justify-content:center;
    background:#eff6ff; border:1px solid #dbeafe;
    font-size:1.15rem; flex:0 0 auto;
}
.agent-hero-title {font-weight:850; color:#0f172a; font-size:1.02rem; line-height:1.3;}
.agent-hero-sub {color:#64748b; font-size:0.83rem; line-height:1.48; margin-top:0.16rem;}
.agent-report-note {font-size:0.79rem; color:#64748b; line-height:1.45; margin-top:0.5rem;}

/* Right-column report typography */
[data-testid="stMainBlockContainer"] h3 {
    font-size:1.04rem !important;
    margin-top:1.12rem !important;
    margin-bottom:0.45rem !important;
    line-height:1.35 !important;
}
[data-testid="stMainBlockContainer"] h4 {font-size:0.98rem !important; line-height:1.35 !important;}
[data-testid="stChatMessage"] {padding:0.65rem 0.75rem;}

@media (max-width: 1100px) {
    .block-container {max-width:100%; padding-left:0.8rem; padding-right:0.8rem;}
    .header-title {font-size:1.72rem;}
    .header-right {display:none;}
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


class MissingIndicatorWithNames:
    def __init__(self, feature_names=None, missing_values=np.nan, features="missing-only"):
        self.feature_names = feature_names
        self.missing_values = missing_values
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features if input_features is not None else [], dtype=object)


def get_lang() -> str:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "zh"
    return st.session_state["lang"]


def tr(key: str, **kwargs) -> str:
    txt = I18N.get(get_lang(), I18N["zh"]).get(key, key)
    return txt.format(**kwargs) if kwargs else txt


def read_json(path: str, default: Dict[str, Any]) -> Dict[str, Any]:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: str, default: str = "") -> str:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return f.read()



def resolve_relative_path(path_str: str, fallback_paths: List[str]) -> str:
    candidates: List[str] = []
    if path_str:
        candidates.append(path_str)
        candidates.append(os.path.join(BASE_DIR, path_str))
    candidates.extend(fallback_paths)
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return path_str or (fallback_paths[0] if fallback_paths else "")


def get_calibration_method(model_meta: Dict[str, Any]) -> str:
    cal_info = model_meta.get("calibration_info", {}) if isinstance(model_meta, dict) else {}
    method = cal_info.get("method") or cal_info.get("requested_method") or "none"
    return str(method).strip() or "none"


def get_probability_source(model_meta: Dict[str, Any]) -> str:
    src = model_meta.get("deployment_probability_source", "calibrated")
    return str(src).strip() or "calibrated"


def format_calibration_label(model_meta: Dict[str, Any]) -> str:
    method = get_calibration_method(model_meta).lower()
    mapping = {"sigmoid": "Sigmoid", "isotonic": "Isotonic", "none": "None"}
    return mapping.get(method, method.capitalize())


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Dict[str, Any]:
    model_meta = read_json(MODEL_META_PATH, {})
    raw_model_path = resolve_relative_path(
        model_meta.get("raw_model_path", ""),
        [RAW_MODEL_FALLBACK],
    )
    deploy_model_path = resolve_relative_path(
        model_meta.get("deployment_model_path", ""),
        [DEPLOY_MODEL_FALLBACK, LEGACY_DEPLOY_MODEL_FALLBACK],
    )

    payload: Dict[str, Any] = {
        "raw_model": None,
        "deploy_model": None,
        "raw_model_path": raw_model_path,
        "deploy_model_path": deploy_model_path,
        "model_meta": model_meta,
        "preprocessor": None,
        "preprocess_meta": read_json(PREPROCESS_META_PATH, {}),
        "report_text": read_text(BEST_REPORT_PATH, ""),
        "load_messages": [],
    }

    try:
        if raw_model_path and os.path.exists(raw_model_path):
            payload["raw_model"] = joblib.load(raw_model_path)
        else:
            payload["load_messages"].append(("warning", f"Raw model not found: {raw_model_path}"))
    except Exception as e:
        payload["load_messages"].append(("error", f"Raw model load failed: {repr(e)}"))

    try:
        if deploy_model_path and os.path.exists(deploy_model_path):
            payload["deploy_model"] = joblib.load(deploy_model_path)
        else:
            payload["load_messages"].append(("warning", f"Deployment model not found: {deploy_model_path}"))
    except Exception as e:
        payload["load_messages"].append(("error", f"Deployment model load failed: {repr(e)}"))

    if os.path.exists(PREPROCESSOR_PATH):
        try:
            payload["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
        except Exception:
            pass
    return payload


def build_input_df(user_inputs: Dict[str, Any], feature_order: List[str]) -> pd.DataFrame:
    row = {}
    for col in feature_order:
        val = user_inputs[col]
        row[col] = int(val) if col == "liver_disease" else float(val)
    return pd.DataFrame([row], columns=feature_order)


def compute_derived_features(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Compute delta features without inventing values for empty widgets.

    The form intentionally opens blank.  Missing entries remain ``None`` until
    the user types a value or explicitly loads an FAH example case.
    """
    derived = dict(raw_inputs)
    mode = raw_inputs.get("delta_input_mode", "pair")

    def _num_or_none(value):
        try:
            if value is None:
                return None
            f = float(value)
            return None if np.isnan(f) else f
        except Exception:
            return None

    if mode == "direct":
        derived["delta_lactate"] = _num_or_none(raw_inputs.get("delta_lactate_input"))
        derived["delta_hemoglobin"] = _num_or_none(raw_inputs.get("delta_hemoglobin_input"))
    else:
        lactate_earlier = _num_or_none(raw_inputs.get("lactate_earlier"))
        lactate_later = _num_or_none(raw_inputs.get("lactate_later"))
        hb_earlier = _num_or_none(raw_inputs.get("hemoglobin_earlier"))
        hb_later = _num_or_none(raw_inputs.get("hemoglobin_later"))
        derived["delta_lactate"] = (
            None if lactate_earlier is None or lactate_later is None
            else lactate_later - lactate_earlier
        )
        derived["delta_hemoglobin"] = (
            None if hb_earlier is None or hb_later is None
            else hb_later - hb_earlier
        )
    return derived


def classify_risk(prob: float, threshold: float) -> Tuple[str, str]:
    if prob < 0.30:
        return tr("low_risk"), "risk-low"
    elif prob < threshold:
        return tr("mid_risk"), "risk-mid"
    else:
        return tr("high_risk"), "risk-high"


def unwrap_model_for_shap(model: Any) -> Any:
    candidates = [model]
    attrs = [
        "estimator", "base_estimator", "estimator_", "base_estimator_",
        "classifier", "classifier_", "model", "model_"
    ]
    for attr in attrs:
        if hasattr(model, attr):
            try:
                candidates.append(getattr(model, attr))
            except Exception:
                pass

    if hasattr(model, "calibrated_classifiers_"):
        try:
            for cc in model.calibrated_classifiers_:
                for attr in ["estimator", "base_estimator", "classifier", "model"]:
                    if hasattr(cc, attr):
                        candidates.append(getattr(cc, attr))
        except Exception:
            pass

    for cand in list(candidates):
        if hasattr(cand, "named_steps"):
            try:
                for step_obj in cand.named_steps.values():
                    candidates.append(step_obj)
            except Exception:
                pass

    for cand in candidates:
        name = cand.__class__.__name__.lower()
        module = cand.__class__.__module__.lower()
        if ("xgb" in name or "xgboost" in name or "tree" in module or "forest" in module) and hasattr(cand, "predict"):
            return cand
    return model



def safe_predict(raw_model: Any, deploy_model: Any, X_raw: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    out = {
        "raw_prob": None,
        "deploy_prob": None,
        "pred": None,
        "used_mode": None,
        "error": None,
    }
    try:
        if raw_model is not None:
            out["raw_prob"] = float(raw_model.predict_proba(X_raw)[0, 1])

        model_for_pred = deploy_model if deploy_model is not None else raw_model
        if model_for_pred is None:
            out["error"] = tr("err_model_none")
            return out

        deploy_prob = float(model_for_pred.predict_proba(X_raw)[0, 1])
        pred = int(deploy_prob >= threshold)
        out.update({
            "deploy_prob": deploy_prob,
            "pred": pred,
            "used_mode": "raw_12_features",
        })
        return out
    except Exception as e:
        out["error"] = tr("pred_fail", detail=repr(e))
        return out


def render_force_plot(base_value: float, shap_values: np.ndarray, features: pd.DataFrame):
    shap.force_plot(
        base_value=base_value,
        shap_values=shap_values,
        features=features.iloc[0],
        matplotlib=True,
        show=False,
        figsize=(14, 2.8),
    )
    fig = plt.gcf()
    fig.set_size_inches(12, 2.8)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.18)
    return fig


def explain_with_shap(model: Any, X_raw: pd.DataFrame) -> Dict[str, Any]:
    result = {"waterfall_fig": None, "force_fig": None, "error": None, "used_model": None, "top_features": []}
    if not SHAP_AVAILABLE:
        result["error"] = tr("shap_na")
        return result

    try:
        model_for_shap = model
        X_for_shap = X_raw.copy()
        feature_names = X_raw.columns.tolist()

        # Route B raw model is typically a Pipeline(preprocess, classifier).
        # Use the transformed matrix for SHAP so that the classifier receives
        # exactly the feature space it was trained on.
        if hasattr(model_for_shap, "named_steps") and "preprocess" in model_for_shap.named_steps and "classifier" in model_for_shap.named_steps:
            preprocess = model_for_shap.named_steps["preprocess"]
            classifier = model_for_shap.named_steps["classifier"]
            X_trans = preprocess.transform(X_raw)
            try:
                feature_names = list(preprocess.get_feature_names_out())
            except Exception:
                feature_names = X_raw.columns.tolist()
            X_for_shap = pd.DataFrame(X_trans, columns=feature_names, index=X_raw.index)
            explain_model = classifier
        else:
            explain_model = unwrap_model_for_shap(model_for_shap)

        result["used_model"] = explain_model.__class__.__name__
        explainer = shap.TreeExplainer(explain_model)
        shap_values = explainer.shap_values(X_for_shap)

        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[-1])[0]
        else:
            sv = np.asarray(shap_values)
            if sv.ndim == 3:
                sv = sv[0, :, -1]
            elif sv.ndim == 2:
                sv = sv[0]
            else:
                sv = sv.reshape(-1)

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            base_val = float(np.array(expected_value).reshape(-1)[-1])
        else:
            base_val = float(expected_value)

        # For Route B the transformed feature names usually remain identical to
        # the 12 raw feature names because the preprocess step is numeric imputation.
        display_feature_names = feature_names
        display_data = X_for_shap.iloc[0].values

        explanation = shap.Explanation(
            values=sv,
            base_values=base_val,
            data=display_data,
            feature_names=display_feature_names,
        )

        # Store a compact numerical summary for the LLM assistant.
        # This is deliberately separate from the figures so the assistant can
        # explain the current patient without trying to interpret an image.
        top_items = []
        try:
            values_arr = np.asarray(sv, dtype=float).reshape(-1)
            data_arr = np.asarray(display_data).reshape(-1)
            names_arr = list(display_feature_names)
            order = np.argsort(np.abs(values_arr))[::-1]
            for idx in order[:8]:
                val = data_arr[idx] if idx < len(data_arr) else None
                if isinstance(val, np.generic):
                    val = val.item()
                top_items.append({
                    "feature": str(names_arr[idx]),
                    "value": val,
                    "shap_value": float(values_arr[idx]),
                })
        except Exception:
            top_items = []
        result["top_features"] = top_items

        plt.close("all")
        shap.plots.waterfall(explanation, max_display=12, show=False)
        wf_fig = plt.gcf()
        wf_fig.set_size_inches(8.4, 5.4)
        try:
            wf_fig.subplots_adjust(left=0.28, right=0.88, top=0.95, bottom=0.12)
        except Exception:
            pass
        result["waterfall_fig"] = wf_fig

        plt.close("all")
        result["force_fig"] = render_force_plot(base_val, sv, X_for_shap)
    except Exception as e:
        result["error"] = tr("shap_fail", detail=repr(e))
    return result


def validate_inputs(raw_inputs: Dict[str, Any]) -> List[str]:
    msgs = []
    deployed = compute_derived_features(raw_inputs)
    for col, meta in FEATURE_META.items():
        lo, hi = meta["hard_range"]
        val = deployed.get(col)
        label = meta["cn"] if get_lang() == "zh" else meta["label"]
        if col == "po2_max" and raw_inputs.get("po2_max_missing"):
            # The deployed pipeline contains numeric imputation; allow PaO2 to be
            # explicitly unavailable rather than fabricating a value for FAH cases.
            continue
        if val is None:
            msgs.append(tr("err_missing", label=label))
            continue
        try:
            fval = float(val)
            if np.isnan(fval):
                msgs.append(tr("err_missing", label=label))
                continue
        except Exception:
            msgs.append(tr("err_missing", label=label))
            continue
        if not (lo <= fval <= hi):
            msgs.append(tr("err_range", label=label, lo=lo, hi=hi))

    if raw_inputs.get("delta_input_mode", "pair") != "direct":
        try:
            hb_min = float(raw_inputs.get("hemoglobin_min"))
            hb_earlier = float(raw_inputs.get("hemoglobin_earlier"))
            hb_later = float(raw_inputs.get("hemoglobin_later"))
            if hb_min > min(hb_earlier, hb_later):
                msgs.append(tr("err_hbmin"))
        except (TypeError, ValueError):
            pass
    return msgs


def get_threshold(model_meta: Dict[str, Any]) -> float:
    try:
        return float(model_meta.get("selected_threshold", DEFAULT_THRESHOLD))
    except Exception:
        return DEFAULT_THRESHOLD


def pretty_metric(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def get_field_label(feat: str) -> str:
    meta = FEATURE_META[feat]
    return meta["cn"] if get_lang() == "zh" else meta["label"]


def get_field_help(feat: str) -> str:
    meta = FEATURE_META[feat]
    return meta["help_zh"] if get_lang() == "zh" else meta["help_en"]


def render_sidebar_lang():
    """Stable bilingual selector.

    Keep the widget label and option values language-neutral so switching the UI
    language does not cause Streamlit to recreate the widget with an empty value.
    """
    current = get_lang()
    if st.session_state.get("lang_selector_sidebar") not in {"zh", "en"}:
        st.session_state["lang_selector_sidebar"] = current

    st.sidebar.markdown("### 🌐 语言 / Language")
    choice = st.sidebar.selectbox(
        "Language",
        options=["zh", "en"],
        format_func=lambda x: "中文" if x == "zh" else "English",
        label_visibility="collapsed",
        key="lang_selector_sidebar",
    )
    if choice not in {"zh", "en"}:
        choice = current
        st.session_state["lang_selector_sidebar"] = current
    st.session_state["lang"] = choice
    st.sidebar.markdown("---")



def render_header(model_meta: Dict[str, Any], report_text: str):
    # Keep deployment parameters internal. The clinical-facing header only
    # communicates what the system does, not calibration/threshold/AUC details.
    if get_lang() == "zh":
        tag1, tag2 = "🩸 48h SIC早期风险评估", "🧠 病例智能解读"
        hero_text = "SIC · 48h"
    else:
        tag1, tag2 = "🩸 48h SIC risk assessment", "🧠 Case interpretation"
        hero_text = "SIC · 48h"

    st.markdown(
        f"""
        <div class="main-card">
            <div class="header-wrap">
                <div class="header-left">
                    <div class="header-title">{tr("title")}</div>
                    <div class="header-sub">{tr("subtitle")}</div>
                    <div class="header-tags">
                        <span class="header-tag">{tag1}</span>
                        <span class="header-tag">{tag2}</span>
                    </div>
                </div>
                <div class="header-right">
                    <div class="header-hero">🩸</div>
                    <div class="header-hero-text">{hero_text}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_info(model_meta: Dict[str, Any]):
    render_sidebar_lang()
    usage_title = "📘 使用说明" if get_lang() == "zh" else "📘 Notes"
    indicator_title = "📋 指标说明" if get_lang() == "zh" else "📋 Input Guide"

    with st.sidebar.expander(usage_title, expanded=True):
        st.markdown(tr("usage_md"))

    # End users do not need model feature codes or deployment ranges.  Show
    # only clinically readable indicator names and units in a collapsed guide.
    rows = []
    for feat in FEATURE_ORDER:
        meta = FEATURE_META[feat]
        rows.append({
            ("指标" if get_lang() == "zh" else "Indicator"): meta["cn"] if get_lang() == "zh" else meta["label"],
            ("单位" if get_lang() == "zh" else "Unit"): meta["unit"],
        })
    with st.sidebar.expander(indicator_title, expanded=False):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_standard_input(feat: str, key_prefix: str = "input_", disabled: bool = False) -> Any:
    """Render one model input with an empty initial state.

    Values appear only after manual entry or after the user clicks ``Load case``.
    """
    meta = FEATURE_META[feat]
    widget_key = f"{key_prefix}{feat}"

    if feat == "liver_disease":
        # Keep the field genuinely blank before manual entry / example loading.
        kwargs = dict(
            options=[0, 1],
            format_func=lambda x: tr("yes") if x == 1 else tr("no"),
            help=get_field_help(feat),
            key=widget_key,
            disabled=disabled,
            placeholder=("请选择" if get_lang() == "zh" else "Select"),
        )
        if widget_key not in st.session_state:
            kwargs["index"] = None
        return st.selectbox(f"{get_field_label(feat)}", **kwargs)

    lo, hi = meta["hard_range"]
    kwargs = dict(
        min_value=float(lo),
        max_value=float(hi),
        step=float(meta["step"]),
        format=meta["format"],
        help=get_field_help(feat),
        key=widget_key,
        disabled=disabled,
    )
    if widget_key not in st.session_state:
        kwargs["value"] = None
    return st.number_input(f"{get_field_label(feat)} [{meta['unit']}]", **kwargs)


def render_delta_card(user_inputs: Dict[str, Any]):
    st.markdown(f"#### 📉 {tr('grp4')}")
    mode_label_map = {tr("delta_mode_pair"): "pair", tr("delta_mode_direct"): "direct"}

    previous_mode = st.session_state.get("delta_input_mode_selector")
    direct_labels = {I18N["zh"]["delta_mode_direct"], I18N["en"]["delta_mode_direct"]}
    pair_labels = {I18N["zh"]["delta_mode_pair"], I18N["en"]["delta_mode_pair"]}
    if previous_mode in direct_labels:
        st.session_state["delta_input_mode_selector"] = tr("delta_mode_direct")
    elif previous_mode in pair_labels:
        st.session_state["delta_input_mode_selector"] = tr("delta_mode_pair")
    elif previous_mode is not None:
        st.session_state.pop("delta_input_mode_selector", None)

    radio_kwargs = dict(
        options=list(mode_label_map.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="delta_input_mode_selector",
    )
    if "delta_input_mode_selector" not in st.session_state:
        radio_kwargs["index"] = 0
    selected_label = st.radio(tr("delta_toggle_label"), **radio_kwargs)
    if selected_label not in mode_label_map:
        selected_label = tr("delta_mode_pair")
    user_inputs["delta_input_mode"] = mode_label_map[selected_label]

    if user_inputs["delta_input_mode"] == "direct":
        c1, c2 = st.columns(2)
        with c1:
            dl_kwargs = dict(
                min_value=float(FEATURE_META["delta_lactate"]["hard_range"][0]),
                max_value=float(FEATURE_META["delta_lactate"]["hard_range"][1]),
                step=float(FEATURE_META["delta_lactate"]["step"]),
                format=FEATURE_META["delta_lactate"]["format"],
                help=get_field_help("delta_lactate"),
                key="delta_lactate_input",
            )
            if "delta_lactate_input" not in st.session_state:
                dl_kwargs["value"] = None
            user_inputs["delta_lactate_input"] = st.number_input(
                f"{get_field_label('delta_lactate')} [mmol/L]", **dl_kwargs
            )
        with c2:
            dh_kwargs = dict(
                min_value=float(FEATURE_META["delta_hemoglobin"]["hard_range"][0]),
                max_value=float(FEATURE_META["delta_hemoglobin"]["hard_range"][1]),
                step=float(FEATURE_META["delta_hemoglobin"]["step"]),
                format=FEATURE_META["delta_hemoglobin"]["format"],
                help=get_field_help("delta_hemoglobin"),
                key="delta_hemoglobin_input",
            )
            if "delta_hemoglobin_input" not in st.session_state:
                dh_kwargs["value"] = None
            user_inputs["delta_hemoglobin_input"] = st.number_input(
                f"{get_field_label('delta_hemoglobin')} [g/dL]", **dh_kwargs
            )
    else:
        def _empty_number(label, key, min_value, max_value):
            kwargs = dict(
                min_value=float(min_value), max_value=float(max_value),
                step=0.1, format="%.1f", key=key,
            )
            if key not in st.session_state:
                kwargs["value"] = None
            return st.number_input(label, **kwargs)

        c1, c2 = st.columns(2)
        with c1:
            user_inputs["lactate_earlier"] = _empty_number(
                f"{tr('lactate_earlier')} [mmol/L]", "lactate_earlier", 0.0, 40.0
            )
        with c2:
            user_inputs["lactate_later"] = _empty_number(
                f"{tr('lactate_later')} [mmol/L]", "lactate_later", 0.0, 40.0
            )
        c3, c4 = st.columns(2)
        with c3:
            user_inputs["hemoglobin_earlier"] = _empty_number(
                f"{tr('hb_earlier')} [g/dL]", "hemoglobin_earlier", 3.0, 25.0
            )
        with c4:
            user_inputs["hemoglobin_later"] = _empty_number(
                f"{tr('hb_later')} [g/dL]", "hemoglobin_later", 3.0, 25.0
            )

    derived = compute_derived_features(user_inputs)
    dl = derived.get("delta_lactate")
    dh = derived.get("delta_hemoglobin")
    if dl is not None and dh is not None:
        note_key = "delta_direct_note" if user_inputs["delta_input_mode"] == "direct" else "auto_delta"
        st.markdown(
            f"<div class='small-chip-note'>{tr(note_key, dl=dl, dh=dh)}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            "请完成变化值相关输入。" if get_lang() == "zh"
            else "Complete the measurements required for the change variables."
        )


def render_input_form() -> Dict[str, Any]:
    st.markdown(f'<div class="section-title">👤 {"患者临床信息" if get_lang()=="zh" else "Patient Inputs"}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small-note">{tr("input_note")}</div>', unsafe_allow_html=True)
    user_inputs: Dict[str, Any] = {}

    with st.container(border=True):
        st.markdown(f"#### 🧪 {tr('grp1')}")
        c1, c2 = st.columns(2)
        with c1:
            user_inputs["pt_max"] = render_standard_input("pt_max")
            user_inputs["rdw"] = render_standard_input("rdw")
        with c2:
            user_inputs["hemoglobin_min"] = render_standard_input("hemoglobin_min")
            user_inputs["mchc"] = render_standard_input("mchc")

    with st.container(border=True):
        st.markdown(f"#### 🧬 {tr('grp2')}")
        c1, c2 = st.columns(2)
        with c1:
            user_inputs["bilirubin_total_max"] = render_standard_input("bilirubin_total_max")
            user_inputs["liver_disease"] = render_standard_input("liver_disease")
        with c2:
            user_inputs["chloride_min"] = render_standard_input("chloride_min")

    with st.container(border=True):
        st.markdown(f"#### 🫁 {tr('grp3')}")
        c1, c2 = st.columns(2)
        with c1:
            user_inputs["sofa_first_day"] = render_standard_input("sofa_first_day")
            user_inputs["abs_neutrophils_max"] = render_standard_input("abs_neutrophils_max")
        with c2:
            po2_missing = st.checkbox(
                "PaO₂ max 未提供" if get_lang() == "zh" else "PaO₂ max unavailable",
                key="input_po2_max_missing",
                help=(
                    "若当前病例无该项数据，可保持勾选状态继续评估。"
                    if get_lang() == "zh" else
                    "Keep this selected when PaO₂ max is unavailable for the current case."
                ),
            )
            po2_display = render_standard_input("po2_max", disabled=po2_missing)
            user_inputs["po2_max_missing"] = bool(po2_missing)
            user_inputs["po2_max"] = np.nan if po2_missing else po2_display
        if po2_missing:
            st.markdown(
                '<div class="fah-note">' + (
                    '当前病例未提供 PaO₂ max；如有结果可取消勾选后手动补充。'
                    if get_lang() == 'zh' else
                    "PaO₂ max is unavailable; uncheck the option and enter a value if it becomes available."
                ) + '</div>', unsafe_allow_html=True,
            )

    with st.container(border=True):
        render_delta_card(user_inputs)

    return user_inputs

def render_prediction(pred_out: Dict[str, Any], threshold: float, model_meta: Dict[str, Any], prob_delta: Any = None):
    deploy_prob = float(pred_out["deploy_prob"])
    pred = int(pred_out["pred"])
    risk_text, risk_css = classify_risk(deploy_prob, threshold)
    pred_text = tr("pred_high") if pred == 1 else tr("pred_low")
    st.markdown(f'<div class="section-title">📌 {"风险评估结果" if get_lang()=="zh" else "Risk Assessment"}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="risk-summary-card"><div class="risk-label">{tr("pred_prob")}</div>'
            f'<div class="risk-prob">{deploy_prob*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="risk-summary-card"><div class="risk-label">{tr("risk_band")}</div>'
            f'<span class="{risk_css}">{risk_text}</span>'
            f'<div class="risk-copy">{pred_text}</div></div>',
            unsafe_allow_html=True,
        )

    st.progress(min(max(deploy_prob, 0.0), 1.0))
    st.caption(
        "该结果用于48小时SIC风险分层与辅助评估，不等同于临床诊断。"
        if get_lang() == "zh" else
        "This result supports 48-hour SIC risk stratification and does not constitute a clinical diagnosis."
    )


def render_shap_block(shap_result: Dict[str, Any]):
    # Preserve the earlier interpretation layout: waterfall and force plots
    # remain peer views under two tabs rather than hiding the force plot.
    st.markdown(f'<div class="section-title">{tr("shap")}</div>', unsafe_allow_html=True)
    if shap_result["error"]:
        st.warning(shap_result["error"])
        return

    tabs = st.tabs([tr("waterfall"), tr("force")])

    with tabs[0]:
        if shap_result["waterfall_fig"] is not None:
            left, center, right = st.columns([0.12, 0.76, 0.12])
            with center:
                st.pyplot(shap_result["waterfall_fig"], width="content")
        else:
            st.info(tr("waterfall_na"))

    with tabs[1]:
        if shap_result["force_fig"] is not None:
            st.pyplot(shap_result["force_fig"], width="stretch")
        else:
            st.info(tr("force_na"))



def _build_case_context(
    deployed_inputs: Dict[str, Any],
    pred_out: Dict[str, Any],
    threshold: float,
    shap_result: Dict[str, Any],
) -> Dict[str, Any]:
    deploy_prob = float(pred_out["deploy_prob"])
    pred = int(pred_out["pred"])
    risk_text, _ = classify_risk(deploy_prob, threshold)
    risk_band_code = "low" if deploy_prob < 0.30 else ("intermediate" if deploy_prob < threshold else "high")
    return {
        "case_id": st.session_state.get("loaded_fah_example"),
        "deploy_prob": deploy_prob,
        "raw_prob": pred_out.get("raw_prob"),
        "threshold": float(threshold),
        "pred_class": "48h SIC高风险" if pred == 1 else "48h SIC风险相对较低",
        "pred_class_code": pred,
        "risk_band": risk_text,
        "risk_band_code": risk_band_code,
        "feature_values": {k: deployed_inputs.get(k) for k in FEATURE_ORDER},
        "shap_top_features": shap_result.get("top_features") or [],
    }


def load_fah_examples() -> pd.DataFrame:
    if not os.path.exists(EXAMPLE_CASES_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(EXAMPLE_CASES_PATH)
    except Exception:
        return pd.DataFrame()


def apply_fah_example(row: pd.Series):
    """Load one de-identified FAH-HMUICU example into the input widgets.

    Examples are curated from the corrected FAHHMU.csv external-validation
    file, which contains all 12 deployed predictors including ``po2_max``.
    """
    for feat in [
        "pt_max", "hemoglobin_min", "rdw", "mchc", "sofa_first_day",
        "abs_neutrophils_max", "bilirubin_total_max", "chloride_min", "liver_disease", "po2_max",
    ]:
        value = row[feat]
        if pd.isna(value):
            continue
        if feat == "liver_disease":
            st.session_state[f"input_{feat}"] = int(value)
        else:
            st.session_state[f"input_{feat}"] = float(value)

    # The corrected FAH source includes PaO2. Curated examples store bilirubin
    # in the deployment unit (mg/dL) and include the observed po2_max value.
    st.session_state["input_po2_max_missing"] = False

    direct_label = I18N[get_lang()]["delta_mode_direct"]
    st.session_state["delta_input_mode_selector"] = direct_label
    st.session_state["delta_lactate_input"] = float(row["delta_lactate"])
    st.session_state["delta_hemoglobin_input"] = float(row["delta_hemoglobin"])
    st.session_state["loaded_fah_example"] = str(row["example_id"])
    # The observed 48h outcome is deliberately not stored in Session State or passed to the Agent.
    # It is reserved for offline QA only, preventing label leakage during interpretation.
    st.session_state.pop("loaded_fah_outcome", None)


def render_fah_example_loader():
    examples = load_fah_examples()
    if examples.empty:
        return

    current_lang = get_lang()
    title = "🏥 一附院示例病例" if current_lang == "zh" else "🏥 FAH-HMUICU Example Cases"
    note = (
        "以下为海南医科大学第一附属医院外部验证队列中的去标识化示例病例，可用于系统演示。"
        if current_lang == "zh"
        else "These are de-identified example cases from the FAH-HMUICU external-validation cohort for system demonstration."
    )
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.caption(note)

    labels = examples["example_id"].tolist()
    c1, c2 = st.columns([0.70, 0.30], vertical_alignment="bottom")
    with c1:
        selected = st.selectbox(
            "选择示例病例" if current_lang == "zh" else "Select an example case",
            labels,
            key="fah_example_selector",
        )
    with c2:
        load_clicked = st.button(
            "载入病例" if current_lang == "zh" else "Load case",
            key="load_fah_example",
            width="stretch",
        )
    if load_clicked:
        row = examples.loc[examples["example_id"].eq(selected)].iloc[0]
        apply_fah_example(row)
        # Loading a new example invalidates any previous prediction/report.
        for key in ["sic_last_prediction", "sic_last_shap_result", "sic_current_case", "sic_single_agent_report"]:
            st.session_state.pop(key, None)
        st.session_state["sic_single_agent_history"] = []
        st.rerun()



def render_agent_references(evidence_ids: List[str], language: str):
    """Render cited sources only, separating study-model basis from medical evidence."""
    if not evidence_ids:
        return
    registry = read_json(SOURCE_REGISTRY_PATH, {"sources": {}})
    sources = registry.get("sources", {}) if isinstance(registry, dict) else {}
    seen = set()
    model_rows = []
    medical_rows = []
    for sid in evidence_ids:
        sid = str(sid or "").strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        meta = sources.get(sid, {}) if isinstance(sources, dict) else {}
        citation = str(meta.get("citation") or "").strip()
        if not citation:
            continue
        if sid.upper().startswith("P"):
            model_rows.append((sid, citation))
        else:
            medical_rows.append((sid, citation))
    if not model_rows and not medical_rows:
        return
    label = "📚 参考依据" if language == "zh" else "📚 Evidence Sources"
    with st.expander(label, expanded=False):
        if model_rows:
            st.markdown("**模型依据**" if language == "zh" else "**Model basis**")
            for sid, citation in model_rows:
                st.markdown(f"**[{sid}]** {citation}")
        if medical_rows:
            st.markdown("**医学循证**" if language == "zh" else "**Medical evidence**")
            for sid, citation in medical_rows[:6]:
                st.markdown(f"**[{sid}]** {citation}")


def scroll_to_qa_history():
    """Smoothly scroll the Streamlit page to the Q&A history section."""
    components.html(
        """
        <script>
        setTimeout(function () {
            const doc = window.parent.document;
            const anchor = doc.getElementById('sic-qa-history-anchor');
            if (anchor) {
                anchor.scrollIntoView({behavior: 'smooth', block: 'start'});
            }
        }, 180);
        </script>
        """,
        height=0,
        width=0,
    )



def _guess_message_language(text: str) -> str:
    """Best-effort language tag for legacy conversation items."""
    value = str(text or "")
    cjk = sum(1 for ch in value if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in value if ch.isascii() and ch.isalpha())
    return "zh" if cjk >= max(2, latin // 6) else "en"


def _normalize_history_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade an old history item to the bilingual cached representation."""
    if not isinstance(item, dict):
        item = {"role": "assistant", "content": str(item)}
    role = str(item.get("role", "assistant"))
    content = str(item.get("content", ""))
    translations = item.get("translations")
    if not isinstance(translations, dict):
        translations = {}
    source_lang = item.get("source_lang")
    if source_lang not in {"zh", "en"}:
        source_lang = _guess_message_language(content)
    if content and not translations.get(source_lang):
        translations[source_lang] = content
    item["role"] = role
    item["content"] = content
    item["source_lang"] = source_lang
    item["translations"] = translations
    return item


def _history_display_content(item: Dict[str, Any], language: str) -> str:
    item = _normalize_history_item(item)
    translated = item.get("translations", {}).get(language)
    return str(translated if translated else item.get("content", ""))


def _ensure_history_language(history: List[Dict[str, Any]], language: str, llm_ok: bool) -> None:
    """Translate uncached conversation entries into the selected UI language.

    The original message remains in ``content``. Translations are cached inside
    each session-state message so subsequent language switches are instant.
    """
    missing_indexes: List[int] = []
    payload: List[Dict[str, str]] = []
    for idx, raw_item in enumerate(history):
        item = _normalize_history_item(raw_item)
        history[idx] = item
        if not item.get("translations", {}).get(language):
            missing_indexes.append(idx)
            payload.append({
                "role": item.get("role", "assistant"),
                "content": item.get("content", ""),
            })

    if not missing_indexes or not llm_ok:
        return

    status_text = "正在切换问答记录语言…" if language == "zh" else "Translating conversation history…"
    try:
        with st.spinner(status_text):
            translated = translate_chat_messages(payload, target_language=language)
        for idx, text in zip(missing_indexes, translated):
            history[idx].setdefault("translations", {})[language] = text
    except Exception:
        # Translation failure must not break the clinical interface; preserve and
        # display the original message as a safe fallback.
        return

def render_sic_assistant():
    """Render the single SIC case-interpretation Agent in a compact right panel."""
    current_lang = get_lang()
    ok, _ = get_llm_status(language=current_lang)
    case_context = st.session_state.get("sic_current_case")

    if not case_context:
        st.info(
            "完成左侧48小时SIC风险预测后，可在此生成病例智能解读。"
            if current_lang == "zh"
            else "Complete the 48-hour SIC prediction on the left to generate a case interpretation here."
        )
        return

    if not ok:
        st.warning(
            "智能解读服务当前不可用，请检查DSV4 API配置；风险预测功能不受影响。"
            if current_lang == "zh"
            else "The interpretation service is unavailable. Check the DSV4 API configuration; risk prediction remains available."
        )
        return

    report_key = "sic_single_agent_report"
    history_key = "sic_single_agent_history"
    st.session_state.setdefault(history_key, [])

    report_item = st.session_state.get(report_key)
    if report_item is not None and not isinstance(report_item, dict):
        report_item = {
            "content": str(report_item),
            "source_lang": _guess_message_language(str(report_item)),
            "translations": {},
        }
        report_item["translations"][report_item["source_lang"]] = report_item["content"]
        st.session_state[report_key] = report_item

    def _get_report_for_language(item: Dict[str, Any], language: str) -> str:
        if not item:
            return ""
        content = str(item.get("content", ""))
        source_lang = item.get("source_lang") if item.get("source_lang") in {"zh", "en"} else _guess_message_language(content)
        translations = item.get("translations") if isinstance(item.get("translations"), dict) else {}
        if content and not translations.get(source_lang):
            translations[source_lang] = content
        if not translations.get(language) and content:
            try:
                translated = translate_chat_messages(
                    [{"role": "assistant", "content": content}], target_language=language
                )[0]
                translations[language] = translated
            except Exception:
                pass
        item["source_lang"] = source_lang
        item["translations"] = translations
        return str(translations.get(language) or content)

    generate_clicked = st.button(
        "生成病例智能解读" if current_lang == "zh" else "Generate Case Interpretation",
        type="primary",
        width="stretch",
        key=f"generate_single_agent_report_{current_lang}",
    )

    if generate_clicked:
        with st.spinner("正在生成病例解读…" if current_lang == "zh" else "Generating case interpretation…"):
            try:
                result = generate_case_report(case_context, language=current_lang)
                report_item = {
                    "content": result.answer,
                    "source_lang": current_lang,
                    "translations": {current_lang: result.answer},
                    "elapsed_seconds": result.elapsed_seconds,
                    "knowledge_modules": result.knowledge_modules,
                    "evidence_ids": result.evidence_ids,
                }
                st.session_state[report_key] = report_item
                st.session_state[history_key] = []
            except Exception as exc:
                st.error(
                    f"病例解读生成失败：{exc}" if current_lang == "zh" else f"Case interpretation failed: {exc}"
                )
                report_item = None

    report_item = st.session_state.get(report_key)
    if not report_item:
        st.caption(
            "智能体将结合风险预测、个体风险因素和本地 SIC 知识库生成结构化病例解读。"
            if current_lang == "zh"
            else "The agent combines risk prediction, patient-level risk factors, and the local SIC knowledge base into a structured case interpretation."
        )
        return

    report_text = _get_report_for_language(report_item, current_lang)
    with st.container(border=True):
        st.markdown("#### 📋 病例综合解读" if current_lang == "zh" else "#### 📋 Case Interpretation")
        st.markdown(report_text)
        st.markdown(
            '<div class="agent-report-note">' + (
                '本报告用于科研展示与模型解释，不等同于临床诊断或治疗建议。'
                if current_lang == 'zh' else
                'This report supports research and model interpretation; it is not a clinical diagnosis or treatment recommendation.'
            ) + '</div>',
            unsafe_allow_html=True,
        )
    render_agent_references(report_item.get("evidence_ids", []), current_lang)

    st.markdown("#### 💬 病例追问" if current_lang == "zh" else "#### 💬 Case Follow-up")
    with st.form(key=f"single_agent_followup_form_{current_lang}", clear_on_submit=True):
        typed = st.text_area(
            "请输入问题" if current_lang == "zh" else "Enter a follow-up question",
            placeholder=(
                "例如：为什么这个患者的预测风险较高？"
                if current_lang == "zh" else
                "For example: Why is this patient's predicted risk high?"
            ),
            height=72,
            key=f"single_agent_followup_text_{current_lang}",
        )
        submitted = st.form_submit_button(
            "提交" if current_lang == "zh" else "Submit", type="primary", width="stretch"
        )

    quick_questions = (
        [
            "为什么这个患者的预测风险是现在这个水平？",
            "哪些因素对本次预测影响最大？",
            "这些主要变量在SIC背景下应该如何理解？",
            "基于当前12个变量，哪些指标最值得继续关注？",
        ]
        if current_lang == "zh" else
        [
            "Why is this patient's predicted risk at this level?",
            "Which factors contributed most to this prediction?",
            "How should the main variables be interpreted in the context of SIC?",
            "Which current variables deserve the closest follow-up?",
        ]
    )
    with st.expander("💡 示例问题" if current_lang == "zh" else "💡 Example Questions", expanded=False):
        for i, q in enumerate(quick_questions):
            if st.button(q, key=f"single_agent_quick_{current_lang}_{i}", width="stretch"):
                st.session_state[f"single_agent_pending_{current_lang}"] = q

    pending = st.session_state.pop(f"single_agent_pending_{current_lang}", None)
    question = (typed or "").strip() if submitted else pending
    history = st.session_state[history_key]
    _ensure_history_language(history, current_lang, ok)

    if question:
        history.append({
            "role": "user", "content": question, "source_lang": current_lang,
            "translations": {current_lang: question},
        })
        try:
            with st.spinner("正在分析…" if current_lang == "zh" else "Analyzing…"):
                result = answer_case_followup(
                    question=question,
                    case_context=case_context,
                    case_report=report_text,
                    language=current_lang,
                    history=[
                        {"role": m["role"], "content": _history_display_content(m, current_lang)}
                        for m in history[:-1]
                    ],
                )
            history.append({
                "role": "assistant", "content": result.answer, "source_lang": current_lang,
                "translations": {current_lang: result.answer},
            })
        except Exception as exc:
            err = f"智能体回答失败：{exc}" if current_lang == "zh" else f"Agent response failed: {exc}"
            history.append({
                "role": "assistant", "content": err, "source_lang": current_lang,
                "translations": {current_lang: err},
            })

    if history:
        st.markdown('<div id="sic-qa-history-anchor"></div>', unsafe_allow_html=True)
        with st.expander("🗂️ 问答记录" if current_lang == "zh" else "🗂️ Conversation", expanded=bool(question)):
            for msg in history:
                with st.chat_message(msg["role"]):
                    st.markdown(_history_display_content(msg, current_lang))
            if st.button(
                "清空问答记录" if current_lang == "zh" else "Clear Conversation",
                key=f"clear_single_agent_chat_{current_lang}",
            ):
                st.session_state[history_key] = []
                st.rerun()
        if question:
            scroll_to_qa_history()

def main():
    _reset_stale_example_session_if_needed()
    artifacts = load_artifacts()
    raw_model = artifacts["raw_model"]
    deploy_model = artifacts["deploy_model"]
    model_meta = artifacts["model_meta"]
    report_text = artifacts["report_text"]
    threshold = get_threshold(model_meta)

    render_sidebar_info(model_meta)
    render_header(model_meta, report_text)

    # Two-column clinical workspace: inputs/prediction on the left and the
    # case-interpretation Agent on the right.  The Agent is visible immediately
    # on first page load instead of appearing only after a long vertical scroll.
    left_col, right_col = st.columns([0.46, 0.54], gap="large")

    with left_col:
        render_fah_example_loader()
        user_inputs = render_input_form()
        input_errors = validate_inputs(user_inputs)

        predict_clicked = st.button(
            tr("run_pred"), type="primary", width='stretch',
            key="run_sic_prediction",
        )

        if predict_clicked and input_errors:
            if get_lang() == "zh":
                st.error("请先完整填写当前病例的必需输入，或点击上方“载入病例”自动填充。")
            else:
                st.error("Complete the required patient inputs, or use “Load case” to populate an example.")
            with st.expander("查看缺失/异常项" if get_lang() == "zh" else "View missing/invalid fields", expanded=False):
                for msg in input_errors:
                    st.markdown(f"- {msg}")

        if predict_clicked and not input_errors:
            deployed_inputs = compute_derived_features(user_inputs)
            X_raw = build_input_df(deployed_inputs, FEATURE_ORDER)

            if raw_model is None and deploy_model is None:
                st.error(tr("err_model_none"))
                st.stop()

            pred_out = safe_predict(raw_model, deploy_model, X_raw, threshold)
            if pred_out["error"] is not None:
                st.error(pred_out["error"])
                st.stop()

            previous_prob = st.session_state.get("last_deploy_prob")
            prob_delta = None if previous_prob is None else float(pred_out["deploy_prob"]) - float(previous_prob)
            st.session_state["last_deploy_prob"] = float(pred_out["deploy_prob"])

            shap_model = raw_model if raw_model is not None else deploy_model
            shap_result = explain_with_shap(shap_model, X_raw)

            preview_payload = {
                **{k: user_inputs.get(k) for k in user_inputs if k not in {"po2_max_missing"}},
                "delta_lactate": deployed_inputs["delta_lactate"],
                "delta_hemoglobin": deployed_inputs["delta_hemoglobin"],
            }
            if user_inputs.get("po2_max_missing"):
                preview_payload["po2_max"] = np.nan

            st.session_state["sic_last_prediction"] = {
                "deployed_inputs": deployed_inputs,
                "pred_out": pred_out,
                "prob_delta": prob_delta,
                "preview_payload": preview_payload,
            }
            st.session_state["sic_last_shap_result"] = shap_result
            st.session_state["sic_current_case"] = _build_case_context(
                deployed_inputs, pred_out, threshold, shap_result
            )

            # A new prediction starts a new Agent report/conversation.
            st.session_state.pop("sic_single_agent_report", None)
            st.session_state["sic_single_agent_history"] = []
            for _lang in ("zh", "en"):
                st.session_state.pop(f"sic_patient_interpretation_{_lang}", None)
                st.session_state.pop(f"sic_patient_interpretation_elapsed_{_lang}", None)
                st.session_state.pop(f"sic_multi_agent_patient_{_lang}", None)

        # Persist prediction and interpretation across Agent/report/chat reruns.
        persisted = st.session_state.get("sic_last_prediction")
        if persisted:
            st.markdown("---")
            render_prediction(
                persisted["pred_out"], threshold, model_meta,
                prob_delta=persisted.get("prob_delta"),
            )
            shap_result = st.session_state.get("sic_last_shap_result") or {
                "waterfall_fig": None, "force_fig": None, "error": tr("shap_na"), "top_features": []
            }
            render_shap_block(shap_result)


    with right_col:
        st.markdown(
            (
                '<div class="agent-hero-card">'
                '<div class="agent-hero-icon">🧠</div>'
                '<div><div class="agent-hero-title">SIC Case Interpretation Agent</div>'
                '<div class="agent-hero-sub">结合风险预测、个体风险因素与本地 SIC 循证知识，生成病例智能解读并支持追问。</div>'
                '</div></div>'
                if get_lang() == 'zh' else
                '<div class="agent-hero-card">'
                '<div class="agent-hero-icon">🧠</div>'
                '<div><div class="agent-hero-title">SIC Case Interpretation Agent</div>'
                '<div class="agent-hero-sub">Combines risk prediction, patient-level risk factors, and local SIC evidence for case interpretation and follow-up.</div>'
                '</div></div>'
            ),
            unsafe_allow_html=True,
        )
        render_sic_assistant()


if __name__ == "__main__":
    main()
