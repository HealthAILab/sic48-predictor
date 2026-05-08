# /usr/bin/python3 -m streamlit run model_routeB_frontend.py
import os
import json
import warnings
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
        "grp4": "变化值特征",
        "delta_mode_pair": "前后值",
        "delta_mode_direct": "直接输入Δ值",
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
        "methods": "📎 结果解读与使用说明",
        "methods_md": "**模型用途**\n- 目标：评估患者在 ICU 入科后 48 小时内发生 SIC 的风险\n- 输入窗口：ICU 入科后前 24 小时\n- 模型类型：{best_model}\n- 风险概率校准方式：{calibration_method}\n- 当前判定阈值：{threshold:.3f}\n\n**临床使用提示**\n- 请尽量录入原始化验或评分结果，并结合患者整体病情进行判断。\n- `delta_lactate` 与 `delta_hemoglobin` 可由系统根据前后两次检测自动计算，也可手动录入变化值。\n- 风险概率越高，提示患者在 48 小时内发生 SIC 的可能性越大。\n- 本结果适用于早期识别、分层管理与临床关注强化，不能替代最终临床诊断与治疗决策。",
        "err_range": "{label} 超出允许范围 [{lo}, {hi}]。",
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
        "grp4": "Delta features",
        "delta_mode_pair": "Earlier/later values",
        "delta_mode_direct": "Direct Δ input",
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
        "methods": "📎 Interpretation and use notes",
        "methods_md": "**Clinical purpose**\n- Target: estimate the risk of SIC within 48 hours after ICU admission\n- Input window: first 24 hours in ICU\n- Model type: {best_model}\n- Risk calibration method: {calibration_method}\n- Current decision threshold: {threshold:.3f}\n\n**Clinical use notes**\n- Whenever possible, enter original laboratory values and scores and interpret the result in the full clinical context.\n- `delta_lactate` and `delta_hemoglobin` can be computed automatically from serial measurements or entered directly as delta values.\n- A higher predicted probability suggests a greater likelihood of SIC within 48 hours.\n- This result is intended for early identification and risk stratification and does not replace final clinical diagnosis or treatment decisions.",
        "err_range": "{label} is outside the allowed range [{lo}, {hi}].",
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
.block-container {padding-top: 15px; padding-bottom: 1rem; max-width: 1380px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);}
.main-card {
    background: linear-gradient(135deg, #f8fafc 0%, #eef4ff 46%, #fff7ed 100%);
    border: 1px solid #dbeafe;
    border-radius: 24px;
    padding: 1.35rem 1.5rem 1.2rem 1.5rem;
    margin-top: 20px;
    margin-bottom: 1rem;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
}
.header-wrap {display:flex; justify-content:space-between; align-items:stretch; gap:1rem;}
.header-left {flex:1.45; min-width:0;}
.header-right {
    flex:0.55; min-width:200px; display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(239,68,68,0.10));
    border-radius: 20px; border: 1px dashed rgba(37,99,235,0.25);
}
.header-hero {font-size: 4rem; line-height: 1; filter: drop-shadow(0 4px 10px rgba(37,99,235,0.12));}
.chip-row {display:flex; flex-wrap:wrap; gap:0.55rem; margin-top:0.85rem;}
.chip {
    display:inline-flex; align-items:center; gap:0.38rem; padding:0.48rem 0.88rem;
    border-radius:999px; font-size:0.92rem; font-weight:700; border:1px solid transparent;
}
.chip-blue {background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe;}
.chip-red {background:#fef2f2; color:#b91c1c; border-color:#fecaca;}
.chip-amber {background:#fff7ed; color:#c2410c; border-color:#fed7aa;}
.chip-green {background:#ecfdf5; color:#047857; border-color:#a7f3d0;}
.chip-slate {background:#f8fafc; color:#334155; border-color:#cbd5e1;}
.section-title {font-size: 1.08rem; font-weight: 800; margin-bottom: 0.45rem; color:#0f172a;}
.small-note {color: #475569; font-size: 0.92rem;}
.caption-box {
    background: linear-gradient(90deg, #eff6ff 0%, #f8fafc 100%);
    border-left: 4px solid #2563eb; padding: 0.85rem 1rem; border-radius: 12px;
}
.input-card {
    background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
    border:1px solid #e2e8f0; border-radius:20px; padding:1rem 1rem 0.85rem 1rem; margin-bottom:0.95rem;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
}
.input-card-blue {border-top: 4px solid #2563eb;}
.input-card-red {border-top: 4px solid #ef4444;}
.input-card-amber {border-top: 4px solid #f59e0b;}
.input-card-teal {border-top: 4px solid #14b8a6;}
.card-head {display:flex; align-items:center; gap:0.5rem; margin-bottom:0.7rem;}
.card-icon {font-size:1.15rem;}
.card-title {font-weight:800; color:#0f172a;}
.metric-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #dbeafe; border-radius: 18px; padding: 1rem; text-align: center;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
}
.risk-low {background: #ecfdf5; color: #065f46; padding: 0.45rem 0.7rem; border-radius: 999px; font-weight: 700; display: inline-block;}
.risk-mid {background: #fffbeb; color: #92400e; padding: 0.45rem 0.7rem; border-radius: 999px; font-weight: 700; display: inline-block;}
.risk-high {background: #fef2f2; color: #991b1b; padding: 0.45rem 0.7rem; border-radius: 999px; font-weight: 700; display: inline-block;}
.small-chip-note {margin-top:0.55rem; font-size:0.90rem; color:#0f766e; font-weight:700;}
.delta-toggle-wrap {
    margin-top: -4.4rem;
    margin-right: 0.55rem;
    margin-bottom: 0.35rem;
    display: flex;
    justify-content: flex-end;
}
.delta-toggle-wrap div[data-testid="stRadio"] {
    background: rgba(255,255,255,0.92);
    border: 1px solid #cbd5e1;
    border-radius: 14px;
    padding: 0.3rem 0.5rem 0.15rem 0.5rem;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
}
.delta-toggle-wrap div[role="radiogroup"] {
    gap: 0.55rem;
}
.delta-toggle-wrap label {
    margin-bottom: 0 !important;
}
.delta-toggle-spacer {height: 0.35rem;}
div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] > div {border-radius: 12px !important;}
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
    derived = dict(raw_inputs)
    mode = raw_inputs.get("delta_input_mode", "pair")
    if mode == "direct":
        derived["delta_lactate"] = float(raw_inputs["delta_lactate_input"])
        derived["delta_hemoglobin"] = float(raw_inputs["delta_hemoglobin_input"])
    else:
        derived["delta_lactate"] = float(raw_inputs["lactate_later"]) - float(raw_inputs["lactate_earlier"])
        derived["delta_hemoglobin"] = float(raw_inputs["hemoglobin_later"]) - float(raw_inputs["hemoglobin_earlier"])
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
    result = {"waterfall_fig": None, "force_fig": None, "error": None, "used_model": None}
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
        val = deployed[col]
        if not (lo <= float(val) <= hi):
            label = meta["cn"] if get_lang() == "zh" else meta["label"]
            msgs.append(tr("err_range", label=label, lo=lo, hi=hi))
    if raw_inputs.get("delta_input_mode", "pair") != "direct":
        if raw_inputs["hemoglobin_min"] > min(raw_inputs["hemoglobin_earlier"], raw_inputs["hemoglobin_later"]):
            msgs.append(tr("err_hbmin"))
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
    st.sidebar.markdown(f"### {tr('lang_label')}")
    choice = st.sidebar.selectbox(
        tr("lang_label"),
        options=["zh", "en"],
        index=0 if get_lang() == "zh" else 1,
        format_func=lambda x: tr("lang_zh") if x == "zh" else tr("lang_en"),
        label_visibility="collapsed",
        key="lang_selector_sidebar",
    )
    st.session_state["lang"] = choice
    st.sidebar.markdown("---")



def render_header(model_meta: Dict[str, Any], report_text: str):
    threshold = get_threshold(model_meta)
    auc = model_meta.get("final_test_metrics", {}).get("auc", None)
    ap = model_meta.get("final_test_metrics", {}).get("ap", None)
    model_name = model_meta.get("best_candidate_model", "XGBoost")
    calib_label = format_calibration_label(model_meta)
    auc_txt = pretty_metric(float(auc), 3) if auc is not None else "N/A"
    ap_txt = pretty_metric(float(ap), 3) if ap is not None else "N/A"

    st.markdown(
        f"""
        <div class="main-card">
            <div class="header-wrap">
                <div class="header-left">
                    <div style="font-size:2.15rem;font-weight:900;color:#0f172a;">{tr("title")}</div>
                    <div style="font-size:1.02rem;color:#64748b;margin-top:0.35rem;line-height:1.7;">
                        {tr("subtitle")}
                    </div>
                    <div class="chip-row">
                        <span class="chip chip-blue">🩸 {tr("task_chip")}</span>
                        <span class="chip chip-amber">📈 {tr("calibr_chip")}: {calib_label}</span>
                    </div>
                    <div class="chip-row">
                        <span class="chip chip-slate">⭐ {tr("best_model")}: {model_name}</span>
                        <span class="chip chip-green">🎯 {tr("deploy_threshold")}: {pretty_metric(threshold, 3)}</span>
                        <span class="chip chip-red">📊 {tr("test_auc_ap")}: {auc_txt} / {ap_txt}</span>
                    </div>
                </div>
                <div class="header-right">
                    <div class="header-hero">🩸📉</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(tr("summary"), expanded=False):
        if report_text.strip():
            st.code(report_text, language="text")
        else:
            st.info(tr("summary_missing"))


def render_sidebar_info(model_meta: Dict[str, Any]):
    render_sidebar_lang()
    st.sidebar.markdown(f"### {tr('usage')}")
    st.sidebar.markdown(tr("usage_md"))
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {tr('range_ref')}")
    rows = []
    for feat in FEATURE_ORDER:
        meta = FEATURE_META[feat]
        rows.append({
            tr("feature_col"): feat,
            tr("meaning_col"): meta["cn"] if get_lang() == "zh" else meta["label"],
            tr("unit_col"): meta["unit"],
            tr("allow_col"): f"{meta['hard_range'][0]}–{meta['hard_range'][1]}",
            tr("typical_col"): f"{meta['typical_range'][0]}–{meta['typical_range'][1]}",
        })
    st.sidebar.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def render_standard_input(feat: str, key_prefix: str = "input_") -> Any:
    meta = FEATURE_META[feat]
    if feat == "liver_disease":
        selected = st.selectbox(
            f"{get_field_label(feat)}",
            options=[0, 1],
            index=int(meta["default"]),
            format_func=lambda x: tr("yes") if x == 1 else tr("no"),
            help=get_field_help(feat),
            key=f"{key_prefix}{feat}",
        )
        return selected
    lo, hi = meta["hard_range"]
    return st.number_input(
        f"{get_field_label(feat)} [{meta['unit']}]",
        min_value=float(lo),
        max_value=float(hi),
        value=float(meta["default"]),
        step=float(meta["step"]),
        format=meta["format"],
        help=get_field_help(feat),
        key=f"{key_prefix}{feat}",
    )


def render_delta_card(user_inputs: Dict[str, Any]):
    st.markdown(
        f"<div class='input-card input-card-teal'><div class='card-head'><span class='card-icon'>📉</span><span class='card-title'>{tr('grp4')}</span></div></div>",
        unsafe_allow_html=True,
    )
    mode_label_map = {
        tr("delta_mode_pair"): "pair",
        tr("delta_mode_direct"): "direct",
    }
    st.markdown("<div class='delta-toggle-wrap'>", unsafe_allow_html=True)
    selected_label = st.radio(
        tr("delta_toggle_label"),
        options=list(mode_label_map.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="delta_input_mode_selector",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    user_inputs["delta_input_mode"] = mode_label_map[selected_label]
    st.markdown("<div class='delta-toggle-spacer'></div>", unsafe_allow_html=True)

    if user_inputs["delta_input_mode"] == "direct":
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            user_inputs["delta_lactate_input"] = st.number_input(
                f"{get_field_label('delta_lactate')} [mmol/L]",
                min_value=float(FEATURE_META["delta_lactate"]["hard_range"][0]),
                max_value=float(FEATURE_META["delta_lactate"]["hard_range"][1]),
                value=float(FEATURE_META["delta_lactate"]["default"]),
                step=float(FEATURE_META["delta_lactate"]["step"]),
                format=FEATURE_META["delta_lactate"]["format"],
                help=get_field_help("delta_lactate"),
                key="delta_lactate_input",
            )
        with r1c2:
            user_inputs["delta_hemoglobin_input"] = st.number_input(
                f"{get_field_label('delta_hemoglobin')} [g/dL]",
                min_value=float(FEATURE_META["delta_hemoglobin"]["hard_range"][0]),
                max_value=float(FEATURE_META["delta_hemoglobin"]["hard_range"][1]),
                value=float(FEATURE_META["delta_hemoglobin"]["default"]),
                step=float(FEATURE_META["delta_hemoglobin"]["step"]),
                format=FEATURE_META["delta_hemoglobin"]["format"],
                help=get_field_help("delta_hemoglobin"),
                key="delta_hemoglobin_input",
            )
    else:
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            user_inputs["lactate_earlier"] = st.number_input(
                f"{tr('lactate_earlier')} [mmol/L]",
                min_value=0.0, max_value=40.0, value=2.0, step=0.1, format="%.1f", key="lactate_earlier"
            )
        with r1c2:
            user_inputs["lactate_later"] = st.number_input(
                f"{tr('lactate_later')} [mmol/L]",
                min_value=0.0, max_value=40.0, value=3.0, step=0.1, format="%.1f", key="lactate_later"
            )
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            user_inputs["hemoglobin_earlier"] = st.number_input(
                f"{tr('hb_earlier')} [g/dL]",
                min_value=3.0, max_value=25.0, value=11.0, step=0.1, format="%.1f", key="hemoglobin_earlier"
            )
        with r2c2:
            user_inputs["hemoglobin_later"] = st.number_input(
                f"{tr('hb_later')} [g/dL]",
                min_value=3.0, max_value=25.0, value=10.0, step=0.1, format="%.1f", key="hemoglobin_later"
            )

    derived = compute_derived_features(user_inputs)
    note_key = "delta_direct_note" if user_inputs["delta_input_mode"] == "direct" else "auto_delta"
    st.markdown(
        f"<div class='small-chip-note'>🧮 {tr(note_key, dl=derived['delta_lactate'], dh=derived['delta_hemoglobin'])}</div>",
        unsafe_allow_html=True
    )

def render_input_form() -> Dict[str, Any]:
    st.markdown(f'<div class="section-title">{tr("input_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small-note">{tr("input_note")}</div>', unsafe_allow_html=True)

    user_inputs: Dict[str, Any] = {}
    card_cols = st.columns(2)

    with card_cols[0]:
        st.markdown(
            f"<div class='input-card input-card-blue'><div class='card-head'><span class='card-icon'>🧪</span><span class='card-title'>{tr('grp1')}</span></div></div>",
            unsafe_allow_html=True,
        )
        for feat in ["pt_max", "hemoglobin_min", "rdw", "mchc"]:
            user_inputs[feat] = render_standard_input(feat)

        st.markdown(
            f"<div class='input-card input-card-amber'><div class='card-head'><span class='card-icon'>🫁</span><span class='card-title'>{tr('grp3')}</span></div></div>",
            unsafe_allow_html=True,
        )
        for feat in ["sofa_first_day", "po2_max", "abs_neutrophils_max"]:
            user_inputs[feat] = render_standard_input(feat)

    with card_cols[1]:
        st.markdown(
            f"<div class='input-card input-card-red'><div class='card-head'><span class='card-icon'>🧬</span><span class='card-title'>{tr('grp2')}</span></div></div>",
            unsafe_allow_html=True,
        )
        for feat in ["bilirubin_total_max", "chloride_min", "liver_disease"]:
            user_inputs[feat] = render_standard_input(feat)

        render_delta_card(user_inputs)

    return user_inputs



def render_prediction(pred_out: Dict[str, Any], threshold: float, model_meta: Dict[str, Any], prob_delta: Any = None):
    deploy_prob = float(pred_out["deploy_prob"])
    raw_prob = pred_out["raw_prob"]
    pred = int(pred_out["pred"])
    risk_text, risk_css = classify_risk(deploy_prob, threshold)
    pred_text = tr("pred_high") if pred == 1 else tr("pred_low")
    st.markdown(f'<div class="section-title">{tr("prediction")}</div>', unsafe_allow_html=True)

    if prob_delta is None:
        delta_note = tr("first_pred_note")
    else:
        delta_note = tr("prob_change_note", delta=prob_delta)

    raw_note = f"{tr('raw_prob_note')}: {raw_prob:.6f}" if raw_prob is not None else f"{tr('raw_prob_note')}: N/A"
    deploy_note = f"{tr('deploy_prob_note')}: {deploy_prob:.6f}"
    source_note = f"{tr('prob_source_label')}: {get_probability_source(model_meta)}"
    threshold_note = tr("current_threshold_note", th=threshold)
    recalc_note = tr("recomputed_note")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="pred-result-card">
                <div class="pred-card-head">
                    <div class="pred-card-head-left">
                        <div class="pred-card-label"><span class="pred-title-icon">📈</span>{tr('deploy_prob_note')}</div>
                    </div>
                </div>
                <div class="pred-card-main">
                    <div class="pred-card-value">{deploy_prob*100:.3f}%</div>
                </div>
                <div class="pred-card-note">{deploy_note}<br>{raw_note}<br>{delta_note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="pred-result-card">
                <div class="pred-card-head">
                    <div class="pred-card-head-left">
                        <div class="pred-card-label"><span class="pred-title-icon">🎯</span>{tr('pred_class')}</div>
                    </div>
                </div>
                <div class="pred-card-main">
                    <div class="pred-card-value-text">{pred_text}</div>
                </div>
                <div class="pred-card-note">{threshold_note}<br>{source_note}<br>{recalc_note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        risk_detail = f"{tr('risk_note')}；{recalc_note}" if get_lang() == "zh" else f"{tr('risk_note')}; {recalc_note}"
        st.markdown(
            f"""
            <div class="pred-result-card">
                <div class="pred-card-head">
                    <div class="pred-card-head-left">
                        <div class="pred-card-label"><span class="pred-title-icon">🚨</span>{tr('risk_band')}</div>
                    </div>
                </div>
                <div class="pred-badge-wrap"><span class="{risk_css}">{risk_text}</span></div>
                <div class="pred-card-note">{risk_detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(min(max(deploy_prob, 0.0), 1.0))
    st.markdown(f'<div class="caption-box">{tr("threshold_note", th=threshold)}</div>', unsafe_allow_html=True)


def render_shap_block(shap_result: Dict[str, Any]):
    st.markdown(f'<div class="section-title">{tr("shap")}</div>', unsafe_allow_html=True)
    if shap_result["error"]:
        st.warning(shap_result["error"])
        return

    st.caption(tr("explainer_model", model=shap_result.get("used_model", "Unknown")))
    tabs = st.tabs([tr("waterfall"), tr("force")])

    with tabs[0]:
        if shap_result["waterfall_fig"] is not None:
            left, center, right = st.columns([0.12, 0.76, 0.12])
            with center:
                st.pyplot(shap_result["waterfall_fig"], width='content')
        else:
            st.info(tr("waterfall_na"))

    with tabs[1]:
        if shap_result["force_fig"] is not None:
            st.pyplot(shap_result["force_fig"], width='stretch')
        else:
            st.info(tr("force_na"))



def main():
    artifacts = load_artifacts()
    raw_model = artifacts["raw_model"]
    deploy_model = artifacts["deploy_model"]
    model_meta = artifacts["model_meta"]
    report_text = artifacts["report_text"]
    threshold = get_threshold(model_meta)

    render_sidebar_info(model_meta)
    render_header(model_meta, report_text)

    user_inputs = render_input_form()
    input_errors = validate_inputs(user_inputs)
    if input_errors:
        for msg in input_errors:
            st.error(msg)

    predict_clicked = st.button(tr("run_pred"), type="primary", width='stretch')
    st.markdown("---")

    if predict_clicked:
        deployed_inputs = compute_derived_features(user_inputs)
        X_raw = build_input_df(deployed_inputs, FEATURE_ORDER)

        st.markdown(f'<div class="section-title">{tr("input_preview")}</div>', unsafe_allow_html=True)
        preview_df = pd.DataFrame([{
            **user_inputs,
            "delta_lactate": deployed_inputs["delta_lactate"],
            "delta_hemoglobin": deployed_inputs["delta_hemoglobin"],
        }])
        st.dataframe(preview_df, width='stretch', hide_index=True)

        if raw_model is None and deploy_model is None:
            st.error(tr("err_model_none"))
            st.stop()

        pred_out = safe_predict(raw_model, deploy_model, X_raw, threshold)
        if pred_out["error"] is not None:
            st.error(pred_out["error"])
            st.stop()

        previous_prob = st.session_state.get("last_deploy_prob")
        prob_delta = None if previous_prob is None else float(pred_out["deploy_prob"]) - float(previous_prob)

        render_prediction(pred_out, threshold, model_meta, prob_delta=prob_delta)
        st.caption(
            tr("inference_path", mode=pred_out["used_mode"])
            + f" | raw={os.path.basename(artifacts['raw_model_path'])} | deploy={os.path.basename(artifacts['deploy_model_path'])}"
        )

        st.session_state["last_deploy_prob"] = float(pred_out["deploy_prob"])

        shap_model = raw_model if raw_model is not None else deploy_model
        shap_result = explain_with_shap(shap_model, X_raw)
        render_shap_block(shap_result)

    with st.expander(tr("methods"), expanded=False):
        st.markdown(
            tr(
                "methods_md",
                best_model=model_meta.get("best_candidate_model", "XGBoost"),
                calibration_method=format_calibration_label(model_meta),
                threshold=threshold,
                prob_source=get_probability_source(model_meta),
            )
        )


if __name__ == "__main__":
    main()
