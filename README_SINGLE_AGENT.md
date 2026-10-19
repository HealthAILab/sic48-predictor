# SIC Case Interpretation Agent — Run Guide

## 1. Configure the DSV4 API

Copy `.env.example` to `.env` and fill in your own endpoint/key/model. Do not commit `.env`.

## 2. Install dependencies

```bash
/usr/bin/python3 -m pip install -r requirements.txt
```

## 3. Start Streamlit

```bash
/usr/bin/python3 -m streamlit run app.py
```

## 4. User flow

1. Enter or load the 12 model variables.
2. Run the 48-hour SIC prediction.
3. Review the XGBoost output and patient-level SHAP plot.
4. Click **生成病例智能解读 / Generate Case Interpretation**.
5. The single Agent internally plans case-specific evidence needs, retrieves local SIC knowledge, and produces a concise five-section structured report.
6. Ask optional follow-up questions below the report.

A new prediction automatically clears the previous case report and follow-up history.

## 2026-09-22 FAH-HMUICU + two-column update

- Example cases now come from the supplied First Affiliated Hospital of Hainan Medical University dataset (`examples/fah_hmuicu_source.csv`) rather than MIMIC-IV.
- Ten de-identified demonstration examples are stored in `examples/fah_hmuicu_example_cases.csv` (5 observed SIC, 5 observed non-SIC).
- The corrected `FAHHMU.csv` contains 204 rows and 13 columns: all 12 deployed predictors plus `sic_diagnosis_48h`. `po2_max` is present for every row.
- The curated web examples now load the observed PaO₂ value rather than marking it as missing.
- The bilirubin values in `FAHHMU.csv` are on a µmol/L-scale and are converted to the model/UI unit mg/dL (divide by 17.1) in the curated example file.
- The raw source contains a small number of obvious range/unit outliers in fields such as hemoglobin, chloride, and RDW. These rows are left unchanged in `data/FAHHMU.csv` but are excluded from the 10 demonstration cases.
- The main interface is now two-column: patient input/prediction/SHAP on the left, and the SIC Case Interpretation Agent on the right from the first page load.
- Prediction and SHAP outputs are persisted in Streamlit session state so they remain visible when the Agent report is generated or follow-up questions are submitted.


### Evidence display
The main structured case report keeps the full deduplicated reference list in one collapsed “参考依据 / Evidence Sources” section. Follow-up answers cite compact source IDs and do not repeat the full bibliography unless a newly retrieved source was not present in the main report.

### Sidebar and Agent UI
The Streamlit sidebar remains collapsible and uses compact expandable sections. The Agent panel uses clinical icons for report generation, evidence, monitoring, follow-up, and conversation history.


## v3 report format

The Agent report is intentionally compact and modeled after a clinical case-summary style:

1. Final Risk Assessment / 最终风险评估
2. Key Variables / 关键变量
3. Reasoning with Evidence / 循证推理
4. Factors Needing Monitoring / 重点监测因素
5. Interpretation Boundaries / 解读边界

The Agent should not repeat all 12 predictors in every section. Key Variables is limited to the leading 3–5 patient-level SHAP contributors, while the reasoning section integrates them with retrieved SIC evidence.

## v8 QA hardening
The v8 bundle automatically invalidates stale FAH example-case session state when the example dataset changes, keeps the observed 48h SIC label out of Agent context, separates model-basis citations from medical evidence, and adds deterministic interpretation QA. For repeated 10-case testing, run `qa/run_fah_agent_qa.py --runs 3 --language en` from the project root.

### v9 input behavior
On a fresh session, the clinical form is intentionally blank. Values are populated only by manual entry or by clicking **Load case / 载入病例**. The application does not use hidden display defaults for missing clinical inputs.
