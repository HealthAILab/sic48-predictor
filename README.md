# SIC48 Case Interpretation Agent

Streamlit application for 48-hour SIC risk prediction, patient-level SHAP explanations, evidence-based case reports, and follow-up questions.

## Run

Use Python 3.11 and install `requirements.txt`, then run:

```sh
streamlit run model_routeB_frontend.py
```

`app.py` is also supported. The legacy entry point is retained for the existing Streamlit Cloud deployment.

## Streamlit Community Cloud

Repository: `HealthAILab/sic48-predictor`; branch: `main`; entry point: `model_routeB_frontend.py`; Python: 3.11.

Configure these values in the app Settings > Secrets (never commit credentials):

```toml
SIC_LLM_API_URL = "your OpenAI-compatible chat completions endpoint"
SIC_LLM_API_KEY = "your API key"
SIC_LLM_MODEL = "your model name"
```

Local runs can use the same environment variables in an untracked `.env` file. Prediction works without an LLM key; Agent reports require a configured, reachable endpoint.

See `README_SINGLE_AGENT.md` for the Agent workflow. The included example cases are demonstration inputs. Outputs support research and do not replace clinical assessment.
