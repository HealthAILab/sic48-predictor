from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from sic_agent import KNOWLEDGE_DIR, KNOWLEDGE_MODULES, get_relevant_knowledge_modules
from .schemas import EvidenceBundle, EvidenceSource

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_REGISTRY_PATH = KNOWLEDGE_DIR / "source_registry.json"

# Additional deterministic hints for model-feature names that may not appear
# verbatim in the natural-language keyword registry.
FEATURE_MODULE_HINTS: Dict[str, Sequence[str]] = {
    "sofa_first_day": ("16_predictor_context.md", "06_clinical_monitoring.md"),
    "po2_max": ("16_predictor_context.md", "06_clinical_monitoring.md", "02_pathophysiology.md"),
    "chloride_min": ("16_predictor_context.md", "06_clinical_monitoring.md"),
    "hemoglobin_min": ("16_predictor_context.md", "05_laboratory_markers.md", "10_differential_diagnosis.md"),
    "abs_neutrophils_max": ("16_predictor_context.md", "02_pathophysiology.md"),
    "bilirubin_total_max": ("16_predictor_context.md", "10_differential_diagnosis.md"),
    "pt_max": ("16_predictor_context.md", "05_laboratory_markers.md", "03_diagnostic_criteria.md"),
    "delta_lactate": ("16_predictor_context.md", "06_clinical_monitoring.md"),
    "delta_hemoglobin": ("16_predictor_context.md", "05_laboratory_markers.md"),
    "rdw": ("16_predictor_context.md", "05_laboratory_markers.md"),
    "mchc": ("16_predictor_context.md", "05_laboratory_markers.md"),
    "liver_disease": ("16_predictor_context.md", "10_differential_diagnosis.md"),
}


def _read_text(filename: str) -> str:
    path = KNOWLEDGE_DIR / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""




def _compact_module_text(text: str, max_chars: int = 2400) -> str:
    """Keep retrieved evidence compact enough for a reasoning-model prompt.

    The P0 knowledge base is curated at module level rather than chunk level.
    Passing four full modules to DSV4 made the synthesis prompt unnecessarily
    long and increased hidden reasoning consumption. This deterministic trim
    preserves the beginning of each curated module (where definitions and core
    evidence statements are stored) while source citations remain separately
    authorized through ``source_registry.json``.
    """
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    cut = value[:max_chars]
    # Prefer ending on a paragraph/sentence boundary when possible.
    for sep in ("\n\n", "。", ". "):
        idx = cut.rfind(sep)
        if idx >= int(max_chars * 0.65):
            cut = cut[: idx + (1 if sep in {"。"} else 0)]
            break
    return cut.rstrip() + "\n[本模块后续内容已为本次病例检索压缩]"

def _module_lookup() -> Dict[str, dict]:
    return {str(m.get("file")): m for m in KNOWLEDGE_MODULES}


def _module_title(module: dict, language: str) -> str:
    if language == "en":
        return str(module.get("title_en") or module.get("title") or module.get("file"))
    return str(module.get("title") or module.get("file"))


def _load_registry() -> dict:
    try:
        return json.loads(SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"sources": {}, "module_sources": {}}


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _feature_hint_files(case_context: dict, max_files: int = 3) -> List[str]:
    ranked = case_context.get("shap_top_features") or []
    files: List[str] = []
    for item in ranked[:5]:
        feat = str(item.get("feature") or "")
        for filename in FEATURE_MODULE_HINTS.get(feat, ()):  # deterministic fallback
            if filename not in files:
                files.append(filename)
        if len(files) >= max_files:
            break
    return files[:max_files]


def retrieve_evidence(
    queries: Sequence[str],
    case_context: dict,
    language: str = "zh",
    max_modules: int = 4,
) -> EvidenceBundle:
    """Retrieve a compact local evidence bundle for a planned case interpretation.

    P0 uses transparent local module retrieval rather than a hosted vector service.
    The planner produces case-specific queries, while deterministic feature hints
    prevent a failed/overly-generic LLM query from losing the most relevant modules.
    """
    lookup = _module_lookup()
    selected_files: List[str] = ["09_model_interpretation.md", "16_predictor_context.md"]

    for query in queries:
        for module in get_relevant_knowledge_modules(str(query), max_modules=2):
            filename = str(module.get("file") or "")
            if filename and filename not in selected_files:
                selected_files.append(filename)
            if len(selected_files) >= max_modules:
                break
        if len(selected_files) >= max_modules:
            break

    if len(selected_files) < max_modules:
        for filename in _feature_hint_files(case_context, max_files=max_modules):
            if filename not in selected_files:
                selected_files.append(filename)
            if len(selected_files) >= max_modules:
                break

    selected_files = selected_files[:max_modules]
    chunks: List[str] = []
    titles: List[str] = []
    for filename in selected_files:
        text = _read_text(filename)
        if not text:
            continue
        text = _compact_module_text(text)
        module = lookup.get(filename, {"file": filename, "title": filename, "title_en": filename})
        titles.append(_module_title(module, language))
        chunks.append(f"## {_module_title(module, language)}\n{text}")

    registry = _load_registry()
    source_ids: List[str] = []
    for filename in selected_files:
        for sid in registry.get("module_sources", {}).get(filename, []):
            if sid not in source_ids:
                source_ids.append(sid)

    sources: List[EvidenceSource] = []
    for sid in source_ids[:5]:
        meta = registry.get("sources", {}).get(sid, {})
        if meta:
            sources.append(EvidenceSource(
                source_id=sid,
                short=str(meta.get("short") or sid),
                citation=str(meta.get("citation") or sid),
            ))

    return EvidenceBundle(
        content="\n\n---\n\n".join(chunks),
        knowledge_modules=titles,
        module_files=selected_files,
        sources=sources,
    )


def retrieve_for_question(question: str, case_context: dict, language: str = "zh", max_modules: int = 2) -> EvidenceBundle:
    queries = [question]
    return retrieve_evidence(queries, case_context=case_context, language=language, max_modules=max_modules)
