# -*- coding: utf-8 -*-
"""
protocol_common.py — Clean-repeatability experiment: FROZEN shared constants.

This file is part of the offline preparation for the clean-request repeatability
experiment (clean_repeatability_preparation/). It is a frozen protocol artifact:
do not edit values here without a documented protocol amendment.

Provenance of every constant:
  - Model/provider:   mci_rulers/.env.local (DEEPSEEK_MODEL=deepseek-v4-flash),
                      config.yaml (deepseek defaults), stability_experiment.py:540-549
                      (temperature from CLI --temperature, max_tokens=16384).
                      Historical run artifacts (run_config.json) do NOT record the
                      model identifier; see protocol.md S4 for the uncertainty note.
  - Rubric:           mci_rubric_audio_en_v1.json, locked by RubricLoader.lock()
                      (bundle_hash asserted below). EN experiment used the EN rubric
                      (language=en selects PROMPT_SYSTEM_MCI_EN).
  - Thresholds:       experiments/hybrid_rulers_full/recording_time_reanalysis/
                      freeze/model_config.json (frozen). Direct 0.50 = prespecified
                      neutral midpoint; Prompt RULERS 0.07 = frozen train-set Youden
                      on normalized score; Hybrid LR 0.55 = frozen train-set Youden.
  - Feature columns:  recording_time_reanalysis/freeze/model_config.json features
                      (6 programmatic + 6 LLM criteria, 12 total), archived in
                      hybrid_rulers_full/hybrid_features_{train,test}.csv.
  - Prompt templates: PROMPT_DIRECT_SYSTEM / PROMPT_DIRECT_USER from
                      experiments/direct_llm_baseline.py (verbatim); RULERS prompt
                      built by mci_rulers_core.build_scoring_prompt (frozen).

NO network calls happen on import. No API keys are read or stored anywhere here.
"""

import hashlib
import json
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREP_DIR = os.path.dirname(_SCRIPT_DIR)                      # clean_repeatability_preparation/
EXPERIMENTS_DIR = os.path.dirname(PREP_DIR)                  # hybrid_rulers_full/
PROJECT_DIR = os.path.dirname(os.path.dirname(EXPERIMENTS_DIR))  # mci_rulers/

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Read-only inputs (never modified by any script in this package)
HISTORICAL_RESULTS_JSONL = os.path.join(
    PROJECT_DIR, "experiments", "stability_audio_en_v4flash_full_test", "results.jsonl"
)
FROZEN_INCLUSION_CSV = os.path.join(
    EXPERIMENTS_DIR, "recording_time_reanalysis", "freeze", "inclusion_exclusion.csv"
)
FROZEN_FEATURES_TRAIN_CSV = os.path.join(EXPERIMENTS_DIR, "hybrid_features_train.csv")
FROZEN_FEATURES_TEST_CSV = os.path.join(EXPERIMENTS_DIR, "hybrid_features_test.csv")
FROZEN_MODEL_CONFIG_JSON = os.path.join(
    EXPERIMENTS_DIR, "recording_time_reanalysis", "freeze", "model_config.json"
)
RUBRIC_FILE = os.path.join(PROJECT_DIR, "mci_rubric_audio_en_v1.json")

# ---------------------------------------------------------------------------
# Frozen cohort numbers (verified from frozen inclusion_exclusion.csv)
# ---------------------------------------------------------------------------

TEST_ROWS_EXPECTED = 127          # phase2_role == "phase2_test"
TEST_SUBJECTS_EXPECTED = 74
REPEATS = 5
METHODS = ("direct", "rulers")    # A = direct, B = rulers; C = hybrid (local, no API)

# ---------------------------------------------------------------------------
# Frozen model / provider settings
# ---------------------------------------------------------------------------

PROVIDER_TYPE = "deepseek"
BASE_URL = "https://api.deepseek.com"
MODEL_DEFAULT = "deepseek-v4-flash"     # from .env.local at historical run time
# --- Pre-run protocol revisions (additive; recorded in protocol.md "Pre-run
# --- revisions"; the frozen historical settings above are untouched) ---
REQUEST_TIMEOUT_S = 120.0   # explicit per-attempt HTTP timeout (matches config.yaml timeout: 120)
SDK_MAX_RETRIES = 0         # SDK-level retries disabled; the outer retry policy owns all retries
REPAIR_DEFAULT = False      # main experiment: repair=off (protocol revision; historical chain had repair=on)
MODEL_ENV = "DEEPSEEK_MODEL"            # env override documented in protocol S4
MODEL_ALIASES_KNOWN = ["deepseek-v4-flash", "deepseek-chat", "gpt-4o"]
TEMPERATURE = 0.3                       # historical run_config.json temperature
MAX_TOKENS = 16384                      # stability_experiment.py:547 (frozen)
MAX_ATTEMPTS_PER_CALL = 3               # 1 planned attempt + at most 2 retries
RETRYABLE_ERROR_TYPES = ("CONNECTION_ERROR", "RATE_LIMIT_ERROR")

# --- Pre-run revision 8 (APPROVED by the author 2026-09-23; recorded in
# --- protocol.md "Pre-run revisions" and approval_record.md) ---
# deepseek-flash (= DeepSeek-V4.1-Flash): thinking mode is ON by default and
# silently IGNORES temperature. The official SDK way to disable it is
# extra_body={"thinking": {"type": "disabled"}} (api-docs.deepseek.com/
# guides/thinking_mode). The approved live run passes
# --thinking-mode disabled; "default" sends no thinking parameter at all
# (the historical frozen behavior, temperature then silently ignored by
# the API) and remains available only as the no-change reference.
THINKING_MODE_CHOICES = ("default", "disabled")
THINKING_EXTRA_BODY = {"thinking": {"type": "disabled"}}
MODEL_REVISION8_PROPOSED = "deepseek-flash"   # current official id (V4.1-Flash)
                                              # deepseek-v4-flash is the retired id,
                                              # routed to the same model per official docs

# ---------------------------------------------------------------------------
# Frozen rubric lock
# ---------------------------------------------------------------------------

# Computed by RubricLoader.lock() over mci_rubric_audio_en_v1.json
# (calculate_hash = sha256 of json.dumps(bundle, sort_keys=True, ensure_ascii=False))
RUBRIC_BUNDLE_HASH_EXPECTED = "c50650c777cca047c5f5c02bea2eadf52c57cd409032d8149b2ebdad190f6d68"
RUBRIC_BUNDLE_HASH_SHORT_EXPECTED = "38cc712159a384a9"

# ---------------------------------------------------------------------------
# Frozen thresholds (recording_time_reanalysis/freeze/model_config.json)
# ---------------------------------------------------------------------------

THRESHOLD_DIRECT = 0.50        # prespecified neutral midpoint on Direct risk (0-1)
THRESHOLD_PROMPT_RULERS = 0.07 # frozen train-set Youden on normalized RULERS risk (0-1)
THRESHOLD_HYBRID_LR = 0.55     # frozen train-set Youden on Hybrid LR proba
THRESHOLD_PROG_LR = 0.55       # frozen train-set Youden on Programmatic LR proba
LEGACY_RULERS_LABEL_THRESHOLD = 0.5  # historical artifact label (NOT a protocol threshold)

# ---------------------------------------------------------------------------
# Frozen feature columns (model_config.json features)
# ---------------------------------------------------------------------------

PROG_COLS = [
    "information_unit_count",
    "information_density",
    "speech_rate",
    "hesitation_rate",
    "long_pause_ratio",
    "transcript_length",
]
LLM_CIDS = ["C01", "C02", "C07", "C08", "C09", "C10"]
LLM_COLS = [f"C_{cid}_median" for cid in LLM_CIDS]
FEATURE_COLS = PROG_COLS + LLM_COLS                 # 12 columns
SIX_ITEM_CIDS = set(LLM_CIDS)                       # the six LLM-driven criteria
ALL_CIDS = [f"C{i:02d}" for i in range(1, 14)]      # C01..C13 (frozen 13-item rubric)

# ---------------------------------------------------------------------------
# Frozen hybrid pipeline (recorded in model_config.json; refit is deterministic)
# ---------------------------------------------------------------------------

HYBRID_PIPELINE = {
    "imputation": "SimpleImputer(strategy='median'), fit on corrected train only",
    "scaling": "StandardScaler(), fit on corrected train only",
    "model": "LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0, solver='lbfgs')",
    "threshold_grid": "np.arange(0.05, 0.96, 0.01), train-set Youden",
    "llm_aggregation_archived": "criterion_wise_median_across_valid_checklists",
}
# Frozen median imputation values are re-derived deterministically at analysis time
# from the frozen train archive (hybrid_features_train.csv); they are not constants.

# ---------------------------------------------------------------------------
# Frozen Direct prompt (verbatim from experiments/direct_llm_baseline.py)
# ---------------------------------------------------------------------------

PROMPT_DIRECT_SYSTEM = """You are a cognitive language screening assistant. Your task is to evaluate a spoken language transcript and determine whether the speaker's language patterns are more consistent with cognitive impairment or normal aging.

**IMPORTANT DISCLAIMER**: This is NOT a medical diagnosis. It is a language screening observation only. Do not make clinical claims.

You must output ONLY a valid JSON object. No Markdown code blocks, no explanation."""

# Cleaned Direct user template.
# Differences from the historical PROMPT_DIRECT_USER (direct_llm_baseline.py:36-71),
# both required by the clean-request protocol and documented in protocol.md S3:
#   1. {text} is inserted IN FULL (no [:3000] slice). All 127 frozen canonical texts
#      are <= 2156 chars, so the historical slice would have been a no-op; verified
#      by the generator (assert len(text) < 3000 for every selected text).
#   2. {sample_id} is the OPAQUE local id (CPR-###), never the real sample id.
# Everything else is verbatim, including the positive/control task wording (normal
# task definition, not sample metadata; see protocol.md S3 leakage rules).
PROMPT_DIRECT_USER_CLEAN = """## Transcript (Cookie Theft picture description task)
{text}

## Your Task
Based on the transcript above, assess whether the speaker's language patterns suggest cognitive impairment or normal control.

Consider:
- Word-finding difficulties (pauses, fillers like "um"/"uh", circumlocutions)
- Syntactic complexity (sentence structure, grammatical errors)
- Semantic coherence (topic maintenance, logical flow)
- Information density (how much information is conveyed)

## Output Format
You MUST output exactly this JSON structure:

{{
  "sample_id": "{sample_id}",
  "predicted_label": "positive",
  "risk_score": 0.75,
  "confidence": 0.8,
  "evidence": [
    "Frequent pauses and fillers suggesting word-finding difficulty",
    "Short, fragmented sentences"
  ],
  "reason": "The speaker shows multiple indicators of lexical retrieval difficulty and reduced syntactic complexity."
}}

## Rules
- `predicted_label` must be "positive" (cognitive impairment) or "control" (normal aging)
- `risk_score` must be between 0.0 (definitely normal) and 1.0 (definitely impaired)
- `confidence` must be between 0.0 (complete guess) and 1.0 (very confident)
- `evidence` must be specific observations from the transcript (at least 1, at most 5)
- `reason` must be a concise summary of your assessment
- If evidence is insufficient, still output your best guess but lower confidence

Now assess the transcript."""


def build_direct_messages_clean(text: str, opaque_id: str) -> list:
    """Cleaned Direct messages for one slot. Deterministic."""
    user_msg = PROMPT_DIRECT_USER_CLEAN.format(text=text, sample_id=opaque_id)
    return [
        {"role": "system", "content": PROMPT_DIRECT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def get_locked_rubric_bundle() -> dict:
    """Load + validate + lock the frozen EN rubric (cached). No network."""
    global _RUBRIC_BUNDLE_CACHE
    if _RUBRIC_BUNDLE_CACHE is not None:
        return _RUBRIC_BUNDLE_CACHE
    from mci_rubric_loader import RubricLoader

    loader = RubricLoader(RUBRIC_FILE)
    loader.load()
    ok, errors = loader.validate()
    if not ok:
        raise ValueError("Frozen rubric failed validation: %s" % "\n".join(errors))
    bundle_hash = loader.lock()
    assert bundle_hash == RUBRIC_BUNDLE_HASH_EXPECTED, (
        "Frozen rubric hash mismatch: %s" % bundle_hash
    )
    _RUBRIC_BUNDLE_CACHE = loader.get_bundle()
    return _RUBRIC_BUNDLE_CACHE


_RUBRIC_BUNDLE_CACHE = None


def build_rulers_messages(text: str) -> list:
    """
    Frozen RULERS (B) messages for one slot: mci_rulers_core.build_scoring_prompt
    with the locked EN rubric bundle. The B prompt contains NO sample id at all
    (frozen template behavior; the historical model echoed sample_id='unknown').
    The text is embedded as the frozen sentence bank: split_sentences_zh(text)
    with sentence-end delimiters in [。！？!?；;] dropped at splits, each line
    prefixed with [sent_id=N]. See protocol.md S3.
    """
    from mci_rulers_core import build_scoring_prompt

    bundle = get_locked_rubric_bundle()
    sample = {"sample_id": "unknown", "text": text}
    return build_scoring_prompt(sample, bundle, text_field="text", id_field="sample_id")


# ---------------------------------------------------------------------------
# Hashing helpers (payload / text integrity)
# ---------------------------------------------------------------------------

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_obj(obj) -> str:
    """Deterministic hash of a JSON-able object (payloads)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj, indent: int = 2):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def save_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
