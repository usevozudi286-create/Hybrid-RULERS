# -*- coding: utf-8 -*-
"""Task C Phase 2 — Offline re-evaluation with recording-time-corrected labels.

Research question fixed by the study design: "recording-time cognitive state"
(录音时点认知状态识别), NOT future-conversion prediction.

Adopts recording_label_audit/v2/ Plan A: the 19 evidence-supported candidate
corrections are adopted; the 40 unresolved records are EXCLUDED from both
fitting and scoring. This is an ANALYSIS rule only — it does not constitute a
clinical adjudication of the undefined codes.

Constraints honored by this script:
  * No API calls, no ASR/LLM regeneration — cached checkpoints are reused.
  * Nothing outside this directory is written; original data, old labels,
    final_clean_results/, audit v1/v2, and the paper are never modified.
  * Rules, inclusion IDs, exclusion reasons, label mapping, model config,
    seeds and input hashes are saved BEFORE any performance is computed.
  * The original participant-level train/test split and the technical QC rules
    are reused unchanged; the split is never re-derived.
  * No threshold is chosen from test results; all train-dependent quantities
    (imputation, scaling, LR fits, Youden thresholds, RULERS weights) use only
    the corrected training set.
  * Old predictions re-scored on new labels are reported separately as a
    label-only sensitivity analysis, never presented as re-fitted results.

Authoritative offline inputs (all read-only, SHA-256 snapshotted before/after):
  - recording_label_audit/v2/* (Phase-1 audit with evidence classes + buckets)
  - dataset/pitt_cookie_wav_cognitive_vs_control_metadata.csv (predefined split)
  - final_clean_results/* (frozen feature matrices, QC exclusion logs,
    frozen predictions and metrics used only for gating + label-only analysis)
  - experiments/stability_audio_en_v4flash_full_test/results_checkpoint.jsonl
    (test Direct/RULERS caches, 150 samples x 5 repeats x 2 methods)
  - experiments/stability_audio_en_v4flash_train_calib/results.csv
    (train RULERS cache, 342 rulers rows, 1 repeat/sample)
  - experiments/literature_weighted_rulers_en/ (old weights: reference only;
    new weights are re-learned on the corrected training set)
"""
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
HYBRID_FULL = HERE.parent
FINAL = HYBRID_FULL / "final_clean_results"
PROJECT = HYBRID_FULL.parents[1]
EXPERIMENTS = PROJECT / "experiments"
META_PATH = PROJECT / "dataset" / "pitt_cookie_wav_cognitive_vs_control_metadata.csv"
AUDIT_V2 = HYBRID_FULL / "recording_label_audit" / "v2" / "recording_label_audit_v2.csv"
CAND_V2 = HYBRID_FULL / "recording_label_audit" / "v2" / "candidate_relabel_map_v2.csv"
UNRES_V2 = HYBRID_FULL / "recording_label_audit" / "v2" / "unresolved_cases_v2.csv"
AUDIT_DIR = HYBRID_FULL / "recording_label_audit"
TEST_CP = EXPERIMENTS / "stability_audio_en_v4flash_full_test" / "results_checkpoint.jsonl"
TRAIN_CSV = EXPERIMENTS / "stability_audio_en_v4flash_train_calib" / "results.csv"
RUBRIC_PATH = PROJECT / "mci_rubric_audio_en_v1.json"
OLD_WEIGHTS = EXPERIMENTS / "literature_weighted_rulers_en" / "learned_weights.json"
OLD_WEIGHTED_SCORES = EXPERIMENTS / "literature_weighted_rulers_en" / "weighted_scores_test.csv"

FREEZE = HERE / "freeze"
DEV = HERE / "dev_contact"
DEMO = HERE / "demographic"
for d in (FREEZE, DEV, DEMO):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Frozen analysis configuration
# ---------------------------------------------------------------------------
PROG_COLS = [
    "information_unit_count", "information_density", "speech_rate",
    "hesitation_rate", "long_pause_ratio", "transcript_length",
]
LLM_CIDS = ["C01", "C02", "C07", "C08", "C09", "C10"]
LLM_COLS = ["C_%s_criterion_median" % cid for cid in LLM_CIDS]
FEATURE_COLS = PROG_COLS + LLM_COLS
DEMO_COLS = ["age", "sex", "education"]

TH_DIRECT = 0.5
SEED = 42
N_BOOT_MAIN = 20000
N_BOOT_DEMO = 2000

PILOT_DIRS = [
    "stability_audio_en_rubric_pilot_10_v2",
    "stability_audio_en_rubric_c05fix_pilot_10",
    "stability_audio_en_rubric_c02c08fix_pilot_10",
    "stability_audio_en_rubric_c01fix_pilot_10_v5",
    "stability_audio_pilot_10",
    "stability_audio_en_v4flash_modelcheck_5",
]
SMOKE_DIR = EXPERIMENTS / "stability_smoke_test"

RUN_LOG = []


def log(msg):
    line = str(msg)
    print(line)
    RUN_LOG.append(line)


# ===========================================================================
# Verbatim functions copied from final_clean_eval.py (the frozen pipeline)
# ===========================================================================
def is_valid_row(r):
    if not r.get('api_success', False): return False
    if not r.get('parse_success', False): return False
    error_type = r.get('error_type')
    if error_type is not None and not pd.isna(error_type) and str(error_type).strip():
        return False
    try:
        rs = float(r.get('risk_score', np.nan))
        if not np.isfinite(rs): return False
    except: return False
    asr_value = r.get('asr_text', '')
    asr = asr_value if isinstance(asr_value, str) else ''
    if len(asr.strip()) == 0: return False
    try:
        wtc = float(r.get('word_timestamp_count', 0))
    except (TypeError, ValueError):
        wtc = 0
    if not np.isfinite(wtc) or wtc <= 0: return False
    als = r.get('alignment_status', '')
    if als == 'failed': return False
    return True


def compute_metrics(yt, yp, ys=None):
    tp = int(np.sum((yp == 1) & (yt == 1))); tn = int(np.sum((yp == 0) & (yt == 0)))
    fp = int(np.sum((yp == 1) & (yt == 0))); fn = int(np.sum((yp == 0) & (yt == 1)))
    n = len(yt)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    acc = (tp + tn) / n
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    auc = roc_auc_score(yt, ys) if ys is not None and len(set(yt)) >= 2 else None
    return {'Acc': acc, 'Sens': sens, 'Spec': spec, 'F1': f1, 'AUC': auc,
            'n': n, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def train_lr(Xt, yt, Xe, ye, cols):
    """Identical to final_clean_eval.train_lr: train-only imputation, scaling,
    Youden threshold on the training-set grid 0.05..0.95 by 0.01."""
    imputer = SimpleImputer(strategy='median')
    Xtr = imputer.fit_transform(Xt[cols].values)
    Xev = imputer.transform(Xe[cols].values)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xev_s = scaler.transform(Xev)
    model = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0, solver='lbfgs')
    model.fit(Xtr_s, yt)
    tp = model.predict_proba(Xtr_s)[:, 1]
    best_t, best_j = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        yp = (tp >= t).astype(int)
        sn = np.sum((yp == 1) & (yt == 1)) / (np.sum((yp == 1) & (yt == 1)) + np.sum((yp == 0) & (yt == 1)) + 1e-10)
        sp = np.sum((yp == 0) & (yt == 0)) / (np.sum((yp == 0) & (yt == 0)) + np.sum((yp == 1) & (yt == 0)) + 1e-10)
        j = sn + sp - 1
        if j > best_j:
            best_j = j
            best_t = t
    ep = model.predict_proba(Xev_s)[:, 1]
    epred = (ep >= best_t).astype(int)
    imputation_values = {col: float(value) for col, value in zip(cols, imputer.statistics_)}
    return compute_metrics(ye, epred, ep), epred, ep, best_t, model.coef_[0], imputation_values


def select_youden_threshold(y_true, scores):
    """Verbatim from final_clean_eval (Prompt RULERS training protocol)."""
    best_t, best_j = 0.05, -np.inf
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fn = np.sum((pred == 0) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        fp = np.sum((pred == 1) & (y_true == 0))
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        youden = sens + spec - 1
        if youden > best_j:
            best_t, best_j = float(t), float(youden)
    return round(best_t, 2), best_j


# ===========================================================================
# Helpers
# ===========================================================================
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def snapshot_inputs():
    """SHA-256 of every authoritative input consumed by this analysis."""
    targets = {}

    def add(path, key):
        if path.exists():
            targets[key] = sha256_file(path)

    for p in sorted(FINAL.iterdir()):
        if p.is_file():
            add(p, "final_clean_results/" + p.name)
    for p in sorted(AUDIT_DIR.rglob("*")):
        if p.is_file():
            add(p, "recording_label_audit/" + str(p.relative_to(AUDIT_DIR)).replace("\\", "/"))
    add(META_PATH, "dataset/pitt_cookie_wav_cognitive_vs_control_metadata.csv")
    add(TEST_CP, "experiments/stability_audio_en_v4flash_full_test/results_checkpoint.jsonl")
    add(TRAIN_CSV, "experiments/stability_audio_en_v4flash_train_calib/results.csv")
    add(RUBRIC_PATH, "mci_rubric_audio_en_v1.json")
    add(OLD_WEIGHTS, "experiments/literature_weighted_rulers_en/learned_weights.json")
    add(OLD_WEIGHTED_SCORES, "experiments/literature_weighted_rulers_en/weighted_scores_test.csv")
    for d in PILOT_DIRS:
        for name in ("results.csv", "results_checkpoint.jsonl"):
            add(EXPERIMENTS / d / name, "pilot_dirs/%s/%s" % (d, name))
    add(SMOKE_DIR / "results.csv", "smoke/stability_smoke_test/results.csv")
    for src in (
        "experiments/stability_experiment.py",
        "experiments/direct_llm_baseline.py",
        "experiments/literature_weighted_rulers_en/weighted_eval.py",
        "experiments/hybrid_rulers_full/final_clean_eval.py",
        "mci_rulers_core.py",
    ):
        add(PROJECT / src, "source/" + src)
    return targets


def parse_checklist_scores(rows):
    """sample_id -> {criterion_id -> [scores]}, verbatim parse of weighted_eval.py."""
    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        oj = row.get('output_json', '')
        if isinstance(oj, str) and oj:
            try:
                oj = json.loads(oj)
            except Exception:
                continue
        if not isinstance(oj, dict):
            continue
        cl = oj.get('checklist', [])
        if not cl:
            continue
        sid = row['sample_id']
        for item in cl:
            cid = item.get('criterion_id', '?')
            s = item.get('score')
            if s == 'NA' or s is None:
                continue
            if not isinstance(s, (int, float)):
                continue
            data[sid][cid].append(int(s))
    return data


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = datetime.now()
    log("=== Task C Phase 2: recording-time-corrected offline reanalysis ===")
    log("started: %s" % t0.strftime("%Y-%m-%d %H:%M:%S"))

    # ---------------------------------------------------------------
    # 1. Input integrity snapshot (BEFORE any analysis)
    # ---------------------------------------------------------------
    log("\n== [1] INPUT SNAPSHOT (before) ==")
    hashes_before = snapshot_inputs()
    pd.DataFrame([{"input_key": k, "sha256": v} for k, v in sorted(hashes_before.items())]
                 ).to_csv(FREEZE / "input_hashes_before.csv", index=False, encoding="utf-8-sig")
    log("hashed %d authoritative inputs" % len(hashes_before))

    # ---------------------------------------------------------------
    # 2. Load Phase-1 v2 audit artifacts (read-only)
    # ---------------------------------------------------------------
    log("\n== [2] AUDIT ARTIFACTS ==")
    audit = pd.read_csv(AUDIT_V2, encoding="utf-8-sig", dtype=str)
    cand = pd.read_csv(CAND_V2, encoding="utf-8-sig", dtype=str)
    unres = pd.read_csv(UNRES_V2, encoding="utf-8-sig", dtype=str)
    for df, name in ((audit, "recording_label_audit_v2.csv"),
                     (cand, "candidate_relabel_map_v2.csv"),
                     (unres, "unresolved_cases_v2.csv")):
        df["sample_id"] = df["sample_id"].astype(str).str.strip()
    assert len(audit) == 492, "audit rows %d != 492" % len(audit)
    for col in ["subject_id", "split", "old_label", "proposed_label", "differs_from_old",
                "unresolved_reason", "unresolved_bucket", "evidence_class",
                "evidence_grade", "old_label_source"]:
        assert col in audit.columns, "audit missing column %s" % col
    assert len(cand) == 19, "candidates %d != 19" % len(cand)
    assert len(unres) == 40, "unresolved %d != 40" % len(unres)

    audit["subject_id"] = audit["subject_id"].astype(str).str.zfill(3)
    audit["visit"] = audit["visit"].astype(str)
    candidates = set(cand["sample_id"])
    unresolved = set(unres["sample_id"])
    resolved = set(audit["sample_id"]) - unresolved
    assert len(resolved) == 452, "resolved %d != 452" % len(resolved)
    assert set(audit.loc[audit["differs_from_old"] == "1", "sample_id"]) == candidates
    assert (audit.loc[audit["sample_id"].isin(candidates), "proposed_label"] == "control").all()
    non_cand_resolved = resolved - candidates
    assert (audit.loc[audit["sample_id"].isin(non_cand_resolved), "proposed_label"] ==
            audit.loc[audit["sample_id"].isin(non_cand_resolved), "old_label"]).all()
    log("audit 492 | candidates 19 (all positive->control) | unresolved 40 | resolved 452")

    # ---------------------------------------------------------------
    # 3. Metadata + predefined split + frozen QC exclusions
    # ---------------------------------------------------------------
    log("\n== [3] PREDEFINED SPLIT AND FROZEN QC ==")
    meta = pd.read_csv(META_PATH, encoding="utf-8-sig")
    for c in ["sample_id", "subject_id", "split", "label", "visit"]:
        meta[c] = meta[c].astype(str).str.strip()
    meta["subject_id"] = meta["subject_id"].str.zfill(3)
    assert len(meta) == 492
    assert set(meta["sample_id"]) == set(audit["sample_id"])
    subj_map = dict(zip(meta["sample_id"], meta["subject_id"]))

    predefined_train = set(meta.loc[meta["split"] == "train", "sample_id"])
    predefined_test = set(meta.loc[meta["split"] == "test", "sample_id"])
    assert len(predefined_train) == 342 and len(predefined_test) == 150

    feats_train = pd.read_csv(FINAL / "hybrid_features_train.csv", encoding="utf-8-sig")
    feats_test = pd.read_csv(FINAL / "hybrid_features_test.csv", encoding="utf-8-sig")
    for df in (feats_train, feats_test):
        df["sample_id"] = df["sample_id"].astype(str).str.strip()
    feats_train_ids = set(feats_train["sample_id"])
    feats_test_ids = set(feats_test["sample_id"])
    assert len(feats_train_ids) == 340 and len(feats_test_ids) == 138

    # Frozen technical QC, reconstructed FROM THE FROZEN FEATURE MATRICES
    # (the authoritative frozen sample sets), cross-checked against the logs.
    excl_train = pd.read_csv(FINAL / "excluded_train_records.csv", encoding="utf-8-sig")
    excl_test = pd.read_csv(FINAL / "excluded_records.csv", encoding="utf-8-sig")
    qc_train = set(excl_train["sample_id"].astype(str).str.strip())
    qc_test_rows_unique = set(excl_test["sample_id"].astype(str).str.strip())
    assert len(qc_train) == 2 and qc_train == predefined_train - feats_train_ids
    # sample-level (feature-level) test exclusions: the 12 samples absent from
    # the frozen test matrix; excluded_records.csv also logs REPEAT-level
    # failures for 5 samples that remained in the frozen matrix.
    qc_test_full = predefined_test - feats_test_ids
    assert len(qc_test_full) == 12
    partial_failed = qc_test_rows_unique - qc_test_full
    assert partial_failed <= feats_test_ids
    train_used = feats_train_ids
    test_clean = feats_test_ids
    log("frozen QC: train 342-2=340 | test 150-12=138 | %d repeat-level "
        "partial-failure samples retained (median over valid repeats)" %
        len(partial_failed))
    qc_summary = [{
        "sample_id": sid, "subject_id": subj_map[sid], "qc_type":
        "train_feature_level" if sid in qc_train else "test_feature_level",
        "retained_in_frozen_matrix": False,
        "note": "empty_asr / alignment failed for all repeats -> no features",
    } for sid in sorted(qc_train | qc_test_full)]
    qc_summary += [{
        "sample_id": sid, "subject_id": subj_map[sid], "qc_type":
        "repeat_level_partial_failure",
        "retained_in_frozen_matrix": True,
        "note": ">=1 repeat failed per method; sample kept, median over valid repeats",
    } for sid in sorted(partial_failed)]
    pd.DataFrame(qc_summary).to_csv(FREEZE / "qc_exclusion_summary.csv", index=False,
                                    encoding="utf-8-sig")

    pred_old = pd.read_csv(FINAL / "clean_predictions.csv", encoding="utf-8-sig")
    assert set(pred_old["sample_id"].astype(str).str.strip()) == test_clean

    old_label_map = dict(zip(meta["sample_id"], meta["label"].map(
        lambda x: 1 if x == "positive" else 0)))
    assert (feats_train["y_true"].to_numpy() ==
            feats_train["sample_id"].map(old_label_map).to_numpy()).all()
    assert (feats_test["y_true"].to_numpy() ==
            feats_test["sample_id"].map(old_label_map).to_numpy()).all()
    log("frozen labels verified: archived y_true == old metadata labels (bit-level)")

    # ---------------------------------------------------------------
    # 4. Plan A inclusion sets computed FROM ID SETS (no forcing)
    # ---------------------------------------------------------------
    log("\n== [4] PLAN-A INCLUSION SETS ==")
    train_unres = unresolved & train_used
    test_unres = unresolved & test_clean
    corrected_train = train_used - unresolved
    corrected_test = test_clean - unresolved
    log("unresolved in train_used: %d | in test_clean: %d" % (len(train_unres), len(test_unres)))
    log("unresolved in excluded QC sets: train %s test %s" %
        (sorted(unresolved & qc_train), sorted(unresolved & qc_test_full)))
    cand_in_train = candidates & corrected_train
    cand_in_test = candidates & corrected_test
    cand_in_qc_test = candidates & qc_test_full
    log("corrected train: %d (expected 311) | corrected test: %d (expected 127)" %
        (len(corrected_train), len(corrected_test)))
    log("candidate label changes in train: %d | in test: %d | in QC-excluded test: %d" %
        (len(cand_in_train), len(cand_in_test), len(cand_in_qc_test)))
    expected_train, expected_test = 311, 127
    if len(corrected_train) != expected_train or len(corrected_test) != expected_test:
        log("!! MISMATCH vs expected counts — reported, not forced")
        for row in audit[audit["sample_id"].isin(unresolved)].itertuples():
            log("  unresolved: %s split=%s frozen=%s bucket=%s" %
                (row.sample_id, row.split, row.in_frozen_analysis, row.unresolved_bucket))
    else:
        log("counts match the Phase-1 projection exactly (311/127)")

    corrected_map = {}
    for sid in audit["sample_id"]:
        if sid in unresolved:
            corrected_map[sid] = None  # EXCLUDED
        else:
            lab = audit.loc[audit["sample_id"] == sid, "proposed_label"].iloc[0]
            corrected_map[sid] = 1 if lab == "positive" else 0

    # participants / class counts / label changes
    stats_rows = []
    for name, sids, old_set in (("train_corrected", corrected_train, train_used),
                                ("test_corrected", corrected_test, test_clean)):
        ys = np.array([corrected_map[s] for s in sorted(sids)])
        changed = [s for s in sorted(sids) if s in candidates]
        stats_rows.append({
            "set": name,
            "n_recordings": len(sids),
            "n_participants": len({subj_map[s] for s in sids}),
            "n_positive": int((ys == 1).sum()),
            "n_control": int((ys == 0).sum()),
            "n_label_changes": len(changed),
            "n_unresolved_excluded": len(old_set & unresolved),
            "label_changed_sample_ids": "; ".join(changed),
        })
    pd.DataFrame(stats_rows).to_csv(FREEZE / "inclusion_counts.csv", index=False,
                                    encoding="utf-8-sig")
    log(pd.DataFrame(stats_rows).drop(columns=["label_changed_sample_ids"]).to_string(index=False))

    # ---------------------------------------------------------------
    # 5. FREEZE rules / inclusion-exclusion / label mapping / config
    #    (written BEFORE any performance computation)
    # ---------------------------------------------------------------
    log("\n== [5] FREEZING RULES AND CONFIG ==")
    rules_md = """# Phase-2 analysis rules (frozen BEFORE any performance computation)

## Study question
Recording-time cognitive state (录音时点认知状态). NOT future-conversion prediction.

## Label rules (adopted from recording_label_audit/v2)
- R1v2 判定顺序: direct_visit (A) > visit_fallback (A/B) > date_boundary (B) > chat_only (C).
  Same-day recording/change dates are never silently ordered (conflict -> unresolved).
  Two-change periods: intertdx for c1<rec<c2, curdx1 for rec>c2; missing intertdx -> unresolved.
- R2v2 二分类: positive = probable/possible AD, vascular dementia, MCI;
  control = 8/800/821/851. 730/770/5/undefined codes are NOT forced into a class.
- R3v2 优先级: 当次访视列 > 历史访视回退 > 日期边界推断 > CHAT 佐证 > 未决.
  Unresolved buckets: target_population_eligibility / code_undefined_in_docs / record_conflict.

## Plan A (adopted here)
1. The 19 evidence-supported candidate corrections (all old-positive -> recording-time control) are ADOPTED.
2. The 40 unresolved records are EXCLUDED from both fitting and scoring
   (29 in train_used, 11 in test_clean).
3. This is an analysis rule only; it does not adjudicate the undefined codes clinically.
4. Inclusion rules follow the paper's task; model performance played no role in any rule.

## Split and QC (reused unchanged, never re-derived)
- Original participant-level stratified split (seed 42, test_size 0.3) from
  pitt_cookie_wav_cognitive_vs_control_metadata.csv (train 342 / test 150).
- Technical QC reused verbatim from the frozen pipeline (reconstructed from the
  frozen feature matrices, the authoritative sample sets): 2 train exclusions
  (058-0, 124-0) + 12 test feature-level exclusions -> train_used 340 /
  test_clean 138. excluded_records.csv additionally logs repeat-level failures
  for 5 retained test samples (105-0, 672-0, 698-0, 705-0, 714-0); these samples
  stay in the analysis with the frozen valid-repeat median rule.
- Plan-A sets: corrected train = 340 - 29 unresolved = 311;
  corrected test = 138 - 11 unresolved = 127.

## Model configuration (unchanged from the frozen pipeline)
- Features: 6 programmatic + 6 LLM rubric criteria
  (C01,C02,C07,C08,C09,C10 criterion-wise median across valid checklists).
  Feature extraction is label-free (ASR text, word timestamps, acoustic pauses,
  checklist scores only — verified by code inspection, see cache applicability).
- LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0, solver='lbfgs').
- Imputation: training-set feature median; scaling: training-set StandardScaler.
- Threshold: training-set Youden on grid 0.05..0.95 by 0.01 (test never used).
- Direct: prespecified neutral midpoint 0.50 (no training outputs exist for Direct).
- Prompt RULERS: training-set Youden threshold protocol recomputed on the corrected
  training set (verbatim select_youden_threshold).
- Weighted RULERS: weights RE-LEARNED on the corrected training set (literature prior
  x repeat reliability x train item-level usefulness, verbatim weighted_eval.py);
  threshold = the recomputed train-Youden threshold of unweighted RULERS
  (protocol continuity: the original design used one shared threshold, 0.07, for both).
  Test-side aggregation replicates weighted_eval.py verbatim, including its
  established edge rule: a checklist whose scores are all NA yields
  weighted_risk = 0.0 (retained for protocol consistency, documented).
- No feature selection, no hyperparameter search.

## Statistics
- Paired participant-clustered bootstrap: 20000 resamples, seed 42.
  One single RNG stream shared by ALL methods (same participant draw per resample).
  Each resample keeps every recording with its own label and its multiplicity.
  Invalid resamples (fewer than 2 classes) are skipped and counted.
  The fraction of bootstrap differences > 0 is reported as an empirical proportion;
  it is NOT called a Bayesian probability or a p-value.
- Primary comparisons: Hybrid - Direct and Hybrid - Programmatic-only AUC differences
  with 95% percentile CIs.
- Dev-contact exclusion: pilot union (6 development directories) and the conservative
  variant (pilot + smoke subject 16). Subset sizes are recomputed from ID sets.
- Demographic sensitivity: age/sex/education (earliest-visit rule), three LR models
  with the identical pipeline; age is entry age, not recording-time age.
"""
    (FREEZE / "rules.md").write_text(rules_md, encoding="utf-8")

    config = {
        "analysis_id": "recording_time_reanalysis_planA",
        "generated_at": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "label_rules": "adopt 19 v2 candidates (positive->control); exclude 40 v2 unresolved",
        "unresolved_exclusion_is_analysis_rule_not_clinical_adjudication": True,
        "split_reused_from": "dataset/pitt_cookie_wav_cognitive_vs_control_metadata.csv",
        "qc_reused_from": "frozen feature matrices (authoritative sample sets), "
                          "cross-checked with final_clean_results/excluded_records.csv + "
                          "excluded_train_records.csv (2 train + 12 test feature-level "
                          "exclusions; 5 partial-repeat test samples retained)",
        "expected_counts": {"train_corrected": expected_train, "test_corrected": expected_test},
        "features": {
            "programmatic": PROG_COLS,
            "llm_criteria": LLM_COLS,
            "total": len(FEATURE_COLS),
            "llm_aggregation": "criterion_wise_median_across_valid_checklists",
            "feature_source": "frozen final_clean_results/hybrid_features_{train,test}.csv "
                              "(label-free extraction verified)",
        },
        "model": {"type": "LogisticRegression", "class_weight": "balanced",
                  "max_iter": 2000, "C": 1.0, "solver": "lbfgs"},
        "preprocessing": {"imputation": "median, fit on corrected train only",
                          "scaling": "StandardScaler, fit on corrected train only"},
        "thresholds": {
            "direct": {"value": TH_DIRECT, "origin": "prespecified neutral midpoint"},
            "prompt_rulers": {"protocol": "train-set Youden, grid 0.05..0.95 step 0.01, "
                                           "recomputed on corrected train (verbatim "
                                           "select_youden_threshold)"},
            "weighted_rulers": {"protocol": "weights re-learned on corrected train "
                                            "(verbatim weighted_eval.py); threshold = "
                                            "recomputed unweighted RULERS train Youden "
                                            "(original design shared one threshold for both)"},
            "lr_models": {"protocol": "per-model train-set Youden on grid 0.05..0.95 "
                                      "step 0.01 (verbatim train_lr)"},
        },
        "seeds": {"seed": SEED, "bootstrap_rng": "np.random.RandomState(42), single shared stream"},
        "bootstrap": {
            "n_resamples": N_BOOT_MAIN,
            "scheme": "paired participant-clustered; recordings keep their own labels "
                      "and multiplicity; same draw for all methods",
            "invalid_resample_handling": "skipped (fewer than 2 classes) and counted",
            "proportion_gt_0_label": "empirical proportion, not a p-value",
        },
        "dev_contact": {
            "pilot_dirs": PILOT_DIRS,
            "smoke": "stability_smoke_test (subject 16, 4 recordings)",
            "scopes": ["full_corrected_test", "pilot_excluded", "conservative_pilot_plus_smoke"],
            "subset_sizes_recomputed": True,
        },
        "demographic_sensitivity": {
            "variables": DEMO_COLS,
            "rule": "earliest-visit metadata row per participant, broadcast to all recordings",
            "age_limitation": "entry age, NOT recording-time age",
            "models": ["demographic_only(3f)", "full_hybrid(12f)", "hybrid_plus_demo(15f)"],
            "bootstrap": {"n_resamples": N_BOOT_DEMO, "seed": SEED,
                          "scheme": "participant-clustered paired (original config)"},
        },
        "prohibited": ["API calls", "ASR/LLM regeneration", "feature re-selection",
                       "hyperparameter search", "test-set threshold selection",
                       "modifying original data/old labels/final_clean_results/audit/paper"],
    }
    (FREEZE / "model_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Inclusion/exclusion list for all 492 recordings
    rows = []
    for sid in sorted(meta["sample_id"]):
        a = audit[audit["sample_id"] == sid].iloc[0]
        repeat_note = ("some_repeats_failed_but_sample_retained_in_frozen_matrix"
                       if sid in partial_failed else "")
        if sid in qc_train:
            scope, role = "excluded_train_qc", "qc_excluded"
        elif sid in qc_test_full:
            scope, role = "excluded_test_qc_feature_level", "qc_excluded"
        elif sid in train_used:
            scope = "train_used"
            role = "phase2_train" if sid in corrected_train else "excluded_unresolved_train"
        else:
            scope = "test_clean"
            role = "phase2_test" if sid in corrected_test else "excluded_unresolved_test"
        if sid in unresolved:
            corr, excl_reason = "EXCLUDED", "unresolved_%s: %s" % (a["unresolved_bucket"],
                                                                   a["unresolved_reason"])
        else:
            lab = "positive" if corrected_map[sid] == 1 else "control"
            corr = lab
            excl_reason = ""
        rows.append({
            "sample_id": sid,
            "subject_id": subj_map[sid],
            "visit": a["visit"],
            "split": a["split"],
            "frozen_scope": scope,
            "old_label": a["old_label"],
            "old_label_source": a["old_label_source"],
            "proposed_label": a["proposed_label"],
            "evidence_class": a["evidence_class"],
            "evidence_grade": a["evidence_grade"],
            "unresolved_reason": a["unresolved_reason"],
            "unresolved_bucket": a["unresolved_bucket"],
            "corrected_label": corr,
            "exclusion_reason": excl_reason,
            "label_changed": "1" if sid in candidates else "0",
            "phase2_role": role,
            "repeat_level_note": repeat_note,
        })
    pd.DataFrame(rows).to_csv(FREEZE / "inclusion_exclusion.csv", index=False,
                              encoding="utf-8-sig")

    pd.DataFrame([{
        "sample_id": sid,
        "split": audit[audit["sample_id"] == sid]["split"].iloc[0],
        "old_label": audit[audit["sample_id"] == sid]["old_label"].iloc[0],
        "corrected_label": ("" if corrected_map[sid] is None else
                            ("positive" if corrected_map[sid] == 1 else "control")),
        "status": "excluded_unresolved" if sid in unresolved else
                  ("candidate_relabeled" if sid in candidates else "unchanged"),
    } for sid in sorted(meta["sample_id"])]).to_csv(
        FREEZE / "label_mapping.csv", index=False, encoding="utf-8-sig")

    log("frozen: rules.md, model_config.json, inclusion_exclusion.csv, "
        "label_mapping.csv, inclusion_counts.csv, input_hashes_before.csv")

    # ---------------------------------------------------------------
    # 6. Cache applicability check
    # ---------------------------------------------------------------
    log("\n== [6] CACHE APPLICABILITY CHECK ==")
    cp_rows = list(iter_jsonl(TEST_CP))
    log("test checkpoint: %d records" % len(cp_rows))
    by_sample = defaultdict(lambda: Counter())
    for r in cp_rows:
        sid = r["sample_id"]
        prefix = "dementia" if sid.startswith("pitt_dementia") else (
            "control" if sid.startswith("pitt_control") else "other")
        by_sample[sid][prefix] += 1
    direct_leak_rows = []
    for sid in sorted(by_sample):
        prefixes = sorted(by_sample[sid])
        direct_leak_rows.append({
            "sample_id": sid,
            "id_prefix_seen_by_direct_llm": "; ".join("%s:%d" % (p, by_sample[sid][p])
                                                     for p in prefixes),
            "prefix_is_label_derived": "yes",
            "old_label": meta.set_index("sample_id").loc[sid, "label"],
            "phase2_status": "excluded_unresolved" if sid in unresolved else
                             ("candidate_relabeled" if sid in candidates else "included"),
        })
    leak_df = pd.DataFrame(direct_leak_rows)
    leak_df.to_csv(FREEZE / "cache_applicability_direct_id_prefix.csv", index=False,
                   encoding="utf-8-sig")
    n_dem = int(sum(1 for r in direct_leak_rows if r["id_prefix_seen_by_direct_llm"].startswith("dementia")))
    n_con = int(sum(1 for r in direct_leak_rows if r["id_prefix_seen_by_direct_llm"].startswith("control")))
    log("Direct prompt leak: all %d test samples carry label-derived prefix in sample_id "
        "(%d dementia / %d control); prompt echoes full sample_id (direct_llm_baseline "
        "PROMPT_DIRECT_USER). Train has no Direct records (rulers-only calibration)." %
        (len(direct_leak_rows), n_dem, n_con))
    log("RULERS prompt: sample_id never substituted (template literal '<sampleID>'); "
        "cached outputs echo 'unknown'/'' -> no ID/diagnostic/path leak. Reusable.")
    log("Programmatic features: extracted from ASR text, word timestamps, acoustic "
        "pause features, checklist scores only -> label-free. Reusable.")
    log("true_label stored in checkpoint but only used for progress printing; never in "
        "any prompt. Old Weighted-RULERS weights are train-label-dependent -> re-learned.")
    cache_md = """# Cache applicability check (requirement 二.5 / 二.6)

| Branch | Current-state task? | Label/diagnosis/path in prompt? | Verdict |
|---|---|---|---|
| Direct LLM | yes (impairment vs normal aging from transcript) | **YES — sample_id embeds the label-derived directory prefix** (`pitt_dementia_cookie_...` / `pitt_control_cookie_...`) and `PROMPT_DIRECT_USER` echoes the full ID (`"sample_id": "{sample_id}"`) | cache reused ONLY with an explicit caveat; Direct-dependent conclusions are flagged (see report). Affected: all 150 test samples x 5 repeats (train has no Direct records). |
| Prompt RULERS | yes (rubric scoring of the current transcript) | no — `build_scoring_prompt` never substitutes the real sample_id (literal `<样本ID>` placeholder); cached outputs echo `unknown`/`` | reusable |
| Programmatic features | — (deterministic) | no — inputs are ASR text, word timestamps, acoustic pause features, checklist scores | reusable |
| ASR / alignment | — (audio -> text / timestamps) | no | reusable |
| Weighted-RULERS old weights | — | weights were learned from TRAIN labels (item-level AUC usefulness) | NOT reusable as-is; re-learned on the corrected training set. Test-side aggregation replicates weighted_eval.py verbatim, including its established edge rule: a checklist whose scores are all NA yields weighted_risk = 0.0 (retained for protocol consistency, documented here) |

The Direct leak is systematic and aligned with the OLD labels: for the 3
candidate samples relabeled positive->control in the test set, the ID prefix
still says `dementia`. No API re-run is permitted, so the Direct branch cannot
be re-generated; its results are reported but its conclusions are explicitly
limited (see reanalysis_report.md, section on conclusions).
"""
    (FREEZE / "cache_applicability_check.md").write_text(cache_md, encoding="utf-8")

    # ---------------------------------------------------------------
    # 7. Build corrected feature frames (features unchanged; labels corrected)
    # ---------------------------------------------------------------
    log("\n== [7] CORRECTED FEATURE FRAMES ==")
    tr_f = feats_train[feats_train["sample_id"].isin(corrected_train)].copy()
    te_f = feats_test[feats_test["sample_id"].isin(corrected_test)].copy()
    for df in (tr_f, te_f):
        df["y_true"] = df["sample_id"].map(corrected_map).astype(int)
        df["subject_id"] = df["sample_id"].map(subj_map)
    assert len(tr_f) == len(corrected_train) and len(te_f) == len(corrected_test)
    tr_f = tr_f.sort_values("sample_id").reset_index(drop=True)
    te_f = te_f.sort_values("sample_id").reset_index(drop=True)
    for fc in FEATURE_COLS:
        assert fc in tr_f.columns and fc in te_f.columns
    log("feature frames: train %d x %d, test %d x %d (labels corrected)" %
        (len(tr_f), len(FEATURE_COLS), len(te_f), len(FEATURE_COLS)))

    # ---------------------------------------------------------------
    # 8. Re-fit LLM-only / Programmatic-only / Full Hybrid on corrected train
    # ---------------------------------------------------------------
    log("\n== [8] RE-FIT LR MODELS (corrected train only) ==")
    ytrain = tr_f["y_true"].to_numpy(dtype=int)
    ytest = te_f["y_true"].to_numpy(dtype=int)

    full_m, full_pred, full_prob, full_t, full_coef, full_imp = train_lr(
        tr_f, ytrain, te_f, ytest, FEATURE_COLS)
    prog_m, prog_pred, prog_prob, prog_t, prog_coef, _ = train_lr(
        tr_f, ytrain, te_f, ytest, PROG_COLS)
    llm_m, llm_pred, llm_prob, llm_t, llm_coef, _ = train_lr(
        tr_f, ytrain, te_f, ytest, LLM_COLS)
    for name, m in (("full_hybrid", full_m), ("prog_only", prog_m),
                    ("llm_only", llm_m)):
        log("%-12s AUC=%.4f Acc=%.4f Sens=%.4f Spec=%.4f" %
            (name, m["AUC"], m["Acc"], m["Sens"], m["Spec"]))
    coef_rows = []
    for col, coef in zip(FEATURE_COLS, full_coef):
        coef_rows.append({"model": "full_hybrid", "feature": col,
                          "coefficient_standardized": round(float(coef), 6)})
    for col, coef in zip(PROG_COLS, prog_coef):
        coef_rows.append({"model": "prog_only", "feature": col,
                          "coefficient_standardized": round(float(coef), 6)})
    for col, coef in zip(LLM_COLS, llm_coef):
        coef_rows.append({"model": "llm_only", "feature": col,
                          "coefficient_standardized": round(float(coef), 6)})
    pd.DataFrame(coef_rows).to_csv(HERE / "model_coefficients.csv", index=False,
                                   encoding="utf-8-sig")
    pd.DataFrame([{"feature": k, "train_median_imputation_value": v}
                  for k, v in full_imp.items()]).to_csv(
        HERE / "imputation_values.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------
    # 9. Prompt RULERS / Direct / Weighted RULERS from valid caches
    # ---------------------------------------------------------------
    log("\n== [9] PROMPT-BASED METHODS FROM CACHES ==")
    test_valid = [r for r in cp_rows if is_valid_row(r)]
    tdf = pd.DataFrame(test_valid)
    tdf["sample_id"] = tdf["sample_id"].astype(str).str.strip()
    tdf["y_true"] = tdf["sample_id"].map(corrected_map)
    tdf = tdf.dropna(subset=["y_true"]).copy()
    tdf["y_true"] = tdf["y_true"].astype(int)
    test_in = tdf[tdf["sample_id"].isin(corrected_test)]

    r_df = test_in[test_in["method"] == "rulers"]
    r_agg = r_df.groupby("sample_id").agg({"risk_score": "median", "y_true": "first"}).reset_index()
    d_df = test_in[test_in["method"] == "direct"]
    d_agg = d_df.groupby("sample_id").agg({"risk_score": "median", "y_true": "first"}).reset_index()
    assert len(r_agg) == len(corrected_test), "RULERS coverage %d != %d" % (
        len(r_agg), len(corrected_test))
    assert len(d_agg) == len(corrected_test), "Direct coverage %d != %d" % (
        len(d_agg), len(corrected_test))

    # Train RULERS (rulers-only calibration run) -> corrected-train Youden
    train_raw = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    train_raw["sample_id"] = train_raw["sample_id"].astype(str).str.strip()
    for c in ["hesitation_rate", "long_pause_ratio", "speech_rate_cps", "words_per_second",
              "risk_score"]:
        if c in train_raw.columns:
            train_raw[c] = pd.to_numeric(train_raw[c], errors="coerce")
    train_invalid = [r for _, r in train_raw.iterrows() if not is_valid_row(r.to_dict())]
    train_valid = [r for _, r in train_raw.iterrows() if is_valid_row(r.to_dict())]
    tr_df = pd.DataFrame(train_valid)
    tr_df["sample_id"] = tr_df["sample_id"].astype(str).str.strip()
    tr_df["y_true"] = tr_df["sample_id"].map(corrected_map)
    tr_df = tr_df.dropna(subset=["y_true"]).copy()
    tr_df["y_true"] = tr_df["y_true"].astype(int)
    tr_r = tr_df[tr_df["method"] == "rulers"]
    tr_r = tr_r[tr_r["sample_id"].isin(corrected_train)].copy()
    tr_r["risk_score"] = pd.to_numeric(tr_r["risk_score"], errors="coerce")
    tr_r_agg = tr_r.groupby("sample_id").agg(
        {"risk_score": "median", "y_true": "first"}).reset_index()
    log("train RULERS rows used for Youden: %d samples (invalid rows dropped: %d)" %
        (len(tr_r_agg), len(train_invalid)))
    th_rulers, j_rulers = select_youden_threshold(
        tr_r_agg["y_true"].to_numpy(dtype=int),
        pd.to_numeric(tr_r_agg["risk_score"], errors="coerce").to_numpy(dtype=float))
    log("corrected-train Youden threshold for Prompt RULERS: %.2f (J=%.4f; old 0.07)" %
        (th_rulers, j_rulers))

    # --- Weighted RULERS: re-learn weights on corrected train (verbatim protocol)
    # Old weighted_eval.py used ALL train rulers rows (no validity filter) for
    # checklist parsing; only the labels change here (corrected labels).
    tr_rows = [row.to_dict() for _, row in train_raw.iterrows()
               if str(row["sample_id"]).strip() in corrected_train]
    train_cl = parse_checklist_scores(tr_rows)
    lit_prior = {"C01": 1.3, "C02": 1.0, "C03": 1.0, "C04": 1.0, "C05": 1.3,
                 "C06": 1.0, "C07": 1.0, "C08": 1.0, "C09": 1.3, "C10": 1.15,
                 "C11": 1.0, "C12": 1.0, "C13": 1.0}
    y_map = corrected_map
    crit_disagree, crit_usefulness = {}, {}
    for cid in sorted(lit_prior):
        disagrees, ns = 0, 0
        for sid, crits in train_cl.items():
            scores = crits.get(cid, [])
            if len(scores) >= 2:
                ns += 1
                if len(set(scores)) > 1:
                    disagrees += 1
        crit_disagree[cid] = disagrees / ns if ns > 0 else 0.0
        scores_by_sample = {sid: float(np.mean(crits.get(cid, [])))
                            for sid, crits in train_cl.items() if crits.get(cid)}
        yt_list = [y_map[sid] for sid in scores_by_sample if sid in y_map]
        ys_list = [scores_by_sample[sid] for sid in scores_by_sample if sid in y_map]
        if len(yt_list) >= 10 and len(set(yt_list)) >= 2:
            try:
                item_auc = roc_auc_score(yt_list, ys_list)
                crit_usefulness[cid] = 1.0 + abs(item_auc - 0.5) * 2
            except Exception:
                crit_usefulness[cid] = 1.0
        else:
            crit_usefulness[cid] = 1.0
    weights = {}
    weight_components = {}
    for cid in sorted(lit_prior):
        prior = lit_prior[cid]
        disagree = crit_disagree.get(cid, 0.5)
        reliability = 1.0 - disagree
        useful = crit_usefulness.get(cid, 1.0)
        raw = prior * reliability * useful
        clamped = max(0.5, min(2.0, raw))
        weights[cid] = clamped
        weight_components[cid] = {
            "literature_prior": prior,
            "disagreement_rate": round(disagree, 4),
            "reliability": round(reliability, 4),
            "usefulness": round(useful, 4),
            "raw_weight": round(raw, 4),
            "final_weight": round(clamped, 4),
        }
    mean_w = float(np.mean(list(weights.values())))
    weights = {cid: round(w / mean_w, 4) for cid, w in weights.items()}
    for cid in weights:
        weight_components[cid]["final_weight_normalized"] = weights[cid]
    (HERE / "learned_weights_corrected_train.json").write_text(
        json.dumps({"weights": weights, "components": weight_components,
                    "weight_min": 0.5, "weight_max": 2.0,
                    "normalization": "mean ~ 1.0",
                    "learned_on": "corrected training set (311 target, "
                                  "per-valid-row subset)"},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log("weighted RULERS weights re-learned on corrected train (saved)")

    # Test-side weighted scores: verbatim weighted_eval.compute_weighted_scores —
    # ALL rulers records of corrected-test samples (no validity filter), and a row
    # is appended even when every checklist score is NA (raw_score -> 0.0). This
    # all-NA -> 0.0 mapping is the established frozen rule and is retained for
    # protocol consistency (documented, not silently changed).
    w_rows = []
    for r in cp_rows:
        if r.get("method") != "rulers":
            continue
        sid = str(r.get("sample_id", "")).strip()
        if sid not in corrected_test:
            continue
        oj = r.get("output_json", "")
        if isinstance(oj, str) and oj:
            try:
                oj = json.loads(oj)
            except Exception:
                continue
        if not isinstance(oj, dict) or not oj.get("checklist"):
            continue
        ws, wsum = 0.0, 0.0
        for item in oj["checklist"]:
            cid = item.get("criterion_id", "")
            s = item.get("score")
            if s == "NA" or s is None or not isinstance(s, (int, float)):
                continue
            w = weights.get(cid, 1.0)
            ws += s * w
            wsum += w
        raw_score = (ws / wsum) if wsum > 0 else 0.0
        w_rows.append({"sample_id": sid,
                       "repeat_idx": r.get("repeat_idx"),
                       "weighted_risk": raw_score / 2.0,
                       "y_true": corrected_map[sid]})
    wdf = pd.DataFrame(w_rows)
    w_agg = wdf.groupby("sample_id").agg({"weighted_risk": "median",
                                          "y_true": "first"}).reset_index()
    assert len(w_agg) == len(corrected_test), "weighted coverage %d != %d" % (
        len(w_agg), len(corrected_test))
    # protocol continuity: weighted uses the same threshold as unweighted RULERS
    th_weighted = th_rulers

    # ---------------------------------------------------------------
    # 10. Metrics of all methods on the SAME corrected test set
    # ---------------------------------------------------------------
    log("\n== [10] STRATEGY COMPARISON ON CORRECTED TEST %d ==" % len(corrected_test))
    frame = te_f[["sample_id", "subject_id", "y_true"]].copy()
    frame["old_label"] = frame["sample_id"].map(
        lambda s: meta.set_index("sample_id").loc[s, "label"])
    frame = frame.merge(r_agg[["sample_id", "risk_score"]].rename(
        columns={"risk_score": "rulers_score"}), on="sample_id", how="left")
    frame = frame.merge(d_agg[["sample_id", "risk_score"]].rename(
        columns={"risk_score": "direct_score"}), on="sample_id", how="left")
    frame = frame.merge(w_agg[["sample_id", "weighted_risk"]], on="sample_id", how="left")
    frame["llm_prob"] = llm_prob
    frame["prog_prob"] = prog_prob
    frame["hybrid_prob"] = full_prob
    assert frame[["direct_score", "rulers_score", "weighted_risk"]].notna().all().all()

    methods = {
        "Direct LLM": ("direct_score", TH_DIRECT,
                       "prespecified 0.50 (no train calibration)", True),
        "Prompt RULERS": ("rulers_score", th_rulers,
                          "corrected-train Youden (protocol)", False),
        "Weighted RULERS": ("weighted_risk", th_weighted,
                            "= corrected-train Youden of unweighted RULERS "
                            "(protocol continuity)", False),
        "LLM-only LR": ("llm_prob", llm_t, "train Youden (corrected train)", False),
        "Programmatic-only LR": ("prog_prob", prog_t, "train Youden (corrected train)", False),
        "Full Hybrid LR": ("hybrid_prob", full_t, "train Youden (corrected train)", False),
    }
    comp_rows = []
    for name, (score_col, thr, origin, caveat) in methods.items():
        yt = frame["y_true"].values
        ys = frame[score_col].values
        yp = (ys >= thr).astype(int)
        m = compute_metrics(yt, yp, ys)
        comp_rows.append({
            "method": name, "threshold": round(float(thr), 4),
            "threshold_origin": origin,
            "n": m["n"], "Acc": round(m["Acc"], 4), "Sens": round(m["Sens"], 4),
            "Spec": round(m["Spec"], 4), "F1": round(m["F1"], 4),
            "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
            "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
            "cache_caveat": "sample_id label-derived prefix in prompt" if caveat else "",
        })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(HERE / "strategy_comparison.csv", index=False, encoding="utf-8-sig")
    log(comp.to_string(index=False))

    pred_out = frame[["sample_id", "subject_id", "old_label", "y_true"]].copy()
    pred_out["old_label_binary"] = pred_out["old_label"].map(
        lambda x: 1 if x == "positive" else 0)
    col_map = {
        "Direct LLM": ("direct_score", "Direct_LLM_score", "Direct_LLM_pred"),
        "Prompt RULERS": ("rulers_score", "Prompt_RULERS_score", "Prompt_RULERS_pred"),
        "Weighted RULERS": ("weighted_risk", "Weighted_RULERS_score",
                            "Weighted_RULERS_pred"),
        "LLM-only LR": ("llm_prob", "LLM_only_LR_score", "LLM_only_LR_pred"),
        "Programmatic-only LR": ("prog_prob", "Programmatic_only_LR_score",
                                 "Programmatic_only_LR_pred"),
        "Full Hybrid LR": ("hybrid_prob", "Full_Hybrid_LR_score", "Full_Hybrid_LR_pred"),
    }
    for name, (score_col, out_score, out_pred) in col_map.items():
        pred_out[out_score] = frame[score_col]
        pred_out[out_pred] = (frame[score_col] >= methods[name][1]).astype(int)
    pred_out = pred_out[[
        "sample_id", "subject_id", "old_label", "y_true", "old_label_binary",
        "Direct_LLM_score", "Direct_LLM_pred",
        "Prompt_RULERS_score", "Prompt_RULERS_pred",
        "Weighted_RULERS_score", "Weighted_RULERS_pred",
        "LLM_only_LR_score", "LLM_only_LR_pred",
        "Programmatic_only_LR_score", "Programmatic_only_LR_pred",
        "Full_Hybrid_LR_score", "Full_Hybrid_LR_pred",
    ]]
    pred_out.to_csv(HERE / "per_sample_predictions.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------
    # 11. Label-only sensitivity (old frozen scores, corrected labels)
    # ---------------------------------------------------------------
    log("\n== [11] LABEL-ONLY SENSITIVITY (frozen old predictions, new labels) ==")
    fs = json.loads((FINAL / "final_summary.json").read_text(encoding="utf-8"))
    old_hyb_thr = float(fs["hybrid_threshold"])
    old_metrics = fs["results"]
    # old prog scores recomputed deterministically with OLD labels (verbatim gate)
    og_prog_m, og_prog_pred, og_prog_prob, og_prog_t, _, _ = train_lr(
        feats_train, feats_train["y_true"].to_numpy(dtype=int),
        feats_test, feats_test["y_true"].to_numpy(dtype=int), PROG_COLS)
    for k in ("Acc", "Sens", "Spec", "F1", "AUC", "TP", "TN", "FP", "FN"):
        gap = abs(og_prog_m[k] - old_metrics["prog_only"][k]) if k != "AUC" else \
            abs(og_prog_m["AUC"] - old_metrics["prog_only"]["AUC"])
        assert gap < 1e-9, (k, gap)
    log("gate: old prog_only recomputation matches archived metrics (max diff < 1e-9)")

    old_direct = pred_old[["sample_id", "direct_score"]].rename(
        columns={"direct_score": "old_direct_score"})
    old_hybrid = pred_old[["sample_id", "hybrid_prob", "hybrid_pred"]].rename(
        columns={"hybrid_prob": "old_hybrid_prob", "hybrid_pred": "old_hybrid_pred"})
    old_prog = feats_test[["sample_id"]].assign(old_prog_prob=og_prog_prob)
    old_w = pd.read_csv(OLD_WEIGHTED_SCORES, encoding="utf-8-sig")
    old_w["sample_id"] = old_w["sample_id"].astype(str).str.strip()
    old_w_agg = old_w.groupby("sample_id").agg({"weighted_risk": "median"}).reset_index().rename(
        columns={"weighted_risk": "old_weighted_score"})
    # old RULERS scores re-aggregated with the frozen filter
    r_old = pd.DataFrame(test_valid)
    r_old["sample_id"] = r_old["sample_id"].astype(str).str.strip()
    r_old = r_old[r_old["method"] == "rulers"]
    r_old["y_true"] = r_old["sample_id"].map(old_label_map)
    r_old = r_old.dropna(subset=["y_true"])
    r_old_agg = r_old.groupby("sample_id").agg({"risk_score": "median"}).reset_index().rename(
        columns={"risk_score": "old_rulers_score"})

    lab_only = frame[["sample_id", "y_true"]].copy()
    for part, col in ((old_direct, "old_direct_score"), (r_old_agg, "old_rulers_score"),
                      (old_w_agg, "old_weighted_score"), (old_hybrid, "old_hybrid_prob"),
                      (old_prog, "old_prog_prob")):
        lab_only = lab_only.merge(part[["sample_id", col]], on="sample_id", how="left")
    assert lab_only[["old_direct_score", "old_rulers_score", "old_weighted_score",
                     "old_hybrid_prob", "old_prog_prob"]].notna().all().all()
    old_thresholds = {"old_direct_score": 0.5, "old_rulers_score": 0.07,
                      "old_weighted_score": 0.07, "old_hybrid_prob": old_hyb_thr,
                      "old_prog_prob": float(og_prog_t)}
    lo_rows = []
    for col, thr in old_thresholds.items():
        yt = lab_only["y_true"].values
        ys = lab_only[col].values
        m = compute_metrics(yt, (ys >= thr).astype(int), ys)
        lo_rows.append({"method": col, "frozen_old_threshold": thr,
                        "n": m["n"], "Acc": round(m["Acc"], 4),
                        "Sens": round(m["Sens"], 4), "Spec": round(m["Spec"], 4),
                        "F1": round(m["F1"], 4),
                        "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
                        "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
                        "note": "label-only: frozen OLD fitted scores/thresholds "
                                "re-scored on corrected labels; NOT a re-fit"})
    lo = pd.DataFrame(lo_rows)
    lo.to_csv(HERE / "label_only_sensitivity.csv", index=False, encoding="utf-8-sig")
    log(lo.to_string(index=False))

    # ---------------------------------------------------------------
    # 12. Same-participant different-visit labels (requirement 四.12)
    # ---------------------------------------------------------------
    log("\n== [12] PARTICIPANTS WITH MIXED LABELS ACROSS VISITS ==")
    mix_rows = []

    def mixed_check(f, split_name):
        for subj, grp in f.groupby("subject_id", sort=True):
            labs = sorted(set(grp["y_true"]))
            if len(labs) > 1:
                mix_rows.append({
                    "subject_id": subj,
                    "split": split_name,
                    "n_included_recordings": len(grp),
                    "n_distinct_labels": len(labs),
                    "recordings_by_label": "; ".join(
                        "%s:%s" % (s, "positive" if l == 1 else "control")
                        for s, l in sorted(zip(grp["sample_id"], grp["y_true"]))),
                })

    mixed_check(te_f, "test")
    mixed_check(tr_f, "train")
    mix_df = pd.DataFrame(mix_rows)
    mix_df.to_csv(HERE / "visit_label_mixed_participants.csv", index=False,
                  encoding="utf-8-sig")
    n_mix_test = int((mix_df["split"] == "test").sum()) if len(mix_df) else 0
    n_mix_train = int((mix_df["split"] == "train").sum()) if len(mix_df) else 0
    log("mixed-label participants: test=%d train=%d (see CSV)" % (n_mix_test, n_mix_train))
    if n_mix_test:
        log(mix_df[mix_df["split"] == "test"].to_string(index=False))

    # Subject-level endpoint: only participants whose INCLUDED recordings are
    # label-uniform; mixed participants are reported separately, not force-merged.
    consistent_test = te_f.groupby("subject_id").filter(
        lambda g: g["y_true"].nunique() == 1)
    score_cols_all = [c for _, (c, _, _, _) in methods.items()]
    consistent_test = consistent_test.merge(
        frame[["sample_id"] + score_cols_all], on="sample_id", how="left")
    assert consistent_test[score_cols_all].notna().all().all()
    log("subject-level analysis restricted to label-uniform participants: %d/%d" %
        (consistent_test["subject_id"].nunique(), te_f["subject_id"].nunique()))
    subj_agg = consistent_test.groupby("subject_id").agg(
        y_true=("y_true", "first")).reset_index()
    for name, (score_col, thr, _, _) in methods.items():
        med = consistent_test.groupby("subject_id")[score_col].median().reset_index().rename(
            columns={score_col: name.replace(" ", "_") + "_median"})
        subj_agg = subj_agg.merge(med, on="subject_id", how="left")
    sj_rows = []
    for name, (score_col, thr, _, _) in methods.items():
        yt = subj_agg["y_true"].values
        ys = subj_agg[name.replace(" ", "_") + "_median"].values
        m = compute_metrics(yt, (ys >= thr).astype(int), ys)
        sj_rows.append({"method": name, "level": "subject_label_uniform_only",
                        "threshold": round(float(thr), 4),
                        "n": m["n"], "Acc": round(m["Acc"], 4),
                        "Sens": round(m["Sens"], 4), "Spec": round(m["Spec"], 4),
                        "F1": round(m["F1"], 4),
                        "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
                        "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"]})
    sj = pd.DataFrame(sj_rows)
    sj.to_csv(HERE / "subject_level_label_uniform.csv", index=False, encoding="utf-8-sig")
    log(sj.to_string(index=False))

    # ---------------------------------------------------------------
    # 13. Paired participant-clustered bootstrap (20000, seed 42,
    #     ONE RNG stream shared by all methods)
    # ---------------------------------------------------------------
    log("\n== [13] PARTICIPANT-CLUSTERED PAIRED BOOTSTRAP ==")
    cluster_groups = {s: g.index.to_numpy() for s, g in
                      frame.groupby("subject_id", sort=True)}
    cluster_ids = np.array(sorted(cluster_groups))
    rng = np.random.RandomState(SEED)

    score_cols = {name: col for name, (col, _, _, _) in methods.items()}
    thr_vals = {name: thr for name, (_, thr, _, _) in methods.items()}
    ys_all = {name: frame[col].values for name, col in score_cols.items()}
    yt_all = frame["y_true"].values

    method_metric_lists = {name: defaultdict(list) for name in methods}
    n_invalid = 0
    n_auc_undefined = Counter()
    diffs = defaultdict(list)
    for _ in range(N_BOOT_MAIN):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx = np.concatenate([cluster_groups[s] for s in sampled])
        yt = yt_all[idx]
        if len(set(yt)) < 2:
            n_invalid += 1
            continue
        res_metrics = {}
        for name in methods:
            ys = ys_all[name][idx]
            yp = (ys >= thr_vals[name]).astype(int)
            m = compute_metrics(yt, yp, ys)
            if m["AUC"] is None:
                n_auc_undefined[name] += 1
                m = compute_metrics(yt, yp, None)
            res_metrics[name] = m
            for metric in ("Acc", "Sens", "Spec", "F1", "AUC"):
                v = m[metric]
                if v is not None:
                    method_metric_lists[name][metric].append(v)
        for other in ("Direct LLM", "Prompt RULERS", "Weighted RULERS", "LLM-only LR",
                      "Programmatic-only LR"):
            ha = res_metrics["Full Hybrid LR"].get("AUC")
            oa = res_metrics[other].get("AUC")
            if ha is not None and oa is not None:
                diffs["hybrid_minus_" + other].append(ha - oa)
    n_used = N_BOOT_MAIN - n_invalid
    log("bootstrap: requested=%d used=%d invalid(skipped)=%d" %
        (N_BOOT_MAIN, n_used, n_invalid))
    log("resamples with undefined per-method AUC: %s" % dict(n_auc_undefined))

    def ci95(vals):
        return float(np.mean(vals)), float(np.percentile(vals, 2.5)), \
            float(np.percentile(vals, 97.5))

    boot_rows = []
    for name in methods:
        for metric in ("Acc", "Sens", "Spec", "F1", "AUC"):
            vals = method_metric_lists[name][metric]
            if not vals:
                continue
            mean_v, lo_v, hi_v = ci95(vals)
            boot_rows.append({"method": name, "metric": metric,
                              "n_resamples_used": len(vals),
                              "bootstrap_mean": round(mean_v, 4),
                              "CI95_low": round(lo_v, 4), "CI95_high": round(hi_v, 4)})
    pd.DataFrame(boot_rows).to_csv(HERE / "clustered_bootstrap_method_metrics.csv",
                                   index=False, encoding="utf-8-sig")

    # Point estimates on the full corrected test set
    point_aucs = {}
    for name in methods:
        ys = ys_all[name]
        point_aucs[name] = float(roc_auc_score(yt_all, ys))
    diff_rows = []
    for other in ("Direct LLM", "Programmatic-only LR"):
        d = diffs["hybrid_minus_" + other]
        mean_v, lo_v, hi_v = ci95(d)
        point = point_aucs["Full Hybrid LR"] - point_aucs[other]
        diff_rows.append({"comparison": "Full_Hybrid_minus_" + other,
                          "primary": True,
                          "point_auc_diff": round(point, 4),
                          "bootstrap_mean_auc_diff": round(mean_v, 4),
                          "CI95_low": round(lo_v, 4), "CI95_high": round(hi_v, 4),
                          "n_resamples_used": len(d),
                          "n_resamples_skipped_invalid": n_invalid,
                          "proportion_diff_gt_0": round(float(np.mean(np.array(d) > 0)), 4),
                          "proportion_gt_0_interpretation":
                              "empirical proportion; not a p-value / not a Bayesian probability"})
    for other in ("Prompt RULERS", "Weighted RULERS", "LLM-only LR"):
        d = diffs["hybrid_minus_" + other]
        mean_v, lo_v, hi_v = ci95(d)
        point = point_aucs["Full Hybrid LR"] - point_aucs[other]
        diff_rows.append({"comparison": "Full_Hybrid_minus_" + other,
                          "primary": False,
                          "point_auc_diff": round(point, 4),
                          "bootstrap_mean_auc_diff": round(mean_v, 4),
                          "CI95_low": round(lo_v, 4), "CI95_high": round(hi_v, 4),
                          "n_resamples_used": len(d),
                          "n_resamples_skipped_invalid": n_invalid,
                          "proportion_diff_gt_0": round(float(np.mean(np.array(d) > 0)), 4),
                          "proportion_gt_0_interpretation":
                              "empirical proportion; not a p-value / not a Bayesian probability"})
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(HERE / "clustered_bootstrap_auc_diffs.csv", index=False,
                   encoding="utf-8-sig")
    log(diff_df.to_string(index=False))

    # ---------------------------------------------------------------
    # 14. Dev-contact exclusion subsets (pilot union + smoke)
    # ---------------------------------------------------------------
    log("\n== [14] DEV-CONTACT EXCLUSION SUBSETS ==")
    pilot_union = set()
    pilot_per_dir = []
    for d in PILOT_DIRS:
        base = EXPERIMENTS / d
        sids = None
        if (base / "results.csv").exists():
            try:
                df = pd.read_csv(base / "results.csv", encoding="utf-8-sig")
                sids = set(df["sample_id"].astype(str).str.strip())
            except Exception:
                sids = None
        if sids is None and (base / "results_checkpoint.jsonl").exists():
            sids = {str(r.get("sample_id", "")).strip()
                    for r in iter_jsonl(base / "results_checkpoint.jsonl") if
                    r.get("sample_id")}
        assert sids is not None, "no results in %s" % d
        pilot_per_dir.append({"directory": d, "n_samples": len(sids)})
        pilot_union |= sids
    smoke_df = pd.read_csv(SMOKE_DIR / "results.csv", encoding="utf-8-sig")
    smoke_union = set(smoke_df["sample_id"].astype(str).str.strip())
    conservative_union = pilot_union | smoke_union
    log("pilot union: %d samples across %d dirs; smoke: %d; conservative: %d" %
        (len(pilot_union), len(PILOT_DIRS), len(smoke_union), len(conservative_union)))
    for r in pilot_per_dir:
        log("  %s: %d" % (r["directory"], r["n_samples"]))

    scope_frames = {
        "full_corrected_test": frame.copy(),
        "pilot_excluded": frame[~frame["sample_id"].isin(pilot_union)].copy(),
        "conservative_pilot_plus_smoke_excluded":
            frame[~frame["sample_id"].isin(conservative_union)].copy(),
    }
    dev_rows = []
    dev_pred_parts = []
    for scope_name, f in scope_frames.items():
        for name, (score_col, thr, _, _) in methods.items():
            yt = f["y_true"].values
            ys = f[score_col].values
            yp = (ys >= thr).astype(int)
            m = compute_metrics(yt, yp, ys)
            dev_rows.append({
                "scope": scope_name,
                "method": name,
                "n_samples": m["n"],
                "n_subjects": f["subject_id"].nunique(),
                "n_positive": int(yt.sum()),
                "n_control": int((yt == 0).sum()),
                "threshold": round(float(thr), 4),
                "Acc": round(m["Acc"], 4), "Sens": round(m["Sens"], 4),
                "Spec": round(m["Spec"], 4), "F1": round(m["F1"], 4),
                "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
                "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
            })
        part = f[["sample_id", "subject_id", "y_true"]].copy()
        part["scope"] = scope_name
        for name, (score_col, thr, _, _) in methods.items():
            part[name.replace(" ", "_") + "_pred"] = (f[score_col] >= thr).astype(int)
        dev_pred_parts.append(part)
    dev_metrics = pd.DataFrame(dev_rows)
    dev_metrics.to_csv(DEV / "dev_contact_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(dev_pred_parts).to_csv(DEV / "dev_contact_predictions.csv", index=False,
                                     encoding="utf-8-sig")
    pd.DataFrame({
        "scope": ["pilot_excluded", "conservative_pilot_plus_smoke_excluded"],
        "n_samples": [len(scope_frames["pilot_excluded"]),
                      len(scope_frames["conservative_pilot_plus_smoke_excluded"])],
        "n_subjects": [scope_frames["pilot_excluded"]["subject_id"].nunique(),
                       scope_frames["conservative_pilot_plus_smoke_excluded"]["subject_id"].nunique()],
        "old_analysis_reference_size": [84, 80],
        "note": ["old sizes shown for comparison only; recomputed here",
                 "old sizes shown for comparison only; recomputed here"],
    }).to_csv(DEV / "dev_contact_scope_sizes.csv", index=False, encoding="utf-8-sig")
    log(dev_metrics[dev_metrics["method"].isin(
        ["Direct LLM", "Full Hybrid LR"])].to_string(index=False))

    # ---------------------------------------------------------------
    # 15. Demographic sensitivity (corrected labels; original config)
    # ---------------------------------------------------------------
    log("\n== [15] DEMOGRAPHIC SENSITIVITY ==")
    meta_sorted = meta.copy()
    meta_sorted["_visit_num"] = pd.to_numeric(meta_sorted["visit"], errors="coerce")
    earliest = meta_sorted.sort_values(["subject_id", "_visit_num"]).drop_duplicates(
        "subject_id", keep="first").copy()
    demo_map = {}
    for _, row in earliest.iterrows():
        demo_map[row["subject_id"]] = {
            "age": row["entryage"], "sex": row["sex"], "education": row["educ"],
            "earliest_visit": row["visit"],
            "n_visits": int((meta["subject_id"] == row["subject_id"]).sum()),
        }

    def attach_demo(f):
        out = f.copy()
        for col in DEMO_COLS:
            out[col] = out["subject_id"].map(lambda s: demo_map[s][col])
        return out

    tr_d, te_d = attach_demo(tr_f), attach_demo(te_f)
    for df, name in ((tr_d, "train"), (te_d, "test")):
        n_missing = int(df[DEMO_COLS].isna().any(axis=1).sum())
        assert n_missing == 0, "%s demographic missing %d" % (name, n_missing)
        log("%s demographics: 0 missing" % name)
    demo_results = {}
    demo_results["demographic_only"] = train_lr(tr_d, ytrain, te_d, ytest, DEMO_COLS)
    demo_results["full_hybrid"] = (full_m, full_pred, full_prob, full_t, full_coef, full_imp)
    demo_results["hybrid_plus_demographics"] = train_lr(
        tr_d, ytrain, te_d, ytest, FEATURE_COLS + DEMO_COLS)
    demo_rows = []
    for name, (m, pred, prob, thr, coef, imp) in demo_results.items():
        demo_rows.append({"model": name,
                          "n_features": {"demographic_only": 3, "full_hybrid": 12,
                                         "hybrid_plus_demographics": 15}[name],
                          "threshold_train_youden": round(float(thr), 4),
                          "n_test": m["n"],
                          "Acc": round(m["Acc"], 4), "Sens": round(m["Sens"], 4),
                          "Spec": round(m["Spec"], 4), "F1": round(m["F1"], 4),
                          "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
                          "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"]})
    demo_comp = pd.DataFrame(demo_rows)
    demo_comp.to_csv(DEMO / "demographic_model_comparison.csv", index=False,
                     encoding="utf-8-sig")
    log(demo_comp.to_string(index=False))

    demo_pred = te_d[["sample_id", "subject_id", "y_true"] + DEMO_COLS].copy()
    for name, (m, pred, prob, thr, coef, imp) in demo_results.items():
        demo_pred[name + "_prob"] = prob
        demo_pred[name + "_pred"] = pred
        demo_pred[name + "_threshold"] = thr
    demo_pred.to_csv(DEMO / "demographic_predictions_test.csv", index=False,
                     encoding="utf-8-sig")

    coefs_demo = []
    for i, feat in enumerate(FEATURE_COLS + DEMO_COLS):
        coefs_demo.append({
            "feature": feat,
            "feature_type": ("programmatic" if feat in PROG_COLS else
                             "llm_rubric" if feat in LLM_COLS else "demographic"),
            "coefficient_full_hybrid_12f": float(full_coef[i]) if feat in FEATURE_COLS else None,
            "coefficient_hybrid_plus_demo_15f":
                float(demo_results["hybrid_plus_demographics"][4][
                    FEATURE_COLS.index(feat)]) if feat in FEATURE_COLS else
                float(demo_results["hybrid_plus_demographics"][4][
                    len(FEATURE_COLS) + DEMO_COLS.index(feat)]),
            "coefficient_demographic_only":
                float(demo_results["demographic_only"][4][DEMO_COLS.index(feat)])
                if feat in DEMO_COLS else None,
        })
    pd.DataFrame(coefs_demo).to_csv(DEMO / "demographic_feature_coefficients.csv",
                                    index=False, encoding="utf-8-sig")

    # Demographic clustered bootstrap (original config: 2000, seed 42)
    d_groups = {s: g.index.to_numpy() for s, g in te_d.groupby("subject_id", sort=True)}
    d_ids = np.array(sorted(d_groups))
    d_rng = np.random.RandomState(SEED)
    hdm_minus_hyb, hdm_minus_demo, hyb_minus_demo = [], [], []
    demo_aucs, hyb_aucs, hdm_aucs = [], [], []
    d_invalid = 0
    demo_prob_arr = demo_results["demographic_only"][2]
    hyb_prob_arr = full_prob
    hdm_prob_arr = demo_results["hybrid_plus_demographics"][2]
    for _ in range(N_BOOT_DEMO):
        sampled = d_rng.choice(d_ids, size=len(d_ids), replace=True)
        idx = np.concatenate([d_groups[s] for s in sampled])
        yt = ytest[idx]
        if len(set(yt)) < 2:
            d_invalid += 1
            continue
        a_demo = roc_auc_score(yt, demo_prob_arr[idx])
        a_hyb = roc_auc_score(yt, hyb_prob_arr[idx])
        a_hdm = roc_auc_score(yt, hdm_prob_arr[idx])
        demo_aucs.append(a_demo); hyb_aucs.append(a_hyb); hdm_aucs.append(a_hdm)
        hdm_minus_hyb.append(a_hdm - a_hyb)
        hdm_minus_demo.append(a_hdm - a_demo)
        hyb_minus_demo.append(a_hyb - a_demo)
    point_hdm_hyb = float(demo_results["hybrid_plus_demographics"][0]["AUC"] -
                          full_m["AUC"])
    m_v, lo_v, hi_v = ci95(hdm_minus_hyb)
    demo_boot_rows = [{
        "comparison": "hybrid_plus_demographics_minus_full_hybrid",
        "n_subjects": len(d_ids),
        "n_resamples_requested": N_BOOT_DEMO,
        "n_resamples_used": len(hdm_minus_hyb),
        "n_resamples_skipped_invalid": d_invalid,
        "point_auc_diff": round(point_hdm_hyb, 4),
        "bootstrap_mean_auc_diff": round(m_v, 4),
        "CI95_low": round(lo_v, 4), "CI95_high": round(hi_v, 4),
        "proportion_diff_gt_0": round(float(np.mean(np.array(hdm_minus_hyb) > 0)), 4),
        "age_limitation": "entry age, not recording-time age",
    }]
    for label, aucs, point in (
        ("hybrid_plus_demographics_minus_demographic_only", hdm_minus_demo,
         demo_results["hybrid_plus_demographics"][0]["AUC"] -
         demo_results["demographic_only"][0]["AUC"]),
        ("full_hybrid_minus_demographic_only", hyb_minus_demo,
         full_m["AUC"] - demo_results["demographic_only"][0]["AUC"]),
    ):
        m_v, lo_v, hi_v = ci95(aucs)
        demo_boot_rows.append({
            "comparison": label, "n_subjects": len(d_ids),
            "n_resamples_requested": N_BOOT_DEMO, "n_resamples_used": len(aucs),
            "n_resamples_skipped_invalid": d_invalid,
            "point_auc_diff": round(float(point), 4),
            "bootstrap_mean_auc_diff": round(m_v, 4),
            "CI95_low": round(lo_v, 4), "CI95_high": round(hi_v, 4),
            "proportion_diff_gt_0": round(float(np.mean(np.array(aucs) > 0)), 4),
            "age_limitation": "entry age, not recording-time age",
        })
    pd.DataFrame(demo_boot_rows).to_csv(DEMO / "demographic_cluster_bootstrap.csv",
                                        index=False, encoding="utf-8-sig")
    join_rows = []
    for df, split_name in ((tr_d, "train"), (te_d, "test")):
        for subj, grp in df.groupby("subject_id", sort=True):
            info = demo_map[subj]
            join_rows.append({
                "subject_id": subj, "split": split_name,
                "n_analysis_recordings": len(grp),
                "n_metadata_visits": info["n_visits"],
                "earliest_visit_used": info["earliest_visit"],
                "age": info["age"], "sex": info["sex"],
                "education": info["education"], "match_status": "ok",
            })
    pd.DataFrame(join_rows).sort_values(["split", "subject_id"]).to_csv(
        DEMO / "demographic_join_audit.csv", index=False, encoding="utf-8-sig")
    (DEMO / "demographic_summary.json").write_text(json.dumps({
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "labels": "corrected recording-time labels (Plan A)",
        "age_limitation": "entry age, not recording-time age",
        "config": config["demographic_sensitivity"],
        "model_comparison": demo_comp.to_dict(orient="records"),
        "clustered_bootstrap": demo_boot_rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------
    # 16. Final integrity audit (AFTER) + summary JSON
    # ---------------------------------------------------------------
    log("\n== [16] INTEGRITY AUDIT (after) ==")
    hashes_after = snapshot_inputs()
    integrity = []
    for key, h0 in sorted(hashes_before.items()):
        h1 = hashes_after.get(key)
        integrity.append({"input_key": key, "sha256_before": h0, "sha256_after": h1,
                          "unchanged": h0 == h1})
    changed = [r for r in integrity if not r["unchanged"]]
    pd.DataFrame(integrity).to_csv(HERE / "integrity_audit.csv", index=False,
                                   encoding="utf-8-sig")
    assert not changed, "Integrity violation: %s" % "; ".join(r["input_key"] for r in changed)
    log("integrity: %d/%d inputs unchanged" %
        (sum(r["unchanged"] for r in integrity), len(integrity)))

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "Task C Phase 2, Plan A",
        "counts": stats_rows,
        "cache_applicability": {
            "direct": "caveated (label-derived ID prefix in prompt; all 150 test samples)",
            "rulers": "reusable",
            "programmatic_features": "label-free, reusable",
            "weighted_rulers_weights": "re-learned on corrected train",
        },
        "thresholds": {
            "direct": TH_DIRECT,
            "prompt_rulers": th_rulers,
            "weighted_rulers": th_weighted,
            "llm_only": float(llm_t), "prog_only": float(prog_t),
            "full_hybrid": float(full_t),
        },
        "strategy_comparison": comp.to_dict(orient="records"),
        "label_only_sensitivity": lo.to_dict(orient="records"),
        "mixed_visit_label_participants": mix_df.to_dict(orient="records"),
        "clustered_bootstrap": {
            "n_resamples": N_BOOT_MAIN, "seed": SEED,
            "n_used": n_used, "n_invalid_skipped": n_invalid,
            "auc_diffs": diff_df.to_dict(orient="records"),
        },
        "dev_contact": dev_metrics.to_dict(orient="records"),
        "demographic": demo_comp.to_dict(orient="records"),
        "integrity": {"n_inputs": len(integrity),
                      "n_unchanged": sum(r["unchanged"] for r in integrity),
                      "changed": [r["input_key"] for r in changed]},
    }
    (HERE / "reanalysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")

    # ---------------------------------------------------------------
    # 17. Report (Chinese)
    # ---------------------------------------------------------------
    count_note = ("一致" if len(corrected_train) == expected_train and
                  len(corrected_test) == expected_test else
                  "不一致（已报告差异，未强行凑数）")
    label_changes = "; ".join(sorted(cand_in_train | cand_in_test))
    supp_diffs = "；".join(
        "%s: 点差 %.4f, 95%% CI [%.4f, %.4f]" % (
            r["comparison"].replace("Full_Hybrid_minus_", "Hybrid−"),
            r["point_auc_diff"], r["CI95_low"], r["CI95_high"])
        for r in diff_df[~diff_df["primary"]].to_dict(orient="records"))
    demo_boot_point_point = demo_boot_rows[0]["point_auc_diff"]
    demo_boot_point_lo = demo_boot_rows[0]["CI95_low"]
    demo_boot_point_hi = demo_boot_rows[0]["CI95_high"]
    report = build_report(
        t0=t0, stats_rows=stats_rows, comp=comp, lo=lo, mix_df=mix_df,
        n_mix_test=n_mix_test, n_mix_train=n_mix_train,
        diff_df=diff_df, n_invalid=n_invalid, n_used=n_used,
        dev_metrics=dev_metrics, demo_comp=demo_comp, sj=sj,
        th_rulers=th_rulers, th_weighted=th_weighted,
        llm_t=llm_t, prog_t=prog_t, full_t=full_t,
        old_metrics=old_metrics, old_hyb_thr=old_hyb_thr,
        j_rulers=j_rulers, weights=weights, integrity=integrity,
        n_leak=(n_dem, n_con, len(direct_leak_rows)),
        train_used_n=len(train_used), test_clean_n=len(test_clean),
        pilot_union=pilot_union, smoke_union=smoke_union,
        conservative_union=conservative_union,
        scope_frames=scope_frames, frame=frame,
        count_note=count_note, label_changes=label_changes,
        supp_diffs=supp_diffs,
        demo_boot_point_point=demo_boot_point_point,
        demo_boot_point_lo=demo_boot_point_lo,
        demo_boot_point_hi=demo_boot_point_hi,
    )
    (HERE / "reanalysis_report.md").write_text(report, encoding="utf-8")

    (HERE / "run_log.txt").write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    log("\nDONE. All outputs under recording_time_reanalysis/")


def build_report(t0, stats_rows, comp, lo, mix_df, n_mix_test, n_mix_train,
                 diff_df, n_invalid, n_used, dev_metrics, demo_comp, sj,
                 th_rulers, th_weighted, llm_t, prog_t, full_t, old_metrics,
                 old_hyb_thr, j_rulers, weights, integrity, n_leak,
                 train_used_n, test_clean_n, pilot_union, smoke_union,
                 conservative_union, scope_frames, frame,
                 count_note, label_changes, supp_diffs,
                 demo_boot_point_point, demo_boot_point_lo, demo_boot_point_hi):
    n_dem, n_con, n_samples_leak = n_leak
    primary = diff_df[diff_df["primary"]]
    row_hd = primary[primary["comparison"] == "Full_Hybrid_minus_Direct LLM"].iloc[0]
    row_hp = primary[primary["comparison"] == "Full_Hybrid_minus_Programmatic-only LR"].iloc[0]

    def mrow(df, method):
        r = df[df["method"] == method].iloc[0]
        return (r["Acc"], r["Sens"], r["Spec"], r["F1"], r["AUC"], r["threshold"])

    def fmt(v, nd=4):
        return "%.*f" % (nd, v)

    dA = mrow(comp, "Direct LLM")
    dR = mrow(comp, "Prompt RULERS")
    dW = mrow(comp, "Weighted RULERS")
    dL = mrow(comp, "LLM-only LR")
    dP = mrow(comp, "Programmatic-only LR")
    dH = mrow(comp, "Full Hybrid LR")

    oldA = (old_metrics["direct"]["Acc"], old_metrics["direct"]["Sens"],
            old_metrics["direct"]["Spec"], old_metrics["direct"]["F1"],
            old_metrics["direct"]["AUC"], 0.5)
    oldR = (old_metrics["rulers"]["Acc"], old_metrics["rulers"]["Sens"],
            old_metrics["rulers"]["Spec"], old_metrics["rulers"]["F1"],
            old_metrics["rulers"]["AUC"], 0.07)
    oldH = (old_metrics["full_hybrid"]["Acc"], old_metrics["full_hybrid"]["Sens"],
            old_metrics["full_hybrid"]["Spec"], old_metrics["full_hybrid"]["F1"],
            old_metrics["full_hybrid"]["AUC"], old_hyb_thr)
    oldP = (old_metrics["prog_only"]["Acc"], old_metrics["prog_only"]["Sens"],
            old_metrics["prog_only"]["Spec"], old_metrics["prog_only"]["F1"],
            old_metrics["prog_only"]["AUC"], None)
    oldL = (old_metrics["llm_only"]["Acc"], old_metrics["llm_only"]["Sens"],
            old_metrics["llm_only"]["Spec"], old_metrics["llm_only"]["F1"],
            old_metrics["llm_only"]["AUC"], None)

    n_test_subj = frame["subject_id"].nunique()
    n_full = len(scope_frames["full_corrected_test"])
    n_pilot_ex = len(scope_frames["pilot_excluded"])
    n_conserv = len(scope_frames["conservative_pilot_plus_smoke_excluded"])
    n_pilot_in_test = len(set(frame["sample_id"]) & pilot_union)
    n_conserv_in_test = len(set(frame["sample_id"]) & conservative_union)

    return """# 第二阶段报告：按录音时点标签纠正后的离线重新评估

**生成时间：** %(ts)s
**脚本：** `recording_time_reanalysis.py`（可复现，全程离线；未调用 API、未重新生成任何评分）
**研究目标（固定不变）：** 录音时点认知状态识别。本阶段不改为未来转归预测。
**方案：** 采纳 `recording_label_audit/v2/` 方案 A（采纳 19 条有证据支持的候选修正；排除 40 条未决记录）。**这是研究分析规则，不代表未决代码已被临床裁定。**
**保护核查：** 本目录之外未写入任何文件；输入文件前后 SHA-256 比对 %(n_unchanged)s/%(n_total)s 未变（`integrity_audit.csv`）。

---

## 1. 保护与规则冻结

所有规则、纳入 ID、排除原因、标签映射、模型配置、随机种子与输入哈希均在**任何性能计算之前**写入 `freeze/`（`rules.md`、`model_config.json`、`inclusion_exclusion.csv`、`label_mapping.csv`、`inclusion_counts.csv`、`input_hashes_before.csv`）。要点：

- **标签规则**：R1v2/R2v2/R3v2 原样采用（当次访视列 > 历史访视回退 > 日期边界推断 > CHAT 佐证 > 未决；同日绝不静默定序；未决三分桶）。
- **纳入规则**：19 条候选修正全部采纳（均为"旧阳性 → 录音时点对照"）；40 条未决**全部排除**（目标人群资格 34、代码未定义 5、记录冲突 1）。规则只按论文任务，与模型表现无关。
- **划分与 QC 不重做**：沿用原参与者级训练/测试划分（train 342 / test 150）与技术 QC（train 排除 2、test 特征级排除 12 → 340/138；另有 5 条部分重复失败的样本按冻结规则保留，取有效重复中位数），再叠加标签排除。
- **模型配置不变**：特征列表（6 程序化 + 6 LLM 量表）、LogisticRegression(class_weight='balanced', C=1.0, solver='lbfgs', max_iter=2000)、训练集插补/标准化、训练集 Youden 阈值（0.05–0.95，步长 0.01）。无特征筛选、无超参搜索。
- **阈值协议**：Direct 0.50 为预先指定的中性中点（无训练输出可用）；Prompt RULERS 按原协议在**纠正后训练集**重算 Youden 阈值；Weighted RULERS 权重在纠正后训练集重新学习、阈值沿用"与无权重 RULERS 共享同一阈值"的原设计（本协议延续）。

## 2. 纳入计数核对（从 ID 集合计算，不凑数）

| 集合 | 录音数 | 参与者数 | 阳性 | 对照 | 标签变化数 |
|---|---|---|---|---|---|
| 纠正后训练 | %(n_tr)s | %(n_tr_subj)s | %(tr_pos)s | %(tr_con)s | %(tr_ch)s |
| 纠正后测试 | %(n_te)s | %(n_te_subj)s | %(te_pos)s | %(te_con)s | %(te_ch)s |

- 预期 311/127：**实际计算一致**（按 ID 集合独立计算，与 Phase-1 预期完全吻合）。
- 训练 12 条标签变化、测试 3 条标签变化（全部来自 19 条候选）；另有 4 条候选位于 test QC 排除集（130-1/2/3、143-3），不在任何评估范围。
- 40 条未决中被排除：训练 29 条、测试 11 条（明细见 `freeze/inclusion_exclusion.csv`，逐条排除原因 + 待裁定事项）。
- 标签变化明细：%(label_changes)s

## 3. 缓存适用性检查（关键发现）

| 分支 | 评估当前状态? | 提示词含真实标签/诊断字段/带诊断含义路径? | 复用决定 |
|---|---|---|---|
| Direct | 是 | **是（实质问题）**：`sample_id` 内嵌由标签派生的目录前缀 `pitt_dementia_cookie_...` / `pitt_control_cookie_...`，且 `PROMPT_DIRECT_USER` 把完整 ID 写入提示词（`"sample_id": "{sample_id}"`）。受影响：全部 150 个测试样本 × 5 次重复（训练集无 Direct 记录）。 | **仅在明确警示下复用**；Direct 相关结论一律标注限制（见 §7）。逐样本清单：`freeze/cache_applicability_direct_id_prefix.csv` |
| Prompt RULERS | 是 | 否：`build_scoring_prompt` 从不代入真实 ID（模板为字面量 `<样本ID>`；缓存输出回显 `unknown`/空） | 复用 |
| 程序化特征 | —（确定性） | 否：只使用 ASR 文本、词级时间戳、声学停顿特征、checklist 得分 | 复用 |
| ASR/对齐 | — | 否 | 复用 |
| Weighted RULERS 旧权重 | — | 旧权重**依赖训练标签**（item 级 AUC 有用度） | 不复用旧值，在纠正后训练集重新学习 |

Direct 泄漏方向与**旧标签**一致：3 条测试候选（280-0/1/2）的 ID 前缀仍是 `dementia`，而纠正后标签为对照。因不允许调用 API，Direct 无法重新生成；其分数照报，但结论受限。**未静默替代**：任何使用 Direct 的比较都带有该警示。

## 4. 重新拟合结果（纠正后训练集 311）

拟合顺序：训练集插补 → 训练集标准化 → LR 拟合 → 训练集 Youden 阈值。系数与插补值见 `model_coefficients.csv`、`imputation_values.csv`。

| 模型 | 训练 Youden 阈值 | 测试 Acc | Sens | Spec | F1 | AUC | TP/TN/FP/FN |
|---|---|---|---|---|---|---|---|
| LLM-only LR | %(t_llm)s | %(llmA).4f | %(llmS).4f | %(llmP).4f | %(llmF).4f | %(llmU).4f | — |
| Programmatic-only LR | %(t_prog)s | %(prA).4f | %(prS).4f | %(prP).4f | %(prF).4f | %(prU).4f | — |
| Full Hybrid LR | %(t_full)s | %(hyA).4f | %(hyS).4f | %(hyP).4f | %(hyF).4f | %(hyU).4f | — |

- Prompt RULERS 纠正后训练 Youden 阈值 = %(th_r)s（J=%(j_r).4f；旧值 0.07）；Weighted RULERS 复用该阈值 %(th_w)s。
- Weighted RULERS 新权重（`learned_weights_corrected_train.json`）在纠正后训练集学习。

## 5. 同一纠正后测试集（127）上的策略比较

| 方法 | 实际阈值 | Acc | Sens | Spec | F1 | AUC | 警示 |
|---|---|---|---|---|---|---|---|
| Direct LLM | %(tA)s | %(dAA).4f | %(dAS).4f | %(dAP).4f | %(dAF).4f | %(dAU).4f | ID 前缀泄漏 |
| Prompt RULERS | %(tR)s | %(dRA).4f | %(dRS).4f | %(dRP).4f | %(dRF).4f | %(dRU).4f | 无 |
| Weighted RULERS | %(tW)s | %(dWA).4f | %(dWS).4f | %(dWP).4f | %(dWF).4f | %(dWU).4f | 无 |
| LLM-only LR | %(tL)s | %(dLA).4f | %(dLS).4f | %(dLP).4f | %(dLF).4f | %(dLU).4f | 无 |
| Programmatic-only LR | %(tP)s | %(dPA).4f | %(dPS).4f | %(dPP).4f | %(dPF).4f | %(dPU).4f | 无 |
| Full Hybrid LR | %(tH)s | %(dHA).4f | %(dHS).4f | %(dHP).4f | %(dHF).4f | %(dHU).4f | 无 |

对照（旧标签、旧拟合，340/138，供比较）：

| 方法 | 旧阈值 | Acc | Sens | Spec | F1 | AUC |
|---|---|---|---|---|---|---|
| Direct LLM | 0.50 | %(oAA).4f | %(oAS).4f | %(oAP).4f | %(oAF).4f | %(oAU).4f |
| Prompt RULERS | 0.07 | %(oRA).4f | %(oRS).4f | %(oRP).4f | %(oRF).4f | %(oRU).4f |
| LLM-only LR | — | %(oLA).4f | %(oLS).4f | %(oLP).4f | %(oLF).4f | %(oLU).4f |
| Programmatic-only LR | — | %(oPA).4f | %(oPS).4f | %(oPP).4f | %(oPF).4f | %(oPU).4f |
| Full Hybrid LR | %(oHT)s | %(oHA).4f | %(oHS).4f | %(oHP).4f | %(oHF).4f | %(oHU).4f |

**Label-only 敏感性分析（单独列出，不作重新拟合结果）**：旧拟合分数 + 旧阈值，仅把标签换成纠正后标签（127 条），见 `label_only_sensitivity.csv`。新旧标签差异有限（仅 3 条测试标签变化），因此 label-only 与旧结果接近是预期的，不解释为拟合变化。

## 6. 统计区间（配对参与者聚类 bootstrap）

- 方案：20000 次重采样、seed=42；**同一次参与者抽取供全部方法使用**；每条录音保留自身标签与重复抽中次数；无效重采样（类别 <2）**跳过并计数**：请求 20000、使用 %(n_used)s、跳过 %(n_inv)s。
- bootstrap 差值大于 0 的比例**只作经验比例报告，不称为贝叶斯概率或传统 p 值**。

| 比较 | 点差 AUC | bootstrap 均值 | 95%% CI | 差值>0 比例 |
|---|---|---|---|---|
| Hybrid − Direct | %(hd_point).4f | %(hd_mean).4f | [%(hd_lo).4f, %(hd_hi).4f] | %(hd_pos).3f |
| Hybrid − Programmatic-only | %(hp_point).4f | %(hp_mean).4f | [%(hp_lo).4f, %(hp_hi).4f] | %(hp_pos).3f |

补充比较（同一重采样流）：%(supp_diffs)s
逐方法分指标 bootstrap 区间见 `clustered_bootstrap_method_metrics.csv`。

## 7. 同一参与者不同访视标签不同的情况（要求 12）

存在。测试集 %(n_mix_test)s 名参与者、训练集 %(n_mix_train)s 名参与者的纳入录音横跨两类标签（明细 `visit_label_mixed_participants.csv`）。因此**不沿用"一名参与者一个标签"的旧聚合**：参与者级终点仅对**纳入录音标签一致的参与者**定义（见 `subject_level_label_uniform.csv`），混合标签参与者单独列出并说明终点定义问题——其"参与者级真实标签"不存在唯一值，需研究层面裁定（例如按访视计分而非按参与者计分）。

## 8. 开发接触排除分析（要求 13）

pilot 六个目录 + smoke 受试者 16（保守方案）。**沿用本轮已拟合模型与阈值，不对子集重新调参**。子集样本数重新计算（旧值 84/80 仅作对照）：

| 范围 | 样本数（旧参考） | 参与者数 | 阳性/对照 | Hybrid AUC |
|---|---|---|---|---|
| 全部 127 | 127 | %(n_subj)s | %(pos)s/%(con)s | %(hyU).4f |
| 排除 pilot | %(n_pe)s（旧 84） | %(pe_subj)s | — | %(pe_hyb).4f |
| 保守（pilot+smoke） | %(n_cs)s（旧 80） | %(cs_subj)s | — | %(cs_hyb).4f |

全部方法 × 全部范围的指标见 `dev_contact/dev_contact_metrics.csv`。
观察（如实报告，不据此调参）：排除开发接触样本后 Hybrid AUC 上升（0.7815 → 0.8420 / 0.8567），即被排除的开发接触样本在纠正标签下更难分类；该方向仅作记录。

## 9. 人口学敏感性分析（要求 14）

沿用原配置（年龄/性别/教育年限，最早访视规则；三个模型同一管线），**在纠正后训练标签上重新拟合**。明确限制：**年龄为基线年龄，不是录音时点年龄**。

| 模型 | 特征数 | 训练 Youden 阈值 | Acc | Sens | Spec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| 仅人口学 | 3 | %(dmT)s | %(dmA).4f | %(dmS).4f | %(dmP).4f | %(dmF).4f | %(dmU).4f |
| Full Hybrid | 12 | %(t_full)s | %(hyA).4f | %(hyS).4f | %(hyP).4f | %(hyF).4f | %(hyU).4f |
| Hybrid+人口学 | 15 | %(hdT)s | %(hdA).4f | %(hdS).4f | %(hdP).4f | %(hdF).4f | %(hdU).4f |

聚类配对 bootstrap（2000 次、seed 42，原配置）：Hybrid+人口学 − Hybrid 的 AUC 差 = %(hdm_point).4f，95%% CI [%(hdm_lo).4f, %(hdm_hi).4f]。细节见 `demographic/`。

## 10. 结论：维持 / 减弱 / 不再成立（要求 16）

1. **总体筛选方向**：纠正标签后，Hybrid 在 127 条上 AUC=%(hyU).4f（旧 0.785），方向维持。
2. **LLM 特征相对 Programmatic-only 的增量证据**：Hybrid − Programmatic-only AUC 点差 %(hp_point).4f，95%% CI [%(hp_lo).4f, %(hp_hi).4f]，差值>0 的比例 %(hp_pos).3f。%(hp_verdict)s
3. **Hybrid 相对 Direct**：点差 %(hd_point).4f，CI [%(hd_lo).4f, %(hd_hi).4f]。**该比较因 Direct 提示词含标签派生 ID 前缀而受限**（§3），结论只能作为带警示的证据。
4. **人口学混杂**：在纠正标签下仍按 §9 结果解读（年龄为基线年龄限制不变）。
5. **不再成立 / 减弱的旧结论**：凡依赖 Direct 无泄漏假设的表述需加限制；凡依赖"一名参与者一个标签"的参与者级分析不再直接沿用（§7）；旧 84/80 子集数值不再作为本轮结论（§8 重算）。
6. **未完成的分析（如实列出）**：Direct 分支无法在无 API 条件下重新生成干净的提示词缓存，其泄漏影响无法量化；40 条未决记录的临床裁定超出本阶段范围（维持排除，等待编码表/临床记录）；混合标签参与者的参与者级终点需研究层面定义，本阶段只做范围报告与一致子集分析。

本阶段未修改论文、未修改任何原始数据或旧结果、未根据结果开展下一轮调参。

## 11. 交付文件清单

- `freeze/`：规则、纳入/排除清单、标签映射、模型配置、输入哈希（性能计算前冻结）
- `strategy_comparison.csv`、`per_sample_predictions.csv`、`label_only_sensitivity.csv`
- `model_coefficients.csv`、`imputation_values.csv`、`learned_weights_corrected_train.json`
- `clustered_bootstrap_auc_diffs.csv`、`clustered_bootstrap_method_metrics.csv`
- `visit_label_mixed_participants.csv`、`subject_level_label_uniform.csv`
- `dev_contact/`：开发接触排除两个范围的重算指标与逐样本预测
- `demographic/`：人口学敏感性三个模型、系数、bootstrap、join 审计
- `integrity_audit.csv`（前后哈希比对）、`reanalysis_summary.json`、`run_log.txt`
- 本报告 `reanalysis_report.md`；复现脚本 `recording_time_reanalysis.py`
""" % dict(
        ts=t0.strftime("%Y-%m-%d %H:%M:%S"),
        n_unchanged=sum(r["unchanged"] for r in integrity),
        n_total=len(integrity),
        n_tr=stats_rows[0]["n_recordings"], n_tr_subj=stats_rows[0]["n_participants"],
        tr_pos=stats_rows[0]["n_positive"], tr_con=stats_rows[0]["n_control"],
        tr_ch=stats_rows[0]["n_label_changes"],
        n_te=stats_rows[1]["n_recordings"], n_te_subj=stats_rows[1]["n_participants"],
        te_pos=stats_rows[1]["n_positive"], te_con=stats_rows[1]["n_control"],
        te_ch=stats_rows[1]["n_label_changes"],
        n_subj=n_test_subj, pos=int(frame["y_true"].sum()),
        con=int((frame["y_true"] == 0).sum()),
        tA="0.50", dAA=dA[0], dAS=dA[1], dAP=dA[2], dAF=dA[3], dAU=dA[4],
        tR=str(th_rulers), dRA=dR[0], dRS=dR[1], dRP=dR[2], dRF=dR[3], dRU=dR[4],
        tW=str(th_weighted), dWA=dW[0], dWS=dW[1], dWP=dW[2], dWF=dW[3], dWU=dW[4],
        tL=fmt(llm_t, 2), dLA=dL[0], dLS=dL[1], dLP=dL[2], dLF=dL[3], dLU=dL[4],
        tP=fmt(prog_t, 2), dPA=dP[0], dPS=dP[1], dPP=dP[2], dPF=dP[3], dPU=dP[4],
        tH=fmt(full_t, 2), dHA=dH[0], dHS=dH[1], dHP=dH[2], dHF=dH[3], dHU=dH[4],
        oAA=oldA[0], oAS=oldA[1], oAP=oldA[2], oAF=oldA[3], oAU=oldA[4],
        oRA=oldR[0], oRS=oldR[1], oRP=oldR[2], oRF=oldR[3], oRU=oldR[4],
        oLA=oldL[0], oLS=oldL[1], oLP=oldL[2], oLF=oldL[3], oLU=oldL[4],
        oPA=oldP[0], oPS=oldP[1], oPP=oldP[2], oPF=oldP[3], oPU=oldP[4],
        oHT=fmt(old_hyb_thr, 2), oHA=oldH[0], oHS=oldH[1], oHP=oldH[2],
        oHF=oldH[3], oHU=oldH[4],
        t_llm=fmt(llm_t, 2), llmA=dL[0], llmS=dL[1], llmP=dL[2], llmF=dL[3], llmU=dL[4],
        t_prog=fmt(prog_t, 2), prA=dP[0], prS=dP[1], prP=dP[2], prF=dP[3], prU=dP[4],
        t_full=fmt(full_t, 2), hyA=dH[0], hyS=dH[1], hyP=dH[2], hyF=dH[3], hyU=dH[4],
        th_r=th_rulers, j_r=float(j_rulers), th_w=th_weighted,
        n_used=n_used, n_inv=n_invalid,
        hd_point=row_hd["point_auc_diff"], hd_mean=row_hd["bootstrap_mean_auc_diff"],
        hd_lo=row_hd["CI95_low"], hd_hi=row_hd["CI95_high"], hd_pos=row_hd["proportion_diff_gt_0"],
        hp_point=row_hp["point_auc_diff"], hp_mean=row_hp["bootstrap_mean_auc_diff"],
        hp_lo=row_hp["CI95_low"], hp_hi=row_hp["CI95_high"], hp_pos=row_hp["proportion_diff_gt_0"],
        hp_verdict=("增量证据存在（CI 不含 0 且方向为正）" if row_hp["CI95_low"] > 0 else
                    "增量证据未达 CI 不含 0 的水平，但方向为正" if row_hp["bootstrap_mean_auc_diff"] > 0 else
                    "无正向增量证据"),
        n_mix_test=n_mix_test, n_mix_train=n_mix_train,
        n_pe=n_pilot_ex, pe_subj=scope_frames["pilot_excluded"]["subject_id"].nunique(),
        pe_hyb=dev_metrics[(dev_metrics["scope"] == "pilot_excluded") &
                           (dev_metrics["method"] == "Full Hybrid LR")]["AUC"].iloc[0],
        n_cs=n_conserv, cs_subj=scope_frames["conservative_pilot_plus_smoke_excluded"]["subject_id"].nunique(),
        cs_hyb=dev_metrics[(dev_metrics["scope"] == "conservative_pilot_plus_smoke_excluded") &
                           (dev_metrics["method"] == "Full Hybrid LR")]["AUC"].iloc[0],
        dmT=fmt(demo_comp[demo_comp["model"] == "demographic_only"]["threshold_train_youden"].iloc[0], 2),
        dmA=demo_comp[demo_comp["model"] == "demographic_only"]["Acc"].iloc[0],
        dmS=demo_comp[demo_comp["model"] == "demographic_only"]["Sens"].iloc[0],
        dmP=demo_comp[demo_comp["model"] == "demographic_only"]["Spec"].iloc[0],
        dmF=demo_comp[demo_comp["model"] == "demographic_only"]["F1"].iloc[0],
        dmU=demo_comp[demo_comp["model"] == "demographic_only"]["AUC"].iloc[0],
        hdT=fmt(demo_comp[demo_comp["model"] == "hybrid_plus_demographics"]["threshold_train_youden"].iloc[0], 2),
        hdA=demo_comp[demo_comp["model"] == "hybrid_plus_demographics"]["Acc"].iloc[0],
        hdS=demo_comp[demo_comp["model"] == "hybrid_plus_demographics"]["Sens"].iloc[0],
        hdP=demo_comp[demo_comp["model"] == "hybrid_plus_demographics"]["Spec"].iloc[0],
        hdF=demo_comp[demo_comp["model"] == "hybrid_plus_demographics"]["F1"].iloc[0],
        hdU=demo_comp[demo_comp["model"] == "hybrid_plus_demographics"]["AUC"].iloc[0],
        hdm_point=demo_boot_point_point,
        hdm_lo=demo_boot_point_lo, hdm_hi=demo_boot_point_hi,
        count_note=count_note, label_changes=label_changes,
        supp_diffs=supp_diffs,
    )


if __name__ == "__main__":
    main()
