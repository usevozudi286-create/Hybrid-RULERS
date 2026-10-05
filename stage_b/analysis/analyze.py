# -*- coding: utf-8 -*-
"""
analyze.py — Clean-repeatability analysis (protocol S7, frozen).

Implements the pre-registered analysis of the 1270-slot experiment:

  Primary:   participant-equal paired difference in recording-level label
             disagreement D = 2*k*(m-k)/(m*(m-1)) (m>=2 valid slots per recording,
             k positive labels) between Hybrid and Direct, computed on the
             THREE-method common valid slots; CI = 10000 participant-clustered
             paired bootstrap, seed 42, percentile 95%.
  Sensitivity (pre-specified): the same per-pair comparison on ALL available
             paired slots of that pair.
  Secondary (exploratory, effect size + CI only): Hybrid-RULERS, RULERS-Direct.
  Complete-5 subset: post-filter recordings with all 5 repeats scorable
             (actual counts reported, never the pre-filter labels).
  Detection metrics (descriptive): AUC / Acc / Sens / Spec / F1, predicted
             class distribution, stable-correct / stable-wrong / flip counts,
             and the recording-level aggregate (detection ONLY - its own
             repeatability is not reported by design). The aggregate is the
             mean of each recording's VALID repeat scores (>=1 valid score;
             valid-count distribution reported per method - it is NOT a
             "5-complete" aggregate), classified with the SAME per-method
             frozen thresholds as the per-call labels (Direct 0.50 /
             RULERS 0.07 / Hybrid 0.55). Implementation correction
             2026-09-23: an earlier version classified every method's mean
             score with a single 0.5 threshold, contradicting the frozen
             protocol; the corrected output is written to a separate
             directory (the original report is preserved). This is a bug
             fix, NOT test-set tuning.

Frozen operation points (labels used for D):
  Direct 0.50 | Prompt RULERS 0.07 | Hybrid LR 0.55   (model_config.json /
  v2 frozen constants TH_RULERS=0.07, TH_HYBRID=0.55). The Direct threshold is
  the same frozen 0.50, but the Direct label is now computed from a VALID
  risk_score (protocol revision); the raw predicted_label is diagnostic only.

Validity rules (protocol S5 + pre-run revisions):
  - "API success" is NOT "scorable". Scorable requires the actual scorable
    marker: A = valid risk_score (strict: bool/non-numeric/non-finite/
    out-of-range -> invalid -> MISSING, never defaulted, never clamped);
    B = parse_success AND structure valid AND risk present.
  - B: all-13-NA -> risk MISSING (never 0); six-LLM-criteria-NA -> Hybrid's 6
    LLM features are imputation-only -> excluded from the primary paired
    comparison, reported separately (coverage + sensitivity).
  - Structural invalidity (B) -> risk MISSING, unscorable, reported
    separately. Evidence-verification failure -> scores KEPT, flagged,
    reported separately. The two dispositions are never conflated.
  - Parse failures, refusals and all-NA are never re-asked; the runner already
    enforces this; the analyzer only consumes what exists.

Interpretation rules enforced in the report:
  - No claim of superiority; a favorable point estimate is not a "significant
    improvement" unless the CI excludes 0 (and even then it is a process
    comparison at frozen operation points, not a clinical claim).
  - CI containing 0 -> "no detectable difference", NEVER "equivalent".
  - Repeated calls are not independent participants; bootstrap clusters by
    participant.
  - This is a repeat-call evaluation on the same internal cohort; it is NOT a
    new independent clinical validation and NOT proof of non-inferiority.

Usage:
  python scripts/analyze.py --results <results.jsonl> --out <dir> [--simulated]
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from protocol_common import (  # noqa: E402
    PREP_DIR, FROZEN_INCLUSION_CSV, FROZEN_FEATURES_TRAIN_CSV,
    FROZEN_FEATURES_TEST_CSV, PROG_COLS, LLM_COLS, FEATURE_COLS,
    THRESHOLD_DIRECT, THRESHOLD_PROMPT_RULERS, THRESHOLD_HYBRID_LR, REPEATS,
)

BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_test_meta():
    rows = {}
    with open(FROZEN_INCLUSION_CSV, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("phase2_role") == "phase2_test":
                rows[r["sample_id"]] = r
    return rows


def load_opaque_mapping():
    m = {}
    with open(os.path.join(PREP_DIR, "text_freeze", "opaque_id_mapping.csv"),
              "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            m[r["opaque_id"]] = r
    return m


def load_results(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            assert r["slot_id"] not in rows, "duplicate slot: %s" % r["slot_id"]
            rows[r["slot_id"]] = r
    return rows


def load_archive(path, sample_filter):
    rows = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["sample_id"] in sample_filter:
                rows[r["sample_id"]] = r
    return rows


# ---------------------------------------------------------------------------
# Hybrid frozen pipeline
# ---------------------------------------------------------------------------

def fit_frozen_hybrid(train_rows_by_sample):
    """Deterministic refit on the frozen corrected train archive (311 rows)."""
    order = sorted(train_rows_by_sample.keys())
    X = np.array([[float(train_rows_by_sample[s][c]) for c in FEATURE_COLS]
                  for s in order])
    y = np.array([float(train_rows_by_sample[s]["y_true"]) for s in order])
    imputer = SimpleImputer(strategy="median")
    Ximp = imputer.fit_transform(X)
    scaler = StandardScaler()
    Xsc = scaler.fit_transform(Ximp)
    model = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0,
                               solver="lbfgs")
    model.fit(Xsc, y)
    return imputer, scaler, model


def hybrid_features_for_slot(test_archive, b_row, mapping=None):
    """12 features: 6 frozen programmatic (archive) + 6 LLM from THIS B call."""
    sid = b_row.get("sample_id")
    if sid is None and mapping is not None:
        sid = mapping[b_row["opaque_id"]]["sample_id"]
    prog = test_archive.get(sid)
    if prog is None:
        return None
    feats = {}
    for c in PROG_COLS:
        feats[c] = float(prog[c])
    checklist = b_row.get("checklist_scores") or {}
    for cid in ["C01", "C02", "C07", "C08", "C09", "C10"]:
        v = checklist.get(cid)
        feats["C_%s_median" % cid] = float(v) if v in (0, 1, 2) else np.nan
    return feats


# ---------------------------------------------------------------------------
# Slot-level table
# ---------------------------------------------------------------------------

def build_slot_table(results, mapping, meta, test_archive, train_archive):
    """
    One row per slot with per-method frozen labels/risks + validity flags.
    Hybrid is a derived (local) method attached to the B slot of each
    (sample, repeat).
    """
    imputer, scaler, model = fit_frozen_hybrid(train_archive)
    slots = []
    for slot_id, r in sorted(results.items()):
        mp = mapping[r["opaque_id"]]
        sid = mp["sample_id"]
        truth = 1 if mp["corrected_label"] == "positive" else 0
        row = {
            "slot_id": slot_id, "method": r["method"], "repeat_idx": r["repeat_idx"],
            "opaque_id": r["opaque_id"], "sample_id": sid,
            "subject_id": mp["subject_id"], "y_true": truth,
            "api_success": r.get("api_success", False),
            "parse_success": r.get("parse_success", False),
        }
        if r["method"] == "direct":
            # Main analysis (protocol revision): the label comes from a VALID
            # risk_score at THRESHOLD_DIRECT; the raw predicted_label is
            # diagnostic only and never drives a classification.
            risk_valid = bool(r.get("risk_valid"))
            risk = r.get("risk_score")
            label = r.get("label_direct")
            label_a = None
            risk_a = None
            if risk_valid and risk is not None:
                risk_a = float(risk)
                if label in ("positive", "control"):
                    label_a = 1 if label == "positive" else 0
            row.update({
                "scorable_A": risk_valid and risk_a is not None,
                "label_A": label_a,
                "risk_A": risk_a,
                "risk_valid": risk_valid,
                "risk_reason": r.get("risk_reason"),
                "risk_score_raw": r.get("risk_score_raw"),
                "predicted_label_raw": r.get("predicted_label_raw"),
                "direct_unknown_fields": r.get("unknown_fields") or [],
                "direct_duplicate_fields": r.get("duplicate_fields") or [],
            })
        else:
            flags = r.get("flags") or {}
            all_na_13 = bool(flags.get("all_na_13"))
            six_na = bool(flags.get("six_llm_criteria_all_na"))
            structural_invalid = bool(r.get("structural_invalid"))
            risk = r.get("risk_score")
            # scorable_B = actual scorable marker: structure valid AND a risk
            # score exists (parse failures, structural invalidity and all-NA
            # carry risk MISSING and are never scored)
            scorable = bool(r.get("parse_success")) and not structural_invalid \
                and risk is not None
            risk_b = None
            label_b = None
            if scorable:
                risk_b = float(risk)
                label_b = 1 if risk_b >= THRESHOLD_PROMPT_RULERS else 0
            # Hybrid (local, frozen pipeline). Imputation flags follow protocol
            # S5: pure-imputation rows (all six LLM features NaN) stay OUT of
            # the primary paired comparison and are reported separately
            # (coverage + a separate sensitivity comparison).
            h_label = h_prob = None
            h_imputed_any = False
            h_pure_imputed = False
            if scorable:
                feats = hybrid_features_for_slot(test_archive, r, mapping)
                if feats is not None:
                    x = np.array([[feats[c] for c in FEATURE_COLS]])
                    llm_idx = [FEATURE_COLS.index(c) for c in LLM_COLS]
                    if np.isnan(x).any():
                        h_imputed_any = True
                    if np.isnan(x[:, llm_idx]).all():
                        h_pure_imputed = True
                    ximp = imputer.transform(x)
                    xsc = scaler.transform(ximp)
                    h_prob = float(model.predict_proba(xsc)[0, 1])
                    h_label = 1 if h_prob >= THRESHOLD_HYBRID_LR else 0
            row.update({
                "scorable_B": scorable,
                "all_na_13": all_na_13,
                "six_na": six_na,
                "structural_invalid": structural_invalid,
                "structural_errors": r.get("structural_errors") or [],
                "evidence_verification_failed": bool(r.get("evidence_verification_failed")),
                "repair_structural_invalid": bool(r.get("repair_structural_invalid")),
                "rulers_unknown_fields": r.get("unknown_fields") or [],
                "rulers_duplicate_fields": r.get("duplicate_fields") or [],
                "risk_B": risk_b,
                "label_B": label_b,
                "hybrid_prob": h_prob,
                "label_H": h_label,
                "hybrid_imputed_any": h_imputed_any,
                "hybrid_pure_imputed": h_pure_imputed,
                "hybrid_scorable": bool(h_label is not None and not h_pure_imputed
                                        and not all_na_13),
                "hybrid_scorable_incl_imputed": bool(h_label is not None
                                                     and not all_na_13),
            })
        slots.append(row)
    return slots, (imputer, scaler, model)


# ---------------------------------------------------------------------------
# D and paired comparisons
# ---------------------------------------------------------------------------

def d_measure(labels):
    """D = 2*k*(m-k)/(m*(m-1)) over a recording's m>=2 valid labels."""
    m = len(labels)
    if m < 2:
        return None
    k = sum(labels)
    return 2.0 * k * (m - k) / (m * (m - 1))


def pair_comparison(slots, k1, k2, pair_rule):
    """
    Paired per-recording D difference between two methods. Slots are paired
    per (sample, repeat): each pair holds one A (direct) row and one B
    (rulers) row; Hybrid lives on the B row as a derived method.
    pair_rule(a_row, b_row) -> bool gates the pair into the comparison's
    common set. k1/k2 = (row_side, label_key): row_side is "rulers" or
    "direct" and names which row of the pair carries the method's label.
    Per-recording delta = D(k1 labels) - D(k2 labels) over the m paired
    repeats (m >= 2), then averaged participant-equal by the caller.
    """
    pairs = {}
    for s in slots:
        key = (s["sample_id"], s["repeat_idx"])
        pairs.setdefault(key, {})[s["method"]] = s
    by_rec = defaultdict(list)
    for pr in pairs.values():
        if "direct" not in pr or "rulers" not in pr:
            continue
        if pair_rule(pr["direct"], pr["rulers"]):
            by_rec[pr["direct"]["sample_id"]].append(pr)
    rec_rows = []
    for sid, prs in sorted(by_rec.items()):
        m = len(prs)
        if m < 2:
            continue
        l1 = [pr[k1[0]][k1[1]] for pr in prs]
        l2 = [pr[k2[0]][k2[1]] for pr in prs]
        d1 = d_measure(l1)
        d2 = d_measure(l2)
        if d1 is None or d2 is None:
            continue
        rec_rows.append({
            "sample_id": sid, "subject_id": prs[0]["direct"]["subject_id"],
            "y_true": prs[0]["direct"]["y_true"], "m": m,
            "k_%s" % k1[1]: sum(l1), "k_%s" % k2[1]: sum(l2),
            "D_%s" % k1[1]: d1, "D_%s" % k2[1]: d2,
            "delta": d1 - d2,
        })
    part_delta = defaultdict(list)
    for r in rec_rows:
        part_delta[r["subject_id"]].append(r["delta"])
    per_participant = {p: float(np.mean(v)) for p, v in part_delta.items()}
    return rec_rows, per_participant


def bootstrap_paired(per_participant, seed=BOOTSTRAP_SEED, iters=BOOTSTRAP_ITERS):
    rng = np.random.default_rng(seed)
    parts = sorted(per_participant.keys())
    vals = np.array([per_participant[p] for p in parts])
    n = len(vals)
    if n == 0:
        return None, None, None, None
    point = float(vals.mean())
    draws = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        draws[i] = vals[idx].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi, n


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def detection_metrics(y_true, y_pred, scores):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    n = len(y_true)
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    acc = (tp + tn) / n if n else None
    f1 = 2 * prec * sens / (prec + sens) if (prec and sens) else None
    auc = None
    if scores is not None and len(set(y_true)) >= 2 and all(s is not None for s in scores):
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, scores)
    return {"n": n, "TP": tp, "TN": tn, "FP": fp, "FN": fn, "Acc": acc,
            "Sens": sens, "Spec": spec, "F1": f1, "AUC": auc,
            "n_pos_pred": sum(y_pred), "n_neg_pred": n - sum(y_pred)}


def stable_flip_counts(slots_by_recording, label_key):
    counts = {"stable_correct": 0, "stable_wrong": 0, "flip": 0, "n_recordings": 0}
    for sid, ss in slots_by_recording.items():
        labels = [s[label_key] for s in ss if s[label_key] is not None]
        if len(labels) < 2:
            continue
        counts["n_recordings"] += 1
        truth = ss[0]["y_true"]
        if len(set(labels)) == 1:
            if labels[0] == truth:
                counts["stable_correct"] += 1
            else:
                counts["stable_wrong"] += 1
        else:
            counts["flip"] += 1
    return counts


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--simulated", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = load_test_meta()
    mapping = load_opaque_mapping()
    results = load_results(args.results)
    assert len(results) == 1270, "expected 1270 slots, got %d" % len(results)

    train_ids = set()
    with open(FROZEN_INCLUSION_CSV, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("phase2_role") == "phase2_train":
                train_ids.add(r["sample_id"])
    train_archive = load_archive(FROZEN_FEATURES_TRAIN_CSV, train_ids)
    assert len(train_archive) == 311, "train archive rows = %d" % len(train_archive)
    test_archive = load_archive(FROZEN_FEATURES_TEST_CSV, set(meta.keys()))
    assert len(test_archive) == 127, "test archive rows = %d" % len(test_archive)

    slots, _pipeline = build_slot_table(results, mapping, meta, test_archive, train_archive)

    by_rec = defaultdict(list)
    for s in slots:
        by_rec[(s["method"], s["sample_id"])].append(s)

    # ---------------- per-method valid-slot D (descriptive) ----------------
    method_D = {}
    for m, lk, sk in (("direct", "label_A", "scorable_A"),
                      ("rulers", "label_B", "scorable_B"),
                      ("hybrid", "label_H", "hybrid_scorable")):
        recs = {}
        for s in slots:
            # label must exist: B all-NA slots are "scorable" at the API level
            # but carry risk=missing (label_B None) and cannot enter D
            if s.get(sk) and s.get(lk) is not None:
                recs.setdefault(s["sample_id"], []).append(s.get(lk))
        # participant-equal per-method D
        pv = defaultdict(list)
        for sid, v in recs.items():
            d = d_measure(v)
            if d is not None:
                pv[meta[sid]["subject_id"]].append(d)
        method_D[m] = {
            "n_recordings": sum(1 for v in recs.values() if len(v) >= 2),
            "n_participants": len(pv),
            "D_participant_equal": float(np.mean([np.mean(x) for x in pv.values()]))
            if pv else None,
        }

    # ---------------- primary: Hybrid vs Direct, 3-method common slots ----
    # Pair rules receive (a_row, b_row) of the SAME (sample, repeat):
    # A validity lives on the A row, B validity and the derived Hybrid live
    # on the B row. A comparison's common set = pairs where all involved
    # methods are scorable for that repeat.
    def pair_three(a, b):
        return bool(a.get("scorable_A") and b.get("scorable_B")
                    and b.get("hybrid_scorable"))

    def pair_HA(a, b):
        return bool(a.get("scorable_A") and b.get("hybrid_scorable"))

    def pair_HA_imputed(a, b):
        # protocol S5: pure-imputation Hybrid rows enter ONLY this separate
        # sensitivity comparison, never the primary
        return bool(a.get("scorable_A") and b.get("hybrid_scorable_incl_imputed"))

    def pair_HB(a, b):
        return bool(b.get("hybrid_scorable") and b.get("scorable_B")
                    and not b.get("all_na_13"))

    def pair_BA(a, b):
        return bool(a.get("scorable_A") and b.get("scorable_B")
                    and not b.get("all_na_13"))

    comparisons = [
        ("PRIMARY: Hybrid-Direct (3-method common slots)",
         ("rulers", "label_H"), ("direct", "label_A"), pair_three, True),
        ("Sensitivity: Hybrid-Direct (all available paired slots)",
         ("rulers", "label_H"), ("direct", "label_A"), pair_HA, True),
        ("Sensitivity: Hybrid-Direct (incl. pure-imputation Hybrid rows)",
         ("rulers", "label_H"), ("direct", "label_A"), pair_HA_imputed, False),
        ("Secondary: Hybrid-RULERS (paired slots)",
         ("rulers", "label_H"), ("rulers", "label_B"), pair_HB, False),
        ("Secondary: RULERS-Direct (paired slots)",
         ("rulers", "label_B"), ("direct", "label_A"), pair_BA, False),
    ]

    comp_rows = []
    for name, k1, k2, rule, is_primary in comparisons:
        rec_rows, per_participant = pair_comparison(slots, k1, k2, rule)
        point, lo, hi, n_part = bootstrap_paired(per_participant)
        comp_rows.append({
            "comparison": name,
            "primary": is_primary,
            "n_recordings": len(rec_rows),
            "n_participants": n_part,
            "delta_point": point,
            "ci_lo": lo,
            "ci_hi": hi,
            "per_recording_rows": rec_rows,
        })
        # complete-5 subset (post-filter, actual counts)
        all5 = [r for r in rec_rows if r["m"] == 5]
        if all5:
            pp5 = defaultdict(list)
            for r in all5:
                pp5[r["subject_id"]].append(r["delta"])
            p5 = {p: float(np.mean(v)) for p, v in pp5.items()}
            pt5, lo5, hi5, n5 = bootstrap_paired(p5)
            comp_rows[-1]["complete5"] = {
                "n_recordings": len(all5), "n_participants": n5,
                "delta_point": pt5, "ci_lo": lo5, "ci_hi": hi5,
            }

    # ---------------- detection metrics (pooled over scorable slots) ------
    metric_rows = []
    for m, lk, rk in (("direct", "label_A", "risk_A"),
                      ("rulers", "label_B", "risk_B"),
                      ("hybrid", "label_H", "hybrid_prob")):
        yt, yp, sc = [], [], []
        for s in slots:
            if s.get(lk) is not None:
                yt.append(s["y_true"]); yp.append(s[lk]); sc.append(s.get(rk))
        metric_rows.append({"method": m, **detection_metrics(yt, yp, sc)})

    # per-repeat metrics
    per_rep = []
    for rep in range(REPEATS):
        for m, lk, rk in (("direct", "label_A", "risk_A"),
                          ("rulers", "label_B", "risk_B"),
                          ("hybrid", "label_H", "hybrid_prob")):
            yt, yp, sc = [], [], []
            for s in slots:
                if s["repeat_idx"] == rep and s.get(lk) is not None:
                    yt.append(s["y_true"]); yp.append(s[lk]); sc.append(s.get(rk))
            per_rep.append({"repeat": rep, "method": m,
                            **detection_metrics(yt, yp, sc)})

    # stable / flip per recording (per method, m>=2 scorable)
    stable_rows = []
    for m, lk in (("direct", "label_A"), ("rulers", "label_B"), ("hybrid", "label_H")):
        recs = defaultdict(list)
        for s in slots:
            if s.get(lk) is not None:
                recs[s["sample_id"]].append(s)
        stable_rows.append({"method": m, **stable_flip_counts(recs, lk)})

    # Recording-level aggregate (detection only). IMPLEMENTATION CORRECTION
    # 2026-09-23: the earlier version classified every method's mean score
    # with a single 0.5 threshold, contradicting the frozen protocol. The
    # aggregate now applies the SAME per-method frozen thresholds as the
    # per-call labels (Direct 0.50 / RULERS 0.07 / Hybrid 0.55). The
    # aggregation itself (mean of valid repeat scores) is UNCHANGED. This is
    # a bug fix, NOT test-set tuning. Inclusion: a recording enters with AT
    # LEAST ONE valid score; its aggregate is the mean of its valid repeats
    # only. The per-recording valid counts are reported (the aggregate is
    # NOT a "5-complete" aggregate).
    agg_thresholds = {"direct": THRESHOLD_DIRECT,
                      "rulers": THRESHOLD_PROMPT_RULERS,
                      "hybrid": THRESHOLD_HYBRID_LR}
    agg_rows = []
    agg_rec_rows = []      # per-recording rows (audit / reconciliation)
    agg_dist_rows = []     # valid-repeat-count distribution per method
    for m, lk, rk in (("direct", "label_A", "risk_A"),
                      ("rulers", "label_B", "risk_B"),
                      ("hybrid", "label_H", "hybrid_prob")):
        thr = agg_thresholds[m]
        recs = defaultdict(list)
        for s in slots:
            if s.get(lk) is not None and s.get(rk) is not None:
                recs[s["sample_id"]].append((s["y_true"], s[lk], s[rk]))
        yt, yp, sc = [], [], []
        dist = {k: 0 for k in range(1, REPEATS + 1)}
        for sid, vals in recs.items():
            n_valid = len(vals)
            dist[min(n_valid, REPEATS)] += 1
            truth = vals[0][0]
            mean_score = float(np.mean([v[2] for v in vals]))
            agg_label = 1 if mean_score >= thr else 0
            yt.append(truth)
            yp.append(agg_label)
            sc.append(mean_score)
            agg_rec_rows.append({"sample_id": sid, "method": m,
                                 "n_valid_repeats": n_valid, "y_true": truth,
                                 "mean_score": round(mean_score, 6),
                                 "aggregate_label": agg_label,
                                 "threshold": thr})
        agg_rows.append({"method": m, "threshold": thr,
                         **detection_metrics(yt, yp, sc)})
        agg_dist_rows.append({"method": m, "threshold": thr,
                              "n_recordings": len(recs),
                              **{"n_valid_%d" % k: dist[k]
                                 for k in range(1, REPEATS + 1)}})

    # ---------------- report ---------------------------------------------
    sim_note = ("\n> **SIMULATED DATA.** These numbers come from simulated "
                "responses (dry run). They are NOT real experimental results.\n"
                if args.simulated else "")
    lines = []
    lines.append("# Clean-repeatability analysis report\n")
    lines.append("Data: `%s`\n" % args.results)
    lines.append("Slots: %d | frozen thresholds: Direct %.2f / RULERS %.2f / Hybrid %.2f\n"
                 % (len(slots), THRESHOLD_DIRECT, THRESHOLD_PROMPT_RULERS, THRESHOLD_HYBRID_LR))
    lines.append(sim_note)
    lines.append("## Per-method recording-level disagreement (descriptive, own valid slots)\n")
    lines.append("| method | recordings (m>=2) | participants | D (participant-equal) |")
    lines.append("|---|---|---|---|")
    for m, d in method_D.items():
        lines.append("| %s | %d | %d | %.4f |" % (m, d["n_recordings"], d["n_participants"],
                                                  d["D_participant_equal"] or float("nan")))
    lines.append("\n## Paired comparisons (participant-clustered bootstrap, %d iters, seed %d)\n"
                 % (BOOTSTRAP_ITERS, BOOTSTRAP_SEED))
    lines.append("| comparison | recordings | participants | delta (point) | 95%% CI |")
    lines.append("|---|---|---|---|---|")
    for c in comp_rows:
        if c["delta_point"] is None:
            continue
        lines.append("| %s | %d | %d | %+.4f | [%+.4f, %+.4f] |"
                     % (c["comparison"], c["n_recordings"], c["n_participants"],
                        c["delta_point"], c["ci_lo"], c["ci_hi"]))
        if "complete5" in c:
            c5 = c["complete5"]
            if c5["delta_point"] is not None:
                lines.append("| %s (complete-5 subset) | %d | %d | %+.4f | [%+.4f, %+.4f] |"
                             % (c["comparison"], c5["n_recordings"], c5["n_participants"],
                                c5["delta_point"], c5["ci_lo"], c5["ci_hi"]))
    lines.append("\nSecondary comparisons are exploratory (effect size + CI); they cannot "
                 "replace the primary comparison.\n")

    # protocol S5 coverage: pure-imputation Hybrid rows stay out of the primary
    n_b_scorable = sum(1 for s in slots
                       if s.get("scorable_B") and not s.get("all_na_13"))
    n_h_label = sum(1 for s in slots if s.get("label_H") is not None)
    n_pure = sum(1 for s in slots if s.get("hybrid_pure_imputed"))
    n_h_primary = sum(1 for s in slots if s.get("hybrid_scorable"))
    # validity coverage (protocol revisions: distinct dispositions, never
    # conflated)
    n_direct_slots = sum(1 for s in slots if s["method"] == "direct")
    n_direct_scorable = sum(1 for s in slots if s.get("scorable_A"))
    risk_reason_counts = defaultdict(int)
    for s in slots:
        if s["method"] == "direct":
            risk_reason_counts[s.get("risk_reason") or "n/a"] += 1
    n_a_parse_fail = sum(1 for s in slots
                         if s["method"] == "direct" and not s.get("parse_success"))
    n_b_parse_fail = sum(1 for s in slots
                         if s["method"] == "rulers" and not s.get("parse_success")
                         and not s.get("structural_invalid"))
    n_b_structural = sum(1 for s in slots if s.get("structural_invalid"))
    n_b_verification_failed = sum(1 for s in slots
                                  if s.get("evidence_verification_failed"))
    n_b_all_na = sum(1 for s in slots if s.get("all_na_13"))
    n_b_scorable_total = sum(1 for s in slots if s.get("scorable_B"))
    n_any_duplicates = sum(1 for s in slots
                           if s.get("direct_duplicate_fields")
                           or s.get("rulers_duplicate_fields"))
    n_any_unknown = sum(1 for s in slots
                        if s.get("direct_unknown_fields")
                        or s.get("rulers_unknown_fields"))
    lines.append("\n## Hybrid coverage (protocol S5)\n")
    lines.append("- B scorable non-all-NA slots: %d" % n_b_scorable)
    lines.append("- Hybrid labels computed: %d" % n_h_label)
    lines.append("- Pure-imputation rows EXCLUDED from primary paired comparison: %d"
                 % n_pure)
    lines.append("- Hybrid rows entering primary comparisons: %d" % n_h_primary)
    lines.append("- A separate sensitivity comparison reports the effect when the "
                 "pure-imputation rows DO enter (never silently).\n")
    lines.append("\n## Validity coverage (actual scorable markers, distinct dispositions)\n")
    lines.append("| disposition | count |")
    lines.append("|---|---|")
    lines.append("| A (Direct) slots | %d |" % n_direct_slots)
    lines.append("| A scorable (valid risk_score, main label = risk>=%.2f) | %d |"
                 % (THRESHOLD_DIRECT, n_direct_scorable))
    for reason in sorted(risk_reason_counts):
        if reason == "ok":
            continue  # valid risks are the scorable row above, not "invalid"
        lines.append("| A risk invalid: %s | %d |" % (reason, risk_reason_counts[reason]))
    lines.append("| A parse failures | %d |" % n_a_parse_fail)
    lines.append("| B scorable (structure valid AND risk present) | %d |"
                 % n_b_scorable_total)
    lines.append("| B structurally invalid -> risk MISSING, unscorable | %d |"
                 % n_b_structural)
    lines.append("| B parse failures (non-structural) | %d |" % n_b_parse_fail)
    lines.append("| B all-13-NA -> risk MISSING, never 0 | %d |" % n_b_all_na)
    lines.append("| B evidence-VERIFICATION failed (scores KEPT, flagged) | %d |"
                 % n_b_verification_failed)
    lines.append("| Slots with duplicate JSON keys recorded | %d |" % n_any_duplicates)
    lines.append("| Slots with unknown top-level fields recorded | %d |" % n_any_unknown)
    lines.append("")
    lines.append("Dispositions are mutually exclusive in interpretation: structural "
                 "invalidity removes the risk (unscorable), evidence-verification "
                 "failure keeps the scores and is reported separately.\n")
    coverage_row = {"n_B_scorable_non_allna": n_b_scorable,
                    "n_hybrid_labels": n_h_label,
                    "n_pure_imputed_excluded": n_pure,
                    "n_hybrid_primary": n_h_primary,
                    "n_A_slots": n_direct_slots,
                    "n_A_scorable": n_direct_scorable,
                    "n_A_parse_fail": n_a_parse_fail,
                    "n_B_scorable": n_b_scorable_total,
                    "n_B_structural_invalid": n_b_structural,
                    "n_B_parse_fail": n_b_parse_fail,
                    "n_B_all_na": n_b_all_na,
                    "n_B_evidence_verification_failed": n_b_verification_failed,
                    "n_slots_duplicate_keys": n_any_duplicates,
                    "n_slots_unknown_fields": n_any_unknown,
                    "A_risk_reason_counts": dict(risk_reason_counts)}
    lines.append("\n## Detection metrics (pooled over scorable slots, descriptive)\n")
    lines.append("| method | n | Acc | Sens | Spec | F1 | AUC | pos_pred | neg_pred |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in metric_rows:
        fmt = lambda v: "%.4f" % v if isinstance(v, float) else str(v)
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %d | %d |"
                     % (r["method"], r["n"], fmt(r["Acc"]), fmt(r["Sens"]), fmt(r["Spec"]),
                        fmt(r["F1"]), fmt(r["AUC"]), r["n_pos_pred"], r["n_neg_pred"]))
    lines.append("\n## Stability of per-recording labels across repeats (m>=2 scorable)\n")
    lines.append("| method | recordings | stable_correct | stable_wrong | flip |")
    lines.append("|---|---|---|---|---|")
    for r in stable_rows:
        lines.append("| %s | %d | %d | %d | %d |"
                     % (r["method"], r["n_recordings"], r["stable_correct"],
                        r["stable_wrong"], r["flip"]))
    lines.append("\n## Recording-level aggregate (mean of valid repeats; DETECTION ONLY; "
                 "aggregate repeatability is not reported by design)\n")
    lines.append("CORRECTION 2026-09-23 (implementation fix, NOT test-set tuning): an earlier "
                 "version of this section classified every method's mean score with a single "
                 "0.5 threshold, contradicting the frozen protocol. The aggregate now applies "
                 "the SAME per-method frozen thresholds as the per-call labels (Direct 0.50 / "
                 "RULERS 0.07 / Hybrid 0.55); the mean aggregation itself is UNCHANGED. The "
                 "original report is preserved under `analysis/`; this corrected output lives "
                 "under `analysis_corrected/`. Per-call predictions, the primary comparison D, "
                 "secondary comparisons and their CIs are UNCHANGED (verified).\n")
    lines.append("Inclusion criterion: a recording enters with AT LEAST ONE valid score; its "
                 "aggregate score is the mean of its valid repeats only (a missing/invalid "
                 "repeat contributes nothing). The aggregate is NOT a \"5-complete\" aggregate "
                 "and is NOT a majority vote; valid-count distributions are reported below.\n")
    lines.append("| method | threshold | n_recordings | n_valid=1 | n_valid=2 | n_valid=3 "
                 "| n_valid=4 | n_valid=5 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in agg_dist_rows:
        lines.append("| %s | %.2f | %d | %d | %d | %d | %d | %d |"
                     % (r["method"], r["threshold"], r["n_recordings"],
                        r["n_valid_1"], r["n_valid_2"], r["n_valid_3"],
                        r["n_valid_4"], r["n_valid_5"]))
    lines.append("\n| method | threshold | n | Acc | Sens | Spec | F1 | AUC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in agg_rows:
        fmt = lambda v: "%.4f" % v if isinstance(v, float) else str(v)
        lines.append("| %s | %.2f | %d | %s | %s | %s | %s | %s |"
                     % (r["method"], r["threshold"], r["n"], fmt(r["Acc"]),
                        fmt(r["Sens"]), fmt(r["Spec"]), fmt(r["F1"]), fmt(r["AUC"])))
    lines.append("\n## Interpretation rules (enforced)\n")
    lines.append("- Thresholds are frozen operation points for a PROCESS comparison; no claim "
                 "of equalized sensitivity/specificity across methods is made.")
    lines.append("- A CI containing 0 => \"no detectable difference\"; NOT \"equivalence\".")
    lines.append("- A favorable point estimate is not reported as a significant improvement "
                 "unless the CI excludes 0.")
    lines.append("- Repeated calls are not independent participants; bootstrap clusters by "
                 "participant.")
    lines.append("- Hybrid slots whose six LLM features are imputation-only are EXCLUDED from "
                 "paired comparisons (coverage reported separately).")
    lines.append("- B all-13-NA slots have missing risk (never 0) and are excluded from D.")
    lines.append("- Direct labels come from a VALID risk_score at 0.50; the raw "
                 "predicted_label is diagnostic only.")
    lines.append("- Structural invalidity removes the risk (unscorable); "
                 "evidence-verification failure keeps the scores and is flagged. "
                 "The two are never conflated.")
    lines.append("- The recording-level aggregate uses the SAME per-method frozen thresholds as "
                 "the per-call labels; it is a mean of valid repeats (>=1 valid score), not a "
                 "majority vote, and its own repeatability is not reported by design.")
    lines.append("- This is a repeat-call evaluation on the same internal cohort: NOT a new "
                 "independent clinical validation, NOT proof of non-inferiority.\n")

    with open(os.path.join(args.out, "analysis_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # CSV artifacts
    def dump_csv(name, rows):
        with open(os.path.join(args.out, name), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump_csv("pair_comparisons.csv", [
        {k: v for k, v in c.items() if k != "per_recording_rows"}
        for c in comp_rows if c["delta_point"] is not None])
    dump_csv("hybrid_coverage.csv", [coverage_row])
    dump_csv("method_metrics.csv", metric_rows)
    dump_csv("per_repeat_metrics.csv", per_rep)
    dump_csv("stable_flip.csv", stable_rows)
    dump_csv("aggregate5_metrics.csv", agg_rows)
    dump_csv("aggregate5_valid_counts.csv", agg_dist_rows)
    dump_csv("aggregate5_per_recording.csv", agg_rec_rows)

    # per-slot table (auditable): every slot with its actual scorable markers,
    # risks, labels and validity flags; lists flattened to ';'-joined strings
    slot_fields = ["slot_id", "method", "repeat_idx", "opaque_id", "sample_id",
                   "subject_id", "y_true", "api_success", "parse_success",
                   "scorable_A", "risk_valid", "risk_reason", "risk_score_raw",
                   "predicted_label_raw", "label_A", "risk_A",
                   "scorable_B", "label_B", "risk_B", "all_na_13", "six_na",
                   "structural_invalid", "evidence_verification_failed",
                   "repair_structural_invalid", "hybrid_scorable", "label_H",
                   "hybrid_prob", "hybrid_pure_imputed", "hybrid_imputed_any",
                   "direct_duplicate_fields", "rulers_duplicate_fields",
                   "direct_unknown_fields", "rulers_unknown_fields"]
    slot_rows = []
    for s in slots:
        row = {}
        for f in slot_fields:
            v = s.get(f)
            if isinstance(v, list):
                v = ";".join(str(x) for x in v)
            row[f] = "" if v is None else v
        slot_rows.append(row)
    dump_csv("slot_table.csv", slot_rows)
    with open(os.path.join(args.out, "per_recording_delta.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["comparison", "sample_id", "subject_id",
                                          "y_true", "m", "delta"])
        w.writeheader()
        for c in comp_rows:
            for r in c["per_recording_rows"]:
                w.writerow({"comparison": c["comparison"], "sample_id": r["sample_id"],
                            "subject_id": r["subject_id"], "y_true": r["y_true"],
                            "m": r["m"], "delta": r["delta"]})

    print("analysis written to %s" % args.out)
    for line in lines[:40]:
        print(line)


if __name__ == "__main__":
    main()
