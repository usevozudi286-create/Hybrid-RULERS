# -*- coding: utf-8 -*-
"""Offline verification of the 2026-09-23 aggregate-threshold correction.

Reads only frozen on-disk artifacts (the two analysis directories). Makes
NO API calls. Verifies:

  1. Per-method aggregate AUC is identical in the OLD and CORRECTED output
     (AUC is rank-based, so a threshold-only fix must not change it).
  2. Every analysis artifact OTHER than the aggregate metrics file is
     byte-identical between results_live/analysis and
     results_live/analysis_corrected (per-call predictions, primary D,
     secondaries, CIs, detection/stability/coverage/validity all unchanged).
  3. Independent recomputation of the AGGREGATE STEP only, from the frozen
     per-slot predictions (slot_table.csv, itself verified byte-identical in
     check 2): mean of valid repeat scores, per-method frozen thresholds
     (Direct 0.50 / RULERS 0.07 / Hybrid 0.55), matched item-by-item against
     aggregate5_per_recording.csv / aggregate5_valid_counts.csv /
     aggregate5_metrics.csv.

This is an implementation-correctness audit of the fix, NOT a new
experiment, NOT tuning. The old artifacts under results_live/analysis are
treated as frozen and are only READ.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OLD = os.path.join(ROOT, "results_live", "analysis")
NEW = os.path.join(ROOT, "results_live", "analysis_corrected")

THRESHOLDS = {"direct": 0.50, "rulers": 0.07, "hybrid": 0.55}
KEYMAP = {"direct": ("label_A", "risk_A"),
          "rulers": ("label_B", "risk_B"),
          "hybrid": ("label_H", "hybrid_prob")}

failures = []


def check(name, ok, detail=""):
    print("%-72s %s%s" % (name, "PASS" if ok else "FAIL",
                          ("  " + detail) if detail else ""))
    if not ok:
        failures.append(name)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# --- 1. AUC unchanged (threshold-only fix must leave rank-based AUC intact)
old_agg = {r["method"]: r for r in read_csv(os.path.join(OLD, "aggregate5_metrics.csv"))}
new_agg = {r["method"]: r for r in read_csv(os.path.join(NEW, "aggregate5_metrics.csv"))}
for m in ("direct", "rulers", "hybrid"):
    check("1. aggregate AUC unchanged (%s)" % m,
          float(old_agg[m]["AUC"]) == float(new_agg[m]["AUC"]),
          "old=%s new=%s" % (old_agg[m]["AUC"], new_agg[m]["AUC"]))

# --- 2. All artifacts other than the aggregate metrics byte-identical
for fn in ["pair_comparisons.csv", "method_metrics.csv", "per_repeat_metrics.csv",
           "stable_flip.csv", "hybrid_coverage.csv", "per_recording_delta.csv",
           "slot_table.csv"]:
    with open(os.path.join(OLD, fn), "rb") as f:
        old_bytes = f.read()
    with open(os.path.join(NEW, fn), "rb") as f:
        new_bytes = f.read()
    check("2. byte-identical: %s" % fn, old_bytes == new_bytes,
          "" if old_bytes == new_bytes else "old=%dB new=%dB" % (len(old_bytes), len(new_bytes)))

# --- 3. Independent recomputation of the aggregate step from per-slot rows
slots = read_csv(os.path.join(OLD, "slot_table.csv"))  # frozen per-call predictions
recs = {}
for s in slots:
    for m, (lk, rk) in KEYMAP.items():
        if s.get(lk) not in ("", None) and s.get(rk) not in ("", None):
            recs.setdefault((s["sample_id"], m), []).append(
                (int(float(s["y_true"])), float(s[rk])))

expected = {}
for (sid, m), vals in recs.items():
    truth = vals[0][0]
    mean_score = sum(v[1] for v in vals) / len(vals)
    label = 1 if mean_score >= THRESHOLDS[m] else 0
    expected[(sid, m)] = (len(vals), truth, round(mean_score, 6), label)

new_rec = read_csv(os.path.join(NEW, "aggregate5_per_recording.csv"))
check("3. per-recording row count matches recomputation",
      len(new_rec) == len(expected),
      "csv=%d recomputed=%d" % (len(new_rec), len(expected)))

n_mismatch = 0
n_checked = 0
for row in new_rec:
    key = (row["sample_id"], row["method"])
    n_checked += 1
    exp = expected.get(key)
    if exp is None:
        n_mismatch += 1
        print("    unexpected row: %s" % (key,))
        continue
    if (int(row["n_valid_repeats"]) != exp[0]
            or int(row["y_true"]) != exp[1]
            or float(row["mean_score"]) != exp[2]
            or int(row["aggregate_label"]) != exp[3]
            or float(row["threshold"]) != THRESHOLDS[row["method"]]):
        n_mismatch += 1
        print("    mismatch: %s %s (exp %s)" % (key, row, exp))
check("3. per-recording rows match item-by-item (n=%d)" % n_checked,
      n_mismatch == 0 and n_checked == len(expected))

# per-method valid-count distributions recomputed independently
new_dist = {r["method"]: r for r in read_csv(os.path.join(NEW, "aggregate5_valid_counts.csv"))}
ok_dist = True
for m in ("direct", "rulers", "hybrid"):
    from collections import Counter
    cnt = Counter(len(vals) for (sid, mm), vals in recs.items() if mm == m)
    row = new_dist[m]
    for k in range(1, 6):
        if int(row["n_valid_%d" % k]) != cnt.get(k, 0):
            ok_dist = False
            print("    dist mismatch %s n_valid_%d: csv=%s recomputed=%d"
                  % (m, k, row["n_valid_%d" % k], cnt.get(k, 0)))
    if int(row["n_recordings"]) != sum(cnt.values()):
        ok_dist = False
        print("    n_recordings mismatch %s: csv=%s recomputed=%d"
              % (m, row["n_recordings"], sum(cnt.values())))
check("3. valid-count distributions match recomputation", ok_dist)

# aggregate metrics recomputed from the reconciled per-recording rows
ok_met = True
for m in ("direct", "rulers", "hybrid"):
    yt, yp = [], []
    for (sid, mm), (n_v, truth, ms, lab) in expected.items():
        if mm == m:
            yt.append(truth)
            yp.append(lab)
    n = len(yt)
    tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
    acc = (tp + tn) / n if n else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    row = new_agg[m]
    for key, val in (("n", n), ("TP", tp), ("TN", tn), ("FP", fp), ("FN", fn),
                     ("Acc", acc), ("Sens", sens), ("Spec", spec)):
        if float(row[key]) != float(val):
            ok_met = False
            print("    metric mismatch %s %s: csv=%s recomputed=%s"
                  % (m, key, row[key], val))
check("3. aggregate confusion metrics match recomputation", ok_met)

# --- 4. Old artifacts preserved and the fix changed ONLY what it should
check("old aggregate5_metrics.csv preserved (no threshold column)",
      "threshold" not in old_agg["direct"])
with open(os.path.join(OLD, "analysis_report.md"), "r", encoding="utf-8") as f:
    old_report = f.read()
check("old analysis_report.md preserved (old heading + old rulers row intact)",
      "Single 5-repeat aggregate" in old_report and "| rulers | 114 | 0.4825 |" in old_report)
# direct used 0.5 all along -> its aggregate row must be unchanged by the fix
met_keys = ("n", "Acc", "Sens", "Spec", "F1")
check("4. direct aggregate metrics unchanged (threshold was already 0.50)",
      all(float(old_agg["direct"][k]) == float(new_agg["direct"][k]) for k in met_keys))
# rulers/hybrid rows MUST have changed (the bug affected exactly these two)
check("4. rulers + hybrid aggregate metrics did change (fix had an effect)",
      any(float(old_agg["rulers"][k]) != float(new_agg["rulers"][k]) for k in met_keys)
      and any(float(old_agg["hybrid"][k]) != float(new_agg["hybrid"][k]) for k in met_keys))

print()
if failures:
    print("FAILED: %d check(s)" % len(failures))
    sys.exit(1)
print("ALL VERIFICATION CHECKS PASSED")
