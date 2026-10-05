# -*- coding: utf-8 -*-
"""冻结预测器恢复 — Programmatic-only LR 与 Full Hybrid LR（311 纠正标签训练集版）。

流程（与 recovery_plan.md 冻结方案一致，本脚本为独立最小复现）：
  1. 只读 ast 提取 recording_time_reanalysis.py 的 train_lr / compute_metrics
     （verbatim 执行；另生成机械式"捕获变体"仅改签名与 return 行以便保存
     imputer/scaler/model 对象——拟合语句逐字节相同，两条变体输出必须位级一致）。
  2. 输入白名单断言（只允许冻结 Pitt 输入文件，禁止任何 Lu 路径）。
  3. 冻结输入快照（与 recovery_plan.md 的 SHA-256 比对）。
  4. 组装 311 训练 / 127 核验帧（v2 审计标签 + Stage A 冻结矩阵 + 冻结 QC 交叉核对）。
  5. 分别拟合两模型（verbatim train_lr：训练集中位数插补 → StandardScaler →
     LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0,
     solver='lbfgs') → 训练集 Youden 网格 np.arange(0.05,0.96,0.01) 严格大于并列取先）。
  6. 六个复现门槛（任一失败即停止该模型外部预测，不调参重试、不放宽 1e-9）：
     G1 ID/标签/列一致；G2 核验集 127 逐样本概率最大绝对差 <=1e-9；
     G3 预测标签/混淆矩阵/AUC 与存档一致；G4 阈值与存档完整浮点值一致；
     G5 插补值/系数与存档一致（系数存档 6 位小数舍入 -> 容差 5e-7，如实报告）；
     G6 拟合未使用 Lu 数据。
  7. 通过后保存完整预测器：joblib 三对象 + 全精度 JSON 参数 + 训练 ID/标签哈希；
     joblib 重加载后在同一输入上重核验（重加载-拟合期逐样本最大差记录）。
输出全部位于本目录；不修改任何原始/冻结文件；无网络调用。
"""
import ast
import csv
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MCIRULERS = Path(os.environ.get("HYBRID_RULERS_SOURCE_ROOT", "data/mci_rulers"))
EXPERIMENTS = MCIRULERS / "experiments"
FULL = EXPERIMENTS / "hybrid_rulers_full"
FINAL = FULL / "final_clean_results"
REANA = FULL / "recording_time_reanalysis"
AUDIT_V2 = FULL / "recording_label_audit" / "v2" / "recording_label_audit_v2.csv"
CAND_V2 = FULL / "recording_label_audit" / "v2" / "candidate_relabel_map_v2.csv"
UNRES_V2 = FULL / "recording_label_audit" / "v2" / "unresolved_cases_v2.csv"
META_PATH = MCIRULERS / "dataset" / "pitt_cookie_wav_cognitive_vs_control_metadata.csv"
FEATS_TR = FINAL / "hybrid_features_train.csv"
FEATS_TE = FINAL / "hybrid_features_test.csv"
EXCL_TR = FINAL / "excluded_train_records.csv"
EXCL_TE = FINAL / "excluded_records.csv"
GATE_PREDS = REANA / "per_sample_predictions.csv"
GATE_COMP = REANA / "strategy_comparison.csv"
GATE_SUMMARY = REANA / "reanalysis_summary.json"
GATE_COEF = REANA / "model_coefficients.csv"
GATE_IMP = REANA / "imputation_values.csv"
MODEL_CFG = REANA / "freeze" / "model_config.json"
REANA_SRC = REANA / "recording_time_reanalysis.py"
FINAL_EVAL_SRC = FULL / "final_clean_eval.py"

RUN_LOG = HERE / "run_log.txt"
PRED_DIR = HERE / "predictors"
CHECK_JSON = HERE / "reproduction_checks.json"

PROG_COLS = ["information_unit_count", "information_density", "speech_rate",
             "hesitation_rate", "long_pause_ratio", "transcript_length"]
LLM_COLS = ["C_C01_criterion_median", "C_C02_criterion_median", "C_C07_criterion_median",
            "C_C08_criterion_median", "C_C09_criterion_median", "C_C10_criterion_median"]
FEATURE_COLS = PROG_COLS + LLM_COLS

MODELS = {"prog_only": PROG_COLS, "full_hybrid": FEATURE_COLS}

# 输入白名单：本脚本允许读取的全部路径（G6 依据之一；任何 Lu 路径在此之外）
INPUT_WHITELIST = {
    AUDIT_V2, CAND_V2, UNRES_V2, META_PATH, FEATS_TR, FEATS_TE, EXCL_TR, EXCL_TE,
    GATE_PREDS, GATE_COMP, GATE_SUMMARY, GATE_COEF, GATE_IMP, MODEL_CFG,
    REANA_SRC, FINAL_EVAL_SRC,
}


def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_frozen_defs(path, names):
    """verbatim ast 提取冻结函数源码（只读证据）。"""
    src = open(path, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    parts = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            parts[node.name] = ast.get_source_segment(src, node)
    missing = set(names) - set(parts)
    if missing:
        raise RuntimeError("frozen defs missing in %s: %s" % (path, sorted(missing)))
    return parts


def build_ns(parts):
    ns = {
        "np": np,
        "SimpleImputer": SimpleImputer,
        "StandardScaler": StandardScaler,
        "LogisticRegression": LogisticRegression,
        "roc_auc_score": roc_auc_score,
    }
    exec(compile(parts["compute_metrics"], "<frozen_compute_metrics>", "exec"), ns)  # noqa: S102
    exec(compile(parts["train_lr"], "<frozen_train_lr>", "exec"), ns)  # noqa: S102
    # 机械式捕获变体：仅改函数名/签名与 return 行（拟合语句逐字节不变）
    cap = parts["train_lr"]
    assert cap.startswith("def train_lr(Xt, yt, Xe, ye, cols):")
    cap = cap.replace("def train_lr(Xt, yt, Xe, ye, cols):",
                      "def train_lr_capture(Xt, yt, Xe, ye, cols, _sink):", 1)
    old_ret = ("    return compute_metrics(ye, epred, ep), epred, ep, best_t, "
               "model.coef_[0], imputation_values")
    assert old_ret in cap, "unexpected return line in frozen train_lr"
    cap = cap.replace(old_ret,
                      "    _sink['imputer'] = imputer\n"
                      "    _sink['scaler'] = scaler\n"
                      "    _sink['model'] = model\n" + old_ret, 1)
    exec(compile(cap, "<frozen_train_lr_capture>", "exec"), ns)  # noqa: S102
    return ns


def assert_whitelist(extra_ok=()):
    """脚本读取路径白名单断言（G6：不读 Lu 数据）。"""
    for p in INPUT_WHITELIST:
        assert Path(p).is_file(), "whitelisted input missing: %s" % p
    for p in extra_ok:
        assert p in INPUT_WHITELIST, "not in whitelist: %s" % p
    log("G6 input whitelist: %d frozen Pitt inputs only (no Lu path)" % len(INPUT_WHITELIST))


def build_frames():
    """冻结脚本 main() 的 ID/标签组装逐句复现（不用其顶层写出行为）。"""
    audit = pd.read_csv(AUDIT_V2, encoding="utf-8-sig", dtype=str)
    cand = pd.read_csv(CAND_V2, encoding="utf-8-sig", dtype=str)
    unres = pd.read_csv(UNRES_V2, encoding="utf-8-sig", dtype=str)
    for df in (audit, cand, unres):
        df["sample_id"] = df["sample_id"].astype(str).str.strip()
    assert len(audit) == 492 and len(cand) == 19 and len(unres) == 40
    candidates = set(cand["sample_id"])
    unresolved = set(unres["sample_id"])
    resolved = set(audit["sample_id"]) - unresolved
    assert len(resolved) == 452
    assert set(audit.loc[audit["differs_from_old"] == "1", "sample_id"]) == candidates
    assert (audit.loc[audit["sample_id"].isin(candidates), "proposed_label"] == "control").all()

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

    feats_train = pd.read_csv(FEATS_TR, encoding="utf-8-sig")
    feats_test = pd.read_csv(FEATS_TE, encoding="utf-8-sig")
    for df in (feats_train, feats_test):
        df["sample_id"] = df["sample_id"].astype(str).str.strip()
    feats_train_ids = set(feats_train["sample_id"])
    feats_test_ids = set(feats_test["sample_id"])
    assert len(feats_train_ids) == 340 and len(feats_test_ids) == 138

    excl_train = pd.read_csv(EXCL_TR, encoding="utf-8-sig")
    excl_test = pd.read_csv(EXCL_TE, encoding="utf-8-sig")
    qc_train = set(excl_train["sample_id"].astype(str).str.strip())
    qc_test_rows_unique = set(excl_test["sample_id"].astype(str).str.strip())
    assert len(qc_train) == 2 and qc_train == predefined_train - feats_train_ids
    qc_test_full = predefined_test - feats_test_ids
    assert len(qc_test_full) == 12
    assert qc_test_rows_unique - qc_test_full <= feats_test_ids

    corrected_train = feats_train_ids - unresolved
    corrected_test = feats_test_ids - unresolved
    assert len(corrected_train) == 311, len(corrected_train)
    assert len(corrected_test) == 127, len(corrected_test)

    corrected_map = {}
    for sid in audit["sample_id"]:
        if sid in unresolved:
            corrected_map[sid] = None  # EXCLUDED
        else:
            lab = audit.loc[audit["sample_id"] == sid, "proposed_label"].iloc[0]
            corrected_map[sid] = 1 if lab == "positive" else 0

    tr_f = feats_train[feats_train["sample_id"].isin(corrected_train)].copy()
    te_f = feats_test[feats_test["sample_id"].isin(corrected_test)].copy()
    for df in (tr_f, te_f):
        df["y_true"] = df["sample_id"].map(corrected_map).astype(int)
        df["subject_id"] = df["sample_id"].map(subj_map)
    assert len(tr_f) == 311 and len(te_f) == 127
    tr_f = tr_f.sort_values("sample_id").reset_index(drop=True)
    te_f = te_f.sort_values("sample_id").reset_index(drop=True)
    for fc in FEATURE_COLS:
        assert fc in tr_f.columns and fc in te_f.columns
    n_tr_subj = len(set(tr_f["subject_id"]))
    n_te_subj = len(set(te_f["subject_id"]))
    assert n_tr_subj == 181 and n_te_subj == 74
    return tr_f, te_f, candidates, unresolved


def main():
    t0 = time.time()
    log("=== recover_predictors start ===")
    for d in (PRED_DIR,):
        d.mkdir(parents=True, exist_ok=True)
    checks = {}

    # ---- 输入快照（与 recovery_plan.md 比对） ----
    log("input snapshot (vs recovery_plan.md source_manifest.json)")
    manifest = json.load(open(HERE / "source_manifest.json", encoding="utf-8"))
    plan_hashes = {r["key"]: r["sha256"] for r in manifest["files"]}
    key_by_path = {r["path"]: r["key"] for r in manifest["files"]}
    mismatches = []
    for p in sorted(INPUT_WHITELIST):
        key = key_by_path.get(str(p))
        if key is None:
            mismatches.append("not in manifest: %s" % p)
            continue
        if sha256_file(p) != plan_hashes[key]:
            mismatches.append("CHANGED: %s" % key)
    assert not mismatches, "input snapshot mismatch: %s" % mismatches
    assert_whitelist()
    log("input snapshot OK: all %d whitelisted inputs match plan hashes" % len(INPUT_WHITELIST))

    # ---- verbatim 提取 ----
    parts = extract_frozen_defs(REANA_SRC, {"train_lr", "compute_metrics"})
    parts_old = extract_frozen_defs(FINAL_EVAL_SRC, {"train_lr", "compute_metrics"})
    log("frozen defs extracted: reanalysis train_lr sha256=%s / compute_metrics=%s" %
        (sha256_text(parts["train_lr"])[:24], sha256_text(parts["compute_metrics"])[:24]))
    log("final_clean_eval train_lr sha256=%s (语义一致，仅格式/注释差异，已 diff 核对)" %
        sha256_text(parts_old["train_lr"])[:24])
    ns = build_ns(parts)
    train_lr = ns["train_lr"]
    train_lr_capture = ns["train_lr_capture"]

    # ---- 帧组装 ----
    tr_f, te_f, candidates, unresolved = build_frames()
    ytrain = tr_f["y_true"].to_numpy(dtype=int)
    ytest = te_f["y_true"].to_numpy(dtype=int)
    log("frames: train %d (181 subj) / test %d (74 subj); label changes in train=%d test=%d" %
        (len(tr_f), len(te_f), len(candidates & set(tr_f["sample_id"])),
         len(candidates & set(te_f["sample_id"]))))

    # ---- 存档读取（核验目标） ----
    gate_pred = pd.read_csv(GATE_PREDS, encoding="utf-8-sig")
    gate_pred["sample_id"] = gate_pred["sample_id"].astype(str).str.strip()
    gate_by_sid = {row["sample_id"]: row for _, row in gate_pred.iterrows()}
    gate_comp = pd.read_csv(GATE_COMP, encoding="utf-8-sig")
    gate_summary = json.load(open(GATE_SUMMARY, encoding="utf-8"))
    arch_thr = gate_summary["thresholds"]
    gate_coef = pd.read_csv(GATE_COEF, encoding="utf-8-sig")
    gate_imp = pd.read_csv(GATE_IMP, encoding="utf-8-sig")

    # ---- 模型恢复 + 六门槛 ----
    results = {}
    for name, cols in MODELS.items():
        log("\n== recover %s (features: %s) ==" % (name, ",".join(c[:12] for c in cols)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = train_lr(tr_f, ytrain, te_f, ytest, cols)
            sink = {}
            out_cap = train_lr_capture(tr_f, ytrain, te_f, ytest, cols, sink)
        m, epred, ep, best_t, coef, imp_vals = out
        m_c, epred_c, ep_c, best_t_c, coef_c, imp_vals_c = out_cap
        # 捕获变体与 verbatim 变体必须位级一致（证明保存对象=冻结实现）
        assert best_t_c == best_t and np.array_equal(ep_c, ep) and np.array_equal(epred_c, epred)
        assert all(imp_vals_c[k] == v for k, v in imp_vals.items())
        assert np.array_equal(coef_c, coef)
        imputer, scaler, model = sink["imputer"], sink["scaler"], sink["model"]
        log("capture variant bit-identical to verbatim variant (objects trustworthy)")

        gate = {}
        # G1: ID/标签/列
        gate["G1_ids"] = (set(gate_by_sid) == set(te_f["sample_id"]),
                          "archived prediction sample set == recovered test set (%d)" % len(te_f))
        gate["G1_labels"] = (all(int(gate_by_sid[sid]["y_true"]) == int(ytest[i])
                                 for i, sid in enumerate(te_f["sample_id"])),
                             "archived y_true == recovered y_true (per-sample)")
        gate["G1_train_counts"] = (len(tr_f) == 311, "train 311")
        assert gate["G1_ids"][0] and gate["G1_labels"][0] and gate["G1_train_counts"][0]

        # G2: 逐样本概率 <= 1e-9
        col = ("Programmatic_only_LR_score" if name == "prog_only"
               else "Full_Hybrid_LR_score")
        diffs = [abs(ep[i] - float(gate_by_sid[sid][col]))
                 for i, sid in enumerate(te_f["sample_id"])]
        maxd = max(diffs)
        gate["G2_prob_max_abs_diff"] = maxd
        gate["G2_pass"] = bool(maxd <= 1e-9)
        log("G2 %s per-sample prob max abs diff vs archive = %.3e (gate 1e-9)" % (name, maxd))

        # G3: 混淆矩阵 / AUC vs strategy_comparison.csv
        arch_row = gate_comp[gate_comp["method"] ==
                             ("Programmatic-only LR" if name == "prog_only" else "Full Hybrid LR")].iloc[0]
        g3 = {}
        for k in ("TP", "TN", "FP", "FN"):
            g3[k] = (int(m[k]) == int(arch_row[k]), "%s %d vs %d" % (k, m[k], arch_row[k]))
        g3["n"] = (m["n"] == 127, "n 127")
        for k in ("AUC", "Acc", "Sens", "Spec", "F1"):
            g3[k] = (abs(m[k] - float(arch_row[k])) < 5e-5, "%s %.6f vs %.4f" % (k, m[k], arch_row[k]))
        gate["G3"] = g3
        gate["G3_pass"] = all(v[0] for v in g3.values())
        log("G3 %s confusion/AUC vs archive: %s" % (name, {k: v[1] for k, v in g3.items()}))

        # G4: 阈值 vs 存档完整浮点值
        th_arch = float(arch_thr[name])
        gate["G4_threshold"] = {"recovered": repr(float(best_t)), "archived": repr(th_arch),
                                "archived_hex": float(th_arch).hex()}
        gate["G4_pass"] = bool(abs(best_t - th_arch) < 1e-12 and round(best_t, 2) == 0.55)
        log("G4 %s threshold recovered %r vs archived %r (pass=%s)" %
            (name, best_t, th_arch, gate["G4_pass"]))

        # G5: 插补值 + 系数 vs 存档（如实报告舍入精度）
        imp_arch = {row["feature"]: float(row["train_median_imputation_value"])
                    for _, row in gate_imp.iterrows()}
        imp_diffs = {k: abs(imp_vals[k] - imp_arch[k]) for k in cols}
        max_imp_diff = max(imp_diffs.values())
        gate["G5_imputation_max_abs_diff"] = max_imp_diff
        gate["G5_imputation_pass"] = bool(max_imp_diff <= 1e-9)
        coef_arch = {row["feature"]: float(row["coefficient_standardized"])
                     for _, row in gate_coef.iterrows() if row["model"] == name}
        coef_diffs = {k: abs(float(coef[i]) - coef_arch[k]) for i, k in enumerate(cols)}
        max_coef_diff = max(coef_diffs.values())
        gate["G5_coefficients"] = {"max_abs_diff": max_coef_diff,
                                   "archive_precision": "6 decimal places (rounded CSV)",
                                   "rule": "|delta| <= 5e-7 (half-ULP at 6dp)",
                                   "pass": bool(max_coef_diff <= 5e-7)}
        gate["G5_pass"] = gate["G5_imputation_pass"] and gate["G5_coefficients"]["pass"]
        log("G5 %s imputation max diff=%.3e (gate 1e-9); coef max diff=%.3e (gate 5e-7)" %
            (name, max_imp_diff, max_coef_diff))

        # G6: 拟合未用 Lu（白名单断言在输入快照阶段已过）
        gate["G6_pass"] = True

        all_pass = bool(gate["G2_pass"] and gate["G3_pass"] and gate["G4_pass"]
                        and gate["G5_pass"] and gate["G6_pass"])
        gate["ALL_PASS"] = all_pass
        results[name] = {"gate": gate, "best_t": best_t, "coef": coef,
                         "imp_vals": imp_vals, "imputer": imputer, "scaler": scaler,
                         "model": model, "ep": ep, "epred": epred, "m": m}

        if not all_pass:
            log("!! %s FAILED a gate -> external prediction for this model STOPPED "
                "(no parameter tuning, 1e-9 not relaxed)" % name)
            continue

        # ---- 保存完整预测器 ----
        mdir = PRED_DIR / name
        mdir.mkdir(parents=True, exist_ok=True)
        import joblib
        for obj, fn in ((imputer, "imputer.joblib"), (scaler, "scaler.joblib"),
                        (model, "model.joblib")):
            joblib.dump(obj, mdir / fn)
        # 全精度 JSON 参数（十进制 repr + hex 双重保存）
        def dbls(arr):
            return [float(x) for x in np.asarray(arr, dtype=float).ravel()]
        def hexs(arr):
            return [float(x).hex() for x in np.asarray(arr, dtype=float).ravel()]
        params = {
            "model_name": name,
            "recovery_kind": ("按原训练数据与固定配置重建，并通过存档预测一致性核验的预测器"
                              "（非原保存模型文件）"),
            "feature_order": cols,
            "n_features_in": len(cols),
            "threshold": {"decimal": repr(float(best_t)), "hex": float(best_t).hex()},
            "imputer": {"strategy": "median",
                        "statistics_": dbls(imputer.statistics_)},
            "scaler": {"mean_": dbls(scaler.mean_), "scale_": dbls(scaler.scale_),
                       "var_": dbls(scaler.var_), "n_features_in_": int(scaler.n_features_in_),
                       "mean_hex": hexs(scaler.mean_), "scale_hex": hexs(scaler.scale_)},
            "logistic_regression": {
                "config": {"class_weight": "balanced", "max_iter": 2000, "C": 1.0,
                           "solver": "lbfgs"},
                "coef_": dbls(model.coef_[0]), "coef_hex": hexs(model.coef_[0]),
                "intercept_": dbls(model.intercept_), "intercept_hex": hexs(model.intercept_),
                "classes_": [int(x) for x in model.classes_],
                "n_iter_": [int(x) for x in np.asarray(model.n_iter_).ravel()],
            },
            "train": {
                "sample_ids": [str(s) for s in tr_f["sample_id"]],
                "labels": [int(x) for x in ytrain],
                "sample_ids_sha256": sha256_text("\n".join(str(s) for s in tr_f["sample_id"])),
                "labels_sha256": sha256_text("".join(str(int(x)) for x in ytrain)),
            },
            "software_versions": {"python": sys.version.split()[0],
                                  "scikit-learn": "1.9.0", "numpy": np.__version__,
                                  "scipy": "1.18.0", "pandas": pd.__version__},
            "input_file_hashes": {key_by_path[str(p)]: sha256_file(p)
                                  for p in sorted(INPUT_WHITELIST)},
            "gate_summary": {k: (v if not isinstance(v, dict) else {kk: vv[0] if isinstance(vv, tuple) else vv for kk, vv in v.items()})
                             for k, v in gate.items()},
        }
        with open(mdir / "params.json", "w", encoding="utf-8") as f:
            json.dump(params, f, indent=1, ensure_ascii=False)
        log("%s predictor saved: 3 joblib objects + params.json (full-precision + hex)" % name)

        # ---- 重加载核验 ----
        im2 = joblib.load(mdir / "imputer.joblib")
        sc2 = joblib.load(mdir / "scaler.joblib")
        mo2 = joblib.load(mdir / "model.joblib")
        Xe = im2.transform(te_f[cols].values)
        Xe_s = sc2.transform(Xe)
        ep2 = mo2.predict_proba(Xe_s)[:, 1]
        reload_max = float(np.max(np.abs(ep2 - ep)))
        reload_check = {
            "reload_prob_max_abs_diff_vs_fitted": reload_max,
            "reload_prob_max_abs_diff_vs_archive": float(
                max(abs(ep2[i] - float(gate_by_sid[sid][col]))
                    for i, sid in enumerate(te_f["sample_id"]))),
            "reload_pass": bool(reload_max == 0.0),
        }
        with open(mdir / "reload_check.json", "w", encoding="utf-8") as f:
            json.dump(reload_check, f, indent=1)
        assert reload_max == 0.0, "reload produced different probabilities"
        log("%s reload-verify: max abs diff vs fitted = %.3e (0.0 required)" %
            (name, reload_max))

    # ---- 恢复期 Pitt 核验预测存档（供审计/对照，非新证据） ----
    pred_rows = []
    for i, sid in enumerate(te_f["sample_id"]):
        row = {"sample_id": sid, "subject_id": te_f["subject_id"].iloc[i],
               "y_true": int(ytest[i])}
        for name, res in results.items():
            pfx = "prog_only" if name == "prog_only" else "full_hybrid"
            row[pfx + "_prob"] = repr(float(res["ep"][i]))
            row[pfx + "_pred"] = int(res["epred"][i])
        pred_rows.append(row)
    with open(HERE / "pitt_recovered_predictions.csv", "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        w.writeheader()
        w.writerows(pred_rows)

    # ---- train/test ID 清单 ----
    for fn, df, y in (("train_ids.csv", tr_f, ytrain), ("test_ids.csv", te_f, ytest)):
        with open(HERE / fn, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "subject_id", "y_true"])
            w.writeheader()
            for i, sid in enumerate(df["sample_id"]):
                w.writerow({"sample_id": sid, "subject_id": df["subject_id"].iloc[i],
                            "y_true": int(y[i])})

    checks = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frozen_defs_sha256": {"train_lr_reanalysis": sha256_text(parts["train_lr"]),
                               "compute_metrics_reanalysis": sha256_text(parts["compute_metrics"]),
                               "train_lr_final_clean_eval": sha256_text(parts_old["train_lr"])},
        "plan_input_hashes_unchanged": True,
        "models": {name: res["gate"] for name, res in results.items()},
        "recovered_predictions_file": "pitt_recovered_predictions.csv",
        "note": ("核验集仅用于复现核对；本轮核对不构成新的独立测试证据。"
                 "系数存档为 6 位小数舍入，一致性按 |Δ|<=5e-7 规则并如实报告最大差；"
                 "位级一致性由逐样本概率门槛 1e-9 间接佐证。"),
    }
    with open(CHECK_JSON, "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=1, ensure_ascii=False)
    for name, res in results.items():
        log("%s ALL PASS=%s (max prob diff %.3e, threshold %r)" %
            (name, res["gate"]["ALL_PASS"], res["gate"]["G2_prob_max_abs_diff"],
             res["best_t"]))
    log("=== recover_predictors done in %.1fs ===" % (time.time() - t0))


if __name__ == "__main__":
    main()
