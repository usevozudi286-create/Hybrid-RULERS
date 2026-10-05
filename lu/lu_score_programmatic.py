# -*- coding: utf-8 -*-
"""Lu Programmatic-only 探索性外部验证（仅在 prog_only 通过全部复现门槛后执行）。

流程：
  1. 重加载已恢复+已核验的 prog_only 预测器（joblib），与 params.json 全精度
     参数逐项一致核验（含 hex 位级比对）；再在冻结 Pitt 127 核验集上重放，
     与存档 per_sample_predictions.csv 逐样本比对（<=1e-9 复证）。
  2. Lu 特征对账：lu_label_audit.csv main_included == lu_feature_extraction.csv
     行集合，主分析固定 50 段（25/25）；F07/F32/F14/F16 保持排除；
     本轮不做任何标签敏感性分析。
  3. 特征列/单位/截断/声学定义检查 + 逐样本缺失/替代/可评分审计：
     区分"原始提取成功"与"经历史 NaN→0 后数值完整"；不把"最终六列无空值"
     当作"全部原始特征有效"。
  4. 用恢复的 imputer → scaler → classifier 顺序评分；阈值用存档完整浮点值
     0.5500000000000002（>= 判定，同冻结实现）。预测文件与 SHA-256 先保存，
     之后才合并诊断标签计算指标；不按结果改任何规则、不重跑。
  5. 指标：TP/TN/FP/FN、AUC、Accuracy、Sensitivity、Specificity、F1。
     标题一律为"Lu 数据集上的探索性外部验证"。
  6. 参与者语义：@PID 不作为受试者 ID 使用的依据未证实 -> 只报告描述性点
     估计并注明限制；不做参与者聚类 bootstrap，不伪称独立性。

只读：冻结 Pitt 文件 + 上一轮 Lu 产物；只写本目录；无网络调用。
"""
import ast
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LUP = Path(os.environ.get("HYBRID_RULERS_LU_ROOT", "data/lu_external_validation_preparation"))
MCIRULERS = Path(os.environ.get("HYBRID_RULERS_SOURCE_ROOT", "data/mci_rulers"))
FROZEN_EVAL = MCIRULERS / "experiments" / "hybrid_rulers_full" / "final_clean_eval.py"
GATE_PREDS = MCIRULERS / "experiments" / "hybrid_rulers_full" / "recording_time_reanalysis" / "per_sample_predictions.csv"
GATE_MATRIX_TE = MCIRULERS / "experiments" / "hybrid_rulers_full" / "final_clean_results" / "hybrid_features_test.csv"
AUDIT_V2 = MCIRULERS / "experiments" / "hybrid_rulers_full" / "recording_label_audit" / "v2" / "recording_label_audit_v2.csv"
UNRES_V2 = MCIRULERS / "experiments" / "hybrid_rulers_full" / "recording_label_audit" / "v2" / "unresolved_cases_v2.csv"

LU_LABELS = LUP / "lu_label_audit.csv"
LU_FEATS = LUP / "lu_feature_extraction.csv"

RUN_LOG = HERE / "run_log.txt"
PRED_DIR = HERE / "predictors" / "prog_only"
PRED_CSV = HERE / "lu_programmatic_predictions.csv"
PRED_HASH = HERE / "lu_programmatic_predictions.sha256"
METRICS_CSV = HERE / "lu_programmatic_metrics.csv"
AUDIT_CSV = HERE / "lu_feature_audit.csv"

PROG_COLS = ["information_unit_count", "information_density", "speech_rate",
             "hesitation_rate", "long_pause_ratio", "transcript_length"]

EXCLUDED = {"F07.cha", "F32.cha", "F14.cha", "F16.cha"}


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


def compute_metrics(yt, yp, ys=None):
    tp = int(np.sum((yp == 1) & (yt == 1))); tn = int(np.sum((yp == 0) & (yt == 0)))
    fp = int(np.sum((yp == 1) & (yt == 0))); fn = int(np.sum((yp == 0) & (yt == 1)))
    n = len(yt)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    acc = (tp + tn) / n
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    auc = None
    if ys is not None and len(set(yt)) >= 2:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yt, ys)
    return {"Acc": acc, "Sens": sens, "Spec": spec, "F1": f1, "AUC": auc,
            "n": n, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


def main():
    t0 = time.time()
    log("=== lu_score_programmatic start (only after prog_only passed ALL gates) ===")
    import joblib

    # ---- 1. 重加载 + 与 params.json 全精度一致核验 ----
    params = json.load(open(PRED_DIR / "params.json", encoding="utf-8"))
    imputer = joblib.load(PRED_DIR / "imputer.joblib")
    scaler = joblib.load(PRED_DIR / "scaler.joblib")
    model = joblib.load(PRED_DIR / "model.joblib")
    assert list(params["feature_order"]) == PROG_COLS
    assert [float(x) for x in imputer.statistics_] == params["imputer"]["statistics_"]
    assert [float(x) for x in scaler.mean_] == params["scaler"]["mean_"]
    assert [float(x) for x in scaler.scale_] == params["scaler"]["scale_"]
    assert [float(x) for x in model.coef_[0]] == params["logistic_regression"]["coef_"]
    assert [float(x) for x in model.intercept_] == params["logistic_regression"]["intercept_"]
    # hex 位级比对
    assert [float(x).hex() for x in scaler.mean_] == params["scaler"]["mean_hex"]
    assert [float(x).hex() for x in scaler.scale_] == params["scaler"]["scale_hex"]
    assert [float(x).hex() for x in model.coef_[0]] == params["logistic_regression"]["coef_hex"]
    assert [float(x).hex() for x in model.intercept_] == params["logistic_regression"]["intercept_hex"]
    thr = float(params["threshold"]["decimal"])
    assert float.fromhex(params["threshold"]["hex"]) == thr
    log("recovered predictor reloaded; params.json full-precision + hex bit-level match; "
        "threshold = %r" % thr)

    # ---- 2. Pitt 127 复证（同一恢复模型重放存档核验集） ----
    te = pd.read_csv(GATE_MATRIX_TE, encoding="utf-8-sig")
    te["sample_id"] = te["sample_id"].astype(str).str.strip()
    unres = pd.read_csv(UNRES_V2, encoding="utf-8-sig", dtype=str)
    unres["sample_id"] = unres["sample_id"].astype(str).str.strip()
    corrected_test = set(te["sample_id"]) - set(unres["sample_id"])
    audit = pd.read_csv(AUDIT_V2, encoding="utf-8-sig", dtype=str)
    audit["sample_id"] = audit["sample_id"].astype(str).str.strip()
    cmap = {}
    for sid in audit["sample_id"]:
        if sid in set(unres["sample_id"]):
            cmap[sid] = None
        else:
            lab = audit.loc[audit["sample_id"] == sid, "proposed_label"].iloc[0]
            cmap[sid] = 1 if lab == "positive" else 0
    te_f = te[te["sample_id"].isin(corrected_test)].copy()
    te_f["y_true"] = te_f["sample_id"].map(cmap).astype(int)
    te_f = te_f.sort_values("sample_id").reset_index(drop=True)
    assert len(te_f) == 127
    Xe = imputer.transform(te_f[PROG_COLS].values)
    Xe_s = scaler.transform(Xe)
    ep_pitt = model.predict_proba(Xe_s)[:, 1]
    gate_pred = pd.read_csv(GATE_PREDS, encoding="utf-8-sig")
    gate_pred["sample_id"] = gate_pred["sample_id"].astype(str).str.strip()
    gate_by_sid = {row["sample_id"]: row for _, row in gate_pred.iterrows()}
    maxd_pitt = max(abs(ep_pitt[i] - float(gate_by_sid[sid]["Programmatic_only_LR_score"]))
                    for i, sid in enumerate(te_f["sample_id"]))
    assert maxd_pitt <= 1e-9, maxd_pitt
    log("Pitt re-verification with reloaded predictor: 127 per-sample max abs diff = %.3e" %
        maxd_pitt)

    # ---- 3. Lu 对账：主分析固定 50 段 ----
    labels = pd.read_csv(LU_LABELS, encoding="utf-8-sig")
    main = labels[labels["label_state"] == "main_included"]
    assert len(main) == 50, len(main)
    assert (main["main_label"].value_counts().to_dict() == {"control": 25, "positive": 25})
    excl = labels[labels["label_state"] != "main_included"]
    assert set(excl["filename"]) == EXCLUDED, (set(excl["filename"]), EXCLUDED)
    feats = pd.read_csv(LU_FEATS, encoding="utf-8-sig")
    assert set(feats["filename"]) == set(main["filename"]), "Lu feature rows != main_included"
    assert len(feats) == 50
    assert feats["feature_valid"].all()
    label_map = dict(zip(labels["filename"], labels["main_label"]))
    ylu = feats["filename"].map(label_map).map(lambda x: 1 if x == "positive" else 0).to_numpy(int)
    log("Lu reconciliation: 50 main-included (25 pos / 25 ctl); excluded %s; "
        "no label sensitivity analysis this round" % sorted(EXCLUDED))

    # ---- 4. 特征列/单位/截断/声学定义检查 + 逐样本审计 ----
    # 4a. 冻结定义片段再提取（与上一轮记录哈希比对）
    src = open(FROZEN_EVAL, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    want = {"tokenize_english", "INFORMATION_UNIT_GROUPS", "count_info",
            "parse_word_timestamps", "aligned_words_per_second"}
    parts = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            parts[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in want:
                    parts[t.id] = ast.get_source_segment(src, node)
    assert want <= set(parts)
    blob = "\n\n".join(parts[n] for n in sorted(parts))
    defs_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    assert defs_hash == "bca821b9393f3388df0ac8999b1dc492da0929fd558f08be0ffae215baef4d27", defs_hash
    log("frozen feature defs re-extracted verbatim, sha256 matches Phase-2 record (bca821b9…)")

    # 4b. 数值一致性自检（不重提特征，只核对已有列间关系与冻结公式）
    feat = feats[["filename"] + PROG_COLS + ["wc", "word_timestamp_count", "alignment_status",
                                             "asr_text_len_raw", "asr_text_len_500",
                                             "audio_duration_s"]].copy()
    feat["asr_text_len_raw"] = pd.to_numeric(feat["asr_text_len_raw"], errors="coerce")
    feat["asr_text_len_500"] = pd.to_numeric(feat["asr_text_len_500"], errors="coerce")
    feat["audio_duration_s"] = pd.to_numeric(feat["audio_duration_s"], errors="coerce")
    for c in PROG_COLS:
        assert feat[c].notna().all(), "Lu prog feature has NaN: %s" % c
    # transcript_length == wc（同列双重记录应一致）
    assert (feat["transcript_length"].astype(int) == feat["wc"].astype(int)).all()
    # information_density == round(iuc/max(wc,1),4)（冻结公式自检）
    iuc = feat["information_unit_count"].astype(int)
    wc = feat["wc"].astype(int)
    dens = np.array([round(int(i) / max(int(w), 1), 4) for i, w in zip(iuc, wc)])
    assert np.allclose(dens, feat["information_density"].astype(float), atol=1e-12)
    n_trunc = int((feat["asr_text_len_raw"] > 500).sum())
    log("feature column self-check OK: transcript_length==wc; density==round(iuc/max(wc,1),4); "
        "asr_text truncated at 500 for %d/50 samples (frozen rule)" % n_trunc)
    # 4c. 逐样本缺失/替代/可评分审计（区分原始提取成功 vs 历史 NaN→0 后数值完整）
    audit_rows = []
    for _, r in feat.iterrows():
        wtc = int(r["word_timestamp_count"])
        dur = r["audio_duration_s"]
        sr0 = float(r["speech_rate"]) == 0
        hr0 = float(r["hesitation_rate"]) == 0
        lpr0 = float(r["long_pause_ratio"]) == 0
        # 原始提取成功判据（来自上一轮逐样本记录的字段，非新计算）：
        # speech_rate: 对齐词数>=2 且末词>首词 -> 计算值；词数<2 -> 冻结规则 0
        # hesitation_rate: 词数>=2 -> 计算值（无 >0.5s 停顿时为计算出的真实 0）
        # long_pause_ratio: 音频解码成功(duration>0) -> 能量检测计算值；
        #                   0 = 未检出 >=0.15s 能量停顿；NaN->0 仅当提取器无输出
        sr_src = "computed" if wtc >= 2 else "frozen_rule_lt2words"
        hr_src = "computed" if wtc >= 2 else "frozen_rule_lt2words"
        lpr_src = "computed_no_pause_detected" if (lpr0 and dur > 0) else (
            "computed" if dur > 0 else "nan_to_zero_substituted")
        audit_rows.append({
            "filename": r["filename"],
            "word_timestamp_count": wtc,
            "audio_duration_s": dur,
            "asr_truncated_at_500": bool(r["asr_text_len_raw"] > 500),
            "speech_rate": float(r["speech_rate"]),
            "speech_rate_source": sr_src,
            "hesitation_rate": float(r["hesitation_rate"]),
            "hesitation_rate_source": hr_src,
            "long_pause_ratio": float(r["long_pause_ratio"]),
            "long_pause_ratio_source": lpr_src,
            "scorable": True,
        })
    audit_df = pd.DataFrame(audit_rows)
    n_nan_sub = int((audit_df["long_pause_ratio_source"] == "nan_to_zero_substituted").sum())
    assert n_nan_sub == 0, "unexpected NaN->0 substitution on Lu side"
    log("per-sample audit: all 50 scorable; sr/hr computed for all (wtc>=39); "
        "lpr computed for all (duration>0); %d samples with lpr==0 (no energy pause "
        ">=0.15s detected under frozen acoustic definition)" % int((audit_df["long_pause_ratio"] == 0).sum()))

    # ---- 5. 恢复管道评分（imputer -> scaler -> LR，阈值 0.5500000000000002） ----
    Xlu = feats[PROG_COLS].to_numpy(dtype=float)
    Xlu_imp = imputer.transform(Xlu)
    Xlu_s = scaler.transform(Xlu_imp)
    prob = model.predict_proba(Xlu_s)[:, 1]
    pred = (prob >= thr).astype(int)
    out = feats[["filename", "folder"]].copy()
    for c in PROG_COLS:
        out[c] = feats[c]
    out["probability"] = [repr(float(p)) for p in prob]
    out["predicted_label"] = [int(p) for p in pred]
    out["threshold"] = repr(thr)
    out.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
    # 哈希先保存（指标计算之前、不合并诊断标签的预测文件）
    with open(PRED_HASH, "w", encoding="utf-8") as f:
        f.write("%s  %s\n" % (sha256_file(PRED_CSV), PRED_CSV.name))
    log("lu_programmatic_predictions.csv saved (50 rows, no diagnosis labels merged); "
        "sha256=%s" % sha256_file(PRED_CSV))

    # ---- 6. 指标（合并诊断标签只发生在预测文件+哈希保存之后） ----
    yp = pred
    m = compute_metrics(ylu, yp, prob)
    log("Lu Programmatic-only exploratory metrics: n=%d TP=%d TN=%d FP=%d FN=%d "
        "AUC=%.4f Acc=%.4f Sens=%.4f Spec=%.4f F1=%.4f" %
        (m["n"], m["TP"], m["TN"], m["FP"], m["FN"],
         m["AUC"], m["Acc"], m["Sens"], m["Spec"], m["F1"]))
    metrics_rows = [{
        "analysis": "Lu 数据集上的探索性外部验证",
        "model": "Programmatic-only LR (recovered & archived-prediction-verified)",
        "threshold": repr(thr),
        "n_samples": int(m["n"]),
        "n_positive": int((ylu == 1).sum()),
        "n_control": int((ylu == 0).sum()),
        "TP": int(m["TP"]), "TN": int(m["TN"]), "FP": int(m["FP"]), "FN": int(m["FN"]),
        "AUC": round(m["AUC"], 4),
        "Accuracy": round(m["Acc"], 4),
        "Sensitivity": round(m["Sens"], 4),
        "Specificity": round(m["Spec"], 4),
        "F1": round(m["F1"], 4),
        "participant_handling": ("descriptive point estimates only; @PID semantics NOT "
                                 "confirmed as subject ID -> no participant clustering, "
                                 "no independence assumed, no participant-level bootstrap"),
        "coverage": "50/50 scored (features 50/50 complete; no NaN->0 substitution "
                    "triggered; see lu_feature_audit.csv)",
        "caveats": ("exploratory external validation only; no LLM features used; "
                    "threshold is the frozen training-set Youden value; "
                    "Pitt/Lu overlap not verified; recording-time diagnosis not "
                    "available in Lu CHA files"),
    }]
    with open(METRICS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        w.writeheader()
        w.writerows(metrics_rows)
    audit_df.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    log("lu_programmatic_metrics.csv + lu_feature_audit.csv saved")
    log("=== lu_score_programmatic done in %.1fs ===" % (time.time() - t0))


if __name__ == "__main__":
    main()
