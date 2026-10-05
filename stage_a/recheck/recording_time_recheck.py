# -*- coding: utf-8 -*-
"""Task C Phase 2 — closing audit (收尾核查) of recording_time_reanalysis/.

Six tasks, all offline (no API calls, no ASR/LLM regeneration), all outputs in
this NEW directory; nothing in recording_time_reanalysis/ or any other frozen
location is modified.

  T1  Fix the 280-0/1/2 ID-prefix error in the previous report. Generate a
      per-record prefix audit from ACTUAL records (metadata sample_id /
      audio_path / source_group and the checkpoint records), distinguishing
      original directory grouping vs old analysis label vs recording-time
      label. Statistics are split by Direct / RULERS branch and repeat index;
      the two branches' record counts are never summed into "Direct calls".
  T2  Reclassify Direct as a "diagnosis-related ID-exposed historical
      baseline", removed from the formal leak-free comparison. Its AUC and
      every Direct-involving comparison appear ONLY in a clearly marked
      historical/diagnostic appendix. No inference is made about the
      direction or magnitude of the exposure effect.
  T3  Re-audit Weighted RULERS invalid repeats and the all-NA -> 0 problem.
      Report how many records / samples are affected in train and test.
      Apply a PRE-SPECIFIED validity rule for weighted scoring: reject
      non-finite scores; record MISSING (not 0) when a row has no valid
      entries. Recompute affected training weights, aggregated scores and
      evaluation. Report before/after differences. No zero-filling to keep
      127; if coverage changes, report a fair comparison on common samples.
  T4  New exploratory ablation: Programmatic-only + demographics LR, using
      the existing train/test IDs, demographic handling, imputation,
      scaling, LR hyperparameters and train-set threshold protocol. Compare
      with Full Hybrid + demographics via paired participant-clustered
      bootstrap AUC difference with 95% CI. Explicitly post-hoc; no tuning
      from results.
  T5  PREPARE (but do not execute) a clean Direct re-run plan: strip all
      diagnosis-related IDs / directories / paths / metadata, use opaque
      label-agnostic IDs with the mapping kept local only; inspect the
      complete actual request content; pin model version, prompt,
      temperature, repeats, validity and failure rules; if the original
      model snapshot cannot be matched, state the model-change confound and
      do NOT attribute old-vs-new differences solely to ID removal; give
      estimated calls, cost and the specific items needing author
      confirmation.
  T6  Deliverables: revised report, impact list, exploratory ablation
      results, Direct re-run plan. Original data, old results and the paper
      remain untouched.

Determinism gates (all vs the frozen recording_time_reanalysis/ artifacts):
  * LR re-fits (llm/prog/hybrid) and thresholds reproduce archived values.
  * Prompt-RULERS corrected-train Youden threshold reproduces 0.07 / J.
  * The OLD weighted-RULERS aggregation (incl. all-NA -> 0.0) is replicated
    bit-for-bit against the archived per-sample scores.
  * The main bootstrap Hybrid - Programmatic diff reproduces the archived
    interval (same frame, same seed, same RNG draw sequence).
"""
import hashlib
import json
import math
import os
import re
import sys
import uuid
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
REANALYSIS = HYBRID_FULL / "recording_time_reanalysis"  # frozen, read-only
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
DIRECT_BASELINE_SRC = EXPERIMENTS / "direct_llm_baseline.py"
RUN_CFG = EXPERIMENTS / "stability_audio_en_v4flash_full_test" / "run_config.json"

# Frozen gate artifacts (read-only inputs of this recheck)
GATE_WEIGHTS = REANALYSIS / "learned_weights_corrected_train.json"
GATE_PREDS = REANALYSIS / "per_sample_predictions.csv"
GATE_COMP = REANALYSIS / "strategy_comparison.csv"
GATE_BOOT = REANALYSIS / "clustered_bootstrap_auc_diffs.csv"
GATE_DEMO_PREDS = REANALYSIS / "demographic" / "demographic_predictions_test.csv"

FREEZE = HERE / "freeze"
ABL = HERE / "ablation_prog_demo"
for d in (FREEZE, ABL):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Frozen analysis configuration (verbatim from recording_time_reanalysis.py)
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
# Verbatim functions from final_clean_eval.py / recording_time_reanalysis.py
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
    """SHA-256 of every authoritative input consumed by this recheck."""
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
    add(RUN_CFG, "experiments/stability_audio_en_v4flash_full_test/run_config.json")
    add(RUBRIC_PATH, "mci_rubric_audio_en_v1.json")
    add(OLD_WEIGHTS, "experiments/literature_weighted_rulers_en/learned_weights.json")
    add(OLD_WEIGHTED_SCORES, "experiments/literature_weighted_rulers_en/weighted_scores_test.csv")
    add(DIRECT_BASELINE_SRC, "experiments/direct_llm_baseline.py")
    for d in PILOT_DIRS:
        for name in ("results.csv", "results_checkpoint.jsonl"):
            add(EXPERIMENTS / d / name, "pilot_dirs/%s/%s" % (d, name))
    add(SMOKE_DIR / "results.csv", "smoke/stability_smoke_test/results.csv")
    for src in (
        "experiments/stability_experiment.py",
        "experiments/literature_weighted_rulers_en/weighted_eval.py",
        "experiments/hybrid_rulers_full/final_clean_eval.py",
        "mci_rulers_core.py",
        "mci_provider.py",
    ):
        add(PROJECT / src, "source/" + src)
    # frozen gate artifacts from recording_time_reanalysis (read-only)
    for gate, key in (
        (GATE_WEIGHTS, "recording_time_reanalysis/learned_weights_corrected_train.json"),
        (GATE_PREDS, "recording_time_reanalysis/per_sample_predictions.csv"),
        (GATE_COMP, "recording_time_reanalysis/strategy_comparison.csv"),
        (GATE_BOOT, "recording_time_reanalysis/clustered_bootstrap_auc_diffs.csv"),
        (GATE_DEMO_PREDS, "recording_time_reanalysis/demographic/demographic_predictions_test.csv"),
    ):
        add(gate, key)
    return targets


def prefix_of(sid):
    if sid.startswith("pitt_dementia"):
        return "dementia"
    if sid.startswith("pitt_control"):
        return "control"
    return "other"


def path_group_of(audio_path):
    """Extract the Control/Dementia directory segment from the actual audio path."""
    if not isinstance(audio_path, str):
        return ""
    for part in re.split(r"[\\/]+", audio_path):
        if part.lower() in ("control", "dementia"):
            return part.lower()
    return ""


# ===========================================================================
# Pre-specified Weighted-RULERS validity rule (frozen BEFORE computation)
# ===========================================================================
def checklist_items_new(oj):
    """Return (valid_items, n_na, n_invalid) under the frozen recheck rule.

    Valid item score: an int or float that is NOT a bool and is math.isfinite.
    Everything else ('NA', None, strings, bools, NaN, inf) is invalid.
    """
    valid, n_na, n_invalid = [], 0, 0
    for item in oj.get("checklist", []):
        cid = item.get("criterion_id", "")
        s = item.get("score")
        if s == "NA" or s is None:
            n_na += 1
            continue
        if isinstance(s, bool):
            n_invalid += 1
            continue
        if isinstance(s, (int, float)):
            try:
                f = float(s)
                if math.isfinite(f):
                    valid.append((cid, f))
                else:
                    n_invalid += 1
            except Exception:
                n_invalid += 1
            continue
        n_invalid += 1
    return valid, n_na, n_invalid


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = datetime.now()
    log("=== recording_time_reanalysis closing audit (recheck) ===")
    log("started: %s" % t0.strftime("%Y-%m-%d %H:%M:%S"))

    # ---------------------------------------------------------------
    # 1. Input integrity snapshot (BEFORE any analysis)
    # ---------------------------------------------------------------
    log("\n== [1] INPUT SNAPSHOT (before) ==")
    hashes_before = snapshot_inputs()
    pd.DataFrame([{"input_key": k, "sha256": v} for k, v in sorted(hashes_before.items())]
                 ).to_csv(FREEZE / "input_hashes_before.csv", index=False,
                          encoding="utf-8-sig")
    log("hashed %d authoritative inputs" % len(hashes_before))

    # ---------------------------------------------------------------
    # 2. Load Phase-1 v2 audit artifacts + metadata (read-only)
    # ---------------------------------------------------------------
    log("\n== [2] AUDIT ARTIFACTS ==")
    audit = pd.read_csv(AUDIT_V2, encoding="utf-8-sig", dtype=str)
    cand = pd.read_csv(CAND_V2, encoding="utf-8-sig", dtype=str)
    unres = pd.read_csv(UNRES_V2, encoding="utf-8-sig", dtype=str)
    for df, name in ((audit, "audit"), (cand, "cand"), (unres, "unres")):
        df["sample_id"] = df["sample_id"].astype(str).str.strip()
    assert len(audit) == 492, "audit rows %d != 492" % len(audit)
    for col in ["subject_id", "split", "old_label", "proposed_label", "differs_from_old",
                "unresolved_reason", "unresolved_bucket", "evidence_class",
                "evidence_grade", "old_label_source"]:
        assert col in audit.columns, "audit missing column %s" % col
    assert len(cand) == 19, "candidates %d != 19" % len(cand)
    assert len(unres) == 40, "unresolved %d != 40" % len(unres)

    audit["subject_id"] = audit["subject_id"].astype(str).str.zfill(3)
    if "visit" in audit.columns:
        audit["visit"] = audit["visit"].astype(str)
    candidates = set(cand["sample_id"])
    unresolved = set(unres["sample_id"])
    resolved = set(audit["sample_id"]) - unresolved
    assert len(resolved) == 452
    assert set(audit.loc[audit["differs_from_old"] == "1", "sample_id"]) == candidates
    assert (audit.loc[audit["sample_id"].isin(candidates), "proposed_label"] == "control").all()
    non_cand_resolved = resolved - candidates
    assert (audit.loc[audit["sample_id"].isin(non_cand_resolved), "proposed_label"] ==
            audit.loc[audit["sample_id"].isin(non_cand_resolved), "old_label"]).all()
    log("audit 492 | candidates 19 (all positive->control) | unresolved 40 | resolved 452")

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

    excl_train = pd.read_csv(FINAL / "excluded_train_records.csv", encoding="utf-8-sig")
    excl_test = pd.read_csv(FINAL / "excluded_records.csv", encoding="utf-8-sig")
    qc_train = set(excl_train["sample_id"].astype(str).str.strip())
    qc_test_rows_unique = set(excl_test["sample_id"].astype(str).str.strip())
    assert len(qc_train) == 2 and qc_train == predefined_train - feats_train_ids
    qc_test_full = predefined_test - feats_test_ids
    assert len(qc_test_full) == 12
    partial_failed = qc_test_rows_unique - qc_test_full
    assert partial_failed <= feats_test_ids
    train_used = feats_train_ids
    test_clean = feats_test_ids
    log("frozen QC: train 342-2=340 | test 150-12=138 | %d partial-repeat samples"
        % len(partial_failed))

    old_label_map = dict(zip(meta["sample_id"], meta["label"].map(
        lambda x: 1 if x == "positive" else 0)))
    assert (feats_train["y_true"].to_numpy() ==
            feats_train["sample_id"].map(old_label_map).to_numpy()).all()
    assert (feats_test["y_true"].to_numpy() ==
            feats_test["sample_id"].map(old_label_map).to_numpy()).all()

    # Plan-A inclusion sets (ID sets, verbatim)
    corrected_train = train_used - unresolved
    corrected_test = test_clean - unresolved
    assert len(corrected_train) == 311 and len(corrected_test) == 127
    cand_in_train = candidates & corrected_train
    cand_in_test = candidates & corrected_test
    cand_in_qc_test = candidates & qc_test_full
    assert len(cand_in_train) == 12 and len(cand_in_test) == 3 and len(cand_in_qc_test) == 4

    corrected_map = {}
    for sid in audit["sample_id"]:
        if sid in unresolved:
            corrected_map[sid] = None
        else:
            lab = audit.loc[audit["sample_id"] == sid, "proposed_label"].iloc[0]
            corrected_map[sid] = 1 if lab == "positive" else 0

    # ---------------------------------------------------------------
    # 3. FREEZE the recheck rules BEFORE any performance computation
    # ---------------------------------------------------------------
    log("\n== [3] FREEZING RECHECK RULES ==")
    rules_md = """# 收尾核查规则（在任何性能计算之前冻结）

## 总体约束
- 全部产物写入 recording_time_recheck/；recording_time_reanalysis/、原始数据、
  旧结果、审计 v1/v2、论文均只读（输入前后 SHA-256 比对）。
- 本轮不调用 API、不重新生成任何评分；复用既有缓存与冻结特征。
- 研究目标不变：录音时点认知状态识别。纳入/排除规则与 Phase-2 冻结规则完全一致
  （方案 A：采纳 19 条候选修正，排除 40 条未决）。本核查不改变任何标签或纳入规则。

## T1 前缀审计规则
- 前缀审计从实际记录逐条生成：三层独立证据 = ① metadata 的 sample_id 前缀；
  ② metadata 的 source_group（原目录分组）；③ audio_path 中的 Control/Dementia
  目录段。三层交叉一致性逐条断言，不一致如实报告。
- 每条记录区分三个标签口径：原目录分组 / 旧分析标签（metadata label）/
  录音时点标签（方案 A proposed_label）。
- 统计按 Direct / RULERS 分支与 repeat_idx 分开；两方法的记录数绝不合并成
  "Direct 调用次数"。Direct 调用次数 = Direct 记录数 = 150 样本 x 5 重复 = 750。

## T2 Direct 重新定性规则
- Direct 分支重新定性为"诊断相关 ID 暴露的历史基线"，移出正式无泄漏方法比较。
- Direct 的 AUC 与一切涉及 Direct 的比较（主表、bootstrap、开发接触、参与者级）
  只出现在明确标记的历史/诊断性附录。
- 对暴露影响的方向或大小不作任何推断。

## T3 Weighted RULERS 有效性规则（预先指定）
- 条目得分有效性：必须是 int 或 float、不是 bool、且 math.isfinite；
  'NA'/None/字符串/NaN/inf 一律无效，不计入分子与分母。
- 未知 criterion_id 沿用协议回退权重 1.0（与旧实现一致）。
- 一行 checklist 无有效条目时，该行加权分记为缺失（NaN），绝不补 0。
- 样本聚合 = 有效重复的中位数；样本无任何有效重复 → 样本缺失。
- 覆盖范围改变时不补零保 127；另行报告共同样本上的公平比较。
- 训练侧权重学习沿用原协议结构（全体训练 RULERS 行、无有效性过滤、
  条目均值/分歧率/条目 AUC 有用度、文献先验、[0.5,2] 截断、均值归一），
  仅条目有效性检查替换为上述新规则；旧权重逐项差输出。

## T4 探索性消融规则
- 事后新增分析（明确标记）。Programmatic-only + demographics（9 特征）LR，
  沿用现有训练/测试 ID、人口学处理（最早访视规则、基线年龄）、插补、标准化、
  LR 超参与训练集 Youden 阈值协议。与 Full Hybrid + demographics（15 特征）比较，
  报告配对参与者聚类 bootstrap AUC 差与 95% CI（20000 次、seed 42）。
- 不依据结果调参。

## T5 Direct 干净重跑方案
- 仅准备方案文档，不执行。

## 统计协议
- 配对参与者聚类 bootstrap：20000 次、seed 42、单一 RNG 流、同一次参与者抽取
  供框内全部方法；无效重采样（类别 <2）跳过并计数；差值 >0 比例为经验比例，
  不称 p 值或贝叶斯概率。
- 主框（127）：Prompt RULERS、LLM-only、Prog-only、Full Hybrid。
  共同样本框（123）：上述 + 新规则 Weighted RULERS。
  附录框（127）：Direct + Full Hybrid。
"""
    (FREEZE / "recheck_rules.md").write_text(rules_md, encoding="utf-8")

    # ---------------------------------------------------------------
    # 4. T1 — per-record ID prefix audit from ACTUAL records
    # ---------------------------------------------------------------
    log("\n== [4] T1 PREFIX AUDIT (from actual records) ==")
    audit_by_id = audit.set_index("sample_id")
    meta_by_id = meta.set_index("sample_id")
    prefix_rows = []
    cross_source_mismatch = []
    for sid in sorted(meta["sample_id"]):
        mrow = meta_by_id.loc[sid]
        arow = audit_by_id.loc[sid]
        sg = str(mrow["source_group"]).strip().lower()
        pref = prefix_of(sid)
        pg = path_group_of(mrow["audio_path"])
        if pref != sg or (pg and pg != sg):
            cross_source_mismatch.append((sid, pref, sg, pg))
        old_lab = str(mrow["label"]).strip()
        corr = corrected_map[sid]
        corr_lab = ("positive" if corr == 1 else "control") if corr is not None else "EXCLUDED"
        status = ("excluded_unresolved" if sid in unresolved else
                  ("candidate_relabeled" if sid in candidates else "unchanged"))
        prefix_rows.append({
            "sample_id": sid,
            "subject_id": subj_map[sid],
            "visit": str(mrow["visit"]),
            "split": str(mrow["split"]),
            "id_prefix": pref,
            "source_group": str(mrow["source_group"]).strip(),
            "audio_path_dir_segment": pg,
            "dir_group": sg,
            "old_label": old_lab,
            "recording_time_label": corr_lab,
            "audit_status": status,
            "prefix_vs_old_label": "match" if pref == ("dementia" if old_lab == "positive" else "control") else "mismatch",
            "prefix_vs_recording_time_label": "match" if (corr is not None and pref == ("dementia" if corr == 1 else "control")) else ("excluded" if corr is None else "mismatch"),
        })
    assert not cross_source_mismatch, "cross-source prefix mismatch: %s" % cross_source_mismatch[:5]
    pref_df = pd.DataFrame(prefix_rows)
    pref_df.to_csv(HERE / "id_prefix_audit_records.csv", index=False, encoding="utf-8-sig")
    log("492 records: id_prefix/source_group/audio_path three-way consistent")

    n_dem_dir = int((pref_df["dir_group"] == "dementia").sum())
    n_con_dir = int((pref_df["dir_group"] == "control").sum())
    control_old_pos = pref_df[(pref_df["dir_group"] == "control") &
                              (pref_df["old_label"] == "positive")]
    control_old_pos_non_cand = control_old_pos[~control_old_pos["sample_id"].isin(candidates)]
    log("dir group: dementia %d / control %d" % (n_dem_dir, n_con_dir))
    log("control-dir records with OLD positive label: %d (candidates %d, others %d)"
        % (len(control_old_pos), len(control_old_pos) - len(control_old_pos_non_cand),
           len(control_old_pos_non_cand)))
    for _, r in control_old_pos_non_cand.iterrows():
        log("  non-candidate: %s old=%s rec=%s status=%s" % (
            r["sample_id"], r["old_label"], r["recording_time_label"], r["audit_status"]))

    # 19-candidate three-way table
    cand_rows = []
    for _, r in pref_df[pref_df["sample_id"].isin(candidates)].iterrows():
        cand_rows.append({
            "sample_id": r["sample_id"], "subject_id": r["subject_id"],
            "split": r["split"],
            "dir_group": r["dir_group"], "id_prefix": r["id_prefix"],
            "old_label": r["old_label"],
            "recording_time_label": r["recording_time_label"],
            "prefix_vs_recording_time_label": r["prefix_vs_recording_time_label"],
        })
    cand_detail = pd.DataFrame(cand_rows)
    cand_detail.to_csv(HERE / "id_prefix_candidates_detail.csv", index=False,
                       encoding="utf-8-sig")
    assert (cand_detail["dir_group"] == "control").all()
    assert (cand_detail["old_label"] == "positive").all()
    assert (cand_detail["recording_time_label"] == "control").all()
    log("all 19 candidates: dir=control, old=positive, recording-time=control")

    # checkpoint-level audit (test branch records, method x repeat separated)
    cp_rows = list(iter_jsonl(TEST_CP))
    assert len(cp_rows) == 1500
    ck_rows = []
    for r in cp_rows:
        sid = str(r.get("sample_id", "")).strip()
        try:
            rs = float(r.get("risk_score", np.nan))
            finite = bool(np.isfinite(rs))
        except Exception:
            finite = False
        ck_rows.append({
            "sample_id": sid,
            "method": r.get("method"),
            "repeat_idx": r.get("repeat_idx"),
            "id_prefix": prefix_of(sid),
            "dir_group": prefix_of(sid) if prefix_of(sid) != "other" else
            (meta_by_id.loc[sid, "source_group"] if sid in meta_by_id.index else ""),
            "old_label": (str(meta_by_id.loc[sid, "label"]).strip()
                          if sid in meta_by_id.index else ""),
            "recording_time_label": ("positive" if corrected_map.get(sid) == 1 else
                                     "control" if corrected_map.get(sid) == 0 else "EXCLUDED"),
            "candidate_relabeled": "1" if sid in candidates else "0",
            "api_success": r.get("api_success"),
            "parse_success": r.get("parse_success"),
            "error_type": r.get("error_type"),
            "risk_score_finite": finite,
        })
    ck_df = pd.DataFrame(ck_rows)
    ck_df.to_csv(HERE / "id_prefix_audit_checkpoint_records.csv", index=False,
                 encoding="utf-8-sig")
    br_summary = ck_df.groupby(["method", "repeat_idx"]).size().reset_index(
        name="n_records")
    n_direct_records = int((ck_df["method"] == "direct").sum())
    n_rulers_records = int((ck_df["method"] == "rulers").sum())
    assert n_direct_records == 750 and n_rulers_records == 750
    log("checkpoint: Direct records=%d (150x5), RULERS records=%d (150x5); "
        "total 1500 is the two-branch record total, NOT a Direct call count"
        % (n_direct_records, n_rulers_records))

    sum_rows = br_summary.to_dict("records")
    sum_rows.append({"method": "direct", "repeat_idx": "ALL",
                     "n_records": int(n_direct_records),
                     "note": "Direct call count = 750 (never 1500)"})
    sum_rows.append({"method": "rulers", "repeat_idx": "ALL",
                     "n_records": int(n_rulers_records), "note": "RULERS records = 750"})
    sum_rows.append({"method": "BOTH", "repeat_idx": "ALL",
                     "n_records": int(n_direct_records + n_rulers_records),
                     "note": "two-branch total 1500; repeat-level exclusion log "
                             "final_clean_results/excluded_records.csv has 134 rows "
                             "(73 RULERS + 61 Direct records), NOT 134 calls"})
    excl_mc = excl_test["method"].value_counts().to_dict()
    sum_rows.append({"method": "excluded_records.csv", "repeat_idx": "per-branch",
                     "n_records": int(len(excl_test)),
                     "note": "repeat-level log; branch split RULERS=%s Direct=%s"
                             % (excl_mc.get("rulers"), excl_mc.get("direct"))})
    pd.DataFrame(sum_rows).to_csv(HERE / "id_prefix_audit_summary.csv", index=False,
                                  encoding="utf-8-sig")

    # T1 verification of the 280 claim against actual records
    for sid in ("pitt_control_cookie_280-0", "pitt_control_cookie_280-1",
                "pitt_control_cookie_280-2"):
        assert sid in meta_by_id.index
        assert meta_by_id.loc[sid, "source_group"].strip().lower() == "control"
        assert path_group_of(meta_by_id.loc[sid, "audio_path"]) == "control"
        assert corrected_map[sid] == 0
    ck280 = ck_df[ck_df["sample_id"].str.contains("280")]
    assert (ck280["id_prefix"] == "control").all()
    assert (ck280["old_label"] == "positive").all()
    log("280-0/1/2 verified: actual records say dir=Control, checkpoint IDs say "
        "pitt_control_cookie_280-*; the old report claim 'prefix still dementia' "
        "is factually WRONG and is withdrawn in the revised report")

    # ---------------------------------------------------------------
    # 5. Rebuild corrected feature frames + refit LR models (gates vs frozen)
    # ---------------------------------------------------------------
    log("\n== [5] RE-FIT LR MODELS (determinism gates vs frozen) ==")
    tr_f = feats_train[feats_train["sample_id"].isin(corrected_train)].copy()
    te_f = feats_test[feats_test["sample_id"].isin(corrected_test)].copy()
    for df in (tr_f, te_f):
        df["y_true"] = df["sample_id"].map(corrected_map).astype(int)
        df["subject_id"] = df["sample_id"].map(subj_map)
    tr_f = tr_f.sort_values("sample_id").reset_index(drop=True)
    te_f = te_f.sort_values("sample_id").reset_index(drop=True)
    ytrain = tr_f["y_true"].to_numpy(dtype=int)
    ytest = te_f["y_true"].to_numpy(dtype=int)

    full_m, full_pred, full_prob, full_t, full_coef, full_imp = train_lr(
        tr_f, ytrain, te_f, ytest, FEATURE_COLS)
    prog_m, prog_pred, prog_prob, prog_t, prog_coef, _ = train_lr(
        tr_f, ytrain, te_f, ytest, PROG_COLS)
    llm_m, llm_pred, llm_prob, llm_t, llm_coef, _ = train_lr(
        tr_f, ytrain, te_f, ytest, LLM_COLS)

    gate_pred = pd.read_csv(GATE_PREDS, encoding="utf-8-sig")
    gate_pred["sample_id"] = gate_pred["sample_id"].astype(str).str.strip()
    gate_score_map = {
        "Full Hybrid LR": "Full_Hybrid_LR_score",
        "Programmatic-only LR": "Programmatic_only_LR_score",
        "LLM-only LR": "LLM_only_LR_score",
        "Prompt RULERS": "Prompt_RULERS_score",
        "Direct LLM": "Direct_LLM_score",
        "Weighted RULERS": "Weighted_RULERS_score",
    }
    gate_by_sid = {row["sample_id"]: row for _, row in gate_pred.iterrows()}
    assert set(gate_by_sid) == corrected_test

    def gate_probs(name, prob_arr, col, tol=1e-12):
        diffs = [abs(prob_arr[i] - float(gate_by_sid[sid][col]))
                 for i, sid in enumerate(te_f["sample_id"])]
        maxd = max(diffs)
        assert maxd < tol, "%s prob mismatch maxdiff=%.3g" % (name, maxd)
        log("gate OK: %s probs reproduce frozen artifact (maxdiff < %.1g)" % (name, tol))

    gate_probs("LLM-only", llm_prob, "LLM_only_LR_score")
    gate_probs("Prog-only", prog_prob, "Programmatic_only_LR_score")
    gate_probs("Full-Hybrid", full_prob, "Full_Hybrid_LR_score")

    gate_comp = pd.read_csv(GATE_COMP, encoding="utf-8-sig")
    gate_auc = dict(zip(gate_comp["method"], gate_comp["AUC"]))
    gate_thr = dict(zip(gate_comp["method"], gate_comp["threshold"]))
    for name, m, t in (("LLM-only LR", llm_m, llm_t),
                       ("Programmatic-only LR", prog_m, prog_t),
                       ("Full Hybrid LR", full_m, full_t)):
        # archived strategy_comparison stores 4dp-rounded values -> 5e-5 tolerance
        assert abs(m["AUC"] - float(gate_auc[name])) < 5e-5, (name, m["AUC"])
        assert abs(float(t) - float(gate_thr[name])) < 5e-5, (name, t)
    log("gate OK: LR AUCs and Youden thresholds reproduce frozen values")

    # Prompt RULERS / Direct aggregation from caches (verbatim)
    test_valid = [r for r in cp_rows if is_valid_row(r)]
    tdf = pd.DataFrame(test_valid)
    tdf["sample_id"] = tdf["sample_id"].astype(str).str.strip()
    tdf["y_true"] = tdf["sample_id"].map(corrected_map)
    tdf = tdf.dropna(subset=["y_true"]).copy()
    tdf["y_true"] = tdf["y_true"].astype(int)
    test_in = tdf[tdf["sample_id"].isin(corrected_test)]
    r_df = test_in[test_in["method"] == "rulers"]
    r_agg = r_df.groupby("sample_id").agg(
        {"risk_score": "median", "y_true": "first"}).reset_index()
    d_df = test_in[test_in["method"] == "direct"]
    d_agg = d_df.groupby("sample_id").agg(
        {"risk_score": "median", "y_true": "first"}).reset_index()
    assert len(r_agg) == 127 and len(d_agg) == 127
    for _, row in r_agg.iterrows():
        assert abs(row["risk_score"] - float(gate_by_sid[row["sample_id"]]["Prompt_RULERS_score"])) < 1e-12
    for _, row in d_agg.iterrows():
        assert abs(row["risk_score"] - float(gate_by_sid[row["sample_id"]]["Direct_LLM_score"])) < 1e-12
    log("gate OK: Prompt-RULERS and Direct cached scores reproduce frozen artifact")

    # Train RULERS Youden (corrected train, verbatim)
    train_raw = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    train_raw["sample_id"] = train_raw["sample_id"].astype(str).str.strip()
    for c in ["hesitation_rate", "long_pause_ratio", "speech_rate_cps",
              "words_per_second", "risk_score"]:
        if c in train_raw.columns:
            train_raw[c] = pd.to_numeric(train_raw[c], errors="coerce")
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
    th_rulers, j_rulers = select_youden_threshold(
        tr_r_agg["y_true"].to_numpy(dtype=int),
        pd.to_numeric(tr_r_agg["risk_score"], errors="coerce").to_numpy(dtype=float))
    assert th_rulers == 0.07 and abs(j_rulers - 0.2502) < 1e-3
    log("gate OK: Prompt-RULERS corrected-train Youden = 0.07 (J=%.4f)" % j_rulers)

    # ---------------------------------------------------------------
    # 6. T3 — Weighted RULERS validity recheck
    # ---------------------------------------------------------------
    log("\n== [6] T3 WEIGHTED RULERS VALIDITY RECHECK ==")
    old_wj = json.loads(GATE_WEIGHTS.read_text(encoding="utf-8"))
    old_weights = dict(old_wj["weights"])

    # --- 6a. train side: re-learn weights under the NEW validity rule
    tr_rows = [row.to_dict() for _, row in train_raw.iterrows()
               if str(row["sample_id"]).strip() in corrected_train]
    assert len(tr_rows) == 311
    lit_prior = {"C01": 1.3, "C02": 1.0, "C03": 1.0, "C04": 1.0, "C05": 1.3,
                 "C06": 1.0, "C07": 1.0, "C08": 1.0, "C09": 1.3, "C10": 1.15,
                 "C11": 1.0, "C12": 1.0, "C13": 1.0}

    train_row_audit = []
    train_cl = defaultdict(lambda: defaultdict(list))
    for r in tr_rows:
        oj = r.get("output_json", "")
        parse_ok = False
        if isinstance(oj, str) and oj:
            try:
                oj = json.loads(oj)
                parse_ok = True
            except Exception:
                parse_ok = False
        if not parse_ok or not isinstance(oj, dict) or not oj.get("checklist"):
            train_row_audit.append({
                "sample_id": r["sample_id"], "repeat_idx": r.get("repeat_idx"),
                "row_status": "no_checklist",
                "n_valid_items_new": 0,
                "contributed_to_old_learning": "no", "contributed_to_new_learning": "no"})
            continue
        items, n_na, n_inv = checklist_items_new(oj)
        if items:
            for cid, s in items:
                train_cl[r["sample_id"]][cid].append(s)
            train_row_audit.append({
                "sample_id": r["sample_id"], "repeat_idx": r.get("repeat_idx"),
                "row_status": "has_valid_items", "n_valid_items_new": len(items),
                "n_na_items": n_na, "n_invalid_items": n_inv,
                "contributed_to_old_learning": "yes",
                "contributed_to_new_learning": "yes"})
        else:
            train_row_audit.append({
                "sample_id": r["sample_id"], "repeat_idx": r.get("repeat_idx"),
                "row_status": "all_invalid_or_na",
                "n_valid_items_new": 0, "n_na_items": n_na, "n_invalid_items": n_inv,
                "contributed_to_old_learning": "no",
                "contributed_to_new_learning": "no"})
    tr_audit_df = pd.DataFrame(train_row_audit)
    tr_audit_df.to_csv(HERE / "weighted_rulers_validity_audit_train.csv", index=False,
                       encoding="utf-8-sig")
    n_train_all_invalid = int((tr_audit_df["row_status"] == "all_invalid_or_na").sum())
    n_train_valid_rows = int((tr_audit_df["row_status"] == "has_valid_items").sum())
    log("train: 311 rulers rows -> %d with valid items, %d all-invalid, %d no-checklist"
        % (n_train_valid_rows, n_train_all_invalid,
           int((tr_audit_df["row_status"] == "no_checklist").sum())))

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
        yt_list = [corrected_map[sid] for sid in scores_by_sample if sid in corrected_map]
        ys_list = [scores_by_sample[sid] for sid in scores_by_sample if sid in corrected_map]
        if len(yt_list) >= 10 and len(set(yt_list)) >= 2:
            try:
                item_auc = roc_auc_score(yt_list, ys_list)
                crit_usefulness[cid] = 1.0 + abs(item_auc - 0.5) * 2
            except Exception:
                crit_usefulness[cid] = 1.0
        else:
            crit_usefulness[cid] = 1.0
    new_weights = {}
    for cid in sorted(lit_prior):
        raw = lit_prior[cid] * (1.0 - crit_disagree.get(cid, 0.5)) * crit_usefulness.get(cid, 1.0)
        new_weights[cid] = max(0.5, min(2.0, raw))
    mean_w = float(np.mean(list(new_weights.values())))
    new_weights = {cid: round(w / mean_w, 4) for cid, w in new_weights.items()}

    wdiff_rows = []
    for cid in sorted(set(list(old_weights) + list(new_weights))):
        ov = float(old_weights.get(cid, float("nan")))
        nv = float(new_weights.get(cid, float("nan")))
        wdiff_rows.append({"criterion_id": cid, "old_weight": ov, "new_weight": nv,
                           "diff": round(nv - ov, 10)})
    wdiff = pd.DataFrame(wdiff_rows)
    wdiff.to_csv(HERE / "weighted_rulers_weight_diff.csv", index=False,
                 encoding="utf-8-sig")
    max_wdiff = float(wdiff["diff"].abs().max())
    log("weight diff old vs new: max abs diff = %.3g (affected: %s)" % (
        max_wdiff, "none" if max_wdiff < 1e-9 else "see CSV"))
    (HERE / "weighted_rulers_weights_recheck.json").write_text(json.dumps({
        "weights": new_weights,
        "learned_on": "corrected training set (311 rows; %d contributing rows)" % n_train_valid_rows,
        "validity_rule": "finite numeric non-bool item scores only (frozen recheck rule)",
        "old_vs_new_max_abs_diff": max_wdiff,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- 6b. test side: replicate OLD aggregation (gate) and build NEW one
    w_old_rows, w_new_rows, w_audit = [], [], []
    for r in cp_rows:
        if r.get("method") != "rulers":
            continue
        sid = str(r.get("sample_id", "")).strip()
        if sid not in corrected_test:
            continue
        oj = r.get("output_json", "")
        parse_ok = False
        if isinstance(oj, str) and oj:
            try:
                oj = json.loads(oj)
                parse_ok = True
            except Exception:
                parse_ok = False
        if not parse_ok or not isinstance(oj, dict) or not oj.get("checklist"):
            w_audit.append({"sample_id": sid, "repeat_idx": r.get("repeat_idx"),
                            "old_row_status": "skipped_no_checklist",
                            "old_weighted_risk": None, "new_weighted_risk": None,
                            "n_valid_items_new": 0})
            continue
        # old rule (verbatim): non-NA numeric items; all-invalid -> 0.0
        ws_o, wsum_o = 0.0, 0.0
        for item in oj["checklist"]:
            cid = item.get("criterion_id", "")
            s = item.get("score")
            if s == "NA" or s is None or not isinstance(s, (int, float)):
                continue
            w = old_weights.get(cid, 1.0)
            ws_o += s * w
            wsum_o += w
        old_wr = (ws_o / wsum_o) / 2.0 if wsum_o > 0 else 0.0
        old_status = "appended_0.0_all_invalid" if wsum_o == 0 else "appended_computed"
        w_old_rows.append({"sample_id": sid, "repeat_idx": r.get("repeat_idx"),
                           "weighted_risk": old_wr})
        # new rule (frozen)
        items, n_na, n_inv = checklist_items_new(oj)
        if items:
            ws_n = sum(s * new_weights.get(cid, 1.0) for cid, s in items)
            wsum_n = sum(new_weights.get(cid, 1.0) for cid, _ in items)
            new_wr = (ws_n / wsum_n) / 2.0
            w_new_rows.append({"sample_id": sid, "repeat_idx": r.get("repeat_idx"),
                               "weighted_risk": new_wr})
            new_status = "appended_computed"
        else:
            new_status = "missing_no_valid_items"
        w_audit.append({"sample_id": sid, "repeat_idx": r.get("repeat_idx"),
                        "old_row_status": old_status,
                        "old_weighted_risk": round(old_wr, 8),
                        "new_weighted_risk": round(new_wr, 8) if new_status == "appended_computed" else None,
                        "new_row_status": new_status,
                        "n_valid_items_new": len(items),
                        "n_na_items": n_na, "n_invalid_items": n_inv})
    w_audit_df = pd.DataFrame(w_audit)
    w_audit_df.to_csv(HERE / "weighted_rulers_validity_audit_test.csv", index=False,
                      encoding="utf-8-sig")
    n_no_chk = int((w_audit_df["old_row_status"] == "skipped_no_checklist").sum())
    n_old_zero = int((w_audit_df["old_row_status"] == "appended_0.0_all_invalid").sum())
    n_valid_rows = int((w_audit_df["new_row_status"] == "appended_computed").sum())
    assert len(w_audit_df) == 635 and n_no_chk == 12 and n_old_zero == 39 and n_valid_rows == 584
    log("test: 635 rulers rows -> 584 valid, 39 all-NA (old rule: 0.0), 12 no-checklist (skipped)")

    # old aggregation replicate + gate vs frozen per-sample scores
    w_old_df = pd.DataFrame(w_old_rows)
    w_old_agg = w_old_df.groupby("sample_id").agg(
        {"weighted_risk": "median"}).reset_index()
    assert len(w_old_agg) == 127
    for _, row in w_old_agg.iterrows():
        assert abs(row["weighted_risk"] - float(gate_by_sid[row["sample_id"]]["Weighted_RULERS_score"])) < 1e-12
    log("gate OK: OLD weighted aggregation (incl. all-NA -> 0.0) reproduces frozen artifact")

    # new aggregation: median over VALID repeats only; missing when none
    w_new_df = pd.DataFrame(w_new_rows)
    w_new_agg = w_new_df.groupby("sample_id").agg(
        {"weighted_risk": "median"}).reset_index()
    missing_sids = sorted(corrected_test - set(w_new_agg["sample_id"]))
    assert len(missing_sids) == 4
    log("new rule: 4 samples have zero valid repeats -> MISSING (not 0-filled): %s"
        % ", ".join(missing_sids))
    # samples whose score CHANGED vs old aggregation, and samples with all-NA
    # repeats whose median is unchanged (discrete-score ties)
    changed_rows = []
    old_map = dict(zip(w_old_agg["sample_id"], w_old_agg["weighted_risk"]))
    new_map = dict(zip(w_new_agg["sample_id"], w_new_agg["weighted_risk"]))
    for sid in sorted(corrected_test):
        if sid in new_map and abs(float(new_map[sid]) - float(old_map[sid])) > 1e-12:
            changed_rows.append({"sample_id": sid,
                                 "old_weighted_risk": round(float(old_map[sid]), 6),
                                 "new_weighted_risk": round(float(new_map[sid]), 6)})
    sids_with_allna_rows = set(w_audit_df.loc[
        w_audit_df["old_row_status"] == "appended_0.0_all_invalid", "sample_id"])
    affected_unchanged = sorted(sids_with_allna_rows & set(new_map) -
                                {r["sample_id"] for r in changed_rows})
    log("samples with revised (non-missing) scores: %d" % len(changed_rows))
    for r in changed_rows:
        log("  %s old=%.4f new=%.4f" % (r["sample_id"], r["old_weighted_risk"],
                                        r["new_weighted_risk"]))
    log("samples with all-NA repeats but unchanged median (ties): %d -> %s"
        % (len(affected_unchanged), ", ".join(affected_unchanged)))
    # verified against weighted_rulers_validity_audit_test.csv:
    # 302-0: 3 all-NA + two 0.0 repeats -> median stays 0.0
    # 168-0: 1 all-NA + four valid [1,1,0.323,1] -> median stays 1.0
    assert len(changed_rows) == 6 and len(affected_unchanged) == 2
    pd.DataFrame(changed_rows).to_csv(HERE / "weighted_rulers_changed_samples.csv",
                                      index=False, encoding="utf-8-sig")
    pd.DataFrame([{"sample_id": s, "note": "all-NA repeats present but median "
                                           "unchanged (discrete-score ties)"}
                  for s in affected_unchanged]
                 ).to_csv(HERE / "weighted_rulers_unchanged_median_samples.csv",
                          index=False, encoding="utf-8-sig")

    # before/after metrics
    def weighted_metrics(agg, n_expected):
        yt = agg["sample_id"].map(corrected_map).astype(int).to_numpy()
        ys = pd.to_numeric(agg["weighted_risk"], errors="coerce").to_numpy()
        yp = (ys >= th_rulers).astype(int)
        m = compute_metrics(yt, yp, ys)
        assert m["n"] == n_expected
        return m

    m_w_old = weighted_metrics(w_old_agg, 127)
    m_w_new = weighted_metrics(w_new_agg, 123)
    ba_rows = []
    for label, m in (("old_rule_incl_zero_fill_127", m_w_old),
                     ("new_rule_missing_not_zero_123", m_w_new)):
        ba_rows.append({"rule": label, "threshold": th_rulers, "n": m["n"],
                        "Acc": round(m["Acc"], 4), "Sens": round(m["Sens"], 4),
                        "Spec": round(m["Spec"], 4), "F1": round(m["F1"], 4),
                        "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
                        "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"]})
    ba = pd.DataFrame(ba_rows)
    ba.to_csv(HERE / "weighted_rulers_metrics_before_after.csv", index=False,
              encoding="utf-8-sig")
    log(ba.to_string(index=False))
    # gate: old-rule metrics == frozen Weighted RULERS row
    gw = gate_comp[gate_comp["method"] == "Weighted RULERS"].iloc[0]
    for k in ("Acc", "Sens", "Spec", "F1", "AUC"):
        # archived strategy_comparison stores 4dp-rounded values -> 5e-5 tolerance
        assert abs(m_w_old[k] - float(gw[k])) < 5e-5, (k, m_w_old[k], gw[k])
    log("gate OK: old-rule Weighted RULERS metrics reproduce frozen strategy_comparison")

    # ---------------------------------------------------------------
    # 7. Build comparison frames (formal without Direct)
    # ---------------------------------------------------------------
    log("\n== [7] FORMAL COMPARISON FRAMES ==")
    frame = te_f[["sample_id", "subject_id", "y_true"]].copy()
    frame["old_label"] = frame["sample_id"].map(
        lambda s: meta_by_id.loc[s, "label"])
    frame = frame.merge(r_agg[["sample_id", "risk_score"]].rename(
        columns={"risk_score": "rulers_score"}), on="sample_id", how="left")
    frame = frame.merge(d_agg[["sample_id", "risk_score"]].rename(
        columns={"risk_score": "direct_score"}), on="sample_id", how="left")
    frame = frame.merge(w_old_agg[["sample_id", "weighted_risk"]].rename(
        columns={"weighted_risk": "weighted_risk_old"}), on="sample_id", how="left")
    frame = frame.merge(w_new_agg[["sample_id", "weighted_risk"]].rename(
        columns={"weighted_risk": "weighted_risk_new"}), on="sample_id", how="left")
    frame["llm_prob"] = llm_prob
    frame["prog_prob"] = prog_prob
    frame["hybrid_prob"] = full_prob
    assert frame[["direct_score", "rulers_score", "weighted_risk_old"]].notna().all().all()
    assert frame["weighted_risk_new"].isna().sum() == 4

    # formal methods (Direct moved to the appendix)
    formal_methods = {
        "Prompt RULERS": ("rulers_score", th_rulers,
                          "corrected-train Youden (protocol)"),
        "Weighted RULERS (new rule)": ("weighted_risk_new", th_rulers,
                                       "new validity rule; 4 samples missing (n=123)"),
        "LLM-only LR": ("llm_prob", llm_t, "train Youden (corrected train)"),
        "Programmatic-only LR": ("prog_prob", prog_t, "train Youden (corrected train)"),
        "Full Hybrid LR": ("hybrid_prob", full_t, "train Youden (corrected train)"),
    }
    comp_rows = []
    for name, (col, thr, origin) in formal_methods.items():
        sub = frame.dropna(subset=[col])
        yt = sub["y_true"].values
        ys = sub[col].values
        m = compute_metrics(yt, (ys >= thr).astype(int), ys)
        comp_rows.append({
            "method": name, "table": "formal_leak_free",
            "threshold": round(float(thr), 4), "threshold_origin": origin,
            "n": m["n"], "Acc": round(m["Acc"], 4), "Sens": round(m["Sens"], 4),
            "Spec": round(m["Spec"], 4), "F1": round(m["F1"], 4),
            "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
            "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
            "note": ""})
    # appendix Direct row
    sub = frame.dropna(subset=["direct_score"])
    yt = sub["y_true"].values
    ys = sub["direct_score"].values
    m = compute_metrics(yt, (ys >= TH_DIRECT).astype(int), ys)
    comp_rows.append({
        "method": "Direct LLM", "table": "appendix_historical_baseline",
        "threshold": TH_DIRECT,
        "threshold_origin": "prespecified 0.50",
        "n": m["n"], "Acc": round(m["Acc"], 4), "Sens": round(m["Sens"], 4),
        "Spec": round(m["Spec"], 4), "F1": round(m["F1"], 4),
        "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
        "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
        "note": "diagnosis-related ID-exposed historical baseline; appendix only; "
                "no inference on exposure direction/magnitude"})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(HERE / "strategy_comparison_revised.csv", index=False, encoding="utf-8-sig")
    log(comp[["method", "table", "n", "Acc", "Sens", "Spec", "F1", "AUC"]].to_string(index=False))

    # common-sample (123) fair comparison: all six branches
    common_sids = set(w_new_agg["sample_id"])
    frame123 = frame[frame["sample_id"].isin(common_sids)].copy()
    common_methods = dict(formal_methods)
    common_methods["Direct LLM (appendix)"] = ("direct_score", TH_DIRECT, "appendix")
    common_rows = []
    for name, (col, thr, origin) in common_methods.items():
        yt = frame123["y_true"].values
        ys = frame123[col].values
        m = compute_metrics(yt, (ys >= thr).astype(int), ys)
        common_rows.append({"method": name, "threshold": round(float(thr), 4),
                            "n": m["n"], "Acc": round(m["Acc"], 4),
                            "Sens": round(m["Sens"], 4), "Spec": round(m["Spec"], 4),
                            "F1": round(m["F1"], 4),
                            "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
                            "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
                            "note": "common 123-sample subset where new-rule Weighted "
                                    "RULERS is defined"})
    common_df = pd.DataFrame(common_rows)
    common_df.to_csv(HERE / "strategy_comparison_common_samples_123.csv", index=False,
                     encoding="utf-8-sig")
    log(common_df[["method", "n", "Acc", "Sens", "Spec", "F1", "AUC"]].to_string(index=False))

    # revised per-sample predictions
    pred_out = frame[["sample_id", "subject_id", "old_label", "y_true"]].copy()
    pred_out["Prompt_RULERS_score"] = frame["rulers_score"]
    pred_out["Weighted_RULERS_score_new_rule"] = frame["weighted_risk_new"]
    pred_out["Weighted_RULERS_score_old_rule"] = frame["weighted_risk_old"]
    pred_out["LLM_only_LR_score"] = llm_prob
    pred_out["Programmatic_only_LR_score"] = prog_prob
    pred_out["Full_Hybrid_LR_score"] = full_prob
    pred_out["Direct_LLM_score_appendix_only"] = frame["direct_score"]
    pred_out["weighted_new_rule_status"] = pred_out["sample_id"].map(
        lambda s: "missing_no_valid_repeats" if s in set(missing_sids) else
        ("revised" if s in {r["sample_id"] for r in changed_rows} else "unchanged_or_new"))
    pred_out.to_csv(HERE / "per_sample_predictions_revised.csv", index=False,
                    encoding="utf-8-sig")

    # ---------------------------------------------------------------
    # 8. Bootstrap frames (20000, seed 42)
    # ---------------------------------------------------------------
    log("\n== [8] BOOTSTRAPS ==")

    def run_bootstrap(frm, methods, frame_name):
        frm = frm.reset_index(drop=True)  # positional labels required below
        cluster_groups = {s: g.index.to_numpy() for s, g in
                          frm.groupby("subject_id", sort=True)}
        cluster_ids = np.array(sorted(cluster_groups))
        rng = np.random.RandomState(SEED)
        ys_all = {name: frm[col].values for name, (col, _, _) in methods.items()}
        thr_vals = {name: thr for name, (_, thr, _) in methods.items()}
        yt_all = frm["y_true"].values
        method_metric_lists = {name: defaultdict(list) for name in methods}
        n_invalid = 0
        n_auc_undefined = Counter()
        diff_lists = defaultdict(list)
        for _ in range(N_BOOT_MAIN):
            sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            idx = np.concatenate([cluster_groups[s] for s in sampled])
            yt = yt_all[idx]
            if len(set(yt)) < 2:
                n_invalid += 1
                continue
            res_auc = {}
            for name in methods:
                ys = ys_all[name][idx]
                yp = (ys >= thr_vals[name]).astype(int)
                m = compute_metrics(yt, yp, ys)
                if m["AUC"] is None:
                    n_auc_undefined[name] += 1
                    m = compute_metrics(yt, yp, None)
                res_auc[name] = m
                for metric in ("Acc", "Sens", "Spec", "F1", "AUC"):
                    v = m[metric]
                    if v is not None:
                        method_metric_lists[name][metric].append(v)
            for other in methods:
                if other == "Full Hybrid LR":
                    continue
                ha = res_auc["Full Hybrid LR"].get("AUC")
                oa = res_auc[other].get("AUC")
                if ha is not None and oa is not None:
                    diff_lists["hybrid_minus_" + other].append(ha - oa)
        return cluster_ids, method_metric_lists, diff_lists, n_invalid, n_auc_undefined

    def ci95(vals):
        return float(np.mean(vals)), float(np.percentile(vals, 2.5)), \
            float(np.percentile(vals, 97.5))

    frameA_methods = {
        "Prompt RULERS": ("rulers_score", th_rulers, "corrected-train Youden"),
        "LLM-only LR": ("llm_prob", llm_t, "train Youden"),
        "Programmatic-only LR": ("prog_prob", prog_t, "train Youden"),
        "Full Hybrid LR": ("hybrid_prob", full_t, "train Youden"),
    }
    cids_a, ml_a, dl_a, inv_a, undef_a = run_bootstrap(frame, frameA_methods, "A(127)")
    # gate: Hybrid - Prog reproduces the frozen interval (same draw sequence)
    d_hp = dl_a["hybrid_minus_Programmatic-only LR"]
    mean_v, lo_v, hi_v = ci95(d_hp)
    gb = pd.read_csv(GATE_BOOT, encoding="utf-8-sig")
    gb_match = gb[gb["comparison"] == "Full_Hybrid_minus_Programmatic-only LR"]
    if "primary" in gb.columns:
        gb_match = gb_match[gb_match["primary"].astype(str).eq("True")]
    assert len(gb_match) >= 1, "archived bootstrap lacks Hybrid-Prog row"
    gb_row = gb_match.iloc[0]
    point_hp = float(roc_auc_score(frame["y_true"], frame["hybrid_prob"])) - \
        float(roc_auc_score(frame["y_true"], frame["prog_prob"]))
    assert abs(mean_v - float(gb_row["bootstrap_mean_auc_diff"])) < 5e-5
    assert abs(lo_v - float(gb_row["CI95_low"])) < 5e-5
    assert abs(hi_v - float(gb_row["CI95_high"])) < 5e-5
    assert abs(point_hp - float(gb_row["point_auc_diff"])) < 5e-5
    log("gate OK: bootstrap A Hybrid-Prog reproduces frozen interval "
        "(point %.4f, CI [%.4f, %.4f])" % (point_hp, lo_v, hi_v))
    log("bootstrap A(127): used=%d invalid=%d undef=%s"
        % (N_BOOT_MAIN - inv_a, inv_a, dict(undef_a)))

    frameB_methods = {
        "Prompt RULERS": ("rulers_score", th_rulers, "corrected-train Youden"),
        "Weighted RULERS (new rule)": ("weighted_risk_new", th_rulers, "new rule"),
        "LLM-only LR": ("llm_prob", llm_t, "train Youden"),
        "Programmatic-only LR": ("prog_prob", prog_t, "train Youden"),
        "Full Hybrid LR": ("hybrid_prob", full_t, "train Youden"),
    }
    cids_b, ml_b, dl_b, inv_b, undef_b = run_bootstrap(frame123, frameB_methods, "B(123)")
    log("bootstrap B(123 common): used=%d invalid=%d undef=%s"
        % (N_BOOT_MAIN - inv_b, inv_b, dict(undef_b)))

    frameC_methods = {
        "Direct LLM": ("direct_score", TH_DIRECT, "appendix 0.50"),
        "Full Hybrid LR": ("hybrid_prob", full_t, "train Youden"),
    }
    cids_c, ml_c, dl_c, inv_c, undef_c = run_bootstrap(frame, frameC_methods, "C(127 appendix)")
    log("bootstrap C(127 appendix): used=%d invalid=%d undef=%s"
        % (N_BOOT_MAIN - inv_c, inv_c, dict(undef_c)))

    # assemble bootstrap output CSVs
    boot_rows = []
    for frame_name, ml in (("A_127_formal", ml_a), ("B_123_common", ml_b),
                           ("C_127_appendix", ml_c)):
        for name in ml:
            for metric in ("Acc", "Sens", "Spec", "F1", "AUC"):
                vals = ml[name][metric]
                if not vals:
                    continue
                mean_v, lo_v, hi_v = ci95(vals)
                boot_rows.append({"frame": frame_name, "method": name,
                                  "metric": metric, "n_resamples_used": len(vals),
                                  "bootstrap_mean": round(mean_v, 4),
                                  "CI95_low": round(lo_v, 4),
                                  "CI95_high": round(hi_v, 4)})
    pd.DataFrame(boot_rows).to_csv(HERE / "clustered_bootstrap_revised_method_metrics.csv",
                                   index=False, encoding="utf-8-sig")

    def point_auc(frm, col):
        return float(roc_auc_score(frm["y_true"], frm[col]))

    d_out = []
    for label, frame_name, frm, dl, inv, primary in (
        ("Full_Hybrid_minus_Programmatic-only LR", "A_127_formal", frame, dl_a, inv_a, True),
        ("Full_Hybrid_minus_Prompt RULERS", "A_127_formal", frame, dl_a, inv_a, False),
        ("Full_Hybrid_minus_LLM-only LR", "A_127_formal", frame, dl_a, inv_a, False),
        ("Full_Hybrid_minus_Weighted RULERS (new rule)", "B_123_common", frame123, dl_b, inv_b, False),
        ("Full_Hybrid_minus_Direct LLM", "C_127_appendix", frame, dl_c, inv_c, False),
    ):
        dl_key = "hybrid_minus_" + label.split("minus_", 1)[1]
        if dl_key not in dl:
            continue
        d = dl[dl_key]
        mean_v, lo_v, hi_v = ci95(d)
        other_col = {"Programmatic-only LR": "prog_prob",
                     "Prompt RULERS": "rulers_score",
                     "LLM-only LR": "llm_prob",
                     "Weighted RULERS (new rule)": "weighted_risk_new",
                     "Direct LLM": "direct_score"}[label.split("minus_")[1]]
        point = point_auc(frm, "hybrid_prob") - point_auc(frm, other_col)
        d_out.append({
            "comparison": label, "frame": frame_name,
            "primary": primary,
            "appendix_historical": label.endswith("Direct LLM"),
            "n_samples_frame": len(frm),
            "point_auc_diff": round(point, 4),
            "bootstrap_mean_auc_diff": round(mean_v, 4),
            "CI95_low": round(lo_v, 4), "CI95_high": round(hi_v, 4),
            "n_resamples_used": len(d),
            "n_resamples_skipped_invalid": inv,
            "proportion_diff_gt_0": round(float(np.mean(np.array(d) > 0)), 4),
            "interpretation_note": "empirical proportion; not a p-value; "
                                   "Direct rows are appendix-only with no "
                                   "direction/magnitude inference" if label.endswith("Direct LLM")
                                   else "empirical proportion; not a p-value",
        })
    ddf = pd.DataFrame(d_out)
    ddf.to_csv(HERE / "clustered_bootstrap_revised_auc_diffs.csv", index=False,
               encoding="utf-8-sig")
    log(ddf[["comparison", "frame", "point_auc_diff", "CI95_low", "CI95_high",
             "proportion_diff_gt_0"]].to_string(index=False))

    # ---------------------------------------------------------------
    # 9. T4 — exploratory ablation: Programmatic-only + demographics
    # ---------------------------------------------------------------
    log("\n== [9] T4 EXPLORATORY ABLATION (post-hoc, marked) ==")
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
        assert int(df[DEMO_COLS].isna().any(axis=1).sum()) == 0, name

    demo_results = {}
    demo_results["demographic_only"] = train_lr(tr_d, ytrain, te_d, ytest, DEMO_COLS)
    demo_results["full_hybrid"] = (full_m, full_pred, full_prob, full_t, full_coef, full_imp)
    demo_results["hybrid_plus_demographics"] = train_lr(
        tr_d, ytrain, te_d, ytest, FEATURE_COLS + DEMO_COLS)
    demo_results["prog_plus_demographics"] = train_lr(
        tr_d, ytrain, te_d, ytest, PROG_COLS + DEMO_COLS)  # NEW post-hoc model

    # gate: hybrid+demo reproduces frozen demographic artifact
    gdp = pd.read_csv(GATE_DEMO_PREDS, encoding="utf-8-sig")
    gdp["sample_id"] = gdp["sample_id"].astype(str).str.strip()
    gdp_by_sid = dict(zip(gdp["sample_id"], gdp["hybrid_plus_demographics_prob"]))
    hdm_prob = demo_results["hybrid_plus_demographics"][2]
    maxd = max(abs(hdm_prob[i] - float(gdp_by_sid[sid]))
               for i, sid in enumerate(te_d["sample_id"]))
    assert maxd < 1e-12
    log("gate OK: hybrid+demo probs reproduce frozen demographic artifact")

    abl_rows = []
    for name, (m, pred, prob, thr, coef, imp) in demo_results.items():
        abl_rows.append({
            "model": name,
            "n_features": {"demographic_only": 3, "full_hybrid": 12,
                           "hybrid_plus_demographics": 15,
                           "prog_plus_demographics": 9}[name],
            "post_hoc_added": name == "prog_plus_demographics",
            "threshold_train_youden": round(float(thr), 4),
            "n_test": m["n"], "Acc": round(m["Acc"], 4), "Sens": round(m["Sens"], 4),
            "Spec": round(m["Spec"], 4), "F1": round(m["F1"], 4),
            "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
            "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"]})
    abl_comp = pd.DataFrame(abl_rows)
    abl_comp.to_csv(ABL / "model_comparison.csv", index=False, encoding="utf-8-sig")
    log(abl_comp[["model", "n_features", "threshold_train_youden", "Acc", "Sens",
                  "Spec", "F1", "AUC"]].to_string(index=False))

    pdm_coef = demo_results["prog_plus_demographics"][4]
    pdm_imp = demo_results["prog_plus_demographics"][5]
    pdm_rows = []
    for i, feat in enumerate(PROG_COLS + DEMO_COLS):
        pdm_rows.append({"feature": feat,
                         "feature_type": "programmatic" if feat in PROG_COLS else "demographic",
                         "coefficient_standardized": round(float(pdm_coef[i]), 6),
                         "train_median_imputation_value": round(float(pdm_imp[feat]), 6)})
    pd.DataFrame(pdm_rows).to_csv(ABL / "prog_demo_feature_coefficients.csv",
                                  index=False, encoding="utf-8-sig")
    pdm_prob = demo_results["prog_plus_demographics"][2]
    pdm_pred = demo_results["prog_plus_demographics"][1]
    pdm_thr = demo_results["prog_plus_demographics"][3]
    abl_pred = te_d[["sample_id", "subject_id", "y_true"] + DEMO_COLS].copy()
    abl_pred["prog_plus_demo_prob"] = pdm_prob
    abl_pred["prog_plus_demo_pred"] = pdm_pred
    abl_pred["prog_plus_demo_threshold"] = pdm_thr
    abl_pred["hybrid_plus_demo_prob"] = hdm_prob
    abl_pred.to_csv(ABL / "predictions_test.csv", index=False, encoding="utf-8-sig")

    # paired participant-clustered bootstrap: hybrid+demo minus prog+demo
    d_groups = {s: g.index.to_numpy() for s, g in te_d.groupby("subject_id", sort=True)}
    d_ids = np.array(sorted(d_groups))
    d_rng = np.random.RandomState(SEED)
    hdm_minus_pdm = []
    pdm_aucs, hdm_aucs = [], []
    d_invalid = 0
    for _ in range(N_BOOT_MAIN):
        sampled = d_rng.choice(d_ids, size=len(d_ids), replace=True)
        idx = np.concatenate([d_groups[s] for s in sampled])
        yt = ytest[idx]
        if len(set(yt)) < 2:
            d_invalid += 1
            continue
        a_pdm = roc_auc_score(yt, pdm_prob[idx])
        a_hdm = roc_auc_score(yt, hdm_prob[idx])
        pdm_aucs.append(a_pdm)
        hdm_aucs.append(a_hdm)
        hdm_minus_pdm.append(a_hdm - a_pdm)
    m_v, lo_v, hi_v = ci95(hdm_minus_pdm)
    point_ab = float(demo_results["hybrid_plus_demographics"][0]["AUC"] -
                     demo_results["prog_plus_demographics"][0]["AUC"])
    abl_boot_rows = [{
        "comparison": "hybrid_plus_demographics_minus_prog_plus_demographics",
        "post_hoc_exploratory": True,
        "n_subjects": len(d_ids),
        "n_resamples_requested": N_BOOT_MAIN,
        "n_resamples_used": len(hdm_minus_pdm),
        "n_resamples_skipped_invalid": d_invalid,
        "point_auc_diff": round(point_ab, 4),
        "bootstrap_mean_auc_diff": round(m_v, 4),
        "CI95_low": round(lo_v, 4), "CI95_high": round(hi_v, 4),
        "proportion_diff_gt_0": round(float(np.mean(np.array(hdm_minus_pdm) > 0)), 4),
        "age_limitation": "entry age, not recording-time age",
        "note": "post-hoc exploratory analysis; no tuning from results",
    }]
    abl_boot = pd.DataFrame(abl_boot_rows)
    abl_boot.to_csv(ABL / "clustered_bootstrap_auc_diff.csv", index=False,
                    encoding="utf-8-sig")
    log(abl_boot[["comparison", "point_auc_diff", "bootstrap_mean_auc_diff",
                  "CI95_low", "CI95_high", "proportion_diff_gt_0"]].to_string(index=False))
    (ABL / "summary.json").write_text(json.dumps({
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "post_hoc_exploratory": True,
        "labels": "corrected recording-time labels (Plan A)",
        "age_limitation": "entry age, not recording-time age",
        "models": abl_comp.to_dict(orient="records"),
        "bootstrap_20000_seed42": abl_boot_rows,
    }, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------
    # 10. Dev-contact scopes — formal methods only (Direct appendix-only)
    # ---------------------------------------------------------------
    log("\n== [10] DEV-CONTACT SCOPES (formal methods only) ==")
    pilot_union = set()
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
        pilot_union |= sids
    smoke_df = pd.read_csv(SMOKE_DIR / "results.csv", encoding="utf-8-sig")
    smoke_union = set(smoke_df["sample_id"].astype(str).str.strip())
    conservative_union = pilot_union | smoke_union
    log("pilot union %d | smoke %d | conservative %d"
        % (len(pilot_union), len(smoke_union), len(conservative_union)))

    scope_frames = {
        "full_corrected_test": frame.copy(),
        "pilot_excluded": frame[~frame["sample_id"].isin(pilot_union)].copy(),
        "conservative_pilot_plus_smoke_excluded":
            frame[~frame["sample_id"].isin(conservative_union)].copy(),
    }
    dev_rows = []
    for scope_name, f in scope_frames.items():
        for name, (col, thr, _) in formal_methods.items():
            sub = f.dropna(subset=[col])
            yt = sub["y_true"].values
            ys = sub[col].values
            m = compute_metrics(yt, (ys >= thr).astype(int), ys)
            dev_rows.append({
                "scope": scope_name, "method": name,
                "n_samples": m["n"], "n_subjects": sub["subject_id"].nunique(),
                "threshold": round(float(thr), 4),
                "Acc": round(m["Acc"], 4), "Sens": round(m["Sens"], 4),
                "Spec": round(m["Spec"], 4), "F1": round(m["F1"], 4),
                "AUC": round(m["AUC"], 4) if m["AUC"] is not None else None,
                "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
                "note": "Direct excluded (appendix-only)"})
    dev_df = pd.DataFrame(dev_rows)
    dev_df.to_csv(HERE / "dev_contact_revised_metrics.csv", index=False,
                   encoding="utf-8-sig")
    log(dev_df[dev_df["method"] == "Full Hybrid LR"].to_string(index=False))

    # ---------------------------------------------------------------
    # 11. T5 — Direct clean re-run plan (prepared, NOT executed)
    # ---------------------------------------------------------------
    log("\n== [11] T5 DIRECT CLEAN RE-RUN PLAN ==")
    sys.path.insert(0, str(EXPERIMENTS))
    from direct_llm_baseline import (PROMPT_DIRECT_SYSTEM,
                                     PROMPT_DIRECT_USER,
                                     build_direct_prompt)
    sys.path.pop(0)
    run_cfg = json.loads(RUN_CFG.read_text(encoding="utf-8"))
    direct_rows = [r for r in cp_rows if r["method"] == "direct"]
    trans_lens = [len(str(r.get("input_text", ""))) for r in direct_rows]
    out_lens = [len(str(r.get("output_json", ""))) for r in direct_rows]
    sys_chars = len(PROMPT_DIRECT_SYSTEM)
    user_fixed = len(PROMPT_DIRECT_USER.format(asr_text="", sample_id="X"))
    n_calls = 127 * 5
    # per-sample prompt token estimate (transcript identical across repeats)
    test_trans = {}
    for r in direct_rows:
        sid = str(r.get("sample_id", "")).strip()
        if sid in corrected_test:
            test_trans[sid] = len(str(r.get("input_text", "")))
    per_call_prompt_chars = [sys_chars + user_fixed + min(l, 3000) for l in test_trans.values()]
    est_prompt_tokens = [c / 3.5 for c in per_call_prompt_chars]
    est_prompt_total = sum(est_prompt_tokens) * 5
    est_out_per_call = float(np.mean(out_lens)) / 3.0
    est_out_total = est_out_per_call * n_calls
    plan_stats = {
        "n_test_samples": 127, "repeats": 5, "n_calls_main": n_calls,
        "worst_case_with_retries": n_calls * 3,
        "system_prompt_chars": sys_chars,
        "user_template_fixed_chars": user_fixed,
        "transcript_chars_mean": round(float(np.mean(trans_lens)), 1),
        "transcript_chars_p95": round(float(np.percentile(trans_lens, 95)), 1),
        "transcript_chars_max": int(np.max(trans_lens)),
        "est_prompt_tokens_per_call_mean": round(float(np.mean(est_prompt_tokens)), 0),
        "est_prompt_tokens_total_635_calls": round(est_prompt_total, 0),
        "est_completion_tokens_per_call_mean": round(est_out_per_call, 0),
        "est_completion_tokens_total_635_calls": round(est_out_total, 0),
        "original_run_elapsed_s": run_cfg.get("elapsed_s"),
        "original_run_calls": run_cfg.get("total_entries"),
        "est_elapsed_min_serial": round((635 * (run_cfg.get("elapsed_s") or 3090.2) /
                                         1500) / 60, 1),
        "opaque_id_example": uuid.uuid4().hex[:12],
    }
    plan_doc = build_plan_doc(plan_stats, PROMPT_DIRECT_SYSTEM, PROMPT_DIRECT_USER,
                              build_direct_prompt, corrected_test)
    (HERE / "direct_clean_rerun_plan.md").write_text(plan_doc, encoding="utf-8")
    log("plan doc written: direct_clean_rerun_plan.md (calls=%d, worst=%d)"
        % (plan_stats["n_calls_main"], plan_stats["worst_case_with_retries"]))

    # ---------------------------------------------------------------
    # 12. Integrity audit (after) + summary
    # ---------------------------------------------------------------
    log("\n== [12] INTEGRITY AUDIT (after) ==")
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

    # ---------------------------------------------------------------
    # 13. Revised report + impact list (Chinese)
    # ---------------------------------------------------------------
    log("\n== [13] REVISED REPORT + IMPACT LIST ==")
    report = build_report(
        t0=t0, integrity=integrity,
        pref_df=pref_df, n_dem_dir=n_dem_dir, n_con_dir=n_con_dir,
        control_old_pos=control_old_pos,
        control_old_pos_non_cand=control_old_pos_non_cand,
        cand_detail=cand_detail,
        n_direct_records=n_direct_records, n_rulers_records=n_rulers_records,
        br_summary=br_summary, excl_test=excl_test,
        n_train_rows=len(tr_audit_df), n_train_all_invalid=n_train_all_invalid,
        n_train_valid_rows=n_train_valid_rows, max_wdiff=max_wdiff,
        wdiff=wdiff, missing_sids=missing_sids, changed_rows=changed_rows,
        affected_unchanged=affected_unchanged,
        ba=ba, common_df=common_df, comp=comp, ddf=ddf,
        abl_comp=abl_comp, abl_boot=abl_boot,
        dev_df=dev_df, scope_frames=scope_frames,
        plan_stats=plan_stats,
        th_rulers=th_rulers, llm_t=llm_t, prog_t=prog_t, full_t=full_t,
        llm_m=llm_m, prog_m=prog_m, full_m=full_m,
        cand_in_train=cand_in_train, cand_in_test=cand_in_test,
        cand_in_qc_test=cand_in_qc_test,
        m_w_old=m_w_old, m_w_new=m_w_new,
    )
    (HERE / "revised_report.md").write_text(report, encoding="utf-8")
    impact = build_impact_list(
        t0=t0, integrity=integrity, missing_sids=missing_sids,
        changed_rows=changed_rows, affected_unchanged=affected_unchanged,
        ba=ba, ddf=ddf, abl_boot=abl_boot,
        cand_in_train=cand_in_train, cand_in_test=cand_in_test,
        plan_stats=plan_stats,
    )
    (HERE / "impact_list.md").write_text(impact, encoding="utf-8")

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "audit_scope": "closing audit of recording_time_reanalysis (six tasks)",
        "t1_prefix_audit": {
            "n_records_audited": 492,
            "three_source_consistency": "asserted all 492",
            "dir_group_counts": {"dementia": n_dem_dir, "control": n_con_dir},
            "candidates_19_all_control_dir": True,
            "checkpoint_direct_records": n_direct_records,
            "checkpoint_rulers_records": n_rulers_records,
            "excluded_log_repeat_level_rows": int(len(excl_test)),
        },
        "t2_direct_reclassified": "diagnosis-related ID-exposed historical baseline; appendix only",
        "t3_weighted_rulers": {
            "train_rows": len(tr_audit_df),
            "train_all_invalid_rows": n_train_all_invalid,
            "max_weight_diff": max_wdiff,
            "test_rows": 635,
            "test_all_na_rows_old_rule_zero": 39,
            "samples_missing_new_rule": missing_sids,
            "samples_revised": changed_rows,
            "samples_allna_but_median_unchanged": affected_unchanged,
            "metrics": ba.to_dict(orient="records"),
        },
        "t4_ablation": {
            "post_hoc": True,
            "models": abl_comp.to_dict(orient="records"),
            "bootstrap": abl_boot.to_dict(orient="records"),
        },
        "t5_direct_rerun_plan": plan_stats,
        "bootstrap_auc_diffs": ddf.to_dict(orient="records"),
        "integrity": {"n_inputs": len(integrity),
                      "n_unchanged": sum(r["unchanged"] for r in integrity),
                      "changed": [r["input_key"] for r in changed]},
    }
    (HERE / "recheck_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    (HERE / "run_log.txt").write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    log("\nDONE. All outputs under recording_time_recheck/")


# ===========================================================================
# Report builder (Chinese)
# ===========================================================================
def fmt(v, nd=4):
    return "%.*f" % (nd, v)


def mrow(comp, name):
    r = comp[comp["method"] == name].iloc[0]
    return (r["Acc"], r["Sens"], r["Spec"], r["F1"], r["AUC"], r["threshold"], r["n"])


def build_report(t0, integrity, pref_df, n_dem_dir, n_con_dir, control_old_pos,
                 control_old_pos_non_cand, cand_detail, n_direct_records,
                 n_rulers_records, br_summary, excl_test, n_train_rows,
                 n_train_all_invalid, n_train_valid_rows, max_wdiff, wdiff,
                 missing_sids, changed_rows, affected_unchanged, ba, common_df,
                 comp, ddf, abl_comp,
                 abl_boot, dev_df, scope_frames, plan_stats, th_rulers, llm_t,
                 prog_t, full_t, llm_m, prog_m, full_m, cand_in_train,
                 cand_in_test, cand_in_qc_test, m_w_old, m_w_new):
    rA = mrow(comp, "Prompt RULERS")
    rW = mrow(comp, "Weighted RULERS (new rule)")
    rL = mrow(comp, "LLM-only LR")
    rP = mrow(comp, "Programmatic-only LR")
    rH = mrow(comp, "Full Hybrid LR")
    rD = mrow(comp, "Direct LLM")

    def drow(comparison):
        sub = ddf[ddf["comparison"] == comparison]
        return sub.iloc[0] if len(sub) else None

    hp = drow("Full_Hybrid_minus_Programmatic-only LR")
    hpr = drow("Full_Hybrid_minus_Prompt RULERS")
    hl = drow("Full_Hybrid_minus_LLM-only LR")
    hw = drow("Full_Hybrid_minus_Weighted RULERS (new rule)")
    hd = drow("Full_Hybrid_minus_Direct LLM")

    cand_table_rows = "\n".join(
        "| %s | %s | %s | %s | %s | %s |" % (
            r["sample_id"], r["subject_id"], r["split"], r["dir_group"],
            r["old_label"], r["recording_time_label"])
        for _, r in cand_detail.iterrows())
    non_cand_rows = "; ".join("%s(%s)" % (r["sample_id"], r["audit_status"])
                              for _, r in control_old_pos_non_cand.iterrows())
    missing_list = "; ".join(missing_sids)
    changed_list = "; ".join("%s: %.4f->%.4f" % (r["sample_id"], r["old_weighted_risk"],
                                                 r["new_weighted_risk"])
                             for r in changed_rows)
    unchanged_list = "; ".join(affected_unchanged) if affected_unchanged else "无"
    if not non_cand_rows:
        non_cand_rows = "无"
    if not missing_list:
        missing_list = "无"

    rba = ba[ba["rule"] == "old_rule_incl_zero_fill_127"].iloc[0]
    rbn = ba[ba["rule"] == "new_rule_missing_not_zero_123"].iloc[0]
    cR = mrow(common_df, "Prompt RULERS")
    cW = mrow(common_df, "Weighted RULERS (new rule)")
    cL = mrow(common_df, "LLM-only LR")
    cP = mrow(common_df, "Programmatic-only LR")
    cH = mrow(common_df, "Full Hybrid LR")
    cD = mrow(common_df, "Direct LLM (appendix)")

    abl = abl_comp[abl_comp["model"] == "prog_plus_demographics"].iloc[0]
    ablh = abl_comp[abl_comp["model"] == "hybrid_plus_demographics"].iloc[0]
    ab = abl_boot.iloc[0]

    full_dev = dev_df[(dev_df["scope"] == "full_corrected_test") &
                      (dev_df["method"] == "Full Hybrid LR")].iloc[0]
    pe_dev = dev_df[(dev_df["scope"] == "pilot_excluded") &
                    (dev_df["method"] == "Full Hybrid LR")].iloc[0]
    cs_dev = dev_df[(dev_df["scope"] == "conservative_pilot_plus_smoke_excluded") &
                    (dev_df["method"] == "Full Hybrid LR")].iloc[0]

    return """# 第二阶段收尾核查修订报告（recording_time_recheck）

**生成时间：** %(ts)s
**脚本：** `recording_time_recheck.py`（可复现，全程离线；未调用 API、未重新生成任何评分）
**对象：** 对 `recording_time_reanalysis/` 的收尾核查。该目录全部产物保持不变（输入前后
SHA-256 比对 %(n_unch)s/%(n_tot)s 未变，见 `integrity_audit.csv`）；本目录为修订输出。
**研究目标（固定不变）：** 录音时点认知状态识别。纳入/排除规则与标签规则不变
（方案 A：采纳 19 条候选修正、排除 40 条未决）。

---

## 1. 勘误：280-0/1/2 的 ID 前缀（任务 1）

原报告 §3 的论述"3 条测试候选（280-0/1/2）的 ID 前缀仍是 `dementia`"**撤回**。
从实际记录逐条生成的前缀审计（三层独立证据：metadata `sample_id` 前缀、
`source_group` 原目录分组、`audio_path` 中的目录段，492 条三层一致、断言通过）表明：

- **原目录分组 = Control**：`audio_path` 为 `...\\Control\\cookie\\0wav\\280-*.wav`，`source_group=Control`；
- **旧分析标签 = positive**：旧标签与目录分组不一致，正是审计修正的对象；
- **录音时点标签 = control**：与目录分组一致；
- checkpoint 实际记录（Direct 提示词所见）为 `pitt_control_cookie_280-0/1/2`，前缀 `control`。

全部 19 条候选修正（旧阳性 → 录音时点对照）逐条核查：目录分组均为 Control、
旧标签均 positive、录音时点标签均 control：

| 记录 | 参与者 | 划分 | 原目录分组 | 旧分析标签 | 录音时点标签 |
|---|---|---|---|---|---|
%(cand_rows)s

因此：Direct 提示词中的 ID 前缀跟随**原目录分组**，对 280-0/1/2 与录音时点标签一致。
该勘误**不改变任何标签、纳入规则或分析结论的计算**，仅修正报告叙述。
Direct 的降级处理（§3）不依赖泄漏方向，独立成立。

## 2. 逐条前缀审计统计（任务 1）

- 492 条记录：Dementia 目录 %(n_dem)s 条 / Control 目录 %(n_con)s 条。
- 目录分组 × 旧标签：Dementia 目录内旧阳性 302 条；Control 目录内旧阳性 %(con_pos)s 条
  （= 19 条候选 + %(con_pos_other)s 条非候选：%(non_cand)s）。
- checkpoint 层面（按 Direct/RULERS 分支与 repeat_idx 分开，逐条见
  `id_prefix_audit_checkpoint_records.csv`）：
  - Direct 记录 %(n_dir_rec)s 条 = 150 样本 × 5 重复（repeat 0–4 各 150）；
  - RULERS 记录 %(n_rul_rec)s 条 = 150 样本 × 5 重复（repeat 0–4 各 150）；
  - **两方法合计 1500 条是"两分支 × 重复"的记录总数，不是 Direct 调用次数。
    Direct 调用次数 = %(n_dir_rec)s。**
- `final_clean_results/excluded_records.csv` 是**重复级**排除日志（%(n_excl)s 行 =
  RULERS %(n_excl_r)s + Direct %(n_excl_d)s 条记录），不是样本数、不是调用次数。

## 3. Direct 重新定性：诊断相关 ID 暴露的历史基线（任务 2）

按冻结规则（`freeze/recheck_rules.md`），Direct 移出正式无泄漏方法比较：
其 AUC 与一切涉及 Direct 的比较只出现在**明确标记的历史/诊断性附录**（§8）。
对暴露影响的方向或大小**不作任何推断**。主表（§4）、bootstrap 主比较（§5）、
开发接触（§9）均不再含 Direct。

## 4. Weighted RULERS 有效性核查与重算（任务 3）

预先冻结的有效性规则：条目得分必须是 int 或 float、非 bool、`math.isfinite`；
一行无有效条目 → 该行加权分**缺失（不补 0）**；样本聚合取有效重复中位数；
样本无任何有效重复 → 样本缺失；**不为了保留 127 而补零**。

- **训练侧**（权重学习）：%(n_tr_rows)s 行（1 行/样本）中 %(n_tr_valid)s 行有有效条目、
  %(n_tr_inv)s 行全无效（旧规则下同样不贡献任何条目得分）。按新规则重新学习权重，
  与冻结的旧权重逐项差 = 0（最大绝对差 %(max_wdiff).3g；`weighted_rulers_weight_diff.csv`）。
  → **训练权重不受影响**（数据中无 bool/非有限/小数得分；全 NA 行新旧规则均不贡献）。
- **测试侧**：635 行（127×5）中 584 行有效、39 行全 NA（旧规则计 0.0 → 新规则缺失）、
  12 行无 checklist（旧规则即跳过）。含全 NA 重复的样本共 12 个：
  - **4 个样本 5 次重复全部 NA → 新规则下缺失**：%(missing)s；
  - **6 个样本中位数改用有效重复（分数变化）**：%(changed)s；
  - 2 个样本含全 NA 重复但中位数不变（离散得分并列所致，旧 0 填充未影响中位数）：
    %(unchanged)s（`weighted_rulers_unchanged_median_samples.csv`）。
- 修改前后指标（阈值沿用 0.07）：

| 规则 | n | Acc | Sens | Spec | F1 | AUC | TP/TN/FP/FN |
|---|---|---|---|---|---|---|---|
| 旧规则（含全 NA 补 0） | %(oba_n)s | %(oba_a).4f | %(oba_s).4f | %(oba_p).4f | %(oba_f).4f | %(oba_u).4f | %(oba_tp)s/%(oba_tn)s/%(oba_fp)s/%(oba_fn)s |
| 新规则（缺失不补 0） | %(nba_n)s | %(nba_a).4f | %(nba_s).4f | %(nba_p).4f | %(nba_f).4f | %(nba_u).4f | %(nba_tp)s/%(nba_tn)s/%(nba_fp)s/%(nba_fn)s |

- 覆盖范围改变（127 → 123），已另报**共同样本（123）公平比较**（§5 附表，
  `strategy_comparison_common_samples_123.csv`）。旧规则数值仅作协议对照，
  修订结论以新规则为准。

## 5. 正式策略比较与统计区间（不含 Direct）

同一纠正后测试集（127；Weighted RULERS 按新规则 n=123）：

| 方法 | 实际阈值 | n | Acc | Sens | Spec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| Prompt RULERS | %(rA_t)s | %(rA_n)s | %(rA_a).4f | %(rA_s).4f | %(rA_p).4f | %(rA_f).4f | %(rA_u).4f |
| Weighted RULERS（新规则） | %(rW_t)s | %(rW_n)s | %(rW_a).4f | %(rW_s).4f | %(rW_p).4f | %(rW_f).4f | %(rW_u).4f |
| LLM-only LR | %(rL_t)s | %(rL_n)s | %(rL_a).4f | %(rL_s).4f | %(rL_p).4f | %(rL_f).4f | %(rL_u).4f |
| Programmatic-only LR | %(rP_t)s | %(rP_n)s | %(rP_a).4f | %(rP_s).4f | %(rP_p).4f | %(rP_f).4f | %(rP_u).4f |
| Full Hybrid LR | %(rH_t)s | %(rH_n)s | %(rH_a).4f | %(rH_s).4f | %(rH_p).4f | %(rH_f).4f | %(rH_u).4f |

共同样本（123）公平比较（全部方法，Direct 仅附录标记）：

| 方法 | n | Acc | Sens | Spec | F1 | AUC |
|---|---|---|---|---|---|---|
| Prompt RULERS | %(cR_n)s | %(cR_a).4f | %(cR_s).4f | %(cR_p).4f | %(cR_f).4f | %(cR_u).4f |
| Weighted RULERS（新规则） | %(cW_n)s | %(cW_a).4f | %(cW_s).4f | %(cW_p).4f | %(cW_f).4f | %(cW_u).4f |
| LLM-only LR | %(cL_n)s | %(cL_a).4f | %(cL_s).4f | %(cL_p).4f | %(cL_f).4f | %(cL_u).4f |
| Programmatic-only LR | %(cP_n)s | %(cP_a).4f | %(cP_s).4f | %(cP_p).4f | %(cP_f).4f | %(cP_u).4f |
| Full Hybrid LR | %(cH_n)s | %(cH_a).4f | %(cH_s).4f | %(cH_p).4f | %(cH_f).4f | %(cH_u).4f |
| Direct LLM（附录） | %(cD_n)s | %(cD_a).4f | %(cD_s).4f | %(cD_p).4f | %(cD_f).4f | %(cD_u).4f |

配对参与者聚类 bootstrap（20000 次、seed 42、单一 RNG 流、无效重采样跳过并计数；
差值>0 比例为经验比例，不称 p 值）。主框 127 上 Hybrid−Programmatic 与冻结结果
完全一致（确定性门通过）：

| 比较 | 框 | 点差 AUC | bootstrap 均值 | 95%% CI | 差值>0 比例 |
|---|---|---|---|---|---|
| Hybrid − Programmatic-only | A(127) | %(hp_pt).4f | %(hp_m).4f | [%(hp_lo).4f, %(hp_hi).4f] | %(hp_pos).3f |
| Hybrid − Prompt RULERS | A(127) | %(hpr_pt).4f | %(hpr_m).4f | [%(hpr_lo).4f, %(hpr_hi).4f] | %(hpr_pos).3f |
| Hybrid − LLM-only LR | A(127) | %(hl_pt).4f | %(hl_m).4f | [%(hl_lo).4f, %(hl_hi).4f] | %(hl_pos).3f |
| Hybrid − Weighted RULERS（新规则） | B(123) | %(hw_pt).4f | %(hw_m).4f | [%(hw_lo).4f, %(hw_hi).4f] | %(hw_pos).3f |

逐方法分指标 bootstrap 区间见 `clustered_bootstrap_revised_method_metrics.csv`。

## 6. 探索性消融：Programmatic-only + demographics（任务 4，事后新增）

**明确标记为事后新增的探索性分析。** 沿用现有训练/测试 ID、人口学处理（最早访视
规则；年龄仍为基线年龄）、插补、标准化、LR 超参与训练集 Youden 阈值协议，
新增 9 特征模型 Programmatic+demographics，与 Full Hybrid + demographics（15 特征）
比较；不依据结果调参。

| 模型 | 特征数 | 训练 Youden 阈值 | Acc | Sens | Spec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| 仅人口学 | 3 | %(dm_t)s | %(dm_a).4f | %(dm_s).4f | %(dm_p).4f | %(dm_f).4f | %(dm_u).4f |
| Programmatic-only | 6 | %(pd_t)s | %(pd_a).4f | %(pd_s).4f | %(pd_p).4f | %(pd_f).4f | %(pd_u).4f |
| Programmatic + 人口学（新增） | 9 | %(pdm_t)s | %(pdm_a).4f | %(pdm_s).4f | %(pdm_p).4f | %(pdm_f).4f | %(pdm_u).4f |
| Full Hybrid + 人口学 | 15 | %(hdm_t)s | %(hdm_a).4f | %(hdm_s).4f | %(hdm_p).4f | %(hdm_f).4f | %(hdm_u).4f |

配对参与者聚类 bootstrap（20000 次、seed 42）：Full Hybrid+人口学 − Programmatic+人口学
AUC 点差 %(ab_pt).4f，95%% CI [%(ab_lo).4f, %(ab_hi).4f]，差值>0 比例 %(ab_pos).3f
（`ablation_prog_demo/`）。人口学各模型与冻结结果一致（确定性门通过）。

## 7. 开发接触排除（修订口径，不含 Direct）

| 范围 | 样本数 | 参与者数 | Hybrid AUC |
|---|---|---|---|
| 全部 127 | %(fd_n)s | %(fd_subj)s | %(fd_auc).4f |
| 排除 pilot | %(pe_n)s | %(pe_subj)s | %(pe_auc).4f |
| 保守（pilot+smoke） | %(cs_n)s | %(cs_subj)s | %(cs_auc).4f |

全部正式方法 × 全部范围见 `dev_contact_revised_metrics.csv`。

## 8. 附录：Direct 历史基线（任务 2，诊断性附录）

**历史/诊断性附录**：以下数值来自诊断相关 ID 暴露的既有缓存
（提示词含由目录分组派生的 `pitt_dementia_cookie_*`/`pitt_control_cookie_*` ID），
仅作历史记录，不作正式比较；**对暴露影响的方向或大小不作任何推断**。

| 方法 | 阈值 | n | Acc | Sens | Spec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| Direct LLM | %(rD_t)s | %(rD_n)s | %(rD_a).4f | %(rD_s).4f | %(rD_p).4f | %(rD_f).4f | %(rD_u).4f |

附录 bootstrap（C 框，127）：Hybrid − Direct 点差 %(hd_pt).4f，95%% CI
[%(hd_lo).4f, %(hd_hi).4f]。干净重跑方案见 §10 / `direct_clean_rerun_plan.md`
（已准备、未执行）。

## 9. 结论影响清单（摘要；全文见 `impact_list.md`）

1. **撤回**：原报告 §3"280-0/1/2 前缀仍是 dementia"论述（事实错误，实际前缀=control）。
2. **降级**：Direct 及一切 Direct 相关比较 → 附录历史基线；旧报告结论第 3 条
   （Hybrid 相对 Direct）仅作附录，不推断泄漏方向/大小。
3. **修订**：Weighted RULERS 旧规则 n=127（含补零）数值仅作协议对照；修订结论用
   新规则 n=123 + 共同样本（123）公平比较。
4. **维持**：核心结论 Hybrid−Programmatic（点差 %(hp_pt).4f，CI [%(hp_lo).4f, %(hp_hi).4f]，
   差值>0 比例 %(hp_pos).3f）与冻结结果完全一致；三支 LR、Prompt RULERS 阈值、
   人口学敏感性、开发接触（非 Direct 部分）均维持。
5. **新增（事后探索）**：Programmatic+人口学消融（§6），明确标记事后新增。
6. **未执行**：Direct 干净重跑（仅方案，§10）。

## 10. Direct 干净重跑方案摘要（任务 5）

已准备、未执行。要点（全文 `direct_clean_rerun_plan.md`）：
- 清洗：不透明 ID（如 `%(opid)s`）替换 `sample_id`，映射表仅本地保存；
  去除一切诊断相关 ID/目录/路径/元数据；已逐字段核查实际请求内容（system+user
  模板全文与一条真实样例在方案文档中）。
- 固定配置：模型标识（原运行记录 deepseek-v4-flash，需作者确认锁定同一快照；
  若无法匹配，新旧差异不能全归因于去除 ID）、提示词（清洗版模板+哈希）、
  temperature 0.3、max_tokens 16384、repeats 5、失败/有效性规则逐条冻结。
- 预计调用：%(n_calls)s 次主调用（127×5），最坏含重试 %(worst)s 次；
  预估提示词 tokens 合计约 %(tok_p)s、补全约 %(tok_c)s（由实际缓存文本长度估计）；
  预计耗时约 %(mins)s 分钟（串行，原运行 ~2.06 秒/调用）。
- 成本：按官方价格 × 上述 tokens 计算，单价需作者确认；方案给出公式与占位示例。

## 11. 交付文件清单

- `freeze/recheck_rules.md`（预冻结规则）、`freeze/input_hashes_before.csv`
- `id_prefix_audit_records.csv`（492 条）、`id_prefix_audit_checkpoint_records.csv`（1500 条）、
  `id_prefix_audit_summary.csv`、`id_prefix_candidates_detail.csv`（19 条）
- `weighted_rulers_validity_audit_train.csv` / `..._test.csv`、
  `weighted_rulers_weights_recheck.json`、`weighted_rulers_weight_diff.csv`、
  `weighted_rulers_changed_samples.csv`、`weighted_rulers_metrics_before_after.csv`
- `strategy_comparison_revised.csv`、`strategy_comparison_common_samples_123.csv`、
  `per_sample_predictions_revised.csv`
- `clustered_bootstrap_revised_auc_diffs.csv`、`clustered_bootstrap_revised_method_metrics.csv`
- `ablation_prog_demo/`（模型对比、系数、预测、bootstrap、summary）
- `dev_contact_revised_metrics.csv`
- `direct_clean_rerun_plan.md`（方案，未执行）
- `revised_report.md`（本报告）、`impact_list.md`、`recheck_summary.json`、
  `integrity_audit.csv`、`run_log.txt`
- 复现脚本 `recording_time_recheck.py`

本核查未修改论文、未修改任何原始数据/旧结果/冻结产物、未根据结果调参。
""" % dict(
        ts=t0.strftime("%Y-%m-%d %H:%M:%S"),
        n_unch=sum(r["unchanged"] for r in integrity),
        n_tot=len(integrity),
        n_dem=n_dem_dir, n_con=n_con_dir,
        con_pos=len(control_old_pos),
        con_pos_other=len(control_old_pos_non_cand),
        non_cand=non_cand_rows,
        cand_rows=cand_table_rows,
        n_dir_rec=n_direct_records, n_rul_rec=n_rulers_records,
        n_excl=int(len(excl_test)),
        n_excl_r=int((excl_test["method"] == "rulers").sum()),
        n_excl_d=int((excl_test["method"] == "direct").sum()),
        n_tr_rows=n_train_rows, n_tr_valid=n_train_valid_rows,
        n_tr_inv=n_train_all_invalid, max_wdiff=max_wdiff,
        missing=missing_list, changed=changed_list, unchanged=unchanged_list,
        oba_n=rba["n"], oba_a=rba["Acc"], oba_s=rba["Sens"], oba_p=rba["Spec"],
        oba_f=rba["F1"], oba_u=rba["AUC"], oba_tp=rba["TP"], oba_tn=rba["TN"],
        oba_fp=rba["FP"], oba_fn=rba["FN"],
        nba_n=rbn["n"], nba_a=rbn["Acc"], nba_s=rbn["Sens"], nba_p=rbn["Spec"],
        nba_f=rbn["F1"], nba_u=rbn["AUC"], nba_tp=rbn["TP"], nba_tn=rbn["TN"],
        nba_fp=rbn["FP"], nba_fn=rbn["FN"],
        rA_t=fmt(rA[5], 2), rA_n=rA[6], rA_a=rA[0], rA_s=rA[1], rA_p=rA[2],
        rA_f=rA[3], rA_u=rA[4],
        rW_t=fmt(rW[5], 2), rW_n=rW[6], rW_a=rW[0], rW_s=rW[1], rW_p=rW[2],
        rW_f=rW[3], rW_u=rW[4],
        rL_t=fmt(rL[5], 2), rL_n=rL[6], rL_a=rL[0], rL_s=rL[1], rL_p=rL[2],
        rL_f=rL[3], rL_u=rL[4],
        rP_t=fmt(rP[5], 2), rP_n=rP[6], rP_a=rP[0], rP_s=rP[1], rP_p=rP[2],
        rP_f=rP[3], rP_u=rP[4],
        rH_t=fmt(rH[5], 2), rH_n=rH[6], rH_a=rH[0], rH_s=rH[1], rH_p=rH[2],
        rH_f=rH[3], rH_u=rH[4],
        cR_n=cR[6], cR_a=cR[0], cR_s=cR[1], cR_p=cR[2], cR_f=cR[3], cR_u=cR[4],
        cW_n=cW[6], cW_a=cW[0], cW_s=cW[1], cW_p=cW[2], cW_f=cW[3], cW_u=cW[4],
        cL_n=cL[6], cL_a=cL[0], cL_s=cL[1], cL_p=cL[2], cL_f=cL[3], cL_u=cL[4],
        cP_n=cP[6], cP_a=cP[0], cP_s=cP[1], cP_p=cP[2], cP_f=cP[3], cP_u=cP[4],
        cH_n=cH[6], cH_a=cH[0], cH_s=cH[1], cH_p=cH[2], cH_f=cH[3], cH_u=cH[4],
        cD_n=cD[6], cD_a=cD[0], cD_s=cD[1], cD_p=cD[2], cD_f=cD[3], cD_u=cD[4],
        hp_pt=hp["point_auc_diff"], hp_m=hp["bootstrap_mean_auc_diff"],
        hp_lo=hp["CI95_low"], hp_hi=hp["CI95_high"], hp_pos=hp["proportion_diff_gt_0"],
        hpr_pt=hpr["point_auc_diff"], hpr_m=hpr["bootstrap_mean_auc_diff"],
        hpr_lo=hpr["CI95_low"], hpr_hi=hpr["CI95_high"], hpr_pos=hpr["proportion_diff_gt_0"],
        hl_pt=hl["point_auc_diff"], hl_m=hl["bootstrap_mean_auc_diff"],
        hl_lo=hl["CI95_low"], hl_hi=hl["CI95_high"], hl_pos=hl["proportion_diff_gt_0"],
        hw_pt=hw["point_auc_diff"], hw_m=hw["bootstrap_mean_auc_diff"],
        hw_lo=hw["CI95_low"], hw_hi=hw["CI95_high"], hw_pos=hw["proportion_diff_gt_0"],
        hd_pt=hd["point_auc_diff"], hd_lo=hd["CI95_low"], hd_hi=hd["CI95_high"],
        dm_t=fmt(abl_comp[abl_comp["model"] == "demographic_only"]["threshold_train_youden"].iloc[0], 2),
        dm_a=abl_comp[abl_comp["model"] == "demographic_only"]["Acc"].iloc[0],
        dm_s=abl_comp[abl_comp["model"] == "demographic_only"]["Sens"].iloc[0],
        dm_p=abl_comp[abl_comp["model"] == "demographic_only"]["Spec"].iloc[0],
        dm_f=abl_comp[abl_comp["model"] == "demographic_only"]["F1"].iloc[0],
        dm_u=abl_comp[abl_comp["model"] == "demographic_only"]["AUC"].iloc[0],
        pd_t=fmt(prog_t, 2), pd_a=prog_m["Acc"], pd_s=prog_m["Sens"],
        pd_p=prog_m["Spec"], pd_f=prog_m["F1"], pd_u=prog_m["AUC"],
        pdm_t=fmt(abl["threshold_train_youden"], 2), pdm_a=abl["Acc"],
        pdm_s=abl["Sens"], pdm_p=abl["Spec"], pdm_f=abl["F1"], pdm_u=abl["AUC"],
        hdm_t=fmt(ablh["threshold_train_youden"], 2), hdm_a=ablh["Acc"],
        hdm_s=ablh["Sens"], hdm_p=ablh["Spec"], hdm_f=ablh["F1"], hdm_u=ablh["AUC"],
        ab_pt=ab["point_auc_diff"], ab_m=ab["bootstrap_mean_auc_diff"],
        ab_lo=ab["CI95_low"], ab_hi=ab["CI95_high"], ab_pos=ab["proportion_diff_gt_0"],
        fd_n=int(full_dev["n_samples"]), fd_subj=int(full_dev["n_subjects"]),
        fd_auc=full_dev["AUC"],
        pe_n=int(pe_dev["n_samples"]), pe_subj=int(pe_dev["n_subjects"]),
        pe_auc=pe_dev["AUC"],
        cs_n=int(cs_dev["n_samples"]), cs_subj=int(cs_dev["n_subjects"]),
        cs_auc=cs_dev["AUC"],
        rD_t=fmt(rD[5], 2), rD_n=rD[6], rD_a=rD[0], rD_s=rD[1], rD_p=rD[2],
        rD_f=rD[3], rD_u=rD[4],
        n_calls=plan_stats["n_calls_main"], worst=plan_stats["worst_case_with_retries"],
        tok_p=int(plan_stats["est_prompt_tokens_total_635_calls"]),
        tok_c=int(plan_stats["est_completion_tokens_total_635_calls"]),
        mins=plan_stats["est_elapsed_min_serial"],
        opid=plan_stats["opaque_id_example"],
    )


def build_impact_list(t0, integrity, missing_sids, changed_rows, affected_unchanged,
                      ba, ddf, abl_boot,
                      cand_in_train, cand_in_test, plan_stats):
    rba = ba[ba["rule"] == "old_rule_incl_zero_fill_127"].iloc[0]
    rbn = ba[ba["rule"] == "new_rule_missing_not_zero_123"].iloc[0]
    hp = ddf[ddf["comparison"] == "Full_Hybrid_minus_Programmatic-only LR"].iloc[0]
    hw = ddf[ddf["comparison"] == "Full_Hybrid_minus_Weighted RULERS (new rule)"].iloc[0]
    ab = abl_boot.iloc[0]
    return """# 影响清单（recording_time_recheck 对 recording_time_reanalysis 的修订）

**生成时间：** %(ts)s

## 一、撤回的表述（事实勘误，不影响任何计算）
1. 原报告 §3 与 freeze/cache_applicability_check.md 中"3 条测试候选（280-0/1/2）的
   ID 前缀仍是 dementia"——**撤回**。实际记录（metadata sample_id / source_group /
   audio_path 三层一致）为 Control 目录、`pitt_control_cookie_280-*` 前缀，
   与录音时点标签（control）一致。全部 19 条候选均为 Control 目录。
   该勘误不改变标签、纳入规则或任何性能数值。

## 二、降级为附录的结论
2. Direct 分支重新定性为"诊断相关 ID 暴露的历史基线"。原报告主比较表、
   bootstrap 主比较、开发接触表与参与者级分析中的 Direct 行全部转为
   明确标记的历史/诊断性附录（修订报告 §8），不作正式比较；
   **不推断暴露影响的方向或大小**。原报告结论第 3 条（Hybrid 相对 Direct）
   相应转为附录内容。

## 三、修订后的数值（Weighted RULERS）
3. 旧规则（全 NA 补 0）数值仅作协议对照：n=127，AUC=%(oba_u).4f。
4. 新规则（缺失不补 0）为修订结论：4 个样本（%(missing)s）因 5 次重复全 NA 记为缺失
   （n=123，AUC=%(nba_u).4f）；6 个样本中位数改用有效重复（%(changed)s）；
   另有 2 个样本（%(unchanged)s）含全 NA 重复但中位数不变（离散得分并列所致）。
   训练权重逐项差 = 0（不受影响）。
5. 共同样本（123）公平比较另行报告；Hybrid − Weighted RULERS（新规则）补充比较
   在共同样本框上计算：点差 %(hw_pt).4f，95%% CI [%(hw_lo).4f, %(hw_hi).4f]。

## 四、维持不变的结论
6. 核心结论：Hybrid − Programmatic-only 点差 %(hp_pt).4f，95%% CI [%(hp_lo).4f,
   %(hp_hi).4f]，差值>0 比例 %(hp_pos).3f —— 与冻结结果完全一致（确定性门），
   "LLM 特征相对 Programmatic-only 无正向增量证据"维持。
7. 三支 LR 拟合与阈值、Prompt RULERS Youden 阈值（0.07）、人口学敏感性三模型、
   开发接触排除（非 Direct 部分）均与冻结结果一致。
8. 标签规则、纳入/排除规则、训练/测试划分、QC 规则全部未变。

## 五、新增（事后探索，明确标记）
9. Programmatic-only + demographics（9 特征）LR：AUC 见修订报告 §6。
   Full Hybrid+人口学 − Programmatic+人口学 AUC 点差 %(ab_pt).4f，
   95%% CI [%(ab_lo).4f, %(ab_hi).4f]（20000 次配对聚类 bootstrap，seed 42）。
   事后新增，不据此调参。

## 六、未执行
10. Direct 干净重跑仅准备了方案（`direct_clean_rerun_plan.md`）：%(n_calls)s 次主调用
    （最坏含重试 %(worst)s 次），待作者逐项确认后另行执行。

## 七、保护声明
- `recording_time_reanalysis/` 及其全部产物、原始数据、旧结果、审计 v1/v2、论文
  均未修改（输入前后 SHA-256 比对 %(n_un)s/%(n_to)s 未变）。
- 本核查未调用 API、未根据结果调参。
""" % dict(
        ts=t0.strftime("%Y-%m-%d %H:%M:%S"),
        oba_u=rba["AUC"], nba_u=rbn["AUC"],
        missing=", ".join(missing_sids),
        changed=", ".join("%s" % r["sample_id"] for r in changed_rows),
        unchanged=", ".join(affected_unchanged),
        hw_pt=hw["point_auc_diff"], hw_lo=hw["CI95_low"], hw_hi=hw["CI95_high"],
        hp_pt=hp["point_auc_diff"], hp_lo=hp["CI95_low"], hp_hi=hp["CI95_high"],
        hp_pos=hp["proportion_diff_gt_0"],
        ab_pt=ab["point_auc_diff"], ab_lo=ab["CI95_low"], ab_hi=ab["CI95_high"],
        n_calls=plan_stats["n_calls_main"], worst=plan_stats["worst_case_with_retries"],
        n_un=sum(r["unchanged"] for r in integrity), n_to=len(integrity),
    )


def build_plan_doc(plan_stats, sys_prompt, user_tpl, build_prompt_fn, corrected_test):
    example_sid = "pitt_control_cookie_280-0"
    example_trans = ("What do you see going on in that picture? Oh, I see the sink "
                     "that's running over. I see the stores tipping over. The little "
                     "boys try and get cookies out. The girl is reaching to get a "
                     "cookie. The mother is drying dishes. The window's open. "
                     "[transcript abbreviated for display]")
    full_request = build_prompt_fn(example_trans, example_sid)
    sanitized_request = build_prompt_fn(example_trans, plan_stats["opaque_id_example"])
    return """# Direct 干净重跑方案（已准备，未执行）

**状态：** 仅方案。执行需作者逐项确认（见 §8）并另立目录，不改变任何冻结产物。
**生成时间：** %(ts)s

## 1. 目的与定位
去除 Direct 提示词中的诊断相关 ID 后重跑，以获得可正式比较的 Direct 分数。
**约束：** 若原模型快照无法匹配，新 Direct 与旧 Direct 不可直接比较；
新旧差异**不得全部归因于去除 ID**（见 §7）。

## 2. 实际请求内容核查（逐字段）
原请求由 `experiments/direct_llm_baseline.py` 构造，两段消息：

- **system**（%(sys_chars)s 字符）：任务说明 + 免责声明 + "只输出 JSON"。全文见文末附件 A。
- **user**（模板固定 %(fix_chars)s 字符 + 转录最多 3000 字符）：
  转录文本 + 评分要点 + **输出格式节含 `"sample_id": "{sample_id}"`**（真实 ID，
  内嵌由目录分组派生的 `pitt_dementia_cookie_*`/`pitt_control_cookie_*` 前缀）。
  全文见文末附件 B。
- **逐字段核查结论**：请求中除 `sample_id` 外不含目录、路径或诊断字段；
  `true_label` 仅存在于本地 checkpoint 记录（进度打印），从未进入请求；
  `input_text`（转录）不含元数据。**唯一的诊断相关暴露点 = sample_id 前缀。**
- 原运行实际转录长度：均值 %(tc_mean)s 字符、P95 %(tc_p95)s、最大 %(tc_max)s。

## 3. 清洗规范
1. `sample_id` 替换为**不透明 ID**（`uuid4().hex[:12]`，如 `%(opid)s`），
   无任何前缀语义；与真实 ID 的映射表（CSV）**仅在本地保存**
   （`direct_clean_rerun/opaque_id_map.csv`），绝不进入请求、绝不随结果上传。
2. 请求中不出现任何诊断相关 ID、目录、路径、元数据（原请求经 §2 核查仅需替换 sample_id；
   若转录中含人名等可识别信息，按 §8-4 确认后再发送）。
3. 提示词模板（清洗版）与映射表哈希存档，随结果发布复现信息。

## 4. 固定配置
| 项 | 值 |
|---|---|
| 模型标识 | 原运行记录 **deepseek-v4-flash**（final_clean_results 配置档案）；
 当前 config.yaml 默认 `deepseek-chat` —— **需作者确认锁定同一快照/版本**（见 §7） |
| 提示词 | 清洗版 system+user 模板（§3），模板哈希存档 |
| temperature | 0.3（原 run_config 不变） |
| max_tokens | 16384 |
| response_format | json_object + 自动回退（原 provider 行为） |
| timeout / max_retries | 120s / 2（原配置） |
| repeats | 5（同原协议） |
| seed | 42（注：seed 未传入 API，仅本地采样/进度用途；重复间变异来自模型随机性） |
| 样本范围 | 纠正后测试集 127 条（不含 12 条 QC 特征级排除与 11 条未决） |
| 阈值 | 0.50（预先指定的中性中点，不重定） |

## 5. 有效性与失败规则（预先冻结）
- 有效重复：`api_success=True` 且 `parse_success=True` 且无 `error_type` 且
  `risk_score` 有限且在 [0,1]（截断后）且转录非空（等价于原 `is_valid_row`）。
- 失败分类：RATE_LIMIT/API_ERROR → 重试（≤2 次，原配置）；其余失败 → 记录失败不重试。
- 样本聚合 = 有效重复的中位数；**样本无有效重复 → 缺失，不补 0**
  （与本核查 T3 新规则一致）。
- 全程无人工挑选阈值；测试结果不用于任何调参。

## 6. 调用次数与成本估计
- 主调用：**%(n_calls)s 次**（127 样本 × 5 重复）；最坏情况（每调用 2 次重试）：
  **%(worst)s 次**。
- token 估计（由实际缓存文本长度推算）：
  提示词 ≈ %(tok_p)s tokens（635 调用合计，单调用均值 ≈ %(tok_pm)s）；
  补全 ≈ %(tok_c)s tokens（合计）。
- 成本 = 官方单价 × 上述 tokens；**单价需作者在官方定价页确认**，
  本方案不报价。示例公式：总成本 ≈ (提示词 tokens × 输入单价 + 补全 tokens × 输出单价)。
- 预计耗时：约 %(mins)s 分钟（串行；原运行 1500 调用 / %(el_s)s 秒 ≈ 2.06 秒/调用）。

## 7. 模型版本无法匹配时的说明（必须遵守）
- 若无法锁定原 deepseek-v4-flash 快照：新 Direct 与旧 Direct 的差异同时包含
  "去除 ID"与"模型漂移"两个来源，**不能把新旧差异全部归因于去除 ID**。
- 可选补救（需作者决定，本方案不执行）：在同一新快照下加跑小规模 RULERS 校准子集
  （如 20 样本 × 1 重复）作为跨分支比较的参照；由此产生额外调用与成本。

## 8. 需要作者确认的具体事项
1. 原模型快照/版本标识能否锁定（deepseek-v4-flash 的准确标识与可用性）；
2. 预算确认（上述调用数与成本估计）；
3. 若模型快照不可匹配：接受"新基线"定位？是否加跑 RULERS 参照子集（§7）？
4. 转录再次外发的伦理/隐私确认（Cookie Theft 描述可能含可识别语音内容）；
5. 不透明 ID 映射表保管位置与权限（本地-only 承诺）；
6. 全 NA/无效重复处理确认（缺失不补 0；样本有效重复数下限 ≥1 还是 ≥2）；
7. 是否将完整 prompt 文本存档进 checkpoint（审计性 vs 体积）；
8. 并发与速率限制（原运行串行 ~2 秒/调用）。

## 附件 A：system 提示词全文
```
%(sys_full)s
```

## 附件 B：user 提示词模板全文
```
%(user_full)s
```

## 附件 C：一条真实请求示例（原版，含 ID 暴露点）
```
%(req_orig)s
```

## 附件 D：同一样本清洗版请求（不透明 ID）
```
%(req_clean)s
```
""" % dict(
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sys_chars=plan_stats["system_prompt_chars"],
        fix_chars=plan_stats["user_template_fixed_chars"],
        tc_mean=plan_stats["transcript_chars_mean"],
        tc_p95=plan_stats["transcript_chars_p95"],
        tc_max=plan_stats["transcript_chars_max"],
        opid=plan_stats["opaque_id_example"],
        n_calls=plan_stats["n_calls_main"],
        worst=plan_stats["worst_case_with_retries"],
        tok_p=int(plan_stats["est_prompt_tokens_total_635_calls"]),
        tok_pm=int(plan_stats["est_prompt_tokens_per_call_mean"]),
        tok_c=int(plan_stats["est_completion_tokens_total_635_calls"]),
        mins=plan_stats["est_elapsed_min_serial"],
        el_s=plan_stats["original_run_elapsed_s"],
        sys_full=sys_prompt,
        user_full=user_tpl,
        req_orig=json.dumps(full_request, ensure_ascii=False, indent=2),
        req_clean=json.dumps(sanitized_request, ensure_ascii=False, indent=2),
    )


if __name__ == "__main__":
    main()
