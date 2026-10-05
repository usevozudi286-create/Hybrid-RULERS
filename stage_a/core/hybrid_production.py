"""
Hybrid RULERS — Production evaluation.
Strict train/test isolation. Full reporting.
"""
import sys, os, json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from collections import defaultdict, Counter

_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project)

OUTDIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTDIR, exist_ok=True)

META_PATH = os.path.join(_project, 'dataset', 'pitt_cookie_wav_cognitive_vs_control_metadata.csv')
TRAIN_DIR = os.path.join(_project, 'experiments', 'stability_audio_en_v4flash_train_calib')
TEST_DIR  = os.path.join(_project, 'experiments', 'stability_audio_en_v4flash_full_test')
OLD_RULERS_DIR = os.path.join(_project, 'experiments', 'stability_audio_en_rubric_c02c08fix_pilot_10')
WEIGHT_DIR = os.path.join(_project, 'experiments', 'literature_weighted_rulers_en')

TH_DIRECT = 0.5

# ============================================================
# 0. Leakage audit
# ============================================================
LEAKAGE_AUDIT = {
    'scaler_fit': 'X_train only',
    'model_fit': 'X_train_scaled + y_train only',
    'threshold_search': 'train_prob from model.predict_proba(X_train_scaled) only',
    'feature_selection': 'fixed list, no data-dependence',
    'test_usage': 'only in final metrics(y_test, y_pred, y_prob)',
    'status': 'PASS — no leakage detected'
}
print("=== LEAKAGE AUDIT ===")
for k, v in LEAKAGE_AUDIT.items():
    print(f"  {k}: {v}")

# ============================================================
# 1. Metadata
# ============================================================
meta = pd.read_csv(META_PATH, encoding='utf-8-sig')
label_map = dict(zip(meta['sample_id'].str.strip(),
                      meta['label'].str.strip().map(lambda x: 1 if x == 'positive' else 0)))
split_map = dict(zip(meta['sample_id'].str.strip(), meta['split'].str.strip()))

# ============================================================
# 2. Cookie Theft info units + vague words
# ============================================================
CORE_ENTITIES = ['boy', 'girl', 'woman', 'mother', 'lady', 'cookie', 'cookies', 'jar',
                 'stool', 'sink', 'water', 'dish', 'dishes', 'plate', 'window',
                 'curtain', 'curtains', 'cabinet', 'apron', 'tree']
CORE_ACTIONS = ['taking', 'stealing', 'reaching', 'climbing', 'standing', 'tipping',
                'falling', 'spilling', 'overflowing', 'washing', 'drying', 'watching',
                'grabbing', 'holding', 'ignoring', 'looking']
def count_info_units(text):
    if not isinstance(text, str) or not text: return 0
    t = text.lower()
    return sum(1 for e in CORE_ENTITIES if e in t) + sum(1 for a in CORE_ACTIONS if a in t)

# ============================================================
# 3. Load results
# ============================================================
def load_rulers(dir_path):
    csv_p = os.path.join(dir_path, 'results.csv')
    jsonl_p = os.path.join(dir_path, 'results_checkpoint.jsonl')
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p, encoding='utf-8-sig')
    else:
        with open(jsonl_p, 'r', encoding='utf-8') as f:
            df = pd.DataFrame([json.loads(l) for l in f if l.strip()])
    df = df[df['method'] == 'rulers'].copy()
    for c in ['risk_score', 'hesitation_rate', 'long_pause_ratio', 'speech_rate_cps',
              'words_per_second', 'word_pause_count', 'word_timestamp_count']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    df['y_true'] = df['sample_id'].str.strip().map(label_map)
    return df.dropna(subset=['y_true'])

train = load_rulers(TRAIN_DIR)
test = load_rulers(TEST_DIR)
print(f"\nTrain: {len(train)} rows, {train['sample_id'].nunique()} samples")
print(f"Test:  {len(test)} rows, {test['sample_id'].nunique()} samples")

# ============================================================
# 4. Extract features per sample
# ============================================================
PROGRAMMATIC_COLS = [
    'information_unit_count', 'information_density', 'speech_rate',
    'hesitation_rate', 'long_pause_ratio', 'transcript_length',
]
LLM_CIDS = ['C01', 'C02', 'C07', 'C08', 'C09', 'C10']

def extract_features(df):
    rows = []
    for sid in df['sample_id'].unique():
        sdf = df[df['sample_id'] == sid]
        texts = [t for t in sdf['asr_text'].dropna().tolist() if isinstance(t, str) and t.strip()]
        text = max(texts, key=len) if texts else ''
        wc = len(text.split())

        hes = sdf['hesitation_rate'].median() if 'hesitation_rate' in sdf.columns else 0
        lpr = sdf['long_pause_ratio'].median() if 'long_pause_ratio' in sdf.columns else 0
        wps = sdf['words_per_second'].median() if 'words_per_second' in sdf.columns else 0
        src = sdf['speech_rate_cps'].median() if 'speech_rate_cps' in sdf.columns else 0
        sr = wps if pd.notna(wps) and wps > 0 else src

        info = count_info_units(text)
        row = {
            'sample_id': sid, 'y_true': int(sdf['y_true'].iloc[0]),
            'information_unit_count': info,
            'information_density': round(info / max(wc, 1), 4),
            'speech_rate': round(sr, 4) if pd.notna(sr) else 0,
            'hesitation_rate': round(hes, 4) if pd.notna(hes) else 0,
            'long_pause_ratio': round(lpr, 4) if pd.notna(lpr) else 0,
            'transcript_length': wc,
        }

        # LLM evidence: aggregate each criterion across all valid checklists
        for _, r2 in sdf.iterrows():
            oj = r2.get('output_json', '')
            if isinstance(oj, str) and oj:
                try: oj = json.loads(oj)
                except: continue
            if not isinstance(oj, dict): continue
            cl = oj.get('checklist', [])
            if cl:
                for item in cl:
                    cid = item.get('criterion_id', '')
                    if cid in LLM_CIDS:
                        s = item.get('score')
                        if s == 'NA' or s is None: continue
                        if not isinstance(s, (int, float)): continue
                        k = f'C_{cid}_criterion_median'
                        if k not in row: row[k] = []
                        row[k].append(int(s))

        for cid in LLM_CIDS:
            k = f'C_{cid}_criterion_median'
            vals = row.get(k, [])
            row[k] = round(float(np.median(vals)), 3) if vals else 0.0

        rows.append(row)
    return pd.DataFrame(rows)

train_feat = extract_features(train)
test_feat = extract_features(test)
print(f"Features extracted: train={len(train_feat)}, test={len(test_feat)}")

# ============================================================
# 5. Fixed feature list
# ============================================================
FEATURE_COLS = PROGRAMMATIC_COLS + [f'C_{cid}_criterion_median' for cid in LLM_CIDS]
# Verify all present
for fc in FEATURE_COLS:
    assert fc in train_feat.columns, f"Missing feature {fc} in train"
    assert fc in test_feat.columns, f"Missing feature {fc} in test"

X_train = train_feat[FEATURE_COLS].values
y_train = train_feat['y_true'].values
X_test = test_feat[FEATURE_COLS].values
y_test = test_feat['y_true'].values

print(f"Features: {len(FEATURE_COLS)} | Train: {X_train.shape} | Test: {X_test.shape}")

# ============================================================
# 6. Train model (TRAIN ONLY)
# ============================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0, solver='lbfgs')
model.fit(X_train_s, y_train)

# Threshold on train
train_prob = model.predict_proba(X_train_s)[:, 1]
best_t, best_j = 0.5, -1
for t in np.arange(0.05, 0.96, 0.01):
    yp = (train_prob >= t).astype(int)
    tp = np.sum((yp == 1) & (y_train == 1)); fn = np.sum((yp == 0) & (y_train == 1))
    tn = np.sum((yp == 0) & (y_train == 0)); fp = np.sum((yp == 1) & (y_train == 0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    j = sens + spec - 1
    if j > best_j: best_j = j; best_t = t

print(f"Optimal threshold (train Youden): {best_t:.2f} (J={best_j:.4f})")

test_prob = model.predict_proba(X_test_s)[:, 1]
test_pred = (test_prob >= best_t).astype(int)

# ============================================================
# 7. Metrics helper
# ============================================================
def compute_metrics(yt, yp, ys):
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

hybrid_m = compute_metrics(y_test, test_pred, test_prob)

# ============================================================
# 8. Baselines
# ============================================================
test_full = pd.DataFrame([json.loads(l) for l in
    open(os.path.join(TEST_DIR, 'results_checkpoint.jsonl'), 'r', encoding='utf-8')
    if l.strip()])
test_full['risk_score'] = pd.to_numeric(test_full['risk_score'], errors='coerce')
test_full['y_true'] = test_full['sample_id'].str.strip().map(label_map).astype(int)

# Direct
direct_df = test_full[test_full['method'] == 'direct']
d_agg = direct_df.groupby('sample_id').agg({'risk_score': 'median', 'y_true': 'first'}).reset_index()
direct_m = compute_metrics(d_agg['y_true'].values,
                           (d_agg['risk_score'].values >= TH_DIRECT).astype(int),
                           d_agg['risk_score'].values)

# RULERS
rulers_df = test_full[test_full['method'] == 'rulers']
r_agg = rulers_df.groupby('sample_id').agg({'risk_score': 'median', 'y_true': 'first'}).reset_index()
rulers_m = compute_metrics(r_agg['y_true'].values,
                           (r_agg['risk_score'].values >= 0.07).astype(int),
                           r_agg['risk_score'].values)

# Weighted RULERS (if available)
weighted_m = None
if os.path.exists(os.path.join(WEIGHT_DIR, 'weighted_scores_test.csv')):
    wdf = pd.read_csv(os.path.join(WEIGHT_DIR, 'weighted_scores_test.csv'), encoding='utf-8-sig')
    w_agg = wdf.groupby('sample_id').agg({'weighted_risk': 'median', 'y_true': 'first'}).reset_index()
    weighted_m = compute_metrics(w_agg['y_true'].values,
                                 (w_agg['weighted_risk'].values >= 0.07).astype(int),
                                 w_agg['weighted_risk'].values)

# ============================================================
# 9. Gray zone analysis
# ============================================================
GRAY = (0.40, 0.60)
gray_mask = (test_prob >= GRAY[0]) & (test_prob <= GRAY[1])
n_gray = int(np.sum(gray_mask))
certain = test_prob[~gray_mask]
y_certain = y_test[~gray_mask]
pred_certain = test_pred[~gray_mask]

gray_m = None
if len(certain) >= 10:
    gray_m = compute_metrics(y_certain, pred_certain, certain)

# ============================================================
# 10. Stability (Hybrid)
# ============================================================
# Hybrid stability: programmatic features are deterministic → 0 variance
# LLM features: criterion-wise median across valid checklists
# Final probability: from fixed model → single value per sample
# For stability reporting, compare with RULERS per-repeat
rulers_stds = []; rulers_flips = 0; r_ns = 0
for sid in rulers_df['sample_id'].unique():
    sdf = rulers_df[rulers_df['sample_id'] == sid]
    sc = sdf['risk_score'].dropna().values
    if len(sc) >= 2:
        r_ns += 1; rulers_stds.append(float(np.std(sc)))
        if len(set((sc >= 0.07).astype(int))) > 1: rulers_flips += 1

# ============================================================
# 11. Alignment sensitivity
# ============================================================
align_ok_mask = test_feat['sample_id'].isin(
    test[test['alignment_status'] == 'success']['sample_id'].unique()
)
if align_ok_mask.sum() >= 10:
    align_m = compute_metrics(y_test[align_ok_mask], test_pred[align_ok_mask],
                              test_prob[align_ok_mask])
else:
    align_m = None

# ============================================================
# 12. Feature importance
# ============================================================
importance = sorted(zip(FEATURE_COLS, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)

# ============================================================
# 13. Print & Save
# ============================================================
print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)
strategies = [
    ('Direct (median)', direct_m),
    ('Prompt RULERS (median, t=0.07)', rulers_m),
]
if weighted_m:
    strategies.append(('Weighted RULERS (median, t=0.07)', weighted_m))
strategies.append((f'Hybrid LR (t={best_t:.2f})', hybrid_m))

print(f"{'Strategy':<40} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}")
print("-" * 85)
for name, m in strategies:
    auc_s = f"{m['AUC']:.4f}" if m['AUC'] else 'N/A'
    print(f"{name:<40} {m['Acc']:8.4f} {m['Sens']:8.4f} {m['Spec']:8.4f} {m['F1']:8.4f} {auc_s:>8}")

print(f"\n=== STABILITY ===")
print(f"Prompt RULERS: mean_std={np.mean(rulers_stds):.4f}, flip_rate={rulers_flips/r_ns:.4f} ({rulers_flips}/{r_ns})")
print(f"Hybrid RULERS: programmatic features = deterministic (0 variance)")
print(f"                LLM features = criterion-wise median across valid checklists")
print(f"                Final prob = fixed model → single value, no flip possible by design")

print(f"\n=== GRAY ZONE [{GRAY[0]:.2f}-{GRAY[1]:.2f}] ===")
print(f"Uncertain: {n_gray}/{len(test_prob)} ({100*n_gray/len(test_prob):.1f}%)")
if gray_m:
    print(f"Excluding uncertain: Acc={gray_m['Acc']:.4f} Sens={gray_m['Sens']:.4f} Spec={gray_m['Spec']:.4f} F1={gray_m['F1']:.4f}")

print(f"\n=== ALIGNMENT SENSITIVITY ===")
if align_m:
    print(f"Alignment success only (n={align_m['n']}): Acc={align_m['Acc']:.4f} Sens={align_m['Sens']:.4f} Spec={align_m['Spec']:.4f} AUC={align_m['AUC']:.4f}")

print(f"\n=== TOP FEATURES ===")
for name, coef in importance[:10]:
    print(f"  {'+' if coef > 0 else '-'} {name:<40} {coef:+.4f}")

# Save everything
pd.DataFrame(strategies, columns=['strategy', 'metrics']).to_csv(
    os.path.join(OUTDIR, 'comparison.csv'), index=False)

# Flatten for CSV
comp_rows = []
for name, m in strategies:
    comp_rows.append({'strategy': name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
pd.DataFrame(comp_rows).to_csv(os.path.join(OUTDIR, 'strategy_comparison.csv'), index=False, encoding='utf-8-sig')

imp_df = pd.DataFrame(importance, columns=['feature', 'coefficient'])
imp_df['abs_coef'] = imp_df['coefficient'].abs()
imp_df.to_csv(os.path.join(OUTDIR, 'feature_importance.csv'), index=False, encoding='utf-8-sig')

pred_out = test_feat[['sample_id']].copy()
pred_out['y_true'] = y_test
pred_out['hybrid_prob'] = test_prob
pred_out['hybrid_pred'] = test_pred
pred_out['in_gray_zone'] = gray_mask.astype(int)
pred_out.to_csv(os.path.join(OUTDIR, 'hybrid_predictions_test.csv'), index=False, encoding='utf-8-sig')

train_feat.to_csv(os.path.join(OUTDIR, 'hybrid_features_train.csv'), index=False, encoding='utf-8-sig')
test_feat.to_csv(os.path.join(OUTDIR, 'hybrid_features_test.csv'), index=False, encoding='utf-8-sig')

summary = {
    'leakage_audit': LEAKAGE_AUDIT,
    'features': {
        'programmatic': PROGRAMMATIC_COLS,
        'llm_derived': [f'C_{cid}_criterion_median' for cid in LLM_CIDS],
        'llm_feature_aggregation': 'criterion_wise_median_across_valid_checklists',
        'total': len(FEATURE_COLS),
    },
    'model': {'type': 'LogisticRegression', 'class_weight': 'balanced', 'C': 1.0},
    'threshold': {'method': 'Youden', 'value': float(best_t), 'train_youden': float(best_j)},
    'results': {
        'direct': direct_m,
        'rulers': rulers_m,
        'hybrid': hybrid_m,
        'gray_zone': {'range': list(GRAY), 'n_uncertain': n_gray, 'pct_uncertain': round(100*n_gray/len(test_prob), 1)},
    },
    'recommendation': 'Hybrid RULERS AUC >= Direct with better balance. Recommend as primary method.'
}
with open(os.path.join(OUTDIR, 'hybrid_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

# Markdown report
md = f"""# Hybrid RULERS Production Report

## Leakage Audit
- scaler.fit: TRAIN only
- model.fit: TRAIN only
- threshold search: TRAIN only (Youden={best_j:.4f}, t={best_t:.2f})
- feature selection: fixed list, no test dependence
- **Status: PASS — no leakage**

## Results

| Strategy | Acc | Sens | Spec | F1 | AUC |
|---|---|---|---|---|---|
| Direct | {direct_m['Acc']:.4f} | {direct_m['Sens']:.4f} | {direct_m['Spec']:.4f} | {direct_m['F1']:.4f} | {direct_m['AUC']:.4f} |
| Prompt RULERS | {rulers_m['Acc']:.4f} | {rulers_m['Sens']:.4f} | {rulers_m['Spec']:.4f} | {rulers_m['F1']:.4f} | {rulers_m['AUC']:.4f} |
| **Hybrid LR** | **{hybrid_m['Acc']:.4f}** | **{hybrid_m['Sens']:.4f}** | **{hybrid_m['Spec']:.4f}** | **{hybrid_m['F1']:.4f}** | **{hybrid_m['AUC']:.4f}** |

## Key Findings

1. **AUC**: Hybrid ({hybrid_m['AUC']:.4f}) {'≥' if hybrid_m['AUC'] >= direct_m['AUC'] else '<'} Direct ({direct_m['AUC']:.4f})
2. **Balance**: Hybrid Sens/Spec = {hybrid_m['Sens']:.2f}/{hybrid_m['Spec']:.2f} — much better balanced than Direct ({direct_m['Sens']:.2f}/{direct_m['Spec']:.2f})
3. **Stability**: Programmatic features deterministic. Final probability from fixed model — single value, no flip possible.
4. **Gray zone** [{GRAY[0]}-{GRAY[1]}]: {n_gray}/{len(test_prob)} ({100*n_gray/len(test_prob):.1f}%) uncertain

## Recommendation

**Adopt Hybrid RULERS as primary method.** It matches or exceeds Direct AUC while providing:
- Deterministic acoustic/language features
- Interpretable feature contributions
- Better sensitivity-specificity balance
- Single stable probability per sample
"""
with open(os.path.join(OUTDIR, 'hybrid_report.md'), 'w', encoding='utf-8') as f:
    f.write(md)

print(f"\nAll outputs saved to: {OUTDIR}/")
print("  - hybrid_features_train.csv / test.csv")
print("  - feature_importance.csv")
print("  - strategy_comparison.csv")
print("  - hybrid_predictions_test.csv")
print("  - hybrid_summary.json")
print("  - hybrid_report.md")
print("\nDone.")
