"""Authoritative offline evaluation for the final Hybrid RULERS analysis.

This script consumes saved ASR/LLM records only. It does not call external
models. All paper-facing metrics must be regenerated from this file.
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2

_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_clean_results')
os.makedirs(OUTDIR, exist_ok=True)

TEST_DIR = os.path.join(_project, 'experiments', 'stability_audio_en_v4flash_full_test')
TRAIN_DIR = os.path.join(_project, 'experiments', 'stability_audio_en_v4flash_train_calib')
META_PATH = os.path.join(_project, 'dataset', 'pitt_cookie_wav_cognitive_vs_control_metadata.csv')
CP = os.path.join(TEST_DIR, 'results_checkpoint.jsonl')
RUBRIC_PATH = os.path.join(_project, 'mci_rubric_audio_en_v1.json')

TH_DIRECT = 0.5
TH_RULERS = 0.07


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None

# ============================================================
# 0. Config
# ============================================================
CONFIG = {
    'analysis_id': 'hybrid_rulers_final_clean_v2',
    'model': 'deepseek-v4-flash',
    'temperature': 0.3,
    'seed': 42,
    'whisper': 'small (CPU)',
    'whisper_backend': 'faster-whisper',
    'whisper_language': 'en',
    'whisper_compute_type': 'int8',
    'whisper_beam_size': 5,
    'whisper_word_timestamps': True,
    'rubric': 'mci_rubric_audio_en_v1.json',
    'rubric_sha256': sha256_file(RUBRIC_PATH),
    'alignment': 'WhisperX, CPU, model=auto',
    'alignment_checkpoint_provenance': 'language-specific default selected by WhisperX; exact checkpoint was not logged',
    'max_tokens': 16384,
    'direct_threshold': TH_DIRECT,
    'direct_threshold_origin': 'prespecified neutral probability midpoint; no Direct training outputs were available for calibration',
    'prompt_rulers_threshold': TH_RULERS,
    'prompt_rulers_threshold_origin': 'lowest maximizer of training-set Youden index on grid 0.05 to 0.95 by 0.01',
    'test_repeats_scheduled_per_method': 5,
    'train_rulers_repeats_scheduled': 1,
    'speech_rate_definition': 'aligned_word_count / (last_word_end - first_word_start)',
    'hesitation_rate_definition': 'word_gaps_over_0.5s / aligned_word_count',
    'long_pause_ratio_definition': 'total_energy_detected_pause_time / audio_duration',
    'software_versions_at_final_audit': {
        name: installed_version(name) for name in [
            'faster-whisper', 'whisperx', 'scikit-learn', 'pandas',
            'numpy', 'openai', 'torch', 'librosa'
        ]
    },
    'software_version_note': 'captured from the project virtual environment during final audit; original run_config did not archive package versions',
}

# ============================================================
# 1. Filter criteria
# ============================================================
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

def excluded_reason(r):
    reasons = []
    if not r.get('api_success', False): reasons.append('api_success=False')
    if not r.get('parse_success', False): reasons.append('parse_success=False')
    et = r.get('error_type')
    if et is not None and not pd.isna(et) and str(et).strip():
        reasons.append('error_type=' + str(et))
    asr_value = r.get('asr_text', '')
    asr = asr_value if isinstance(asr_value, str) else ''
    if len(asr.strip()) == 0: reasons.append('empty_asr')
    try:
        wtc = float(r.get('word_timestamp_count', 0))
    except (TypeError, ValueError):
        wtc = 0
    if not np.isfinite(wtc) or wtc <= 0: reasons.append(f'word_timestamp_count={wtc}')
    if r.get('alignment_status', '') == 'failed': reasons.append('alignment=failed')
    return '; '.join(reasons) if reasons else 'unknown'

# ============================================================
# 2. Load & filter
# ============================================================
print("=== LOADING ===")
with open(CP, 'r', encoding='utf-8') as f:
    all_rows = [json.loads(l) for l in f if l.strip()]
print(f"Total records: {len(all_rows)}")

# Filter
valid_rows = [r for r in all_rows if is_valid_row(r)]
excluded_rows = [r for r in all_rows if not is_valid_row(r)]
print(f"Valid: {len(valid_rows)} | Excluded: {len(excluded_rows)}")

# Build valid df
df = pd.DataFrame(valid_rows)
meta = pd.read_csv(META_PATH, encoding='utf-8-sig')
label_map = dict(zip(meta['sample_id'].str.strip(), meta['label'].str.strip().map(lambda x: 1 if x=='positive' else 0)))
sample_subject_map = dict(zip(meta['sample_id'].astype(str).str.strip(), meta['subject_id'].astype(str).str.strip()))
train_subjects = set(meta.loc[meta['split'].eq('train'), 'subject_id'].astype(str).str.strip())
test_subjects = set(meta.loc[meta['split'].eq('test'), 'subject_id'].astype(str).str.strip())
subject_overlap = train_subjects & test_subjects
if subject_overlap:
    raise ValueError('Train/test participant overlap: ' + ', '.join(sorted(subject_overlap)))
df['y_true'] = df['sample_id'].str.strip().map(label_map)
df = df.dropna(subset=['y_true']); df['y_true'] = df['y_true'].astype(int)

print(f"After label merge: {len(df)} rows, {df['sample_id'].nunique()} samples")

# ============================================================
# 3. Common test set: exclude samples with 0 valid Direct OR 0 valid RULERS
# ============================================================
direct_valid = set(df[df['method']=='direct']['sample_id'].unique())
rulers_valid = set(df[df['method']=='rulers']['sample_id'].unique())
common_samples = direct_valid & rulers_valid
all_test_samples = set(meta.loc[meta['split'].eq('test'), 'sample_id'].astype(str).str.strip())
excluded_from_common = all_test_samples - common_samples
total_test_samples = 150

print(f"\n=== COMMON TEST SET ===")
print(f"Direct valid samples: {len(direct_valid)}")
print(f"RULERS valid samples: {len(rulers_valid)}")
print(f"Common samples: {len(common_samples)}")
print(f"Excluded from common: {len(excluded_from_common)}")
for s in sorted(excluded_from_common):
    d_ok = s in direct_valid
    r_ok = s in rulers_valid
    d_rows = [r for r in excluded_rows if r['sample_id']==s and r['method']=='direct']
    r_rows = [r for r in excluded_rows if r['sample_id']==s and r['method']=='rulers']
    reasons = set()
    for rx in d_rows + r_rows:
        reasons.add(excluded_reason(rx))
    print(f"  {s}: direct_valid={d_ok} rulers_valid={r_ok} reasons={reasons}")

# Filter to common
df_common = df[df['sample_id'].isin(common_samples)].copy()
print(f"\nCommon analysis set: {len(df_common)} rows, {len(common_samples)} samples")

# Per-sample+method repeat counts
for m in ['direct','rulers']:
    mdf = df_common[df_common['method']==m]
    rc = mdf.groupby('sample_id').size()
    print(f"{m}: {rc.min()}-{rc.max()} repeats/sample, mean={rc.mean():.1f}")
    low = rc[rc < 5]
    if len(low) > 0:
        print(f"  Low-repeat ({len(low)} samples):")
        for sid, n in sorted(low.items()):
            print(f"    {sid}: {n}/5")

# ============================================================
# 4. Baseline metrics (Direct + RULERS risk-score median)
# ============================================================
def compute_metrics(yt, yp, ys=None):
    tp=int(np.sum((yp==1)&(yt==1))); tn=int(np.sum((yp==0)&(yt==0)))
    fp=int(np.sum((yp==1)&(yt==0))); fn=int(np.sum((yp==0)&(yt==1)))
    n=len(yt); sens=tp/(tp+fn)if(tp+fn)>0 else 0.0; spec=tn/(tn+fp)if(tn+fp)>0 else 0.0
    prec=tp/(tp+fp)if(tp+fp)>0 else 0.0; acc=(tp+tn)/n
    f1=2*prec*sens/(prec+sens)if(prec+sens)>0 else 0.0
    auc=roc_auc_score(yt,ys)if ys is not None and len(set(yt))>=2 else None
    return {'Acc':acc,'Sens':sens,'Spec':spec,'F1':f1,'AUC':auc,'n':n,'TP':tp,'TN':tn,'FP':fp,'FN':fn}

# Direct risk-score median
d_df = df_common[df_common['method']=='direct']
d_agg = d_df.groupby('sample_id').agg({'risk_score':'median','y_true':'first'}).reset_index()
d_yp = (d_agg['risk_score'].values>=TH_DIRECT).astype(int)
d_m = compute_metrics(d_agg['y_true'].values, d_yp, d_agg['risk_score'].values)

# RULERS risk-score median
r_df = df_common[df_common['method']=='rulers']
r_agg = r_df.groupby('sample_id').agg({'risk_score':'median','y_true':'first'}).reset_index()
r_yp = (r_agg['risk_score'].values>=TH_RULERS).astype(int)
r_m = compute_metrics(r_agg['y_true'].values, r_yp, r_agg['risk_score'].values)


def repeated_score_stability(method_df, threshold):
    flip_flags = []
    score_stds = []
    for _, sample_rows in method_df.groupby('sample_id'):
        scores = pd.to_numeric(sample_rows['risk_score'], errors='coerce').dropna().to_numpy()
        if len(scores) < 2:
            continue
        labels = (scores >= threshold).astype(int)
        flip_flags.append(int(len(set(labels)) > 1))
        score_stds.append(float(np.std(scores, ddof=0)))
    return {
        'n_samples_with_at_least_2_repeats': len(flip_flags),
        'label_flip_rate': float(np.mean(flip_flags)) if flip_flags else np.nan,
        'mean_score_std': float(np.mean(score_stds)) if score_stds else np.nan,
        'std_ddof': 0,
    }


stability = {
    'Direct LLM': repeated_score_stability(d_df, TH_DIRECT),
    'Prompt RULERS': repeated_score_stability(r_df, TH_RULERS),
    'Full Hybrid LR (fixed feature vector)': {
        'n_samples_with_at_least_2_repeats': len(common_samples),
        'label_flip_rate': 0.0,
        'mean_score_std': 0.0,
        'std_ddof': 0,
        'scope': 'conditional downstream determinism; not end-to-end repeatability',
    },
}

# ============================================================
# 5. Hybrid features (same logic as hybrid_production)
# ============================================================
# Each group represents one lexicon-based information unit. Exact token
# matching avoids substring collisions, while grouped variants avoid counting
# singular/plural forms or common referential synonyms twice.
INFORMATION_UNIT_GROUPS = {
    'boy': {'boy'},
    'girl': {'girl'},
    'woman': {'woman', 'mother', 'lady'},
    'cookie': {'cookie', 'cookies'},
    'jar': {'jar'},
    'stool': {'stool'},
    'sink': {'sink'},
    'water': {'water'},
    'dish': {'dish', 'dishes', 'plate', 'plates'},
    'window': {'window'},
    'curtain': {'curtain', 'curtains'},
    'cabinet': {'cabinet'},
    'apron': {'apron'},
    'tree': {'tree'},
    'taking': {'take', 'takes', 'taking', 'took'},
    'stealing': {'steal', 'steals', 'stealing', 'stole'},
    'reaching': {'reach', 'reaches', 'reaching', 'reached'},
    'climbing': {'climb', 'climbs', 'climbing', 'climbed'},
    'standing': {'stand', 'stands', 'standing', 'stood'},
    'tipping': {'tip', 'tips', 'tipping', 'tipped'},
    'falling': {'fall', 'falls', 'falling', 'fell'},
    'spilling': {'spill', 'spills', 'spilling', 'spilled'},
    'overflowing': {'overflow', 'overflows', 'overflowing', 'overflowed'},
    'washing': {'wash', 'washes', 'washing', 'washed'},
    'drying': {'dry', 'dries', 'drying', 'dried'},
    'watching': {'watch', 'watches', 'watching', 'watched'},
    'grabbing': {'grab', 'grabs', 'grabbing', 'grabbed'},
    'holding': {'hold', 'holds', 'holding', 'held'},
    'ignoring': {'ignore', 'ignores', 'ignoring', 'ignored'},
    'looking': {'look', 'looks', 'looking', 'looked'},
}
LLM_CIDS = ['C01','C02','C07','C08','C09','C10']
PROG_COLS = [
    'information_unit_count', 'information_density', 'speech_rate',
    'hesitation_rate', 'long_pause_ratio', 'transcript_length',
]

def tokenize_english(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def count_info(text):
    tokens = set(tokenize_english(text))
    return sum(bool(tokens & variants) for variants in INFORMATION_UNIT_GROUPS.values())


def parse_word_timestamps(value):
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str) and value.strip():
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return []
    else:
        return []
    words = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item['start'])
            end = float(item['end'])
        except (KeyError, TypeError, ValueError):
            continue
        if end >= start:
            words.append((start, end))
    return sorted(words)


def aligned_words_per_second(row):
    words = parse_word_timestamps(row.get('word_timestamps_json', ''))
    if len(words) < 2:
        return np.nan
    duration = words[-1][1] - words[0][0]
    return len(words) / duration if duration > 0 else np.nan

def extract_hybrid_features(rulers_df_clean):
    """Extract hybrid features from ALREADY FILTERED rulers data."""
    rows = []
    for sid in sorted(rulers_df_clean['sample_id'].unique()):
        sdf = rulers_df_clean[rulers_df_clean['sample_id']==sid]
        texts = [t for t in sdf['asr_text'].dropna().tolist() if isinstance(t,str) and len(t.strip())>0]
        text = max(texts, key=len) if texts else ''
        wc = len(tokenize_english(text))
        if wc == 0: continue  # SKIP if no valid text
        hes = pd.to_numeric(sdf['hesitation_rate'], errors='coerce').median()
        lpr = pd.to_numeric(sdf['long_pause_ratio'], errors='coerce').median()
        aligned_rates = sdf.apply(aligned_words_per_second, axis=1)
        sr_val = pd.to_numeric(aligned_rates, errors='coerce').median()
        info = count_info(text)
        row = {
            'sample_id': sid, 'y_true': int(sdf['y_true'].iloc[0]),
            'information_unit_count': info, 'information_density': round(info/max(wc,1),4),
            'speech_rate': round(sr_val,4) if pd.notna(sr_val) else 0,
            'hesitation_rate': round(hes,4) if pd.notna(hes) else 0,
            'long_pause_ratio': round(lpr,4) if pd.notna(lpr) else 0,
            'transcript_length': wc,
        }
        # LLM evidence: aggregate each criterion across all valid checklists
        for _, r2 in sdf.iterrows():
            oj = r2.get('output_json','')
            if isinstance(oj,str) and oj:
                try: oj = json.loads(oj)
                except: continue
            if isinstance(oj,dict) and oj.get('checklist'):
                for item in oj['checklist']:
                    cid = item.get('criterion_id','')
                    if cid in LLM_CIDS:
                        s = item.get('score')
                        if s!='NA' and s is not None and isinstance(s,(int,float)):
                            k = f'C_{cid}_criterion_median'
                            if k not in row: row[k] = []
                            row[k].append(int(s))
        for cid in LLM_CIDS:
            k = f'C_{cid}_criterion_median'
            vals = row.get(k,[])
            row[k] = round(float(np.median(vals)),3) if vals else np.nan
        rows.append(row)
    features = pd.DataFrame(rows)
    if features['speech_rate'].isna().any():
        bad = features.loc[features['speech_rate'].isna(), 'sample_id'].astype(str)
        raise ValueError('Cannot derive aligned speech rate for: ' + ', '.join(bad))
    return features

# Also extract train features (same way, from filtered train data)
train_raw = pd.read_csv(os.path.join(TRAIN_DIR, 'results.csv'), encoding='utf-8-sig') if os.path.exists(os.path.join(TRAIN_DIR, 'results.csv')) else pd.DataFrame()
if train_raw.empty and os.path.exists(os.path.join(TRAIN_DIR, 'results_checkpoint.jsonl')):
    with open(os.path.join(TRAIN_DIR, 'results_checkpoint.jsonl'), 'r', encoding='utf-8') as f:
        train_raw = pd.DataFrame([json.loads(l) for l in f if l.strip()])
train_raw = train_raw[train_raw['method']=='rulers'].copy()
for c in ['hesitation_rate','long_pause_ratio','speech_rate_cps','words_per_second']:
    if c in train_raw.columns: train_raw[c] = pd.to_numeric(train_raw[c], errors='coerce')
# Filter train same way
train_invalid_rows = [r.to_dict() for _, r in train_raw.iterrows() if not is_valid_row(r.to_dict())]
train_valid = [r for _, r in train_raw.iterrows() if is_valid_row(r.to_dict())]
train_clean = pd.DataFrame(train_valid) if train_valid else train_raw
train_clean['y_true'] = train_clean['sample_id'].str.strip().map(label_map)
train_clean = train_clean.dropna(subset=['y_true']); train_clean['y_true'] = train_clean['y_true'].astype(int)

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

train_rulers_threshold, train_rulers_youden = select_youden_threshold(
    train_clean['y_true'].to_numpy(dtype=int),
    pd.to_numeric(train_clean['risk_score'], errors='coerce').to_numpy(dtype=float),
)
if train_rulers_threshold != TH_RULERS:
    raise AssertionError(
        f'Prompt RULERS threshold drift: expected {TH_RULERS}, '
        f'training Youden selected {train_rulers_threshold}'
    )
CONFIG['prompt_rulers_training_youden'] = round(train_rulers_youden, 4)

test_hybrid = extract_hybrid_features(r_df)
train_hybrid = extract_hybrid_features(train_clean[train_clean['method']=='rulers'])

print(f"\nHybrid features: train={len(train_hybrid)} test={len(test_hybrid)}")

FEATURE_COLS = PROG_COLS + [f'C_{cid}_criterion_median' for cid in LLM_CIDS]
for fc in FEATURE_COLS:
    assert fc in train_hybrid.columns, f"Missing {fc} in train"
    assert fc in test_hybrid.columns, f"Missing {fc} in test"

# ============================================================
# 6. Train models
# ============================================================
def train_lr(Xt, yt, Xe, ye, cols):
    imputer = SimpleImputer(strategy='median')
    Xtr = imputer.fit_transform(Xt[cols].values)
    Xev = imputer.transform(Xe[cols].values)
    scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xev_s = scaler.transform(Xev)
    model = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0, solver='lbfgs')
    model.fit(Xtr_s, yt)
    tp = model.predict_proba(Xtr_s)[:,1]
    best_t, best_j = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        yp = (tp>=t).astype(int)
        sn = np.sum((yp==1)&(yt==1))/(np.sum((yp==1)&(yt==1))+np.sum((yp==0)&(yt==1))+1e-10)
        sp = np.sum((yp==0)&(yt==0))/(np.sum((yp==0)&(yt==0))+np.sum((yp==1)&(yt==0))+1e-10)
        j = sn+sp-1
        if j>best_j: best_j=j; best_t=t
    ep = model.predict_proba(Xev_s)[:,1]
    epred = (ep>=best_t).astype(int)
    imputation_values = {
        col: float(value) for col, value in zip(cols, imputer.statistics_)
    }
    return compute_metrics(ye, epred, ep), epred, ep, best_t, model.coef_[0], imputation_values

ytrain = train_hybrid['y_true'].values
ytest = test_hybrid['y_true'].values

full_m, full_pred, full_prob, full_t, full_coef, full_imputation = train_lr(
    train_hybrid, ytrain, test_hybrid, ytest, FEATURE_COLS)
prog_m, prog_pred, prog_prob, prog_t, prog_coef, _ = train_lr(
    train_hybrid, ytrain, test_hybrid, ytest, PROG_COLS)
llm_m, llm_pred, llm_prob, llm_t, llm_coef, _ = train_lr(
    train_hybrid, ytrain, test_hybrid, ytest,
    [f'C_{cid}_criterion_median' for cid in LLM_CIDS])

# Weighted RULERS
weights_path = os.path.join(_project, 'experiments', 'literature_weighted_rulers_en', 'learned_weights.json')
w_m = None
if os.path.exists(weights_path):
    with open(weights_path) as f: wdata = json.load(f)
    weights = wdata['weights']
    # Compute weighted score for test
    w_scores = []
    for _, row in r_df.iterrows():
        oj = row.get('output_json','')
        if isinstance(oj,str) and oj:
            try: oj = json.loads(oj)
            except: continue
        if not isinstance(oj,dict) or not oj.get('checklist'): continue
        ws, wsum = 0, 0
        for item in oj['checklist']:
            cid = item.get('criterion_id',''); s = item.get('score')
            if s=='NA' or s is None or not isinstance(s,(int,float)): continue
            w = weights.get(cid, 1.0); ws += s*w; wsum += w
        if wsum > 0: w_scores.append({'sample_id': row['sample_id'], 'weighted_risk': (ws/wsum)/2.0, 'y_true': row['y_true']})
    if w_scores:
        wdf = pd.DataFrame(w_scores)
        w_agg = wdf.groupby('sample_id').agg({'weighted_risk':'median','y_true':'first'}).reset_index()
        w_yp = (w_agg['weighted_risk'].values>=TH_RULERS).astype(int)
        w_m = compute_metrics(w_agg['y_true'].values, w_yp, w_agg['weighted_risk'].values)

# ============================================================
# 7. Bootstrap CI
# ============================================================
N_BOOT = 2000; SEED = 42; rng = np.random.RandomState(SEED)
d_yt_bs = d_agg['y_true'].values; d_ys_bs = d_agg['risk_score'].values; d_yp_bs = (d_ys_bs>=TH_DIRECT).astype(int)
h_yt_bs = test_hybrid['y_true'].values; h_ys_bs = full_prob; h_yp_bs = full_pred

def bootstrap_ci(yt, yp, ys, nboot, rng):
    n = len(yt); metrics = []
    for _ in range(nboot):
        idx = rng.choice(n, size=n, replace=True)
        m = compute_metrics(yt[idx], yp[idx], ys[idx])
        metrics.append(m)
    return metrics

d_bs = bootstrap_ci(d_yt_bs, d_yp_bs, d_ys_bs, N_BOOT, rng)
h_bs = bootstrap_ci(h_yt_bs, h_yp_bs, h_ys_bs, N_BOOT, rng)

def ci95(vals):
    return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

boot_rows = []
for metric in ['Acc','Sens','Spec','F1','AUC']:
    dv = [m[metric] for m in d_bs if m[metric] is not None]
    hv = [m[metric] for m in h_bs if m[metric] is not None]
    dm,dl,dh = ci95(dv); hm,hl,hh = ci95(hv)
    boot_rows.append({'Metric':metric, 'Direct_mean':round(dm,4),'Direct_CI95_low':round(dl,4),'Direct_CI95_high':round(dh,4),
                      'Hybrid_mean':round(hm,4),'Hybrid_CI95_low':round(hl,4),'Hybrid_CI95_high':round(hh,4)})

# AUC difference: align the two score vectors by sample_id before paired resampling.
# The extraction paths do not guarantee the same row order.
paired_scores = d_agg[['sample_id', 'y_true', 'risk_score']].merge(
    test_hybrid[['sample_id']].assign(hybrid_prob=full_prob),
    on='sample_id', how='inner', validate='one_to_one').reset_index(drop=True)
paired_yt = paired_scores['y_true'].values
paired_direct = paired_scores['risk_score'].values
paired_hybrid = paired_scores['hybrid_prob'].values
auc_diff_point = roc_auc_score(paired_yt, paired_hybrid) - roc_auc_score(paired_yt, paired_direct)
paired_rng = np.random.RandomState(SEED)
auc_diffs = []
for _ in range(N_BOOT):
    idx = paired_rng.choice(len(paired_yt), size=len(paired_yt), replace=True)
    da = roc_auc_score(paired_yt[idx], paired_direct[idx]) if len(set(paired_yt[idx]))>=2 else 0.5
    ha = roc_auc_score(paired_yt[idx], paired_hybrid[idx]) if len(set(paired_yt[idx]))>=2 else 0.5
    auc_diffs.append(ha-da)
_, ad_l, ad_h = ci95(auc_diffs)
ad_m = auc_diff_point
p_pos = np.mean(np.array(auc_diffs)>0)

# Participant-clustered paired bootstrap. Each sampled participant contributes
# all of their retained recordings, preserving within-participant dependence.
paired_scores['subject_id'] = paired_scores['sample_id'].map(sample_subject_map)
if paired_scores['subject_id'].isna().any():
    raise ValueError('Missing subject_id for one or more paired samples')

cluster_groups = {
    subject_id: group.index.to_numpy()
    for subject_id, group in paired_scores.groupby('subject_id', sort=True)
}
cluster_ids = np.array(sorted(cluster_groups))
cluster_rng = np.random.RandomState(SEED)
cluster_direct_metrics = []
cluster_hybrid_metrics = []
cluster_auc_diffs = []
for _ in range(N_BOOT):
    sampled_clusters = cluster_rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
    idx = np.concatenate([cluster_groups[s] for s in sampled_clusters])
    yt = paired_yt[idx]
    if len(set(yt)) < 2:
        continue
    d_scores = paired_direct[idx]
    h_scores = paired_hybrid[idx]
    d_metrics = compute_metrics(yt, (d_scores >= TH_DIRECT).astype(int), d_scores)
    h_metrics = compute_metrics(yt, (h_scores >= full_t).astype(int), h_scores)
    cluster_direct_metrics.append(d_metrics)
    cluster_hybrid_metrics.append(h_metrics)
    cluster_auc_diffs.append(h_metrics['AUC'] - d_metrics['AUC'])

cluster_boot_rows = []
for metric in ['Acc','Sens','Spec','F1','AUC']:
    dv = [m[metric] for m in cluster_direct_metrics if m[metric] is not None]
    hv = [m[metric] for m in cluster_hybrid_metrics if m[metric] is not None]
    dm,dl,dh = ci95(dv); hm,hl,hh = ci95(hv)
    cluster_boot_rows.append({
        'Metric': metric,
        'Direct_mean': round(dm,4), 'Direct_CI95_low': round(dl,4), 'Direct_CI95_high': round(dh,4),
        'Hybrid_mean': round(hm,4), 'Hybrid_CI95_low': round(hl,4), 'Hybrid_CI95_high': round(hh,4),
    })
_, cluster_ad_l, cluster_ad_h = ci95(cluster_auc_diffs)
cluster_p_pos = np.mean(np.array(cluster_auc_diffs) > 0)

# McNemar
d_sample_preds = dict(zip(d_agg['sample_id'].values, d_yp_bs))
h_sample_preds = dict(zip(test_hybrid['sample_id'].values, h_yp_bs))
common_ids = sorted(set(d_sample_preds.keys()) & set(h_sample_preds.keys()))
dp = np.array([d_sample_preds[s] for s in common_ids])
hp = np.array([h_sample_preds[s] for s in common_ids])
common_yt = np.array([d_agg[d_agg['sample_id']==s]['y_true'].iloc[0] for s in common_ids])
d_err = (dp != common_yt).astype(int); h_err = (hp != common_yt).astype(int)
b = int(np.sum((d_err==1)&(h_err==0))); c = int(np.sum((d_err==0)&(h_err==1)))
mcn_stat = (abs(b-c)-1)**2/(b+c) if (b+c)>0 else 0
mcn_p = 1 - chi2.cdf(mcn_stat, 1) if (b+c)>0 else 1.0

# Subject-level
subj_to_samples = defaultdict(list)
for _, r in meta.iterrows():
    subj_to_samples[str(r['subject_id']).strip()].append(str(r['sample_id']).strip())
subj_label = dict(zip(meta['subject_id'].astype(str).str.strip(), meta['label'].str.strip().map(lambda x: 1 if x=='positive' else 0)))

d_sample_scores = dict(zip(d_agg['sample_id'].values, d_agg['risk_score'].values))
h_sample_scores = dict(zip(test_hybrid['sample_id'].values, full_prob))
subj_rows = []
for subj, samps in subj_to_samples.items():
    if subj not in subj_label: continue
    tl = subj_label[subj]
    ds = [d_sample_scores[s] for s in samps if s in d_sample_scores]
    hs = [h_sample_scores[s] for s in samps if s in h_sample_scores]
    if ds and hs:
        subj_rows.append({'subject_id':subj,'y_true':tl,'direct_median':np.median(ds),'hybrid_median':np.median(hs)})
subj_df = pd.DataFrame(subj_rows)
d_sj_yp = (subj_df['direct_median'].values>=TH_DIRECT).astype(int)
h_sj_yp = (subj_df['hybrid_median'].values>=0.5).astype(int)
d_sj_m = compute_metrics(subj_df['y_true'].values, d_sj_yp, subj_df['direct_median'].values)
h_sj_m = compute_metrics(subj_df['y_true'].values, h_sj_yp, subj_df['hybrid_median'].values)

# Gray zone
GRAY = (0.40, 0.60)
gray_mask = (full_prob>=0.40)&(full_prob<=0.60)
ng = int(np.sum(gray_mask))
cert_mask = ~gray_mask
cert_m = compute_metrics(ytest[cert_mask], full_pred[cert_mask], full_prob[cert_mask]) if cert_mask.sum()>=10 else None

# Sensitivity analysis restricted to samples with all five valid repeats in
# both prompt-based branches. The fitted model and threshold remain unchanged.
direct_repeat_counts = d_df.groupby('sample_id').size()
rulers_repeat_counts = r_df.groupby('sample_id').size()
complete_repeat_ids = sorted(
    set(direct_repeat_counts[direct_repeat_counts == 5].index) &
    set(rulers_repeat_counts[rulers_repeat_counts == 5].index)
)
complete_scores = paired_scores[paired_scores['sample_id'].isin(complete_repeat_ids)].copy()
complete_direct = compute_metrics(
    complete_scores['y_true'].to_numpy(),
    (complete_scores['risk_score'].to_numpy() >= TH_DIRECT).astype(int),
    complete_scores['risk_score'].to_numpy(),
)
complete_hybrid = compute_metrics(
    complete_scores['y_true'].to_numpy(),
    (complete_scores['hybrid_prob'].to_numpy() >= full_t).astype(int),
    complete_scores['hybrid_prob'].to_numpy(),
)

# ============================================================
# 8. Print & Save
# ============================================================
print(f"\n{'='*80}")
print(f"FINAL CLEAN RESULTS (n={len(common_samples)} samples)")
print(f"{'='*80}")
strategies = [
    ('Direct (median)', d_m),
    ('Prompt RULERS (median, t=0.07)', r_m),
]
if w_m: strategies.append(('Weighted RULERS', w_m))
strategies += [
    ('LLM-only LR', llm_m),
    ('Programmatic-only LR', prog_m),
    ('Full Hybrid LR', full_m),
]
print(f"{'Method':<35} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}")
print('-'*80)
for name, m in strategies:
    a = f"{m['AUC']:.4f}" if m['AUC'] else 'N/A'
    print(f"{name:<35} {m['Acc']:8.4f} {m['Sens']:8.4f} {m['Spec']:8.4f} {m['F1']:8.4f} {a:>8}")

print(f"\n--- Bootstrap 95% CI ---")
for r in boot_rows:
    print(f"{r['Metric']}: Direct {r['Direct_mean']:.3f} [{r['Direct_CI95_low']:.3f},{r['Direct_CI95_high']:.3f}] | Hybrid {r['Hybrid_mean']:.3f} [{r['Hybrid_CI95_low']:.3f},{r['Hybrid_CI95_high']:.3f}]")

print(f"\nAUC diff: {ad_m:.4f} [{ad_l:.4f}, {ad_h:.4f}], P(Hybrid>Direct)={p_pos:.3f}")
print(f"Clustered AUC diff: {ad_m:.4f} [{cluster_ad_l:.4f}, {cluster_ad_h:.4f}], P(Hybrid>Direct)={cluster_p_pos:.3f}")
print(f"McNemar: b={b} c={c} chi2={mcn_stat:.3f} p={mcn_p:.4f}")
print(f"Subject Direct:  Acc={d_sj_m['Acc']:.4f} AUC={d_sj_m['AUC']:.4f}")
print(f"Subject Hybrid:  Acc={h_sj_m['Acc']:.4f} AUC={h_sj_m['AUC']:.4f}")
print(f"Gray zone: {ng}/{len(full_prob)} uncertain")
if cert_m: print(f"  Excluding: Acc={cert_m['Acc']:.4f} Sens={cert_m['Sens']:.4f} Spec={cert_m['Spec']:.4f}")

# ============================================================
# 9. Save all
# ============================================================
# Strategy comparison
comp = []
for name, m in strategies:
    comp.append({'Method':name,**{k:round(v,4)if isinstance(v,float)else v for k,v in m.items()}})
pd.DataFrame(comp).to_csv(os.path.join(OUTDIR,'strategy_comparison.csv'),index=False,encoding='utf-8-sig')

# Bootstrap CI
pd.DataFrame(boot_rows).to_csv(os.path.join(OUTDIR,'bootstrap_ci.csv'),index=False,encoding='utf-8-sig')

# AUC diff
pd.DataFrame([{'auc_diff':round(ad_m,4),'CI95_low':round(ad_l,4),'CI95_high':round(ad_h,4),'P_Hybrid_gt_Direct':round(p_pos,3)}]).to_csv(os.path.join(OUTDIR,'auc_difference_bootstrap.csv'),index=False,encoding='utf-8-sig')

# Participant-clustered bootstrap
pd.DataFrame(cluster_boot_rows).to_csv(os.path.join(OUTDIR,'cluster_bootstrap_ci.csv'),index=False,encoding='utf-8-sig')
pd.DataFrame([{
    'n_subjects': len(cluster_ids), 'n_resamples': len(cluster_auc_diffs),
    'auc_diff': round(ad_m,4), 'CI95_low': round(cluster_ad_l,4),
    'CI95_high': round(cluster_ad_h,4),
    'P_Hybrid_gt_Direct': round(cluster_p_pos,3),
}]).to_csv(os.path.join(OUTDIR,'cluster_auc_difference_bootstrap.csv'),index=False,encoding='utf-8-sig')

# McNemar
pd.DataFrame([{'n_common':len(common_ids),'b':b,'c':c,'chi2':round(mcn_stat,3),'p':round(mcn_p,4)}]).to_csv(os.path.join(OUTDIR,'mcnemar.csv'),index=False,encoding='utf-8-sig')

# Subject level
pd.DataFrame([
    {'Level':'Subject','Method':'Direct','Acc':round(d_sj_m['Acc'],4),'Sens':round(d_sj_m['Sens'],4),'Spec':round(d_sj_m['Spec'],4),'F1':round(d_sj_m['F1'],4),'AUC':round(d_sj_m['AUC'],4)},
    {'Level':'Subject','Method':'Hybrid','Acc':round(h_sj_m['Acc'],4),'Sens':round(h_sj_m['Sens'],4),'Spec':round(h_sj_m['Spec'],4),'F1':round(h_sj_m['F1'],4),'AUC':round(h_sj_m['AUC'],4)},
    {'Level':'Sample','Method':'Direct','Acc':round(d_m['Acc'],4),'Sens':round(d_m['Sens'],4),'Spec':round(d_m['Spec'],4),'F1':round(d_m['F1'],4),'AUC':round(d_m['AUC'],4)},
    {'Level':'Sample','Method':'Hybrid','Acc':round(full_m['Acc'],4),'Sens':round(full_m['Sens'],4),'Spec':round(full_m['Spec'],4),'F1':round(full_m['F1'],4),'AUC':round(full_m['AUC'],4)},
]).to_csv(os.path.join(OUTDIR,'subject_level_results.csv'),index=False,encoding='utf-8-sig')

# Feature importance
coefs = sorted(zip(FEATURE_COLS, full_coef), key=lambda x: abs(x[1]), reverse=True)
pd.DataFrame(coefs, columns=['feature','coefficient']).to_csv(os.path.join(OUTDIR,'feature_importance.csv'),index=False,encoding='utf-8-sig')

# Exact feature matrices used by the final model.
train_hybrid.to_csv(os.path.join(OUTDIR, 'hybrid_features_train.csv'), index=False, encoding='utf-8-sig')
test_hybrid.to_csv(os.path.join(OUTDIR, 'hybrid_features_test.csv'), index=False, encoding='utf-8-sig')
missingness_rows = []
for split_name, feature_df in [('train', train_hybrid), ('test', test_hybrid)]:
    for feature in FEATURE_COLS:
        n_missing = int(feature_df[feature].isna().sum())
        missingness_rows.append({
            'split': split_name,
            'feature': feature,
            'n_samples': len(feature_df),
            'n_missing_before_imputation': n_missing,
            'missing_fraction': n_missing / len(feature_df),
            'train_median_imputation_value': full_imputation[feature],
        })
pd.DataFrame(missingness_rows).to_csv(
    os.path.join(OUTDIR, 'feature_missingness_and_imputation.csv'),
    index=False,
    encoding='utf-8-sig',
)

# Per-sample predictions used by the clean paper figures and audit trail.
clean_predictions = paired_scores.merge(
    test_hybrid[['sample_id']].assign(hybrid_pred=full_pred),
    on='sample_id', how='left', validate='one_to_one')
clean_predictions['direct_pred'] = (clean_predictions['risk_score'] >= TH_DIRECT).astype(int)
clean_predictions['in_gray_zone'] = (
    (clean_predictions['hybrid_prob'] >= GRAY[0]) &
    (clean_predictions['hybrid_prob'] <= GRAY[1])
).astype(int)
clean_predictions.rename(columns={'risk_score':'direct_score'}, inplace=True)
clean_predictions.to_csv(os.path.join(OUTDIR,'clean_predictions.csv'),index=False,encoding='utf-8-sig')

if cert_m:
    pd.DataFrame([{
        'subset':'Full clean test set', **full_m
    }, {
        'subset':'Gray-zone excluded', **cert_m
    }]).to_csv(os.path.join(OUTDIR,'grayzone_results.csv'), index=False, encoding='utf-8-sig')

# Excluded samples log
excluded_log = []
for r in excluded_rows:
    excluded_log.append({'sample_id':r['sample_id'],'method':r['method'],'repeat_idx':r['repeat_idx'],
                         'error_type':r.get('error_type',''),'reason':excluded_reason(r)})
pd.DataFrame(excluded_log).to_csv(os.path.join(OUTDIR,'excluded_records.csv'),index=False,encoding='utf-8-sig')
train_excluded_log = [{
    'sample_id': row.get('sample_id'),
    'method': row.get('method'),
    'repeat_idx': row.get('repeat_idx'),
    'error_type': row.get('error_type', ''),
    'reason': excluded_reason(row),
} for row in train_invalid_rows]
pd.DataFrame(train_excluded_log).to_csv(
    os.path.join(OUTDIR, 'excluded_train_records.csv'), index=False, encoding='utf-8-sig')

# Repeat coverage, including the full repeat-count distribution.
coverage = []
coverage_distribution = []
for m in ['direct','rulers']:
    mdf = df_common[df_common['method']==m]
    rc = mdf.groupby('sample_id').size()
    coverage.append({'method':m,'n_samples':len(rc),'min_repeats':int(rc.min()),'max_repeats':int(rc.max()),'mean_repeats':round(rc.mean(),1)})
    for repeat_count, n_samples in rc.value_counts().sort_index().items():
        coverage_distribution.append({
            'method': m,
            'valid_repeat_count': int(repeat_count),
            'n_samples': int(n_samples),
        })
pd.DataFrame(coverage).to_csv(os.path.join(OUTDIR,'repeat_coverage.csv'),index=False,encoding='utf-8-sig')
pd.DataFrame(coverage_distribution).to_csv(
    os.path.join(OUTDIR, 'repeat_coverage_distribution.csv'), index=False, encoding='utf-8-sig')

# Stability values are recomputed from the filtered raw records, eliminating
# figure constants without provenance.
stability_rows = []
for method, values in stability.items():
    stability_rows.append({'method': method, **values})
pd.DataFrame(stability_rows).to_csv(
    os.path.join(OUTDIR, 'stability_metrics.csv'), index=False, encoding='utf-8-sig')

# Complete-repeat sensitivity analysis.
repeat_sensitivity_rows = []
for method, values in [('Direct LLM', complete_direct), ('Full Hybrid LR', complete_hybrid)]:
    repeat_sensitivity_rows.append({
        'subset': '5_valid_repeats_in_both_branches',
        'method': method,
        **values,
    })
pd.DataFrame(repeat_sensitivity_rows).to_csv(
    os.path.join(OUTDIR, 'repeat_completeness_sensitivity.csv'), index=False, encoding='utf-8-sig')

source_manifest = {
    'authoritative_script': os.path.abspath(__file__),
    'python': platform.python_version(),
    'input_files': {
        'test_checkpoint': {'path': CP, 'sha256': sha256_file(CP)},
        'train_results': {
            'path': os.path.join(TRAIN_DIR, 'results.csv'),
            'sha256': sha256_file(os.path.join(TRAIN_DIR, 'results.csv')),
        },
        'metadata': {'path': META_PATH, 'sha256': sha256_file(META_PATH)},
        'rubric': {'path': RUBRIC_PATH, 'sha256': sha256_file(RUBRIC_PATH)},
    },
    'output_directory': OUTDIR,
    'source_of_truth': 'final_summary.json and CSV files in this directory',
}
with open(os.path.join(OUTDIR, 'analysis_manifest.json'), 'w', encoding='utf-8') as stream:
    json.dump(source_manifest, stream, indent=2, ensure_ascii=False)

# Summary JSON
summary = {
    'config': CONFIG,
    'original_samples': 150,
    'training_samples_scheduled': int(meta['split'].eq('train').sum()),
    'training_samples_used': len(train_hybrid),
    'training_samples_excluded_by_qc': sorted(
        str(row.get('sample_id')) for row in train_invalid_rows
    ),
    'common_analysis_samples': len(common_samples),
    'excluded_samples': sorted(excluded_from_common),
    'total_excluded_records': len(excluded_rows),
    'participant_split': {
        'train_participants': len(train_subjects),
        'test_participants': len(test_subjects),
        'overlap': len(subject_overlap),
    },
    'results': {
        'direct': d_m, 'rulers': r_m, 'full_hybrid': full_m,
        'prog_only': prog_m, 'llm_only': llm_m,
    },
    'auc_diff': {'mean': round(ad_m,4), 'ci95': [round(ad_l,4), round(ad_h,4)], 'p_hybrid_gt_direct': round(p_pos,3)},
    'clustered_auc_diff': {
        'n_subjects': len(cluster_ids), 'n_resamples': len(cluster_auc_diffs),
        'mean': round(ad_m,4),
        'ci95': [round(cluster_ad_l,4), round(cluster_ad_h,4)],
        'p_hybrid_gt_direct': round(cluster_p_pos,3),
    },
    'mcnemar': {'b': b, 'c': c, 'chi2': round(mcn_stat,3), 'p': round(mcn_p,4)},
    'stability': stability,
    'complete_repeat_sensitivity': {
        'selection_rule': 'five valid repeats in both Direct and Prompt RULERS branches',
        'n_samples': len(complete_scores),
        'direct': complete_direct,
        'full_hybrid': complete_hybrid,
    },
    'gray_zone': {'range': list(GRAY), 'n_uncertain': ng, 'pct': round(100*ng/len(full_prob),1)},
    'hybrid_threshold': round(full_t, 2),
    'feature_schema': {
        'programmatic_features': PROG_COLS,
        'llm_features': [f'C_{cid}_criterion_median' for cid in LLM_CIDS],
        'llm_feature_aggregation': 'criterion_wise_median_across_valid_checklists',
        'information_unit_groups': {
            name: sorted(variants) for name, variants in INFORMATION_UNIT_GROUPS.items()
        },
        'information_unit_matching': 'case-normalized exact token matching; one count per semantic group',
        'speech_rate': CONFIG['speech_rate_definition'],
        'hesitation_rate': CONFIG['hesitation_rate_definition'],
        'long_pause_ratio': CONFIG['long_pause_ratio_definition'],
        'missing_value_policy': 'training-set feature median via SimpleImputer; test labels never used',
        'training_median_imputation_values': full_imputation,
        'total_features': len(FEATURE_COLS),
    },
}
with open(os.path.join(OUTDIR, 'final_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print(f"\nAll outputs saved to: {OUTDIR}/")
print("Done.")
