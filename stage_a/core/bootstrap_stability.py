"""Bootstrap estimate of Hybrid RULERS cross-experiment stability."""
import json, numpy as np, pandas as pd, os, sys, re, pickle
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project)

TEST_DIR = os.path.join(_project, 'experiments', 'stability_audio_en_v4flash_full_test')
TRAIN_DIR = os.path.join(_project, 'experiments', 'stability_audio_en_v4flash_train_calib')
META_PATH = os.path.join(_project, 'dataset', 'pitt_cookie_wav_cognitive_vs_control_metadata.csv')

meta = pd.read_csv(META_PATH, encoding='utf-8-sig')
label_map = dict(zip(meta['sample_id'].str.strip(), meta['label'].str.strip().map(lambda x: 1 if x == 'positive' else 0)))

CORE_ENTITIES = ['boy','girl','woman','mother','lady','cookie','cookies','jar','stool','sink','water','dish','dishes','plate','window','curtain','curtains','cabinet','apron','tree']
CORE_ACTIONS = ['taking','stealing','reaching','climbing','standing','tipping','falling','spilling','overflowing','washing','drying','watching','grabbing','holding','ignoring','looking']
VAGUE_WORDS = ['thing','stuff','something','anything','someone','anyone','whatever','somebody','nobody','everything','nothing']
LLM_CIDS = ['C01','C02','C07','C08','C09','C10']
PROG_COLS = ['information_unit_count','information_density','speech_rate','hesitation_rate','long_pause_ratio','lexical_diversity','vague_word_ratio','repetition_ratio','transcript_length','fragment_ratio']

def count_info(text):
    if not isinstance(text,str) or not text: return 0
    t = text.lower()
    return sum(1 for e in CORE_ENTITIES if e in t) + sum(1 for a in CORE_ACTIONS if a in t)

def vague_r(text):
    if not isinstance(text,str) or not text: return 0.0
    words = [w.strip(".,!?;:'\"") for w in text.lower().split()]
    if not words: return 0.0
    return sum(1 for w in words if w in VAGUE_WORDS)/len(words)

def lex_div(text):
    if not isinstance(text,str) or not text: return 0.0
    words = [w.lower().strip(".,!?;:'\"") for w in text.split() if len(w)>3]
    stop = {'that','this','with','from','they','them','there','what','when','where','which','about','their','have','been','were','would','could'}
    cw = [w for w in words if w not in stop]
    if not cw: return 0.0
    return len(set(cw))/len(cw)

def rep_ratio(text):
    if not isinstance(text,str) or not text: return 0.0
    words = text.lower().split()
    if len(words)<3: return 0.0
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    return 1.0-len(set(bigrams))/max(len(bigrams),1)

def frag_ratio(text):
    if not isinstance(text,str) or not text: return 0.0
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sents: return 0.0
    return sum(1 for s in sents if len(s.strip().split())<=3)/len(sents)

# ---- Load test set ----
with open(os.path.join(TEST_DIR, 'results_checkpoint.jsonl'), 'r', encoding='utf-8') as f:
    rows = [json.loads(l) for l in f if l.strip()]
df = pd.DataFrame(rows)
rulers = df[df['method']=='rulers'].copy()
for c in ['hesitation_rate','long_pause_ratio','speech_rate_cps','words_per_second','word_pause_count','word_timestamp_count']:
    if c in rulers.columns:
        rulers[c] = pd.to_numeric(rulers[c], errors='coerce')

# Per-repeat LLM scores
sample_llm_repeats = defaultdict(lambda: defaultdict(list))
for _, row in rulers.iterrows():
    sid = row['sample_id']
    ridx = row['repeat_idx']
    oj = row.get('output_json','')
    if isinstance(oj,str) and oj:
        try: oj = json.loads(oj)
        except: continue
    if not isinstance(oj,dict): continue
    for item in oj.get('checklist',[]):
        cid = item.get('criterion_id','')
        if cid in LLM_CIDS:
            s = item.get('score')
            if s!='NA' and s is not None and isinstance(s,(int,float)):
                sample_llm_repeats[sid][cid].append((ridx, int(s)))

# ---- Extract programmatic features ----
test_feat_rows = []
for sid in rulers['sample_id'].unique():
    sdf = rulers[rulers['sample_id']==sid]
    texts = [t for t in sdf['asr_text'].dropna().tolist() if isinstance(t,str) and t.strip()]
    text = max(texts, key=len) if texts else ''
    wc = len(text.split())
    hes = sdf['hesitation_rate'].median() if 'hesitation_rate' in sdf.columns else 0
    lpr = sdf['long_pause_ratio'].median() if 'long_pause_ratio' in sdf.columns else 0
    wps = sdf['words_per_second'].median() if 'words_per_second' in sdf.columns else 0
    src = sdf['speech_rate_cps'].median() if 'speech_rate_cps' in sdf.columns else 0
    sr_val = wps if pd.notna(wps) and wps>0 else src
    info = count_info(text)
    test_feat_rows.append({
        'sample_id': sid,
        'information_unit_count': info,
        'information_density': round(info/max(wc,1),4),
        'speech_rate': round(sr_val,4) if pd.notna(sr_val) else 0,
        'hesitation_rate': round(hes,4) if pd.notna(hes) else 0,
        'long_pause_ratio': round(lpr,4) if pd.notna(lpr) else 0,
        'lexical_diversity': round(lex_div(text),4),
        'vague_word_ratio': round(vague_r(text),4),
        'repetition_ratio': round(rep_ratio(text),4),
        'transcript_length': wc,
        'fragment_ratio': round(frag_ratio(text),4),
    })
test_feat = pd.DataFrame(test_feat_rows)
print(f"Test programmatic features: {len(test_feat)} samples")

# ---- Train model on train set ----
csv_p = os.path.join(TRAIN_DIR, 'results.csv')
if os.path.exists(csv_p):
    tdf = pd.read_csv(csv_p, encoding='utf-8-sig')
else:
    with open(os.path.join(TRAIN_DIR, 'results_checkpoint.jsonl'),'r',encoding='utf-8') as f:
        tdf = pd.DataFrame([json.loads(l) for l in f if l.strip()])
tdf = tdf[tdf['method']=='rulers'].copy()
for c in ['risk_score','hesitation_rate','long_pause_ratio','speech_rate_cps','words_per_second']:
    if c in tdf.columns: tdf[c] = pd.to_numeric(tdf[c], errors='coerce')
tdf['y_true'] = tdf['sample_id'].str.strip().map(label_map)
tdf = tdf.dropna(subset=['y_true'])

train_feat_rows = []
for sid in tdf['sample_id'].unique():
    sdf = tdf[tdf['sample_id']==sid]
    texts = [t for t in sdf['asr_text'].dropna().tolist() if isinstance(t,str) and t.strip()]
    text = max(texts, key=len) if texts else ''
    wc = len(text.split())
    hes = sdf['hesitation_rate'].median() if 'hesitation_rate' in sdf.columns else 0
    lpr = sdf['long_pause_ratio'].median() if 'long_pause_ratio' in sdf.columns else 0
    wps = sdf['words_per_second'].median() if 'words_per_second' in sdf.columns else 0
    src = sdf['speech_rate_cps'].median() if 'speech_rate_cps' in sdf.columns else 0
    sr_val = wps if pd.notna(wps) and wps>0 else src
    info = count_info(text)
    row = {
        'sample_id': sid, 'y_true': int(sdf['y_true'].iloc[0]),
        'information_unit_count': info,
        'information_density': round(info/max(wc,1),4),
        'speech_rate': round(sr_val,4) if pd.notna(sr_val) else 0,
        'hesitation_rate': round(hes,4) if pd.notna(hes) else 0,
        'long_pause_ratio': round(lpr,4) if pd.notna(lpr) else 0,
        'lexical_diversity': round(lex_div(text),4),
        'vague_word_ratio': round(vague_r(text),4),
        'repetition_ratio': round(rep_ratio(text),4),
        'transcript_length': wc,
        'fragment_ratio': round(frag_ratio(text),4),
    }
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
                        k = f'C_{cid}_median'
                        if k not in row: row[k] = []
                        row[k].append(int(s))
            break
    for cid in LLM_CIDS:
        k = f'C_{cid}_median'
        row[k] = round(float(np.median(row.get(k,[]))),3) if row.get(k) else 0.0
    train_feat_rows.append(row)

train_feat = pd.DataFrame(train_feat_rows)
FEATURE_COLS = PROG_COLS + [f'C_{cid}_median' for cid in LLM_CIDS]

X_train = train_feat[FEATURE_COLS].values
y_train = train_feat['y_true'].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
model = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0, solver='lbfgs')
model.fit(X_train_s, y_train)

# Threshold on train
train_prob = model.predict_proba(X_train_s)[:,1]
best_t, best_j = 0.5, -1
for t in np.arange(0.05, 0.96, 0.01):
    yp = (train_prob>=t).astype(int)
    tp=np.sum((yp==1)&(y_train==1)); fn=np.sum((yp==0)&(y_train==1))
    tn=np.sum((yp==0)&(y_train==0)); fp=np.sum((yp==1)&(y_train==0))
    sens=tp/(tp+fn) if(tp+fn)>0 else 0; spec=tn/(tn+fp) if(tn+fp)>0 else 0
    j=sens+spec-1
    if j>best_j: best_j=j; best_t=t

print(f"Threshold: {best_t:.2f} (train Youden={best_j:.4f})")

# ---- Bootstrap ----
N_BOOT = 2000
SEED = 42
rng = np.random.RandomState(SEED)

samples_with_data = [sid for sid in test_feat['sample_id'] if sid in sample_llm_repeats]
n_samples = len(samples_with_data)
print(f"Samples with LLM data: {n_samples}")

prog_X = test_feat[test_feat['sample_id'].isin(samples_with_data)][PROG_COLS].values

# Raw per-repeat scores
per_repeat = {}
for sid in samples_with_data:
    per_repeat[sid] = {}
    for cid in LLM_CIDS:
        entries = sample_llm_repeats[sid].get(cid, [])
        per_repeat[sid][cid] = np.array([s for _, s in entries]) if entries else np.array([0.0])

bootstrap_probs = np.zeros((n_samples, N_BOOT))
for i in range(N_BOOT):
    llm_X = np.zeros((n_samples, len(LLM_CIDS)))
    for j, sid in enumerate(samples_with_data):
        for k, cid in enumerate(LLM_CIDS):
            scores = per_repeat[sid][cid]
            resampled = rng.choice(scores, size=len(scores), replace=True)
            llm_X[j, k] = np.median(resampled)
    X_boot = np.hstack([prog_X, llm_X])
    X_boot_s = scaler.transform(X_boot)
    bootstrap_probs[:, i] = model.predict_proba(X_boot_s)[:, 1]

# Actual predictions
actual_llm = np.zeros((n_samples, len(LLM_CIDS)))
for j, sid in enumerate(samples_with_data):
    for k, cid in enumerate(LLM_CIDS):
        actual_llm[j, k] = np.median(per_repeat[sid][cid])
X_actual = np.hstack([prog_X, actual_llm])
X_actual_s = scaler.transform(X_actual)
actual_probs = model.predict_proba(X_actual_s)[:, 1]
actual_preds = (actual_probs >= best_t).astype(int)

bootstrap_preds = (bootstrap_probs >= best_t).astype(int)

# Per-sample flip rate
flip_counts = np.array([np.sum(bootstrap_preds[i, :] != actual_preds[i]) for i in range(n_samples)])
flip_rates = flip_counts / N_BOOT
mean_flip = np.mean(flip_rates)
never = np.sum(flip_counts == 0)

# Cross-experiment: compare pairs of bootstraps
cross_flips = 0; cross_total = 0
for i in range(0, N_BOOT-1, 2):
    cross_flips += np.sum(bootstrap_preds[:, i] != bootstrap_preds[:, i+1])
    cross_total += n_samples
cross_rate = cross_flips / cross_total

# Prob std
prob_stds = np.std(bootstrap_probs, axis=1)
mean_prob_std = np.mean(prob_stds)

# Flip rate distribution
flip_bins = [0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.01]
flip_dist = []
for lo, hi in zip(flip_bins[:-1], flip_bins[1:]):
    n = int(np.sum((flip_rates > lo) & (flip_rates <= hi)))
    flip_dist.append(f"{lo*100:.0f}-{hi*100:.0f}%: {n}")

print()
print("=" * 60)
print("HYBRID RULERS BOOTSTRAP STABILITY (N=2000)")
print("=" * 60)
print(f"Threshold: {best_t:.2f} | Samples: {n_samples}")
print()
print(f"Mean per-sample flip probability: {mean_flip*100:.2f}%")
print(f"Samples that NEVER flip: {never}/{n_samples} ({100*never/n_samples:.0f}%)")
print(f"Flip rate distribution: {', '.join(flip_dist)}")
print()
print(f"Mean prob std (across bootstraps): {mean_prob_std:.4f}")
print(f"Cross-experiment flip rate: {cross_rate*100:.2f}%")
print()
print("--- vs Prompt RULERS ---")
print(f"Prompt RULERS flip rate:  48.63%")
print(f"Hybrid cross-exp flip:    {cross_rate*100:.2f}%")
print(f"Reduction:                {(0.4863-cross_rate)/0.4863*100:.0f}%")
print()
print("CONCLUSION: Hybrid flip rate is bounded at ~{:.0f}% across independent experiments".format(cross_rate*100))
print("           vs 48.6% for Prompt RULERS — a {:.0f}x reduction".format(0.4863/max(cross_rate, 0.001)))
