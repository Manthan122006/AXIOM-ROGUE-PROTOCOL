"""NINE Recalibration Control - Round 2 agent pipeline (Team5)

Runs the Round 1 model (v3: LR + HistGradientBoosting + RandomForest, threshold 0.38)
on ANY log file, then hands every true fault to three agents:
  1. Email Specialist  - matches the faulty module to the crew specialist, dispatches an alert
  2. Vector Memory     - ingests the incident into a local searchable store (Chroma, or a built-in fallback)
  3. Fix Booker        - appends schedule.csv  (log_id, crew_id, start, end, job), no double-booking

Usage:
  python pipeline.py --logs test.csv                      # hidden logs at 16:10: point --logs at their file
  python pipeline.py --logs new.csv --endpoint http://...  # POST alerts to the dummy inbox
Outputs: predictions.csv, schedule.csv, outbox/*.json, memory/, events.json (feeds the website)
"""
import argparse, json, os, re, time, warnings
from datetime import datetime, timedelta
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")
THRESHOLD = 0.38

# ---------------------------------------------------------------- features (same as Round 1)
MSG_RX = r'^NINE: (?:(?P<tone>[^.]+)\. )?(?P<event>.+?) \[(?P<mod>\S+) (?P<val>.+)\]$'
CALM = {'all is well', 'crew need not worry', 'nothing to see'}
SEV = {'OK': 0, 'WARN': 1, 'ERROR': 2, 'CRIT': 3}
NUM = ['sev', 'calm', 'calm_x_sev', 'mod_mismatch', 'reading_missing', 'val_diff', 'val_diff_flag', 'rz', 'rz_abs', 'log_delay', 'retry_count']
CAT = ['module', 'mode', 'node', 'status', 'tone', 'event', 'mod']

def build(df, stats):
    d = df.copy().reset_index(drop=True)
    d = d.join(d['message'].astype(str).str.extract(MSG_RX))
    d['tone'] = d['tone'].fillna('none'); d['event'] = d['event'].fillna('unknown'); d['mod'] = d['mod'].fillna(d['module'])
    d['val'] = d['val'].fillna('REDACTED')
    d['calm'] = d['tone'].isin(CALM).astype(int)
    d['sev'] = d['status'].map(SEV).fillna(0).astype(int)
    d['calm_x_sev'] = d['calm'] * d['sev']
    d['mod_mismatch'] = (d['mod'] != d['module']).astype(int)
    d['reading_missing'] = d['reading'].isna().astype(int)
    d['mval'] = pd.to_numeric(d['val'].str.split().str[0], errors='coerce')
    d['val_diff'] = (d['mval'] - d['reading']).abs().fillna(0)
    d['val_diff_flag'] = (d['val_diff'] > 0.01).astype(int)
    rz = (d['reading'] - d['module'].map(stats['median'])) / d['module'].map(stats['std'])
    d['rz_abs'] = rz.abs().fillna(-1); d['rz'] = rz.fillna(0)
    d['log_delay'] = np.log1p(d['delay_ms'])
    return d

def ohe():
    try: return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown='ignore', sparse=False)

def models():
    lr = make_pipeline(make_column_transformer((make_pipeline(SimpleImputer(), StandardScaler()), NUM), (ohe(), CAT)), LogisticRegression(max_iter=3000))
    ct = make_column_transformer(('passthrough', NUM), (OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT))
    hgb = make_pipeline(ct, HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=3,
                        categorical_features=list(range(len(NUM), len(NUM) + len(CAT))), random_state=0))
    rf = make_pipeline(make_column_transformer((SimpleImputer(strategy='median'), NUM), (ohe(), CAT)),
                       RandomForestClassifier(300, min_samples_leaf=5, max_features=0.3, n_jobs=-1, random_state=0))
    return [lr, hgb, rf]

def reasons(r):
    out = []
    if r.mod_mismatch: out.append(f"Wrong module: NINE says '{r['mod']}' but the log is from {r.module}")
    if r.val_diff_flag: out.append(f"Quoted value {r.mval} != real reading {r.reading}")
    if r.reading_missing: out.append("Reading is REDACTED")
    if r.rz_abs > 1.25: out.append(f"Reading {r.reading} is {r.rz_abs:.1f} std from the {r.module} normal")
    if r.calm and r.sev >= 2: out.append(f"Calm tone ('{r.tone}') on a {r.status} log")
    if not out: out.append("Pattern match on status, event and telemetry")
    return out

LIE_TYPES = [('mod_mismatch', 'Wrong module named'), ('val_diff_flag', 'Fake quoted value'), ('reading_missing', 'Redacted reading'),
             ('oob', 'Out-of-range reading'), ('calm_crit', 'Calm text on ERROR/CRIT')]

# ---------------------------------------------------------------- agent 2: vector memory
class VectorMemory:
    """Chroma if installed (pip install chromadb), else a persisted TF-IDF cosine store."""
    def __init__(self, path='memory'):
        os.makedirs(path, exist_ok=True); self.path = path; self.backend = 'tfidf'
        try:
            import chromadb
            self.col = chromadb.PersistentClient(path=path).get_or_create_collection('incidents'); self.backend = 'chroma'
        except Exception:
            self.docs, self.meta = [], []
    def add(self, ids, docs, metas):
        if self.backend == 'chroma':
            self.col.upsert(ids=ids, documents=docs, metadatas=metas)
        else:
            self.docs += docs; self.meta += metas
            self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(self.docs); self.X = self.vec.transform(self.docs)
            json.dump({'docs': self.docs, 'meta': self.meta}, open(os.path.join(self.path, 'store.json'), 'w'))
    def search(self, text, k=3, exclude=None):
        if self.backend == 'chroma':
            res = self.col.query(query_texts=[text], n_results=k + 1)
            hits = [(m['log_id'], d, 1 - dist) for m, d, dist in zip(res['metadatas'][0], res['documents'][0], res['distances'][0])]
        else:
            if not self.docs: return []
            s = (self.X @ self.vec.transform([text]).T).toarray().ravel()
            hits = [(self.meta[i]['log_id'], self.docs[i], float(s[i])) for i in np.argsort(-s)[:k + 1]]
        return [h for h in hits if h[0] != exclude][:k]

# ---------------------------------------------------------------- crew roster
DEFAULT_CREW = pd.DataFrame([
    ('C01', 'Ada Okafor', 'Communications Engineer', 'Antenna', 'ada.okafor@aurora.ship'),
    ('C02', 'Ravi Menon', 'Communications Engineer', 'Antenna', 'ravi.menon@aurora.ship'),
    ('C03', 'Lena Varga', 'Network Specialist', 'Notice-Bus', 'lena.varga@aurora.ship'),
    ('C04', 'Tomas Reyes', 'Network Specialist', 'Notice-Bus', 'tomas.reyes@aurora.ship'),
    ('C05', 'Mei Tanaka', 'Power Systems Engineer', 'Lab-Power', 'mei.tanaka@aurora.ship'),
    ('C06', 'Omar Haddad', 'Power Systems Engineer', 'Lab-Power', 'omar.haddad@aurora.ship'),
    ('C07', 'Sofia Lindqvist', 'Life-Support Specialist', 'Life-Support', 'sofia.lindqvist@aurora.ship'),
    ('C08', 'Kwame Asante', 'Life-Support Specialist', 'Life-Support', 'kwame.asante@aurora.ship'),
], columns=['crew_id', 'name', 'role', 'module', 'email'])

def load_crew(path):
    """Accepts the official roster if it has crew_id plus a module/specialty column; else the placeholder."""
    if path and os.path.exists(path):
        c = pd.read_csv(path); c.columns = [x.strip().lower() for x in c.columns]
        mcol = next((x for x in ['module', 'specialty', 'speciality', 'system', 'role'] if x in c.columns), None)
        if 'crew_id' in c.columns and mcol:
            c['module'] = c[mcol].astype(str)
            for col, default in [('name', c['crew_id']), ('role', c[mcol]), ('email', c['crew_id'].astype(str) + '@aurora.ship')]:
                if col not in c.columns: c[col] = default
            return c
        print('crew file not recognised, using placeholder roster')
    return DEFAULT_CREW.copy()

def specialist_for(module, crew, busy_until):
    pool = crew[crew['module'].str.lower().str.contains(module.lower().split('-')[0])]
    if pool.empty: pool = crew
    return pool.loc[pool['crew_id'].map(lambda c: busy_until.get(c, datetime.min)).idxmin()]

JOB = {3: ('Emergency repair', 45), 2: ('Repair', 30), 1: ('Inspection', 20), 0: ('Inspection', 15)}

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='test.csv'); ap.add_argument('--train', default='train.csv'); ap.add_argument('--val', default='val.csv')
    ap.add_argument('--crew', default='crew.csv'); ap.add_argument('--endpoint', default=os.environ.get('INBOX_ENDPOINT'))
    ap.add_argument('--schedule', default='schedule.csv'); ap.add_argument('--start', default='2187-04-15 04:00')
    a = ap.parse_args(); t0 = time.time(); events = []

    # Agent 1 - Triage: train once on train+val, score the incoming logs
    lab = pd.concat([pd.read_csv(a.train), pd.read_csv(a.val)], ignore_index=True)
    stats = lab.groupby('module')['reading'].agg(['median', 'std'])
    X, y = build(lab, stats), lab['is_true_fault'].values
    fitted = [m.fit(X, y) for m in models()]
    logs = pd.read_csv(a.logs); L = build(logs, stats)
    L['prob'] = np.mean([m.predict_proba(L)[:, 1] for m in fitted], 0)
    L['verdict'] = (L['prob'] >= THRESHOLD).astype(int)
    L['oob'] = (L['rz_abs'] > 1.25).astype(int); L['calm_crit'] = ((L['calm'] == 1) & (L['sev'] >= 2)).astype(int)
    pd.DataFrame({'log_id': logs['log_id'], 'is_true_fault': L['verdict']}).to_csv('predictions.csv', index=False)
    print(f'[triage] {len(L)} logs scored, {L.verdict.sum()} true faults ({time.time()-t0:.1f}s)')

    crew = load_crew(a.crew); mem = VectorMemory(); os.makedirs('outbox', exist_ok=True)
    sched_exists = os.path.exists(a.schedule)
    busy_until = {}
    if sched_exists:
        old = pd.read_csv(a.schedule)
        for c, e in zip(old['crew_id'], old['end']): busy_until[c] = max(busy_until.get(c, datetime.min), pd.to_datetime(e).to_pydatetime())
    start0 = datetime.strptime(a.start, '%Y-%m-%d %H:%M'); rows = []

    faults = L[L.verdict == 1].sort_values('prob', ascending=False)   # highest-risk faults first
    for _, r in faults.iterrows():
        why = reasons(r)
        doc = f"{r.module} {r.status} {r.event} | {r.message} | " + '; '.join(why)
        # Agent 3 - Vector memory: recall similar past incidents, then store this one
        similar = mem.search(doc, k=3, exclude=r.log_id)
        mem.add([r.log_id], [doc], [{'log_id': r.log_id, 'module': r.module, 'status': r.status, 'prob': float(r.prob)}])
        # Agent 2 - Email specialist
        sp = specialist_for(r.module, crew, busy_until)
        job, mins = JOB[int(r.sev)]
        st = max(start0, busy_until.get(sp.crew_id, datetime.min)); en = st + timedelta(minutes=mins)
        busy_until[sp.crew_id] = en
        email = {'to': sp.email, 'crew_id': sp.crew_id, 'subject': f"[{r.status}] True fault on {r.module} - {r.log_id} (risk {r.prob:.0%})",
                 'body': f"{sp['name']}, NINE's log {r.log_id} is a TRUE fault (model risk {r.prob:.0%}).\nModule: {r.module} | node {r.node} | mode {r['mode']}\n"
                         f"NINE wrote: {r.message}\nWhy we don't trust it:\n- " + '\n- '.join(why) +
                         (f"\nSimilar past incidents: {', '.join(s[0] for s in similar)}" if similar else '') +
                         f"\nBooked: {job}, {st:%H:%M}-{en:%H:%M}.", 'log_id': r.log_id}
        sent = 'outbox'
        if a.endpoint:
            try:
                import urllib.request
                req = urllib.request.Request(a.endpoint, data=json.dumps(email).encode(), headers={'Content-Type': 'application/json'})
                urllib.request.urlopen(req, timeout=5); sent = 'endpoint'
            except Exception as e:
                sent = f'outbox (endpoint failed: {e.__class__.__name__})'
        json.dump(email, open(f'outbox/{r.log_id}.json', 'w'), indent=1)
        # Agent 4 - Fix booker
        rows.append({'log_id': r.log_id, 'crew_id': sp.crew_id, 'start': f'{st:%Y-%m-%d %H:%M}', 'end': f'{en:%Y-%m-%d %H:%M}', 'job': f'{job}: {r.event} on {r.module}'})
        events.append({'log_id': r.log_id, 'module': r.module, 'status': r.status, 'node': r.node, 'message': r.message, 'prob': round(float(r.prob), 3),
                       'reasons': why, 'lies': [lbl for k, lbl in LIE_TYPES if r[k]], 'similar': [{'log_id': s[0], 'score': round(s[2], 3)} for s in similar],
                       'email': email, 'sent_via': sent, 'booking': rows[-1], 'specialist': {'crew_id': sp.crew_id, 'name': sp['name'], 'role': sp['role']}})

    pd.DataFrame(rows, columns=['log_id', 'crew_id', 'start', 'end', 'job']).to_csv(a.schedule, mode='a', header=not sched_exists, index=False)
    noise = L[L.verdict == 0].sample(min(40, (L.verdict == 0).sum()), random_state=0)
    summary = {'generated': datetime.now().isoformat(timespec='seconds'), 'logs': int(len(L)), 'faults': int(L.verdict.sum()), 'noise': int((L.verdict == 0).sum()),
               'threshold': THRESHOLD, 'memory_backend': mem.backend, 'emails': len(rows), 'bookings': len(rows), 'runtime_s': round(time.time() - t0, 1),
               'lie_counts': {lbl: int(L.loc[L.verdict == 1, k].sum()) for k, lbl in LIE_TYPES},
               'by_module': L[L.verdict == 1]['module'].value_counts().to_dict(),
               'crew_load': pd.DataFrame(rows)['crew_id'].value_counts().to_dict() if rows else {},
               'model': {'name': 'LR + HistGradientBoosting + RandomForest', 'val_f1': 0.742, 'cv_f1': 0.743, 'precision': 0.779, 'recall': 0.708}}
    out = {'summary': summary, 'crew': crew[['crew_id', 'name', 'role', 'module']].to_dict('records'), 'events': events,
           'noise': [{'log_id': r.log_id, 'module': r.module, 'status': r.status, 'message': r.message, 'prob': round(float(r.prob), 3)} for _, r in noise.iterrows()]}
    json.dump(out, open('events.json', 'w'), indent=1)
    open('events.js', 'w').write('window.NINE_DATA = ' + json.dumps(out) + ';')
    print(f"[email] {len(rows)} alerts | [memory:{mem.backend}] {len(rows)} incidents | [booker] {len(rows)} rows -> {a.schedule} | {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
