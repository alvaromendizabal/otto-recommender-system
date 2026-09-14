"""Bounded sparse representation learning on certified PRE-cutoff history.

R11: shared session basis, IDF versus no IDF. R12: action-specific PPMI
forward-context basis versus direction removed. This is feature learning, not
ranker/algorithm tuning. Vocabulary selection never consults query targets.
"""
from __future__ import annotations
import gc
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from core import checked, save_arrays, save_json, sha, forward_folds, array_digest
from latent_features import normalize, positions
from source_contract import HISTORY_END

DIMENSION = 32
VOCABULARY_CAP = 60000
EDGE_RECORD_CAP = 8000000
MAX_GAP = 5
MAX_MS = 1800000
SEED = 20260912


def read(path):
    return json.loads(Path(path).read_text())


def training_anchor_sessions(ids, query_ts, anchors, postings):
    """Freeze source selection using earliest-fold TRAINING prefixes only.

    Later validation anchors cannot expand the learned vocabulary or basis.
    Postings themselves reference only strictly pre-cutoff historical sessions.
    """
    ids=np.asarray(ids);query_ts=np.asarray(query_ts);anchors=np.asarray(anchors);postings=np.asarray(postings)
    if anchors.shape!=(len(ids),4) or postings.ndim!=2 or postings.shape[1]!=4:
        raise ValueError('Training-anchor source schema differs')
    train=forward_folds(ids,query_ts)[0]['train']
    selected_anchors=np.unique(anchors[train][anchors[train]>=0])
    selected_sessions=np.unique(postings[np.isin(postings[:,0],selected_anchors),1])
    if not len(selected_anchors) or not len(selected_sessions):raise ValueError('No training-only historical source')
    return selected_sessions, {'selection':'earliest chronological fold training anchors only',
        'training_ids_sha256':array_digest(ids[train]),'training_anchor_count':len(selected_anchors),
        'training_selected_history_sessions':len(selected_sessions),'validation_anchors_used':False}


def source_selection(ctx):
    root=ctx['shared'];cohort_receipt=read(root/'cohort.json');postings_receipt=read(root/'postings.json')
    checked(root/'cohort.npz',cohort_receipt['sha256']);checked(root/'postings.npz',postings_receipt['sha256'])
    with np.load(root/'cohort.npz',allow_pickle=False) as z:ids=z['ids'];anchors=z['anchors']
    if not np.array_equal(ids,ctx['ids']):raise ValueError('Source query identity changed')
    with np.load(root/'postings.npz',allow_pickle=False) as z:postings=z['postings']
    return training_anchor_sessions(ids,np.asarray(ctx['cohort']['query_ts'],np.int64),anchors,postings)


def source_events(ctx):
    root = ctx['shared']
    receipt = read(root/'history.json')
    checked(root/'history.npz', receipt['sha256'])
    selected, selection = source_selection(ctx)
    with np.load(root/'history.npz', allow_pickle=False) as z:
        events = z['events'].copy()
    events=validate_events(events, ctx['excluded'])
    events=events[np.isin(events[:,0],selected)]
    if not np.array_equal(np.unique(events[:,0]),selected):raise ValueError('Selected history incomplete')
    return events


def validate_events(events, excluded=()):
    x = np.asarray(events)
    if x.ndim != 2 or x.shape[1] != 5 or x.dtype.kind not in 'iu' or not len(x):
        raise ValueError('Expected nonempty (session,item,ts,action,original_index) integer history')
    if len(x) > 10000000 or len(np.unique(x[:, 0])) > 400000:
        raise ValueError('Historical source cap exceeded')
    if (x < 0).any() or (x[:, 2] >= HISTORY_END).any() or not np.isin(x[:, 3], (0,1,2)).all():
        raise ValueError('Invalid events or history cutoff crossed')
    if np.isin(x[:, 0], np.asarray(excluded, dtype=np.int64)).any():
        raise ValueError('Study session contamination')
    order = np.lexsort((x[:, 4], x[:, 0])); x = x[order]
    same = x[1:, 0] == x[:-1, 0]
    if ((np.diff(x[:, 4]) <= 0) & same).any() or ((np.diff(x[:, 2]) < 0) & same).any():
        raise ValueError('Duplicate original indices or reversed chronology')
    return x


def select_vocabulary(events, cap=VOCABULARY_CAP):
    if not isinstance(cap, int) or cap < 2 or cap > VOCABULARY_CAP:
        raise ValueError('Invalid vocabulary cap')
    # DISTINCT(session,item) frequency; repeated clicks do not inflate selection.
    pairs = np.unique(events[:, [0, 1]], axis=0)
    item, count = np.unique(pairs[:, 1], return_counts=True)
    order = np.lexsort((item, -count))[:cap]
    sort = np.argsort(item[order]); chosen = order[sort]
    return item[chosen].astype(np.int64), count[chosen].astype(np.int64)


def incidence(events, vocabulary):
    sessions = np.unique(events[:, 0])
    si = np.searchsorted(sessions, events[:, 0]); ii, keep = positions(vocabulary, events[:, 1])
    shape = (len(vocabulary), len(sessions))
    def binary(mask):
        m = sparse.coo_matrix((np.ones(int(mask.sum()), np.float32), (ii[mask], si[mask])), shape=shape).tocsr()
        m.sum_duplicates(); m.data[:] = 1.; m.sort_indices()
        return m
    return [binary(keep), *[binary(keep & (events[:, 3] == k)) for k in range(3)]]


def matrix_arrays(m):
    m = sparse.csr_matrix(m, dtype=np.float32); m.sum_duplicates(); m.sort_indices()
    return dict(data=m.data, indices=m.indices, indptr=m.indptr, shape=np.asarray(m.shape, np.int64))


def restore_matrix(path):
    with np.load(path, allow_pickle=False) as a:
        m = sparse.csr_matrix((a['data'], a['indices'], a['indptr']), shape=tuple(a['shape']))
    if not np.isfinite(m.data).all() or (m.data < 0).any():
        raise ValueError('Invalid cached sparse matrix')
    return m


def pair_rows(events, vocabulary, offset):
    """Each retained-row offset is only an enumeration device; original gaps govern.

    Because event indices strictly increase, any pair with original gap<=5 is
    at most five retained rows apart. Missing rows are NOT compressed.
    Return (session, source_vocab_index, target_vocab_index, target_action).
    """
    if offset not in range(1, MAX_GAP+1):
        raise ValueError('Invalid enumerator offset')
    a, b = events[:-offset], events[offset:]
    ia, ka = positions(vocabulary, a[:, 1]); ib, kb = positions(vocabulary, b[:, 1])
    gap = b[:, 4] - a[:, 4]; elapsed = b[:, 2] - a[:, 2]
    use = ((a[:, 0] == b[:, 0]) & (gap > 0) & (gap <= MAX_GAP) & (elapsed >= 0)
           & (elapsed <= MAX_MS) & (a[:, 1] != b[:, 1]) & ka & kb)
    forward = np.column_stack((a[use, 0], ia[use], ib[use], b[use, 3])).astype(np.int64)
    reverse = np.column_stack((a[use, 0], ib[use], ia[use], a[use, 3])).astype(np.int64)
    return forward, reverse


def count_pairs(rows, n_items, action):
    r = rows[rows[:, 3] == action, :3]
    # Removing session only AFTER distinctness makes support count sessions.
    r = np.unique(r, axis=0)
    m = sparse.coo_matrix((np.ones(len(r), np.float32), (r[:, 1], r[:, 2])),
                          shape=(n_items, n_items)).tocsr()
    m.sum_duplicates(); m.sort_indices()
    return m


def ppmi(m):
    m = sparse.csr_matrix(m, dtype=np.float64)
    if m.nnz == 0:
        return m.astype(np.float32)
    total = float(m.sum()); row = np.asarray(m.sum(1)).ravel(); col = np.asarray(m.sum(0)).ravel()
    coo = m.tocoo()
    values = np.maximum(0., np.log(coo.data * total / (row[coo.row] * col[coo.col])))
    out = sparse.coo_matrix((values.astype(np.float32), (coo.row, coo.col)), shape=m.shape).tocsr()
    out.eliminate_zeros(); out.sort_indices()
    return out


def svd(m, seed=SEED):
    if m.nnz < 2 or min(m.shape) < 2:
        raise ValueError('Insufficient historical representation support; no model fitting')
    rank = min(DIMENSION, min(m.shape)-1)
    u, s, vt = randomized_svd(m, n_components=rank, n_oversamples=8, n_iter=2,
                              power_iteration_normalizer='QR', flip_sign=True, random_state=seed)
    if not (np.isfinite(u).all() and np.isfinite(s).all() and np.isfinite(vt).all()):
        raise ValueError('Nonfinite factorization')
    return u, s, vt


def padded(x):
    result = np.zeros((len(x), DIMENSION), np.float32)
    result[:, :x.shape[1]] = normalize(x)
    return result


def session_embedding(matrices, use_idf):
    all_events = matrices[0]
    df = np.asarray(all_events.sum(1)).ravel()
    item_weight = 1 + np.log((1+all_events.shape[1])/(1+df)) if use_idf else np.ones(len(df))
    length = np.asarray(all_events.sum(0)).ravel()
    dw = sparse.diags(item_weight.astype(np.float32))
    sw = sparse.diags((1/np.sqrt(np.maximum(length,1))).astype(np.float32))
    weighted = (dw @ all_events @ sw).tocsr()
    _, singular, vt = svd(weighted)
    query = padded(weighted @ vt.T)
    targets = np.stack([padded(dw @ m @ sw @ vt.T) for m in matrices[1:]])
    return np.broadcast_to(query, targets.shape).copy(), targets, singular


def transition_embedding(counts):
    association = ppmi(counts)
    u, singular, vt = svd(association)
    scale = np.sqrt(singular)
    return padded(u*scale), padded(vt.T*scale), singular


def representation_identity(ctx, root):
    _, selection = source_selection(ctx)
    return {'historical_cache_sha256': read(ctx['shared']/'history.json')['sha256'],
            'historical_cutoff_ms': HISTORY_END, 'query_labels_read': False,
            'source_selection': selection, 'postings_sha256':read(ctx['shared']/'postings.json')['sha256'],
            'protocol_sha256': sha(root/'protocol.json'), 'implementation_sha256': sha(root/'latent_models.py'),
            'numpy': importlib.metadata.version('numpy'), 'scipy': importlib.metadata.version('scipy'),
            'scikit_learn': importlib.metadata.version('scikit-learn'),
            'vocabulary_selection': 'historical distinct-session frequency only; stable item-ID ties',
            'rank_cap': DIMENSION, 'vocabulary_cap': VOCABULARY_CAP}


def store_matrix(path, matrix, unit):
    receipt = path.with_suffix('.json')
    if receipt.exists():
        r=read(receipt)
        if r['unit'] != unit: raise ValueError('Sparse cache identity differs')
        checked(path, r['sha256']); return r
    if path.exists(): raise ValueError('Orphan sparse cache preserved')
    digest=save_arrays(path, **matrix_arrays(matrix))
    r={'unit':unit, 'sha256':digest,'rows':matrix.shape[0],'columns':matrix.shape[1],'nnz':matrix.nnz}
    save_json(receipt,r);return r


def prepare(ctx, root, out, round_no, before=lambda:None, progress=lambda **kwargs:None):
    identity=representation_identity(ctx,root); folder=out/'representation_inputs';folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'contract.json',identity); cid=sha(folder/'contract.json')
    done=folder/'manifest.json'
    if done.exists():
        manifest=read(done)
        if manifest['contract_id']!=cid: raise ValueError('Prepared data identity differs')
        for rel,digest in manifest['files'].items(): checked(folder/rel,digest)
        return {'status':f'ROUND{round_no:02d}_REPRESENTATION_INPUTS_READY','reused':True}
    before(); events=source_events(ctx); vocabulary,df=select_vocabulary(events)
    if len(vocabulary)<16: raise ValueError('Insufficient historical vocabulary; stop and review')
    digest=save_arrays(folder/'vocabulary.npz',items=vocabulary,distinct_session_frequency=df)
    files={'vocabulary.npz':digest}; progress(stage='historical_vocabulary',completed=len(vocabulary),total=VOCABULARY_CAP)
    if round_no==11:
        matrices=incidence(events,vocabulary)
        for k,m in enumerate(matrices):
            before();name=f'incidence-{k}.npz';r=store_matrix(folder/name,m,{'contract_id':cid,'view':k});files[name]=r['sha256']
            progress(stage='session_incidence',completed=k+1,total=4)
        diagnostics={'session_count':matrices[0].shape[1],'nonzero_incidence':matrices[0].nnz,'source_history_events':len(events)}
    else:
        row_count=0; chunks=[]
        for offset in range(1,MAX_GAP+1):
            before();path=folder/f'pairs-{offset}.npz';receipt=path.with_suffix('.json')
            unit={'contract_id':cid,'offset':offset}
            if receipt.exists():
                r=read(receipt)
                if r['unit']!=unit: raise ValueError('Pair checkpoint identity differs')
                checked(path,r['sha256'])
                with np.load(path,allow_pickle=False) as z:f=z['forward'];b=z['reverse']
            else:
                if path.exists():raise ValueError('Orphan pair checkpoint preserved')
                f,b=pair_rows(events,vocabulary,offset)
                if row_count+len(f)+len(b)>EDGE_RECORD_CAP:raise ValueError('Eight-million pair-record bound; stop, no silent trimming')
                digest=save_arrays(path,forward=f,reverse=b)
                save_json(receipt,{'unit':unit,'sha256':digest})
            row_count+=len(f)+len(b)
            if row_count>EDGE_RECORD_CAP:raise ValueError('Pair-record bound exceeded')
            chunks.append((f,b));files[path.name]=sha(path)
            progress(stage='original_index_pair_chunks',completed=offset,total=MAX_GAP)
        forward=np.concatenate([v[0] for v in chunks]); unsigned=np.concatenate([v for pair in chunks for v in pair]);del chunks
        for arm,rows in [('primary',forward),('ablation',unsigned)]:
            for kind in range(3):
                before();m=count_pairs(rows,len(vocabulary),kind);name=f'{arm}-{kind}.npz'
                r=store_matrix(folder/name,m,{'contract_id':cid,'arm':arm,'kind':kind});files[name]=r['sha256']
                progress(stage='typed_session_unique_context_counts',completed=(0 if arm=='primary' else 3)+kind+1,total=6)
        diagnostics={'pair_records':row_count,'source_history_events':len(events),'source_history_sessions':len(np.unique(events[:,0]))}
    save_json(done,{'contract_id':cid,'files':files,'vocabulary_size':len(vocabulary),
                    'status':f'ROUND{round_no:02d}_REPRESENTATION_INPUTS_READY','query_labels_read':False,**diagnostics})
    return {'status':f'ROUND{round_no:02d}_REPRESENTATION_INPUTS_READY','reused':False,**diagnostics}


def learn(ctx, root, out, round_no, before=lambda:None, progress=lambda **kwargs:None):
    folder=out/'representation_inputs'; manifest=read(folder/'manifest.json')
    if read(folder/'contract.json')!=representation_identity(ctx,root):raise ValueError('Representation source or environment changed')
    for rel,digest in manifest['files'].items():checked(folder/rel,digest)
    dest=out/'representations';dest.mkdir(exist_ok=True)
    contract={'prepared_sha256':sha(folder/'manifest.json'),'protocol_sha256':sha(root/'protocol.json'),
              'feature_implementation_sha256':sha(root/'latent_features.py'),'representation_code_sha256':sha(root/'latent_models.py')}
    save_json(dest/'contract.json',contract);cid=sha(dest/'contract.json')
    with np.load(folder/'vocabulary.npz',allow_pickle=False) as z:vocabulary=z['items']
    inventory={};units=0
    for arm in ('primary','ablation'):
        kinds=[None] if round_no==11 else list(range(3))
        for kind in kinds:
            path=dest/(f'{arm}.npz' if kind is None else f'{arm}-{kind}.npz');receipt=path.with_suffix('.json')
            unit={'contract_id':cid,'arm':arm,'kind':kind}
            if receipt.exists():
                r=read(receipt)
                if r['unit']!=unit:raise ValueError('Saved representation identity differs')
                checked(path,r['sha256'])
            else:
                if path.exists():raise ValueError('Orphan latent representation preserved')
                before()
                if round_no==11:
                    matrices=[restore_matrix(folder/f'incidence-{k}.npz') for k in range(4)]
                    q,t,s=session_embedding(matrices,arm=='primary');del matrices
                else:
                    q,t,s=transition_embedding(restore_matrix(folder/f'{arm}-{kind}.npz'))
                digest=save_arrays(path,query=q,target=t,singular_values=s)
                save_json(receipt,{'unit':unit,'sha256':digest,'effective_rank':len(s),'self_supervised_factorizations':1})
            inventory[path.name]=sha(path);units+=1
            progress(stage='historical_factorizations',completed=units,total=2 if round_no==11 else 6)
            gc.collect()
    save_json(dest/'manifest.json',{'status':f'ROUND{round_no:02d}_REPRESENTATIONS_READY','contract_id':cid,
               'files':inventory,'vocabulary_sha256':sha(folder/'vocabulary.npz'),'query_labels_read':False,
               'representation_fits_planned':2 if round_no==11 else 6,'ranker_fits':0})
    load_embeddings(root,out,round_no)
    return {'status':f'ROUND{round_no:02d}_REPRESENTATIONS_READY','verified_units':units,'ranker_fits':0}


def load_embeddings(root,out,round_no):
    folder=out/'representations';manifest=read(folder/'manifest.json')
    if manifest['contract_id']!=sha(folder/'contract.json'):raise ValueError('Representation contract hash differs')
    contract=read(folder/'contract.json')
    expected={'prepared_sha256':sha(out/'representation_inputs/manifest.json'),'protocol_sha256':sha(root/'protocol.json'),
              'feature_implementation_sha256':sha(root/'latent_features.py'),'representation_code_sha256':sha(root/'latent_models.py')}
    if contract!=expected:raise ValueError('Representation code/configuration changed')
    vocabpath=out/'representation_inputs/vocabulary.npz';checked(vocabpath,manifest['vocabulary_sha256'])
    with np.load(vocabpath,allow_pickle=False) as z:vocabulary=z['items']
    result={}
    for arm in ('primary','ablation'):
        qs=[];ts=[]
        for kind in ([None] if round_no==11 else range(3)):
            name=f'{arm}.npz' if kind is None else f'{arm}-{kind}.npz';checked(folder/name,manifest['files'][name])
            with np.load(folder/name,allow_pickle=False) as z:qs.append(z['query']);ts.append(z['target'])
        q=qs[0] if round_no==11 else np.stack(qs);t=ts[0] if round_no==11 else np.stack(ts)
        for a in (q,t):
            if a.shape!=(3,len(vocabulary),DIMENSION) or not np.isfinite(a).all():raise ValueError('Latent representation schema changed')
            norms=np.linalg.norm(a,axis=2)
            if not np.all((norms<1e-8)|np.isclose(norms,1,atol=2e-5)):raise ValueError('Latent vectors must be unit length or zero')
        result[arm]=(q,t)
    return vocabulary,result
