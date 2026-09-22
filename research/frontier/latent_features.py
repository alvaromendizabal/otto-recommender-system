"""Historical graph-only low-rank item contexts, plus label-blind affinity features.

A = D_out^-1/2 log1p(W_time) D_in^-1/2; randomized range finding,
one stabilized power iteration, small projected Gram eigendecomposition.
The item representation is row-normalized approximate U sqrt(S).
This is NOT an implementation of NetMF, ALS, or Word2Vec.
"""
from __future__ import annotations
import gc, os, time
from pathlib import Path
import numpy as np
from scipy import sparse, linalg
from .contracts import require, load, save, digest, identity

SCOPES=('all','clicks','carts','orders')
NAMES=tuple(f'latent_{s}_{a}' for s in SCOPES for a in ('max_cosine','mean_cosine','position_cosine'))+(
 'latent_last_cosine','latent_candidate_known','latent_prefix_known_share','latent_candidate_norm')

def array_checkpoint(path, contract, maker):
    """Numpy checkpoints commit data and its staged receipt; no orphan is trusted."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rec=path.with_suffix('.json');pending=path.with_suffix('.pending.json');tmp=path.with_suffix('.pending.npy')
    require(not any(p.is_symlink() for p in (path,rec,pending,tmp)), 'ARRAY_SYMLINK_REJECTED')
    if pending.exists() and not rec.exists():
        r=load(pending);require(r['contract']==contract,'ARRAY_PENDING_CONTRACT')
        chosen=path if path.exists() else tmp;checked_array(chosen,r)
        if chosen==tmp:os.replace(tmp,path)
        os.replace(pending,rec)
    if rec.exists():
        r=load(rec);require(r['contract']==contract,'ARRAY_CONTRACT_CHANGED');checked_array(path,r)
        return np.load(path,mmap_mode='r',allow_pickle=False),True
    require(not path.exists(),'UNCOMMITTED_ARRAY_PRESERVED')
    # A .pending.npy without a staged receipt was never committed; do not use it.
    a=np.asarray(maker(),dtype=np.float32)
    require(a.ndim in (1,2) and np.isfinite(a).all(),'INVALID_NEW_ARRAY')
    with tmp.open('wb') as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
    r={'contract':contract,'shape':list(a.shape),'dtype':a.dtype.str,'sha256':digest(tmp)}
    save(pending,r);os.replace(tmp,path);os.replace(pending,rec)
    del a;gc.collect()
    return np.load(path,mmap_mode='r',allow_pickle=False),False

def checked_array(path,r):
    require(Path(path).is_file() and digest(path)==r['sha256'],'ARRAY_CHECKSUM')
    a=np.load(path,mmap_mode='r',allow_pickle=False)
    require(list(a.shape)==r['shape'] and a.dtype.str==r['dtype'] and not a.dtype.hasobject,'ARRAY_SCHEMA')

def normalized_graph(graph):
    require(sparse.issparse(graph) and graph.shape[0]==graph.shape[1],'SQUARE_SPARSE_GRAPH')
    a=graph.astype(np.float32,copy=True).tocsr();a.sum_duplicates();a.eliminate_zeros()
    require(a.nnz>0 and np.isfinite(a.data).all() and np.all(a.data>=0),'GRAPH_WEIGHT_DOMAIN')
    a.data=np.log1p(a.data)
    row=np.asarray(a.sum(1)).ravel();col=np.asarray(a.sum(0)).ravel()
    ro=np.divide(1.,np.sqrt(row),out=np.zeros_like(row),where=row>0)
    co=np.divide(1.,np.sqrt(col),out=np.zeros_like(col),where=col>0)
    for start in range(0,a.shape[0],20000):
        end=min(start+20000,a.shape[0]);b,e=a.indptr[start],a.indptr[end]
        a.data[b:e]*=np.repeat(ro[start:end],np.diff(a.indptr[start:end+1]))*co[a.indices[b:e]]
    require(np.isfinite(a.data).all(),'NORMALIZED_GRAPH_FINITE')
    return a,row>0

def embed(graph, folder, contract, rank=64, oversampling=8, seed=20260924,
          boundary=lambda:None, emit=lambda *a,**kw:None, max_seconds=600):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cid=identity({'source':contract,'rank':rank,'oversampling':oversampling,'seed':seed,'power_iterations':1,'formula':'normalized_log_time_graph_U_sqrtS_v1'})
    done=folder/'embedding.json'
    if done.exists():
        r=load(done);require(r['contract']==cid,'EMBEDDING_CONTRACT_CHANGED')
        e,_=array_checkpoint(folder/'vectors.npy',cid,lambda:(_ for _ in ()).throw(RuntimeError('missing completed embedding')))
        n,_=array_checkpoint(folder/'norms.npy',cid,lambda:(_ for _ in ()).throw(RuntimeError('missing completed norms')))
        require(digest(folder/'vectors.npy')==r['vectors_sha256'] and digest(folder/'norms.npy')==r['norms_sha256'],'EMBEDDING_SEAL')
        return e,n,r
    tick=time.monotonic()
    def gate():
        boundary()
        if time.monotonic()-tick>max_seconds:raise TimeoutError('EMBEDDING_PHASE_BUDGET; completed factors preserved')
    gate();a,known=normalized_graph(graph);n=a.shape[0];l=rank+oversampling
    require(1<=rank and l<n,'EMBEDDING_RANK_SHAPE')
    emit('embedding_graph_ready',items=n,edges=a.nnz,dimensions=rank)
    def q0():
        gate();emit('embedding_range_projection',phase=1,total_phases=3)
        omega=np.random.default_rng(seed).standard_normal((n,l),dtype=np.float32)
        y=a@omega;del omega
        q,_=linalg.qr(y,mode='economic',overwrite_a=True,check_finite=False);gate();return q
    q,_=array_checkpoint(folder/'range_q0.npy',cid,q0)
    def q1():
        gate();emit('embedding_power_iteration',phase=2,total_phases=3)
        z=a.T@q;z,_=linalg.qr(z,mode='economic',overwrite_a=True,check_finite=False)
        y=a@z;del z
        qq,_=linalg.qr(y,mode='economic',overwrite_a=True,check_finite=False);gate();return qq
    q1a,_=array_checkpoint(folder/'range_q1.npy',cid,q1);del q;gc.collect()
    # A norms checkpoint is written inside the representation computation, before
    # vectors are committed. Every component still binds to the same contract.
    def vectors():
        gate();emit('embedding_projected_factorization',phase=3,total_phases=3)
        b=a.T@q1a;gram=np.zeros((l,l),np.float64)
        for start in range(0,len(b),20000):
            gate();bb=np.asarray(b[start:start+20000],np.float64);gram+=bb.T@bb
        del b,bb;gc.collect()
        eig,v=np.linalg.eigh((gram+gram.T)*.5);order=np.argsort(eig)[::-1][:rank]
        require(float(eig.min())>=-1e-6,'PROJECTED_GRAM_NOT_PSD')
        vals=np.maximum(eig[order],0.);require(vals[0]>0 and np.isfinite(vals).all(),'INVALID_SPECTRUM')
        e=np.asarray(q1a@np.asarray(v[:,order],np.float32),np.float32)
        e*=np.power(vals,.25).astype(np.float32);e[~known]=0
        norms=np.linalg.norm(e,axis=1).astype(np.float32)
        for start in range(0,len(e),20000):
            gate();nn=norms[start:start+20000];e[start:start+20000]/=np.maximum(nn[:,None],1e-12)
        array_checkpoint(folder/'norms.npy',cid,lambda:norms)
        save(folder/'spectrum.json',{'contract':cid,'singular_values':np.sqrt(vals).tolist(),'graph_frobenius_squared':sum(float(np.dot(vv,vv)) for start in range(0,len(a.data),1000000) for vv in [a.data[start:start+1000000].astype(np.float64)])})
        gate();return e
    e,reused=array_checkpoint(folder/'vectors.npy',cid,vectors)
    norms,_=array_checkpoint(folder/'norms.npy',cid,lambda:(_ for _ in ()).throw(RuntimeError('EMBEDDING_NORMS_MISSING')))
    r={'status':'HISTORICAL_EMBEDDING_READY','contract':cid,'dimensions':rank,'items':n,'nonzero_items':int((norms>1e-12).sum()),
       'edges':a.nnz,'vectors_sha256':digest(folder/'vectors.npy'),'norms_sha256':digest(folder/'norms.npy'),
       'elapsed_seconds':time.monotonic()-tick,'labels_used':False,'scope':'One approximate normalized time-graph factorization; not Word2Vec/ALS/NetMF.'}
    save(done,r);return e,norms,r

def affinity(prefix,aids,embeddings,norms):
    prefix.validate();ids=np.asarray(aids)
    require(ids.ndim==1 and ids.dtype.kind in 'iu' and np.all(ids>=0) and len(np.unique(ids))==len(ids),'AFFINITY_CANDIDATE_IDS')
    e=np.asarray(embeddings);norms=np.asarray(norms);require(e.ndim==2 and norms.ndim==1 and len(norms)==len(e),'EMBEDDING_SHAPE')
    def lookup(a):
        a=np.asarray(a);valid=a<len(e);out=np.zeros((len(a),e.shape[1]),np.float32);out[valid]=e[a[valid]]
        known=np.zeros(len(a),bool);known[valid]=norms[a[valid]]>1e-12
        out[~known]=0
        return out,known
    cand,ck=lookup(ids);recent=prefix.aid[-20:][::-1];kinds=prefix.kind[-20:][::-1]
    context,pk=lookup(recent);sim=np.clip(cand@context.T,-1,1)
    cols=[];position=1/np.sqrt(np.arange(1,len(recent)+1))
    for s in (-1,0,1,2):
        mask=pk if s==-1 else pk&(kinds==s)
        if mask.any():
            part=sim[:,mask];w=position[mask];cols.extend([part.max(1),part.mean(1),part@w/w.sum()])
        else:cols.extend([np.zeros(len(ids)) for _ in range(3)])
    raw=np.zeros(len(ids),np.float32);valid=ids<len(e);raw[valid]=norms[ids[valid]]
    cols.extend([sim[:,0] if pk[0] else np.zeros(len(ids)),ck.astype(np.float32),np.full(len(ids),pk.mean()),raw])
    x=np.column_stack(cols).astype(np.float32)
    require(x.shape==(len(ids),16) and np.isfinite(x).all(),'AFFINITY_FEATURES_FINITE')
    return x
