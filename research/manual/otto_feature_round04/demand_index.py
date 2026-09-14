"""Bounded exact event index from complete JSONL sessions in permitted raw TRAIN.

Every selected study session is excluded wholesale. Future events remain unread by
feature lookups: an hourly snapshot consults only bins strictly before its cutoff.
No test input, labels, learned graph or target-dependent sampling is used.
"""
from __future__ import annotations
import hashlib
import math
import time
from pathlib import Path
import numpy as np
import orjson
from core import save_arrays, save_json, sha, checked
from demand_features import HOUR, WINDOWS

ACTION={'clicks':0,'carts':1,'orders':2}
SHARD_WIDTH=250000
CHUNK_BYTES=64*1024**2

class PauseIndex(Exception):pass

def aggregate(keys,counts,last):
    k,c,t=map(np.asarray,(keys,counts,last))
    if not len(k):return np.empty(0,np.int64),np.empty(0,np.int64),np.empty(0,np.int64)
    if k.shape!=c.shape or c.shape!=t.shape or (c<=0).any() or (k<0).any():raise ValueError('Invalid event aggregates')
    ix=np.argsort(k,kind='stable');k=k[ix];c=c[ix];t=t[ix]
    begin=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1]
    return k[begin],np.add.reduceat(c,begin),np.maximum.reduceat(t,begin)

def chunk(raw,start,*,minimum_hour,maximum_hour,excluded,catalogue,budget_bytes=CHUNK_BYTES):
    """Read complete JSONL lines; exact byte offsets make normal pauses resumable."""
    bins=maximum_hour-minimum_hour+1
    if minimum_hour<0 or bins<169 or budget_bytes<1:raise ValueError('Invalid hourly index range')
    keys=[];ts=[];global_counts=np.zeros((bins,3),np.int64)
    seen=used=excluded_count=future_sessions=events=0
    with Path(raw).open('rb') as f:
        f.seek(start)
        if start:
            f.seek(start-1)
            if f.read(1)!=b'\n':raise ValueError('Resume offset is not a complete-session boundary')
        digest=hashlib.sha256()
        while True:
            line=f.readline(16*1024**2+1)
            if not line:break
            if len(line)>16*1024**2:raise ValueError('Raw session line exceeds resource limit')
            digest.update(line);seen+=1
            s=orjson.loads(line)
            session=s.get('session');ev=s.get('events')
            if type(session) is not int or session<0 or not isinstance(ev,list) or not ev:
                raise ValueError('Raw session schema invalid')
            if session in excluded:excluded_count+=1
            elif type(ev[0].get('ts')) is int and ev[0]['ts']>=maximum_hour*HOUR:
                future_sessions+=1
            else:
                previous=-1
                for e in ev:
                    aid,t,kind=e.get('aid'),e.get('ts'),e.get('type')
                    if type(aid) is not int or aid<0 or type(t) is not int or t<0 or t<previous or kind not in ACTION:
                        raise ValueError('Malformed or unordered raw training events')
                    previous=t;events+=1
                    if t>=maximum_hour*HOUR:continue
                    h=max(0,t//HOUR-minimum_hour+1);a=ACTION[kind]
                    if h>=bins:raise ValueError('Future event passed exclusive index cutoff')
                    # Global totals include ALL items, independent of candidate catalogue filtering.
                    global_counts[h,a]+=1;used+=1
                    if aid in catalogue:
                        keys.append((aid*3+a)*bins+h);ts.append(t)
            if f.tell()-start>=budget_bytes:break
        end=f.tell()
    k=np.asarray(keys,np.int64);t=np.asarray(ts,np.int64)
    k,c,t=aggregate(k,np.ones(len(k),np.int64),t)
    return {'keys':k,'counts':c,'last':t,'global_counts':global_counts}, {
        'start':start,'end':end,'raw_chunk_sha256':digest.hexdigest(),'sessions_read':seen,
        'study_sessions_excluded':excluded_count,'later_sessions_skipped':future_sessions,
        'events_inspected':events,'events_before_maximum_cutoff':used,
        'indexed_item_hour_action_rows':len(k)}

def group_cumsum(keys,counts,bins):
    if not len(keys):return np.empty(0,np.int64)
    groups=keys//bins;start=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1]
    sums=np.cumsum(counts,dtype=np.int64);offset=np.r_[0,sums[start[1:]-1]]
    return sums-np.repeat(offset,np.diff(np.r_[start,len(keys)]))

class HourlyIndex:
    def __init__(self,keys,cumulative,last,global_counts,minimum_hour,maximum_hour):
        self.keys=np.asarray(keys,np.int64);self.cumulative=np.asarray(cumulative,np.int64);self.last=np.asarray(last,np.int64)
        self.minimum_hour=int(minimum_hour);self.maximum_hour=int(maximum_hour);self.bins=maximum_hour-minimum_hour+1
        self.global_counts=np.asarray(global_counts,np.int64)
        if (self.keys.shape!=self.cumulative.shape or self.keys.shape!=self.last.shape
            or self.global_counts.shape!=(self.bins,3) or (np.diff(self.keys)<=0).any()
            or (self.cumulative<=0).any() or (self.last<0).any() or (self.last>=maximum_hour*HOUR).any()
            or (self.global_counts<0).any()):raise ValueError('Invalid merged hourly index')
        self.global_prefix=np.vstack([np.zeros((1,3),np.int64),self.global_counts.cumsum(0)])

    def _before(self,groups,relative_hour):
        boundary=groups*self.bins+relative_hour
        ix=np.searchsorted(self.keys,boundary,side='left')-1
        vals=np.zeros(groups.shape,np.int64);last=np.full(groups.shape,-1,np.int64)
        if len(self.keys):
            safe=np.clip(ix,0,len(self.keys)-1);valid=(ix>=0)&(self.keys[safe]//self.bins==groups)
            vals[valid]=self.cumulative[safe[valid]];last[valid]=self.last[safe[valid]]
        return vals,last

    def snapshot(self,aids,cutoff):
        aa=np.asarray(aids)
        if aa.ndim!=1 or aa.dtype.kind not in 'iu' or np.unique(aa).size!=aa.size or (aa<0).any():
            raise ValueError('Unique integer candidates required')
        if cutoff%HOUR or not self.minimum_hour+168<=cutoff//HOUR<=self.maximum_hour:
            raise ValueError('Snapshot outside certified hourly range')
        end=cutoff//HOUR-self.minimum_hour+1
        groups=aa[:,None]*3+np.arange(3)[None,:]
        before,last=self._before(groups,end);counts=np.empty((len(aa),5,3),np.int64)
        totals=np.empty((5,3),np.int64)
        for j,h in enumerate(WINDOWS):
            past,_=self._before(groups,end-h);counts[:,j]=before-past
            totals[j]=self.global_prefix[end]-self.global_prefix[end-h]
        if (counts<0).any():raise ValueError('Negative cumulative difference')
        return counts,last,totals

    @classmethod
    def load(cls,out):
        import json
        m=json.loads((out/'index_manifest.json').read_text());c=m['contract']
        ks=[];cs=[];ls=[]
        for r in m['shards']:
            p=out/r['path'];checked(p,r['sha256'])
            with np.load(p,allow_pickle=False) as z:ks.append(z['keys']);cs.append(z['cumulative']);ls.append(z['last'])
        checked(out/'global.npz',m['global_sha256'])
        with np.load(out/'global.npz',allow_pickle=False) as z:glob=z['counts']
        return cls(np.concatenate(ks),np.concatenate(cs),np.concatenate(ls),glob,c['minimum_hour'],c['maximum_hour'])

def build(raw,out,contract,*,deadline,progress,log,chunk_bytes=CHUNK_BYTES):
    import json
    out.mkdir(parents=True,exist_ok=True);save_json(out/'index_contract.json',contract)
    cid=sha(out/'index_contract.json');minimum=contract['minimum_hour'];maximum=contract['maximum_hour'];bins=maximum-minimum+1
    excluded=set(contract['excluded_sessions']);catalogue=set(contract['candidate_catalogue'])
    n_shards=int(max(catalogue)//SHARD_WIDTH+1);position=0;number=0;receipts=[]
    if n_shards>16:raise ValueError('Unexpected catalogue range')
    raw_size=Path(raw).stat().st_size
    while position<raw_size:
        receipt=out/'chunks'/f'{number:04d}.json'
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if r['contract_id']!=cid or r['start']!=position or not position<r['end']<=raw_size:
                raise ValueError('Raw-chunk checkpoint lineage differs')
            for s in r['shards']:checked(out/s['path'],s['sha256'])
            checked(out/r['global']['path'],r['global']['sha256'])
        else:
            if time.monotonic()>deadline-20:raise PauseIndex('Completed raw chunks preserved; resume index only after reviewing progress')
            start=time.monotonic();a,r=chunk(raw,position,minimum_hour=minimum,maximum_hour=maximum,
                              excluded=excluded,catalogue=catalogue,budget_bytes=chunk_bytes)
            r['contract_id']=cid;r['shards']=[]
            # Unit output is immutable and verified before committing its receipt.
            shard_ids=(a['keys']//bins//3)//SHARD_WIDTH
            for s in range(n_shards):
                mask=shard_ids==s
                rel=f'chunks/{number:04d}-s{s:02d}.npz'
                hh=save_arrays(out/rel,keys=a['keys'][mask],counts=a['counts'][mask],last=a['last'][mask])
                r['shards'].append({'path':rel,'sha256':hh})
            rel=f'chunks/{number:04d}-global.npz';hh=save_arrays(out/rel,counts=a['global_counts'])
            r['global']={'path':rel,'sha256':hh}
            save_json(receipt,r)
            log('raw_chunk_complete',chunk=number,raw_bytes_done=r['end'],raw_bytes_total=raw_size,
                seconds=round(time.monotonic()-start,3),source_sessions_excluded=r['study_sessions_excluded'])
        position=r['end'];number+=1;receipts.append(r)
        progress.update(stage='raw_demand_index',completed=position,total=raw_size,completed_chunks=number)
    shards=[]
    for s in range(n_shards):
        if time.monotonic()>deadline-20:raise PauseIndex('Raw scan complete; completed merged shards preserved')
        p=out/'shards'/f'{s:02d}.npz';rp=p.with_suffix('.json')
        if rp.exists():
            rr=json.loads(rp.read_text())
            if rr['contract_id']!=cid:raise ValueError('Merged shard contract differs')
            checked(p,rr['sha256'])
        else:
            kk=[];cc=[];tt=[]
            for r in receipts:
                with np.load(out/r['shards'][s]['path'],allow_pickle=False) as z:
                    kk.append(z['keys']);cc.append(z['counts']);tt.append(z['last'])
            k,c,t=aggregate(np.concatenate(kk),np.concatenate(cc),np.concatenate(tt))
            cumulative=group_cumsum(k,c,bins);hh=save_arrays(p,keys=k,cumulative=cumulative,last=t)
            rr={'contract_id':cid,'sha256':hh,'rows':len(k)};save_json(rp,rr)
            del kk,cc,tt,k,c,t,cumulative
        shards.append({'path':str(p.relative_to(out)),'sha256':rr['sha256'],'rows':rr['rows']})
        progress.update(stage='merge_demand_shards',completed=s+1,total=n_shards)
    glob=np.zeros((bins,3),np.int64)
    for r in receipts:
        with np.load(out/r['global']['path'],allow_pickle=False) as z:glob+=z['counts']
    gh=save_arrays(out/'global.npz',counts=glob)
    result={'status':'ROUND04_INDEX_READY','contract':contract,'contract_id':cid,'shards':shards,
        'global_sha256':gh,'raw_bytes_processed':position,'raw_chunks':len(receipts),
        'raw_sessions_read':sum(r['sessions_read'] for r in receipts),
        'source_study_sessions_excluded':sum(r['study_sessions_excluded'] for r in receipts),
        'source_sessions_skipped_after_last_cutoff':sum(r['later_sessions_skipped'] for r in receipts),
        'global_events_before_last_cutoff':int(glob.sum()),'candidate_catalogue_size':len(catalogue)}
    save_json(out/'index_manifest.json',result)
    return result
