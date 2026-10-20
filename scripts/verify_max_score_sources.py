"""Read-only early/middle/late numerical snapshot acceptance for exact sources.

All thresholds and w0 below are engineering fixtures, never an investment policy.
Raw source publication/price semantics remain unverified. No real execution is
claimed. Output is LOCAL ONLY (contains exact source paths and fingerprints).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import polars as pl
from optimizer import InProcessOptimizerClient
from optimizer.transport.shm import ShmOptimizerClient


def fingerprint(path: Path) -> dict:
    s=path.stat()
    with path.open('rb') as f:
        first=f.read(65536);f.seek(max(0,s.st_size-65536));last=f.read(65536)
    return {'path':str(path.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'first_last_64k_sha256':hashlib.sha256(first+last).hexdigest()}


def read_dates(path: Path, dates: list[str]) -> dict[str,dict[str,float]]:
    lf=pl.scan_parquet(path);names=lf.collect_schema().names()
    date=next((x for x in ['__index_level_0__','trade_day','date'] if x in names),None)
    if date is None:raise ValueError(f'no supported date field: {path.name}')
    frame=lf.filter(pl.col(date).dt.strftime('%Y-%m-%d').is_in(dates)).collect()
    if frame.height!=len(dates):raise ValueError(f'date coverage/uniqueness failure: {path.name}')
    result={}
    for row in frame.iter_rows(named=True):
        stamp=str(row.pop(date))[:10]
        result[stamp]=row
    return result


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--market-root',type=Path,required=True);p.add_argument('--barra-root',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);selection=json.loads(a.selection.read_text());dates=[r['date'] for r in selection['dates']]
    if selection.get('purpose')!='engineering_acceptance':raise ValueError('explicit engineering selection required')
    started=time.perf_counter();paths={'pred':a.predictions,**{k:a.market_root/f for k,f in {'industry':'industry_citics_2019_1.parquet','listed':'listed_days.parquet','normal':'nosuspend_days.parquet','st':'is_st.parquet','turnover':'total_turnover.parquet','close':'close.parquet','cap':'VALUATION_market_cap_3.parquet'}.items()}}
    fingerprints=[fingerprint(x) for x in paths.values()];tables={key:read_dates(path,dates) for key,path in paths.items()};size={}
    for stamp in dates:
        path=a.barra_root/stamp[:4]/'SIZE.parquet';fingerprints.append(fingerprint(path));frame=pl.scan_parquet(path).filter(pl.col('date').cast(pl.String)==stamp).collect()
        if frame.select(pl.struct(['date','symbol']).is_duplicated().any()).item():raise ValueError('duplicate SIZE keys')
        if frame['version'].unique().to_list()!=['v3_0']:raise ValueError('SIZE version changed')
        size[stamp]=dict(frame.select('symbol','factor_value').iter_rows())
    read_ms=(time.perf_counter()-started)*1000
    report={'purpose':'engineering_acceptance','source_point_in_time_status':'unverified','real_execution':'BLOCKED','price_and_eligibility_semantics':'unverified','selection':selection,'source_fingerprints':fingerprints,'read_align_ms':read_ms,'snapshots':[]}
    socket=a.output/'service.sock'
    with (a.output/'service.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','optimizer.transport.shm.service','--socket',str(socket),'--max-assets','6000'],stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+15
            while not socket.exists():
                if proc.poll() is not None or time.monotonic()>deadline:raise RuntimeError('service startup failed')
                time.sleep(.01)
            with InProcessOptimizerClient(schema_version='1.2',max_assets=6000) as direct,ShmOptimizerClient(socket,schema_version='1.2',request_timeout_ms=15000) as shm:
                for stamp in dates:
                    raw=tables['pred'][stamp];finite=sorted(s for s,v in raw.items() if v is not None and np.isfinite(v));candidates=[];excluded={};rules={'listed':lambda v:v>=180,'normal':lambda v:v>=10,'st':lambda v:v==0,'turnover':lambda v:v>0}
                    for symbol in finite:
                        reason=next((key for key,predicate in rules.items() if (v:=tables[key][stamp].get(symbol)) is None or not np.isfinite(v) or not predicate(v)),None)
                        if reason:excluded[reason]=excluded.get(reason,0)+1
                        else:candidates.append(symbol)
                    n=len(candidates);scores=np.array([raw[s] for s in candidates],dtype=np.float64);z=np.array([size[stamp].get(s,np.nan) for s in candidates],dtype=np.float64);ind=np.array([tables['industry'][stamp].get(s,np.nan) for s in candidates],dtype=np.float64)
                    row={'date':stamp,'grid_denominator':len(raw),'finite_prediction_denominator':len(finite),'candidate_count':n,'explicit_eligibility_exclusions':excluded,'missing_size':int(np.count_nonzero(~np.isfinite(z))),'invalid_industry':int(np.count_nonzero(~np.isfinite(ind)|(ind<=0)|(ind!=np.floor(ind)))),'missing_close':sum(not (tables['close'][stamp].get(s) or 0)>0 for s in candidates),'missing_cap':sum(not (tables['cap'][stamp].get(s) or 0)>0 for s in candidates),'test_w0':'uniform synthetic, not historical holdings','complete_held_count':n,'missing_held_scores':0}
                    assert len(finite)==n+sum(excluded.values())
                    if row['missing_size'] or row['invalid_industry']:
                        row.update(status='BLOCKED',blocker='raw required exposure missing; no completeness filtering or fill');report['snapshots'].append(row);continue
                    codes=ind.astype(np.int64);unique=sorted(set(codes.tolist()));w0=np.full(n,.8/n,dtype=np.float64);mean=float(z@w0)
                    constraints=[{'type':'max_weight','upper':.01},{'type':'size','exposure_id':'raw_SIZE_v3_0','unit':'source_raw_unknown_standardization','lower':mean-.05,'upper':mean+.05},{'type':'industry','classification_id':'citics_2019_1','bounds':[{'industry_id':str(c),'lower':0.,'upper':float(w0[codes==c].sum()+.03)} for c in unique]},{'type':'turnover','definition':'stock_half_l1','limit':.2}]
                    req={'schema_version':'1.2','request_id':'real-'+stamp,'data_context':{'as_of':stamp+'T15:00:00+08:00','planned_execution_at':stamp+'T16:00:00+08:00','frequency':'daily'},'portfolio_policy':{'long_only':True,'fully_invested':False,'allow_cash':True,'allow_leverage':False,'cash_policy':'retain','target_budget':.8},'model':{'type':'max_score','version':'1','config':{}},'universe':{'asset_ids':candidates},'inputs':{'scores':scores,'current_weights':w0,'current_cash':np.array([.2]),'size_exposure':z,'industry_codes':codes},'constraints':constraints,'snapshot':{'valuation_at':stamp+'T15:00:00+08:00','weight_denominator':'total_nav','asset_order_hash':hashlib.sha256(json.dumps(candidates,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),'source_point_in_time_status':'unverified','size':{'exposure_id':'raw_SIZE_v3_0','field':'factor_value:SIZE','source':'BAR-001','version':'v3_0','unit':'source_raw_unknown_standardization','transform':'none','semantics_status':'unverified'},'industry':{'classification_id':'citics_2019_1','source':'MKT-024','version':'snapshot_unverified','level':'1','code_to_id':{str(c):str(c) for c in unique}}},'options':{'timeout_ms':10000,'deterministic':True}}
                    results=[];weights=[]
                    for client in [direct,shm]:
                        lease,wait=client.optimize(req)
                        try:weights.append(np.array(lease.weights,copy=True));results.append({'wait_ms':wait,'diagnostics':lease.response['diagnostics']})
                        finally:lease.release()
                    error=float(np.max(np.abs(weights[0]-weights[1])));assert error<=1e-8
                    row.update(status='PASS',max_weight_abs_difference=error,results=results,nonzero_weights=int(np.count_nonzero(weights[0])),constraints=constraints,asset_order_hash=req['snapshot']['asset_order_hash']);report['snapshots'].append(row)
        finally:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
    report['status']='PASS' if all(r['status']=='PASS' for r in report['snapshots']) else 'BLOCKED'
    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['source_fingerprints','snapshots']},indent=2));print([(r['date'],r['status'],r['candidate_count']) for r in report['snapshots']])

if __name__=='__main__':main()
