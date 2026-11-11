#!/usr/bin/env python3
"""Atomiq N10/N11/N12, routed by poutcome=unknown, global 300-row OOF.

Two modes:
  cached  : use existing N7-N9 OOF file; evaluate N10/N12 immediately.
            N11 is NOT computable from saved probabilities alone.
  retrain : refit F2 on the full outer-training fold, train N8 and N11 only
            on unknown training rows, and evaluate N10/N11/N12 on all 300.
            Requires Qiskit and adjacent previous Atomiq scripts.

N10: for unknown rows, (1-alpha)*F2 + alpha*N8_ring_top40.
N11: for unknown rows, F2-style classical + 15q features + 6q connected
     pairs + 14q ring singles and 40 train-selected connected pairs, fit as
     ONE classifier on unknown outer-training rows only.
N12: for unknown rows, mean(N8_ring_top25, N8_ring_top40, N8_ring_top60).
Non-unknown rows: global fold-trained F2, never a 46-person-only classifier.

All model fitting and pair selection is within outer train folds. Overall
OOF hyperparameter comparisons are exploratory model selection; do not treat
best OOF alpha as an independent generalization estimate.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

SEEDS=(42,179,316)
PERCENT=(0,10,20,30,40,50,60,80,100)
N8_KEYS=('p_N8_ring_top25','p_N8_ring_top40','p_N8_ring_top60')

def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),1e-10,1-1e-10)
    ix=np.argsort(-p,kind='stable')[:min(10,len(p))]
    return {'n':int(len(y)),'positives':int(y.sum()),
            'PR_AUC':float(average_precision_score(y,p)) if y.sum() else np.nan,
            'ROC_AUC':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
            'Brier':float(brier_score_loss(y,p)),
            'LogLoss':float(log_loss(y,p,labels=[0,1])),
            'hits_at_10':int(y[ix].sum()),
            'precision_at_10':float(y[ix].sum()/len(ix)),
            'recall_at_10':float(y[ix].sum()/y.sum()) if y.sum() else np.nan}

def evaluate(df,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    assert df.groupby('seed').size().eq(300).all(), 'Must have all 300 customers per seed'
    assert not df.duplicated(['seed','customer_id']).any()
    pcols=[c for c in df if c.startswith('p_')]
    assert not df[pcols].isna().any().any()
    unknown=df.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
    rows=[]; selected=[]
    for seed,d in df.groupby('seed',sort=True):
        u=d.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
        y=d.label.to_numpy(dtype=int)
        for model, col in [('Baseline_F2','p_Baseline_F2')]:
            for subset,sm in [('all',np.ones(len(y),bool)),('unknown',u),('nonunknown',~u)]:
                rows.append({'seed':int(seed),'model':model,'subset':subset,**metrics(y[sm],d[col].to_numpy()[sm])})
        for col in N8_KEYS:
            if col not in d:raise ValueError('Missing '+col)
        pbase=d.p_Baseline_F2.to_numpy(dtype=float)
        p40=d.p_N8_ring_top40.to_numpy(dtype=float)
        pavg=d[list(N8_KEYS)].mean(axis=1).to_numpy(dtype=float)
        candidates={'N12_ring25_40_60_avg':pavg,
                    'N11_F2features_plus_ring40':d['p_N11_F2features_plus_ring40'].to_numpy(dtype=float) if 'p_N11_F2features_plus_ring40' in d else None}
        for a in PERCENT:
            candidates[f'N10_F2_{100-a}_N8_{a}']=(1-a/100)*pbase+(a/100)*p40
        for name,newunknown in candidates.items():
            if newunknown is None:continue
            p=pbase.copy();p[u]=newunknown[u]
            for subset,sm in [('all',np.ones(len(y),bool)),('unknown',u),('nonunknown',~u)]:
                rows.append({'seed':int(seed),'model':name,'subset':subset,**metrics(y[sm],p[sm])})
            selected.extend({'seed':int(seed),'customer_id':cid,'poutcome':hist,'label':int(yy),
                             'model':name,'probability':float(pp),'rank':int(rr)}
                            for rr,(cid,hist,yy,pp) in enumerate(sorted(zip(d.customer_id,d.poutcome,y,p),key=lambda z:-z[3])[:10],1))
    by_seed=pd.DataFrame(rows)
    summary=by_seed.groupby(['model','subset'],as_index=False).agg(
        PR_AUC_mean=('PR_AUC','mean'),ROC_AUC_mean=('ROC_AUC','mean'),
        Brier_mean=('Brier','mean'),LogLoss_mean=('LogLoss','mean'),
        hits_mean=('hits_at_10','mean'),hits_min=('hits_at_10','min'),
        PR_AUC_sd=('PR_AUC','std'))
    by_seed.to_csv(out/'by_seed.csv',index=False)
    summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(selected).to_csv(out/'top10.csv',index=False)
    df.to_csv(out/'source_oof.csv',index=False)
    print('\n=== All 300, sorted by PR-AUC (descriptive) ===')
    print(summary[summary.subset=='all'].sort_values('PR_AUC_mean',ascending=False).to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\n=== Unknown subgroup ===')
    print(summary[summary.subset=='unknown'].sort_values('PR_AUC_mean',ascending=False).to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nSaved:',out.resolve())
    return summary

def from_cached(path):
    d=pd.read_csv(path)
    required={'seed','customer_id','poutcome','label','p_Baseline_F2'}|set(N8_KEYS)
    if required-set(d):raise ValueError(f'Missing CSV columns: {sorted(required-set(d))}')
    return d

def retrain(train,folds,seeds):
    import atomiq_stage1_AB_optimization as base
    import atomiq_stage1_15q_plus_6q_feature_fusion as f2
    import atomiq_unknown_14q_N1_N3 as n1
    import atomiq_unknown_14q_N4_N6 as n4
    import atomiq_unknown_14q_N7_N9 as n7
    data=pd.read_csv(train,sep='\t')
    if len(data)!=300:raise ValueError(f'Expected 300 rows, got {len(data)}')
    if data.customer_id.duplicated().any():raise ValueError('Duplicate customer_id')
    y=data.y.astype(str).str.lower().eq('yes').astype(int).to_numpy()
    u=data.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
    strata=2*y+u.astype(int)
    if np.bincount(strata).min()<folds:raise ValueError('Not enough cases in strata')
    cols=['p_Baseline_F2',*N8_KEYS,'p_N11_F2features_plus_ring40']
    allrows=[]
    for seed in seeds:
        preds={k:np.full(300,np.nan) for k in cols}
        folds_assign=np.zeros(300,int)
        cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
        for fold,(itr,iva) in enumerate(cv.split(data,strata),1):
            print(f'seed={seed}, fold={fold}/{folds}: building F2 and ring features',flush=True)
            tr=data.iloc[itr].reset_index(drop=True);va=data.iloc[iva].reset_index(drop=True)
            yu=y[itr]; utrain=tr.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
            uvalid=va.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
            ut=tr.loc[utrain].reset_index(drop=True);uv=va.loc[uvalid].reset_index(drop=True)
            uy=yu[utrain]
            q15tr,q15va=f2.make_quantum_features(tr,va,base.ORDER,True)
            q6tr,q6va=f2.make_quantum_features(tr,va,f2.Q6_COLS,False)
            pbase=f2.fit_predict(tr,yu,va,q15tr,q15va,q6tr,q6va,variant=2)
            preds['p_Baseline_F2'][iva]=pbase
            encoder=base.FoldAngleEncoder(n1.COLUMNS).fit(ut)
            a=np.vstack((encoder.transform(ut),encoder.transform(uv)))
            q=n7.compute_ring(a,0.7,1)
            qt,qv=q[:len(ut)],q[len(ut):]
            qclass=n1.classical_preprocessor()
            ct=np.asarray(qclass.fit_transform(ut[n1.COLUMNS]),dtype=float)
            cvmat=np.asarray(qclass.transform(uv[n1.COLUMNS]),dtype=float)
            for k in [25,40,60]:
                sel=n1.choose_pairs_train_only(qt,uy,k)
                xt=np.c_[qt[:,:14],qt[:,14:][:,sel]]
                xv=np.c_[qv[:,:14],qv[:,14:][:,sel]]
                pn=n4.fit_probs(xt,uy,xv,seed)
                name=f'p_N8_ring_top{k}'
                pred=pbase.copy();pred[uvalid]=pn
                preds[name][iva]=pred
                if k==40:
                    # N11: F2-style features + ring40 in ONE unknown-only classifier.
                    # All pair screening (including q15) uses unknown TRAIN labels only.
                    q15ut=q15tr[utrain];q15uv=q15va[uvalid]
                    q6ut=q6tr[utrain];q6uv=q6va[uvalid]
                    sel15=f2.choose_pairs(q15ut,uy)
                    x15tr=np.c_[q15ut[:,:15],q15ut[:,15:][:,sel15],q6ut[:,6:]]
                    x15va=np.c_[q15uv[:,:15],q15uv[:,15:][:,sel15],q6uv[:,6:]]
                    xtr=np.c_[ct,x15tr,xt];xva=np.c_[cvmat,x15va,xv]
                    pn11=n4.fit_probs(xtr,uy,xva,seed)
                    pn=pbase.copy();pn[uvalid]=pn11
                    preds['p_N11_F2features_plus_ring40'][iva]=pn
            folds_assign[iva]=fold
        if not all(np.isfinite(v).all() for v in preds.values()):raise RuntimeError('Missing OOF predictions')
        for i, cid in enumerate(data.customer_id):
            allrows.append({'seed':seed,'customer_id':cid,'poutcome':data.poutcome.iloc[i],
                            'label':int(y[i]),'fold':int(folds_assign[i]),**{name:float(v[i]) for name,v in preds.items()}})
    return pd.DataFrame(allrows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['cached','retrain'],default='cached')
    ap.add_argument('--oof',default='all_oof(3).csv')
    ap.add_argument('--train',default='Stage1_Train.tsv')
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    ap.add_argument('--outdir',default='N10_N12_history_results')
    args=ap.parse_args()
    d=from_cached(args.oof) if args.mode=='cached' else retrain(args.train,args.folds,args.seeds)
    evaluate(d,args.outdir)
if __name__=='__main__':main()
