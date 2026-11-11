#!/usr/bin/env python3
"""N16 vs N17 vs Friend Hybrid: aligned, leakage-controlled Nested CV.

Models (all have non-unknown = 70% F2 + 30% friend's LR Compact,
plus one shared shrinkage history log-odds coefficient gamma):
 N16: unknown = 50% F2 + 50% N8 Ring40
 N17: unknown = 85% N16_unknown + 15% friend's LR Compact
 HYBRID: unknown = 85% N16_unknown + 15% friend's original Hybrid Ensemble
           (mean of LR Compact, QuantumKernelKLR, QuantumVQC).

For every inner/outer split all model components are RE-FITTED on training rows.
Gamma is chosen separately for EACH candidate using ONLY inner-OOF predictions.
Also reports inner-selected model (N16/N17/HYBRID) evaluated on held-out outer folds.

Dependencies: all scripts required by atomiq_N15_strict_nested_history.py
and friend's cloned repo at --friend-repo with src/qcampaign/...
Use --self-test for a QUICK synthetic-backend pipeline test (no Qiskit).

This compares genuine friend Hybrid only when the original repository is available.
Previously saved averaged OOF predictions must NOT be used for this experiment.

Example:
 git clone https://github.com/whatdislol/atomiq-qiskit-fall-fest.git friend_atomiq
 python -m pip install -r friend_atomiq/requirements.txt
 python atomiq_N17_same_nested_three_models.py --train Stage1_Train.tsv \\
   --friend-repo friend_atomiq --outer-folds 5 --inner-folds 3 --seeds 42 179 316
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import atomiq_N15_strict_nested_history as n15
import atomiq_N16_fixed_N10_nested_gamma as n16

CANDIDATES=('N16','N17','N17_FriendHybrid')
GAMMAS=(0.0,0.1,0.2,0.35,0.5)
EPS=1e-8

def load_friend_classes(root):
    """Use the author's original models; no joblib loading."""
    root=Path(root).expanduser().resolve()
    pkg=root/'src'/'qcampaign'
    if not (pkg/'models'/'quantum.py').is_file():
        raise FileNotFoundError(f'Original friend repo not found at {pkg}. Clone it first.')
    sys.path.insert(0,str(root/'src'))
    from qcampaign.models.classical import LRCompact
    from qcampaign.models.quantum import QuantumKernelKLR,QuantumVQC,HybridEnsemble
    from qcampaign.data import FEATURES
    return LRCompact,QuantumKernelKLR,QuantumVQC,HybridEnsemble,FEATURES

def fake_hybrid(train,valid,y,seed):
    """Deterministic label-independent synthetic features + train-only fit, for tests."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cols=['age','balance','campaign','previous','day']
    def fm(d):
        a=d[cols].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(float)
        a[:,1]=np.sign(a[:,1])*np.log1p(np.abs(a[:,1]))
        a[:,2]=np.log1p(np.maximum(a[:,2],0))
        return a
    sc=StandardScaler().fit(fm(train))
    model=LogisticRegression(C=.1,max_iter=300).fit(sc.transform(fm(train)),y)
    p=model.predict_proba(sc.transform(fm(valid)))[:,1]
    return p,p,{'fake':True}

def fit_all(train,valid,y,seed,friend_classes=None,fake=False):
    """Recompute all constituent models on ONLY this training split."""
    tr=train.reset_index(drop=True)
    va=valid.reset_index(drop=True)
    y=np.asarray(y,dtype=int)
    src=n15.fit_sources(tr,va,y,seed,fake=fake)
    if fake:
        lr,qk,params=fake_hybrid(tr,va,y,seed)
        # Mimic a distinct ensemble while preserving split isolation.
        qk=np.clip(.75*lr+.25*float(y.mean()),EPS,1-EPS)
        vqc=np.clip(.6*lr+.4*float(y.mean()),EPS,1-EPS)
        hy=(lr+qk+vqc)/3.
        src['friend']=lr
        src['hybrid']=hy
        src['friend_hybrid_params']=params
    else:
        if friend_classes is None:
            raise ValueError('Supply --friend-repo for the actual Friend Hybrid')
        LR,QK,VQC,Hybrid,feature_cols=friend_classes
        Xtr=tr[list(feature_cols)].copy(); Xva=va[list(feature_cols)].copy()
        # Exact original pipelines: independently tune LR/QK/VQC on split training.
        fitted={}
        for name,cls in [('lr_compact',LR),('qk_klr',QK),('vqc',VQC)]:
            print(f'      training friend component {name}',flush=True)
            fitted[name]=cls().fit(Xtr,y)
        src['friend']=fitted['lr_compact'].predict_proba(Xva)[:,1]
        src['hybrid']=Hybrid(fitted).predict_proba(Xva)[:,1]
        src['friend_hybrid_params']={k:str(m.best_params_) for k,m in fitted.items()}
    for col in ['f2','ring','friend','hybrid','delta']:
        a=np.asarray(src[col],float)
        if len(a)!=len(va) or not np.isfinite(a).all():
            raise RuntimeError(f'Invalid {col} in fit_all')
    return src

def predict(src,history,model,gamma):
    h=pd.Series(history).astype(str).str.strip().str.lower().to_numpy()
    u=(h=='unknown')
    f2=np.asarray(src['f2'],float)
    n8=np.asarray(src['ring'],float)
    lr=np.asarray(src['friend'],float)
    hy=np.asarray(src['hybrid'],float)
    delta=np.asarray(src['delta'],float)
    un=.5*f2+.5*n8
    if model=='N17': un=.85*un+.15*lr
    elif model=='N17_FriendHybrid':un=.85*un+.15*hy
    elif model!='N16':raise ValueError(f'Unknown candidate: {model}')
    nonbase=.7*f2+.3*lr
    non=n15.n14.sigmoid(n15.n14.logit(nonbase)+gamma*delta)
    out=np.where(u,un,non)
    return np.clip(out,EPS,1-EPS)

def eval_metrics(y,p):
    return n16.metrics(y,p)

def obj(y,p,rule):
    m=eval_metrics(y,p)
    if rule=='brier':return m['Brier']
    if rule=='brier_ap':return m['Brier']-0.03*m['PR_AUC']
    if rule=='logloss':return m['LogLoss']
    raise ValueError(rule)

def run(trainfile,outdir,outer_folds=5,inner_folds=3,seeds=(42,179,316),
        gammas=GAMMAS,objective='brier_ap',friend_repo=None,fake=False):
    df=pd.read_csv(trainfile,sep='\t')
    must={'customer_id','poutcome','y'}
    if not must.issubset(df.columns):raise ValueError(f'Missing columns: {must-set(df.columns)}')
    if not df.customer_id.is_unique:raise ValueError('Duplicate customer IDs')
    y=df.y.astype(str).str.strip().str.lower().eq('yes').to_numpy(int)
    if y.sum()<min(outer_folds,inner_folds):raise ValueError('Not enough positives')
    friend_classes=None if fake else load_friend_classes(friend_repo)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    labels=(*CANDIDATES,'INNER_SELECTED')
    rows=[];inner_rows=[];choices=[];fitrows=[]
    for seed in seeds:
        cv=StratifiedKFold(outer_folds,shuffle=True,random_state=seed)
        strata=n15.strata_for(df,y,outer_folds)
        allp={m:np.full(len(df),np.nan) for m in labels}
        folds=np.full(len(df),-1,int)
        for fold,(it,iv) in enumerate(cv.split(df,strata),1):
            tr=df.iloc[it].reset_index(drop=True)
            va=df.iloc[iv].reset_index(drop=True)
            yt=y[it]
            cv_inner=StratifiedKFold(inner_folds,shuffle=True,random_state=seed+1000+fold)
            inner_strata=n15.strata_for(tr,yt,inner_folds)
            inner_src={c:np.full(len(tr),np.nan) for c in ('f2','ring','friend','hybrid','delta')}
            print(f'[seed={seed} outer={fold}/{outer_folds}] Inner CV',flush=True)
            for j,(a,b) in enumerate(cv_inner.split(tr,inner_strata),1):
                src=fit_all(tr.iloc[a],tr.iloc[b],yt[a],seed+fold*101+j,friend_classes,fake)
                for c in inner_src:inner_src[c][b]=src[c]
                fitrows.append({'seed':seed,'outer_fold':fold,'inner_fold':j,'stage':'inner',
                                'friend_params':str(src.get('friend_hybrid_params'))})
            if any(not np.isfinite(a).all() for a in inner_src.values()):
                raise RuntimeError('Incomplete inner OOF')
            bests={}
            for model in CANDIDATES:
                for gamma in gammas:
                    p=predict(inner_src,tr.poutcome,model,gamma)
                    val=obj(yt,p,objective);ms=eval_metrics(yt,p)
                    inner_rows.append({'seed':seed,'outer_fold':fold,'candidate':model,
                                       'gamma':gamma,'objective':val,**ms})
                    if model not in bests or (val,gamma)<(bests[model][0],bests[model][1]):
                        bests[model]=(val,gamma)
            # Model selection is inside outer-training only.
            winner=min(CANDIDATES,key=lambda m:(bests[m][0],CANDIDATES.index(m)))
            choices.append({'seed':seed,'outer_fold':fold,'selected_model':winner,
                            'selected_gamma':bests[winner][1],
                            **{f'gamma_{m}':bests[m][1] for m in CANDIDATES},
                            **{f'objective_{m}':bests[m][0] for m in CANDIDATES}})
            print(f'  selected {winner}, gamma={bests[winner][1]}; outer refit',flush=True)
            fresh=fit_all(tr,va,yt,seed+fold*10001,friend_classes,fake)
            fitrows.append({'seed':seed,'outer_fold':fold,'inner_fold':-1,'stage':'outer',
                            'friend_params':str(fresh.get('friend_hybrid_params'))})
            for model in CANDIDATES:
                allp[model][iv]=predict(fresh,va.poutcome,model,bests[model][1])
            allp['INNER_SELECTED'][iv]=allp[winner][iv]
            folds[iv]=fold
        if any(not np.isfinite(v).all() for v in allp.values()):raise RuntimeError('Incomplete outer OOF')
        for i in range(len(df)):
            rows.append({'seed':seed,'fold':folds[i],'customer_id':df.customer_id.iloc[i],
                         'poutcome':df.poutcome.iloc[i],'label':int(y[i]),
                         **{f'p_{m}':float(allp[m][i]) for m in labels}})
    oof=pd.DataFrame(rows);oof.to_csv(out/'nested_oof.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'chosen_by_outer_fold.csv',index=False)
    pd.DataFrame(inner_rows).to_csv(out/'inner_grid_scores.csv',index=False)
    pd.DataFrame(fitrows).to_csv(out/'friend_refits.csv',index=False)
    byseed=[];tops=[]
    for seed,sd in oof.groupby('seed',sort=True):
        h=sd.poutcome.astype(str).str.lower().to_numpy()
        yy=sd.label.to_numpy(int)
        for model in labels:
            p=sd[f'p_{model}'].to_numpy(float)
            for subset,sel in [('all',np.ones(len(sd),bool)),('unknown',h=='unknown'),('nonunknown',h!='unknown')]:
                byseed.append({'seed':seed,'model':model,'subset':subset,**eval_metrics(yy[sel],p[sel])})
            order=np.lexsort((sd.customer_id.to_numpy(str),-p))[:10]
            for rank,idx in enumerate(order,1):
                tops.append({'seed':seed,'model':model,'rank':rank,'customer_id':sd.customer_id.iloc[idx],
                             'probability':p[idx],'label':yy[idx],'poutcome':h[idx]})
    b=pd.DataFrame(byseed);b.to_csv(out/'by_seed.csv',index=False)
    pd.DataFrame(tops).to_csv(out/'top10.csv',index=False)
    summ=b.groupby(['model','subset'],as_index=False).agg(
        PR_AUC_mean=('PR_AUC','mean'),PR_AUC_sd=('PR_AUC','std'),
        ROC_AUC_mean=('ROC_AUC','mean'),Brier_mean=('Brier','mean'),
        LogLoss_mean=('LogLoss','mean'),hits_mean=('hits_at_10','mean'),
        hits_min=('hits_at_10','min'),hits_max=('hits_at_10','max'))
    summ.to_csv(out/'summary.csv',index=False)
    print('\nAll customers:')
    print(summ.query("subset=='all'").sort_values('PR_AUC_mean',ascending=False).to_string(index=False))
    print('\nSaved to:',out.resolve())
    return summ

def self_test(train):
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        result=run(train,temp,outer_folds=3,inner_folds=2,seeds=(42,),
                   gammas=(0,.2),fake=True)
        oof=pd.read_csv(Path(temp)/'nested_oof.csv')
        assert len(oof)==300
        assert oof.customer_id.nunique()==300
        assert set(result.model)==set((*CANDIDATES,'INNER_SELECTED'))
        assert set(result.subset)=={'all','unknown','nonunknown'}
        assert result.Brier_mean.notna().all()
    print('SELF-TEST OK (synthetic model features; NOT scientific results).')

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--train',default='Stage1_Train.tsv')
    a.add_argument('--friend-repo',default='friend_atomiq',help='Cloned original repository path')
    a.add_argument('--outdir',default='N17_same_nested_results')
    a.add_argument('--outer-folds',type=int,default=5)
    a.add_argument('--inner-folds',type=int,default=3)
    a.add_argument('--seeds',type=int,nargs='+',default=[42,179,316])
    a.add_argument('--gamma-grid',type=float,nargs='+',default=list(GAMMAS))
    a.add_argument('--objective',choices=['brier_ap','brier','logloss'],default='brier_ap')
    a.add_argument('--self-test',action='store_true')
    x=a.parse_args()
    if x.self_test:self_test(x.train)
    else:run(x.train,x.outdir,x.outer_folds,x.inner_folds,x.seeds,x.gamma_grid,
             x.objective,x.friend_repo)

if __name__=='__main__':main()
