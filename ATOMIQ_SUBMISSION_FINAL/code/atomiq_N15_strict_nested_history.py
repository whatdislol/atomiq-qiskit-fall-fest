#!/usr/bin/env python3
"""Strict nested CV for routed Stage-1 predictor, using fresh F2/N8/LR fits.

Outer 5-fold x seeds: untouched evaluation. Inner stratified 3-fold: choose:
  alpha: N8 weight in unknown = (1-alpha)*F2 + alpha*N8 Ring40
  w:     friend's LR weight in non-unknown = (1-w)*F2 + w*LR
  gamma: ONE shared coefficient for shrunk history log-odds shifts in
         success/failure/other, determined from TRAINING labels only.

Group prior shrinkage uses pseudo=40 prior observations toward training prevalence.
No separate per-history fitted gamma parameters. Zero is included in the gamma
search, so the correction can be rejected by nested CV.

Important: Inner F2, N8, LR *must* be refitted on the actual inner-train rows.
Using existing 'source_oof(1).csv' as inner predictions would leak information.
Prediction/calibration model selection NEVER uses the outer held-out labels.

Usage:
 python atomiq_N15_strict_nested_history.py --train Stage1_Train.tsv --outer-folds 5 --inner-folds 3 --seeds 42 179 316
 python atomiq_N15_strict_nested_history.py --self-test

Requires adjacent scripts from this chat:
 atomiq_stage1_AB_optimization.py
 atomiq_stage1_15q_plus_6q_feature_fusion.py
 atomiq_unknown_14q_N1_N3.py
 atomiq_unknown_14q_N4_N6.py
 atomiq_unknown_14q_N7_N9.py
 atomiq_N14_history_shared_F2_friend.py
Needs Qiskit only in non-self-test mode. High compute demand: 15 outer fits
and 45 inner fits of two quantum maps. Try --seeds 42 first.
"""
from __future__ import annotations
import argparse
import itertools
import tempfile
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

import atomiq_stage1_AB_optimization as base
import atomiq_unknown_14q_N1_N3 as n1
import atomiq_unknown_14q_N4_N6 as n4
import atomiq_unknown_14q_N7_N9 as n7
import atomiq_N14_history_shared_F2_friend as n14

EPS=1e-8

def strata_for(frame, y, folds):
    is_unknown=frame.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
    strat=2*np.asarray(y,dtype=int)+is_unknown.astype(int)
    if np.bincount(strat,minlength=4).min() >= folds:
        return strat
    if np.bincount(np.asarray(y,dtype=int),minlength=2).min() >= folds:
        warnings.warn('Insufficient history/y cells: stratifying by label only')
        return np.asarray(y,dtype=int)
    raise ValueError('Not enough positive/negative cases for requested folds')

def fit_sources(train, valid, ytrain, seed, fake=False, friend_tune=True):
    """Fresh fits; never uses any labels belonging to valid."""
    train=train.reset_index(drop=True)
    valid=valid.reset_index(drop=True)
    ytrain=np.asarray(ytrain,dtype=int)
    p_f2=np.asarray(n7.make_f2_outer(train,valid,ytrain,seed,fake=fake),float)
    p_friend,chosen,inner_loss=n14.train_friend_lr(train,ytrain,valid,inner_folds=3 if fake else 5)
    unknown_train=train.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
    unknown_valid=valid.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
    p_ring=p_f2.copy()
    if unknown_valid.any():
        t=train.loc[unknown_train].reset_index(drop=True)
        v=valid.loc[unknown_valid].reset_index(drop=True)
        yy=ytrain[unknown_train]
        if len(np.unique(yy))<2:
            raise ValueError('Not enough unknown positive cases to fit ring')
        encoder=base.FoldAngleEncoder(n1.COLUMNS).fit(t)
        angles=np.vstack([encoder.transform(t),encoder.transform(v)])
        quantum=(n7.fake_ring(angles,.7,1) if fake else n7.compute_ring(angles,.7,1))
        qt,qv=quantum[:len(t)],quantum[len(t):]
        indices=n1.choose_pairs_train_only(qt,yy,40)
        xt=np.c_[qt[:,:14],qt[:,14:][:,indices]]
        xv=np.c_[qv[:,:14],qv[:,14:][:,indices]]
        p_ring[unknown_valid]=n4.fit_probs(xt,yy,xv,int(seed))
    delta,history_params=n14.history_offsets(train,ytrain,valid,pseudo=40.0)
    return {'f2':p_f2,'ring':p_ring,'friend':np.asarray(p_friend,float),
            'delta':np.asarray(delta,float),'friend_choice':chosen,
            'friend_inner_logloss':inner_loss,'history_params':history_params}

def make_candidate(source, history, alpha, w, gamma):
    u=np.asarray(history).astype(str)=='unknown'
    p_f2=np.asarray(source['f2'],float)
    p=p_f2.copy()
    p[u]=(1-alpha)*p_f2[u]+alpha*np.asarray(source['ring'])[u]
    p_non=(1-w)*p_f2+w*np.asarray(source['friend'],float)
    p[~u]=n14.sigmoid(n14.logit(p_non[~u])+gamma*np.asarray(source['delta'])[~u])
    return np.clip(p,EPS,1-EPS)

def score(y,p):
    y=np.asarray(y,int);p=np.clip(np.asarray(p,float),EPS,1-EPS)
    k=min(10,len(y));sel=np.argsort(-p,kind='stable')[:k]
    return {'n':len(y),'positives':int(y.sum()),
            'PR_AUC':float(average_precision_score(y,p)) if y.sum() else np.nan,
            'ROC_AUC':float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan,
            'Brier':float(brier_score_loss(y,p)),
            'LogLoss':float(log_loss(y,p,labels=[0,1])),
            'hits_at_10':int(y[sel].sum()),
            'precision_at_10':float(y[sel].sum()/k),
            'recall_at_10':float(y[sel].sum()/y.sum()) if y.sum() else np.nan}

def run(train_path, output_dir, outer_folds, inner_folds, seeds, alpha_grid,
        friend_grid, gamma_grid, fake=False, verbose=True):
    d=pd.read_csv(train_path,sep='\t')
    if d.customer_id.duplicated().any():raise ValueError('Duplicate customer IDs')
    if not {'poutcome','y','customer_id'}.issubset(d):raise ValueError('Missing required train columns')
    y=d.y.astype(str).str.strip().str.lower().eq('yes').to_numpy(dtype=int)
    out=Path(output_dir);out.mkdir(exist_ok=True,parents=True)
    all_rows=[];chosen_rows=[];inner_rows=[];fit_rows=[]
    grid=list(itertools.product(alpha_grid,friend_grid,gamma_grid))
    for seed in seeds:
        cv=StratifiedKFold(n_splits=outer_folds,shuffle=True,random_state=seed)
        fold_id=np.full(len(d),-1,int)
        p_selected=np.full(len(d),np.nan)
        p_baseline=np.full(len(d),np.nan)
        p_f2=np.full(len(d),np.nan)
        p_n10=np.full(len(d),np.nan)
        split_strata=strata_for(d,y,outer_folds)
        for fold,(outer_tr,outer_va) in enumerate(cv.split(d,split_strata),1):
            tr=d.iloc[outer_tr].reset_index(drop=True)
            va=d.iloc[outer_va].reset_index(drop=True)
            yt=y[outer_tr]
            yi=strata_for(tr,yt,inner_folds)
            cv_inner=StratifiedKFold(n_splits=inner_folds,shuffle=True,random_state=seed+1000+fold)
            source_inner={x:np.full(len(tr),np.nan) for x in ('f2','ring','friend','delta')}
            print(f'seed={seed} outer={fold}/{outer_folds}: fitting {inner_folds} inner folds',flush=True)
            for ifold,(itr,iva) in enumerate(cv_inner.split(tr,yi),1):
                src=fit_sources(tr.iloc[itr],tr.iloc[iva],yt[itr],seed+fold*101+ifold,fake=fake)
                for key in source_inner: source_inner[key][iva]=src[key]
                fit_rows.append({'seed':seed,'outer_fold':fold,'inner_fold':ifold,
                                 'stage':'inner','friend_best_k':src['friend_choice'][0],
                                 'friend_best_C':src['friend_choice'][1]})
            if not all(np.isfinite(v).all() for v in source_inner.values()):
                raise RuntimeError('Inner OOF incomplete')
            h=tr.poutcome.astype(str).str.lower().to_numpy()
            # Choose hyperparameters strictly from inner validation predictions.
            candidate_scores=[]
            for alpha,w,gamma in grid:
                p=make_candidate(source_inner,h,alpha,w,gamma)
                m=score(yt,p)
                objective=m['Brier']-0.03*m['PR_AUC']
                candidate_scores.append((objective,m['Brier'],-m['PR_AUC'],
                                         alpha,w,gamma,m))
            best=min(candidate_scores,key=lambda x:(x[0],x[1],x[2],x[3],x[4],x[5]))
            _,_,_,alpha,w,gamma,_=best
            chosen_rows.append({'seed':seed,'outer_fold':fold,'chosen_alpha':alpha,
                                'chosen_friend_weight':w,'chosen_gamma':gamma,
                                'inner_objective':best[0],
                                'inner_PR_AUC':best[6]['PR_AUC'],
                                'inner_Brier':best[6]['Brier']})
            for objective,brier,negap,a,ww,g,m in candidate_scores:
                inner_rows.append({'seed':seed,'outer_fold':fold,'alpha':a,
                    'friend_weight':ww,'gamma':g,'objective':objective,
                    'PR_AUC':m['PR_AUC'],'Brier':m['Brier']})
            # Absolutely untouched outer validation: retrain every constituent.
            print(f'  selected alpha={alpha:g}, LR={w:g}, gamma={gamma:g}; refit outer train',flush=True)
            fresh=fit_sources(tr,va,yt,seed+fold*10001,fake=fake)
            hv=va.poutcome.astype(str).str.lower().to_numpy()
            selected=make_candidate(fresh,hv,alpha,w,gamma)
            baseline=make_candidate(fresh,hv,0.5,0.3,0.0)
            n10=make_candidate(fresh,hv,0.5,0.0,0.0)
            p_selected[outer_va]=selected
            p_baseline[outer_va]=baseline
            p_f2[outer_va]=fresh['f2']
            p_n10[outer_va]=n10
            fold_id[outer_va]=fold
            fit_rows.append({'seed':seed,'outer_fold':fold,'stage':'outer',
                             'friend_best_k':fresh['friend_choice'][0],
                             'friend_best_C':fresh['friend_choice'][1]})
        assert np.isfinite(p_selected).all() and (fold_id>0).all()
        for i,cid in enumerate(d.customer_id):
            all_rows.append({'seed':seed,'fold':fold_id[i],'customer_id':cid,
                'poutcome':d.poutcome.iloc[i],'label':y[i],
                'p_nested_selected':p_selected[i],
                'p_fixed_no_history':p_baseline[i],
                'p_N10_F2_only':p_n10[i],'p_F2_only':p_f2[i]})
    allp=pd.DataFrame(all_rows)
    results=[]
    for seed,sd in allp.groupby('seed'):
        unknown=sd.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
        for label,col in [('Nested_selected','p_nested_selected'),
                          ('Fixed_N10_F2_70_Friend_30','p_fixed_no_history'),
                          ('N10_F2_nonunknown','p_N10_F2_only'),
                          ('F2_all','p_F2_only')]:
            yp=sd.label.to_numpy(int); pp=sd[col].to_numpy(float)
            for group,mask in [('all',np.ones(len(sd),bool)),('unknown',unknown),
                               ('nonunknown',~unknown)]:
                results.append({'seed':seed,'model':label,'subset':group,
                                **score(yp[mask],pp[mask])})
    m=pd.DataFrame(results)
    ag=m.groupby(['model','subset'],as_index=False).agg(
        PR_AUC_mean=('PR_AUC','mean'),PR_AUC_sd=('PR_AUC','std'),
        ROC_AUC_mean=('ROC_AUC','mean'),Brier_mean=('Brier','mean'),
        LogLoss_mean=('LogLoss','mean'),hits_mean=('hits_at_10','mean'),
        hits_min=('hits_at_10','min'))
    allp.to_csv(out/'nested_oof.csv',index=False)
    pd.DataFrame(chosen_rows).to_csv(out/'chosen_by_outer_fold.csv',index=False)
    pd.DataFrame(inner_rows).to_csv(out/'inner_grid_scores.csv',index=False)
    pd.DataFrame(fit_rows).to_csv(out/'friend_refits.csv',index=False)
    m.to_csv(out/'by_seed.csv',index=False)
    ag.to_csv(out/'summary.csv',index=False)
    if verbose:
        for subset in ('all','unknown','nonunknown'):
            print('\n'+subset+' RESULTS:')
            print(ag[ag.subset.eq(subset)].sort_values('PR_AUC_mean',ascending=False).to_string(index=False))
    return ag

def self_test():
    base_dir=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        ag=run(base_dir/'Stage1_Train.tsv',td,3,2,[42],
               [0,.5,1],[0,.3],[0,.35],fake=True,verbose=False)
        assert set(ag.subset)=={'all','unknown','nonunknown'}
        assert np.isfinite(ag.Brier_mean).all()
        assert len(pd.read_csv(Path(td)/'nested_oof.csv'))==300
    print('SELF-TEST PASSED (simulated features only; not real Qiskit results)')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',default='Stage1_Train.tsv')
    p.add_argument('--outdir',default='N15_nested_results')
    p.add_argument('--outer-folds',type=int,default=5)
    p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,179,316])
    p.add_argument('--alpha-grid',nargs='+',type=float,default=[0,.4,.5,.6])
    p.add_argument('--friend-grid',nargs='+',type=float,default=[0,.15,.3,.4])
    p.add_argument('--gamma-grid',nargs='+',type=float,default=[0,.2,.35])
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test()
    else:run(args.train,args.outdir,args.outer_folds,args.inner_folds,args.seeds,
             args.alpha_grid,args.friend_grid,args.gamma_grid)
