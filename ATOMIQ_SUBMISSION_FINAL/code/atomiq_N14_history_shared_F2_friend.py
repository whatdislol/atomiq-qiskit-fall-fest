#!/usr/bin/env python3
"""Atomiq Stage 1: matched-OOF N10 + regularized history offsets + friend's LR Compact.

Inputs:
  Stage1_Train.tsv and 'source_oof(1).csv' from N10/N11 retraining.

For every (seed, fold):
 - Existing F2/N8/N11 validation predictions are reused (they are outer-fold OOF).
 - Friend's LR Compact is fit on ONLY that fold's training rows; its original
   feature generation, top-k L1 selection and inner 5-fold log-loss tuning are
   recreated from the team's public code. This does NOT reuse friend's
   differently-folded historical OOF scores or unpickle the joblib file.
 - History offsets use *only outer-training labels*: a Bayesian-shrunk
   success/failure/other prior-odds correction, with explicit strength grid.

The history correction is exploratory and may DOUBLE COUNT poutcome because F2
already uses it. Compare to offsets=0 and global non-routed baselines.
All hyperparameter combination scores are descriptive model search on the same
300 labels, NOT honest nested model-selection performance estimates.

Usage:
 python atomiq_N14_history_shared_F2_friend.py --train Stage1_Train.tsv \
   --oof 'source_oof(1).csv' --outdir N14_history_results
"""
from __future__ import annotations
import argparse
import itertools
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS=1e-7

class CampaignFeatures(BaseEstimator, TransformerMixin):
    """Faithful reproduction of public qcampaign.features.CampaignFeatures."""
    def fit(self, X, y=None):
        self.month_mode_ = X['month'].mode().iloc[0]
        return self
    def transform(self, X):
        d=pd.DataFrame(index=X.index)
        age=pd.to_numeric(X.age)
        balance=pd.to_numeric(X.balance)
        d['age']=age.astype(float)
        d['age_sq']=(age-40.0)**2/100.0
        d['bal_slog']=np.sign(balance)*np.log1p(np.abs(balance))
        d['day']=pd.to_numeric(X.day).astype(float)
        d['log_campaign']=np.log(pd.to_numeric(X.campaign).clip(lower=1))
        d['prev_contacted']=(pd.to_numeric(X.pdays)>=0).astype(float)
        d['log_pdays']=np.log1p(pd.to_numeric(X.pdays).clip(lower=0))
        d['log_previous']=np.log1p(pd.to_numeric(X.previous))
        d['pout_success']=(X.poutcome=='success').astype(float)
        d['pout_failure']=(X.poutcome=='failure').astype(float)
        d['contact_unknown']=(X.contact=='unknown').astype(float)
        d['housing_yes']=(X.housing=='yes').astype(float)
        d['loan_yes']=(X.loan=='yes').astype(float)
        d['month_is_mode']=(X.month==self.month_mode_).astype(float)
        return d


def friend_pipeline(k,C):
    return Pipeline([
        ('features',CampaignFeatures()),
        ('scale',StandardScaler()),
        ('select',SelectFromModel(LogisticRegression(C=0.3,penalty='l1',solver='liblinear',max_iter=5000),
                                  max_features=k, threshold=-np.inf)),
        ('clf',LogisticRegression(C=C,max_iter=5000))
    ])


def train_friend_lr(xtr,ytr,xva,inner_folds=5):
    # Original grid k={4,6,8}, C={0.1,0.3,1}; inner 5 folds / seed 7.
    inner_folds=min(inner_folds,int(np.bincount(ytr).min()))
    splits=list(StratifiedKFold(n_splits=inner_folds,shuffle=True,random_state=7).split(xtr,ytr))
    chosen=None; best=float('inf')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k,C in itertools.product((4,6,8),(.1,.3,1.0)):
            oof=np.full(len(xtr),np.nan)
            for a,b in splits:
                model=friend_pipeline(k,C)
                model.fit(xtr.iloc[a],ytr[a])
                oof[b]=model.predict_proba(xtr.iloc[b])[:,1]
            loss=log_loss(ytr,np.clip(oof,EPS,1-EPS),labels=[0,1])
            if loss<best-1e-12:
                chosen=(k,C);best=loss
        final=friend_pipeline(*chosen)
        final.fit(xtr,ytr)
    return final.predict_proba(xva)[:,1],chosen,best


def logit(p):
    p=np.clip(p,EPS,1-EPS)
    return np.log(p/(1-p))

def sigmoid(x):
    x=np.asarray(x,float)
    return 1/(1+np.exp(-np.clip(x,-100,100)))


def history_offsets(xtr,ytr,xva,pseudo=20.0):
    """Train-only smoothed subgroup log-odds shifts for non-unknown groups.

    Shrink subgroup likelihood to overall train prevalence. This adjustment is
    not fit to held-out F2 labels, and the fit never sees the evaluation fold.
    """
    base=float(np.mean(ytr))
    groups=xtr.poutcome.astype(str).str.lower().to_numpy()
    out=np.zeros(len(xva),float)
    params={}
    for g in ('success','failure','other'):
        m=(groups==g)
        n=int(m.sum()); pos=int(ytr[m].sum())
        post=(pos+pseudo*base)/(n+pseudo)
        delta=logit(post)-logit(base)
        params[g]={'n':n,'positive':pos,'posterior':float(post),'delta':float(delta)}
        out[xva.poutcome.astype(str).str.lower().to_numpy()==g]=delta
    return out,params


def score(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    order=np.argsort(-p,kind='stable')[:min(10,len(y))]
    hits=int(y[order].sum())
    return dict(n=len(y),positives=int(y.sum()),PR_AUC=(float(average_precision_score(y,p)) if y.sum() else np.nan),
                ROC_AUC=(float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan),
                Brier=float(brier_score_loss(y,p)),LogLoss=float(log_loss(y,p,labels=[0,1])),
                hits_at_10=hits,precision_at_10=hits/len(order),
                recall_at_10=hits/int(y.sum()) if y.sum() else np.nan)


def run(train_file,oof_file,outdir,betas,lr_weights,offset_weights,pseudo):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(train_file,sep='\t')
    raw=pd.read_csv(oof_file)
    required={'seed','customer_id','fold','poutcome','label','p_Baseline_F2',
              'p_N8_ring_top40','p_N11_F2features_plus_ring40'}
    if required-set(raw.columns):
        raise ValueError(f'Missing OOF columns: {sorted(required-set(raw.columns))}')
    if train.customer_id.duplicated().any() or raw.duplicated(['seed','customer_id']).any():
        raise ValueError('Duplicate customer IDs')
    orig=train.set_index('customer_id',drop=False)
    ymap=train.set_index('customer_id').y.astype(str).str.lower().eq('yes').astype(int)
    if not set(raw.customer_id).issubset(set(train.customer_id)):
        raise ValueError('Some OOF IDs are not in train TSV')
    if not np.array_equal(raw.label.to_numpy(int),raw.customer_id.map(ymap).to_numpy(int)):
        raise ValueError('OOF labels disagree with TSV')
    if len(raw)!=len(train)*raw.seed.nunique():
        raise ValueError('Expected a complete OOF prediction for each seed')
    rows=[]; per_seed=[]; selected=[]; tuning=[]; priors=[]
    for seed,d in raw.groupby('seed',sort=True):
        d=d.copy().sort_values('customer_id').reset_index(drop=True)
        if set(d.customer_id)!=set(train.customer_id):raise ValueError('Missing IDs in seed '+str(seed))
        p_friend=np.full(len(d),np.nan); delta=np.full(len(d),np.nan)
        for fold in sorted(d.fold.unique()):
            vi=np.flatnonzero(d.fold.to_numpy()==fold)
            tr_ids=d.loc[d.fold.ne(fold),'customer_id'].to_list()
            va_ids=d.iloc[vi].customer_id.to_list()
            xtr=orig.loc[tr_ids].reset_index(drop=True)
            xva=orig.loc[va_ids].reset_index(drop=True)
            ytr=ymap.loc[tr_ids].to_numpy(int)
            p_friend[vi],(k,C),inner_loss=train_friend_lr(xtr,ytr,xva)
            delta[vi],group_params=history_offsets(xtr,ytr,xva,pseudo)
            tuning.append({'seed':seed,'fold':fold,'best_k':k,'best_C':C,'inner_logloss':inner_loss})
            for group,par in group_params.items():
                priors.append({'seed':seed,'fold':fold,'group':group,**par})
        if not np.isfinite(p_friend).all() or not np.isfinite(delta).all():
            raise RuntimeError('Incomplete fold predictions')
        base=d.p_Baseline_F2.to_numpy(float)
        n8=d.p_N8_ring_top40.to_numpy(float)
        n11=d.p_N11_F2features_plus_ring40.to_numpy(float)
        y=d.label.to_numpy(int)
        is_unknown=d.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
        for beta in betas:
            p10=.5*base+.5*n8
            p_unknown=(1-beta)*p10+beta*n11
            for lr_w in lr_weights:
                p_nonbase=(1-lr_w)*base+lr_w*p_friend
                for off_w in offset_weights:
                    p_non=sigmoid(logit(p_nonbase)+off_w*delta)
                    p=p_unknown.copy()
                    p[~is_unknown]=p_non[~is_unknown]
                    key=f'B{round(beta*100):03d}_LR{round(lr_w*100):03d}_H{round(off_w*100):03d}'
                    subset_masks={'all':np.ones(len(y),bool),'unknown':is_unknown,
                                  'nonunknown':~is_unknown}
                    for group in ('success','failure','other'):
                        subset_masks[group]=d.poutcome.eq(group).to_numpy()
                    for subset,mask in subset_masks.items():
                        rows.append({'seed':seed,'model':key,'beta':beta,'friend_lr_weight':lr_w,
                                     'history_strength':off_w,'subset':subset,**score(y[mask],p[mask])})
                    if (beta==0 and lr_w==0 and off_w==0) or (beta==0 and lr_w==.3 and off_w==0):
                        pass
                    selected.extend({'seed':seed,'model':key,'customer_id':cid,'poutcome':hist,
                                     'label':int(yy),'probability':float(pp)}
                                    for cid,hist,yy,pp in zip(d.customer_id,d.poutcome,y,p))
        # Recomputed friend LR OOF, only ONE row per seed/customer
        per_seed.extend({'seed':seed,'customer_id':cid,'label':int(yy),'poutcome':hist,'fold':fold,
                         'p_F2':float(pf),'p_Ring40':float(pr),'p_N11':float(pn),'p_friend_LR':float(pl),
                         'history_delta':float(de)}
                        for cid,yy,hist,fold,pf,pr,pn,pl,de in zip(d.customer_id,y,d.poutcome,d.fold,
                                                                   base,n8,n11,p_friend,delta))
        print(f'Finished seed {seed}',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'by_seed.csv',index=False)
    summary=df.groupby(['model','beta','friend_lr_weight','history_strength','subset'],as_index=False).agg(
        PR_AUC=('PR_AUC','mean'),ROC_AUC=('ROC_AUC','mean'),Brier=('Brier','mean'),
        LogLoss=('LogLoss','mean'),hits_mean=('hits_at_10','mean'),hits_min=('hits_at_10','min'),
        PR_AUC_sd=('PR_AUC','std'))
    summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(per_seed).to_csv(out/'source_oof_with_refit_friend.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'friend_lr_inner_tuning.csv',index=False)
    pd.DataFrame(priors).to_csv(out/'history_train_fold_priors.csv',index=False)
    s=summary[summary.subset.eq('all')].copy()
    print('\nBASELINE N10 + F2 (beta=0, friend=0, history=0):')
    print(s[np.isclose(s.beta,0)&np.isclose(s.friend_lr_weight,0)&np.isclose(s.history_strength,0)].to_string(index=False))
    print('\nTop 10 exploratory PR-AUC:')
    print(s.nlargest(10,'PR_AUC')[['model','PR_AUC','Brier','hits_mean','hits_min']].to_string(index=False))
    print('\nTop 10 exploratory Brier:')
    print(s.nsmallest(10,'Brier')[['model','PR_AUC','Brier','hits_mean','hits_min']].to_string(index=False))
    # fixed, interpretable candidates + exploratory score winners
    candidates=['B000_LR000_H000','B000_LR030_H000','B000_LR030_H015',
                'B000_LR030_H035','B020_LR030_H015','B050_LR030_H015']
    candidates+=list(s.nlargest(3,'PR_AUC').model)+list(s.nsmallest(3,'Brier').model)
    candidates+=list(s.nlargest(3,'hits_mean').model)
    candidates=list(dict.fromkeys(candidates))
    pd.DataFrame(selected).query('model in @candidates').to_csv(out/'selected_candidates_oof.csv',index=False)
    summary[summary.model.isin(candidates)].to_csv(out/'selected_candidates_summary.csv',index=False)
    print('\nSaved:',out.resolve())
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',default='Stage1_Train.tsv')
    ap.add_argument('--oof',default='source_oof(1).csv')
    ap.add_argument('--outdir',default='N14_history_results')
    ap.add_argument('--betas',nargs='+',type=float,default=[0,.2,.5])
    ap.add_argument('--lr-weights',nargs='+',type=float,default=[0,.15,.3,.4])
    ap.add_argument('--history-strengths',nargs='+',type=float,default=[0,.15,.35,.6])
    ap.add_argument('--pseudo',type=float,default=20)
    args=ap.parse_args()
    for name,vals in [('betas',args.betas),('lr weights',args.lr_weights),('history strengths',args.history_strengths)]:
        if any(v<0 or v>1 for v in vals):raise ValueError(name+' must be in [0,1]')
    if args.pseudo<=0:raise ValueError('--pseudo must be > 0')
    run(args.train,args.oof,args.outdir,args.betas,args.lr_weights,args.history_strengths,args.pseudo)

if __name__=='__main__':main()
