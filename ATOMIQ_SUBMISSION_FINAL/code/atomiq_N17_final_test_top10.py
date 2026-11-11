#!/usr/bin/env python3
"""Fit actual N17+Friend Hybrid on ALL labeled training data, predict 100 test customers.
Requires original friend repo and Qiskit dependencies, plus local N17/N15 modules.
This is not the rank-fusion submission; it creates an independent N17 output.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from atomiq_N17_same_nested_three_models import load_friend_classes, fit_all, predict

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',default='Stage1_Train.tsv')
    ap.add_argument('--test',default='Stage1_Test.tsv')
    ap.add_argument('--friend-repo',default='friend_atomiq')
    ap.add_argument('--gamma',type=float,default=0.2)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--outdir',default='N17_final_test')
    args=ap.parse_args()
    train=pd.read_csv(args.train,sep='\t')
    test=pd.read_csv(args.test,sep='\t')
    if not {'customer_id','poutcome','y'}.issubset(train):raise ValueError('Train requires customer_id, poutcome and y')
    if not {'customer_id','poutcome'}.issubset(test):raise ValueError('Test requires customer_id and poutcome')
    if len(test)!=100:raise ValueError(f'Expected 100 test rows, got {len(test)}')
    if train.customer_id.duplicated().any() or test.customer_id.duplicated().any():raise ValueError('Duplicate customer IDs')
    if set(train.customer_id)&set(test.customer_id):raise ValueError('Train/test customer ID overlap')
    y=train.y.astype(str).str.strip().str.lower().eq('yes').to_numpy(int)
    if not {0,1}.issubset(set(y)):raise ValueError('Training labels must contain yes and no')
    cls=load_friend_classes(args.friend_repo)
    src=fit_all(train,test,y,args.seed,cls,fake=False)
    probabilities=predict(src,test.poutcome,'N17_FriendHybrid',args.gamma)
    if not np.isfinite(probabilities).all() or not np.all((probabilities>=0)&(probabilities<=1)):
        raise ValueError('Invalid probabilities')
    result=pd.DataFrame({'customer_id':test.customer_id.astype(str),'predicted_probability':probabilities})
    result=result.sort_values(['predicted_probability','customer_id'],ascending=[False,True]).reset_index(drop=True)
    result['rank']=np.arange(1,len(result)+1)
    result['top10_flag']=(result['rank']<=10).astype(int)
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'stage1_N17_FriendHybrid.csv',index=False,float_format='%.10f')
    result.iloc[:10].to_csv(out/'N17_FriendHybrid_top10.csv',index=False,float_format='%.10f')
    print(result.head(10).to_string(index=False))
    print('Saved:',out.resolve())

if __name__=='__main__': main()
