#!/usr/bin/env python3
"""Stage 2 end-to-end: exact combinatorial optimum and exact 6-of-10 subspace QAOA.
The shot results are NUMERICAL simulations, not measurements on quantum hardware.
Dependencies: numpy, pandas, scipy. Same customer ordering as Stage 1 probability rank.
"""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def native(v):
    if isinstance(v,dict):return {str(k):native(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [native(x) for x in v]
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.floating):return float(v)
    if isinstance(v,np.ndarray):return native(v.tolist())
    return v

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--stage1',default='stage1_result.csv')
    pa.add_argument('--data',default='stage2_customer_inputs_complete.csv')
    pa.add_argument('--outdir',default='stage2_final_results')
    pa.add_argument('--depth',type=int,default=2)
    pa.add_argument('--starts',type=int,default=5)
    pa.add_argument('--maxiter',type=int,default=400)
    pa.add_argument('--shots',type=int,default=8192)
    pa.add_argument('--seed',type=int,default=42)
    pa.add_argument('--no-qaoa',action='store_true')
    args=pa.parse_args()
    if args.depth<1 or args.starts<1 or args.shots<1:raise ValueError('Positive depth, starts, shots required')
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    allp=pd.read_csv(args.stage1,dtype={'customer_id':str})
    req=['customer_id','predicted_probability'];assert all(c in allp for c in req)
    assert len(allp) in (10,100) and allp.customer_id.is_unique
    assert allp.predicted_probability.between(0,1).all()
    allp=allp.sort_values(['predicted_probability','customer_id'],ascending=[False,True]).reset_index(drop=True)
    if 'top10_flag' in allp:
        assert set(allp.loc[allp.top10_flag.astype(int).eq(1),'customer_id'])==set(allp.head(10).customer_id)
    econ=pd.read_csv(args.data,dtype={'customer_id':str})
    assert econ.customer_id.is_unique
    require=['expected_deposit_value','contact_cost','capacity_units']
    assert all(c in econ for c in require)
    df=allp.head(10)[req].merge(econ,on='customer_id',how='left',validate='one_to_one')
    if df[require].isna().any().any():raise ValueError('Stage2 economic inputs missing')
    df.insert(1,'qubit_index',np.arange(10))
    df['expected_net_value']=df.predicted_probability*df.expected_deposit_value-df.contact_cost
    n=10;k=6;budget=25;capacity=22
    val=df.expected_net_value.to_numpy(float);cost=df.contact_cost.to_numpy(float);cap=df.capacity_units.to_numpy(float)
    combinations=list(itertools.combinations(range(n),k))
    masks=np.array([sum(1<<j for j in subset) for subset in combinations],dtype=int)
    obj=np.array([val[list(subset)].sum() for subset in combinations]);co=np.array([cost[list(subset)].sum() for subset in combinations]);ca=np.array([cap[list(subset)].sum() for subset in combinations]);feas=(co<=budget)&(ca<=capacity)
    assert feas.any()
    exact_index=int(np.argmax(np.where(feas,obj,-np.inf)))
    best_indices=combinations[exact_index];exact_value=float(obj[exact_index]);bestmask=int(masks[exact_index]);bestbits=format(bestmask,'010b')
    def ids(inds):return df.iloc[list(inds)].customer_id.tolist()
    df['selected_exact']=df.qubit_index.isin(best_indices).astype(int)
    df['candidate_bit_position']=df.qubit_index
    df.to_csv(out/'stage2_candidates.csv',index=False)
    pd.DataFrame({'bitstring':[format(m,'010b') for m in masks],'customer_ids':[';'.join(ids(subset)) for subset in combinations],'objective':obj,'contact_cost':co,'capacity_units':ca,'feasible':feas}).sort_values('objective',ascending=False).to_csv(out/'all_210_subsets.csv',index=False)
    report={'candidate_ids':df.customer_id.tolist(),'selected_ids_exact':ids(best_indices),'selected_bitstring_q9_to_q0':bestbits,'objective_exact':exact_value,'budget_limit':budget,'budget_used':float(co[exact_index]),'capacity_limit':capacity,'capacity_used':float(ca[exact_index]),'subsets_tested':len(combinations),'feasible_subsets':int(feas.sum()),'max_cost_over_six':float(max(co)),'max_capacity_over_six':float(max(ca)),'qubit_order':'q0 is first row of candidates and rightmost bit of printed bitstring'}
    if not args.no_qaoa:
        # Exact simulation in six-excitation subspace, dimension C(10,6)=210.
        # Initial Dicke state = equal amplitudes for all six-of-ten bitstrings.
        # Cost unitary: RZ(-gamma*v_i) on each q_i -> exp(+i*gamma*objective(mask)) up to global phase.
        # Mixer: RXX(2*beta), then RYY(2*beta) for each edge in two even/odd sweeps.
        # On |01>,|10>, this is exp[-i*beta*(XX+YY)] -> cos(2*beta), -i*sin(2*beta).
        reverse={int(m):ix for ix,m in enumerate(masks)}
        edge_pairs=[]
        for j in range(n-1):
            starts=[];ends=[]
            for ix,m in enumerate(masks):
                if ((m>>j)&1)==0 and ((m>>(j+1))&1)==1:
                    other=int(m^(1<<j)^(1<<(j+1)))
                    starts.append(ix);ends.append(reverse[other])
            edge_pairs.append((np.asarray(starts,dtype=int),np.asarray(ends,dtype=int)))
        mixer_edges=list(range(0,n-1,2))+list(range(1,n-1,2))
        initial=np.ones(len(masks),dtype=np.complex128)/np.sqrt(len(masks))
        def state(theta):
            psi=initial.copy()
            for ell in range(args.depth):
                gamma,beta=theta[2*ell],theta[2*ell+1]
                psi*=np.exp(1j*gamma*obj)
                c=np.cos(2*beta);s=-1j*np.sin(2*beta)
                for edge in mixer_edges:
                    a,b=edge_pairs[edge]
                    ap=psi[a].copy();bp=psi[b].copy()
                    psi[a]=c*ap+s*bp;psi[b]=s*ap+c*bp
            return psi
        def energy(theta):return -float(np.dot(np.abs(state(theta))**2,obj))
        rng=np.random.default_rng(args.seed)
        attempts=[]
        for start in range(args.starts):
            theta=np.empty(2*args.depth)
            theta[0::2]=rng.uniform(-1.5,1.5,args.depth)
            theta[1::2]=rng.uniform(-.8,.8,args.depth)
            r=minimize(energy,theta,method='COBYLA',options={'maxiter':args.maxiter,'rhobeg':.25,'tol':1e-7})
            attempts.append(r)
        res=min(attempts,key=lambda x:x.fun)
        dist=np.abs(state(res.x))**2;dist/=dist.sum()
        outcomes=rng.choice(len(dist),size=args.shots,p=dist)
        counts=np.bincount(outcomes,minlength=len(dist))
        measured_idx=np.flatnonzero(counts)
        best_observed_idx=int(max(measured_idx,key=lambda ix:obj[ix] if feas[ix] else -np.inf))
        sorted_measurements=sorted(measured_idx,key=lambda ix:(-int(counts[ix]),-float(obj[ix])))
        measurement_df=pd.DataFrame([{'bitstring':format(int(masks[ix]),'010b'),'customer_ids':';'.join(ids(combinations[ix])),'shots':int(counts[ix]),'frequency':float(counts[ix]/args.shots),'objective':float(obj[ix]),'budget_used':float(co[ix]),'capacity_used':float(ca[ix]),'feasible':bool(feas[ix])} for ix in sorted_measurements])
        measurement_df.to_csv(out/'simulated_qaoa_bitstrings.csv',index=False)
        report['qaoa_numerical_simulation']={'algorithm':'Dicke-initialized cardinality-preserving XY-QAOA','simulator':'custom NumPy 210-dimensional exact subspace emulator; shots sampled classically (NOT hardware shots)','depth_p':args.depth,'restarts':args.starts,'maxiter_per_restart':args.maxiter,'seed':args.seed,'shots':args.shots,'optimized_parameters_gamma_beta_interleaved':res.x.tolist(),'optimizer_nfev_best_restart':int(res.nfev),'expected_objective_under_qaoa_distribution':float(-res.fun),'exact_optimum_probability_ideal':float(dist[exact_index]),'observed_exact_optimum_shots':int(counts[exact_index]),'best_observed_bitstring_q9_to_q0':format(int(masks[best_observed_idx]),'010b'),'best_observed_selected_ids':ids(combinations[best_observed_idx]),'best_observed_objective':float(obj[best_observed_idx]),'best_observed_cost':float(co[best_observed_idx]),'best_observed_capacity':float(ca[best_observed_idx]),'optimality_gap_best_observed':float(exact_value-obj[best_observed_idx]),'probability_mass_feasible':float(dist[feas].sum()),'number_of_distinct_bitstrings_observed':len(measured_idx),'number_of_qaoa_rz_gates_excluding_state_preparation':10*args.depth,'number_of_qaoa_two_qubit_rotations_excluding_state_preparation':18*args.depth,'transpiled_depth_and_gate_counts':'not available without Qiskit transpiler and chosen backend; Dicke preparation excluded'}
    (out/'stage2_summary.json').write_text(json.dumps(native(report),ensure_ascii=False,indent=2))
    print(json.dumps(native(report),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
