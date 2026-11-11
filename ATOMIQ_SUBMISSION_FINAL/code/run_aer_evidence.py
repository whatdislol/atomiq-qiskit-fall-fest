#!/usr/bin/env python3
"""Atomiq: independently reproducible Qiskit Aer evidence for stages 2 and 3.

- Does NOT claim hardware execution.
- Uses full-precision Stage 1 predictions.
- Stage 2 QAOA angles are preoptimized by a 210-dimensional NumPy simulator.
- Checks Qiskit Statevector agrees with that exact subspace simulator before sampling.
- Saves circuit diagrams, counts, backend configuration, gate/depth and noise tests.
"""
from __future__ import annotations
import argparse, csv, itertools, json, math, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def probs_of_outcomes(probabilities):
    d = np.array([1.0])
    for p in probabilities:
        n = np.zeros(len(d)+1)
        n[:-1] += d * (1-p)
        n[1:] += d * p
        d = n
    return d


def count_gates(qc):
    r = {'total_instructions':len(qc.data), 'depth': qc.depth(), 'count_ops':{str(k):int(v) for k,v in qc.count_ops().items()}, 'one_qubit':0, 'two_qubit':0, 'three_or_more_qubit':0, 'swap_count':0, 'num_qubits':qc.num_qubits}
    for op in qc.data:
        n=len(op.qubits)
        if n==1:r['one_qubit']+=1
        elif n==2:r['two_qubit']+=1
        elif n>2:r['three_or_more_qubit']+=1
        if op.operation.name=='swap':r['swap_count']+=1
    return r


def read_inputs(root):
    s1 = pd.read_csv(root/'stage1_result.csv')
    req={'customer_id','predicted_probability','rank','top10_flag'}
    assert req.issubset(s1.columns), 'Stage1 file missing required columns'
    assert len(s1)==100 and s1.customer_id.is_unique
    assert s1.predicted_probability.between(0,1).all()
    s1=s1.sort_values(['predicted_probability','customer_id'], ascending=[False,True],kind='stable').reset_index(drop=True)
    assert s1['rank'].astype(int).tolist()==list(range(1,101)), 'Rank must equal descending-probability order'
    assert s1.top10_flag.astype(int).tolist()==[1]*10+[0]*90
    econ=pd.read_csv(root/'N17_top10_economics.csv')
    req_e={'customer_id','contact_cost','expected_deposit_value','capacity_units'}
    assert req_e.issubset(econ.columns) and econ.customer_id.is_unique
    candidates=s1.head(10)[['customer_id','predicted_probability']].merge(econ.drop(columns=['predicted_probability','rank','top10_flag'],errors='ignore'),on='customer_id',validate='one_to_one',how='left',sort=False)
    assert candidates[sorted(req_e-{'customer_id'})].notna().all().all()
    return s1,candidates


def stage2(root, out, s1, cands, shots, seed):
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator

    prev=json.loads((root/'stage2_summary.json').read_text())
    n,k=10,6
    ids=list(cands.customer_id)
    assert ids==prev['candidate_ids'], 'Stage 1 ranking differs from earlier QAOA optimization; reoptimize angles'
    w=(cands.predicted_probability*cands.expected_deposit_value-cands.contact_cost).to_numpy(float)
    co=cands.contact_cost.to_numpy(float)
    ca=cands.capacity_units.to_numpy(float)
    all_sets=list(itertools.combinations(range(n), k))
    masks=np.array([sum(1<<i for i in v) for v in all_sets],dtype=int)
    vv=np.array([w[list(v)].sum() for v in all_sets])
    budgets=np.array([co[list(v)].sum() for v in all_sets])
    capacity=np.array([ca[list(v)].sum() for v in all_sets])
    feas=(budgets<=25)&(capacity<=22)
    assert np.any(feas)
    exact_idx=int(np.argmax(np.where(feas,vv,-np.inf)))
    optmask=int(masks[exact_idx]);optval=float(vv[exact_idx])
    expected_ids=[ids[j] for j in all_sets[exact_idx]]
    assert expected_ids == prev['selected_ids_exact'], 'Expected optimum differs from saved Stage 2; stop'
    # Precision from stage1 can differ slightly from rounded previous report (normal).
    assert abs(optval-prev['objective_exact'])<1e-4
    parms=prev['qaoa_numerical_simulation']['optimized_parameters_gamma_beta_interleaved']
    p=int(prev['qaoa_numerical_simulation']['depth_p'])
    assert len(parms)==2*p

    initial=np.zeros(1<<n,dtype=complex)
    initial[masks]=1/math.sqrt(len(masks))
    qc=QuantumCircuit(n)
    qc.initialize(initial,list(range(n)))
    for layer in range(p):
        gamma,beta=parms[2*layer:2*layer+2]
        # IMPORTANT: for relative phase exp(+i*gamma*w*x), RZ(+gamma*w), NOT RZ(-gamma*w).
        # RZ(phi)|1> / RZ(phi)|0> = exp(+i*phi).
        for j,wi in enumerate(w): qc.rz(float(gamma*wi),j)
        for parity in (0,1):
            for j in range(parity,n-1,2):
                qc.rxx(2*float(beta),j,j+1)
                qc.ryy(2*float(beta),j,j+1)

    # Critical numerical-vs-Qiskit circuit validation (same order + correct sign).
    psi_expected=np.ones(len(masks),dtype=complex)/math.sqrt(len(masks))
    rev={int(m):ii for ii,m in enumerate(masks)}
    edges=[]
    for j in range(n-1):
        aa=[];bb=[]
        for ii,m in enumerate(masks):
            if ((m>>j)&1)==0 and ((m>>(j+1))&1)==1:
                aa.append(ii);bb.append(rev[int(m^(1<<j)^(1<<(j+1)))])
        edges.append((np.array(aa),np.array(bb)))
    for layer in range(p):
        gamma,beta=parms[2*layer:2*layer+2]
        psi_expected*=np.exp(1j*gamma*vv)
        c=np.cos(2*beta);s=-1j*np.sin(2*beta)
        for j in list(range(0,n-1,2))+list(range(1,n-1,2)):
            ia,ib=edges[j]
            u=psi_expected[ia].copy();v=psi_expected[ib].copy()
            psi_expected[ia]=c*u+s*v
            psi_expected[ib]=s*u+c*v
    from_circuit=np.abs(Statevector.from_instruction(qc).data)**2
    from_subspace=np.abs(psi_expected)**2
    diff=float(np.max(np.abs(from_circuit[masks]-from_subspace)))
    leakage=float(np.sum(from_circuit)-np.sum(from_circuit[masks]))
    if diff>1e-8 or abs(leakage)>1e-8:
        raise RuntimeError(f'Quantum circuit and NumPy QAOA disagree: max_prob_diff={diff:.3g} leakage={leakage:.3g}. Stop; do not submit!')
    expected_qaoa=float(np.dot(from_circuit[masks],vv))
    ideal_opt_prob=float(from_circuit[optmask])
    assert abs(expected_qaoa-prev['qaoa_numerical_simulation']['expected_objective_under_qaoa_distribution'])<1e-4

    qc_m=qc.copy();qc_m.measure_all()
    backend=AerSimulator(method='statevector')
    t0=time.perf_counter();tqc=transpile(qc_m,backend=backend,optimization_level=1,seed_transpiler=seed)
    transpilation_sec=time.perf_counter()-t0
    t0=time.perf_counter()
    job=backend.run(tqc,shots=shots,seed_simulator=seed)
    counts={str(b).replace(' ',''):int(c) for b,c in job.result().get_counts().items()}
    runtime=time.perf_counter()-t0
    measured=[]
    for bits,c in counts.items():
        mask=int(bits,2)
        pick=[i for i in range(n) if mask>>i&1]
        value=float(w[pick].sum());cost=float(co[pick].sum());cap=float(ca[pick].sum())
        measured.append({'bitstring_q9_to_q0':bits,'count':c,'selected_ids':';'.join(ids[i] for i in pick),'num_selected':len(pick),'objective':value,'budget':cost,'capacity':cap,'feasible':len(pick)==6 and cost<=25 and cap<=22})
    measured=sorted(measured,key=lambda x:(-x['count'], -x['objective']))
    pd.DataFrame(measured).to_csv(out/'stage2_aer_counts.csv',index=False)
    valid=[x for x in measured if x['feasible']]
    if not valid: raise RuntimeError('No feasible QAOA shot; increase shots')
    best=max(valid,key=lambda x:x['objective'])
    top=max(measured,key=lambda x:x['count'])
    min_gap=max(0., optval-best['objective'])
    report={'backend':'Qiskit AerSimulator (local simulator; NOT hardware)','algorithm':'Dicke+XY QAOA','seed':seed,'shots':shots,'candidate_ids_in_q0_to_q9_order':ids,'optimized_angles_gamma_beta_interleaved':parms,'depth_p':p,'classical_feasible_subsets':int(feas.sum()),'classical_enumerated_subsets':len(feas),'classical_exact_objective':optval,'classical_exact_selected_ids':expected_ids,'classical_exact_bitstring_q9_to_q0':format(optmask,'010b'),'classical_exact_cost':float(budgets[exact_idx]),'classical_exact_capacity':float(capacity[exact_idx]),'statevector_vs_subspace_max_probability_discrepancy':diff,'statevector_outside_six_excitation_probability':leakage,'statevector_expected_objective':expected_qaoa,'statevector_exact_optimum_probability':ideal_opt_prob,'measured_best_feasible_objective':best['objective'],'measured_best_feasible_bitstring_q9_to_q0':best['bitstring_q9_to_q0'],'measured_best_selected_ids':best['selected_ids'].split(';'),'measured_optimality_gap':min_gap,'most_frequent_bitstring_q9_to_q0':top['bitstring_q9_to_q0'],'most_frequent_count':top['count'],'exact_optimum_shots':counts.get(format(optmask,'010b'),0),'observed_feasible_shots':sum(x['count'] for x in valid),'transpilation_seconds':transpilation_sec,'execution_seconds':runtime,'backend_options':{'method':'statevector','seed_simulator':seed},'original_circuit':count_gates(qc_m),'transpiled_circuit':count_gates(tqc),'bitstring_convention':'q9...q0; rightmost is first customer'}
    dump(out/'stage2_aer_evidence.json',report)
    (out/'stage2_original_circuit.txt').write_text(str(qc_m.draw(output='text',fold=120)))
    (out/'stage2_transpiled_circuit.txt').write_text(str(tqc.draw(output='text',fold=120)))
    return report


def stage3(root,out,s1,stage2report,shots,seed,noise_shots):
    from qiskit import QuantumCircuit,transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    ids=stage2report['classical_exact_selected_ids']
    idx=s1.set_index('customer_id')
    probs=np.array([float(idx.loc[c,'predicted_probability']) for c in ids])
    assert all(0<=p<=1 for p in probs)
    expected=probs_of_outcomes(probs)
    exact=float(expected[3:].sum())
    qc=QuantumCircuit(6,6)
    angles=[2*math.asin(math.sqrt(float(p))) for p in probs]
    for j,theta in enumerate(angles):qc.ry(theta,j)
    qc.measure(range(6),range(6))
    backend=AerSimulator(method='statevector')
    tqc=transpile(qc,backend=backend,optimization_level=1,seed_transpiler=seed)
    def run(tqc,backend,shots,seed):
        t0=time.perf_counter();counts={str(b).replace(' ',''):int(c) for b,c in backend.run(tqc,shots=shots,seed_simulator=seed).result().get_counts().items()};elapsed=time.perf_counter()-t0
        success=sum(c for b,c in counts.items() if b.count('1')>=3)
        p=success/shots
        se=math.sqrt(p*(1-p)/shots)
        return counts,{'shots':shots,'success_shots':success,'estimated_probability':p,'exact_probability':exact,'signed_error':p-exact,'absolute_error':abs(p-exact),'approximate_95pct_shot_interval':[max(0,p-1.96*se),min(1,p+1.96*se)],'execution_seconds':elapsed}
    counts,ideal=run(tqc,backend,shots,seed)
    pd.DataFrame([{'bitstring_q5_to_q0':f'{i:06b}','ones':int(f'{i:06b}'.count('1')),'measured_count':counts.get(f'{i:06b}',0),'success':int(f'{i:06b}'.count('1')>=3)} for i in range(64)]).to_csv(out/'stage3_aer_counts.csv',index=False)
    result={'backend':'Qiskit AerSimulator (simulator; NOT hardware)','selected_customer_ids_q0_to_q5':ids,'probabilities_q0_to_q5':probs.tolist(),'ry_angles_radians':angles,'exact_probability_at_least_three':exact,'exact_probability_by_subscription_count':expected.tolist(),'bitstring_convention':'q5...q0; rightmost bit is first selected customer','ideal_aer':ideal,'original_circuit':count_gates(qc),'transpiled_circuit':count_gates(tqc)}
    (out/'stage3_circuit.txt').write_text(str(tqc.draw(output='text')))
    if noise_shots>0:
        # Explicit illustrative gate/readout noise assumptions; not a device calibration.
        # Using a single-qubit depolarizing channel for the Ry gates and an asymmetric readout error.
        nm=NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(.001,1),['ry'])
        nm.add_all_qubit_readout_error(ReadoutError([[.98,.02],[.03,.97]]))
        noisy_backend=AerSimulator(noise_model=nm,method='density_matrix')
        noisy_tqc=transpile(qc,noisy_backend,optimization_level=0,seed_transpiler=seed)
        noisy_counts,noisy=run(noisy_tqc,noisy_backend,noise_shots,seed+1)
        result['illustrative_noise_model']={'depolarizing_error_1q':.001,'readout_P_0to1':.02,'readout_P_1to0':.03,'is_device_calibrated':False}
        result['noisy_aer']=noisy
        result['noisy_transpiled_circuit']=count_gates(noisy_tqc)
        pd.DataFrame([{'bitstring_q5_to_q0':f'{i:06b}','measured_count':noisy_counts.get(f'{i:06b}',0)} for i in range(64)]).to_csv(out/'stage3_noisy_aer_counts.csv',index=False)
    dump(out/'stage3_aer_evidence.json',result)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,default=Path('submission_inputs'))
    ap.add_argument('--output-dir',type=Path,default=Path('aer_evidence'))
    ap.add_argument('--shots',type=int,default=8192)
    ap.add_argument('--noise-shots',type=int,default=8192,help='0 to disable illustrative Stage 3 noise test')
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if args.shots<=0 or args.noise_shots<0:ap.error('invalid shot count')
    try:
        import qiskit,qiskit_aer
    except ImportError as e: raise SystemExit('Install qiskit and qiskit-aer in your active Python environment (see README).') from e
    args.output_dir.mkdir(parents=True,exist_ok=True)
    s1,cands=read_inputs(args.input_dir)
    s2=stage2(args.input_dir,args.output_dir,s1,cands,args.shots,args.seed)
    s3=stage3(args.input_dir,args.output_dir,s1,s2,args.shots,args.seed,args.noise_shots)
    metadata={'python':sys.version,'platform':platform.platform(),'qiskit_version':qiskit.__version__,'qiskit_aer_version':qiskit_aer.__version__,'numpy':np.__version__,'pandas':pd.__version__,'seed':args.seed,'simulator_only':True}
    dump(args.output_dir/'versions_and_environment.json',metadata)
    print('Stage2 exact objective',s2['classical_exact_objective'],'Aer observed gap',s2['measured_optimality_gap'])
    print('Stage3 exact P',s3['exact_probability_at_least_three'],'Aer measured P',s3['ideal_aer']['estimated_probability'])
    print('EVIDENCE SAVED:',args.output_dir.resolve())

if __name__=='__main__':main()
