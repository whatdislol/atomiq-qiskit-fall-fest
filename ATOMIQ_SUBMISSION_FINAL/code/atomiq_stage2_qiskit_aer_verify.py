#!/usr/bin/env python3
"""Qiskit Aer validation of the numerically optimized Stage 2 Dicke+XY QAOA.
Requires qiskit, qiskit-aer, numpy, pandas. This is a SIMULATOR, not hardware.
To reproduce optimizer parameters, run atomiq_stage2_complete.py first.
"""
import argparse, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage1', default='stage1_result.csv')
    ap.add_argument('--data',default='stage2_customer_inputs_complete.csv')
    ap.add_argument('--summary',default='stage2_final_results/stage2_summary.json')
    ap.add_argument('--outdir',default='stage2_final_results')
    ap.add_argument('--shots',type=int,default=8192)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    report=json.loads(Path(args.summary).read_text())
    theta=np.asarray(report['qaoa_numerical_simulation']['optimized_parameters_gamma_beta_interleaved'],dtype=float)
    depth=int(report['qaoa_numerical_simulation']['depth_p'])
    s=pd.read_csv(args.stage1).sort_values(['predicted_probability','customer_id'],ascending=[False,True]).head(10)
    df=s[['customer_id','predicted_probability']].merge(pd.read_csv(args.data),on='customer_id',validate='one_to_one')
    assert df.customer_id.tolist()==report['candidate_ids'], 'Candidate order changed'
    w=(df.predicted_probability*df.expected_deposit_value-df.contact_cost).to_numpy(float)
    n=10;k=6
    dicke=np.zeros(2**n,dtype=complex)
    for inds in itertools.combinations(range(n),k):
        dicke[sum(1<<j for j in inds)]=1/np.sqrt(210)
    qc=QuantumCircuit(n,n)
    qc.initialize(dicke,list(range(n)))
    for layer in range(depth):
        gamma,beta=theta[2*layer:2*layer+2]
        for j,v in enumerate(w):qc.rz(-float(gamma*v),j)
        for shift in (0,1):
            for j in range(shift,n-1,2):
                qc.rxx(2*float(beta),j,j+1)
                qc.ryy(2*float(beta),j,j+1)
    qc.measure(list(range(n)),list(range(n)))
    backend=AerSimulator(seed_simulator=args.seed)
    tqc=transpile(qc,backend,seed_transpiler=args.seed,optimization_level=1)
    counts=backend.run(tqc,shots=args.shots).result().get_counts()
    rec=[]
    for bits,count in counts.items():
        bits=bits.replace(' ','')
        selected=[j for j in range(n) if bits[-1-j]=='1']
        obj=sum(w[j] for j in selected)
        cost=float(df.iloc[selected].contact_cost.sum());capacity=float(df.iloc[selected].capacity_units.sum())
        rec.append({'bitstring':bits,'shots':count,'selected_ids':';'.join(df.iloc[selected].customer_id),'objective':float(obj),'budget_used':cost,'capacity_used':capacity,'feasible':len(selected)==6 and cost<=25 and capacity<=22})
    scores=pd.DataFrame(rec).sort_values('objective',ascending=False)
    feasible=scores[scores.feasible]
    assert len(feasible)>0
    best=feasible.iloc[0]
    out=Path(args.outdir);out.mkdir(exist_ok=True,parents=True)
    scores.to_csv(out/'qiskit_aer_measured_bitstrings.csv',index=False)
    details={'backend':'Qiskit AerSimulator','seed':args.seed,'shots':args.shots,'optimized_angles':theta.tolist(),'transpiled_circuit_depth':tqc.depth(),'one_qubit_gates':sum(len(inst.qubits)==1 for inst in tqc.data),'two_qubit_gates':sum(len(inst.qubits)==2 for inst in tqc.data),'best_observed_bitstring':best.bitstring,'selected_ids':best.selected_ids.split(';'),'best_observed_objective':float(best.objective),'gap':float(report['objective_exact']-best.objective),'top_shot_bitstring':max(counts,key=counts.get),'top_shot_frequency':max(counts.values())/args.shots}
    (out/'qiskit_aer_validation.json').write_text(json.dumps(details,indent=2,ensure_ascii=False))
    print(json.dumps(details,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
