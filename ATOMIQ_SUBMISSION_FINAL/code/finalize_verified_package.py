#!/usr/bin/env python3
"""Only package official results and REAL locally generated Qiskit Aer evidence."""
import argparse, json, zipfile, shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evidence-dir',type=Path,default=Path('aer_evidence'))
    p.add_argument('--package-source',type=Path,default=Path('submission_existing'))
    p.add_argument('--out',type=Path,default=Path('ATOMICQ_WITH_QISKIT_AER_EVIDENCE.zip'))
    args=p.parse_args()
    required=['stage2_aer_evidence.json','stage3_aer_evidence.json','versions_and_environment.json','stage2_aer_counts.csv','stage3_aer_counts.csv','stage2_original_circuit.txt','stage2_transpiled_circuit.txt','stage3_circuit.txt']
    for name in required:
        if not (args.evidence_dir/name).exists():p.error('Missing Aer evidence: '+name+'; run run_aer_evidence.py first')
    e2=json.loads((args.evidence_dir/'stage2_aer_evidence.json').read_text())
    e3=json.loads((args.evidence_dir/'stage3_aer_evidence.json').read_text())
    assert 'AerSimulator' in e2['backend'] and 'AerSimulator' in e3['backend']
    assert e2['statevector_vs_subspace_max_probability_discrepancy']<1e-8
    assert len(e2['classical_exact_selected_ids'])==6
    assert e2['classical_exact_selected_ids']==e3['selected_customer_ids_q0_to_q5']
    assert e2['observed_feasible_shots']<=e2['shots']
    assert e2['exact_optimum_shots']<=e2['shots']
    assert 0<=e3['ideal_aer']['estimated_probability']<=1
    assert abs(e3['exact_probability_at_least_three']-.677951491947)<1e-4
    if not args.package_source.is_dir():p.error('Place previous verified stage files in '+str(args.package_source))
    official=['stage1_result.csv','stage2_result.csv','stage2_result.json','stage3_result.csv','stage3_result.json']
    for x in official:
        if not (args.package_source/x).exists():p.error('Missing official file: '+x)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.out,'w',zipfile.ZIP_DEFLATED) as z:
        for file in args.package_source.rglob('*'):
            if file.is_file():z.write(file,'results/'+str(file.relative_to(args.package_source)))
        for file in args.evidence_dir.rglob('*'):
            if file.is_file():z.write(file,'evidence/'+str(file.relative_to(args.evidence_dir)))
        for script in ('run_aer_evidence.py','finalize_verified_package.py','README_QISKIT_AER.md','requirements_aer.txt'):
            z.write(Path(__file__).parent/script,'code/'+script)
        for inputname in ('N17_top10_economics.csv','stage2_summary.json'):
            z.write(Path(__file__).parent/inputname,'inputs/'+inputname)
    print('Validated Aer evidence and official Stage1-3 results:',args.out.resolve())
    print('Backend:',e2['backend'],'(not real quantum hardware)')

if __name__=='__main__':main()
