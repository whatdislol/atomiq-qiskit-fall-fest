#!/usr/bin/env python3
import csv,json,math
from pathlib import Path
root=Path(__file__).resolve().parent
with (root/'stage1_result.csv').open() as f: a=list(csv.DictReader(f))
s2=json.loads((root/'stage2_result.json').read_text());s3=json.loads((root/'stage3_result.json').read_text())
e2=json.loads((root/'evidence/qiskit_aer/stage2_aer_evidence.json').read_text());e3=json.loads((root/'evidence/qiskit_aer/stage3_aer_evidence.json').read_text())
assert len(a)==100 and len(set(r['customer_id'] for r in a))==100
assert all(0<=float(r['predicted_probability'])<=1 for r in a)
sorted_ids=[r['customer_id'] for r in sorted(a,key=lambda r:(-float(r['predicted_probability']),r['customer_id']))]
assert sorted_ids==[r['customer_id'] for r in a]
assert all(int(r['rank'])==j+1 and int(r['top10_flag'])==int(j<10) for j,r in enumerate(a))
assert sorted_ids[:10]==s2['candidate_ids']==e2['candidate_ids_in_q0_to_q9_order']
assert s2['selected_ids']==s3['selected_ids_q0_to_q5']==e3['selected_customer_ids_q0_to_q5']
assert len(s2['selected_ids'])==6 and s2['budget_used']<=s2['budget_limit'] and s2['capacity_used']<=s2['capacity_limit']
assert s2['optimality_gap_measured_best']==0 and e2['exact_optimum_shots']==236 and e2['observed_feasible_shots']==8192
assert e3['ideal_aer']['success_shots']==5515 and e3['noisy_aer']['success_shots']==5392
assert math.isclose(s3['exact_probability'],e3['exact_probability_at_least_three'],rel_tol=0,abs_tol=1e-12)
assert all(math.isclose(float(a[next(i for i,r in enumerate(a) if r['customer_id']==cid)]['predicted_probability']),p,rel_tol=0,abs_tol=1e-9) for cid,p in zip(s3['selected_ids_q0_to_q5'],s3['probabilities_q0_to_q5']))
print('PASS: All Stage 1 / Stage 2 / Stage 3 data and Aer evidence are mutually consistent')
