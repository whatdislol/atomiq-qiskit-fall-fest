# ATOMIQ Hackathon — Final Submission (Qiskit Aer verified)

**Submission model:** N17 + Friend Hybrid (history correction γ=0.2). This folder contains the 100-customer Stage 1 predictions, the Stage 2 six-person optimized decision, and the Stage 3 outcome estimation, with actual **Qiskit AerSimulator** shot counts and code.

## Files for judges (root)

- `stage1_result.csv` — 100 customers, probabilities, ranks, exactly 10 selected candidates.
- `stage2_candidates.csv` — ten Stage 2 inputs, their economics and expected net values.
- `stage2_result.csv` and `stage2_result.json` — **Aer-based** Stage 2 optimization and classical exact benchmark.
- `stage3_result.csv` and `stage3_result.json` — **Aer-based** Stage 3 outcomes, exact benchmark, and illustrative noise model.

## Main verified outcomes

- **Stage 2 selected:** C0081, C0079, C0031, C0327, C0374, C0222.
- **Expected net value (full-precision Stage 1 probabilities):** 43.183204089; cost 14 / 25; capacity 11 / 22. Exact classical search checked all 210 six-person combinations.
- **QAOA:** 10 qubits, p=2, Dicke+XY; **8192** Qiskit AerSimulator shots; best sampled bitstring `0101001111` matches exact optimizer; best-sample gap 0; optimal bitstring measured **236** times; all 8192 shots feasible.
- **Stage 3:** 6 qubits, six Ry rotations; theoretical independent-response probability of at least 3 subscribers **0.677951472**; ideal Aer **5515/8192 = 0.673217773**; synthetic noisy Aer **5392/8192 = 0.658203125**.

## Organization

- `evidence/qiskit_aer/`: raw measured counts, evidence JSON, circuit diagrams and environment versions.
- `code/`: N17 inference / validation and Stage 2 / 3 Qiskit reproduction scripts; installation notes.
- `data/`: the given labeled train and unlabeled test inputs.
- `reproduce_inputs/`: exact input files for the Aer reproduction command (including QAOA angles and ten-customer economics).
- `reports/`: technical analysis, exact-vs-Aer explanation and presentation outline.
- `supplementary/numpy_baseline/`: previous **NumPy-only** simulated outputs, retained for provenance and **not** the primary submitted results.
- `MANIFEST_SHA256.txt`: per-file checksums.

## Reproduction

Python 3.11+ recommended; the captured environment was Python 3.11.18, Qiskit 2.6.2, Aer 0.18.2, numpy 2.4.6 and pandas 3.1.1 (see `evidence/qiskit_aer/versions_and_environment.json`).

```bash
python -m pip install -r code/requirements_aer.txt
python code/run_aer_evidence.py \
  --input-dir reproduce_inputs \
  --output-dir aer_rerun \
  --shots 8192 --noise-shots 8192 --seed 42
```

For the complete Stage 1 refit, the **external friend's `friend_atomiq` repository is required** and is not included in this archive. The Stage 1 prediction file is already included, but full end-to-end reproducibility requires pinning and including that dependency and installing its compatible libraries. See `code/atomiq_N17_final_test_top10.py`. The Stage 2 and Stage 3 code and recorded evidence are supplied.

## Important scientific limitations

1. **Aer is a local simulator; no quantum hardware run was performed.** Do not present Aer counts as hardware shots.
2. Stage 2 Dicke state is prepared with an opaque Qiskit `initialize`; the listed circuit depth 12 is **not** a native-gate, hardware-ready compiled depth. No quantum advantage is established; 210 classical combinations are trivial to enumerate.
3. The Stage 2 optimality gap 0 refers to the **best sampled** bitstring, not to the most frequent sample or the mean QAOA objective.
4. The Stage 3 noisy Aer experiment uses an illustrative, **not device-calibrated**, noise model. The Stage 3 shot uncertainty does not capture Stage 1 probability/model uncertainty.
5. Full-precision Stage 1 probabilities produce Stage 2 value 43.183204089; earlier rounded reports show 43.183213. The Aer figure is the primary result.

## Validation

Run `python verify_submission.py` for cross-stage identity, probability, constraint and evidence integrity checks. Review `reports/AER_VERIFIED_TECHNICAL_REPORT.md` for the detailed methodology.
