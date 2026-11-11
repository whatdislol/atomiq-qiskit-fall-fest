# Atomiq Hackathon — Stage 2, N17 + Friend Hybrid

## Stage 1 linkage
These calculations use the **N17 + Friend Hybrid** ten probabilities provided by the user, **not** the earlier F2+A0+Friend LR consensus. N17 top-ten IDs in descending probability: C0081, C0079, C0031, C0327, C0183, C0208, C0374, C0193, C0222, C0318. **The Stage 1 submission must match this same N17 ranking and probability set** if this Stage 2 result is submitted. The present `N17_top10.csv` is a ten-person input snapshot, NOT the Stage 1 100-person submission.

## Optimization formulation
For candidate `i`, let p_i be N17 probability, D_i expected deposit value, c_i contact cost, h_i capacity units, and x_i ∈ {0,1} contact decision. Expected net value w_i = p_i D_i − c_i. Maximize `Z = Σ_i w_i x_i` subject to `Σ_i x_i = 6`, `Σ_i c_i x_i ≤ 25`, `Σ_i h_i x_i ≤ 22`.

## Candidate table and classical optimum
See `results/stage2_candidates.csv` for full 10 rows. Exact optimum is the six customers **C0081, C0079, C0031, C0327, C0374, C0222**. In Stage 1 ranked qubit order q0...q9, their binary contact decisions are `[1,1,1,1,0,0,1,0,1,0]`. Standard displayed bitstring q9...q0 is **`0101001111`**. Exact objective **43.183213**, total contact cost **14/25**, capacity **11/22**. All **210 of 210** 6-of-10 combinations satisfy both inequalities for this particular candidate set; maximum cost 19, maximum capacity 17. Thus constraints on cost/capacity are redundant after imposing six selections in this instance only.

## QUBO / Ising formulation
The exact constrained problem above can be encoded with unconstrained pseudo-Boolean objective:
`E(x) = −Σ_i w_i x_i + A(Σ_i x_i − 6)^2`
for a suitable positive A. Given that all six-of-ten selections are feasible, no cost/capacity slack variables or penalties are needed for this specific instance. Instead of optimizing a penalty, our simulation **enforces cardinality exactly** by preparing a uniform Dicke state and using a Hamming-weight-preserving XY mixer.

Cost Hamiltonian: `H_C = −Σ_i w_i (I−Z_i)/2`. Initial state: `|D_10^6> = (1/√210) Σ_{|x|=6} |x>`. Phase operator: `e^{−i γ H_C}` is equivalent up to global phase to `Π_i RZ_i(−γ w_i)`. Mixer for an edge `(i,j)`: `e^{−i β(X_iX_j+Y_iY_j)}`, implemented using `RXX(2β)` then `RYY(2β)` for each nearest-neighbor edge, applied in alternating even/odd edge sweeps. Two layers use 20 RZ and 36 RXX/RYY rotations before state preparation and any backend transpilation.

## Quantum/hybrid numerical experiment
The provided `atomiq_stage2_N17.py` uses a **custom NumPy statevector simulator restricted to the 210-dimensional fixed-Hamming-weight subspace** (classical simulation, not physical quantum hardware nor Qiskit Aer). Parameters: p=2 layers, COBYLA 5 restarts × 400 max function evaluations, simulation seed 42. Optimized `[γ1,β1,γ2,β2] = [0.8644973638401579, 1.1101127977966894, 0.4277927114086896, 0.32948773731137304]`. Expectation `E[Z] = 33.72006665946626`; probability of exact-optimum bitstring in simulated final state = `0.02882817785318697` (2.8828%). From **8192 simulated draws**, the exact optimum `0101001111` occurred **242** times; the best feasible observed output is therefore exactly the classical optimum, net value **43.183213**, resulting gap **0.000000**. Across the 8192 simulated draws, **191 distinct bitstrings** appeared. It is essential to distinguish **best observed** from **most frequent** and **expected** objective. This result does **not demonstrate quantum advantage**.

## Bitstring-to-customer decoding
Bits q0..q9 correspond to the ten customers in N17 probability order, and q0 is the **rightmost** bit in the printed ten-character string. `0101001111` is q9..q0. Select those `i` for which bit q_i=1; verify exactly six unique customers, cost ≤25 and capacity ≤22. See `results/simulated_qaoa_bitstrings.csv` and `results/all_210_subsets.csv` for complete diagnostics.

## Gates, depth, noise, and what is not yet verified
10 logical qubits, QAOA p=2. Cost phase layer: 10 RZ × 2. XY mixer: 9 nearest-neighbor edges ×2 two-qubit rotations ×2 layers =36 rotations. Dicke initialization is **not included** in these rotation counts. Circuit depth, compiled backend-specific CX/SWAP counts, and noise results cannot be claimed until the provided `atomiq_stage2_qiskit_aer_verify.py` is run using Qiskit Aer on a specified backend. No hardware test was performed. The provided verification script reproduces the optimized angles and separately records Qiskit Aer simulated measurement results; these may vary from NumPy sampling due to randomness.

## Files
- `N17_top10.csv`: ten predicted probabilities supplied by user (NOT full Stage 1 submission).
- `N17_top10_economics.csv`: corresponding Stage 2 economic parameters.
- `atomiq_stage2_N17.py`: full classical enumeration and NumPy XY-QAOA simulation.
- `atomiq_stage2_qiskit_aer_verify.py`: Qiskit Aer validation (requires Qiskit and Aer).
- `results/stage2_candidates.csv`: all candidates, weights, exact selection flags.
- `results/all_210_subsets.csv`: all feasible combinations and objectives.
- `results/simulated_qaoa_bitstrings.csv`: simulated measurement frequencies and decoded IDs.
- `results/stage2_summary.json`: structured results.
- `results/stage2_submission_summary.csv`: compact required submission facts.
