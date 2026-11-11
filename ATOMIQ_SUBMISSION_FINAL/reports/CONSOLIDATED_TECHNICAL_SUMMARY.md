# Technical summary — N17 hybrid quantum-classical deposit campaign

## Pipeline
Train on 300 labeled customers, predict all 100 test probabilities with the N17 + Friend Hybrid estimator (history gamma 0.2), select ten by descending probability and ID tie-break, optimize six contact decisions, and compute P(at least 3 subscriptions).

## Stage 1
N17 combines F2 quantum correlation features, a ring-like additional quantum feature expert for unknown prior campaign outcomes, and Friend hybrid (LR Compact, quantum kernel and VQC) for the unknown stratum; the non-unknown stratum uses F2 and Friend LR with history adjustment. Exploratory OOF comparisons suggest accuracy improvements, but the repeated selection of model families on the same 300 samples makes those scores optimistic. The official Stage 1 file must list all 100 customers, rank 1–100, with exactly ten flags and probabilities in [0,1]. Current package supplies only the reported top-ten snapshot until the user's 100-row local CSV is incorporated.

## Stage 2
Optimize sum_i ((p_i D_i - c_i) x_i) with x_i binary, cardinality six, cost ceiling 25 and capacity ceiling 22. Exact exhaustive search of C(10,6)=210 subsets selects C0081, C0079, C0031, C0327, C0374, C0222, objective 43.183213, cost 14 and capacity 11. All 210 cardinality-six subsets meet costs/capacities, so using |D_10^6> Dicke preparation and a cardinality-preserving XY mixer is feasible without inequality penalties in this instance. The cost Hamiltonian is -sum_i w_i (I-Z_i)/2. NumPy 210-state simulator with p=2, gamma/beta parameters [0.8644973638, 1.1101127978, 0.4277927114, 0.3294877373], 8192 simulated shots, obtains optimal bitstring q9...q0 = 0101001111 242 times, best observed objective 43.183213, observed gap 0, distribution expectation 33.720067. No hardware/Aer output is asserted.

## Stage 3
Using exactly the Stage 2 six in order q0–q5, prepare Ry(2 arcsin sqrt(p_i)) for each and measure. Count >=3 ones. Analytic Poisson-binomial P(K>=3) = 0.677951491947. Python independent Bernoulli simulation with seed 42 observed 5582/8192 = 0.681396484375; abs discrepancy ~0.003445. Six Ry, six measurement, zero entangling gates. Aer physical/transpiled depth has not been recorded.

## Limitations and integrity
No test labels were used. Stage 2/3 probabilities are rounded six-decimal snapshots of N17 top-ten provided in chat. All three stages must be validated against the exact 100-row Stage 1 N17 output before uploading. Claims of quantum advantage are not supported. Noise and physical connectivity metrics require additional backend runs.
