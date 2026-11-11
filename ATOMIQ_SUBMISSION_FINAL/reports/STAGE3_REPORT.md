# Stage 3 — N17 + Friend Hybrid (gamma=0.2)

## Objective and Stage 2 input

Using the exact six customers selected in Stage 2, estimate the probability of **at least three** customers subscribing. Customer responses are modelled as independent Bernoulli random variables with the Stage 1 probabilities below.

| Qubit | Customer | Probability p | Gate |
|---|---|---:|---|
| q0 | C0081 | 0.849352 | Ry(2 asin sqrt(p)) |
| q1 | C0079 | 0.729566 | Ry(2 asin sqrt(p)) |
| q2 | C0031 | 0.701745 | Ry(2 asin sqrt(p)) |
| q3 | C0327 | 0.264112 | Ry(2 asin sqrt(p)) |
| q4 | C0374 | 0.226869 | Ry(2 asin sqrt(p)) |
| q5 | C0222 | 0.182735 | Ry(2 asin sqrt(p)) |

## Quantum circuit

Prepare each of six qubits using `Ry(theta_i)`, with `theta_i = 2*asin(sqrt(p_i))`. The resulting state is the tensor product

`|psi> = tensor_i (sqrt(1-p_i)|0> + sqrt(p_i)|1>)`.

Measure all six qubits. Qiskit counts return bit strings in **q5 q4 q3 q2 q1 q0 order** (left-to-right); `q0` is the rightmost bit, corresponding to **C0081**. An outcome is a success when the bit string has **at least three ones**. There are 6 rotation gates, 6 measurements, no entangling gates; circuit depth depends on transpilation and the target backend.

## Exact reference

For `K = sum X_i`, compute the Poisson-binomial distribution using the recursion

`P_{i+1}(k) = (1-p_i)P_i(k) + p_i P_i(k-1)`.

For this input, `P(K>=3) = 0.6779514919470052`, so **67.795149%** is the exact reference value.

| K | Exact probability |
|---:|---:|
| 0 | 0.005649891451 |
| 1 | 0.065338303784 |
| 2 | 0.251060312819 |
| 3 | 0.389826667022 |
| 4 | 0.227945257637 |
| 5 | 0.055418364762 |
| 6 | 0.004761202526 |

## Numerical shot test (seed 42)

The default Python program independently samples ideal Bernoulli outcomes, mathematically equivalent to sampling the ideal Ry circuit output distribution. This **is not** a Qiskit Aer run and **is not** a real hardware measurement.

- Shots: **8192**
- Success shots (K >= 3): **5582**
- Estimate: **0.681396484375** (68.139648%)
- Absolute error vs exact: **0.003444992428** (0.344499 percentage points)
- Wilson 95% confidence interval: **[0.671224, 0.691399]**
- Estimated shot standard error: ~0.00516.

The result is consistent with the exact benchmark at the expected sampling precision. The independent-outcomes assumption and errors in Stage 1 probabilities are not included in this shot-noise interval.

## Reproducibility

```bash
python atomiq_stage3_N17.py --backend numpy --shots 8192 --seed 42
```

To use actual Qiskit Aer simulator measurements (in an environment with `qiskit` and `qiskit-aer` installed):

```bash
python atomiq_stage3_N17.py --backend aer --shots 8192 --seed 42
```

Qiskit Aer and the Python reference sampler need not produce identical *shot counts* for the same seed because their random-number generators differ. They should agree statistically with the exact reference.

Files: `stage3_submission.json`, `stage3_summary.csv`, `stage3_selected_customers.csv`, `stage3_outcome_distribution.csv`, `stage3_bitstring_counts.csv`.

## Limitation

Qiskit transpiler statistics (basis gate counts, backend-specific depths, SWAPs) and noisy-device simulation require an actual Qiskit backend and transpilation settings; none were measured here. All six probabilities must match the Stage 1 probabilities actually submitted, and the selected IDs must match the Stage 2 output.
