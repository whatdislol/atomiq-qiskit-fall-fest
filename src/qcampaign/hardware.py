"""Backends, transpilation metrics, noisy simulation and readout-error mitigation."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import StagedPassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from . import config


def fake_backend(name: str):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def _two_qubit_ops(ops: dict) -> int:
    return sum(v for k, v in ops.items() if k in {"cx", "cz", "ecr", "swap", "rzz", "rxx", "ryy", "cp", "crx", "cry",
                                                   "crz", "iswap", "xx_plus_yy", "rzx", "ch", "cu", "cswap"})


def gate_counts(qc: QuantumCircuit) -> dict:
    """Counts 1q / 2q / 3q+ gates by arity (measure, barrier, delay excluded)."""
    one = two = multi = 0
    for inst in qc.data:
        if inst.operation.name in ("measure", "barrier", "delay", "reset"):
            continue
        k = len(inst.qubits)
        if k == 1:
            one += 1
        elif k == 2:
            two += 1
        else:
            multi += 1
    return {"one_qubit": one, "two_qubit": two, "multi_qubit": multi}


def transpile_report(qc: QuantumCircuit, backend, optimization_level: int = 3, seed=None):
    """Transpile for a backend and report logical/physical qubits, depths, gate and SWAP counts.

    SWAPs are counted after the routing stage (before basis translation turns them into
    2-qubit native gates), using the same seeded preset pass manager.
    """
    seed = config.SEEDS["transpiler"] if seed is None else seed
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, seed_transpiler=seed)
    tqc = pm.run(qc)
    route_pm = StagedPassManager(stages=["init", "layout", "routing"], init=pm.init, layout=pm.layout, routing=pm.routing)
    routed = route_pm.run(qc)
    swaps = routed.count_ops().get("swap", 0)
    layout = tqc.layout.final_index_layout() if tqc.layout is not None else list(range(qc.num_qubits))
    active = {tqc.find_bit(q).index for inst in tqc.data for q in inst.qubits
              if inst.operation.name not in ("barrier",)}
    logical = qc.remove_final_measurements(inplace=False)
    rep = {
        "backend": backend.name,
        "optimization_level": optimization_level,
        "seed_transpiler": seed,
        "logical_qubits": qc.num_qubits,
        "physical_qubits_used": len(active),
        "physical_layout": layout,
        "original_depth": logical.depth(),
        "original_ops": dict(logical.count_ops()),
        "original_gate_counts": gate_counts(logical),
        "transpiled_depth": tqc.depth(),
        "transpiled_two_qubit_depth": tqc.depth(lambda i: len(i.qubits) == 2),
        "transpiled_ops": dict(tqc.count_ops()),
        "transpiled_gate_counts": gate_counts(tqc),
        "swap_count_after_routing": int(swaps),
    }
    return tqc, rep


def noisy_simulator(backend, seed=None):
    seed = config.SEEDS["noise_sim"] if seed is None else seed
    return AerSimulator.from_backend(backend, seed_simulator=seed)


def ideal_simulator(seed):
    return AerSimulator(method="automatic", seed_simulator=seed)


def run_counts(sim, tqc, shots):
    res = sim.run(tqc, shots=shots).result()
    return res.get_counts()


# ----------------------------------------------------------------------------- readout mitigation
def calibrate_readout(backend, physical_qubits, shots, seed):
    """Measure the per-qubit confusion matrices by running |0..0> and |1..1> calibration
    circuits on the noisy simulator (tensored / local readout calibration)."""
    n = len(physical_qubits)
    sim = noisy_simulator(backend, seed)
    mats = []
    circs = []
    for prep in (0, 1):
        qc = QuantumCircuit(backend.num_qubits, n)
        if prep:
            for pq in physical_qubits:
                qc.x(pq)
        for i, pq in enumerate(physical_qubits):
            qc.measure(pq, i)
        circs.append(qc)
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend, seed_transpiler=config.SEEDS["transpiler"],
                                      initial_layout=None)
    tc = pm.run(circs)
    res = sim.run(tc, shots=shots).result()
    c0, c1 = res.get_counts(0), res.get_counts(1)
    for i in range(n):
        def p_one(counts):
            tot = sum(counts.values())
            return sum(v for k, v in counts.items() if k[::-1][i] == "1") / tot
        p10 = p_one(c0)          # measured 1 | prepared 0
        p01 = 1 - p_one(c1)      # measured 0 | prepared 1
        mats.append(np.array([[1 - p10, p01], [p10, 1 - p01]]))
    return mats


def mitigate_counts(counts: dict, mats: list[np.ndarray]) -> np.ndarray:
    """Tensored inverse of per-qubit confusion matrices applied to the measured distribution,
    followed by projection onto the probability simplex (clip negatives, renormalise).
    Returns a probability vector indexed by the integer value of the bitstring."""
    n = len(mats)
    shots = sum(counts.values())
    p = np.zeros(2 ** n)
    for k, v in counts.items():
        p[int(k.replace(" ", ""), 2)] += v / shots
    t = p.reshape([2] * n)  # axis 0 = most significant bit = qubit n-1
    for q in range(n):
        inv = np.linalg.inv(mats[q])
        ax = n - 1 - q
        t = np.moveaxis(np.tensordot(inv, np.moveaxis(t, ax, 0), axes=(1, 0)), 0, ax)
    p = t.reshape(-1)
    p = np.clip(p, 0, None)
    return p / p.sum()
