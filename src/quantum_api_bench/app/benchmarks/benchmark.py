import time
import numpy as np
from dataclasses import dataclass
from typing import Callable
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2

@dataclass
class BenchmarkResults:
    algorithm: str
    backend_type: str  # ideal, noisy_sim, real
    shots: int
    runtime_ms: float
    counts: dict
    circuit_depth: int
    gate_count: int
    fidelity: float | None  # use none for real hardware
    correct_result: bool
    metadata: dict

def run_benchmark(
        qc : QuantumCircuit,
        algorithm_name: str,
        expected_key: str,
        backend,
        backend_type: str,
        shots: int = 1024,
        ideal_statevector: Statevector | None = None
) -> BenchmarkResults:

    # translates my circuit into the native gate set supported by the current IBM hardware
    # this is the circuit that gets actually executed
    transpiled = transpile(qc, backend=backend, optimization_level=1)

    ### NOTE: Both of the depth and gate count increase after transpilation ###
    # Number of layers of gates. Tells us how long the computation takes relative to coherence time
    # A deeper circuit is going to be more exposed to noise because the qubits have more time
    # to decohere before measurement
    circuit_depth=transpiled.depth()

    # The raw total number of operations.
    gate_count=sum(transpiled.count_ops().values())

    sampler = BackendSamplerV2(backend=backend)

    # the job runs "shots" times. Since quantum measurement is probabilistic, you need many samples
    # to build up a statistical picture of the output distribution. Each shot collapses the quantum
    # state to a classical bitstring and the sampler collects them
    start = time.perf_counter()
    job = sampler.run([transpiled], shots=shots)
    result = job.result()
    elapsed_ms = time.perf_counter() - start

    ### extract counts from bitarray ###
    pub_results = result[0]
    # counts is a dict matching observed bitstrings to how many times they occurred
    # Should have one answer dominating the counts for low noise hardware
    counts = pub_results.data.meas.get_counts()

    ### compare sim to real ###
    fidelity = None
    if ideal_statevector is not None:
        ideal_sim = AerSimulator(method='statevector')
        # remove measurements to get state vector
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        fidelity = float(state_fidelity(sv, ideal_sim))

    total_counts = sum(counts.values())
    correct_counts = counts.get(expected_key, 0)

    return BenchmarkResults(
        algorithm=algorithm_name,
        backend_type=backend_type,
        shots=shots,
        runtime_ms=elapsed_ms,
        counts=counts,
        circuit_depth=transpiled.depth(),
        gate_count=sum(transpiled.count_ops().values()),
        fidelity=fidelity,
        correct_result=(correct_counts / total_counts) > 0.9,
        metadata={
            'success_probability': correct_counts / total_counts,
            'top_result': max(correct_counts, key=correct_counts.get)
        }
    )
