import time
import math
from dataclasses import dataclass
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.primitives import BackendSamplerV2
from qiskit_ibm_runtime import SamplerV2 as IBMSampler
from qiskit.quantum_info import Statevector, partial_trace


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
    threshold = 0.9 if backend_type == "ideal" else 0.7

    # translates my circuit into the native gate set supported by the current IBM hardware
    # this is the circuit that gets actually executed
    transpiled = transpile(qc, backend=backend, optimization_level=1)

    ### NOTE: Both of the depth and gate count increase after transpilation ###
    # Number of layers of gates. Tells us how long the computation takes relative to coherence time
    # A deeper circuit is going to be more exposed to noise because the qubits have more time

    start = time.perf_counter()
    if backend_type == "real":
        sampler = IBMSampler(mode=backend)
    else:
        sampler = BackendSamplerV2(backend=backend)

    # the job runs "shots" times. Since quantum measurement is probabilistic, you need many samples
    # to build up a statistical picture of the output distribution. Each shot collapses the quantum
    # state to a classical bitstring and the sampler collects them
    job = sampler.run([transpiled], shots=shots)
    result = job.result()
    elapsed_ms = time.perf_counter() - start

    ### extract counts from bitarray ###
    pub_results = result[0]
    # counts is a dict matching observed bitstrings to how many times they occurred
    # Should have one answer dominating the counts for low noise hardware
    counts = pub_results.data.meas.get_counts()

    ### compare sim to real ###
    if backend_type == "ideal":
        fidelity = 1.0
    else:
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv_ideal = Statevector.from_instruction(qc_no_meas)

        n_measured = qc.num_clbits
        n_total = qc.num_qubits
        n_ancilla = n_total - n_measured

        if n_ancilla > 0:
            # Trace out ancilla qubit(s) — DJ has 1 ancilla, Grover has none
            dm = partial_trace(sv_ideal, [n_total - 1])
            ideal_probs_raw = {
                format(i, f'0{n_measured}b'): float(dm.data[i, i].real)
                for i in range(2 ** n_measured)
            }
        else:
            # No ancilla — use statevector probabilities directly
            ideal_probs_raw = {
                format(i, f'0{n_measured}b'): float(p)
                for i, p in enumerate(sv_ideal.probabilities())
            }

        # Reverse bit order to match measurement convention
        ideal_probs = {k[::-1]: v for k, v in ideal_probs_raw.items()}

        total = sum(counts.values())
        measured_probs = {k: v / total for k, v in counts.items()}

        fidelity = sum(
            math.sqrt(ideal_probs.get(k, 0) * measured_probs.get(k, 0))
            for k in set(ideal_probs) | set(measured_probs)
        )

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
        correct_result=(correct_counts / total_counts) > threshold,
        metadata={
            'success_probability': correct_counts / total_counts,
            'top_result': max(counts, key=counts.get)
        }
    )
