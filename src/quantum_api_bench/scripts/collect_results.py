import json
from datetime import datetime
from quantum_api_bench.app.quantum_algorithms.deutsch_josza import deutsch_jozsa
from quantum_api_bench.app.quantum_algorithms.grover import grover
from quantum_api_bench.app.backends.backend import quantum_backend
from quantum_api_bench.app.benchmarks.benchmark import run_benchmark


SHOTS = 1024
RESULTS_FILE = "../../data_and_plots/benchmark_results.json"

# Define all experiments
# (algorithm, params, expected_key, label)
EXPERIMENTS = [
    ("dj", {"case": 1, "n_qubits": 3}, "000",  "dj_constant_3q"),
    ("dj", {"case": 3, "n_qubits": 3}, "111",  "dj_balanced_3q"),  # any non-zero answer
    ("dj", {"case": 3, "n_qubits": 4}, "1111", "dj_balanced_4q"),  # any non-zero answer
    ("grover", {"target": "11"},   "11",   "grover_2q"),
    ("grover", {"target": "101"},  "101",  "grover_3q"),
    ("grover", {"target": "1011"}, "1011", "grover_4q"),
]

results = {}


def get_circuit(algorithm, params):
    if algorithm == "dj":
        return deutsch_jozsa(**params)
    elif algorithm == "grover":
        return grover(**params)


def run_on_simulator(noisy=False):
    label_suffix = "noisy" if noisy else "ideal"
    print(f"\n=== Running {label_suffix} simulator ===")
    if noisy:
        backend = quantum_backend.get_noisy_simulator()  # IBM calibrated noise
    else:
        backend = quantum_backend.get_simulator()

    for algo, params, expected_key, label in EXPERIMENTS:
        print(f"  {label}...")
        qc = get_circuit(algo, params)
        # For balanced DJ, just check it's not all zeros
        key = expected_key if expected_key else "0" * qc.num_clbits
        result = run_benchmark(qc, algo, key, backend,
                               label_suffix, shots=SHOTS)
        results[f"{label}_{label_suffix}"] = {
            "success_probability": result.metadata["success_probability"],
            "correct_result": result.correct_result,
            "circuit_depth": result.circuit_depth,
            "gate_count": result.gate_count,
            "runtime_ms": result.runtime_ms,
            "counts": result.counts,
            "top_result": result.metadata["top_result"],
        }
        print(f"    success_prob={result.metadata['success_probability']:.3f} "
              f"depth={result.circuit_depth}")


def run_on_real_hardware():
    print(f"\n=== Submitting to real hardware ===")
    backend = quantum_backend.get_real_backend()

    for algo, params, expected_key, label in EXPERIMENTS:
        print(f"  {label}...")
        qc = get_circuit(algo, params)
        key = expected_key if expected_key else "0" * qc.num_clbits
        result = run_benchmark(
            qc, algo, key, backend, "real", shots=SHOTS
        )
        results[f"{label}_real"] = {
            "success_probability": result.metadata["success_probability"],
            "correct_result": result.correct_result,
            "circuit_depth": result.circuit_depth,
            "gate_count": result.gate_count,
            "runtime_ms": result.runtime_ms,
            "counts": result.counts,
            "top_result": result.metadata["top_result"],
        }
        print(f"    top={result.metadata['top_result']} "
              f"success_prob={result.metadata['success_probability']:.3f}")


if __name__ == "__main__":
    print(f"Starting benchmark collection at {datetime.now()}")

    run_on_simulator(noisy=False)
    run_on_simulator(noisy=True)

    use_real = input("\nRun on real hardware? (y/n): ").strip().lower()
    if use_real == 'y':
        run_on_real_hardware()

    # Save all results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")