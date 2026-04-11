import pytest
from qiskit_aer import AerSimulator
from quantum_api_bench.app.benchmarks.benchmark import run_benchmark, BenchmarkResults
from quantum_api_bench.app.quantum_algorithms.deutsch_josza import deutsch_jozsa
from quantum_api_bench.app.quantum_algorithms.grover import grover


@pytest.fixture
def simulator():
    return AerSimulator()


class TestBenchmarkPipeline:

    def test_returns_benchmark_result(self, simulator):
        qc = deutsch_jozsa(case=1)
        result = run_benchmark(qc, "dj", "00", simulator, "ideal")
        assert isinstance(result, BenchmarkResults)

    def test_runtime_is_positive(self, simulator):
        qc = deutsch_jozsa(case=1)
        result = run_benchmark(qc, "dj", "00", simulator, "ideal")
        assert result.runtime_ms > 0

    def test_circuit_depth_is_positive(self, simulator):
        qc = deutsch_jozsa(case=1)
        result = run_benchmark(qc, "dj", "00", simulator, "ideal")
        assert result.circuit_depth > 0

    def test_counts_sum_to_shots(self, simulator):
        shots = 512
        qc = deutsch_jozsa(case=1)
        result = run_benchmark(qc, "dj", "00", simulator, "ideal", shots=shots)
        assert sum(result.counts.values()) == shots

    def test_dj_constant_passes_correctness_check(self, simulator):
        qc = deutsch_jozsa(case=1, n_qubits=2)
        result = run_benchmark(qc, "dj", "00", simulator, "ideal")
        assert result.correct_result is True

    def test_grover_passes_correctness_check(self, simulator):
        qc = grover("11")
        result = run_benchmark(qc, "grover", "11", simulator, "ideal")
        assert result.correct_result is True

    def test_metadata_contains_success_probability(self, simulator):
        qc = deutsch_jozsa(case=1)
        result = run_benchmark(qc, "dj", "00", simulator, "ideal")
        assert "success_probability" in result.metadata
        assert 0.0 <= result.metadata["success_probability"] <= 1.0