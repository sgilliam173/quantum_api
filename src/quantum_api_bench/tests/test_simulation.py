import pytest
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2
from quantum_api_bench.app.quantum_algorithms.deutsch_josza import deutsch_jozsa
from quantum_api_bench.app.quantum_algorithms.grover import grover


SHOTS = 2048
simulator = AerSimulator()
sampler = BackendSamplerV2(backend=simulator)


def get_counts(qc, shots=SHOTS):
    """Helper: run a circuit and return counts dict."""
    job = sampler.run([qc], shots=shots)
    return job.result()[0].data.meas.get_counts()


class TestDeutschJozsaSimulation:
    """
    Deutsch-Jozsa is deterministic — it should ALWAYS give the
    correct answer with 100% probability on an ideal simulator.
    """

    @pytest.mark.parametrize("case", [1, 2])
    def test_constant_function_returns_all_zeros(self, case):
        """Constant functions must measure all zeros."""
        qc = deutsch_jozsa(case=case, n_qubits=2)
        counts = get_counts(qc)
        # All shots should give "00"
        total = sum(counts.values())
        assert counts.get("00", 0) / total > 0.99, (
            f"Case {case} should be constant (all zeros), got {counts}"
        )

    @pytest.mark.parametrize("case", [3, 4])
    def test_balanced_function_never_returns_all_zeros(self, case):
        """Balanced functions must NEVER measure all zeros."""
        qc = deutsch_jozsa(case=case, n_qubits=2)
        counts = get_counts(qc)
        total = sum(counts.values())
        zero_fraction = counts.get("00", 0) / total
        assert zero_fraction < 0.01, (
            f"Case {case} should be balanced (no zeros), got {counts}"
        )

    def test_scales_to_more_qubits(self):
        """Algorithm should work for n > 1 input qubits."""
        for n in [2, 3, 4]:
            qc = deutsch_jozsa(case=1, n_qubits=n)
            counts = get_counts(qc)
            total = sum(counts.values())
            expected_zero = "0" * n
            assert counts.get(expected_zero, 0) / total > 0.99


class TestGroverSimulation:
    """
    Grover's is probabilistic but should succeed with high probability
    after the optimal number of iterations.
    """

    @pytest.mark.parametrize("target", ["00", "01", "10", "11"])
    def test_finds_target(self, target):
        qc = grover(target)
        counts = get_counts(qc)
        total = sum(counts.values())
        success_prob = counts.get(target, 0) / total
        assert success_prob > 0.8, (
            f"Grover failed to find '{target}': success prob = {success_prob:.2f}, counts = {counts}"
        )

    def test_three_qubit_grover(self):
        """Harder search space — still should succeed."""
        target = "101"
        qc = grover(target)
        counts = get_counts(qc)
        total = sum(counts.values())
        assert counts.get(target, 0) / total > 0.7