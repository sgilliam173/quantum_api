import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from quantum_api_bench.app.quantum_algorithms.deutsch_josza import (
    deutsch_jozsa, build_dj_oracle
)
from quantum_api_bench.app.quantum_algorithms.grover import (
    grover, build_grover_oracle
)


class TestDeutschJozsa:

    def test_returns_quantum_circuit(self):
        qc = deutsch_jozsa(case=1)
        assert isinstance(qc, QuantumCircuit)

    def test_invalid_case_raises(self):
        with pytest.raises(ValueError):
            deutsch_jozsa(case=99)

    def test_circuit_has_measurements(self):
        qc = deutsch_jozsa(case=1)
        assert qc.num_clbits > 0

    @pytest.mark.parametrize("case", [1, 2, 3, 4])
    def test_all_cases_build_without_error(self, case):
        qc = deutsch_jozsa(case=case)
        assert qc is not None

    def test_oracle_is_unitary(self):
        """Quantum oracles must be unitary — verifies no irreversible ops."""
        for case in [1, 2, 3, 4]:
            oracle = build_dj_oracle(case)
            # Remove barriers before checking unitarity
            op = Operator(oracle)
            assert op.is_unitary(), f"Oracle for case {case} is not unitary"

    def test_circuit_width_matches_n_qubits(self):
        for n in [1, 2, 3]:
            qc = deutsch_jozsa(case=3, n_qubits=n)
            # n input qubits + 1 ancilla
            assert qc.num_qubits == n + 1


class TestGrover:

    def test_returns_quantum_circuit(self):
        qc = grover("11")
        assert isinstance(qc, QuantumCircuit)

    def test_circuit_width_matches_target(self):
        for target in ["1", "10", "101"]:
            qc = grover(target)
            assert qc.num_qubits == len(target)

    def test_oracle_is_unitary(self):
        for target in ["0", "1", "10", "11"]:
            oracle = build_grover_oracle(target)
            op = Operator(oracle)
            assert op.is_unitary()

    def test_circuit_has_measurements(self):
        qc = grover("11")
        assert qc.num_clbits > 0