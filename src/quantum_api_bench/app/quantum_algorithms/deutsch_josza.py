from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

def build_dj_oracle(case: int, n_qubits: int = 1) -> QuantumCircuit:
    """
    Builds the oracle for Deutsch-Jozsa.
    n_qubits: number of INPUT qubits (ancilla is separate)
    case 1: constant f(x) = 0
    case 2: constant f(x) = 1
    case 3: balanced (CNOT-based)
    case 4: balanced with X flip
    """
    if case not in [1, 2, 3, 4]:
        raise ValueError('case must be in [1, 2, 3, 4]')

    # n_qubits input + 1 ancilla
    oracle = QuantumCircuit(n_qubits + 1)

    if case == 1:
        pass  # constant 0: do nothing
    elif case == 2:
        oracle.x(n_qubits)  # constant 1: flip ancilla
    elif case == 3:
        for q in range(n_qubits):
            oracle.cx(q, n_qubits)  # balanced: CNOT each input to ancilla
    elif case == 4:
        for q in range(n_qubits):
            oracle.cx(q, n_qubits)
        oracle.x(n_qubits)  # balanced with phase kick

    return oracle


def deutsch_jozsa(case: int, n_qubits: int = 1) -> QuantumCircuit:
    """
    Full Deutsch-Jozsa circuit.
    Returns circuit — caller handles execution.
    """
    oracle = build_dj_oracle(case, n_qubits)
    total = n_qubits + 1

    qr = QuantumRegister(total)
    cr = ClassicalRegister(n_qubits, name='meas')  # explicit name
    qc = QuantumCircuit(qr, cr)

    qc.x(n_qubits)
    qc.h(range(total))
    qc.barrier()
    qc.compose(oracle, inplace=True)
    qc.barrier()
    qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    return qc