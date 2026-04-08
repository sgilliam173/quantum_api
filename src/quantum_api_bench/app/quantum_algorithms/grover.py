from qiskit import QuantumCircuit
import numpy as np


def build_grover_oracle(target: str) -> QuantumCircuit:
    """
    Marks the target state with a phase flip.
    target: bitstring e.g. '101' (little-endian, LSB first)
    """
    n = len(target)
    oracle = QuantumCircuit(n)

    # Flip qubits where target bit is '0' so CZ sees all |1⟩
    for i, bit in enumerate(target):
        if bit == '0':
            oracle.x(i)

    # Multi-controlled Z (phase flip on |11...1⟩)
    oracle.h(n - 1)
    oracle.mcx(list(range(n - 1)), n - 1)  # Toffoli generalization
    oracle.h(n - 1)

    # Unflip
    for i, bit in enumerate(target):
        if bit == '0':
            oracle.x(i)

    return oracle


def build_diffuser(n: int) -> QuantumCircuit:
    """
    Grover diffuser: reflection about |s⟩ = H⊗n |0...0⟩
    Amplifies the marked state after each oracle call.
    """
    diffuser = QuantumCircuit(n)
    diffuser.h(range(n))
    diffuser.x(range(n))
    diffuser.h(n - 1)
    diffuser.mcx(list(range(n - 1)), n - 1)
    diffuser.h(n - 1)
    diffuser.x(range(n))
    diffuser.h(range(n))
    return diffuser


def grover(target: str) -> QuantumCircuit:
    """
    Full Grover circuit. Optimal iterations ≈ π/4 * sqrt(N).
    """
    n = len(target)
    N = 2 ** n
    iterations = max(1, round((np.pi / 4) * np.sqrt(N)))

    oracle = build_grover_oracle(target)
    diffuser = build_diffuser(n)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))  # uniform superposition

    for _ in range(iterations):
        qc.barrier()
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n), range(n))
    return qc