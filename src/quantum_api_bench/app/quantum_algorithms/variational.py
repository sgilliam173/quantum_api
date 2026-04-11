from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

def build_variational_circuit(n_qubits: int, depth: int = 2) -> QuantumCircuit:
    """
    Hardware-efficient ansatz: alternating Ry rotations and CNOT entanglers.
    Parameters: theta[i] are variational angles (optimized classically).
    """
    theta = ParameterVector("θ", length=n_qubits * depth)
    qc = QuantumCircuit(n_qubits, n_qubits)

    idx = 0
    for layer in range(depth):
        # Rotation layer
        for q in range(n_qubits):
            qc.ry(theta[idx], q)
            idx += 1
        # Entanglement layer (skip on last depth if desired)
        if layer < depth - 1:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc