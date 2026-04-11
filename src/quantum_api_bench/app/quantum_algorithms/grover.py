import numpy as np
import math
from qiskit import QuantumCircuit, ClassicalRegister



def build_grover_oracle(target: str) -> QuantumCircuit:
    """
    Marks the target state with a phase flip (-1).
    Uses the standard approach: flip qubits where target=0,
    apply multi-controlled Z, then unflip.
    """
    n = len(target)
    oracle = QuantumCircuit(n)

    # Flip qubits where target bit is '0'
    for i, bit in enumerate(target):
        if bit == '0':
            oracle.x(i)

    # Apply multi-controlled Z as phase flip on |11...1>
    if n == 1:
        # Single qubit: Z gate flips phase of |1>
        oracle.z(0)
    elif n == 2:
        # Two qubits: CZ gate
        oracle.cz(0, 1)
    else:
        # n > 2: decompose as H + MCX + H on last qubit
        oracle.h(n - 1)
        oracle.mcx(list(range(n - 1)), n - 1)
        oracle.h(n - 1)

    # Unflip qubits
    for i, bit in enumerate(target):
        if bit == '0':
            oracle.x(i)

    return oracle


def build_diffuser(n: int) -> QuantumCircuit:
    """
    Grover diffuser: 2|s><s| - I
    Reflects about the uniform superposition state.
    """
    diffuser = QuantumCircuit(n)
    diffuser.h(range(n))
    diffuser.x(range(n))

    # Multi-controlled Z
    if n == 1:
        diffuser.z(0)
    elif n == 2:
        diffuser.cz(0, 1)
    else:
        diffuser.h(n - 1)
        diffuser.mcx(list(range(n - 1)), n - 1)
        diffuser.h(n - 1)

    diffuser.x(range(n))
    diffuser.h(range(n))
    return diffuser


def grover(target: str) -> QuantumCircuit:
    n = len(target)
    N = 2 ** n
    reversed_target = target[::-1]

    qc = QuantumCircuit(n)
    qc.add_register(ClassicalRegister(n, name='meas'))

    # Special case: 1 qubit, just prepare the target state directly
    if n == 1:
        if target == '1':
            qc.x(0)  # flip to |1> if target is 1
        qc.measure(0, 0)
        return qc

    iterations = max(1, math.floor((np.pi / 4) * np.sqrt(N)))
    oracle = build_grover_oracle(reversed_target)
    diffuser = build_diffuser(n)

    qc.h(range(n))
    for _ in range(iterations):
        qc.barrier()
        qc.compose(oracle, qubits=list(range(n)), inplace=True)
        qc.compose(diffuser, qubits=list(range(n)), inplace=True)

    qc.measure(range(n), range(n))
    return qc