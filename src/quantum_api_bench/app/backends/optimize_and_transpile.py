from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from quantum_api_bench.app.backends.backend import quantum_backend


def transpile_for_backend(qc: QuantumCircuit, use_real: bool = False):
    """
    Transpiles a circuit for the target backend at optimization level 3.
    """
    backend = quantum_backend.get_real_backend() if use_real \
              else quantum_backend.get_simulator()
    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
    return pm.run(qc)