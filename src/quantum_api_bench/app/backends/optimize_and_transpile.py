from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from quantum_api_bench.app.backends.backend import Backend


target = Backend.backend_name.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)

qc_isa = pm.run(qc)