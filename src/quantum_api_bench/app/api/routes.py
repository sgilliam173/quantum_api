from fastapi import APIRouter, HTTPException
from .schemas import DJRequest, GroverRequest, BenchmarkResponse
from ..quantum_algorithms.deutsch_josza import deutsch_jozsa
from ..quantum_algorithms.grover import grover
from ..backends.backend import quantum_backend
from ..benchmarks.benchmark import run_benchmark
from qiskit_aer import AerSimulator

router = APIRouter(prefix="/api/v1", tags=["quantum"])


def _resolve_backend(backend_type: str):
    if backend_type == "ideal":
        return quantum_backend.get_simulator(), "ideal"
    elif backend_type == "noisy":
        return quantum_backend.get_noisy_simulator(), "noisy_sim"
    elif backend_type == "real":
        return quantum_backend.get_real_backend(), "real"
    else:
        raise HTTPException(400, f"Unknown backend: {backend_type}")


@router.post("/deutsch-jozsa", response_model=BenchmarkResponse)
def run_dj(req: DJRequest):
    try:
        qc = deutsch_jozsa(req.case, req.n_qubits)
        backend, btype = _resolve_backend(req.backend)
        # Constant → all zeros; balanced → non-zero
        expected = "0" * req.n_qubits if req.case in [1, 2] else None
        result = run_benchmark(
            qc, "deutsch-jozsa", expected or "1" * req.n_qubits,
            backend, btype, req.shots
        )
        return BenchmarkResponse(**result.__dict__)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/grover", response_model=BenchmarkResponse)
def run_grover(req: GroverRequest):
    try:
        qc = grover(req.target)
        backend, btype = _resolve_backend(req.backend)
        result = run_benchmark(
            qc, "grover", req.target, backend, btype, req.shots
        )
        return BenchmarkResponse(**result.__dict__)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/health")
def health():
    return {"status": "ok"}