from pydantic import BaseModel
from typing import Literal, Optional


class DJRequest(BaseModel):
    case: int                          # 1-4
    n_qubits: int = 1
    backend: Literal["ideal", "noisy", "real"] = "ideal"
    shots: int = 1024


class GroverRequest(BaseModel):
    target: str                        # e.g. "101"
    backend: Literal["ideal", "noisy", "real"] = "ideal"
    shots: int = 1024


class BenchmarkResponse(BaseModel):
    algorithm: str
    backend_type: str
    shots: int
    runtime_ms: float
    counts: dict
    circuit_depth: int
    gate_count: int
    fidelity: Optional[float]
    correct_result: bool
    metadata: dict