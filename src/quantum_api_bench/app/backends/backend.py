from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class QuantumBackend:
    use_real_hardware: bool = False
    ibm_backend_name: str = "ibm_brisbane"
    _service: Optional[QiskitRuntimeService] = field(default=None, repr=False)
    _noise_model: Optional[NoiseModel] = field(default=None, repr=False)

    def get_simulator(self) -> AerSimulator:
        """Noiseless simulator."""
        return AerSimulator()

    def get_noisy_simulator(self) -> AerSimulator:
        """Simulator with noise model pulled from real backend."""
        noise = self._get_noise_model()
        return AerSimulator(noise_model=noise)

    def get_real_backend(self):
        """Returns real IBM backend. Requires saved credentials."""
        svc = self._get_service()
        return svc.backend(self.ibm_backend_name)

    def get_sampler(self, noisy: bool = False, real: bool = False) -> Sampler:
        if real:
            backend = self.get_real_backend()
            return Sampler(mode=backend)
        elif noisy:
            return BackendSamplerV2(backend=self.get_noisy_simulator())
        else:
            return BackendSamplerV2(backend=self.get_simulator())

    def _get_service(self) -> QiskitRuntimeService:
        if self._service is None:
            token = os.getenv("IBM_QUANTUM_TOKEN")
            if not token:
                raise EnvironmentError(
                    "IBM_QUANTUM_TOKEN not set. "
                    "Run: export IBM_QUANTUM_TOKEN=your_token"
                )
            QiskitRuntimeService.save_account(
                channel="ibm_quantum", token=token, overwrite=True
            )
            self._service = QiskitRuntimeService(channel="ibm_quantum")
        return self._service

    def _get_noise_model(self) -> NoiseModel:
        if self._noise_model is None:
            backend = self.get_real_backend()
            self._noise_model = NoiseModel.from_backend(backend)
        return self._noise_model


# import this everywhere
quantum_backend = QuantumBackend()