from fastapi import FastAPI
from quantum_api_bench.app.api.routes import router

app = FastAPI(
    title="Quantum API Bench",
    description="Benchmarking quantum algorithms across simulators and IBM hardware",
    version="0.1.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)