"""FastAPI application for Diffusion Boltzmann Sampler."""

from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .routes import sampling_router, training_router, analysis_router
from .routes.lj_sampling import router as lj_router
from .websocket import sample_websocket_handler
from ..ml.systems.ising import IsingModel
from ..ml.models.score_network import ScoreNetwork
from ..ml.models.diffusion import DiffusionProcess


# Global state for ML models
class AppState:
    """Application state holding ML models."""

    def __init__(self):
        self.device = "cpu"
        self.ising_model: IsingModel = None
        self.score_network: ScoreNetwork = None
        self.diffusion: DiffusionProcess = None
        self.is_training = False
        self.training_progress = 0.0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    # Initialize default models
    state.ising_model = IsingModel(size=32)
    state.diffusion = DiffusionProcess()
    state.score_network = ScoreNetwork(
        in_channels=1,
        base_channels=32,
        time_embed_dim=64,
        num_blocks=3,
    ).to(state.device)

    print("Models initialized")
    yield

    # Cleanup
    print("Shutting down")


app = FastAPI(
    title="Diffusion Boltzmann Sampler",
    description="Neural sampling from Boltzmann distributions using score-based diffusion models",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sampling_router, prefix="/sample", tags=["Sampling"])
app.include_router(training_router, prefix="/training", tags=["Training"])
app.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])
app.include_router(lj_router, prefix="/sample", tags=["Lennard-Jones"])


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with version info.

    Returns:
        Health status including API version and available features.
    """
    return {
        "status": "healthy",
        "version": app.version,
        "title": app.title,
        "features": {
            "mcmc_sampling": True,
            "diffusion_sampling": True,
            "websocket_streaming": True,
            "lennard_jones": True,
        },
    }


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get current configuration."""
    return {
        "lattice_size": state.ising_model.size if state.ising_model else 32,
        "T_critical": IsingModel.T_CRITICAL,
        "device": state.device,
        "is_training": state.is_training,
        "training_progress": state.training_progress,
    }


@app.websocket("/ws/sample")
async def websocket_sample(websocket: WebSocket):
    """WebSocket endpoint for streaming sampling results.

    Expects JSON message with:
    - temperature: float
    - lattice_size: int
    - sampler: "mcmc" or "diffusion"
    - num_steps: int
    """
    await sample_websocket_handler(websocket, state)


def get_state() -> AppState:
    """Get global app state (for dependency injection in routes)."""
    return state


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
