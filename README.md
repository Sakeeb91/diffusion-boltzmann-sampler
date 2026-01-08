# Diffusion Models as Boltzmann Samplers

A neural approach to statistical mechanics sampling using score-based diffusion models. This project trains diffusion models to sample from Boltzmann distributions, leveraging the insight that the score function equals the physical force field.

## Overview

Traditional Markov Chain Monte Carlo (MCMC) methods for sampling Boltzmann distributions `p(x) ∝ exp(-E(x)/kT)` are often slow to converge, especially near phase transitions. This project implements score-based diffusion models as fast neural samplers, providing:

- **Faster sampling** than traditional MCMC after training
- **Temperature generalization** from a single trained model
- **Physical interpretability** where the learned score function represents forces
- **Interactive visualization** of the sampling process

## Target Systems

1. **2D Ising Model** - Lattice spin system with well-understood phase transitions
2. **Lennard-Jones Fluid** - Continuous particle system for molecular dynamics

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         System Architecture                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Frontend   │    │   Backend    │    │      ML Engine           │  │
│  │   (React)    │◄──►│  (FastAPI)   │◄──►│      (PyTorch)          │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│         │                   │                        │                  │
│         ▼                   ▼                        ▼                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │ Plotly/D3.js │    │  WebSocket   │    │  Score Network (U-Net)   │  │
│  │ Animations   │    │  Streaming   │    │  Diffusion Scheduler     │  │
│  │ Controls     │    │  REST API    │    │  MCMC Baseline           │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Score-Based Diffusion

The score function `s(x) = ∇log p(x)` for a Boltzmann distribution is:

```
s(x) = -∇E(x) / kT
```

This is exactly the **force field** divided by temperature. The diffusion model learns to denoise samples by following this force field backward in time.

### Forward Process (Noising)

```
dx = σ(t) dW
```

Add Gaussian noise progressively, turning any distribution into pure noise.

### Reverse Process (Denoising)

```
dx = -σ(t)² s_θ(x,t) dt + σ(t) dW
```

Neural network `s_θ` learns to reverse the noising, generating samples from the target distribution.

## Project Structure

```
diffusion-boltzmann-sampler/
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── sampling.py      # Sampling endpoints
│   │   │   ├── training.py      # Training endpoints
│   │   │   └── analysis.py      # Analysis endpoints
│   │   └── main.py              # FastAPI app
│   ├── ml/
│   │   ├── models/
│   │   │   ├── score_network.py # Score function neural network
│   │   │   └── diffusion.py     # Diffusion process
│   │   ├── systems/
│   │   │   ├── ising.py         # 2D Ising model
│   │   │   └── lennard_jones.py # Lennard-Jones fluid
│   │   ├── samplers/
│   │   │   ├── mcmc.py          # Baseline MCMC sampler
│   │   │   └── diffusion.py     # Diffusion sampler
│   │   └── training/
│   │       └── trainer.py       # Training loop
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── IsingVisualizer.tsx
│   │   │   ├── DiffusionAnimation.tsx
│   │   │   ├── CorrelationPlot.tsx
│   │   │   └── ControlPanel.tsx
│   │   ├── hooks/
│   │   ├── services/
│   │   └── App.tsx
│   └── package.json
├── docs/
│   └── IMPLEMENTATION_PLAN.md
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Sakeeb91/diffusion-boltzmann-sampler.git
cd diffusion-boltzmann-sampler

# Backend setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
```

### Running the Application

```bash
# Terminal 1: Start backend
cd backend
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Visit `http://localhost:5173` for the interactive interface.

## Features

### Interactive Visualizations

- **Lattice Viewer**: Real-time Ising model spin configuration
- **Diffusion Animation**: Watch the denoising process step-by-step
- **Energy Landscape**: 2D projection of the energy surface
- **Correlation Functions**: Compare neural sampler vs MCMC

### Parameter Controls

- Temperature slider with phase transition indicator
- Lattice size selection (8x8 to 64x64)
- Diffusion steps control
- MCMC comparison toggle

## Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| ML Framework | PyTorch | Industry standard, excellent autodiff |
| Backend | FastAPI | Async support, automatic OpenAPI docs |
| Frontend | React + TypeScript | Type safety, component reusability |
| Visualization | Plotly.js | Interactive plots, animation support |
| State Management | Zustand | Lightweight, minimal boilerplate |
| Styling | Tailwind CSS | Rapid prototyping, consistent design |

## Mathematical Background

### Ising Model Energy

```
E(s) = -J Σ_{<i,j>} s_i s_j - h Σ_i s_i
```

Where `s_i ∈ {-1, +1}`, `J` is coupling strength, `h` is external field.

### Lennard-Jones Potential

```
U(r) = 4ε [(σ/r)^12 - (σ/r)^6]
```

Total energy: `E = Σ_{i<j} U(|r_i - r_j|)`

### Score Matching Loss

```
L(θ) = E_t E_{x(t)} [ ||s_θ(x(t), t) - ∇log p(x(t))||² ]
```

For known energy functions, the target score is computable analytically.

## Validation

The neural sampler is validated against gold-standard Metropolis-Hastings MCMC by comparing:

1. **Magnetization distribution** `P(M)` for Ising model
2. **Radial distribution function** `g(r)` for Lennard-Jones
3. **Autocorrelation times** (neural should be much faster)
4. **Energy histograms** at various temperatures

## Contributing

Contributions are welcome. Please see the GitHub Issues for current tasks and the implementation plan in `docs/IMPLEMENTATION_PLAN.md`.

## License

MIT License

## References

- Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution
- Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models
- Noe, F., et al. (2019). Boltzmann Generators
- Newman, M. E. J., & Barkema, G. T. (1999). Monte Carlo Methods in Statistical Physics
