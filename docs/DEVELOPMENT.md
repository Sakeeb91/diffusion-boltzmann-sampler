# Development Guide

This guide covers setting up and working with the Diffusion Boltzmann Sampler codebase.

## Prerequisites

- Python 3.9+
- Node.js 18+
- Git

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/Sakeeb91/diffusion-boltzmann-sampler.git
cd diffusion-boltzmann-sampler

# Backend setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
cd ..
```

## Running the Application

### Backend

```bash
# From project root, with venv activated
cd backend
uvicorn api.main:app --reload --port 8000
```

The API will be available at:
- REST API: http://localhost:8000
- OpenAPI docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws/sample

### Frontend

```bash
# From project root
cd frontend
npm run dev
```

The frontend will be available at http://localhost:5173

## Project Structure

```
diffusion-boltzmann-sampler/
├── backend/
│   ├── api/              # FastAPI application
│   │   ├── main.py       # App entry point
│   │   ├── routes/       # API endpoints
│   │   ├── models.py     # Pydantic models
│   │   └── middleware.py # Custom middleware
│   ├── ml/               # Machine learning code
│   │   ├── systems/      # Physical systems (Ising, etc.)
│   │   ├── models/       # Neural network models
│   │   ├── samplers/     # Sampling algorithms
│   │   └── training/     # Training loop
│   ├── tests/            # Backend tests
│   ├── config.py         # Configuration
│   └── utils.py          # Utilities
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── hooks/        # Custom hooks
│   │   ├── services/     # API client
│   │   ├── store/        # Zustand store
│   │   └── config/       # Environment config
│   └── package.json
├── docs/                 # Documentation
├── requirements.txt      # Python dependencies
└── pytest.ini           # Test configuration
```

## Development Workflow

### Running Tests

**Backend:**
```bash
# All tests
pytest

# Specific file
pytest backend/tests/test_ising.py

# With coverage
pytest --cov=backend --cov-report=html
```

**Frontend:**
```bash
cd frontend

# Watch mode
npm test

# Single run
npm run test:run

# With coverage
npm run test:coverage
```

### Code Quality

**Backend:**
```bash
# Linting
ruff check backend/

# Type checking
mypy backend/ --ignore-missing-imports

# Formatting
black backend/
isort backend/
```

**Frontend:**
```bash
cd frontend

# Linting
npm run lint

# Type checking
npx tsc --noEmit
```

### Adding a New Endpoint

1. Create the route in `backend/api/routes/`
2. Add Pydantic models to `backend/api/models.py`
3. Register the router in `backend/api/main.py`
4. Add tests in `backend/tests/`

Example:
```python
# backend/api/routes/myroute.py
from fastapi import APIRouter
from ..models import MyRequest, MyResponse

router = APIRouter()

@router.post("/myendpoint", response_model=MyResponse)
async def my_endpoint(request: MyRequest):
    return MyResponse(...)
```

### Adding a New Component

1. Create component in `frontend/src/components/`
2. Export from `frontend/src/components/index.ts`
3. Add tests in same directory as `Component.test.tsx`

Example:
```typescript
// frontend/src/components/MyComponent.tsx
export function MyComponent({ prop }: { prop: string }) {
  return <div>{prop}</div>;
}
```

## Environment Variables

### Backend

Create `.env` in project root:
```env
DEBUG=true
DEFAULT_LATTICE_SIZE=32
DEFAULT_TEMPERATURE=2.27
```

### Frontend

Create `frontend/.env`:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_DEFAULT_LATTICE_SIZE=32
```

See `frontend/.env.example` for all options.

## Common Issues

### CORS Errors

If you see CORS errors, ensure:
1. Backend is running on port 8000
2. Frontend is running on port 5173
3. The backend CORS settings include your frontend URL

### WebSocket Connection Failed

1. Check backend is running
2. Ensure no firewall blocking WebSocket
3. Check browser console for specific errors

### Import Errors (Python)

Run from project root with:
```bash
cd backend
uvicorn api.main:app --reload
```

Not:
```bash
cd backend/api
python main.py  # Wrong!
```

## Contributing

1. Create a feature branch from `main`
2. Make changes with atomic commits
3. Ensure all tests pass
4. Submit a pull request

Commit message format:
```
type(scope): description

feat(api): add new endpoint
fix(frontend): resolve state issue
test(backend): add Ising model tests
docs: update README
```
