# NeuralCanvas WASM

Interactive neural network visualizer with local browser inference via WebAssembly (Rust) and a React UI.

Italian version: `README-IT.md`

## Project vision

NeuralCanvas WASM is built around two goals:

- make the neural network forward pass visible in real time;
- demonstrate a modern frontend + Rust/WASM pipeline for performance-oriented use cases.

The user draws a digit on a canvas, the tensor is normalized to 28x28 (MNIST-style), sent to the Rust/WASM engine, and the frontend displays:

- predicted class;
- confidence;
- internal layer activations;
- dynamic neuron/synapse visualization.

## Tech stack

- Frontend: React + TypeScript + Vite + Tailwind CSS
- Inference engine: Rust + `wasm-bindgen` + `serde`
- Backend API: Rust + Axum + SQLx + PostgreSQL
- Local runtime: browser (client-side inference)
- Containerization: DevContainer + Docker multi-stage

## Why WASM + React

- **Performance**: numeric forward-pass computation runs in Rust, reducing overhead vs JS-only implementations.
- **UI responsiveness**: React handles rendering and interaction, while CPU-bound logic is delegated to WASM.
- **Portability**: once compiled, the module runs on modern browsers without plugins.
- **Separation of concerns**: mathematical domain in Rust, UX/presentation in React.

## Architecture

```text
Drawing Canvas (React)
   -> preprocessing 28x28 + normalization [0..1]
   -> Float32Array(784)
   -> NeuralEngine (Rust -> WASM)
        -> Dense + ReLU + Dense + ReLU + Dense + Softmax
        -> predicted_label + probabilities + layer_activations
   -> NetworkVisualizer (React)
```

### Main modules

- `frontend/src/DrawingCanvas.tsx`: input capture, bounding box, MNIST-style resize, normalization.
- `frontend/src/App.tsx`: WASM bootstrap, backend model fetch, inference orchestration, data collection.
- `frontend/src/NetworkVisualizer.tsx`: dynamic rendering of layers, neurons, and synapses.
- `wasm-engine/src/lib.rs`: math core + exported WASM `NeuralEngine` API.
- `backend/src/main.rs`: backend API (health, model asset provider, dataset management).

### Backend API quick reference

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | No | Backend health check + model version |
| `GET` | `/api/model/:version` | No | Serves model JSON requested by frontend |
| `POST` | `/api/datasets/labels` | Admin | Stores one labeled sample |
| `GET` | `/api/datasets/stats` | Admin | Returns dataset metrics (total + counts by label) |
| `DELETE` | `/api/datasets/labels` | Admin | Deletes all samples |
| `DELETE` | `/api/datasets/labels/:label` | Admin | Deletes all samples for one label |
| `DELETE` | `/api/datasets/samples/:sample_id` | Admin | One-shot undo for last saved sample |

Notes:

- `Admin` means `Authorization: Basic ...` with `ADMIN_USERNAME` and `ADMIN_PASSWORD`.
- `GET /api/model/:version` remains public so inference can work without login.

## Repository structure

```text
.
├── frontend/         # React app + WASM integration
├── wasm-engine/      # Rust crate compiled to WebAssembly
├── backend/          # Rust API + PostgreSQL integration
├── Dockerfile        # Production multi-stage build
└── docker-compose.yml
```

## Requirements

### Option A - local development

- Node.js 20+
- npm
- Rust toolchain (stable)
- `wasm-pack`

Install `wasm-pack`:

```bash
cargo install wasm-pack
```

### Option B - DevContainer (recommended)

Only Docker is required on the host. Open the repository in Cursor/VS Code and select "Reopen in Container".

## Running locally (host)

1. Clone and prepare environment:

```bash
cp .env.example .env
```

2. Build the WASM module where the frontend imports it:

```bash
cd wasm-engine
wasm-pack build --target web --out-dir ../frontend/src/wasm
```

3. Start frontend:

```bash
cd ../frontend
npm install
npm run dev
```

App is available at `http://localhost:5173` (default Vite port).

## Running with Docker

To run the composed infrastructure:

```bash
cp .env.example .env
docker compose up --build
```

Declared services:

- frontend (Vite dev server, hot reload)
- backend (Axum API + PostgreSQL)
- postgres

With stack running:

- frontend: `http://localhost:3000`
- backend health: `http://localhost:8080/health`
- backend model API: `http://localhost:8080/api/model/v1.0`

## Label manager and authentication

The UI does not require login at startup. Users can draw and run inference immediately.

For dataset operations (labeling/metrics/delete/undo):

- open `Manage Labels & Metrics`;
- if unauthenticated, an admin login modal appears;
- valid credentials open the label manager panel.

Panel operations:

- `Save Sample` stores current sample (disabled on empty canvas);
- `Undo last label` undoes only the latest saved labeling operation;
- `Delete label X` removes all samples for a given label;
- `Delete ALL samples` removes the full dataset (destructive action with dedicated confirmation).

## Production build

Multi-stage build (Rust WASM -> frontend bundle -> NGINX):

```bash
docker build -t neuralcanvas-wasm .
docker run --rm -p 8080:80 neuralcanvas-wasm
```

## Testing

The Rust crate currently includes:

- **Unit tests** in `wasm-engine/src/lib.rs` for core math:
  - `dense_forward` (matmul + bias)
  - `relu`
  - `softmax` (including numerical stability with large values)
  - deterministic pipeline (`dense -> relu -> dense -> softmax`)
  - negative cases (invalid shapes, empty logits)
- **Integration tests** in `wasm-engine/tests/neural_engine_integration.rs`:
  - valid model loading
  - end-to-end prediction on reduced model
  - error handling for invalid input size and inconsistent model shape
- **WASM API tests** in `wasm-engine/tests/wasm_api.rs` (target `wasm32`):
  - `NeuralEngine` construction via `wasm-bindgen` API
  - `predict` serialized payload validation
  - error handling for invalid bridge-side input

Run tests:

```bash
cd wasm-engine
cargo test
```

Run WASM tests (Node.js):

```bash
cd wasm-engine
wasm-pack test --node
```

## Environment variables

Defined in `.env.example`:

- backend/runtime:
  - `PORT`
  - `DATABASE_TARGET`
  - `DATABASE_URL_LOCAL`
  - `DATABASE_URL_DOCKER`
  - `MODEL_VERSION`
  - `MODEL_ASSET_PATH`
  - `ADMIN_USERNAME`
  - `ADMIN_PASSWORD`
  - `POSTGRES_USER`
  - `POSTGRES_PASSWORD`
  - `POSTGRES_DB`
- frontend:
  - `VITE_API_BASE_URL`
  - `VITE_MODEL_VERSION`
  - `VITE_TELEMETRY_ENABLED`

## Project status and roadmap

### Completed

- local WASM inference
- interactive canvas + 28x28 preprocessing
- neural internal state visualization
- unit/integration/WASM tests on Rust crate
- CI pipeline with lint + test + WASM/frontend build
- Rust backend + PostgreSQL (health, model asset provider, persistent data collector)
- frontend-backend integration (`/api/model/:version`, `/api/datasets/labels`)
- admin auth for dataset APIs + on-demand protected UI panel
- label manager with bulk/per-label delete and one-shot undo

### Next steps
- add end-to-end tests for admin flows (login, delete, undo)
- add rate-limit and/or token sessions for auth hardening
- add backend audit logs for destructive dataset operations
