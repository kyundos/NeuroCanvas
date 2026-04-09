# Frontend - NeuralCanvas WASM

React frontend responsible for:

- user input via drawing canvas;
- preprocessing to normalized 28x28 tensor;
- Rust WebAssembly module invocation;
- prediction, confidence, and internal activation visualization;
- optional labeled sample submission to backend (data collection).

Italian version: `README-IT.md`

## Requirements

- Node.js 20+
- npm
- WASM module generated in `src/wasm`
- reachable backend API (`VITE_API_BASE_URL`)

## Quick setup

From repository root:

```bash
cd wasm-engine
wasm-pack build --target web --out-dir ../frontend/src/wasm
cd ../frontend
npm install
npm run dev
```

Frontend is available on `http://localhost:5173` (or `http://localhost:3000` when started via Docker Compose).

## npm scripts

- `npm run dev`: start Vite development server
- `npm run build`: TypeScript build + production bundle
- `npm run preview`: local production preview
- `npm run lint`: frontend linting

## Main components

- `src/App.tsx`: WASM bootstrap + backend model fetch + inference orchestration + data collection
- `src/DrawingCanvas.tsx`: stroke capture and preprocessing
- `src/NetworkVisualizer.tsx`: neuron/synapse rendering and network state
- `src/types.ts`: shared inference response types

## Notes

Frontend first tries loading model data from backend (`/api/model/:version`) and uses `/model.json` as local fallback.
