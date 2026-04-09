# NeuralCanvas WASM

Visualizzatore interattivo di reti neurali con inferenza locale nel browser tramite WebAssembly (Rust) e UI React.

## Visione del progetto

NeuralCanvas WASM nasce con due obiettivi:

- rendere "visibile" il forward pass di una rete neurale in tempo reale;
- dimostrare una pipeline moderna frontend + Rust/WASM adatta a use case ad alte prestazioni.

L'utente disegna una cifra su canvas, il tensore viene normalizzato in formato 28x28 (MNIST style), inviato al motore Rust compilato in WASM e il frontend mostra:

- classe predetta;
- confidenza;
- attivazioni interne dei layer;
- visualizzazione dinamica di neuroni e connessioni.

## Stack tecnologico

- Frontend: React + TypeScript + Vite + Tailwind CSS
- Inference engine: Rust + `wasm-bindgen` + `serde`
- Backend API: Rust + Axum + SQLx + PostgreSQL
- Runtime locale: browser (inference client-side)
- Containerization: DevContainer + Docker multi-stage

## Perche' WASM + React

- **Performance**: il calcolo numerico del forward pass gira in Rust, riducendo overhead rispetto a una versione solo JS.
- **Reattivita' UI**: React gestisce rendering e interazione, mentre il lavoro CPU-bound e' delegato al modulo WASM.
- **Portabilita'**: una volta compilato, il modulo gira in browser moderni senza plugin.
- **Separazione delle responsabilita'**: dominio matematico in Rust, presentazione e UX in React.

## Architettura

```text
Drawing Canvas (React)
   -> preprocessing 28x28 + normalization [0..1]
   -> Float32Array(784)
   -> NeuralEngine (Rust -> WASM)
        -> Dense + ReLU + Dense + ReLU + Dense + Softmax
        -> predicted_label + probabilities + layer_activations
   -> NetworkVisualizer (React)
```

### Moduli principali

- `frontend/src/DrawingCanvas.tsx`: acquisizione input, bounding box, resize MNIST-style e normalizzazione.
- `frontend/src/App.tsx`: bootstrap WASM, fetch modello dal backend, orchestrazione inferenza e data collection.
- `frontend/src/NetworkVisualizer.tsx`: rendering dinamico di layer, neuroni e sinapsi.
- `wasm-engine/src/lib.rs`: core matematico + API `NeuralEngine` esportata in WASM.
- `backend/src/main.rs`: API backend (health, model asset provider, dataset management).

### Backend API quick reference

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | No | Healthcheck backend + model version |
| `GET` | `/api/model/:version` | No | Serve il modello richiesto dal frontend |
| `POST` | `/api/datasets/labels` | Admin | Salva un sample etichettato |
| `GET` | `/api/datasets/stats` | Admin | Ritorna metriche dataset (totale + count per label) |
| `DELETE` | `/api/datasets/labels` | Admin | Elimina tutti i sample |
| `DELETE` | `/api/datasets/labels/:label` | Admin | Elimina tutti i sample di una label |
| `DELETE` | `/api/datasets/samples/:sample_id` | Admin | Undo one-shot dell'ultimo sample salvato |

Note:

- `Admin` corrisponde a header `Authorization: Basic ...` con `ADMIN_USERNAME` e `ADMIN_PASSWORD`.
- `GET /api/model/:version` resta pubblico per consentire inferenza senza login iniziale.

## Struttura repository

```text
.
├── frontend/         # React app + integrazione WASM
├── wasm-engine/      # Rust crate compilato in WebAssembly
├── backend/          # API Rust + integrazione PostgreSQL
├── Dockerfile        # Multi-stage build produzione
└── docker-compose.yml
```

## Requisiti

### Opzione A - sviluppo locale

- Node.js 20+
- npm
- Rust toolchain (stable)
- `wasm-pack`

Installazione `wasm-pack`:

```bash
cargo install wasm-pack
```

### Opzione B - DevContainer (consigliata)

Serve solo Docker sull'host. Apri la repo in Cursor/VS Code e seleziona "Reopen in Container".

## Avvio del progetto (host locale)

1. Clona e prepara l'ambiente:

```bash
cp .env.example .env
```

2. Compila il modulo WASM e pubblicalo dove il frontend lo importa:

```bash
cd wasm-engine
wasm-pack build --target web --out-dir ../frontend/src/wasm
```

3. Avvia il frontend:

```bash
cd ../frontend
npm install
npm run dev
```

App disponibile su `http://localhost:5173` (porta Vite default).

## Avvio con Docker

Per esplorare l'infrastruttura composta:

```bash
cp .env.example .env
docker compose up --build
```

Servizi dichiarati:

- frontend (Vite dev server, hot-reload)
- backend (Axum API + PostgreSQL)
- postgres

Con stack attivo:

- frontend: `http://localhost:3000`
- backend health: `http://localhost:8080/health`
- backend model API: `http://localhost:8080/api/model/v1.0`

## Label manager e autenticazione

La UI non richiede login all'apertura. L'utente puo' disegnare e vedere inferenza subito.

Per operazioni dataset (labeling/metrics/delete/undo):

- apri il pannello con `Manage Labels & Metrics`;
- se non autenticato, compare un modal login admin;
- credenziali valide => pannello label manager disponibile.

Operazioni nel pannello:

- `Save Sample` salva il sample corrente (disabilitato su canvas vuoto);
- `Undo last label` annulla solo l'ultima operazione di labeling salvata;
- `Delete label X` elimina tutti i sample di una label;
- `Delete ALL samples` elimina tutto il dataset (azione distruttiva con conferma dedicata).

## Build produzione

Build multi-stage (Rust WASM -> frontend bundle -> NGINX):

```bash
docker build -t neuralcanvas-wasm .
docker run --rm -p 8080:80 neuralcanvas-wasm
```

## Testing

Attualmente il crate Rust include:

- **Unit test** in `wasm-engine/src/lib.rs` per il core matematico:
  - `dense_forward` (matmul + bias)
  - `relu`
  - `softmax` (inclusa stabilita' numerica con valori grandi)
  - pipeline deterministica (`dense -> relu -> dense -> softmax`)
  - casi negativi (shape invalide, logits vuoti)
- **Integration test** in `wasm-engine/tests/neural_engine_integration.rs`:
  - caricamento modello valido
  - predizione end-to-end su modello ridotto
  - error handling su input size e shape modello non coerenti
- **WASM API test** in `wasm-engine/tests/wasm_api.rs` (target `wasm32`):
  - costruzione `NeuralEngine` via API `wasm-bindgen`
  - validazione payload serializzato di `predict`
  - error handling su input invalido lato bridge JS/WASM

Esecuzione:

```bash
cd wasm-engine
cargo test
```

Esecuzione test WASM (Node.js):

```bash
cd wasm-engine
wasm-pack test --node
```

## Variabili ambiente

Definite in `.env.example`:

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

## Stato progetto e roadmap

### Completato

- inferenza locale in WASM
- canvas interattivo + preprocessing 28x28
- visualizzazione stato interno rete
- unit/integration/WASM test sul crate Rust
- CI pipeline con lint + test + build WASM/frontend
- backend Rust + PostgreSQL (health, model asset provider, data collector persistente)
- integrazione frontend -> backend (`/api/model/:version`, `/api/datasets/labels`)
- autenticazione admin su dataset API + panel UI protetto on-demand
- label manager con delete bulk/per-label e undo one-shot ultima etichettatura

### Prossimi step
- aggiungere test end-to-end per i flussi admin (login, delete, undo)
- introdurre rate-limit e/o sessione token per hardening autenticazione
- aggiungere audit log backend per operazioni distruttive dataset
