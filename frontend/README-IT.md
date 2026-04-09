# Frontend - NeuralCanvas WASM

Frontend React responsabile di:

- input utente via canvas;
- preprocessing del disegno in tensore 28x28 normalizzato;
- invocazione del modulo WebAssembly Rust;
- visualizzazione di prediction, confidence e attivazioni interne.
- invio opzionale di sample etichettati al backend (data collection).

## Requisiti

- Node.js 20+
- npm
- modulo WASM generato in `src/wasm`
- backend API raggiungibile (`VITE_API_BASE_URL`)

## Setup rapido

Dalla root repository:

```bash
cd wasm-engine
wasm-pack build --target web --out-dir ../frontend/src/wasm
cd ../frontend
npm install
npm run dev
```

Frontend disponibile su `http://localhost:5173` (oppure `http://localhost:3000` se avviato via Docker Compose).

## Script npm

- `npm run dev`: avvio sviluppo con Vite
- `npm run build`: build TypeScript + bundle produzione
- `npm run preview`: preview build locale
- `npm run lint`: linting frontend

## Componenti principali

- `src/App.tsx`: bootstrap WASM + fetch modello backend + orchestration inferenza + data collection
- `src/DrawingCanvas.tsx`: acquisizione tratto e preprocessing
- `src/NetworkVisualizer.tsx`: rendering neuroni/sinapsi e stato rete
- `src/types.ts`: tipi condivisi per risultato inferenza

## Note

Il frontend prova a caricare il modello da backend (`/api/model/:version`) e usa `/model.json` come fallback locale.
