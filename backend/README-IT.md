# Backend - NeuralCanvas WASM

Backend Rust per due motivi principali:

- **Model Asset Provider**: espone il modello versione-specifico via API.
- **Data Collector**: salva campioni etichettati dall'utente in PostgreSQL per futuri retraining.

## Perche' questo step e' importante

Chiude il gap tra demo frontend e prodotto evolvibile:

- rende il caricamento modello tracciabile per versione (`MODEL_VERSION`);
- introduce una pipeline dati reale (raccolta sample persistita su DB);
- mantiene API stabili per frontend e futuri job di training.

## Endpoint disponibili

- `GET /health`
  - Health-check del servizio.
- `GET /api/model/:version`
  - Ritorna il JSON del modello se la versione richiesta coincide con `MODEL_VERSION`.
- `POST /api/datasets/labels`
  - Salva un sample etichettato (`label`, `pixels`, `source`) nella tabella `labeled_samples`.
- `GET /api/datasets/stats`
  - Ritorna metriche aggregate (`total_samples`, conteggio per label).

## Payload data collector

```json
{
  "label": 3,
  "pixels": [0.0, 0.12, 1.0],
  "source": "frontend"
}
```

Vincoli:

- `label` deve essere tra `0` e `9`;
- `pixels` deve avere lunghezza `784`;
- ogni valore pixel deve stare in `0.0..=1.0`.

## Avvio locale

```bash
cd backend
cargo run
```

## Variabili ambiente

- `PORT` (obbligatoria)
- `DATABASE_URL` (obbligatoria, fail-fast)
- `MODEL_VERSION` (obbligatoria)
- `MODEL_ASSET_PATH` (opzionale, default: `../frontend/public/model.json`)

## Schema database

Al bootstrap il backend crea automaticamente:

- `labeled_samples`
  - `id BIGSERIAL PRIMARY KEY`
  - `created_at TIMESTAMPTZ`
  - `source TEXT`
  - `label SMALLINT CHECK 0..9`
  - `pixels JSONB`
