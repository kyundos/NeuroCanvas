# Backend - NeuralCanvas WASM

Rust backend for two main purposes:

- **Model Asset Provider**: serves versioned model assets through API.
- **Data Collector**: stores user-labeled samples in PostgreSQL for future retraining.

Italian version: `README-IT.md`

## Why this matters

This closes the gap between a frontend demo and an evolvable product:

- keeps model loading traceable by version (`MODEL_VERSION`);
- introduces a real data pipeline (samples persisted in DB);
- keeps stable APIs for frontend and future training jobs.

## Available endpoints

- `GET /health`
  - Service health check.
- `GET /api/model/:version`
  - Returns model JSON when requested version matches `MODEL_VERSION`.
- `POST /api/datasets/labels`
  - Stores a labeled sample (`label`, `pixels`, `source`) in `labeled_samples`.
- `GET /api/datasets/stats`
  - Returns aggregate metrics (`total_samples`, counts by label).
- `DELETE /api/datasets/labels`
  - Deletes all stored samples.
- `DELETE /api/datasets/labels/:label`
  - Deletes all samples for one label.
- `DELETE /api/datasets/samples/:sample_id`
  - One-shot undo for one sample by id.

## Data collector payload

```json
{
  "label": 3,
  "pixels": [0.0, 0.12, 1.0],
  "source": "frontend"
}
```

Constraints:

- `label` must be in `0..=9`;
- `pixels` must have length `784`;
- each pixel value must be in `0.0..=1.0`.

## Local run

```bash
cd backend
cargo run
```

## Environment variables

- `PORT` (required)
- `DATABASE_TARGET` (required unless `DATABASE_URL` is set)
- `DATABASE_URL_LOCAL` (required when `DATABASE_TARGET=local`)
- `DATABASE_URL_DOCKER` (required when `DATABASE_TARGET=docker`)
- `MODEL_VERSION` (required)
- `MODEL_ASSET_PATH` (optional, default: `../frontend/public/model.json`)
- `ADMIN_USERNAME` (required for dataset endpoints)
- `ADMIN_PASSWORD` (required for dataset endpoints)

## Database schema

At startup, backend auto-creates:

- `labeled_samples`
  - `id BIGSERIAL PRIMARY KEY`
  - `created_at TIMESTAMPTZ`
  - `source TEXT`
  - `label SMALLINT CHECK 0..9`
  - `pixels JSONB`
