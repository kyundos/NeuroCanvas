use anyhow::{Context, Result};
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{postgres::PgPoolOptions, PgPool};
use std::{env, net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::{fs, net::TcpListener};
use tower_http::cors::CorsLayer;

#[derive(Clone)]
struct AppState {
    model_version: String,
    model_asset_path: PathBuf,
    admin_username: String,
    admin_password: String,
    pool: PgPool,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    model_version: String,
}

#[derive(Serialize)]
struct ModelResponse {
    version: String,
    model: Value,
}

#[derive(Deserialize)]
struct LabelSampleRequest {
    pixels: Vec<f32>,
    label: u8,
    source: Option<String>,
}

#[derive(Serialize)]
struct LabelSampleStored {
    saved: bool,
    sample_id: i64,
}

#[derive(Serialize)]
struct DeleteSamplesResponse {
    deleted_count: u64,
}

#[derive(Serialize)]
struct DatasetStatsResponse {
    total_samples: i64,
    counts_by_label: Vec<LabelCount>,
}

#[derive(Serialize, sqlx::FromRow)]
struct LabelCount {
    label: i16,
    count: i64,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let database_url = resolve_database_url()?;
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await
        .context("failed to connect to PostgreSQL")?;

    init_schema(&pool).await?;

    let config = AppState {
        model_version: required_env("MODEL_VERSION")?,
        model_asset_path: env::var("MODEL_ASSET_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("../frontend/public/model.json")),
        admin_username: required_env("ADMIN_USERNAME")?,
        admin_password: required_env("ADMIN_PASSWORD")?,
        pool,
    };
    let port = required_env("PORT")?
        .parse::<u16>()
        .context("PORT must be a valid u16 number")?;

    let app_state = Arc::new(config);
    let app = Router::new()
        .route("/health", get(health))
        .route("/api/model/{version}", get(get_model))
        .route(
            "/api/datasets/labels",
            post(post_label_sample).delete(delete_all_samples),
        )
        .route(
            "/api/datasets/labels/{label}",
            delete(delete_samples_by_label),
        )
        .route(
            "/api/datasets/samples/{sample_id}",
            delete(delete_sample_by_id),
        )
        .route("/api/datasets/stats", get(get_dataset_stats))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(addr).await?;
    println!("backend listening on {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn init_schema(pool: &PgPool) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS labeled_samples (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            source TEXT NOT NULL,
            label SMALLINT NOT NULL CHECK (label >= 0 AND label <= 9),
            pixels JSONB NOT NULL
        );
        "#,
    )
    .execute(pool)
    .await
    .context("failed to run database schema initialization")?;
    Ok(())
}

fn required_env(name: &str) -> Result<String> {
    env::var(name).with_context(|| format!("missing required environment variable: {name}"))
}

fn resolve_database_url() -> Result<String> {
    if let Ok(url) = env::var("DATABASE_URL") {
        return Ok(url);
    }

    let target = env::var("DATABASE_TARGET").unwrap_or_else(|_| "local".to_string());
    let key = match target.as_str() {
        "local" => "DATABASE_URL_LOCAL",
        "docker" => "DATABASE_URL_DOCKER",
        other => {
            anyhow::bail!(
                "invalid DATABASE_TARGET value: {other} (expected 'local' or 'docker')"
            )
        }
    };

    required_env(key)
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model_version: state.model_version.clone(),
    })
}

async fn get_model(
    Path(version): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ModelResponse>, (StatusCode, Json<ErrorResponse>)> {
    if version != state.model_version {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("model version {version} not available"),
            }),
        ));
    }

    let raw_model = fs::read_to_string(&state.model_asset_path)
        .await
        .map_err(map_internal_io("failed to read model asset"))?;

    let model_json: Value = serde_json::from_str(&raw_model)
        .map_err(map_internal_parse("model asset contains invalid JSON"))?;

    Ok(Json(ModelResponse {
        version,
        model: model_json,
    }))
}

async fn post_label_sample(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(payload): Json<LabelSampleRequest>,
) -> Result<Json<LabelSampleStored>, (StatusCode, Json<ErrorResponse>)> {
    ensure_authorized(&headers, &state)?;
    validate_label_payload(&payload)?;

    let source = payload.source.unwrap_or_else(|| "frontend".to_string());
    let pixels_json = serde_json::to_value(payload.pixels)
        .map_err(map_internal_parse("failed to serialize pixels payload"))?;

    let sample_id: i64 = sqlx::query_scalar(
        r#"
        INSERT INTO labeled_samples (source, label, pixels)
        VALUES ($1, $2, $3)
        RETURNING id
        "#,
    )
    .bind(source)
    .bind(payload.label as i16)
    .bind(pixels_json)
    .fetch_one(&state.pool)
    .await
    .map_err(map_internal_db("failed to store labeled sample"))?;

    Ok(Json(LabelSampleStored {
        saved: true,
        sample_id,
    }))
}

async fn get_dataset_stats(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<DatasetStatsResponse>, (StatusCode, Json<ErrorResponse>)> {
    ensure_authorized(&headers, &state)?;

    let total_samples: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM labeled_samples")
        .fetch_one(&state.pool)
        .await
        .map_err(map_internal_db("failed to count dataset samples"))?;

    let counts_by_label: Vec<LabelCount> = sqlx::query_as::<_, LabelCount>(
        r#"
        SELECT label, COUNT(*) AS count
        FROM labeled_samples
        GROUP BY label
        ORDER BY label ASC
        "#,
    )
    .fetch_all(&state.pool)
    .await
    .map_err(map_internal_db("failed to fetch label statistics"))?;

    Ok(Json(DatasetStatsResponse {
        total_samples,
        counts_by_label,
    }))
}

async fn delete_all_samples(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<DeleteSamplesResponse>, (StatusCode, Json<ErrorResponse>)> {
    ensure_authorized(&headers, &state)?;
    let deleted_count = sqlx::query("DELETE FROM labeled_samples")
        .execute(&state.pool)
        .await
        .map_err(map_internal_db("failed to delete all samples"))?
        .rows_affected();

    Ok(Json(DeleteSamplesResponse { deleted_count }))
}

async fn delete_samples_by_label(
    Path(label): Path<u8>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<DeleteSamplesResponse>, (StatusCode, Json<ErrorResponse>)> {
    ensure_authorized(&headers, &state)?;
    if label > 9 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "label must be in range 0..=9".to_string(),
            }),
        ));
    }

    let deleted_count = sqlx::query("DELETE FROM labeled_samples WHERE label = $1")
        .bind(label as i16)
        .execute(&state.pool)
        .await
        .map_err(map_internal_db("failed to delete samples by label"))?
        .rows_affected();

    Ok(Json(DeleteSamplesResponse { deleted_count }))
}

async fn delete_sample_by_id(
    Path(sample_id): Path<i64>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<DeleteSamplesResponse>, (StatusCode, Json<ErrorResponse>)> {
    ensure_authorized(&headers, &state)?;
    let deleted_count = sqlx::query("DELETE FROM labeled_samples WHERE id = $1")
        .bind(sample_id)
        .execute(&state.pool)
        .await
        .map_err(map_internal_db("failed to undo sample"))?
        .rows_affected();

    Ok(Json(DeleteSamplesResponse { deleted_count }))
}

fn validate_label_payload(
    payload: &LabelSampleRequest,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if payload.label > 9 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "label must be in range 0..=9".to_string(),
            }),
        ));
    }

    if payload.pixels.len() != 28 * 28 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "pixels must have length 784 (28x28)".to_string(),
            }),
        ));
    }

    if payload.pixels.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "pixels values must be normalized in range 0.0..=1.0".to_string(),
            }),
        ));
    }

    Ok(())
}

fn map_internal_io(
    message: &'static str,
) -> impl FnOnce(std::io::Error) -> (StatusCode, Json<ErrorResponse>) {
    move |err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("{message}: {err}"),
            }),
        )
    }
}

fn map_internal_parse(
    message: &'static str,
) -> impl FnOnce(serde_json::Error) -> (StatusCode, Json<ErrorResponse>) {
    move |err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("{message}: {err}"),
            }),
        )
    }
}

fn map_internal_db(
    message: &'static str,
) -> impl FnOnce(sqlx::Error) -> (StatusCode, Json<ErrorResponse>) {
    move |err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("{message}: {err}"),
            }),
        )
    }
}

fn ensure_authorized(
    headers: &HeaderMap,
    state: &Arc<AppState>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let encoded = BASE64_STANDARD.encode(format!(
        "{}:{}",
        state.admin_username, state.admin_password
    ));
    let expected = format!("Basic {encoded}");

    let provided = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();

    if provided != expected {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "invalid admin credentials".to_string(),
            }),
        ));
    }

    Ok(())
}
