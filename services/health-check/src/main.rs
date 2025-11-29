use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

#[derive(Clone)]
struct AppState {
    websocket_url: String,
}

#[derive(Serialize, Deserialize)]
struct HealthResponse {
    status: String,
    websocket_available: bool,
    timestamp: u64,
    service: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn check_websocket(ws_url: &str) -> Result<bool, String> {
    match timeout(Duration::from_secs(5), connect_async(ws_url)).await {
        Ok(Ok((mut ws_stream, _))) => {
            // Try to close gracefully
            let _ = ws_stream.close(None).await;
            Ok(true)
        }
        Ok(Err(e)) => Err(format!("WebSocket connection failed: {}", e)),
        Err(_) => Err("WebSocket connection timeout".to_string()),
    }
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ws_check = check_websocket(&state.websocket_url).await;

    let (ws_available, error) = match ws_check {
        Ok(available) => (available, None),
        Err(e) => {
            error!("WebSocket health check failed: {}", e);
            (false, Some(e))
        }
    };

    let status = if ws_available { "healthy" } else { "degraded" };
    let http_status = if ws_available {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let response = HealthResponse {
        status: status.to_string(),
        websocket_available: ws_available,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        service: "kyutai-stt-server".to_string(),
        error,
    };

    (http_status, Json(response))
}

async fn info_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "Kyutai STT Health Check Service",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": {
            "/health": "Health check endpoint",
            "/info": "Service information"
        }
    }))
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let websocket_url = std::env::var("WEBSOCKET_URL")
        .unwrap_or_else(|_| "ws://localhost:9000/api/asr-streaming?token=public_token".to_string());

    let health_port = std::env::var("HEALTH_PORT")
        .unwrap_or_else(|_| "8001".to_string())
        .parse::<u16>()
        .expect("HEALTH_PORT must be a valid port number");

    info!("Health check service starting...");
    info!("Monitoring WebSocket at: {}", websocket_url);
    info!("Health endpoint will be available at: http://0.0.0.0:{}/health", health_port);

    let state = Arc::new(AppState {
        websocket_url: websocket_url.clone(),
    });

    // Configure CORS to allow requests from any origin
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/info", get(info_handler))
        .layer(cors)
        .with_state(state);

    let addr = format!("0.0.0.0:{}", health_port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind to address");

    info!("Health check service listening on {}", addr);

    axum::serve(listener, app)
        .await
        .expect("Server failed to start");
}
