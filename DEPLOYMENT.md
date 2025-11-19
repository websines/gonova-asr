# Kyutai STT Universal Deployment Guide

This guide covers deploying the Kyutai STT system with WebSocket support, health checks, and integration with multiple sources (web browsers, SIP/phone calls, mobile apps, etc.).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Universal STT System                     │
└─────────────────────────────────────────────────────────────┘

Input Sources:                    Services:                Output:
┌─────────────┐                  ┌──────────────┐
│   Browser   │──────┐          │ Moshi Server │
│  (Web Mic)  │      │          │ Port: 8080   │──────▶ Transcripts
└─────────────┘      │          │  WebSocket   │         + Timestamps
                     │          └──────────────┘         + VAD
┌─────────────┐      │                  │
│ SIP Gateway │──────┼─── WebSocket ────┤
│  (Asterisk/ │      │      Audio       │
│ FreeSWITCH) │      │      Stream      │
└─────────────┘      │                  │
                     │          ┌──────────────┐
┌─────────────┐      │          │ Health Check │
│ Mobile Apps │──────┘          │ Port: 8001   │──────▶ Status JSON
└─────────────┘                 │   HTTP       │
                                └──────────────┘
```

## System Components

### 1. Moshi WebSocket Server (Port 8080)
- **Purpose**: Real-time speech-to-text transcription
- **Technology**: Rust with CUDA acceleration
- **Capacity**: 30-60 concurrent streams on RTX 3090
- **Protocol**: WebSocket with PCM audio input (24kHz, mono, int16)

### 2. Health Check Service (Port 8001)
- **Purpose**: Service monitoring and health status
- **Technology**: Rust (Axum framework)
- **Endpoints**:
  - `GET /health` - Returns service health status
  - `GET /info` - Returns service information

### 3. Web Client (Port 8000)
- **Purpose**: Browser-based microphone testing
- **Technology**: HTML5 + Web Audio API
- **Features**: Real-time transcription display

## Prerequisites

### CUDA System Requirements
- NVIDIA GPU (RTX 3090 or better)
- CUDA Toolkit 11.0+ with `nvcc`
- NVIDIA Driver 450.80.02+
- 24GB+ VRAM for optimal performance

### Software Requirements
- Rust 1.70+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Python 3.8+ (for web server and scripts)
- Git

## Quick Start (CUDA System)

### 1. Clone and Navigate
```bash
cd /path/to/gonova-asr/delayed-streams-modeling
```

### 2. Deploy Everything
```bash
./deployment/deploy.sh
```

This will:
- Install moshi-server with CUDA support
- Build the health check service
- Start all services
- Display service URLs

### 3. Test Health Check
```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "websocket_available": true,
  "timestamp": 1700000000,
  "service": "kyutai-stt-server"
}
```

### 4. Test Web Client
Open browser to: `http://localhost:8000`

1. Click "Connect"
2. Click "Start Recording"
3. Speak into microphone
4. See real-time transcription

## Manual Deployment

### Step 1: Install Moshi Server
```bash
cargo install --features cuda moshi-server
```

### Step 2: Build Health Check Service
```bash
cd services/health-check
cargo build --release
cd ../..
```

### Step 3: Start Moshi Server
```bash
moshi-server worker --config configs/config-stt-en_fr-hf.toml
```

### Step 4: Start Health Check Service
```bash
WEBSOCKET_URL="ws://localhost:8080" HEALTH_PORT=8001 \
  ./services/health-check/target/release/stt-health-check
```

### Step 5: Start Web Client
```bash
cd web-client
python3 -m http.server 8000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOSHI_PORT` | 8080 | WebSocket server port |
| `HEALTH_PORT` | 8001 | Health check HTTP port |
| `WEB_PORT` | 8000 | Web client port |
| `WEBSOCKET_URL` | ws://localhost:8080 | WebSocket server URL for health checks |
| `STT_MODEL` | kyutai/stt-1b-en_fr | Model to use |

## Production Deployment

### Using Systemd (Recommended)

1. Run deployment script and choose option 2 or 3:
```bash
./deployment/deploy.sh
# Choose: 3 (Both)
```

2. Install systemd services:
```bash
sudo mv /tmp/kyutai-stt*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable kyutai-stt kyutai-stt-health
sudo systemctl start kyutai-stt kyutai-stt-health
```

3. Check status:
```bash
sudo systemctl status kyutai-stt
sudo systemctl status kyutai-stt-health
```

### Using Docker (Alternative)

Create `Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy project
WORKDIR /app
COPY . .

# Build services
RUN cargo install --features cuda moshi-server
RUN cd services/health-check && cargo build --release

# Expose ports
EXPOSE 8080 8001

# Start script
CMD ["./deployment/deploy.sh"]
```

Build and run:
```bash
docker build -t kyutai-stt .
docker run --gpus all -p 8080:8080 -p 8001:8001 kyutai-stt
```

### Reverse Proxy (Nginx)

For production with SSL:

```nginx
# WebSocket proxy
upstream moshi_backend {
    server localhost:8080;
}

# Health check proxy
upstream health_backend {
    server localhost:8001;
}

server {
    listen 443 ssl;
    server_name stt.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # WebSocket endpoint
    location /ws {
        proxy_pass http://moshi_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://health_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Integration Guides

See [SIP_INTEGRATION.md](./SIP_INTEGRATION.md) for detailed integration guides for:
- SIP/VoIP phone systems
- Web browsers
- Mobile applications
- Other platforms

## Monitoring

### Health Check Monitoring

Add to your monitoring system:
```bash
#!/bin/bash
# health_monitor.sh
while true; do
    RESPONSE=$(curl -s http://localhost:8001/health)
    STATUS=$(echo $RESPONSE | jq -r '.status')

    if [ "$STATUS" != "healthy" ]; then
        echo "ALERT: STT Service unhealthy"
        # Send alert (email, Slack, PagerDuty, etc.)
    fi

    sleep 60
done
```

### Prometheus Metrics

Moshi server exposes metrics at `/metrics` (port 8080). Configure Prometheus:

```yaml
scrape_configs:
  - job_name: 'kyutai-stt'
    static_configs:
      - targets: ['localhost:8080']
```

## Performance Tuning

### GPU Memory Optimization

Adjust batch size in config file (`configs/config-stt-en_fr-hf.toml`):
```toml
[model]
batch_size = 32  # Lower if you have VRAM issues, higher for more throughput
```

### Concurrent Connections

Monitor GPU memory usage:
```bash
watch -n 1 nvidia-smi
```

Typical capacity:
- RTX 3090 (24GB): 30-60 streams
- A100 (40GB): 100-150 streams
- H100 (80GB): 400+ streams

## Troubleshooting

### Issue: "CUDA not found"
**Solution**: Install CUDA Toolkit and ensure `nvcc` is in PATH
```bash
nvcc --version
```

### Issue: WebSocket connection refused
**Solution**: Check if moshi-server is running
```bash
ps aux | grep moshi-server
curl http://localhost:8001/health
```

### Issue: No transcription output
**Solution**: Check audio format (must be 24kHz, mono, int16 PCM)

### Issue: High latency
**Solution**:
- Reduce batch size
- Check GPU utilization
- Ensure real-time factor > 1.0

## Stopping Services

```bash
./deployment/stop.sh
```

## Logs

All logs are stored in `logs/` directory:
- `moshi-server.log` - WebSocket server logs
- `health-check.log` - Health check service logs
- `web-server.log` - Web client server logs

## Support

- GitHub Issues: https://github.com/kyutai-labs/delayed-streams-modeling/issues
- Documentation: https://kyutai.org/next/stt
- Model: https://huggingface.co/kyutai/stt-1b-en_fr
