# Quick Start Guide

Get up and running with Kyutai STT in 5 minutes.

## For Your RTX 3090 System (Production)

### 1. Transfer Files

From your Mac (development machine):
```bash
cd /Users/subhankarchowdhury/devfolder/gonova/gonova-asr
tar -czf kyutai-stt.tar.gz delayed-streams-modeling/

# Transfer to CUDA system
scp kyutai-stt.tar.gz user@your-3090-server:/path/to/destination/
```

On your RTX 3090 system:
```bash
tar -xzf kyutai-stt.tar.gz
cd delayed-streams-modeling
```

### 2. Install Prerequisites (One-Time Setup)

**Required:** Rust + CUDA Toolkit + Python

```bash
./deployment/install-prerequisites.sh
```

This will:
- ✅ Install Rust (if not present)
- ✅ Check CUDA is available
- ✅ Verify Python is installed
- ⚡ Takes ~2 minutes

**Already have Rust/CUDA?** Skip this step.

### 3. One-Command Deploy

```bash
./deployment/deploy.sh
```

Choose option **1** (Start services now) or **3** (Create systemd + start)

⏱️ **First run**: Takes ~5-10 minutes to compile moshi-server with CUDA
📦 **Subsequent runs**: Instant (already compiled)

### 4. Verify It's Working

```bash
# Check health endpoint
curl http://localhost:8002/health

# Should return:
# {
#   "status": "healthy",
#   "websocket_available": true,
#   "timestamp": ...,
#   "service": "kyutai-stt-server"
# }
```

### 5. Test with Web Interface

Open browser: `http://localhost:9002`

1. Click **Connect**
2. Click **Start Recording**
3. Speak into your microphone
4. Watch real-time transcription appear

## Ports Used

⚠️ **Updated to avoid conflicts with your existing services:**

| Service | Port | URL |
|---------|------|-----|
| Moshi WebSocket Server | **9000** | `ws://localhost:9000` |
| Health Check API | **8002** | `http://localhost:8002` (nginx configured) |
| Web Client | **9002** | `http://localhost:9002` |

## Custom Ports

If you need different ports, set environment variables before deploying:

```bash
export MOSHI_PORT=9500
export HEALTH_PORT=9501
export WEB_PORT=9502
./deployment/deploy.sh
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env with your ports
```

## Stop Services

```bash
./deployment/stop.sh
```

## Next Steps

### Connect from JavaScript
```javascript
const ws = new WebSocket('ws://localhost:9000');

ws.onopen = () => console.log('Connected');
ws.onmessage = (event) => console.log('Transcript:', event.data);

// Send audio (24kHz, mono, int16 PCM)
ws.send(audioBuffer);
```

### Connect from SIP/Phone System

See [SIP_INTEGRATION.md](./SIP_INTEGRATION.md) for:
- Asterisk integration
- FreeSWITCH integration
- Twilio integration

### Monitor Service Health

```bash
# Continuous monitoring
watch -n 5 'curl -s http://localhost:8002/health | jq'

# Check logs
tail -f logs/moshi-server.log
tail -f logs/health-check.log
```

### Scale for Production

See [DEPLOYMENT.md](./DEPLOYMENT.md) for:
- Systemd service setup
- Nginx reverse proxy configuration
- Docker deployment
- Load balancing

## Troubleshooting

### "Port already in use"
```bash
# Find what's using the port
sudo lsof -i :9000

# Use different ports
export MOSHI_PORT=9500
./deployment/deploy.sh
```

### "CUDA not found"
```bash
# Check CUDA
nvcc --version
nvidia-smi

# If missing, install CUDA Toolkit:
# https://developer.nvidia.com/cuda-downloads
```

### "No audio/transcription"
- Verify audio format: 24kHz, mono, int16 PCM
- Check microphone permissions in browser
- Review logs: `tail -f logs/*.log`

## Architecture Diagram

```
Browser (http://localhost:9002)
   │
   │ WebSocket
   ▼
Moshi Server (ws://localhost:9000)
   │
   ▼
[RTX 3090 GPU] → Real-time Transcription
   │
   ▼
Transcript Output


Health Check (http://localhost:8002/health)
   │
   ▼
Service Status JSON
```

## Models Available

### kyutai/stt-1b-en_fr (Default, Recommended)
- ✅ English + French
- ✅ 500ms latency
- ✅ Semantic VAD
- ✅ 30-60 concurrent streams on RTX 3090

### kyutai/stt-2.6b-en
- ✅ English only
- ⚠️ 2.5s latency
- ✅ Higher accuracy
- ⚠️ 20-40 concurrent streams on RTX 3090

Switch models by editing the config file path in `deployment/deploy.sh`

## Resources

- **Full Deployment Guide**: [DEPLOYMENT.md](./DEPLOYMENT.md)
- **SIP Integration**: [SIP_INTEGRATION.md](./SIP_INTEGRATION.md)
- **Project Overview**: [PROJECT_README.md](./PROJECT_README.md)
- **Kyutai STT Docs**: https://kyutai.org/next/stt
- **Model on HuggingFace**: https://huggingface.co/kyutai/stt-1b-en_fr

## Support

Issues? Questions?
- GitHub: https://github.com/kyutai-labs/delayed-streams-modeling/issues
- Check logs in `logs/` directory
- Review documentation files above
