# gonova-asr

Production-ready Kyutai STT system with universal WebSocket API, health monitoring, and multi-platform support.

## 🎯 Features

- ✅ **Universal WebSocket API** - Connect from web, SIP, mobile, any platform
- ✅ **Real-time Transcription** - 500ms latency with semantic VAD
- ✅ **Health Monitoring** - HTTP endpoint for service status
- ✅ **SIP Integration** - Phone call transcription (Asterisk/FreeSWITCH/Twilio)
- ✅ **Production Ready** - Systemd services, monitoring, logging
- ✅ **High Throughput** - 30-60 concurrent streams on RTX 3090

## 🚀 Quick Start

```bash
# 1. Install prerequisites (one-time)
./deployment/install-prerequisites.sh

# 2. Deploy everything
./deployment/deploy.sh

# 3. Test
curl http://localhost:8002/health
```

Full guide: [QUICKSTART.md](./QUICKSTART.md)

## 📊 Architecture

```
Browser/SIP/Mobile → WebSocket (Port 9000) → Moshi STT → GPU → Transcription
                                                 ↓
                                         Health Check (Port 8002)
```

## 📚 Documentation

- **[QUICKSTART.md](./QUICKSTART.md)** - 5-minute setup guide
- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - Full deployment guide
- **[SIP_INTEGRATION.md](./SIP_INTEGRATION.md)** - Phone system integration
- **[PROJECT_README.md](./PROJECT_README.md)** - Complete reference

## 🔧 Services

| Service | Port | Description |
|---------|------|-------------|
| Moshi WebSocket | 9000 | Audio streaming + transcription |
| Health Check | 8002 | HTTP health status |
| Web Client | 9002 | Browser test interface |

## 🛠️ What's Included

- **Health Check Service** (Rust) - HTTP monitoring with JSON status
- **Web Client** - Browser-based test interface with real-time transcription
- **Deployment Scripts** - One-command setup with systemd support
- **SIP Integration** - Complete examples for Asterisk, FreeSWITCH, Twilio

## 📦 Requirements

- NVIDIA GPU with CUDA support
- Rust (auto-installed by setup script)
- Python 3
- CUDA Toolkit 11.0+

## 🎯 Use Cases

- Real-time phone call transcription
- Voice assistants and chatbots
- Live captioning for video calls
- Customer support transcription
- Voice command systems

## 📖 Based On

Built using [Kyutai's delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) framework with production enhancements.

## 📄 License

- Python code: MIT
- Rust backend: Apache 2.0
- STT model weights: CC-BY 4.0

## 🔗 Links

- [Kyutai STT Docs](https://kyutai.org/next/stt)
- [Model on HuggingFace](https://huggingface.co/kyutai/stt-1b-en_fr)
- [Original Repository](https://github.com/kyutai-labs/delayed-streams-modeling)
