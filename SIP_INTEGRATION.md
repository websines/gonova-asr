# SIP/VoIP Integration Guide

This guide explains how to integrate Kyutai STT with SIP/VoIP systems for real-time phone call transcription.

## Architecture

```
┌──────────────┐       RTP Audio        ┌─────────────────┐
│  SIP Client  │◀──────────────────────▶│  SIP Gateway    │
│ (Phone Call) │      (μ-law/A-law)     │ (Asterisk/      │
└──────────────┘                        │  FreeSWITCH)    │
                                        └─────────────────┘
                                                │
                                                │ Audio Processing:
                                                │ 1. Extract RTP stream
                                                │ 2. Decode μ-law → PCM
                                                │ 3. Resample → 24kHz
                                                │ 4. Convert to int16
                                                │
                                                ▼
                                        ┌─────────────────┐
                                        │ WebSocket Relay │
                                        │   (Bridge)      │
                                        └─────────────────┘
                                                │
                                                │ WebSocket
                                                │ PCM 24kHz int16
                                                ▼
                                        ┌─────────────────┐
                                        │  Moshi Server   │
                                        │   Port: 8080    │
                                        └─────────────────┘
                                                │
                                                ▼
                                        Real-time Transcript
```

## Option 1: Asterisk Integration

### Prerequisites
- Asterisk 18+
- Node.js 16+ (for WebSocket bridge)
- sox (for audio format conversion)

### Installation

#### 1. Install Asterisk
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install asterisk asterisk-modules

# CentOS/RHEL
sudo yum install asterisk asterisk-modules
```

#### 2. Configure Asterisk for External Media

Edit `/etc/asterisk/extensions.conf`:
```ini
[default]
; Route incoming calls to transcription
exten => _X.,1,Answer()
    same => n,Set(CHANNEL(hangup_handler_push)=cleanup,s,1)
    same => n,Stasis(transcription-app)
    same => n,Hangup()

[cleanup]
exten => s,1,NoOp(Call ended)
```

Edit `/etc/asterisk/http.conf`:
```ini
[general]
enabled=yes
bindaddr=127.0.0.1
bindport=8088
```

Edit `/etc/asterisk/ari.conf`:
```ini
[general]
enabled = yes

[transcription]
type = user
read_only = no
password = your-secure-password
```

Restart Asterisk:
```bash
sudo systemctl restart asterisk
```

### 3. Create WebSocket Bridge

Create `sip-websocket-bridge/package.json`:
```json
{
  "name": "sip-websocket-bridge",
  "version": "1.0.0",
  "dependencies": {
    "ari-client": "^2.2.0",
    "ws": "^8.14.0",
    "wav": "^1.0.2"
  }
}
```

Create `sip-websocket-bridge/index.js`:
```javascript
const ari = require('ari-client');
const WebSocket = require('ws');
const { Readable } = require('stream');

const ARI_URL = 'http://localhost:8088';
const ARI_USER = 'transcription';
const ARI_PASS = 'your-secure-password';
const STT_WS_URL = 'ws://localhost:8080';

// Audio format conversion constants
const ASTERISK_SAMPLE_RATE = 8000; // Asterisk default
const TARGET_SAMPLE_RATE = 24000;  // Kyutai STT requirement
const RESAMPLE_FACTOR = TARGET_SAMPLE_RATE / ASTERISK_SAMPLE_RATE; // 3

// μ-law to PCM conversion table (standard ITU-T G.711)
const MULAW_TO_PCM = new Int16Array(256);
(function initMuLawTable() {
    for (let i = 0; i < 256; i++) {
        let mulaw = ~i;
        let sign = (mulaw & 0x80);
        let exponent = (mulaw >> 4) & 0x07;
        let mantissa = mulaw & 0x0F;
        let sample = ((mantissa << 3) + 0x84) << exponent;
        if (sign) sample = -sample;
        MULAW_TO_PCM[i] = sample;
    }
})();

class AudioProcessor {
    constructor(sttWebSocket) {
        this.sttWs = sttWebSocket;
        this.buffer = Buffer.alloc(0);
    }

    // Convert μ-law to 16-bit PCM
    muLawToPCM(muLawData) {
        const pcm = Buffer.alloc(muLawData.length * 2);
        for (let i = 0; i < muLawData.length; i++) {
            pcm.writeInt16LE(MULAW_TO_PCM[muLawData[i]], i * 2);
        }
        return pcm;
    }

    // Simple linear interpolation resampling
    resample(pcmData, fromRate, toRate) {
        const ratio = toRate / fromRate;
        const samplesIn = pcmData.length / 2;
        const samplesOut = Math.floor(samplesIn * ratio);
        const output = Buffer.alloc(samplesOut * 2);

        for (let i = 0; i < samplesOut; i++) {
            const srcPos = i / ratio;
            const srcIndex = Math.floor(srcPos);
            const fraction = srcPos - srcIndex;

            if (srcIndex + 1 < samplesIn) {
                const sample1 = pcmData.readInt16LE(srcIndex * 2);
                const sample2 = pcmData.readInt16LE((srcIndex + 1) * 2);
                const interpolated = Math.round(sample1 + (sample2 - sample1) * fraction);
                output.writeInt16LE(interpolated, i * 2);
            } else {
                const sample = pcmData.readInt16LE(srcIndex * 2);
                output.writeInt16LE(sample, i * 2);
            }
        }

        return output;
    }

    processChunk(muLawData) {
        // Convert μ-law to PCM
        const pcm8k = this.muLawToPCM(muLawData);

        // Resample from 8kHz to 24kHz
        const pcm24k = this.resample(pcm8k, ASTERISK_SAMPLE_RATE, TARGET_SAMPLE_RATE);

        // Send to STT server
        if (this.sttWs.readyState === WebSocket.OPEN) {
            this.sttWs.send(pcm24k);
        }
    }
}

async function main() {
    console.log('Starting SIP to WebSocket Bridge...');

    const client = await ari.connect(ARI_URL, ARI_USER, ARI_PASS);

    client.on('StasisStart', async (event, channel) => {
        console.log(`New call from ${channel.caller.number}`);

        // Create WebSocket connection to STT server
        const sttWs = new WebSocket(STT_WS_URL);
        const audioProcessor = new AudioProcessor(sttWs);

        sttWs.on('open', () => {
            console.log('Connected to STT server');
        });

        sttWs.on('message', (data) => {
            // Handle transcription results
            const transcript = data.toString();
            console.log(`Transcript: ${transcript}`);

            // You can store this in a database, send to a webhook, etc.
        });

        sttWs.on('error', (error) => {
            console.error('STT WebSocket error:', error);
        });

        // Create external media channel for raw audio
        const extChannel = await client.channels.externalMedia({
            app: 'transcription-app',
            external_host: '127.0.0.1:8088',
            format: 'ulaw',
            direction: 'both'
        });

        // Answer the call
        await channel.answer();

        // Create a bridge
        const bridge = client.Bridge();
        await bridge.create({ type: 'mixing' });
        await bridge.addChannel({ channel: channel.id });
        await bridge.addChannel({ channel: extChannel.id });

        // Start receiving audio
        extChannel.on('ChannelStateChange', (event) => {
            console.log('Channel state:', event.channel.state);
        });

        // Get audio stream
        const mediaStream = await extChannel.getChannelVar({
            variable: 'EXTERNALMEDIARTP'
        });

        // Process incoming audio packets
        extChannel.on('media', (data) => {
            audioProcessor.processChunk(data);
        });

        // Handle call hangup
        channel.on('StasisEnd', async () => {
            console.log('Call ended');
            sttWs.close();
            await bridge.destroy();
        });
    });

    client.start('transcription-app');
    console.log('Bridge ready and listening for calls');
}

main().catch(console.error);
```

### 4. Run the Bridge

```bash
cd sip-websocket-bridge
npm install
node index.js
```

### 5. Test with a Call

Make a call to your Asterisk server. The bridge will:
1. Receive the call
2. Extract RTP audio
3. Convert μ-law → PCM → 24kHz
4. Stream to Moshi WebSocket server
5. Print transcriptions to console

## Option 2: FreeSWITCH Integration

### Prerequisites
- FreeSWITCH 1.10+
- mod_audio_stream

### Installation

#### 1. Install FreeSWITCH
```bash
# Ubuntu/Debian
wget -O - https://files.freeswitch.org/repo/deb/debian-release/fsstretch-archive-keyring.asc | apt-key add -
echo "deb https://files.freeswitch.org/repo/deb/debian-release/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/freeswitch.list
apt-get update
apt-get install freeswitch-all
```

#### 2. Enable mod_audio_stream

Edit `/etc/freeswitch/autoload_configs/modules.conf.xml`:
```xml
<load module="mod_audio_stream"/>
```

Create `/etc/freeswitch/autoload_configs/audio_stream.conf.xml`:
```xml
<configuration name="audio_stream.conf" description="Audio Stream">
  <settings>
    <param name="websocket-url" value="ws://localhost:8080"/>
    <param name="sample-rate" value="24000"/>
    <param name="channels" value="1"/>
  </settings>
</configuration>
```

#### 3. Create Dialplan

Edit `/etc/freeswitch/dialplan/default.xml`:
```xml
<extension name="transcription">
  <condition field="destination_number" expression="^transcribe$">
    <action application="answer"/>
    <action application="audio_stream" data="start"/>
    <action application="park"/>
  </condition>
</extension>
```

Restart FreeSWITCH:
```bash
systemctl restart freeswitch
```

## Option 3: Twilio Integration

For cloud-based SIP, use Twilio with a WebSocket relay:

### 1. Create Twilio TwiML App

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="wss://your-server.com/ws/transcribe"/>
    </Start>
    <Say>Please speak after the beep</Say>
    <Record maxLength="60" transcribe="false"/>
</Response>
```

### 2. Create WebSocket Relay

```javascript
// twilio-relay.js
const WebSocket = require('ws');
const express = require('express');

const app = express();
const wss = new WebSocket.Server({ noServer: true });

app.use(express.urlencoded({ extended: true }));

const server = app.listen(3000);

server.on('upgrade', (request, socket, head) => {
    wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request);
    });
});

wss.on('connection', (twilioWs) => {
    console.log('Twilio connected');

    const sttWs = new WebSocket('ws://localhost:8080');

    twilioWs.on('message', (message) => {
        const msg = JSON.parse(message);

        if (msg.event === 'media') {
            // Twilio sends μ-law base64 encoded
            const audio = Buffer.from(msg.media.payload, 'base64');

            // Convert and send to STT
            // (use same conversion logic as Asterisk example)
            sttWs.send(convertToSTTFormat(audio));
        }
    });

    sttWs.on('message', (transcript) => {
        console.log('Transcript:', transcript);
        // Send back to Twilio or store
    });
});

console.log('Twilio relay listening on port 3000');
```

## Testing

### Test Audio Format

Use this script to verify your audio is correctly formatted:

```python
# test_audio_format.py
import struct
import wave

def check_audio_format(filename):
    with wave.open(filename, 'rb') as wav:
        print(f"Channels: {wav.getnchannels()} (should be 1)")
        print(f"Sample width: {wav.getsampwidth()} bytes (should be 2)")
        print(f"Sample rate: {wav.getframerate()} Hz (should be 24000)")
        print(f"Frames: {wav.getnframes()}")

check_audio_format('test_audio.wav')
```

### Monitor Transcriptions

```bash
# Watch health endpoint
watch -n 1 'curl -s http://localhost:8001/health | jq .'

# Monitor logs
tail -f logs/moshi-server.log
```

## Production Considerations

### 1. Load Balancing

For high call volume, use multiple STT servers behind a load balancer:

```
┌─────────┐     ┌──────────────┐     ┌──────────┐
│ SIP GW  │────▶│ Load Balancer│────▶│ STT Srv 1│
└─────────┘     │  (HAProxy)   │     ├──────────┤
                └──────────────┘     │ STT Srv 2│
                                     ├──────────┤
                                     │ STT Srv 3│
                                     └──────────┘
```

### 2. Persistent Storage

Store transcriptions in a database:

```javascript
sttWs.on('message', async (data) => {
    const transcript = JSON.parse(data);

    await db.query(
        'INSERT INTO transcripts (call_id, timestamp, text) VALUES (?, ?, ?)',
        [callId, Date.now(), transcript.text]
    );
});
```

### 3. Real-time Delivery

Send transcripts via webhook:

```javascript
const axios = require('axios');

sttWs.on('message', async (data) => {
    await axios.post('https://your-app.com/webhook/transcript', {
        call_id: callId,
        transcript: data,
        timestamp: Date.now()
    });
});
```

## Troubleshooting

### Issue: No audio received
- Check SIP codec (should be μ-law or A-law)
- Verify RTP stream with `tcpdump -i any -w capture.pcap port 5060`

### Issue: Garbled transcription
- Audio format mismatch (check sample rate and encoding)
- Use `sox` to verify: `sox input.wav -n stat`

### Issue: High latency
- Reduce network hops
- Use UDP for RTP
- Check STT server load

## Security

### 1. Authentication

Add API key authentication to the bridge:

```javascript
const API_KEY = process.env.STT_API_KEY;

sttWs = new WebSocket(STT_WS_URL, {
    headers: { 'Authorization': `Bearer ${API_KEY}` }
});
```

### 2. Encryption

Use TLS for all connections:
- SIP: SIPS (TLS)
- WebSocket: WSS (TLS)
- HTTP: HTTPS

### 3. Rate Limiting

Implement rate limiting to prevent abuse:

```javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // limit each IP to 100 requests per windowMs
});

app.use(limiter);
```

## Additional Resources

- Asterisk ARI: https://wiki.asterisk.org/wiki/display/AST/Asterisk+REST+Interface
- FreeSWITCH: https://freeswitch.org/confluence/
- Twilio Streams: https://www.twilio.com/docs/voice/twiml/stream
- μ-law codec: https://en.wikipedia.org/wiki/G.711
