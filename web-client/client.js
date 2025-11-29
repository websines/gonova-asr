class STTClient {
    constructor() {
        this.ws = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.audioWorkletNode = null;
        this.isRecording = false;
        this.isConnected = false;

        this.initializeUI();
    }

    initializeUI() {
        this.connectBtn = document.getElementById('connectBtn');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.serverUrlInput = document.getElementById('serverUrl');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.transcript = document.getElementById('transcript');
        this.errorMsg = document.getElementById('errorMsg');

        this.connectBtn.addEventListener('click', () => this.toggleConnection());
        this.startBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
    }

    showError(message) {
        this.errorMsg.textContent = message;
        this.errorMsg.classList.add('show');
        setTimeout(() => {
            this.errorMsg.classList.remove('show');
        }, 5000);
    }

    updateStatus(status, dotClass = '') {
        this.statusText.textContent = status;
        this.statusDot.className = 'status-dot ' + dotClass;
    }

    async toggleConnection() {
        if (this.isConnected) {
            this.disconnect();
        } else {
            await this.connect();
        }
    }

    async connect() {
        const serverUrl = this.serverUrlInput.value.trim();
        if (!serverUrl) {
            this.showError('Please enter a server URL');
            return;
        }

        this.updateStatus('Connecting...', '');
        this.connectBtn.disabled = true;

        try {
            this.ws = new WebSocket(serverUrl);

            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateStatus('Connected', 'connected');
                this.connectBtn.textContent = 'Disconnect';
                this.connectBtn.disabled = false;
                this.startBtn.disabled = false;
                console.log('WebSocket connected');
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showError('WebSocket connection error');
                this.disconnect();
            };

            this.ws.onclose = (event) => {
                console.log('WebSocket closed:', {
                    code: event.code,
                    reason: event.reason,
                    wasClean: event.wasClean
                });
                this.disconnect();
            };

        } catch (error) {
            console.error('Connection error:', error);
            this.showError('Failed to connect: ' + error.message);
            this.connectBtn.disabled = false;
        }
    }

    disconnect() {
        if (this.isRecording) {
            this.stopRecording();
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.isConnected = false;
        this.updateStatus('Disconnected', '');
        this.connectBtn.textContent = 'Connect';
        this.connectBtn.disabled = false;
        this.startBtn.disabled = true;
        this.stopBtn.disabled = true;
    }

    async handleMessage(data) {
        try {
            // Server sends msgpack-encoded binary messages
            if (data instanceof Blob) {
                const arrayBuffer = await data.arrayBuffer();
                const message = MessagePack.decode(new Uint8Array(arrayBuffer));
                console.log('Received message:', message);

                if (message.type === 'Word' && message.text) {
                    this.appendTranscript(message.text);
                } else if (message.type === 'EndWord') {
                    // Word boundary - could add punctuation or spacing logic here
                } else if (message.type === 'Step') {
                    // VAD predictions - could visualize pause detection
                } else if (message.type === 'Marker') {
                    console.log('Stream marker received:', message.id);
                }
            } else if (typeof data === 'string') {
                // Fallback for text messages
                const message = JSON.parse(data);
                console.log('Received text message:', message);
                if (message.text) {
                    this.appendTranscript(message.text);
                }
            }
        } catch (error) {
            console.error('Error handling message:', error);
        }
    }

    appendTranscript(text) {
        this.transcript.textContent += text + ' ';
        // Auto-scroll to bottom
        this.transcript.parentElement.scrollTop = this.transcript.parentElement.scrollHeight;
    }

    async startRecording() {
        try {
            this.updateStatus('Requesting microphone access...', '');

            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 24000, // Kyutai STT expects 24kHz
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                }
            });

            // Create audio context with 24kHz sample rate
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 24000
            });

            const source = this.audioContext.createMediaStreamSource(this.mediaStream);

            // Create script processor for audio processing
            // Using 4096 buffer size for ~170ms chunks at 24kHz
            const bufferSize = 4096;
            this.audioWorkletNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

            // Create msgpack encoder with float32 forced (required by moshi-server)
            const msgpackEncoder = new MessagePack.Encoder({ forceFloat32: true });

            let audioChunkCount = 0;
            this.audioWorkletNode.onaudioprocess = (event) => {
                if (this.isRecording && this.ws && this.ws.readyState === WebSocket.OPEN) {
                    const inputData = event.inputBuffer.getChannelData(0);

                    // Convert Float32Array to regular array of floats
                    const pcmArray = Array.from(inputData);

                    // Send as msgpack-encoded message (format expected by moshi-server)
                    const message = { type: "Audio", pcm: pcmArray };
                    const encoded = msgpackEncoder.encode(message);

                    try {
                        this.ws.send(encoded);
                        audioChunkCount++;
                        if (audioChunkCount <= 3) {
                            console.log(`Sent audio chunk #${audioChunkCount}, size: ${encoded.byteLength} bytes, samples: ${pcmArray.length}`);
                        }
                    } catch (e) {
                        console.error('Error sending audio:', e);
                    }
                }
            };

            source.connect(this.audioWorkletNode);
            this.audioWorkletNode.connect(this.audioContext.destination);

            this.isRecording = true;
            this.updateStatus('Recording...', 'recording');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.connectBtn.disabled = true;

            console.log('Recording started');

        } catch (error) {
            console.error('Error starting recording:', error);
            this.showError('Failed to access microphone: ' + error.message);
            this.updateStatus('Connected', 'connected');
        }
    }

    stopRecording() {
        this.isRecording = false;

        if (this.audioWorkletNode) {
            this.audioWorkletNode.disconnect();
            this.audioWorkletNode = null;
        }

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        this.updateStatus('Connected', 'connected');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.connectBtn.disabled = false;

        console.log('Recording stopped');
    }

}

// Initialize the client when the page loads
const sttClient = new STTClient();

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (sttClient.isRecording) {
        sttClient.stopRecording();
    }
    if (sttClient.isConnected) {
        sttClient.disconnect();
    }
});
