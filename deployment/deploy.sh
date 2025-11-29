#!/bin/bash

# Kyutai STT Deployment Script for CUDA Systems
# This script sets up and runs the complete STT system with health checks

set -e

echo "=========================================="
echo "Kyutai STT System - Deployment Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration - Hardcoded ports
MOSHI_PORT=9000
HEALTH_PORT=8001
WEB_PORT=9002
STT_MODEL=${STT_MODEL:-"kyutai/stt-1b-en_fr"}
CONFIG_FILE=${CONFIG_FILE:-"configs/config-stt-en_fr-hf.toml"}

echo -e "${GREEN}Configuration:${NC}"
echo "  Moshi Server Port: $MOSHI_PORT"
echo "  Health Check Port: $HEALTH_PORT"
echo "  Web Client Port: $WEB_PORT"
echo "  STT Model: $STT_MODEL"
echo "  Config File: $CONFIG_FILE"
echo ""

# Check for CUDA
echo -e "${YELLOW}Checking CUDA availability...${NC}"
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}WARNING: nvcc not found. CUDA may not be available.${NC}"
    echo "The system will attempt to continue, but GPU acceleration may not work."
else
    echo -e "${GREEN}CUDA found: $(nvcc --version | grep release)${NC}"
fi
echo ""

# Check for Rust
echo -e "${YELLOW}Checking Rust installation...${NC}"
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}ERROR: Rust/Cargo not found.${NC}"
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi
echo -e "${GREEN}Rust found: $(rustc --version)${NC}"
echo ""

# Install moshi-server if not already installed
echo -e "${YELLOW}Checking moshi-server installation...${NC}"
if ! command -v moshi-server &> /dev/null; then
    echo -e "${YELLOW}Installing moshi-server with CUDA support...${NC}"
    cargo install --features cuda moshi-server
    echo -e "${GREEN}moshi-server installed successfully${NC}"
else
    echo -e "${GREEN}moshi-server already installed${NC}"
fi
echo ""

# Build health check service
echo -e "${YELLOW}Building health check service...${NC}"
cd services/health-check
cargo build --release
cd ../..
echo -e "${GREEN}Health check service built successfully${NC}"
echo ""

# Create systemd service files (optional)
create_systemd_services() {
    echo -e "${YELLOW}Creating systemd service files...${NC}"

    # Moshi server service
    cat > /tmp/kyutai-stt.service << EOF
[Unit]
Description=Kyutai STT WebSocket Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(which moshi-server) worker --config $CONFIG_FILE --port $MOSHI_PORT
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Health check service
    cat > /tmp/kyutai-stt-health.service << EOF
[Unit]
Description=Kyutai STT Health Check Service
After=network.target kyutai-stt.service
Requires=kyutai-stt.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="WEBSOCKET_URL=ws://localhost:$MOSHI_PORT"
Environment="HEALTH_PORT=$HEALTH_PORT"
ExecStart=$(pwd)/services/health-check/target/release/stt-health-check
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    echo -e "${GREEN}Systemd service files created in /tmp/${NC}"
    echo "To install them, run:"
    echo "  sudo mv /tmp/kyutai-stt*.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable kyutai-stt kyutai-stt-health"
    echo "  sudo systemctl start kyutai-stt kyutai-stt-health"
    echo ""
}

# Start services
start_services() {
    echo -e "${YELLOW}Starting services...${NC}"
    echo ""

    # Start moshi-server in background
    echo -e "${GREEN}Starting Moshi WebSocket Server on port $MOSHI_PORT...${NC}"
    moshi-server worker --config $CONFIG_FILE --port $MOSHI_PORT > logs/moshi-server.log 2>&1 &
    MOSHI_PID=$!
    echo "Moshi Server PID: $MOSHI_PID"
    echo $MOSHI_PID > logs/moshi.pid

    # Wait for moshi-server to start
    echo "Waiting for Moshi server to initialize..."
    sleep 5

    # Start health check service in background
    echo -e "${GREEN}Starting Health Check Service on port $HEALTH_PORT...${NC}"
    WEBSOCKET_URL="ws://localhost:$MOSHI_PORT" HEALTH_PORT=$HEALTH_PORT \
        ./services/health-check/target/release/stt-health-check > logs/health-check.log 2>&1 &
    HEALTH_PID=$!
    echo "Health Check PID: $HEALTH_PID"
    echo $HEALTH_PID > logs/health.pid

    # Start simple web server for the client
    echo -e "${GREEN}Starting Web Client Server on port $WEB_PORT...${NC}"
    cd web-client
    python3 -m http.server $WEB_PORT > ../logs/web-server.log 2>&1 &
    WEB_PID=$!
    echo "Web Server PID: $WEB_PID"
    echo $WEB_PID > ../logs/web.pid
    cd ..

    echo ""
    echo -e "${GREEN}=========================================="
    echo "All services started successfully!"
    echo "==========================================${NC}"
    echo ""
    echo "Service URLs:"
    echo "  WebSocket Server: ws://localhost:$MOSHI_PORT"
    echo "  Health Check: http://localhost:$HEALTH_PORT/health"
    echo "  Web Client: http://localhost:$WEB_PORT"
    echo ""
    echo "Logs are available in the 'logs/' directory"
    echo ""
    echo "To stop all services, run: ./deployment/stop.sh"
}

# Create logs directory
mkdir -p logs

# Main menu
echo "What would you like to do?"
echo "1) Start services now"
echo "2) Create systemd service files (for production)"
echo "3) Both"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        start_services
        ;;
    2)
        create_systemd_services
        ;;
    3)
        create_systemd_services
        start_services
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac
