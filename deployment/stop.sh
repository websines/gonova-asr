#!/bin/bash

# Kyutai STT - Stop Script
# Stops all running services

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Stopping Kyutai STT Services"
echo "=========================================="
echo ""

# Function to stop a service by PID file
stop_service() {
    local name=$1
    local pid_file=$2

    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping $name (PID: $PID)...${NC}"
            kill $PID
            sleep 2
            if ps -p $PID > /dev/null 2>&1; then
                echo -e "${RED}Force killing $name...${NC}"
                kill -9 $PID
            fi
            echo -e "${GREEN}$name stopped${NC}"
        else
            echo -e "${YELLOW}$name is not running${NC}"
        fi
        rm "$pid_file"
    else
        echo -e "${YELLOW}No PID file found for $name${NC}"
    fi
}

# Stop all services
stop_service "Web Server" "logs/web.pid"
stop_service "Health Check Service" "logs/health.pid"
stop_service "Moshi Server" "logs/moshi.pid"

# Also try to kill by process name as backup
echo ""
echo -e "${YELLOW}Checking for any remaining processes...${NC}"
pkill -f "moshi-server" 2>/dev/null && echo -e "${GREEN}Killed remaining moshi-server processes${NC}" || true
pkill -f "stt-health-check" 2>/dev/null && echo -e "${GREEN}Killed remaining health-check processes${NC}" || true

echo ""
echo -e "${GREEN}All services stopped successfully${NC}"
