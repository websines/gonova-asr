#!/bin/bash

# Prerequisites Installer for Kyutai STT
# Run this once on your CUDA system before deploying

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Kyutai STT - Prerequisites Installer"
echo "=========================================="
echo ""

# Check OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo -e "${GREEN}OS: Linux${NC}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
    echo -e "${YELLOW}OS: macOS (CUDA not available)${NC}"
else
    echo -e "${RED}Unsupported OS${NC}"
    exit 1
fi

echo ""

# 1. Check/Install Rust
echo -e "${YELLOW}[1/4] Checking Rust installation...${NC}"
if command -v cargo &> /dev/null; then
    echo -e "${GREEN}✓ Rust already installed: $(rustc --version)${NC}"
else
    echo -e "${YELLOW}Installing Rust...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    echo -e "${GREEN}✓ Rust installed successfully${NC}"
fi

echo ""

# 2. Check CUDA
echo -e "${YELLOW}[2/4] Checking CUDA installation...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}✓ CUDA installed: $CUDA_VERSION${NC}"

    # Check nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA driver installed${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo -e "${YELLOW}⚠ nvidia-smi not found, but nvcc is present${NC}"
    fi
else
    echo -e "${RED}✗ CUDA not found${NC}"
    echo ""
    echo "Please install CUDA Toolkit from:"
    echo "https://developer.nvidia.com/cuda-downloads"
    echo ""
    echo "After installation, add to ~/.bashrc or ~/.zshrc:"
    echo 'export PATH=/usr/local/cuda/bin:$PATH'
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
    echo ""
    echo "Then run: source ~/.bashrc (or ~/.zshrc)"
    exit 1
fi

echo ""

# 3. Check/Install OpenSSL development libraries
echo -e "${YELLOW}[3/4] Checking OpenSSL development libraries...${NC}"
if [ "$OS" == "linux" ]; then
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        if dpkg -l | grep -q libssl-dev; then
            echo -e "${GREEN}✓ OpenSSL dev libraries installed${NC}"
        else
            echo -e "${YELLOW}Installing OpenSSL dev libraries...${NC}"
            sudo apt-get update
            sudo apt-get install -y libssl-dev pkg-config
            echo -e "${GREEN}✓ OpenSSL dev libraries installed${NC}"
        fi
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        if rpm -qa | grep -q openssl-devel; then
            echo -e "${GREEN}✓ OpenSSL dev libraries installed${NC}"
        else
            echo -e "${YELLOW}Installing OpenSSL dev libraries...${NC}"
            sudo yum install -y openssl-devel pkg-config
            echo -e "${GREEN}✓ OpenSSL dev libraries installed${NC}"
        fi
    elif command -v dnf &> /dev/null; then
        # Fedora
        if rpm -qa | grep -q openssl-devel; then
            echo -e "${GREEN}✓ OpenSSL dev libraries installed${NC}"
        else
            echo -e "${YELLOW}Installing OpenSSL dev libraries...${NC}"
            sudo dnf install -y openssl-devel pkg-config
            echo -e "${GREEN}✓ OpenSSL dev libraries installed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Unknown package manager. Please install openssl-dev manually.${NC}"
    fi
else
    echo -e "${GREEN}✓ macOS - OpenSSL available via system${NC}"
fi

echo ""

# 4. Check Python
echo -e "${YELLOW}[4/4] Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Python installed: $PYTHON_VERSION${NC}"

    # Python is installed, check for dev headers
    if [ "$OS" == "linux" ]; then
        echo -e "${YELLOW}Checking Python development headers...${NC}"
        if command -v apt-get &> /dev/null; then
            if dpkg -l | grep -q python3-dev; then
                echo -e "${GREEN}✓ Python dev headers installed${NC}"
            else
                echo -e "${YELLOW}Installing Python dev headers...${NC}"
                sudo apt-get install -y python3-dev
                echo -e "${GREEN}✓ Python dev headers installed${NC}"
            fi
        elif command -v yum &> /dev/null || command -v dnf &> /dev/null; then
            PKG_MGR=$(command -v dnf &> /dev/null && echo "dnf" || echo "yum")
            if rpm -qa | grep -q python3-devel; then
                echo -e "${GREEN}✓ Python dev headers installed${NC}"
            else
                echo -e "${YELLOW}Installing Python dev headers...${NC}"
                sudo $PKG_MGR install -y python3-devel
                echo -e "${GREEN}✓ Python dev headers installed${NC}"
            fi
        fi
    fi
else
    echo -e "${YELLOW}Python not found. Installing...${NC}"
    if [ "$OS" == "linux" ]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-dev
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3 python3-pip python3-devel
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3 python3-pip python3-devel
        else
            echo -e "${RED}Cannot auto-install Python. Please install manually.${NC}"
            exit 1
        fi
    fi
    echo -e "${GREEN}✓ Python installed${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}All prerequisites installed!${NC}"
echo "=========================================="
echo ""
echo "You can now run the deployment:"
echo "  ./deployment/deploy.sh"
echo ""

# Show system info
echo "System Information:"
echo "  Rust: $(rustc --version)"
if command -v nvcc &> /dev/null; then
    echo "  CUDA: $(nvcc --version | grep release | awk '{print $5}')"
fi
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi
echo "  Python: $(python3 --version)"
