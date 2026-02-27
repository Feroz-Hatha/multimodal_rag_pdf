#!/usr/bin/env bash
# =============================================================================
# EC2 Setup Script — PDF Multimodal RAG
# Run this once after SSHing into a fresh Ubuntu 22.04 EC2 instance.
# Usage: bash deploy/setup_ec2.sh
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/Feroz-Hatha/multimodal_rag_pdf.git"
APP_DIR="$HOME/PDF_Multimodal_RAG"
VENV="$APP_DIR/py312_pdf_rag"

echo "=== [1/8] System update ==="
sudo apt-get update -y && sudo apt-get upgrade -y

echo "=== [2/8] System dependencies ==="
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -y
sudo apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    python3-pip \
    poppler-utils \
    libgl1 libglib2.0-0 \
    nginx \
    git \
    curl

echo "=== [3/8] Node.js 20 (for building React) ==="
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

echo "=== [4/8] Clone repository ==="
git clone "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"

echo "=== [5/8] Python virtual environment ==="
python3.12 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r requirements.txt

echo "=== [6/8] Build React frontend ==="
cd "$APP_DIR/web"
npm ci
npm run build
cd "$APP_DIR"

echo "=== [7/8] Create data directories ==="
mkdir -p data/uploads data/processed data/chroma_db

echo "=== [8/8] Nginx + systemd ==="
# Nginx site config
sudo cp deploy/nginx.conf /etc/nginx/sites-available/pdf-rag
sudo ln -sf /etc/nginx/sites-available/pdf-rag /etc/nginx/sites-enabled/pdf-rag
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# systemd service
sudo cp deploy/pdf-rag.service /etc/systemd/system/pdf-rag.service
sudo systemctl daemon-reload
sudo systemctl enable pdf-rag

echo ""
echo "============================================================"
echo " Setup complete!"
echo " Next: create .env (see .env.example), then run:"
echo "   sudo systemctl start pdf-rag"
echo "   sudo systemctl status pdf-rag"
echo "============================================================"
