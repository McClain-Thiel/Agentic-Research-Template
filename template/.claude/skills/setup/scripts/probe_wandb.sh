#!/usr/bin/env bash
# Probe Weights & Biases authentication and project access.
# Exits 0 on success, non-zero on failure.

set -euo pipefail

echo "=== Weights & Biases Probe ==="

# Check if wandb is installed
if ! command -v wandb &> /dev/null; then
    echo "[FAIL] wandb CLI not found -- install with: uv pip install wandb"
    exit 1
fi

# Check auth status
echo "Checking W&B authentication..."
if ! wandb login --relogin &> /dev/null; then
    echo "[FAIL] W&B authentication failed -- check your WANDB_API_KEY"
    exit 1
fi

# Get entity info
ENTITY=$(wandb login 2>/dev/null | grep -oP 'Logged in as \K[^ ]+' || echo "")
if [ -n "${ENTITY}" ]; then
    echo "[OK] Authenticated as: ${ENTITY}"
else
    echo "[OK] Authenticated (entity details unavailable)"
fi

# Verify project access
PROJECT="${WANDB_PROJECT:-[[ package_name ]]}"
if [ -n "${WANDB_ENTITY:-}" ]; then
    echo "Checking project access: ${WANDB_ENTITY}/${PROJECT}"
    if wandb projects "${WANDB_ENTITY}" 2>/dev/null | grep -q "${PROJECT}"; then
        echo "[OK] Project '${PROJECT}' accessible"
    else
        echo "[OK] Project '${PROJECT}' not found but can be created"
    fi
else
    echo "[INFO] WANDB_ENTITY not set -- will use default entity"
fi

echo "[OK] W&B probe passed"
exit 0
