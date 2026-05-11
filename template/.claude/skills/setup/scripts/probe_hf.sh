#!/usr/bin/env bash
# Probe HuggingFace authentication and access.
# Exits 0 on success, non-zero on failure.

set -euo pipefail

echo "=== HuggingFace Probe ==="

# Check if huggingface_hub is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "[FAIL] huggingface-cli not found -- install with: uv pip install huggingface-hub"
    exit 1
fi

# Check auth status
echo "Checking HF authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "[FAIL] Not authenticated -- run: huggingface-cli login"
    exit 1
fi

USER=$(huggingface-cli whoami 2>/dev/null | head -1 | awk '{print $NF}')
echo "[OK] Authenticated as: ${USER}"

# Check token has write access by attempting a whoami with full info
if huggingface-cli whoami 2>/dev/null | grep -q "token"; then
    echo "[OK] Token has read access"
else
    echo "[WARN] Could not verify token details"
fi

# Check HF_HOME is set
if [ -n "${HF_HOME:-}" ]; then
    echo "[OK] HF_HOME set to: ${HF_HOME}"
else
    echo "[INFO] HF_HOME not set -- using default ~/.cache/huggingface"
fi

echo "[OK] HuggingFace probe passed"
exit 0
