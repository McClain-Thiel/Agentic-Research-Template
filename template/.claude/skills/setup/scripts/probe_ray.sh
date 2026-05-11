#!/usr/bin/env bash
# Probe Ray and Anyscale configuration.
# Exits 0 on success, non-zero on failure.

set -euo pipefail

echo "=== Ray Probe ==="

# Check if ray is installed
if ! python3 -c "import ray" &> /dev/null; then
    echo "[FAIL] Ray not installed -- install with: uv pip install ray[default]"
    exit 1
fi

echo "[OK] Ray installed: $(python3 -c 'import ray; print(ray.__version__)')"

# Check if RAY_CLUSTER_ADDRESS is reachable
if [ -n "${RAY_CLUSTER_ADDRESS:-}" ]; then
    echo "Checking Ray cluster: ${RAY_CLUSTER_ADDRESS}..."
    if python3 -c "
import urllib.request
import sys
try:
    urllib.request.urlopen('${RAY_CLUSTER_ADDRESS}', timeout=5)
    sys.exit(0)
except Exception as e:
    print(f'Connection failed: {e}')
    sys.exit(1)
"; then
        echo "[OK] Ray cluster reachable: ${RAY_CLUSTER_ADDRESS}"
    else
        echo "[WARN] Ray cluster not reachable: ${RAY_CLUSTER_ADDRESS}"
        # Don't fail -- cluster might be down temporarily
    fi
else
    echo "[INFO] RAY_CLUSTER_ADDRESS not set"
fi

# Check Anyscale credentials
if [ -n "${ANYSCALE_API_KEY:-}" ]; then
    echo "Checking Anyscale credentials..."
    if python3 -c "
import urllib.request
import json
import sys
import os
req = urllib.request.Request(
    'https://console.anyscale.com/api/v2/whoami',
    headers={'Authorization': f'Bearer {os.environ[\"ANYSCALE_API_KEY\"]}'}
)
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
        print(f'Authenticated as: {data.get(\"email\", \"unknown\")}')
        sys.exit(0)
except Exception as e:
    print(f'Anyscale auth failed: {e}')
    sys.exit(1)
"; then
        echo "[OK] Anyscale credentials valid"
    else
        echo "[FAIL] Anyscale credentials invalid"
        exit 1
    fi
else
    echo "[INFO] ANYSCALE_API_KEY not set -- Anyscale integration disabled"
fi

echo "[OK] Ray probe passed"
exit 0
