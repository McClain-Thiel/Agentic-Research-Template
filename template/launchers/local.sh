#!/usr/bin/env bash
# Local launcher — runs training synchronously on the local machine.
# Usage: ./launchers/local.sh <command> [args...]
#
# Writes status file before and after, calls push-results on completion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source environment file if set
if [ -n "${ENV_FILE:-}" ] && [ -f "${ENV_FILE}" ]; then
    set -a
    source "${ENV_FILE}"
    set +a
fi

# Build the command
COMMAND="$*"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LAUNCHER="local"
HOSTNAME="$(hostname)"
PID="$$"

# Status file
mkdir -p "${PROJECT_ROOT}/journal/runs"
STATUS_FILE="${PROJECT_ROOT}/journal/runs/${RUN_ID}.yaml"

# Write pending status
write_status() {
    local status="$1"
    local completed_at="${2:-null}"
    cat > "${STATUS_FILE}" <<EOF
run_id: ${RUN_ID}
launcher: ${LAUNCHER}
status: ${status}
pid: ${PID}
host: ${HOSTNAME}
started_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
completed_at: ${completed_at}
command: ${COMMAND}
EOF
}

echo "[local] Starting run ${RUN_ID}"
write_status "pending"

# Update to running
write_status "running"

# Run the command
if eval "${COMMAND}"; then
    echo "[local] Run ${RUN_ID} completed successfully"
    write_status "completed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # Push results to storage
    cd "${PROJECT_ROOT}"
    uv run python -c "from [[ package_name ]].infra.storage import sync_results; sync_results()" || true
else
    echo "[local] Run ${RUN_ID} failed"
    write_status "failed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 1
fi
