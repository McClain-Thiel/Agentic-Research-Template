#!/usr/bin/env bash
# SLURM launcher — STUB.
#
# Should generate an sbatch script, submit via sbatch, poll the job, and
# sync results on completion.
#
# To enable: implement the body and ensure SLURM_* env vars are set
# (account, partition, qos, time, gpus, cpus, mem) in .env.

set -euo pipefail

echo "[slurm] Launcher is a stub. Implement it before using LAUNCHER=slurm." >&2
exit 1
