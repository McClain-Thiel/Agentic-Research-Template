#!/usr/bin/env bash
# EC2 launcher — STUB.
#
# Should start an EC2 instance (or use a pre-existing one), rsync the
# project, run training over SSH, rsync results back, then stop the
# instance.
#
# To enable: implement the body and set EC2_* env vars in .env.

set -euo pipefail

echo "[ec2] Launcher is a stub. Implement it before using LAUNCHER=ec2." >&2
exit 1
