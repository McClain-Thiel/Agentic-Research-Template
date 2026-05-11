#!/usr/bin/env python3
"""Ray launcher — STUB.

Should submit a job to a Ray cluster (RAY_CLUSTER_ADDRESS), stream logs,
and sync results on completion. To enable: implement the body and set
RAY_CLUSTER_ADDRESS (or ANYSCALE_API_KEY) in .env.
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "[ray] Launcher is a stub. Implement it before using LAUNCHER=ray.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
