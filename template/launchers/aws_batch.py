#!/usr/bin/env python3
"""AWS Batch launcher — STUB.

Should submit a Batch job via boto3, poll for completion, stream logs from
CloudWatch, and sync results on success. To enable: implement the body and
set BATCH_* env vars in .env.
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "[aws_batch] Launcher is a stub. Implement it before using LAUNCHER=aws_batch.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
