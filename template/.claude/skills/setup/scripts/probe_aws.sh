#!/usr/bin/env bash
# Probe AWS configuration and resource access.
# Exits 0 on success, non-zero on failure.

set -euo pipefail

echo "=== AWS Probe ==="

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "[FAIL] AWS CLI not found -- install with: pip install awscli"
    exit 1
fi

# Check credentials via get-caller-identity
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "[FAIL] AWS credentials not configured -- run: aws configure"
    exit 1
fi

IDENTITY=$(aws sts get-caller-identity --output json 2>/dev/null)
ACCOUNT=$(echo "${IDENTITY}" | grep -oP '"Account": "\K[0-9]+' || echo "unknown")
USER=$(echo "${IDENTITY}" | grep -oP '"Arn": "[^"]+"' | sed 's/.*\///;s/"$//')
echo "[OK] Authenticated -- Account: ${ACCOUNT}, User: ${USER}"

# Check S3 bucket access if configured
if [ "${STORAGE_BACKEND:-}" = "s3" ] && [ -n "${S3_BUCKET:-}" ]; then
    echo "Checking S3 bucket access: ${S3_BUCKET}..."
    if aws s3 ls "s3://${S3_BUCKET}" &> /dev/null; then
        echo "[OK] S3 bucket accessible: ${S3_BUCKET}"
    else
        echo "[FAIL] Cannot access S3 bucket: ${S3_BUCKET}"
        exit 1
    fi
fi

# Check Batch job definition if launcher is aws_batch
if [ "${LAUNCHER:-}" = "aws_batch" ] && [ -n "${BATCH_JOB_DEFINITION:-}" ]; then
    echo "Checking Batch job definition: ${BATCH_JOB_DEFINITION}..."
    if aws batch describe-job-definitions \
        --job-definition-name "${BATCH_JOB_DEFINITION}" \
        --status ACTIVE &> /dev/null; then
        echo "[OK] Batch job definition found: ${BATCH_JOB_DEFINITION}"
    else
        echo "[FAIL] Batch job definition not found: ${BATCH_JOB_DEFINITION}"
        exit 1
    fi
fi

# Check EC2 launch template if launcher is ec2
if [ "${LAUNCHER:-}" = "ec2" ] && [ -n "${EC2_LAUNCH_TEMPLATE:-}" ]; then
    echo "Checking EC2 launch template: ${EC2_LAUNCH_TEMPLATE}..."
    if aws ec2 describe-launch-template-versions \
        --launch-template-id "${EC2_LAUNCH_TEMPLATE}" \
        &> /dev/null; then
        echo "[OK] EC2 launch template found: ${EC2_LAUNCH_TEMPLATE}"
    else
        echo "[FAIL] EC2 launch template not found: ${EC2_LAUNCH_TEMPLATE}"
        exit 1
    fi
fi

echo "[OK] AWS probe passed"
exit 0
