# Multi-stage Dockerfile for [[ package_name ]]
# Targets:
#   base   - Runtime dependencies only
#   train  - Training image (production)
#   dev    - Development environment

# =============================================================================
# Base stage: CUDA runtime + Python + runtime dependencies
# =============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy project configuration
COPY pyproject.toml ./

# Create a minimal uv.lock if one doesn't exist (for template)
# In production, copy an existing uv.lock for reproducible builds
RUN uv sync --no-dev --no-install-project || uv pip install torch pydantic pydantic-settings fsspec wandb huggingface-hub hf-xet marimo rich loguru

# =============================================================================
# Train stage: Production training image
# =============================================================================
FROM base AS train

# Copy source code
COPY src/ ./src/

# Install the package in non-editable mode
RUN uv pip install --no-deps -e .

# Set entrypoint
ENTRYPOINT ["python", "-m", "[[ package_name ]].train"]

# =============================================================================
# Dev stage: Full development environment
# =============================================================================
FROM base AS dev

# Install all dependencies including dev
RUN uv sync --all-extras

# Copy full repository
COPY . .

# Install the package in editable mode with dev dependencies
RUN uv pip install -e ".[dev]"

# Default to bash for interactive development
CMD ["/bin/bash"]
