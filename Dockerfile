# syntax=docker/dockerfile:1

# ================================
# Stage 1: Build environment (Rust + Python 3.11)
# ================================
FROM python:3.11-slim-bookworm AS builder

# Install Rust + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . "$HOME/.cargo/env"

ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory and copy repo
WORKDIR /mistralrs
COPY . .

# Build only mistralrs-pyo3 (faster than whole workspace)
RUN cargo build --release -p mistralrs-pyo3

# Build Python wheel using maturin (Python 3.11)
WORKDIR /mistralrs/mistralrs-pyo3
RUN pip install maturin \
    && maturin build --release --interpreter python3.11


# ================================
# Stage 2: Runtime (Python 3.11)
# ================================
FROM python:3.11-slim-bookworm AS runtime
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp-dev \
    ca-certificates \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install runpod

# Copy mistralrs-pyo3 Python wheel from builder
COPY --from=builder /mistralrs/target/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl

# Copy chat templates (still useful for some models)
COPY --from=builder /mistralrs/chat_templates /chat_templates

# Copy your Python handler
WORKDIR /app
COPY handler.py /app/handler.py

# HuggingFace cache (persist between runs if mounted)
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache \
    PYTHONUNBUFFERED=1

# Run RunPod handler
CMD ["python", "handler.py"]
