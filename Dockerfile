# syntax=docker/dockerfile:1

# ================================
# Stage 1: Builder (optional, caches Rust deps)
# ================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Install Rust + build tools + Python for caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev git \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /mistralrs
COPY . .
ARG CUDA_COMPUTE_CAP=80
ARG RAYON_NUM_THREADS=4
ARG RUST_NUM_THREADS=4
ARG RUSTFLAGS="-Z threads=${RUST_NUM_THREADS}"
ARG WITH_FEATURES="cuda,cudnn"
# Precompile Rust extension to cache deps
RUN cargo build --release -p mistralrs-pyo3


# ================================
# Stage 2: Runtime (Python + CUDA 12.4 + cuDNN + Rust toolchain)
# ================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS runtime
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    libomp-dev ca-certificates libssl-dev curl build-essential git pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3.11 is default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install Rust (needed for maturin develop)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install runtime Python dependencies
RUN pip install --no-cache-dir runpod maturin

# Copy source code
WORKDIR /mistralrs
COPY --from=builder /mistralrs /mistralrs

# Build Rust Python extension directly in runtime
WORKDIR /mistralrs/mistralrs-pyo3
RUN maturin build --release --skip-auditwheel --features cuda

WORKDIR /mistralrs
RUN pip install --no-cache-dir target/wheels/mistralrs-*.whl
# Copy chat templates
COPY --from=builder /mistralrs/chat_templates /chat_templates

# Copy Python handler
WORKDIR /app
COPY handler.py /app/handler.py

# HuggingFace cache
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache \
    PYTHONUNBUFFERED=1

# Ensure NVIDIA runtime visibility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["python", "handler.py"]
