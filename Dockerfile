# syntax=docker/dockerfile:1

# ================================
# Stage 1: Build environment (CUDA + Rust + Python 3.11)
# ================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN <<HEREDOC
    apt-get update
    apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
        libssl-dev \
        python3.11 \
        python3-pip \
        python3-venv \
        git \
        ca-certificates \
        libomp-dev
    rm -rf /var/lib/apt/lists/*
HEREDOC

# Install Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup update nightly && rustup default nightly

# Set working directory and copy repo
WORKDIR /mistralrs
COPY . .

ARG CUDA_COMPUTE_CAP=80
ARG RAYON_NUM_THREADS=4
ARG RUST_NUM_THREADS=4
ARG RUSTFLAGS="-Z threads=${RUST_NUM_THREADS}"
ARG WITH_FEATURES="cuda,cudnn"
RUN cargo build --release --workspace --features "${WITH_FEATURES}"

# Build only mistralrs-pyo3
WORKDIR /mistralrs/mistralrs-pyo3
RUN pip3 install --no-cache-dir maturin \
    && maturin build --release --interpreter python3.11

# ================================
# Stage 2: Runtime environment (CUDA + Python 3.11)
# ================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN <<HEREDOC
    apt-get update
    apt-get install -y --no-install-recommends \
        libomp-dev \
        ca-certificates \
        libssl-dev \
        python3.11 \
        python3-pip \
        curl \
        git
    rm -rf /var/lib/apt/lists/*
HEREDOC

# Install Python dependencies
RUN pip3 install --no-cache-dir runpod

# Copy mistralrs-pyo3 Python wheel from builder
COPY --from=builder /mistralrs/target/wheels/*.whl /tmp/
RUN pip3 install --no-cache-dir /tmp/*.whl

# Copy chat templates
COPY --from=builder /mistralrs/chat_templates /chat_templates

# Copy Python handler
WORKDIR /app
COPY handler.py /app/handler.py

# Set HuggingFace cache to network volume
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf_cache \
    PYTHONUNBUFFERED=1

# Optional: symlink /workspace → /runpod-volume if handler expects /workspace
RUN ln -s /runpod-volume /workspace || true

# Run RunPod handler
CMD ["python3", "handler.py"]
