# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libomp-dev ca-certificates libssl-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3.11 is default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install Python deps (use prebuilt CUDA wheel for mistralrs)
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir runpod mistralrs-cuda -v \
    && python -m pip show runpod \
    && python -m pip show mistralrs-cuda
# Copy chat templates (optional if mistralrs-cuda already ships them)
WORKDIR /chat_templates
COPY chat_templates /chat_templates

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