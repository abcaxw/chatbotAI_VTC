# CUDA runtime + cuDNN (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Python + system deps for OCR/PDF/CV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    tesseract-ocr tesseract-ocr-vie \
    poppler-utils \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg wget curl ca-certificates git dos2unix \
  && rm -rf /var/lib/apt/lists/*

# Ensure python/pip shims
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install Python deps (clean requirements file first)
COPY requirements.txt /app/requirements.txt
RUN dos2unix /app/requirements.txt || true && \
    sed -i '1s/^\xEF\xBB\xBF//' /app/requirements.txt && \
    grep -q "^# -*- coding: utf-8 -*-" /app/requirements.txt || sed -i '1i # -*- coding: utf-8 -*-' /app/requirements.txt && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Runtime dirs
RUN mkdir -p /app/uploads /app/models

# Expose app ports (document-api:8000, rag-api:8501)
EXPOSE 8000 8501

# Default self-test (compose will override)
CMD ["bash", "-lc", "python - <<'PY'\nimport torch,sys\nprint('torch', torch.__version__, 'cuda_build', getattr(torch.version,'cuda',None), 'cuda_available', torch.cuda.is_available())\nPY"]
