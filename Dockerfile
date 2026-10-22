FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates \
      python3 python3-venv python3-pip \
      ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Build CrispASR (Linux, tanpa Metal) ----
WORKDIR /opt
COPY crispasr/ /opt/CrispASR/
RUN cd /opt/CrispASR \
 && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_OPENMP=ON -DCRISPASR_BUILD_TESTS=OFF \
 && cmake --build build -j"$(nproc)" \
 && cp build/bin/crispasr /usr/local/bin/crispasr \
 && crispasr --version

# ---- Python API ----
COPY app/ /opt/app/
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r /opt/app/requirements.txt
ENV PATH="/opt/venv/bin:$PATH"

# ---- Models (baked untuk dev; produksi: GCS/init-container) ----
COPY models/ /models/

WORKDIR /opt/app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
