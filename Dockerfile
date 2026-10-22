# ---- Stage 1: Builder (compile CrispASR) ----
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
COPY crispasr/ /opt/CrispASR/
RUN cd /opt/CrispASR \
 && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_OPENMP=ON -DCRISPASR_BUILD_TESTS=OFF \
 && cmake --build build -j"$(nproc)" \
 && cp build/bin/crispasr /usr/local/bin/crispasr

# ---- Stage 2: Runtime (slim, TANPA models — di-fetch initContainer dari HF) ----
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
      ffmpeg curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/crispasr /usr/local/bin/crispasr
COPY app/ /opt/app/
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r /opt/app/requirements.txt
ENV PATH="/opt/venv/bin:$PATH"
# Models di-mount ke /models oleh initContainer (lihat chart deployment.yaml)
WORKDIR /opt/app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
