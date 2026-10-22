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
 && mkdir -p /out/bin /out/lib \
 && cp build/bin/crispasr /out/bin/ \
 && find build -name "lib*.so*" -exec cp -a {} /out/lib/ \; \
 && ls /out/lib | head -5

# ---- Stage 2: Runtime (slim, TANPA models — di-fetch initContainer dari HF) ----
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
      ffmpeg curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /out/bin/crispasr /usr/local/bin/crispasr
COPY --from=builder /out/lib/ /usr/local/lib/
RUN ldconfig && crispasr --version
COPY app/ /opt/app/
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r /opt/app/requirements.txt
ENV PATH="/opt/venv/bin:$PATH"
# Models di-mount ke /models oleh initContainer (lihat chart deployment.yaml)
WORKDIR /opt/app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
