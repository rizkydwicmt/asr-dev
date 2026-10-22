import os
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

CRISPASR = os.getenv("CRISPASR_BIN", "/usr/local/bin/crispasr")
MODEL = os.getenv("ASR_MODEL", "/models/ggml-large-v3-turbo.bin")
VAD_MODEL = os.getenv("ASR_VAD_MODEL", "/models/ggml-silero-v6.2.0.bin")
DIARIZE_MODEL = os.getenv("ASR_DIARIZE_MODEL", "/models/Nemotron-3-Diarization.q8_0.gguf")
PROMPT = os.getenv(
    "ASR_PROMPT",
    "Telepon call center. Percakapan bahasa Indonesia sehari-hari. "
    "Istilah umum: janji temu, appointment, waiting list, self check-in, "
    "WhatsApp, asuransi, nomor telepon.",
)
WORK_DIR = Path(os.getenv("ASR_WORK_DIR", "/tmp/asr-jobs"))
TIMEOUT = int(os.getenv("ASR_TIMEOUT", "600"))
MAX_UPLOAD = int(os.getenv("ASR_MAX_UPLOAD_MB", "200"))

WORK_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="ASR Dev", version="dev")


@app.get("/api/health")
def health():
    try:
        v = subprocess.run([CRISPASR, "--version"], capture_output=True, text=True, timeout=15)
        return {"status": "ok", "crispasr": v.stdout.strip() or v.stderr.strip()}
    except Exception as e:  # pragma: no cover
        raise HTTPException(503, f"crispasr not healthy: {e}")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(True),
):
    if (file.size or 0) > MAX_UPLOAD * 1024 * 1024:
        raise HTTPException(413, f"file > {MAX_UPLOAD}MB")

    ext = Path(file.filename or "audio.ogg").suffix.lower() or ".ogg"
    if ext not in {".ogg", ".wav", ".mp3", ".m4a", ".flac"}:
        raise HTTPException(415, f"format {ext} tidak didukung; pakai ogg/wav/mp3/m4a/flac")

    job = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job
    job_dir.mkdir()
    src = job_dir / f"input{ext}"
    src.write_bytes(await file.read())

    prefix = str(job_dir / "out")
    cmd = [
        CRISPASR,
        "--model", MODEL,
        "--file", str(src),
        "-l", "id",
        "-mc", "0", "-sns",
        "-et", "2.2", "-lpt", "-1.0", "-nth", "0.6", "-bs", "5",
        "--prompt", PROMPT, "--carry-initial-prompt",
        "--vad", "--vad-model", VAD_MODEL,
    ]
    if diarize:
        cmd += ["--diarize", "--diarize-method", "sortformer", "--diarize-model", DIARIZE_MODEL]
    cmd += ["-otxt", "-osrt", "-of", prefix]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"transcribe timeout >{TIMEOUT}s")
    if proc.returncode != 0:
        raise HTTPException(500, f"crispasr gagal: {proc.stderr[-800:]}")

    text = _read(Path(prefix + ".txt"))
    srt = _read(Path(prefix + ".srt"))
    if not text.strip():
        raise HTTPException(500, "output kosong")

    return JSONResponse(
        {
            "job_id": job,
            "filename": file.filename,
            "diarize": diarize,
            "transcript": text.strip(),
            "srt": srt,
            "srt_url": f"/api/jobs/{job}/srt",
        }
    )


@app.get("/api/jobs/{job}/srt")
def job_srt(job: str):
    p = WORK_DIR / job / "out.srt"
    if not p.exists():
        raise HTTPException(404, "job tidak ditemukan")
    return PlainTextResponse(p.read_text(encoding="utf-8", errors="replace"), media_type="text/plain")


@app.get("/api/jobs/{job}/audio-original")
def job_audio(job: str):
    for f in (WORK_DIR / job).glob("input.*"):
        return FileResponse(f)
    raise HTTPException(404, "job tidak ditemukan")


app.mount("/", StaticFiles(directory="/opt/app/static", html=True), name="ui")
