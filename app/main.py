import os
import subprocess
import threading
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

# registry job in-memory (dev) — status + hasil
JOBS: dict[str, dict] = {}


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def _run_job(job_id: str, cmd: list[str]):
    """Jalan di thread background — crispasr proses, hasil ditandai lewat out.txt."""
    JOBS[job_id]["status"] = "processing"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        if proc.returncode != 0:
            JOBS[job_id].update(status="error", error=(proc.stderr or proc.stdout)[-800:])
            return
        text = _read(WORK_DIR / job_id / "out.txt")
        if not text.strip():
            JOBS[job_id].update(status="error", error="output kosong")
            return
        JOBS[job_id].update(
            status="done", transcript=text.strip(), srt=_read(WORK_DIR / job_id / "out.srt")
        )
    except subprocess.TimeoutExpired:
        JOBS[job_id].update(status="error", error=f"timeout >{TIMEOUT}s")
    except Exception as e:  # pragma: no cover
        JOBS[job_id].update(status="error", error=str(e))


def _build_cmd(job_id: str, src: Path, diarize: bool) -> list[str]:
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
    cmd += ["-otxt", "-osrt", "-of", str(WORK_DIR / job_id / "out")]
    return cmd


@app.get("/api/health")
def health():
    try:
        v = subprocess.run([CRISPASR, "--version"], capture_output=True, text=True, timeout=15)
    except Exception as e:  # pragma: no cover
        raise HTTPException(503, f"crispasr not healthy: {e}")
    if v.returncode != 0:
        # jujur: binary rusak/library hilang → 503, bukan 200 dengan pesan error
        raise HTTPException(503, f"crispasr broken: {(v.stderr or v.stdout).strip()[:300]}")
    return {"status": "ok", "crispasr": (v.stdout or v.stderr).strip()}


@app.post("/api/transcribe")
def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(True),
    wait: bool = Form(False),  # true = mode lama (blok sampai beres; buat curl)
):
    if (file.size or 0) > MAX_UPLOAD * 1024 * 1024:
        raise HTTPException(413, f"file > {MAX_UPLOAD}MB")

    ext = Path(file.filename or "audio.ogg").suffix.lower() or ".ogg"
    if ext not in {".ogg", ".wav", ".mp3", ".m4a", ".flac"}:
        raise HTTPException(415, f"format {ext} tidak didukung; pakai ogg/wav/mp3/m4a/flac")

    job = uuid.uuid4().hex[:12]
    (WORK_DIR / job).mkdir()
    src = WORK_DIR / job / f"input{ext}"
    src.write_bytes(file.file.read())

    JOBS[job] = {"status": "queued"}
    cmd = _build_cmd(job, src, diarize)
    threading.Thread(target=_run_job, args=(job, cmd), daemon=True).start()

    if wait:  # mode kompatibel (curl): blok sampai selesai
        import time
        while JOBS[job]["status"] in ("queued", "processing"):
            time.sleep(2)
        j = JOBS[job]
        if j["status"] == "error":
            raise HTTPException(500, j["error"])
        return JSONResponse(
            {
                "job_id": job,
                "filename": file.filename,
                "diarize": diarize,
                "transcript": j["transcript"],
                "srt": j["srt"],
                "srt_url": f"/api/jobs/{job}/srt",
            }
        )

    # mode default: langsung balik job_id — client polling
    return JSONResponse(
        {"job_id": job, "status": "queued", "poll_url": f"/api/jobs/{job}"}
    )


@app.get("/api/jobs/{job}")
def job_status(job: str):
    j = JOBS.get(job)
    if j:
        resp = {"job_id": job, "status": j["status"]}
        if j["status"] == "done":
            resp.update(
                transcript=j["transcript"], srt=j["srt"], srt_url=f"/api/jobs/{job}/srt"
            )
        elif j["status"] == "error":
            resp["error"] = j["error"]
        return resp
    # fallback: job dibuat sebelum pod restart — cek file langsung
    d = WORK_DIR / job
    if (d / "out.txt").exists():
        return {
            "job_id": job,
            "status": "done",
            "transcript": _read(d / "out.txt").strip(),
            "srt": _read(d / "out.srt"),
            "srt_url": f"/api/jobs/{job}/srt",
        }
    raise HTTPException(404, "job tidak ditemukan")


@app.get("/api/jobs/{job}/srt")
def job_srt(job: str):
    p = WORK_DIR / job / "out.srt"
    if not p.exists():
        raise HTTPException(404, "job tidak ditemukan")
    return PlainTextResponse(p.read_text(encoding="utf-8", errors="replace"), media_type="text/plain")


app.mount("/", StaticFiles(directory="/opt/app/static", html=True), name="ui")
