import os
import time
import tempfile
import logging
import asyncio
import json
import subprocess
import warnings
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import requests
import uvicorn
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r"std\(\): degrees of freedom is <= 0",
    category=UserWarning,
    module=r"pyannote\.audio\.models\.blocks\.pooling",
)
warnings.filterwarnings(
    "ignore",
    message=r"The MPEG_LAYER_III subtype is unknown to TorchAudio",
    category=UserWarning,
    module=r"torchaudio\._backend\.soundfile_backend",
)

OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
# pyannote 로컬 모델 경로 (다운로드 후 지정)
# 예: ./models/pyannote/speaker-diarization-3.1
DIARIZATION_MODEL  = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
HF_TOKEN           = os.getenv("HF_TOKEN", "")          # 최초 다운로드 시에만 필요
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}

whisper_model: Optional[WhisperModel]   = None
diarize_pipeline: Optional[Pipeline]    = None


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def _load_whisper() -> WhisperModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctype  = "float16" if device == "cuda" else "int8"
    logger.info(f"Whisper 로드: {WHISPER_MODEL_SIZE} / {device} / {ctype}")
    return WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=ctype)


def _load_diarize() -> Pipeline:
    """
    pyannote speaker-diarization-3.1 (로컬 또는 HF Hub)
    GPU 없을 경우 CPU 로 폴백.
    토큰은 환경변수로 huggingface_hub 가 자동 인식.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # HF_TOKEN → huggingface_hub 표준 환경변수로 주입
    if HF_TOKEN:
        os.environ.setdefault("HF_HUB_TOKEN", HF_TOKEN)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)

    # 로컬 경로가 있으면 로컬 우선
    local = Path(DIARIZATION_MODEL)
    if local.exists():
        pipeline = Pipeline.from_pretrained(str(local))
    else:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)

    pipeline = pipeline.to(device)
    logger.info(f"Diarization 파이프라인 로드 완료 ({device})")
    return pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    global whisper_model, diarize_pipeline
    whisper_model    = await asyncio.to_thread(_load_whisper)
    diarize_pipeline = await asyncio.to_thread(_load_diarize)
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ──────────────────────────────────────────────
# 화자 분리 + STT 결합
# ──────────────────────────────────────────────
def _run_diarize_and_stt(tmp_path: str, progress_cb):
    """
    1) pyannote로 화자 세그먼트 추출
    2) faster-whisper로 단어 단위 타임스탬프 전사
    3) 각 단어를 화자 세그먼트에 매핑
    반환: (대화록 str, 원문 str, 화자 목록)
    """
    # MP3 등 비-WAV 파일은 ffmpeg로 WAV 변환 (libmpg123 C-level 에러 우회)
    wav_path: Optional[str] = None
    if not tmp_path.lower().endswith(".wav"):
        wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_tmp.close()
        wav_path = wav_tmp.name
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg 오디오 변환 실패: {err}")
        audio_path = wav_path
    else:
        audio_path = tmp_path

    try:
        # --- 화자 분리 ---
        diarization = diarize_pipeline(audio_path)

        # 세그먼트를 [(start, end, speaker), ...] 리스트로 변환
        dia_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            dia_segments.append((turn.start, turn.end, speaker))

        progress_cb(10)   # diarization 완료 10%

        # --- STT (단어 단위 타임스탬프) ---
        segments_iter, info = whisper_model.transcribe(
            audio_path,
            language="ko",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True,          # ← 핵심: 단어 단위 타임스탬프
        )

        duration = info.duration
        all_words = []
        for seg in segments_iter:
            if seg.words:
                for w in seg.words:
                    all_words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
            # STT 진행률 10~90%
            pct = 10 + int((seg.end / duration) * 80)
            progress_cb(min(90, pct))

        # --- 단어 → 화자 매핑 ---
        def find_speaker(start, end):
            best_spk = "SPEAKER_00"
            best_overlap = 0.0
            for ds, de, spk in dia_segments:
                overlap = min(end, de) - max(start, ds)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_spk = spk
            return best_spk

        # 화자 번호를 사람 이름으로 (SPEAKER_00 → 화자 1)
        speaker_map: dict[str, str] = {}
        counter = 1

        def label(spk):
            nonlocal counter
            if spk not in speaker_map:
                speaker_map[spk] = f"화자 {counter}"
                counter += 1
            return speaker_map[spk]

        # 연속적으로 같은 화자의 단어를 묶어 대화록 생성
        dialogue_lines = []
        raw_words = []
        cur_speaker = None
        cur_words = []

        for w in all_words:
            spk = find_speaker(w["start"], w["end"])
            raw_words.append(w["word"])
            if spk != cur_speaker:
                if cur_words:
                    dialogue_lines.append({
                        "speaker": label(cur_speaker),
                        "text": " ".join(cur_words),
                        "start": cur_words[0] if cur_words else "",
                    })
                cur_speaker = spk
                cur_words = [w["word"]]
            else:
                cur_words.append(w["word"])

        if cur_words:
            dialogue_lines.append({
                "speaker": label(cur_speaker),
                "text": " ".join(cur_words),
            })

        # 대화록 문자열
        dialogue_str = "\n".join(
            f"[{d['speaker']}] {d['text']}" for d in dialogue_lines
        )
        original_text = " ".join(raw_words)
        speakers = list(speaker_map.values())

        progress_cb(100)
        return dialogue_str, original_text, speakers, dialogue_lines

    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


# ──────────────────────────────────────────────
# 요약 (Ollama)
# ──────────────────────────────────────────────
def _summarize(dialogue: str) -> str:
    prompt = f"""다음은 회의 대화록입니다 (화자별로 구분되어 있습니다). 아래 형식으로 한국어로 작성해 주세요.

1. **핵심 요약**: 회의의 주요 논의 사항을 3~5문장으로 간결하게 정리하세요.
2. **할 일 목록 (Action Items)**: 회의에서 결정된 실행 항목을 담당자(화자), 기한과 함께 번호 목록으로 정리하세요. 명시되지 않은 경우 '미지정'으로 표기하세요.
3. **화자별 주요 발언**: 각 화자의 핵심 발언을 1~2문장으로 요약하세요.

---
[회의 대화록]
{dialogue}
---

위 형식에 맞춰 작성하세요."""

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1536},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama 요약 오류: {e}")
        return "요약 생성 중 오류가 발생했습니다."


# ──────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"지원하지 않는 형식: {suffix}")

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    async def event_generator():
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def progress_cb(pct: int):
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "percent": pct})

        def run_all():
            try:
                dialogue, original, speakers, lines = _run_diarize_and_stt(tmp_path, progress_cb)
                loop.call_soon_threadsafe(queue.put_nowait, {
                    "type": "stt_done",
                    "dialogue": dialogue,
                    "original": original,
                    "speakers": speakers,
                    "lines": lines,
                })
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})

        asyncio.create_task(asyncio.to_thread(run_all))

        dialogue_text = ""
        original_text = ""
        speakers = []
        dialogue_lines = []

        try:
            while True:
                item = await queue.get()
                if item["type"] == "progress":
                    yield f"data: {json.dumps(item)}\n\n"
                elif item["type"] == "stt_done":
                    dialogue_text  = item["dialogue"]
                    original_text  = item["original"]
                    speakers       = item["speakers"]
                    dialogue_lines = item["lines"]
                    break
                elif item["type"] == "error":
                    yield f"data: {json.dumps(item)}\n\n"
                    return

            if not original_text.strip():
                yield f"data: {json.dumps({'type':'error','message':'음성을 감지할 수 없습니다.'})}\n\n"
                return

            # 요약 단계
            yield f"data: {json.dumps({'type':'progress','percent':100,'step':'summarize'})}\n\n"
            summary = await asyncio.to_thread(_summarize, dialogue_text)

            result = {
                "type": "result",
                "filename":       file.filename,
                "original_text":  original_text,
                "dialogue_text":  dialogue_text,
                "dialogue_lines": dialogue_lines,
                "summary":        summary,
                "speakers":       speakers,
                "speaker_count":  len(speakers),
                "char_count":     len(original_text),
                "word_count":     len(original_text.split()),
                "elapsed_sec":    round(time.perf_counter() - t0, 1),
            }
            yield f"data: {json.dumps(result)}\n\n"

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        ollama_ok = r.status_code == 200
    except:
        pass
    return {
        "status":         "ok" if (whisper_model and diarize_pipeline and ollama_ok) else "degraded",
        "whisper_model":  WHISPER_MODEL_SIZE if whisper_model else "not loaded",
        "diarization":    "loaded" if diarize_pipeline else "not loaded",
        "ollama_ok":      ollama_ok,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
