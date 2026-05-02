# 로컬 AI 회의록 자동화 서버 — 설치 및 실행 가이드

## 시스템 요구 사항
- Ubuntu Server 20.04 / 22.04
- NVIDIA RTX 2080 Ti (11 GB VRAM)
- CUDA 11.8 이상 + cuDNN
- Python 3.10 이상
- Ollama 설치 완료

---

## 1단계 — CUDA 환경 확인

```bash
nvidia-smi          # GPU 인식 확인
nvcc --version      # CUDA 버전 확인 (11.8 이상 권장)
```

---

## 2단계 — Python 가상 환경 구성

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3단계 — PyTorch (CUDA) 설치

> `requirements.txt` 설치 전에 CUDA 버전에 맞는 PyTorch를 먼저 설치해야 합니다.

```bash
# CUDA 11.8 기준
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 기준
# pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

---

## 4단계 — 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 5단계 — Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 (미설치 시)
curl -fsSL https://ollama.com/install.sh | sh

# Ollama 서비스 시작
ollama serve &

# 한국어 LLM 모델 다운로드 (약 6 GB)
ollama pull yanolja/EEVE-Korean-Instruct-10.8B
```

---

## 6단계 — 서버 실행

```bash
# 기본 실행
python main.py

# 또는 uvicorn 직접 실행
uvicorn main:app --host 0.0.0.0 --port 8000

# 환경 변수로 모델/URL 변경 가능
OLLAMA_BASE_URL=http://localhost:11434 \
OLLAMA_MODEL=yanolja/EEVE-Korean-Instruct-10.8B \
WHISPER_MODEL_SIZE=large-v3 \
python main.py
```

서버가 시작되면 Whisper `large-v3` 모델을 자동으로 로드합니다 (최초 실행 시 약 1.5 GB 다운로드).

---

## 7단계 — API 사용

### 엔드포인트 목록

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/transcribe` | 오디오 파일 업로드 → STT + 요약 반환 |
| GET | `/health` | 서버 상태 확인 |
| GET | `/docs` | Swagger UI |

### 예시 — curl

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@meeting.mp3"
```

### 예시 — Python

```python
import requests

with open("meeting.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": ("meeting.wav", f, "audio/wav")},
    )

data = response.json()
print("=== 원문 ===")
print(data["original_text"])
print("\n=== 요약 ===")
print(data["summary"])
```

### 응답 형식 (JSON)

```json
{
  "filename": "meeting.mp3",
  "original_text": "오늘 회의에서는 ...",
  "summary": "## 핵심 요약\n...\n\n## 할 일 목록\n1. ..."
}
```

---

## VRAM 사용량 참고

| 단계 | 사용 모델 | VRAM 소모 |
|------|-----------|-----------|
| STT | Whisper large-v3 (float16) | ~6 GB |
| LLM | EEVE-Korean-10.8B (Ollama) | ~7 GB |

> STT 완료 후 LLM 요청을 보내는 **순차 처리** 방식으로 설계되어 있습니다.  
> Ollama는 별도 프로세스로 VRAM을 관리하므로 두 모델이 동시에 11 GB를 점유하지 않습니다.

---

## 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| `CUDA out of memory` | VRAM 부족 | `WHISPER_MODEL_SIZE=medium` 으로 변경 |
| `503 Ollama 연결 실패` | Ollama 미실행 | `ollama serve` 실행 후 재시도 |
| 변환 결과가 비어있음 | 무음 파일 | 오디오 파일에 음성이 있는지 확인 |
| 느린 처리 속도 | CPU fallback | `nvidia-smi`로 GPU 인식 여부 확인 |
