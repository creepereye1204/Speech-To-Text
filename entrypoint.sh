#!/bin/bash
set -e

OLLAMA_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
MODEL="${OLLAMA_MODEL:-hf.co/teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf:Q5_K_M}"

until curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; do
    echo "[entrypoint] Ollama 대기 중..."
    sleep 5
done

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | cut -d: -f1 | tr '[:upper:]' '[:lower:]')
if curl -sf "$OLLAMA_URL/api/tags" | grep -qi "$MODEL_SHORT"; then
    echo "[entrypoint] 모델 확인: $MODEL"
else
    echo "[entrypoint] 모델 다운로드: $MODEL"
    curl -sf -X POST "$OLLAMA_URL/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$MODEL\", \"stream\": false}" \
        --max-time 1800
    echo ""
fi

echo "[entrypoint] Gunicorn 시작"
exec gunicorn main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 600 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile -
