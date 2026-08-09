#!/usr/bin/env bash
# Завантажує модель vosk-model-uk-v3-lgraph (тир з динамічним графом, обов'язковий
# для грамматики/guided decoding — див. docs/reading-speed-feature-design.md, 3.1).
set -euo pipefail

MODEL_NAME="vosk-model-uk-v3-lgraph"
MODEL_URL="https://alphacephei.com/vosk/models/${MODEL_NAME}.zip"
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/vosk_models"

mkdir -p "$TARGET_DIR"

if [ -d "$TARGET_DIR/$MODEL_NAME" ]; then
  echo "Модель вже завантажена: $TARGET_DIR/$MODEL_NAME"
  exit 0
fi

echo "Завантаження $MODEL_URL (~325 МБ)..."
curl -L --fail -o "$TARGET_DIR/$MODEL_NAME.zip" "$MODEL_URL"

echo "Розпакування..."
unzip -q "$TARGET_DIR/$MODEL_NAME.zip" -d "$TARGET_DIR"
rm "$TARGET_DIR/$MODEL_NAME.zip"

echo "Готово: $TARGET_DIR/$MODEL_NAME"
