#!/usr/bin/env bash
# Downloads the face-detection model used for the 9:16 layout (about 230 KB).
# Whisper downloads its own model on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${FACE_MODEL_PATH:-./data/models/face_detection_yunet_2023mar.onnx}"
mkdir -p "$(dirname "$DEST")"
if [ -s "$DEST" ]; then
  echo "Face model already present: $DEST"
  exit 0
fi
curl -fsSL -o "$DEST" \
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
echo "Saved face model to $DEST"
