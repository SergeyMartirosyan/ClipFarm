#!/usr/bin/env bash
# Downloads the face-detection model (9:16 layout) and the sound model (game sounds), ~4.5 MB total.
# Whisper downloads its own model on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
fetch() {  # fetch <destination> <url>
  mkdir -p "$(dirname "$1")"
  if [ -s "$1" ]; then echo "Already present: $1"; return; fi
  curl -fsSL -o "$1.part" "$2" && mv "$1.part" "$1" && echo "Saved $1"
}
# Face detection (YuNet, ~230 KB)
fetch "${FACE_MODEL_PATH:-./data/models/face_detection_yunet_2023mar.onnx}" \
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
# Game-sound tagging (YAMNet TFLite, ~4 MB) and its class names
fetch "${SOUND_MODEL_PATH:-./data/models/yamnet.tflite}" \
  "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
fetch "${SOUND_LABELS_PATH:-./data/models/yamnet_class_map.csv}" \
  "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
