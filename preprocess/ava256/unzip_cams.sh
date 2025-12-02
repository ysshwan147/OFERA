#!/bin/bash

DATASET_ROOT="/root/Dataset/ava-256-dataset"
TARGET_CAMS=("401168" "401875")

for subject_dir in "$DATASET_ROOT"/*; do
  IMAGE_DIR="$subject_dir/decoder/image"

  if [ -d "$IMAGE_DIR" ]; then
    for cam_id in "${TARGET_CAMS[@]}"; do
      ZIP_FILE="$IMAGE_DIR/cam${cam_id}.zip"
      if [ -f "$ZIP_FILE" ]; then
        echo "Unzipping $ZIP_FILE..."
        unzip -q "$ZIP_FILE" -d "$IMAGE_DIR"
      fi
    done
  fi
done
