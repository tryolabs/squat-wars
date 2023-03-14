#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./squat_wars/models"
else
  DATA_DIR="$1"
fi

# Create folder for models
if [ ! -d "$DATA_DIR" ]; then
  mkdir $DATA_DIR
fi

# Download TF Lite models

# TPU models
FILE=${DATA_DIR}/movenet_lightning_tpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://raw.githubusercontent.com/google-coral/test_data/master/movenet_single_pose_lightning_ptq_edgetpu.tflite' \
    -o ${FILE}
fi

FILE=${DATA_DIR}/movenet_thunder_tpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://raw.githubusercontent.com/google-coral/test_data/master/movenet_single_pose_thunder_ptq_edgetpu.tflite' \
    -o ${FILE}
fi

# CPU models
FILE=${DATA_DIR}/movenet_lightning.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite' \
    -o ${FILE}
fi

FILE=${DATA_DIR}/movenet_thunder.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite' \
    -o ${FILE}
fi

echo -e "Downloaded files are in ${DATA_DIR}"