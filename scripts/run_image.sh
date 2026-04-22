#!/bin/bash

export TMPDIR=$PWD/tmp
mkdir -p $TMPDIR

pip install --no-cache-dir --target=$TMPDIR \
    pillow numpy torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

export PYTHONPATH=$TMPDIR:$PYTHONPATH

IMAGE_NAME=$(basename "$1")

echo "Processing $IMAGE_NAME"

python process_image.py "$IMAGE_NAME"

# make sure file exists
if ls *.npy 1> /dev/null 2>&1; then
    echo "Embedding created"
else
    echo "FAILED to create embedding"
    touch failed.npy
fi
