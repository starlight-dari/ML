#!/bin/bash

# Google Drive 파일 ID 및 출력 파일명
FILE_ID="1Vf5vdnlW4Y2VurWzIK6o_Nxe0t4K7UQm"
OUTPUT_FILE="sam_vit_b_01ec64.pth"

FILE_ID2="1_EWkG2dUNIfkzrYfDPcOhP-osKxngX1V"
OUTPUT_FILE2="table5_pidinet.pth"

# 모델 파일 다운로드 (기존 파일이 없을 경우만 다운로드)
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Downloading $OUTPUT_FILE from Google Drive..."
    gdown --id "$FILE_ID" -O "$OUTPUT_FILE"
else
    echo "$OUTPUT_FILE already exists. Skipping download."
fi

if [ ! -f "$OUTPUT_FILE2" ]; then
    echo "Downloading $OUTPUT_FILE2 from Google Drive..."
    gdown --id "$FILE_ID2" -O "$OUTPUT_FILE2"
else
    echo "$OUTPUT_FILE2 already exists. Skipping download."
fi