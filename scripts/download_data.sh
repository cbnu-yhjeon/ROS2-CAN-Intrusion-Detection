#!/usr/bin/env bash
# HCRL Car-Hacking 데이터 (원 저장소가 공유한 Google Drive 압축본) 내려받기
#   scripts/download_data.sh [대상 디렉토리=data]
# 데이터 사용 시 HCRL 인용 요구사항을 따른다. 저장소에는 넣지 않는다.
set -euo pipefail
DEST="${1:-data}"; ID=1gKQdjjNxbO0t2VrAFyPLnAqdSGpvotVg
mkdir -p "$DEST"
curl -L -o "$DEST/hcrl.zip" "https://drive.usercontent.google.com/download?id=$ID&export=download&confirm=t"
unzip -q -o "$DEST/hcrl.zip" 'Data/raw/*' -d "$DEST"
rm "$DEST/hcrl.zip"
ls -la "$DEST/Data/raw"
