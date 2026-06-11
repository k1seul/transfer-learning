#!/bin/bash
# download_data.sh — CUB-200-2011 + CUB-200-Paintings 데이터셋 다운로드
#
# Usage:
#   bash download_data.sh [DATA_DIR]
#   DATA_DIR default: ./data
#
# After this script:
#   DATA_DIR/
#   ├── CUB_200_2011/images/001.Black_footed_Albatross/...   (11,788 images)
#   └── CUB_200_Paintings/001.Black_footed_Albatross/...     ( 3,047 images)

set -e

DATA_DIR="${1:-./data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "====================================="
echo "  UDA Dataset Downloader"
echo "  Target: $DATA_DIR"
echo "====================================="

# ── CUB-200-2011 (photos) ─────────────────────────────────────────────────────
CUB_TGZ="CUB_200_2011.tgz"
CUB_URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

if [ -d "CUB_200_2011/images" ]; then
    echo "[SKIP] CUB_200_2011/images already exists"
else
    echo ""
    echo "[1/2] Downloading CUB-200-2011 (~1.1GB) ..."
    if command -v wget &>/dev/null; then
        wget -c "$CUB_URL" -O "$CUB_TGZ"
    else
        curl -L "$CUB_URL" -o "$CUB_TGZ"
    fi

    echo "  Extracting ..."
    tar -xzf "$CUB_TGZ"
    rm "$CUB_TGZ"
    echo "  Done: CUB_200_2011/ ($(find CUB_200_2011/images -name '*.jpg' | wc -l) images)"
fi

# ── CUB-200-Paintings ─────────────────────────────────────────────────────────
# Source: DomainBed / Fine-Grained UDA benchmarks
# Two possible sources — try in order

PAINT_DIR="CUB_200_Paintings"
PAINT_ZIP="CUB_200_Paintings.zip"

# Primary source: known Google Drive export (add your own link if different)
# GDRIVE_ID="<REPLACE_WITH_GDRIVE_FILE_ID>"

# Fallback: direct HTTP if hosted
# PAINT_URL="https://example.com/CUB_200_Paintings.zip"

if [ -d "$PAINT_DIR" ]; then
    echo "[SKIP] CUB_200_Paintings already exists"
else
    echo ""
    echo "[2/2] Downloading CUB-200-Paintings (~80MB) ..."

    # ── Option A: gdown (Google Drive) ─────────────────────────
    if command -v gdown &>/dev/null && [ -n "${GDRIVE_ID:-}" ]; then
        gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "$PAINT_ZIP"

    # ── Option B: direct URL ────────────────────────────────────
    elif [ -n "${PAINT_URL:-}" ]; then
        if command -v wget &>/dev/null; then
            wget -c "$PAINT_URL" -O "$PAINT_ZIP"
        else
            curl -L "$PAINT_URL" -o "$PAINT_ZIP"
        fi

    # ── Option C: manual instructions ──────────────────────────
    else
        echo ""
        echo "  ⚠  CUB-200-Paintings 자동 다운로드 링크가 없습니다."
        echo "     아래 방법 중 하나로 직접 다운로드하세요:"
        echo ""
        echo "  방법 1) 강의 자료 또는 교수님이 제공한 링크 사용"
        echo ""
        echo "  방법 2) DomainBed 기반 스크립트:"
        echo "    pip install gdown"
        echo "    gdown <GDRIVE_ID> -O $DATA_DIR/CUB_200_Paintings.zip"
        echo "    unzip $DATA_DIR/CUB_200_Paintings.zip -d $DATA_DIR/"
        echo ""
        echo "  방법 3) 로컬 머신에서 복사:"
        echo "    scp -r /local/path/CUB_200_Paintings  server:$DATA_DIR/"
        echo ""
        echo "  기대 구조:"
        echo "    $DATA_DIR/CUB_200_Paintings/"
        echo "    ├── 001.Black_footed_Albatross/"
        echo "    ├── 002.Laysan_Albatross/"
        echo "    └── ... (200 classes, ~3,047 images total)"
        exit 0
    fi

    echo "  Extracting ..."
    unzip -q "$PAINT_ZIP"
    rm "$PAINT_ZIP"

    # normalize dir name if needed
    [ -d "cub_paintings" ] && mv "cub_paintings" "$PAINT_DIR"
    [ -d "CUB_Paintings" ] && mv "CUB_Paintings" "$PAINT_DIR"

    echo "  Done: CUB_200_Paintings/ ($(find $PAINT_DIR -name '*.jpg' -o -name '*.png' | wc -l) images)"
fi

# ── Verify structure ──────────────────────────────────────────────────────────
echo ""
echo "====================================="
echo "  Verification"
echo "====================================="

CUB_COUNT=$(find CUB_200_2011/images -name "*.jpg" 2>/dev/null | wc -l)
PAINT_COUNT=$(find CUB_200_Paintings -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
CUB_CLASSES=$(ls CUB_200_2011/images 2>/dev/null | wc -l)
PAINT_CLASSES=$(ls CUB_200_Paintings 2>/dev/null | wc -l)

echo "  CUB-200-2011 : ${CUB_COUNT} images  / ${CUB_CLASSES} classes  (expected: 11788 / 200)"
echo "  CUB-Paintings: ${PAINT_COUNT} images / ${PAINT_CLASSES} classes (expected: 3047  / 200)"

if [ "$CUB_CLASSES" -eq 200 ] && [ "$PAINT_CLASSES" -eq 200 ]; then
    echo ""
    echo "  ✓ Both datasets ready!"
    echo ""
    echo "  Run experiment with:"
    echo "    python run_one.py --config configs/ctop_sweep.json --idx 0 \\"
    echo "      # (data paths are set per config; update if DATA_DIR != ./data)"
else
    echo ""
    echo "  ✗ One or more datasets incomplete."
fi
