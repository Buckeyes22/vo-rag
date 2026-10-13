#!/usr/bin/env bash
# Bootstrap a RunPod pod for farm OCR (VOR-1434). Runs ON the pod, in $WORKDIR,
# after ocr_worker.py, bootstrap.sh and run_units.sh were copied there.
#
# Lessons baked in (2026-09-26 incidents):
# - pip's default torch wheel (cu130) silently ran on the CPU under the pods' CUDA 12.8
#   driver: torch is pinned to 2.8.0+cu128 and setup ABORTS unless torch.cuda.is_available().
# - parallel workers raced RapidOCR's first-use model download: ONE process warms the
#   models before the pool starts, and setup checks they are on disk.
# Exit codes: 0 ready, 3 no CUDA, 4 models not warmed, other = install failure.
set -euo pipefail

WORKDIR=${WORKDIR:-/workspace}
PYTHON=${PYTHON:-python3}
VENV_PY=${VENV_PY:-$WORKDIR/venv/bin/python}
# One model cache for the warm-up and the pool (run_units.sh uses the same default):
# warming any other cache leaves the workers to race the first-use download.
HF_HOME=${HF_HOME:-$WORKDIR/hf}
export HF_HOME
unset HF_TOKEN
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
TORCH_PINS=("torch==2.8.0+cu128" "torchvision==0.23.0+cu128")
BASE_PINS=(docling==2.97.0 docling-core==2.78.0 pypdfium2==5.9.0 rapidocr==3.8.1 pymupdf)

log() { echo "bootstrap: $*" >&2; }

make_venv() {
    "$PYTHON" -m venv "$WORKDIR/venv"
}

install_deps() {
    "$VENV_PY" -m pip install -q "${BASE_PINS[@]}"
    "$VENV_PY" -m pip install -q --force-reinstall "${TORCH_PINS[@]}" --index-url "$TORCH_INDEX_URL"
    "$VENV_PY" -m pip install -q "numpy<2.4"
}

require_cuda() {
    if ! "$VENV_PY" -c 'import sys, torch; print("torch", torch.__version__, "cuda", torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 3)'; then
        log "torch.cuda.is_available() is False: refusing to OCR on the CPU"
        echo "NO_CUDA" > "$WORKDIR/setup.failed"
        return 3
    fi
}

models_failed() {
    log "$1 (see warm.err)"
    echo "NO_MODELS" > "$WORKDIR/setup.failed"
    return 4
}

warm_models() {
    # One process only: this is the first use that downloads the models, into $HF_HOME
    # (docling) and RapidOCR's package dir.
    "$VENV_PY" -c 'import fitz; d = fitz.open(); p = d.new_page(); p.insert_text((72, 72), "warm up the OCR models", fontsize=24); d.save("warm.pdf")'
    "$VENV_PY" ocr_worker.py warm.pdf warm.json > warm.note 2> warm.err || true
    if [[ -z $(find "$HF_HOME" -type f -print -quit 2>/dev/null) ]]; then
        models_failed "HF cache $HF_HOME is empty after the warm-up"
        return 4
    fi
    # Every RapidOCR .pth must load: a partial download has the right name but not the bytes.
    if ! "$VENV_PY" -c 'import glob, os, sys, torch, rapidocr; fs = glob.glob(os.path.join(os.path.dirname(rapidocr.__file__), "models", "*.pth")); assert len(fs) >= 3, fs; [torch.load(f, map_location="cpu", weights_only=False) for f in fs]; print("models ok", len(fs))'; then
        models_failed "RapidOCR models are missing or do not load (torch.load)"
        return 4
    fi
}

main() {
    cd "$WORKDIR"
    rm -f setup.done setup.failed
    make_venv
    install_deps
    require_cuda
    "$VENV_PY" ocr_worker.py --check
    warm_models
    echo READY > setup.done
    log "ready"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
