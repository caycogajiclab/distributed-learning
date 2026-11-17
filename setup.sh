#!/usr/bin/env bash
# CCDL — environment setup (macOS / Linux)
#
# Creates a Python environment (venv or conda), installs the dependencies,
# editable-installs the repo (so `import ccdl` works) and optionally unpacks
# data_archive.zip. Just run it and answer the questions (Enter = default):
#     ./setup.sh
#
# Two setups:
#   legacy (paper)  Python 3.8 + the exact versions that produced the paper's
#                   results (requirements.txt). Use this to reproduce them.
#   updated         Python 3.12 + recent versions (updated-requirements.txt).
#                   Not fully tested.
#
# venv needs the interpreter already installed (python3.8 / python3.12, see
# setup_python.sh); conda installs it for you.
#
# Options (skip the questions; handy for scripted runs):
#   --setup legacy|updated   --manager venv|conda   --env FOLDER_OR_NAME
#   --cpu (CPU-only torch)   --extract-data (unpack data_archive.zip)
#   -h | --help
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SETUP="" MANAGER="" ENV_LOCATION="" CPU_TORCH=0 DO_EXTRACT=0
IS_MACOS=0; [ "$(uname -s)" = "Darwin" ] && IS_MACOS=1
INTERACTIVE=0; [ $# -eq 0 ] && [ -t 0 ] && INTERACTIVE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --setup)        SETUP="${2:?}"; shift 2 ;;
    --manager)      MANAGER="${2:?}"; shift 2 ;;
    --env)          ENV_LOCATION="${2:?}"; shift 2 ;;
    --cpu)          CPU_TORCH=1; shift ;;
    --extract-data) DO_EXTRACT=1; shift ;;
    -h|--help)      sed -n '2,/^set -euo/{/^set/d;s/^# \{0,1\}//;p;}' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

ask() { local r; read -r -p "$1" r; echo "${r:-$2}"; }
warn_updated() {
  echo
  echo "  WARNING: the 'updated' setup has NOT been fully tested."
  echo "           To reproduce the paper's results, use the 'legacy' setup."
  echo
}
command -v conda >/dev/null 2>&1 && default_mgr=conda || default_mgr=venv

# ---- questions (interactive) / defaults (scripted) --------------------------
if [ "$INTERACTIVE" -eq 1 ]; then
  echo "Which setup?"
  echo "  1) legacy (paper) — Python 3.8 + exact versions used for the paper's results"
  echo "  2) updated        — Python 3.12 + recent versions (not fully tested)"
  case "$(ask "Choice [1]: " 1)" in 2|updated) SETUP=updated; warn_updated ;; esac
  MANAGER="$(ask "Environment manager (conda/venv) [$default_mgr]: " "$default_mgr")"
  if [ "$MANAGER" = conda ]; then ENV_LOCATION="$(ask "Conda environment name [ccdl]: " ccdl)"
  else ENV_LOCATION="$(ask "Environment folder [.venv]: " .venv)"; fi
  if [ "$IS_MACOS" -eq 0 ]; then
    case "$(ask "Install the CPU-only build of torch (no CUDA GPU)? [y/N]: " n)" in [yY]*) CPU_TORCH=1 ;; esac
  fi
  case "$(ask "Extract data_archive.zip (data/ + figures/ + source_data/, only needed for the paper's figures)? [y/N]: " n)" in
    [yY]*) DO_EXTRACT=1 ;;
  esac
  echo
fi
SETUP="${SETUP:-legacy}"; MANAGER="${MANAGER:-$default_mgr}"

case "$SETUP" in
  legacy)  PY_VERSION=3.8;  [ "$MANAGER" = conda ] && PY_VERSION=3.8.10; REQ_FILE=requirements.txt ;;
  updated) PY_VERSION=3.12; REQ_FILE=updated-requirements.txt; [ "$INTERACTIVE" -eq 1 ] || warn_updated ;;
  *) echo "Unknown setup '$SETUP' (use legacy | updated)" >&2; exit 2 ;;
esac

# ---- create the environment; PY = its python --------------------------------
echo ">> setup: $SETUP (Python $PY_VERSION), manager: $MANAGER"
case "$MANAGER" in
  venv)
    ENV_LOCATION="${ENV_LOCATION:-.venv}"
    PYTHON="python$(echo "$PY_VERSION" | cut -d. -f1,2)"
    if ! "$PYTHON" -c 'import ensurepip' >/dev/null 2>&1; then
      echo "ERROR: '$PYTHON' (with venv support) not found. Install it with" >&2
      echo "       'sudo bash setup_python.sh', or use conda (--manager conda)." >&2
      exit 1
    fi
    "$PYTHON" -m venv --clear "$ENV_LOCATION"
    PY="$ENV_LOCATION/bin/python"
    ACTIVATE_HINT="source $ENV_LOCATION/bin/activate" ;;
  conda)
    ENV_LOCATION="${ENV_LOCATION:-ccdl}"
    conda create -y -n "$ENV_LOCATION" -c conda-forge --override-channels "python=$PY_VERSION" pip
    PY="$(conda run -n "$ENV_LOCATION" python -c 'import sys; print(sys.executable)')"
    ACTIVATE_HINT="conda activate $ENV_LOCATION" ;;
  *) echo "Unknown manager '$MANAGER' (use conda | venv)" >&2; exit 2 ;;
esac

# ---- install the requirements -----------------------------------------------
# requirements.txt pins the CUDA 11.3 torch build; adapt the torch line for
# macOS (plain PyPI build) or CPU-only (+cpu build) on a temporary copy.
"$PY" -m pip install --upgrade pip
REQ_TMP="$(mktemp)"; trap 'rm -f "$REQ_TMP"' EXIT
if [ "$SETUP" = legacy ] && [ "$IS_MACOS" -eq 1 ]; then
  sed -e 's/+cu113//' -e '/^--extra-index-url/d' "$REQ_FILE" > "$REQ_TMP"
elif [ "$SETUP" = legacy ] && [ "$CPU_TORCH" -eq 1 ]; then
  sed -e 's/+cu113/+cpu/' -e 's#whl/cu113#whl/cpu#' "$REQ_FILE" > "$REQ_TMP"
else
  cp "$REQ_FILE" "$REQ_TMP"
  if [ "$CPU_TORCH" -eq 1 ] && [ "$IS_MACOS" -eq 0 ]; then
    echo ">> installing CPU-only torch"
    "$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu "$(grep -E '^torch([=<>!~ ;]|$)' "$REQ_FILE")"
  fi
fi
echo ">> installing $REQ_FILE"
"$PY" -m pip install -r "$REQ_TMP"
echo ">> editable install of ccdl"
"$PY" -m pip install -e . --no-deps

# Legacy torch 1.11 libraries are flagged as needing an executable stack, which
# glibc >= 2.41 refuses ("cannot enable executable stack"). Clear the flag.
if [ "$SETUP" = legacy ] && [ "$IS_MACOS" -eq 0 ]; then
  echo ">> legacy torch: clearing the executable-stack flag (needed on recent Linux)"
  "$PY" -m pip install -q patchelf
  PATCHELF="$(dirname "$PY")/patchelf"
  # locate torch WITHOUT importing it (importing is what fails)
  TORCH_DIR="$("$PY" -c 'import importlib.util as u; print(u.find_spec("torch").submodule_search_locations[0])')"
  for so in "$TORCH_DIR"/lib/*.so*; do
    if [ -f "$so" ] && "$PATCHELF" --print-execstack "$so" | grep -q X; then
      "$PATCHELF" --clear-execstack "$so"; echo "   cleared: $(basename "$so")"
    fi
  done
fi

# ---- working folders / data -------------------------------------------------
if [ "$DO_EXTRACT" -eq 1 ]; then
  command -v unzip >/dev/null 2>&1 || { echo "ERROR: 'unzip' not found" >&2; exit 1; }
  echo ">> extracting data_archive.zip (existing files are kept)"
  unzip -n -q data_archive.zip || [ $? -eq 1 ]   # rc 1 = harmless warning
fi

echo
echo "Done. Activate the environment with:  $ACTIVATE_HINT"
[ "$SETUP" = updated ] && warn_updated
exit 0
