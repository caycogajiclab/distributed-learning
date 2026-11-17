#!/usr/bin/env bash
#
# CCDL — install a system Python the standard way (no uv, no conda), Debian/Ubuntu
# --------------------------------------------------------------------------------
# Needed only for the venv route of setup.sh:
#   legacy (paper) setup -> Python 3.8   (no longer shipped by current Ubuntu/Debian)
#   updated setup        -> Python 3.12  (shipped by some releases, e.g. Ubuntu 24.04)
# Installs the interpreter + venv/dev packages, from the distro if available,
# otherwise from the deadsnakes PPA. (conda users don't need this script.)
#
# Note on the exact patch: apt provides the latest 3.X.y patch (e.g. 3.8.20),
# not exactly 3.8.10. With the pinned package versions this is expected to
# reproduce results fine; for 3.8.10 exactly use conda or pyenv.
#
# Usage:  sudo bash setup_python.sh          # asks which version
#         sudo bash setup_python.sh 3.12     # or give it directly (3.8 | 3.12)
#
set -euo pipefail

VER="${1:-}"
if [ -z "$VER" ]; then
  read -r -p "Python for which setup? 1) legacy (3.8)  2) updated (3.12) [1]: " c
  case "$c" in 2|updated) VER=3.12 ;; *) VER=3.8 ;; esac
fi
case "$VER" in 3.8|3.12) ;; *) echo "Unsupported version '$VER' (use 3.8 | 3.12)" >&2; exit 2 ;; esac

if command -v "python$VER" >/dev/null 2>&1 && "python$VER" -c 'import ensurepip' >/dev/null 2>&1; then
  echo "python$VER with venv support is already installed."
  exit 0
fi

apt update
# use the distro's packages when it has them, otherwise add deadsnakes
if ! apt-cache show "python$VER-venv" >/dev/null 2>&1; then
  add-apt-repository -y ppa:deadsnakes/ppa
  apt update
fi
apt install -y "python$VER" "python$VER-venv" "python$VER-dev"

echo
"python$VER" --version
echo "Next, from the repo root:  ./setup.sh   (choose venv)"
