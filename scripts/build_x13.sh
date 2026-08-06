#!/usr/bin/env bash
# Build the Census X-13ARIMA-SEATS binary for THIS platform and install it
# into the MacroPy package tree (src/MacroPy/bin/<system>-<machine>/x13as),
# where MacroPy.x13.x13_path() discovers it.
#
# Requirements: gfortran and make (macOS: `brew install gcc`; Debian/Ubuntu:
# `apt-get install gfortran make`). Safe to rerun; ~2 minutes.
set -euo pipefail

VERSION="v1-1-b62"
URL="https://www2.census.gov/software/x-13arima-seats/x13as/unix-linux/program-archives/x13as_asciisrc-${VERSION}.tar.gz"

here="$(cd "$(dirname "$0")/.." && pwd)"
system="$(uname -s | tr '[:upper:]' '[:lower:]')"
machine="$(uname -m)"
case "$machine" in aarch64) machine=arm64 ;; amd64) machine=x86_64 ;; esac
dest="${here}/src/MacroPy/bin/${system}-${machine}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
echo "downloading ${URL}"
curl -fsSL "$URL" -o "${work}/src.tar.gz"
tar xzf "${work}/src.tar.gz" -C "$work"
cd "${work}/x13as_asciisrc-${VERSION}"

# legacy F77 needs relaxed checks on modern gfortran; macOS forbids -static
make -f makefile.gf FFLAGS="-O2 -std=legacy -fallow-argument-mismatch" -j8 \
  || gfortran -o x13as_ascii ./*.o

printf '' | ./x13as_ascii | head -2
mkdir -p "$dest"
install -m 755 x13as_ascii "${dest}/x13as"
echo "installed ${dest}/x13as"
