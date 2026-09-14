#!/usr/bin/env bash
# One-time: copy Bessica-D v1.0 STL files into this package so ILStudio is self-contained.
# Usage: ./vendor_meshes.sh /path/to/directory/with/STL/files
# The directory may be named anything; all *.STL / *.stl in it (non-recursive) are copied.

set -euo pipefail

if [[ "${1:-}" == "" ]]; then
  echo "Usage: $0 /path/to/stl_directory"
  echo "  Copies STL files into:"
  echo "    mujoco_model/meshes/Bessica-D_v1_0/"
  echo "    assets/meshes/Bessica-D_v1_0/"
  exit 1
fi

SRC=$(realpath "$1")
ROOT="$(cd "$(dirname "$0")" && pwd)"
DST_MJ="$ROOT/mujoco_model/meshes/Bessica-D_v1_0"
DST_URDF="$ROOT/assets/meshes/Bessica-D_v1_0"

if [[ ! -d "$SRC" ]]; then
  echo "Error: not a directory: $SRC"
  exit 1
fi

mkdir -p "$DST_MJ" "$DST_URDF"

n=0
for pat in "$SRC"/*.STL "$SRC"/*.stl; do
  [[ -e "$pat" ]] || continue
  base=$(basename "$pat")
  cp -f "$pat" "$DST_MJ/$base"
  cp -f "$pat" "$DST_URDF/$base"
  n=$((n + 1))
done

if [[ "$n" -eq 0 ]]; then
  echo "Error: no *.STL or *.stl files found in: $SRC"
  exit 1
fi

echo "Copied $n mesh file(s) into:"
echo "  $DST_MJ"
echo "  $DST_URDF"
