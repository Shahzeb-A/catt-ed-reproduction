#!/usr/bin/env bash
# One time setup: venv, upstream code at the pinned SHA, release assets with sha256 checks.
# Needs git, curl and Python 3.12 (uv is used when present, plain venv + pip otherwise).
# The upstream benchmarking/ folder (WikiNews and CATT benchmark gold) is deliberately NOT
# fetched: benchmark data is pulled only after its licence is approved.
# Safe to rerun; every step checks its own result.
set -euo pipefail
cd "$(dirname "$0")/.."
export LC_ALL=C

SHA="$(cat UPSTREAM_SHA)"
REL="https://github.com/abjadai/catt/releases/download/v2"

# --- environment -------------------------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then uv venv --python 3.12 .venv; else python3.12 -m venv .venv; fi
fi
if command -v uv >/dev/null; then
  uv pip install -q --python .venv/bin/python -r requirements.txt
else
  .venv/bin/python -m pip install -q -r requirements.txt
fi

# --- upstream code, without benchmarking/ ------------------------------------------------
# blobless clone and the sparse pattern set BEFORE checkout, so benchmarking/ blobs are never fetched
[ -d vendor/catt/.git ] || git clone -q --filter=blob:none --no-checkout https://github.com/abjadai/catt vendor/catt
if [ "$(git -C vendor/catt rev-parse HEAD 2>/dev/null)" != "$SHA" ] || [ ! -f vendor/catt/ed_pl.py ]; then
  git -C vendor/catt sparse-checkout set --no-cone '/*' '!/benchmarking/'
  git -C vendor/catt checkout -q "$SHA"
fi
test "$(git -C vendor/catt rev-parse HEAD)" = "$SHA" || { echo "vendor/catt is not at $SHA"; exit 1; }
test ! -e vendor/catt/benchmarking || { echo "benchmarking/ must not be checked out"; exit 1; }

# No benchmark blob may sit in the local object store. Both commands read only what is already
# local (ls-tree needs trees, rev-list --missing=print never fetches), so the check itself cannot
# download anything, on any git version.
present=$(comm -12 \
  <(git -C vendor/catt ls-tree -r HEAD -- benchmarking | awk '$2=="blob"{print $3}' | sort -u) \
  <(git -C vendor/catt rev-list --objects --no-walk --missing=print HEAD | awk '$1 !~ /^\?/ {print $1}' | sort -u) \
  | grep -c . || true)
[ "$present" -eq 0 ] || { echo "$present benchmark blobs are in vendor/catt/.git; delete vendor/catt and rerun"; exit 1; }

# Every upstream file used here must be byte identical to the pinned tree (hash-object never fetches).
git -C vendor/catt ls-tree HEAD | awk '$2=="blob"{print $3" "$4}' | while read -r oid f; do
  [ "$(git -C vendor/catt hash-object "$f")" = "$oid" ] || { echo "vendor/catt/$f differs from $SHA"; exit 1; }
done

# --- release assets ----------------------------------------------------------------------
sha256_ok() {  # sha256 file
  if command -v sha256sum >/dev/null; then echo "$1  $2" | sha256sum -c --status -
  else echo "$1  $2" | shasum -a 256 -c --status -; fi
}
mkdir -p data/release
fetch() {  # name sha256
  local f="data/release/$1"
  if ! sha256_ok "$2" "$f" 2>/dev/null; then
    curl -fL --retry 3 -C - -o "$f" "$REL/$1" || { rm -f "$f"; curl -fL --retry 3 -o "$f" "$REL/$1"; }
  fi
  sha256_ok "$2" "$f" || { echo "sha256 mismatch: $f (delete it and rerun)"; exit 1; }
  echo "ok $f"
}
fetch best_ed_mlm_ns_epoch_178.pt c6297ec54f9af518a59d4b9e5831d3ad4da304435b8c7fb969f6e3a16816ac23
fetch dataset.zip fa332ca9d3118ed8bc974debc681ba3f9d5ebca3cb824073b24bef0314e5fb00

# only val/ and test/ are extracted; train.txt (1.1 GB) is streamed from the zip when needed
.venv/bin/python - <<'PY'
import zipfile
z = zipfile.ZipFile("data/release/dataset.zip")
for n in z.namelist():
    if n.startswith(("dataset/val/", "dataset/test/")):
        z.extract(n, "data/ds")
PY
sha256_ok 4e851ff836f0a178abb15d9f4a8bcf92748b77cc0d67030d9fae2fbe698baa12 data/ds/dataset/test/test.txt \
  || { echo "sha256 mismatch: test.txt"; exit 1; }
echo "setup ok"
