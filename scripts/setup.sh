#!/usr/bin/env bash
# One time setup: venv, upstream code at the pinned SHA, release assets with sha256 checks.
# The upstream benchmarking/ folder (WikiNews and CATT benchmark gold) is deliberately NOT
# checked out: benchmark data is pulled only after its licence is approved.
set -euo pipefail
cd "$(dirname "$0")/.."

SHA="$(cat UPSTREAM_SHA)"
REL="https://github.com/abjadai/catt/releases/download/v2"

if [ ! -d .venv ]; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi

if [ ! -d vendor/catt/.git ]; then
  git clone --filter=blob:none --no-checkout https://github.com/abjadai/catt vendor/catt
  git -C vendor/catt sparse-checkout set --no-cone '/*' '!/benchmarking/'
  git -C vendor/catt checkout -q "$SHA"
fi
test "$(git -C vendor/catt rev-parse HEAD)" = "$SHA" || { echo "vendor/catt is not at $SHA"; exit 1; }
test ! -e vendor/catt/benchmarking || { echo "benchmarking/ must not be checked out"; exit 1; }
# a partial clone fetches blobs lazily; make sure no benchmark blob reached the local object store
for b in $(git -C vendor/catt ls-tree -r HEAD benchmarking | awk '{print $3}'); do
  if GIT_NO_LAZY_FETCH=1 git -C vendor/catt cat-file -e "$b" 2>/dev/null; then
    echo "benchmark blob $b is in vendor/catt/.git; delete vendor/catt and rerun"; exit 1
  fi
done

mkdir -p data/release
fetch() {  # name sha256
  local f="data/release/$1"
  [ -f "$f" ] || curl -fL --retry 3 -C - -o "$f" "$REL/$1"
  echo "$2  $f" | shasum -a 256 -c -
}
fetch best_ed_mlm_ns_epoch_178.pt c6297ec54f9af518a59d4b9e5831d3ad4da304435b8c7fb969f6e3a16816ac23
fetch dataset.zip fa332ca9d3118ed8bc974debc681ba3f9d5ebca3cb824073b24bef0314e5fb00

mkdir -p data/ds
[ -f data/ds/dataset/test/test.txt ] || unzip -q -o data/release/dataset.zip 'dataset/val/*' 'dataset/test/*' -d data/ds
echo "4e851ff836f0a178abb15d9f4a8bcf92748b77cc0d67030d9fae2fbe698baa12  data/ds/dataset/test/test.txt" | shasum -a 256 -c -
echo "setup ok"
