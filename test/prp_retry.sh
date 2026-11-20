#!/usr/bin/env bash
# End-to-end tests of the PRP retry after a repeatable error (Task::execute, FFTConfig nextFit):  with the fault build's
# PRPLL_PRP_FAULT (src/Gpu.cpp) a residue bit flips after iteration k at every pass on the FFT given, so its Gerbicz check fails the
# same way twice.
#
#   once    one repeatable error:  retry on the next FFT, back to the first FFT past the failing iteration, the right res64
#   twice   two:  the second stays on the next FFT to the end, the right res64
#   both    the next FFT fails too:  the test stops
#   fft     with -fft:  no retry, the test stops
#
#   test/prp_retry.sh [once|twice|both|fft|all]      (default all)
# Environment:  DEVICE (default 0), WORK (default test/work-prp-retry), FAULT_PRPLL (default build-prpfaults/prpll, built with
#               -DPRPLL_PRP_FAULTS if missing; CUDA=1 for a CUDA one), E (default 2000003), K1, K2 (the fault iterations:  not
#               multiples of the block size, and between the checks at multiples of E/8 rounded to the log step), FAIL1 (the check
#               that sees K1's fault:  default 500000)

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
DEVICE=${DEVICE:-0}
WORK=${WORK:-$HERE/work-prp-retry}
FAULT_PRPLL=${FAULT_PRPLL:-$ROOT/build-prpfaults/prpll}
E=${E:-2000003}
K1=${K1:-300500}
K2=${K2:-800500}
FAIL1=${FAIL1:-500000}       # the check that sees the K1 fault
WHAT=${1:-all}

fails=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }

if [[ ! -x "$FAULT_PRPLL" ]]; then
  make -s -C "$ROOT" BIN="$(dirname "$FAULT_PRPLL")" CPPFLAGS=-DPRPLL_PRP_FAULTS ${CUDA:+CUDA=$CUDA} || exit 1
fi
mkdir -p "$WORK"

# run <dir> [env assignments...] -- <args...>:  a fresh <dir> with one PRP line, prpll run in it (output in <dir>/out)
run() {
  local d=$1; shift
  local envs=()
  while [[ $1 != -- ]]; do envs+=("$1"); shift; done
  shift
  rm -rf "$d"; mkdir -p "$d"
  echo "PRP=0123456789ABCDEF0123456789ABCDEF,1,2,$E,-1,70,0" > "$d/worktodo-0.txt"
  (cd "$d" && env "${envs[@]}" "$FAULT_PRPLL" -device "$DEVICE" -log 10000 -noclean "$@" >> out 2>&1)
}

res64() { grep -o '"res64":"[0-9a-f]*"' "$1/results-0.txt" 2>/dev/null | cut -d'"' -f4; }
fftLen() { grep -o '"fft-length":[0-9]*' "$1/results-0.txt" 2>/dev/null | cut -d: -f2; }
gerbicz() { grep -o '"gerbicz":[0-9]*' "$1/results-0.txt" 2>/dev/null | cut -d: -f2; }

# The reference:  no fault.  Its FFT is the first FFT of the other runs.
ref=$WORK/ref
run "$ref" --
REF=$(res64 "$ref")
SPEC=$(grep -o 'FFT: [^ ]* [^ ]*' "$ref/out" | head -1 | cut -d' ' -f3)
[[ -n "$REF" && -n "$SPEC" ]] || { echo "the reference run failed (see $ref/out)"; exit 1; }
echo "reference: $E on $SPEC, res64 $REF, fft-length $(fftLen "$ref")"

if [[ $WHAT = all || $WHAT = once ]]; then
  d=$WORK/once
  run "$d" PRPLL_PRP_FAULT="$K1:$SPEC" --
  if [[ "$(res64 "$d")" = "$REF" ]] && grep -q "Consistent error .* at " "$d/out" && grep -q "until it passes the failing iteration" "$d/out" \
     && grep -q "passed iteration $FAIL1 (the error was at $FAIL1), back to FFT $SPEC" "$d/out" && [[ "$(fftLen "$d")" = "$(fftLen "$ref")" ]] \
     && [[ "$(gerbicz "$d")" = 2 ]]; then
    pass "once:  retried, back to $SPEC at $FAIL1, 2 errors, res64 $REF"
  else
    fail "once (see $d/out)"
  fi
fi

if [[ $WHAT = all || $WHAT = twice ]]; then
  d=$WORK/twice
  run "$d" PRPLL_PRP_FAULT="$K1:$SPEC,$K2:$SPEC" --
  if [[ "$(res64 "$d")" = "$REF" ]] && [[ "$(grep -c "Consistent error .* at " "$d/out")" = 2 ]] && grep -q "for the rest of the test" "$d/out" \
     && [[ "$(grep -c "back to FFT" "$d/out")" = 1 ]] && [[ "$(fftLen "$d")" != "$(fftLen "$ref")" ]] && [[ "$(gerbicz "$d")" = 4 ]]; then
    pass "twice:  stayed on the next FFT, 4 errors, res64 $REF"
  else
    fail "twice (see $d/out)"
  fi
fi

if [[ $WHAT = all || $WHAT = both ]]; then
  d=$WORK/both
  run "$d" PRPLL_PRP_FAULT="$K1" --
  if [[ -z "$(res64 "$d")" ]] && grep -q "Repeatable error on FFT .* too, will stop" "$d/out"; then
    pass "both:  stopped"
  else
    fail "both (see $d/out)"
  fi
fi

if [[ $WHAT = all || $WHAT = fft ]]; then
  d=$WORK/fft
  run "$d" PRPLL_PRP_FAULT="$K1" -- -fft "$SPEC"
  if [[ -z "$(res64 "$d")" ]] && grep -q "No FFT more accurate than $SPEC to retry on (-fft given), will stop" "$d/out"; then
    pass "fft:  no retry with -fft"
  else
    fail "fft (see $d/out)"
  fi
fi

echo "$fails failure(s)"
exit $((fails > 0))
