#!/usr/bin/env bash
# End-to-end tests of P-1 with stage 2 in prpll, from "Pminus1=" worktodo lines:  stage 1, its GCD, stage 2 and its GCD, the result
# line, the hand-off to mprime when stage 2 declines, resume across the stage 1 / stage 2 boundary and inside stage 2, and the stage 2
# checks under simulated faults; and "Pfactor=" lines (pfactor:  the bounds prpll chooses equal check_bounds' for the stage 2 room it
# logged, kept across a restart, stage 1 alone without memory, no P-1 when it can't pay).  GMP (stage1_oracle, check_stage2) is the
# oracle; prpll links neither.
#
#   test/pm1/stage2.sh [factor|decline|resume|fault|fft|tunefile|pfactor|all]      (default all)
# Environment:  DEVICE (default 1, the CPU OpenCL device here), WORK (default test/pm1/work-stage2), PRPLL (default
#               build-release/prpll), FAULT_PRPLL (default build-pm1faults/prpll, built here with -DPRPLL_PM1_FAULTS if missing).

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DEVICE=${DEVICE:-1}
WORK=${WORK:-$HERE/work-stage2}
PRPLL=${PRPLL:-$ROOT/build-release/prpll}
FAULT_PRPLL=${FAULT_PRPLL:-$ROOT/build-pm1faults/prpll}
ORACLE=$HERE/stage1_oracle
CHECK=$HERE/check_stage2
BOUNDS=$HERE/check_bounds
WHAT=${1:-all}
TINY=(-maxAlloc 100M)       # leaves stage 2 less than its 256 MB reserve:  it declines, stage 1 still runs

fails=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }

make -s -C "$HERE" stage1_oracle check_stage2 || exit 1
[ -x "$PRPLL" ] || { echo "no $PRPLL (run make)"; exit 1; }
mkdir -p "$WORK"

# setup <dir> <p> <B1> <B2>:  a fresh run directory with one Pminus1 line (and an empty mprime directory mp)
setup() {
  rm -rf "$1"; mkdir -p "$1/mp"
  echo "Pminus1=1,2,$2,-1,$3,$4" > "$1/worktodo-0.txt"
}

# run <dir> <prpll> <args...>:  runs prpll in <dir>, appending to <dir>/out
run() {
  local d=$1 bin=$2; shift 2
  (cd "$d" && "$bin" -device "$DEVICE" -log 10000 -noclean "$@" >> out 2>&1)
}

lastSave() { ls -v "$1/$2-$3/"*."$2" 2>/dev/null | tail -1; }      # <dir> <p1|p2> <p>

# stage2Oracle <dir> <p> <B1> <B2>:  GMP's stage 1 residue equal to prpll's (final p1 savefile), the final stage 2 accumulator (p2
# savefile) equal to GMP's product for the D prpll used, and GMP's gcd(acc, 2^p-1) in <dir>/gcd2.  Needs -noclean.
stage2Oracle() {
  local d=$1 p=$2 B1=$3 B2=$4 D
  "$ORACLE" "$p" "$B1" "$(lastSave "$d" p1 "$p")" > "$d/oracle1" 2>&1
  grep -q "residue EQUAL" "$d/oracle1" || return 1
  "$CHECK" savefile "$p" "$(lastSave "$d" p1 "$p")" "$d/x.bin" > /dev/null || return 1
  "$CHECK" savefile "$p" "$(lastSave "$d" p2 "$p")" "$d/acc.bin" > "$d/oracle2" || return 1
  D=$(grep -o 'Stage 2 B1=[0-9]* B2=[0-9]*: D=[0-9]*' "$d/out" | tail -1 | sed 's/.*D=//')
  "$CHECK" acc "$p" "$B1" "$B2" "$D" "$d/x.bin" "$d/acc.bin" >> "$d/oracle2" || return 1
  "$CHECK" gcd "$p" "$d/acc.bin" | cut -d' ' -f2 > "$d/gcd2"
}

# result <dir> <p> <B1> <B2 or 0> <factor or NF>:  exactly one P-1 result line, with "b2" iff B2 > 0, and that status/factor;
# the worktodo line gone
result() {
  local d=$1 p=$2 B1=$3 B2=$4 want=$5 r
  [ -s "$d/worktodo-0.txt" ] && return 1
  r=$(cat "$d/results-0.txt" 2>/dev/null)
  [ "$(echo "$r" | grep -c .)" = 1 ] || return 1
  if [ "$B2" != 0 ]; then
    echo "$r" | grep -q "\"exponent\":$p, \"worktype\":\"P-1\", \"b1\":$B1, \"b2\":$B2, \"fft-length\"" || return 1
  else
    echo "$r" | grep -q "\"exponent\":$p, \"worktype\":\"P-1\", \"b1\":$B1, \"fft-length\"" || return 1
    echo "$r" | grep -q '"b2"' && return 1
  fi
  if [ "$want" = NF ]; then
    echo "$r" | grep -q '"status":"NF"' && ! echo "$r" | grep -q factors
  else
    echo "$r" | grep -q "\"status\":\"F\".*\"factors\":\[\"$want\"\]"
  fi
}

times() { echo "$(grep -o 'Stage 2 done: [0-9]* multiplications, [0-9]*s, [0-9.]* us/mul' "$1/out" | tail -1); $(grep -o 'stage 2 GCD: [0-9.]* s' "$1/out" | tail -1)"; }

# e2e <name> <p> <B1> <B2> <known factor that must divide the gcd, or NF> [prpll args]:  stage 2 run by prpll from the worktodo line
e2e() {
  local name=$1 p=$2 B1=$3 B2=$4 known=$5; shift 5
  local d=$WORK/$name g
  setup "$d" "$p" "$B1" "$B2"
  run "$d" "$PRPLL" "$@"
  if ! grep -q "P-1 stage 1 found no factor" "$d/out" || ! stage2Oracle "$d" "$p" "$B1" "$B2"; then fail "$name: oracle (see $d)"; return; fi
  g=$(cat "$d/gcd2"); [ "$g" = 1 ] && g=NF
  local ok=1
  result "$d" "$p" "$B1" "$B2" "$g" || ok=0
  if [ "$known" = NF ]; then [ "$g" = NF ] || ok=0; else python3 -c "import sys; sys.exit(int('$g') % $known != 0)" || ok=0; fi
  [ -n "$(ls "$d/mp")" ] && ok=0
  if [ $ok = 1 ]; then
    pass "$name: p=$p B1=$B1 B2=$B2 -> $g (= GMP gcd(acc); known $known), acc = GMP's product ($(grep -o '[0-9]* primes, [0-9]* pairs, [0-9]* multiplications' "$d/oracle2")); $(times "$d")"
  else
    fail "$name (see $d)"
  fi
}

factor_tests() {
  #   M400381:   671839319     = 2 * 839 * 400381 + 1          (839 in (500, 200000])
  #   M1000099:  1872347344039 = 2 * 3^2 * 104009 * 1000099 + 1 (104009 in (50000, 150000]); mprime.sh's hand-off test finds it too
  e2e s2-400381 400381 500 200000 671839319
  e2e s2-1000099 1000099 50000 150000 1872347344039
  e2e s2-nf 400033 1000 50000 NF
  e2e s2-inplace 400381 500 3000 671839319 -use INPLACE=1
  # The default P-1 FFT is an NTT type (here M61); an explicit -fft of another supported type (FP64) is used by both stages
  e2e s2-fp64 400381 500 3000 671839319 -fft 256:1:256
  local d=$WORK/s2-fp64
  if [ "$(grep -c 'FFT: 128K 256:1:256:' "$d/out")" = 2 ]; then pass "s2-fp64: -fft 256:1:256 (FP64) used by stage 1 and stage 2"; else fail "s2-fp64: FFT (see $d)"; fi
  d=$WORK/s2-400381
  if [ "$(grep -c 'FFT: 128K 3:256:1:256:' "$d/out")" = 2 ]; then pass "default P-1 FFT: 3:256:1:256 (M61, first in the untuned order) for both stages"; else fail "default P-1 FFT (see $d)"; fi
}

decline_tests() {
  local d p=400381 B1=500 B2=200000
  # -mprimeDir:  the stage 1 residue (equal to GMP's) and the line go to mprime, prpll writes no result
  d=$WORK/decline-mp
  setup "$d" $p $B1 $B2
  run "$d" "$PRPLL" "${TINY[@]}" -mprimeDir mp
  local f; f=$(ls "$d/mp" | grep -v worktodo)
  if grep -q "Stage 2 declined" "$d/out" && [ ! -e "$d/results-0.txt" ] && [ ! -s "$d/worktodo-0.txt" ] \
     && grep -q "^Pminus1=1,2,$p,-1,$B1,$B2\$" "$d/mp/worktodo.add" && "$ORACLE" $p $B1 "$d/mp/$f" > "$d/oracle-mp" 2>&1; then
    pass "declined with -mprimeDir: handed to mprime ($(grep -o 'Stage 2 declined: .*' "$d/out" | head -1); $f equal to GMP)"
  else
    fail "declined with -mprimeDir (see $d)"
  fi
  # without -mprimeDir:  NF for B1 alone (no "b2"), the p1 savefile kept
  d=$WORK/decline-nomp
  setup "$d" $p $B1 $B2
  run "$d" "$PRPLL" "${TINY[@]}"
  if grep -q "P-1 stage 2 (B2=$B2) is not run" "$d/out" && result "$d" $p $B1 0 NF && [ -n "$(lastSave "$d" p1 $p)" ] && [ -z "$(ls "$d/mp")" ]; then
    pass "declined without -mprimeDir: NF for B1=$B1 alone, savefile kept"
  else
    fail "declined without -mprimeDir (see $d)"
  fi
}

resume_tests() {
  local d p=400381 B1=500 B2=200000 f=671839319
  # Stage 1 finished in an earlier run (here one where stage 2 declined), the line given again:  stage 1 is loaded, not redone
  d=$WORK/resume-after-stage1
  setup "$d" $p $B1 $B2
  run "$d" "$PRPLL" "${TINY[@]}"
  echo "Pminus1=1,2,$p,-1,$B1,$B2" > "$d/worktodo-0.txt"
  rm -f "$d/results-0.txt"
  local before; before=$(ls -l --time-style=full-iso "$d/p1-$p")
  run "$d" "$PRPLL"
  if grep -q "P-1 stage 1 already done" "$d/out" && [ "$(ls -l --time-style=full-iso "$d/p1-$p")" = "$before" ] \
     && stage2Oracle "$d" $p $B1 $B2 && result "$d" $p $B1 $B2 "$(cat "$d/gcd2")" && [ "$(cat "$d/gcd2")" != 1 ]; then
    pass "restart after stage 1: stage 1 loaded, not rerun; stage 2 -> $(cat "$d/gcd2") = GMP"
  else
    fail "restart after stage 1 (see $d)"
  fi

  # -iters stops stage 2 (stage 1 is shorter, so it finishes); the next run loads stage 1, skips its GCD and resumes stage 2
  d=$WORK/resume-iters
  setup "$d" $p $B1 $B2
  run "$d" "$PRPLL" -iters 4000
  run "$d" "$PRPLL" -iters 4000
  run "$d" "$PRPLL"
  if [ "$(grep -c 'Stage 2 stopped at k=' "$d/out")" = 2 ] && [ "$(grep -c 'Stage 2 resuming at k=' "$d/out")" = 2 ] \
     && [ "$(grep -c 'P-1 stage 1 GCD skipped' "$d/out")" = 2 ] && [ "$(grep -c 'P-1 stage 1 already done' "$d/out")" = 2 ] \
     && [ "$(grep -c 'Exception "stop requested"' "$d/out")" = 2 ] \
     && stage2Oracle "$d" $p $B1 $B2 && result "$d" $p $B1 $B2 "$(cat "$d/gcd2")" && [ "$(cat "$d/gcd2")" != 1 ]; then
    pass "stage 2 stopped twice by -iters and resumed ($(grep -o 'resuming at k=[0-9]*' "$d/out" | paste -sd,)), acc = GMP, $(cat "$d/gcd2")"
  else
    fail "stage 2 resume with -iters (see $d)"
  fi

  # SIGINT inside stage 2, then resume
  d=$WORK/resume-sigint
  setup "$d" $p $B1 $B2
  (cd "$d" && exec "$PRPLL" -device "$DEVICE" -log 10000 -noclean >> out 2>&1) &
  local pid=$!
  # (the log file, not out:  stdout into a file is block-buffered)
  until grep -q "Stage 2 table of" "$d/gpuowl-0.log" 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 0.2; done
  sleep 3
  kill -INT $pid; wait $pid
  run "$d" "$PRPLL"
  if grep -q "Stage 2 stopped at k=" "$d/out" && grep -q "Stage 2 resuming at k=" "$d/out" && stage2Oracle "$d" $p $B1 $B2 \
     && result "$d" $p $B1 $B2 "$(cat "$d/gcd2")" && [ "$(cat "$d/gcd2")" != 1 ]; then
    pass "SIGINT in stage 2 ($(grep -o 'stopped at k=[0-9]*' "$d/out")), resumed, acc = GMP"
  else
    fail "SIGINT in stage 2 (see $d)"
  fi
}

fault_tests() {
  if [ ! -x "$FAULT_PRPLL" ]; then
    make -s -C "$ROOT" -j8 BIN=build-pm1faults CPPFLAGS=-DPRPLL_PM1_FAULTS > /dev/null || { fail "fault build"; return; }
  fi
  local p=400381 B1=500 B2=30000 d kind
  # table: a table entry overwritten (caught at the next checkpoint by its sum);  A: A_k overwritten (caught against x^((kD)^2));
  # acc: the accumulator overwritten in a duplicated block.  Each goes back to the last checkpoint and ends with GMP's accumulator.
  for kind in table A acc; do
    d=$WORK/fault-$kind
    setup "$d" $p $B1 $B2
    PRPLL_STAGE2_FAULT=$kind,5 run "$d" "$FAULT_PRPLL"
    if grep -q "TEST FAULT: a simulated '$kind'" "$d/out" && grep -q "Stage 2 going back to k=" "$d/out" && stage2Oracle "$d" $p $B1 $B2 \
       && result "$d" $p $B1 $B2 "$(cat "$d/gcd2")"; then
      pass "stage 2 fault $kind: $(grep 'EE stage 2' "$d/out" | head -1 | sed 's/.*EE stage 2: //'), went back, acc = GMP"
    else
      fail "stage 2 fault $kind (see $d)"
    fi
  done
}

# FFT choice:  a stage 2 FFT that can't be built is replaced (no -fft) without redoing stage 1; an -fft is never replaced
fft_tests() {
  if [ ! -x "$FAULT_PRPLL" ]; then
    make -s -C "$ROOT" -j8 BIN=build-pm1faults CPPFLAGS=-DPRPLL_PM1_FAULTS > /dev/null || { fail "fault build"; return; }
  fi
  local d p=400381 B1=500 B2=3000 f=671839319
  # The first stage 2 FFT (M61 here) "fails to build" (fault build):  the next one (FP32+M31) runs stage 2, stage 1 ran once
  d=$WORK/fft-fallback
  setup "$d" $p $B1 $B2
  PRPLL_STAGE2_BUILD_FAULT=1 run "$d" "$FAULT_PRPLL"
  local first; first=$(grep -o "Stage 2 can't build FFT [^ ]*" "$d/out" | head -1 | cut -d' ' -f6)
  if [ -n "$first" ] && grep -q "trying the next FFT" "$d/out" && [ "$(grep -c 'P-1 stage 1 B1=' "$d/out")" = 1 ] \
     && ! grep -q "already done" "$d/out" && [ "$(grep -o 'FFT: [^ ]* [^ ]*' "$d/out" | sort -u | wc -l)" -ge 2 ] \
     && stage2Oracle "$d" $p $B1 $B2 && result "$d" $p $B1 $B2 $f; then
    pass "stage 2 FFT $first rejected (simulated build failure), stage 2 on $(grep -o 'FFT: [^ ]* [^ ]*' "$d/out" | tail -1 | cut -d' ' -f3), stage 1 run once, acc = GMP"
  else
    fail "stage 2 FFT fallback (see $d)"
  fi
  # No stage 2 FFT can be built:  stop, the line and the savefiles stay; the next run loads stage 1 and finishes
  d=$WORK/fft-nonebuild
  setup "$d" $p $B1 $B2
  PRPLL_STAGE2_BUILD_FAULT=99 run "$d" "$FAULT_PRPLL"
  local stopped=0
  grep -q 'Exception "P-1 stage 2: no FFT could be built"' "$d/out" && [ -s "$d/worktodo-0.txt" ] && [ ! -e "$d/results-0.txt" ] \
    && [ -n "$(lastSave "$d" p1 $p)" ] && [ "$(grep -c 'P-1 stage 1 B1=' "$d/out")" = 1 ] && stopped=1
  run "$d" "$PRPLL"
  if [ $stopped = 1 ] && grep -q "P-1 stage 1 already done" "$d/out" && stage2Oracle "$d" $p $B1 $B2 && result "$d" $p $B1 $B2 $f; then
    pass "no stage 2 FFT builds: stopped once with the line and savefiles kept; the rerun loads stage 1 and finds $f"
  else
    fail "no stage 2 FFT builds (see $d)"
  fi
  # An -fft whose kernels "fail to build" in stage 2:  one line, stop, nothing replaced
  d=$WORK/fft-user-build
  setup "$d" $p $B1 $B2
  PRPLL_STAGE2_BUILD_FAULT=1 run "$d" "$FAULT_PRPLL" -fft 1:256:1:256
  if grep -q "the kernels of the -fft FFT 1:256:1:256.* can't be built .*P-1 does not replace it" "$d/out" \
     && grep -q 'Exception "P-1 stage 2: the -fft FFT can.t be built"' "$d/out" && ! grep -q "Stage 2 B1=" "$d/out" \
     && [ -s "$d/worktodo-0.txt" ] && [ ! -e "$d/results-0.txt" ] && [ "$(grep -o 'FFT: [^ ]* [^ ]*' "$d/out" | sort -u | wc -l)" = 1 ]; then
    pass "-fft with a stage 2 build failure: $(grep -o 'P-1 stage 2: the kernels.*' "$d/out" | head -1 | cut -c1-90)..., stopped, line kept"
  else
    fail "-fft with a stage 2 build failure (see $d)"
  fi
  # An -fft that fits stage 1 but not stage 2's margin (1:256:1:256 handles 5.31M; stage 2 of 5280007 needs 5280007 + 0.5 * 128K)
  d=$WORK/fft-user-margin
  setup "$d" 5280007 50 1000
  run "$d" "$PRPLL" -fft 1:256:1:256
  if grep -q "P-1 stage 2 of 5280007 can't run on the -fft given, and P-1 does not replace it: 1:256:1:256.* handles up to [0-9]*, needs 5345543" "$d/out" \
     && ! grep -q "Stage 2 B1=" "$d/out" && [ -s "$d/worktodo-0.txt" ] && [ ! -e "$d/results-0.txt" ] && [ -n "$(lastSave "$d" p1 5280007)" ]; then
    pass "-fft too small for stage 2: $(grep -o 'P-1 stage 2 of 5280007.*' "$d/out" | head -1), stopped, line and p1 savefile kept"
  elif grep -q "P-1 found a factor in stage 1" "$d/out"; then
    fail "-fft margin case: stage 1 found a factor, pick another exponent (see $d)"
  else
    fail "-fft too small for stage 2 (see $d)"
  fi
  # An -fft too large for stage 1:  stopped before any work, not replaced
  d=$WORK/fft-user-large
  setup "$d" $p $B1 $B2
  run "$d" "$PRPLL" -fft 1:1K:2:256
  if grep -q "P-1 stage 1 of $p can't run on the -fft given, and P-1 does not replace it: 1:1K:2:256.* is too large" "$d/out" \
     && ! grep -q 'FFT: ' "$d/out" && [ -s "$d/worktodo-0.txt" ]; then
    pass "-fft too large for stage 1: $(grep -o 'P-1 stage 1 of.*' "$d/out" | head -1), stopped, line kept"
  else
    fail "-fft too large for stage 1 (see $d)"
  fi
}

# Stored FFT choices:  stage 2 takes the fitting tune-pm1.txt line (-tune pm1) with its settings, else the tune.txt line (PRP's); stage
# 1 never reads tune-pm1.txt; the command line beats a line's settings; an -fft takes its tune-pm1.txt line's settings in stage 2 and its
# tune.txt line's in stage 1
tunefile_tests() {
  local d p=400381 B1=500 B2=3000 f=671839319
  # stage2FFT <dir>:  the FFT stage 2 ran on (the last "FFT:" line);  stage1FFT:  the first
  stage2FFT() { grep -o 'FFT: [^ ]* [^ ]*' "$1/out" | tail -1 | cut -d' ' -f3; }
  stage1FFT() { grep -o 'FFT: [^ ]* [^ ]*' "$1/out" | head -1 | cut -d' ' -f3; }
  check() {     # <dir> <name> <stage 1 FFT> <stage 2 FFT> <source> <grep pattern that must be in out or ''>
    if [ "$(stage1FFT "$1")" = "$3" ] && [ "$(stage2FFT "$1")" = "$4" ] && grep -q "Stage 2 FFT $4 ($5)" "$1/out" \
       && { [ -z "$6" ] || grep -q "$6" "$1/out"; } && stage2Oracle "$1" $p $B1 $B2 && result "$1" $p $B1 $B2 $f; then
      pass "$2: stage 1 on $3, stage 2 on $4 ($5)${6:+, '$6'}, acc = GMP"
    else
      fail "$2 (see $1)"
    fi
  }

  d=$WORK/tf-pm1
  setup "$d" $p $B1 $B2
  echo "  100.0 4:256:1:256:202 # 0, TAIL_KERNELS=3" > "$d/tune-pm1.txt"
  run "$d" "$PRPLL"
  check "$d" "tune-pm1.txt line" 3:256:1:256:101 4:256:1:256:202 tune-pm1.txt "4:256:1:256:202: tune.txt sets TAIL_KERNELS=3"

  d=$WORK/tf-prp
  setup "$d" $p $B1 $B2
  echo "  100.0 1:256:1:256:202 # 0, TAIL_KERNELS=1" > "$d/tune.txt"
  run "$d" "$PRPLL"
  check "$d" "tune.txt line" 1:256:1:256:202 1:256:1:256:202 tune.txt "1:256:1:256:202: tune.txt sets TAIL_KERNELS=1"

  d=$WORK/tf-both
  setup "$d" $p $B1 $B2
  echo "  100.0 1:256:1:256:202 # 0, TAIL_KERNELS=1" > "$d/tune.txt"
  echo "  100.0 4:256:1:256:202 # 0, TAIL_KERNELS=3" > "$d/tune-pm1.txt"
  run "$d" "$PRPLL"
  check "$d" "tune-pm1.txt and tune.txt lines" 1:256:1:256:202 4:256:1:256:202 tune-pm1.txt "4:256:1:256:202: tune.txt sets TAIL_KERNELS=3"

  d=$WORK/tf-cmdline
  setup "$d" $p $B1 $B2
  echo "  100.0 4:256:1:256:202 # 0, TAIL_KERNELS=3" > "$d/tune-pm1.txt"
  run "$d" "$PRPLL" -use TAIL_KERNELS=0
  check "$d" "command line over the tune-pm1.txt line" 3:256:1:256:101 4:256:1:256:202 tune-pm1.txt \
    "tune.txt TAIL_KERNELS=3 overridden by command line TAIL_KERNELS=0"

  # an -fft's settings:  stage 2 from its tune-pm1.txt line, stage 1 from its tune.txt line; and the same -fft without a tune-pm1.txt
  # line takes the tune.txt line's in stage 2 too
  d=$WORK/tf-spec
  setup "$d" $p $B1 $B2
  echo "  100.0 4:256:1:256:202 # 0, TAIL_KERNELS=1" > "$d/tune.txt"
  echo "  100.0 4:256:1:256:202 # 0, TAIL_KERNELS=3" > "$d/tune-pm1.txt"
  run "$d" "$PRPLL" -fft 4:256:1:256:202
  if [ "$(grep -c '4:256:1:256:202: tune.txt sets TAIL_KERNELS=1' "$d/out")" = 1 ]; then
    check "$d" "-fft with both lines" 4:256:1:256:202 4:256:1:256:202 -fft "4:256:1:256:202: tune.txt sets TAIL_KERNELS=3"
  else
    fail "-fft with both lines: stage 1 settings (see $d)"
  fi
  # tune-pm1.txt as -tune pm1 wrote it on PoCL (7 types at 128K):  for p = 3000017 the 1:256:1:256 line with its TAIL_KERNELS=1; -use on
  # the command line wins over it.  (Stage 1 has no tune.txt here:  the untuned order.)
  local p0=$p; p=3000017
  local tuned='13130.2  256:1:256:101 # 2496528
17984.5 1:256:1:256:202 # 5248123, TAIL_KERNELS=1
25403.4 4:256:1:256:202 # 6633554, TAIL_KERNELS=3'
  for v in plain cmdline; do
    d=$WORK/tf-tuned-$v
    setup "$d" $p $B1 $B2
    echo "$tuned" > "$d/tune-pm1.txt"
    if [ $v = plain ]; then run "$d" "$PRPLL"; else run "$d" "$PRPLL" -use TAIL_KERNELS=2; fi
    local want="1:256:1:256:202: tune.txt sets TAIL_KERNELS=1"
    [ $v = cmdline ] && want="tune.txt TAIL_KERNELS=1 overridden by command line TAIL_KERNELS=2"
    local s1; s1=$(stage1FFT "$d")
    if [ "$(stage2FFT "$d")" = 1:256:1:256:202 ] && grep -q "Stage 2 FFT 1:256:1:256:202 (tune-pm1.txt)" "$d/out" && grep -q "$want" "$d/out" \
       && stage2Oracle "$d" $p $B1 $B2 && result "$d" $p $B1 $B2 "$(sed 's/^1$/NF/' "$d/gcd2")"; then
      pass "tuner-written tune-pm1.txt ($v): stage 1 on $s1, stage 2 on 1:256:1:256:202 (tune-pm1.txt), '$want', acc = GMP, $(sed 's/^1$/NF/' "$d/gcd2")"
    else
      fail "tuner-written tune-pm1.txt ($v) (see $d)"
    fi
  done
  p=$p0

  # A tune.txt line with the narrow carry (PRP tunes both widths) where stage 2's doubled products need the wide one (19.04 bits per word
  # on 256:2:256, narrow carry limit 20):  stage 1 keeps the line's carry, stage 2 widens it
  d=$WORK/tf-carry
  setup "$d" 4990049 11 1000
  echo "  100.0 256:2:256:101:0 # 0" > "$d/tune.txt"
  run "$d" "$PRPLL"
  if [ "$(stage1FFT "$d")" = 256:2:256:101:0 ] && grep -q "Stage 2 FFT 256:2:256:101:1 (tune.txt)" "$d/out" && [ "$(stage2FFT "$d")" = 256:2:256:101:1 ] \
     && { grep -q "P-1 found a factor in stage 1" "$d/out" || stage2Oracle "$d" 4990049 11 1000; }; then
    pass "tune.txt 256:2:256:101:0 at 19.04 bpw: stage 1 on :0, stage 2 on 256:2:256:101:1 (wide carry)"
  else
    fail "narrow carry widened for stage 2 (see $d)"
  fi

  d=$WORK/tf-spec2
  setup "$d" $p $B1 $B2
  echo "  100.0 4:256:1:256:202 # 0, TAIL_KERNELS=1" > "$d/tune.txt"
  run "$d" "$PRPLL" -fft 4:256:1:256:202
  if [ "$(grep -c '4:256:1:256:202: tune.txt sets TAIL_KERNELS=1' "$d/out")" = 2 ]; then
    check "$d" "-fft with a tune.txt line only" 4:256:1:256:202 4:256:1:256:202 -fft ""
  else
    fail "-fft with a tune.txt line only: settings (see $d)"
  fi
}

# ---- Pfactor lines:  the bounds chosen by prpll (src/Pm1Bounds.cpp), checked against check_bounds for the same stage 2 room ----

# psetup <dir> <p> <tfBits> <testsSaved>
psetup() {
  rm -rf "$1"; mkdir -p "$1/mp"
  echo "Pfactor=N/A,1,2,$2,-1,$3,$4" > "$1/worktodo-0.txt"
}

# chosen <dir>:  "B1 B2" of the bounds prpll logged (the last choice)
chosen() { grep -o 'P-1 bounds.*: B1=[0-9]* B2=[0-9]*' "$1/out" | tail -1 | sed 's/.*B1=\([0-9]*\) B2=\([0-9]*\)/\1 \2/'; }

# modelOK <dir> <p> <tf> <ts>:  the logged bounds are check_bounds' choice for the stage 2 room prpll logged
modelOK() {
  local room; room=$(grep -o 'room for [0-9]* table buffers' "$1/out" | tail -1 | grep -o '[0-9]*')
  [ -n "$room" ] || return 1
  local want; want=$("$BOUNDS" "$2,$3,$4,$room" | sed 's/.*: B1=\([0-9]*\) B2=\([0-9]*\) .*/\1 \2/')
  [ "$(chosen "$1")" = "$want" ]
}

pfactor_tests() {
  make -s -C "$HERE" check_bounds || { fail "check_bounds does not build"; return; }
  local d B1 B2 p=400381 f=671839319
  # A stage 2 factor:  M400381, 671839319 = 2 * 839 * 400381 + 1;  testsSaved 0.3 gives B1 < 839 < B2.  acc = GMP's product
  d=$WORK/pf-stage2
  psetup "$d" $p 25 0.3
  run "$d" "$PRPLL" -maxAlloc 1G
  read -r B1 B2 <<< "$(chosen "$d")"
  if [ -n "$B1" ] && [ "$B1" -lt 839 ] && [ "$B2" -gt 839 ] && modelOK "$d" $p 25 0.3 && stage2Oracle "$d" $p "$B1" "$B2" \
     && [ "$(cat "$d/gcd2")" = $f ] && result "$d" $p "$B1" "$B2" $f; then
    pass "pf-stage2: Pfactor $p tf 25 saved 0.3 -> B1=$B1 B2=$B2 (= check_bounds), factor $f in stage 2 (= GMP gcd(acc)), result has the bounds"
  else
    fail "pf-stage2 (see $d)"
  fi
  # A stage 1 factor (testsSaved 1:  B1 > 839):  the result has b1 alone
  d=$WORK/pf-stage1
  psetup "$d" $p 25 1
  run "$d" "$PRPLL" -maxAlloc 1G
  read -r B1 B2 <<< "$(chosen "$d")"
  if [ -n "$B1" ] && [ "$B1" -gt 839 ] && [ "$B2" -gt "$B1" ] && modelOK "$d" $p 25 1 && result "$d" $p "$B1" 0 $f \
     && grep -q "P-1 found a factor in stage 1: $f" "$d/out"; then
    pass "pf-stage1: Pfactor $p tf 25 saved 1 -> B1=$B1 B2=$B2 (= check_bounds), factor $f in stage 1, result b1=$B1 without b2"
  else
    fail "pf-stage1 (see $d)"
  fi
  # No memory for stage 2:  B2 = B1, chosen for stage 1 alone, logged
  d=$WORK/pf-nomem
  psetup "$d" $p 25 1
  run "$d" "$PRPLL" "${TINY[@]}"
  read -r B1 B2 <<< "$(chosen "$d")"
  if [ -n "$B1" ] && [ "$B1" = "$B2" ] && grep -q "room for 0 table buffers" "$d/out" && modelOK "$d" $p 25 1 \
     && grep -q "P-1 without stage 2 (B2 = B1): too little GPU memory" "$d/out" && result "$d" $p "$B1" 0 $f; then
    pass "pf-nomem: -maxAlloc 100M -> B1=B2=$B1 (= check_bounds without stage 2), logged, factor $f in stage 1"
  else
    fail "pf-nomem (see $d)"
  fi
  # No memory, and stage 1 alone does not pay at M1000099 TF 50 (while P-1 with a stage 2 would):  stage 1 alone runs all the same
  d=$WORK/pf-nomem-neg
  psetup "$d" 1000099 50 1
  run "$d" "$PRPLL" "${TINY[@]}"
  read -r B1 B2 <<< "$(chosen "$d")"
  if [ -n "$B1" ] && [ "$B1" = "$B2" ] && modelOK "$d" 1000099 50 1 && grep -q "stage 1 alone does not pay for itself either" "$d/out" \
     && grep -q '"exponent":1000099, "worktype":"P-1", "b1":'"$B1"', "fft-length"' "$d/results-0.txt" && [ ! -s "$d/worktodo-0.txt" ]; then
    pass "pf-nomem-neg: no stage 2 room, stage 1 alone not paying -> still run with B1=B2=$B1 (= check_bounds), result b1 only"
  else
    fail "pf-nomem-neg (see $d)"
  fi
  # Resume:  stopped in stage 1, restarted with less memory (which alone would choose B2 = B1):  the savefile's bounds are kept
  d=$WORK/pf-resume
  psetup "$d" 400033 25 0.3
  run "$d" "$PRPLL" -maxAlloc 1G -iters 300
  local first; first=$(chosen "$d")
  run "$d" "$PRPLL" "${TINY[@]}"
  read -r B1 B2 <<< "$first"
  if [ -n "$B1" ] && [ "$B2" -gt "$B1" ] && grep -q "P-1 bounds from the stage 1 savefile: B1=$B1 B2=$B2" "$d/out" \
     && head -1 "$(lastSave "$d" p1 400033)" | grep -q "^OWL P1 2 N=1\*2^400033-1 B1=$B1 B2=$B2 " \
     && [ "$(grep -c 'P-1 bounds (no factor' "$d/out")" = 1 ]; then
    # The second run's stage 2 declines (-maxAlloc 100M):  NF for B1 alone, as a Pminus1 line with these bounds would give
    if result "$d" 400033 "$B1" 0 NF && grep -q "P-1 stage 2 (B2=$B2) is not run" "$d/out"; then
      pass "pf-resume: bounds B1=$B1 B2=$B2 kept from the p1 savefile (version 2 header) across a restart with -maxAlloc 100M"
    else
      fail "pf-resume: result (see $d)"
    fi
  else
    fail "pf-resume (see $d)"
  fi
  # Not worth it even with stage 2's largest table (no test saved):  nothing runs, no result, the line is deleted
  d=$WORK/pf-zero
  psetup "$d" $p 25 0
  run "$d" "$PRPLL" -maxAlloc 1G
  if grep -q "P-1 is not worth its cost for $p" "$d/out" && [ ! -e "$d/results-0.txt" ] && [ ! -s "$d/worktodo-0.txt" ] \
     && ! grep -q "P-1 stage 1 B1" "$d/out"; then
    pass "pf-zero: testsSaved 0 -> no P-1, line deleted, no result"
  else
    fail "pf-zero (see $d)"
  fi
  # Bad lines are not run:  TF depth >= 100, a negative testsSaved
  for bad in "Pfactor=N/A,1,2,$p,-1,150,1" "Pfactor=N/A,1,2,$p,-1,70,-1"; do
    d=$WORK/pf-bad
    rm -rf "$d"; mkdir -p "$d"; echo "$bad" > "$d/worktodo-0.txt"
    run "$d" "$PRPLL" -maxAlloc 1G
    if [ ! -e "$d/results-0.txt" ] && grep -q "No work to do" "$d/out" && [ -s "$d/worktodo-0.txt" ]; then
      pass "pf-bad: '$bad' not run"
    else
      fail "pf-bad: '$bad' (see $d)"
    fi
  done
}

case $WHAT in
  factor) factor_tests ;;
  decline) decline_tests ;;
  resume) resume_tests ;;
  fault) fault_tests ;;
  fft) fft_tests ;;
  tunefile) tunefile_tests ;;
  pfactor) pfactor_tests ;;
  all) factor_tests; decline_tests; resume_tests; fault_tests; fft_tests; tunefile_tests; pfactor_tests ;;
esac

echo "$fails failure(s)"
[ $fails = 0 ]
