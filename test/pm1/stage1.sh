#!/usr/bin/env bash
# End-to-end tests of P-1 stage 1 in prpll, with GMP (stage1_oracle) as the oracle.  GMP is used only by the oracle, never by prpll.
#
#   test/pm1/stage1.sh [residue|resume|fault|all]      (default all)
# Environment:  DEVICE (default 1, the CPU OpenCL device here), WORK (default test/pm1/work), PRPLL (default build-release/prpll),
#               FAULT_PRPLL (default build-pm1faults/prpll, built here with -DPRPLL_PM1_FAULTS if missing).
# Exponents below about 393K cannot run (the smallest FFT is 128K words at no less than 3 bits per word).

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DEVICE=${DEVICE:-1}
WORK=${WORK:-$HERE/work}
PRPLL=${PRPLL:-$ROOT/build-release/prpll}
FAULT_PRPLL=${FAULT_PRPLL:-$ROOT/build-pm1faults/prpll}
ORACLE=$HERE/stage1_oracle
WHAT=${1:-all}

fails=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }

make -s -C "$HERE" stage1_oracle || exit 1
[ -x "$PRPLL" ] || { echo "no $PRPLL (run make)"; exit 1; }
mkdir -p "$WORK"

# setup <dir> <p> <B1> <B2>:  a fresh run directory with one Pminus1 line
setup() {
  rm -rf "$1"; mkdir -p "$1/mp"
  echo "Pminus1=1,2,$2,-1,$3,$4" > "$1/worktodo-0.txt"
}

# run <dir> <prpll> <args...>:  runs prpll in <dir>, appending to <dir>/out
run() {
  local d=$1 bin=$2; shift 2
  (cd "$d" && "$bin" -device "$DEVICE" -mprimeDir mp -log 10000 -noclean "$@" >> out 2>&1)
}

# oracle <dir> <p> <B1>:  compares the residue in the mprime savefile with GMP's
oracle() {
  local f
  f=$(ls "$1/mp" | grep -v worktodo)
  "$ORACLE" "$2" "$3" "$1/mp/$f" > "$1/oracle" 2>&1
  grep -q "residue EQUAL" "$1/oracle"
}

residue_tests() {
  # p, B1, B2, extra prpll arguments
  local cases=(
    "400009 1000 0"
    "400009 300000 0"
    "1000003 50000 2000000"
    "1257787 100000 100000"
    "1257787 30000 0 -fft 1:256:1:256"
    "2000003 20000 0 -fft 4:256:1:256"
    "4000037 10000 0"
  )
  [ -n "${RES_CASES:-}" ] && IFS=';' read -r -a cases <<< "$RES_CASES"     # e.g. RES_CASES="400009 1000 0;1000003 50000 0"
  for c in "${cases[@]}"; do
    set -- $c
    local p=$1 B1=$2 B2=$3; shift 3
    local d=$WORK/res-$p-$B1${1:+-fft}
    setup "$d" "$p" "$B1" "$B2"
    run "$d" "$PRPLL" "$@"
    local want=$B2; [ "$B2" -gt "$B1" ] || want=$((B1 + 1))
    if oracle "$d" "$p" "$B1" && grep -q "Pminus1=1,2,$p,-1,$B1,$want\$" "$d/mp/worktodo.add" && ! [ -s "$d/worktodo-0.txt" ] \
       && grep -q "stage 1 complete, residue handed to mprime" "$d/out" && grep -q " 0 Gerbicz errors" "$d/out"; then
      pass "residue p=$p B1=$B1 $* ($(grep -o 'FFT: [^ ]* [^ ]*' "$d/out" | head -1); $(grep -o 'res64=[0-9a-f]*' "$d/oracle"); line B2=$want)"
    else
      fail "residue p=$p B1=$B1 $*  (see $d)"
    fi
  done
}

resume_tests() {
  local p=1000003 B1=50000 d
  # -iters: stop at the first sub-block boundary at or after N iterations, then resume from the savefile
  d=$WORK/resume-iters
  setup "$d" $p $B1 0
  run "$d" "$PRPLL" -iters 25000
  run "$d" "$PRPLL" -iters 20000
  run "$d" "$PRPLL"
  local nLoads; nLoads=$(grep -c "P-1 loaded" "$d/out")
  if [ "$nLoads" = 2 ] && [ "$(grep -c 'stop requested' "$d/out")" = 2 ] && oracle "$d" $p $B1; then
    pass "stop with -iters twice, resumed twice ($(grep 'P-1 loaded' "$d/out" | grep -o 'OK *[0-9]*' | tr -s ' ' | paste -sd,)), residue equal"
  else
    fail "resume with -iters (see $d)"
  fi

  # SIGINT part way, then resume
  d=$WORK/resume-sigint
  setup "$d" $p $B1 0
  (cd "$d" && exec "$PRPLL" -device "$DEVICE" -mprimeDir mp -log 10000 -noclean >> out 2>&1) &
  local pid=$!
  until ls "$d/p1-$p/"*.p1 > /dev/null 2>&1 || ! kill -0 $pid 2>/dev/null; do sleep 0.2; done    # the first verified block
  kill -INT $pid; wait $pid
  run "$d" "$PRPLL"
  if grep -q "Stopping, please wait" "$d/out" && grep -q "P-1 loaded" "$d/out" && oracle "$d" $p $B1; then
    pass "SIGINT at $(grep 'P-1 loaded' "$d/out" | grep -o 'OK *[0-9]*' | tr -s ' '), resumed, residue equal"
  else
    fail "resume after SIGINT (see $d)"
  fi

  # A finished stage 1 whose hand-off did not happen (the worktodo line is still there):  only the hand-off is redone
  d=$WORK/resume-done
  setup "$d" $p 20000 0
  run "$d" "$PRPLL"
  echo "Pminus1=1,2,$p,-1,20000,0" > "$d/worktodo-0.txt"
  rm "$d/mp/"*
  run "$d" "$PRPLL"
  if grep -q "P-1 stage 1 already done" "$d/out" && oracle "$d" $p 20000; then
    pass "a finished stage 1 is not run again"
  else
    fail "finished stage 1 (see $d)"
  fi

  # A savefile of another B1 is set aside, not continued
  d=$WORK/resume-otherB1
  setup "$d" $p 20000 0
  run "$d" "$PRPLL" -iters 10000
  echo "Pminus1=1,2,$p,-1,30000,0" > "$d/worktodo-0.txt"
  run "$d" "$PRPLL"
  if grep -q "savefile is for B1=20000" "$d/out" && oracle "$d" $p 30000; then
    pass "a savefile of another B1 is set aside"
  else
    fail "other B1 (see $d)"
  fi
}

fault_tests() {
  if [ ! -x "$FAULT_PRPLL" ]; then
    make -s -C "$ROOT" -j8 BIN=build-pm1faults CPPFLAGS=-DPRPLL_PM1_FAULTS > /dev/null || { fail "fault build"; return; }
  fi
  local p=1000003 B1=50000 d spec
  # flip: one bit of x;  square: x * 4 (a perfect square: invisible to a Jacobi check);  check: one bit of the check product;
  # two faults in different blocks
  for spec in 33333:flip 50001:square 40000:check "20000:square,60000:flip"; do
    d=$WORK/fault-${spec//[:,]/-}
    setup "$d" $p $B1 0
    PRPLL_PM1_FAULT=$spec run "$d" "$FAULT_PRPLL"
    local nFaults; nFaults=$(grep -c "TEST FAULT" "$d/out")
    local nFailed; nFailed=$(grep -c "Gerbicz check failed" "$d/out")
    if [ "$nFaults" -ge 1 ] && [ "$nFailed" = "$nFaults" ] && grep -q " $nFaults Gerbicz errors" "$d/out" && oracle "$d" $p $B1; then
      pass "fault $spec: $nFaults injected, $nFailed caught ($(grep 'Gerbicz check failed' "$d/out" | grep -o 'failed at [0-9]*' | paste -sd,)), rolled back, residue equal"
    else
      fail "fault $spec (see $d)"
    fi
  done
}

case $WHAT in
  residue) residue_tests ;;
  resume) resume_tests ;;
  fault) fault_tests ;;
  all) residue_tests; resume_tests; fault_tests ;;
esac

echo "$fails failure(s)"
[ $fails = 0 ]
