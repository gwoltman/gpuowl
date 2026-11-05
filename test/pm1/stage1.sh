#!/usr/bin/env bash
# End-to-end tests of P-1 stage 1 and its GCD in prpll, with GMP (stage1_oracle) as the oracle.  GMP is used only by the oracle, never
# by prpll.
#
#   test/pm1/stage1.sh [residue|resume|fault|jacobi|all]      (default all)
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
FLIP=$HERE/p1flip
WHAT=${1:-all}

fails=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }

make -s -C "$HERE" stage1_oracle p1flip || exit 1
[[ -x "$PRPLL" ]] || { echo "no $PRPLL (run make)"; exit 1; }
mkdir -p "$WORK"

# setup <dir> <p> <B1> <B2>:  a fresh run directory with one Pminus1 line
setup() {
  rm -rf "$1"; mkdir -p "$1/mp"
  echo "Pminus1=1,2,$2,-1,$3,$4" > "$1/worktodo-0.txt"
}

# run <dir> <prpll> <args...>:  runs prpll in <dir> (with -mprimeDir mp unless NOMP is set), appending to <dir>/out
run() {
  local d=$1 bin=$2; shift 2
  local mp=(-mprimeDir mp); [[ -n "${NOMP:-}" ]] && mp=()
  (cd "$d" && "$bin" -device "$DEVICE" "${mp[@]}" -log 10000 -noclean "$@" >> out 2>&1)
}

# lastSave <dir> <p>:  prpll's most advanced P-1 savefile
# shellcheck disable=SC2012  # the names are prpll's own savefile names: no spaces or newlines
lastSave() { ls "$1/p1-$2/"*.p1 2>/dev/null | sort | tail -1; }

# oracle <dir> <p> <B1>:  compares the final residue in prpll's own savefile with GMP's (needs -noclean); GMP's gcd(x - 1, 2^p-1)
# is then in <dir>/gcd
oracle() {
  "$ORACLE" "$2" "$3" "$(lastSave "$1" "$2")" > "$1/oracle" 2>&1
  grep '^gcd ' "$1/oracle" | cut -d' ' -f2 > "$1/gcd"
  grep -q "residue EQUAL" "$1/oracle"
}

# outcome <dir> <p> <B1> <B2> [nomp]:  after oracle, checks what prpll did with the GMP gcd:  a factor -> an "F" result with it, B1 only;
# no factor and B2 <= B1 -> an "NF" result for B1; no factor and B2 > B1 -> the residue handed to mprime (equal to GMP's) and no
# result line of prpll's, or without -mprimeDir an "NF" result for B1 only.  The worktodo line is gone in every case.  The B2 > B1
# cases here run with -maxAlloc 100M, so that stage 2 declines (prpll's own stage 2:  stage2.sh).
outcome() {
  local d=$1 p=$2 B1=$3 B2=$4 nomp=${5:-} g; g=$(cat "$d/gcd")
  [[ -s "$d/worktodo-0.txt" ]] && return 1
  if [[ "$g" != 1 ]] || [[ "$B2" -le "$B1" ]] || [[ -n "$nomp" ]]; then
    local r; r=$(cat "$d/results-0.txt" 2>/dev/null)
    [[ "$(echo "$r" | grep -c .)" = 1 ]] || return 1
    echo "$r" | grep -q "\"exponent\":$p, \"worktype\":\"P-1\", \"b1\":$B1, \"fft-length\"" || return 1
    if [[ "$g" = 1 ]]; then
      echo "$r" | grep -q '"status":"NF"' && ! echo "$r" | grep -q factors || return 1
    else
      echo "$r" | grep -q "\"status\":\"F\".*\"factors\":\[\"$g\"\]" || return 1
    fi
    [[ -z "$(ls "$d/mp" 2>/dev/null)" ]] || return 1
  else
    [[ -e "$d/results-0.txt" ]] && return 1
    # shellcheck disable=SC2010  # mprime's savefile name (p<exponent>-style): no spaces or newlines
    "$ORACLE" "$p" "$B1" "$d/mp/$(ls "$d/mp" | grep -v worktodo)" > "$d/oracle-mp" 2>&1 || return 1
    grep -q "Pminus1=1,2,$p,-1,$B1,$B2\$" "$d/mp/worktodo.add" || return 1
  fi
}

residue_tests() {
  # p, B1, B2, extra prpll arguments
  local cases=(
    "400009 1000 0"
    "400009 300000 0"
    "1000003 50000 2000000 -maxAlloc 100M"
    "1257787 100000 100000"
    "1257787 30000 0 -fft 1:256:1:256"
    "2000003 20000 0 -fft 4:256:1:256"
    "4000037 10000 0"
    "400381 1000 0"
    "400381 500 0"
    "400381 500 1000 -maxAlloc 100M"
    "400381 1000 2000"
    "1000253 50000 0"
    "400033 1000 0"
  )
  [[ -n "${RES_CASES:-}" ]] && IFS=';' read -r -a cases <<< "$RES_CASES"     # e.g. RES_CASES="400009 1000 0;1000003 50000 0"
  for c in "${cases[@]}"; do
    local words; read -r -a words <<< "$c"; set -- "${words[@]}"
    local p=$1 B1=$2 B2=$3; shift 3
    local d=$WORK/res-$p-$B1-$B2${1:+-fft}
    setup "$d" "$p" "$B1" "$B2"
    run "$d" "$PRPLL" "$@"
    if oracle "$d" "$p" "$B1" && outcome "$d" "$p" "$B1" "$B2" && ! grep -q "Gerbicz check failed" "$d/out"; then
      pass "residue p=$p B1=$B1 B2=$B2 $* ($(grep -o 'FFT: [^ ]* [^ ]*' "$d/out" | head -1); $(grep -o 'res64=[0-9a-f]*' "$d/oracle"); gcd $(cat "$d/gcd") = prpll's; $(grep -o 'GCD: [0-9.]* s' "$d/out"))"
    else
      fail "residue p=$p B1=$B1 B2=$B2 $*  (see $d)"
    fi
  done

  # No -mprimeDir and B2 > B1, stage 2 declined:  stage 1 and its GCD, the result for B1 alone, the savefile kept for a later stage 2
  local d=$WORK/res-nomp
  setup "$d" 400381 500 1000
  NOMP=1 run "$d" "$PRPLL" -maxAlloc 100M
  if oracle "$d" 400381 500 && outcome "$d" 400381 500 1000 nomp && grep -q "stage 2 (B2=1000) is not run" "$d/out"; then
    pass "no -mprimeDir, B2 > B1: NF for B1=500 only, stage 2 not claimed, savefile kept"
  else
    fail "no -mprimeDir (see $d)"
  fi
}

resume_tests() {
  local p=1000003 B1=50000 d
  # -iters: stop at the first sub-block boundary at or after N iterations, then resume from the savefile
  d=$WORK/resume-iters
  setup "$d" "$p" "$B1" 0
  run "$d" "$PRPLL" -iters 25000
  run "$d" "$PRPLL" -iters 20000
  run "$d" "$PRPLL"
  local nLoads; nLoads=$(grep -c "P-1 loaded" "$d/out")
  if [[ "$nLoads" = 2 ]] && [[ "$(grep -c 'stop requested' "$d/out")" = 2 ]] && oracle "$d" "$p" "$B1"; then
    pass "stop with -iters twice, resumed twice ($(grep 'P-1 loaded' "$d/out" | grep -o 'OK *[0-9]*' | tr -s ' ' | paste -sd,)), residue equal"
  else
    fail "resume with -iters (see $d)"
  fi

  # SIGINT part way, then resume
  d=$WORK/resume-sigint
  setup "$d" "$p" "$B1" 0
  (cd "$d" && exec "$PRPLL" -device "$DEVICE" -mprimeDir mp -log 10000 -noclean >> out 2>&1) &
  local pid=$!
  until ls "$d/p1-$p/"*.p1 > /dev/null 2>&1 || ! kill -0 "$pid" 2>/dev/null; do sleep 0.2; done    # the first verified block
  kill -INT "$pid"; wait "$pid"
  run "$d" "$PRPLL"
  if grep -q "Stopping, please wait" "$d/out" && grep -q "P-1 loaded" "$d/out" && oracle "$d" "$p" "$B1"; then
    pass "SIGINT at $(grep 'P-1 loaded' "$d/out" | grep -o 'OK *[0-9]*' | tr -s ' '), resumed, residue equal"
  else
    fail "resume after SIGINT (see $d)"
  fi

  # A finished stage 1 whose result was not written (the worktodo line is still there):  only the GCD and the result are redone
  d=$WORK/resume-done
  setup "$d" "$p" 20000 0
  run "$d" "$PRPLL"
  echo "Pminus1=1,2,$p,-1,20000,0" > "$d/worktodo-0.txt"
  rm "$d/results-0.txt"
  run "$d" "$PRPLL"
  if grep -q "P-1 stage 1 already done" "$d/out" && oracle "$d" "$p" 20000 && outcome "$d" "$p" 20000 0; then
    pass "a finished stage 1 is not run again"
  else
    fail "finished stage 1 (see $d)"
  fi

  # A savefile of another B1 is set aside, not continued
  d=$WORK/resume-otherB1
  setup "$d" "$p" 20000 0
  run "$d" "$PRPLL" -iters 10000
  echo "Pminus1=1,2,$p,-1,30000,0" > "$d/worktodo-0.txt"
  run "$d" "$PRPLL"
  if grep -q "savefile is for B1=20000" "$d/out" && oracle "$d" "$p" 30000; then
    pass "a savefile of another B1 is set aside"
  else
    fail "other B1 (see $d)"
  fi
}

fault_tests() {
  if [[ ! -x "$FAULT_PRPLL" ]]; then
    make -s -C "$ROOT" -j8 BIN=build-pm1faults CPPFLAGS=-DPRPLL_PM1_FAULTS > /dev/null || { fail "fault build"; return; }
  fi
  local p=1000003 B1=50000 d spec
  # flip: one bit of x;  square: x * 4 (a perfect square: invisible to a Jacobi check);  check: one bit of the check product;
  # two faults in different blocks
  for spec in 33333:flip 50001:square 40000:check "20000:square,60000:flip"; do
    d=$WORK/fault-${spec//[:,]/-}
    setup "$d" "$p" "$B1" 0
    PRPLL_PM1_FAULT=$spec run "$d" "$FAULT_PRPLL"
    local nFaults; nFaults=$(grep -c "TEST FAULT" "$d/out")
    local nFailed; nFailed=$(grep -c "Gerbicz check failed" "$d/out")
    if [[ "$nFaults" -ge 1 ]] && [[ "$nFailed" = "$nFaults" ]] && grep -q "\"gerbicz\":$nFaults}" "$d/results-0.txt" && oracle "$d" "$p" "$B1"; then
      pass "fault $spec: $nFaults injected, $nFailed caught ($(grep 'Gerbicz check failed' "$d/out" | grep -o 'failed at [0-9]*' | paste -sd,)), rolled back, residue equal"
    else
      fail "fault $spec (see $d)"
    fi
  done
}

jacobi_tests() {
  local p=400009 B1=20000 d
  # A good savefile passes the load check, and the final residue the end check
  d=$WORK/jacobi-good
  setup "$d" "$p" "$B1" 0
  run "$d" "$PRPLL" -iters 10000
  run "$d" "$PRPLL" -use PM1_JACOBI=1
  if [[ "$(grep -c 'Jacobi check at [0-9]*: OK' "$d/out")" = 2 ]] && oracle "$d" "$p" "$B1" && outcome "$d" "$p" "$B1" 0; then
    pass "Jacobi: good savefile and final residue pass ($(grep -o 'Jacobi check at [0-9]*: OK ([-0-9]*, expected [-0-9]*), [0-9.]* s' "$d/out" | paste -sd';'))"
  else
    fail "Jacobi on a good savefile (see $d)"
  fi

  # One bit of the newest savefile flipped (CRC rewritten, so only the residue can tell):  the Jacobi symbol is wrong about half the
  # time.  Rejected:  the previous savefile is used and the result is right.  Accepted:  nothing can see it (the Gerbicz check starts
  # from the loaded residue), and the residue is wrong.
  local nRej=0 nAcc=0 bit
  for bit in 5 100 1000 4321 77777 123456 200000 300001 350000 399999; do
    d=$WORK/jacobi-flip-$bit
    setup "$d" "$p" "$B1" 0
    run "$d" "$PRPLL" -iters 15000
    local f; f=$(lastSave "$d" "$p")
    "$FLIP" "$f" "$bit"
    run "$d" "$PRPLL" -use PM1_JACOBI=1
    if grep -q "P-1 savefile fails the Jacobi check" "$d/out"; then
      if oracle "$d" "$p" "$B1" && outcome "$d" "$p" "$B1" 0 && grep -q '"jacobi":1}' "$d/results-0.txt"; then
        nRej=$((nRej + 1)); pass "Jacobi: bit $bit of $(basename "$f") flipped: rejected, went back to $(grep -o 'OK *[0-9]* P-1 loaded' "$d/out" | tr -s ' ' | cut -d' ' -f2), residue equal"
      else
        fail "Jacobi: bit $bit rejected but the run is wrong (see $d)"
      fi
    elif grep -q "Jacobi check at [0-9]*: OK" "$d/out" && ! oracle "$d" "$p" "$B1"; then
      nAcc=$((nAcc + 1)); pass "Jacobi: bit $bit of $(basename "$f") flipped: symbol unchanged, accepted, final residue wrong as expected (Jacobi blind spot)"
    else
      fail "Jacobi: bit $bit (see $d)"
    fi
  done
  echo "Jacobi flips: $nRej rejected, $nAcc accepted"
  [[ "$nRej" -gt 0 ]] && [[ "$nAcc" -gt 0 ]] || fail "Jacobi flips: expected both outcomes"

  # The final residue replaced by 2^p-1 - x after its Gerbicz check (fault build):  the end check fails it and the block is redone;
  # without -use PM1_JACOBI=1 the wrong residue goes through
  if [[ ! -x "$FAULT_PRPLL" ]]; then
    make -s -C "$ROOT" -j8 BIN=build-pm1faults CPPFLAGS=-DPRPLL_PM1_FAULTS > /dev/null || { fail "fault build"; return; }
  fi
  local nBits; nBits=$("$ORACLE" "$p" "$B1" | grep -o 'E bits=[0-9]*' | cut -d= -f2)
  d=$WORK/jacobi-negate
  setup "$d" "$p" "$B1" 0
  PRPLL_PM1_FAULT=$nBits:negate run "$d" "$FAULT_PRPLL" -use PM1_JACOBI=1
  if grep -q "TEST FAULT negate" "$d/out" && grep -q "P-1 Jacobi check failed at $nBits" "$d/out" && oracle "$d" "$p" "$B1" \
     && outcome "$d" "$p" "$B1" 0 && grep -q '"gerbicz":0, "jacobi":1}' "$d/results-0.txt"; then
    pass "Jacobi: final residue negated after its Gerbicz check: caught, block redone, residue equal"
  else
    fail "Jacobi: negated final residue (see $d)"
  fi
  d=$WORK/jacobi-negate-off
  setup "$d" "$p" "$B1" 0
  PRPLL_PM1_FAULT=$nBits:negate run "$d" "$FAULT_PRPLL"
  if grep -q "TEST FAULT negate" "$d/out" && ! oracle "$d" "$p" "$B1"; then
    pass "Jacobi off: the negated final residue goes through (what the end check is for)"
  else
    fail "Jacobi off: negated final residue (see $d)"
  fi
}

case $WHAT in
  residue) residue_tests ;;
  resume) resume_tests ;;
  fault) fault_tests ;;
  jacobi) jacobi_tests ;;
  all) residue_tests; resume_tests; fault_tests; jacobi_tests ;;
esac

echo "$fails failure(s)"
[[ "$fails" = 0 ]]
