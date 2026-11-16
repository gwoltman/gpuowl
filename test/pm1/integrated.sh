#!/usr/bin/env bash
# End-to-end tests of the P-1 inside a PRP test (here with the bounds of the hidden -use PRPPM1_B1[,PRPPM1_B2] keys, src/Pm1Prp.cpp), with GMP (stage1_oracle) as the oracle for every accumulator
# savefile (its residue must be 3^(E mod 2^next), its prp64 the res64 of 3^(2^k)) and for the factor.
#
#   test/pm1/integrated.sh [factor|nf|carry|resume|fault|onoff|crash|stage2|stage2fail|watch|handoff|all]      (default all)
# Environment:  DEVICE (default 1, the CPU OpenCL device here), WORK (default test/pm1/work-integrated), PRPLL (default
#               build-release/prpll), FAULT_PRPLL (default build-pm1faults/prpll, built with -DPRPLL_PM1_FAULTS if missing).

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DEVICE=${DEVICE:-1}
WORK=${WORK:-$HERE/work-integrated}
PRPLL=${PRPLL:-$ROOT/build-release/prpll}
FAULT_PRPLL=${FAULT_PRPLL:-$ROOT/build-pm1faults/prpll}
ORACLE=$HERE/stage1_oracle
WHAT=${1:-all}

fails=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }

make -s -C "$HERE" stage1_oracle || exit 1
[[ -x "$PRPLL" ]] || { echo "no $PRPLL (run make)"; exit 1; }
if [[ ! -x "$FAULT_PRPLL" ]]; then make -s -C "$ROOT" BIN=build-pm1faults CPPFLAGS=-DPRPLL_PM1_FAULTS || exit 1; fi
mkdir -p "$WORK"

# setup <dir> <p>:  a fresh run directory with one PRP line
setup() {
  rm -rf "$1"; mkdir -p "$1"
  echo "PRP=0123456789ABCDEF0123456789ABCDEF,1,2,$2,-1,70,0" > "$1/worktodo-0.txt"
}

# run <dir> <prpll> <args...>:  runs prpll in <dir>, appending to <dir>/out
run() {
  local d=$1 bin=$2; shift 2
  (cd "$d" && "$bin" -device "$DEVICE" -log 1000 -noclean "$@" >> out 2>&1)
}

# accOK <dir> <p> <B1>:  every accumulator savefile agrees with GMP;  the GMP gcd is then in <dir>/gcd
accOK() {
  local files; mapfile -t files < <(ls "$1/p1prp-$2/"*.p1prp 2>/dev/null)
  [[ ${#files[@]} -gt 0 ]] || return 1
  "$ORACLE" "$2" "$3" "${files[@]}" > "$1/oracle" 2>&1 || return 1
  grep '^gcd ' "$1/oracle" | cut -d' ' -f2 > "$1/gcd"
}

# resultOK <dir> <p> <B1> <B2> <factor|NF>:  exactly one P-1 result line, integrated, with this outcome
resultOK() {
  local r; r=$(grep '"worktype":"P-1"' "$1/results-0.txt" 2>/dev/null)
  [[ "$(echo "$r" | grep -c .)" = 1 ]] || return 1
  echo "$r" | grep -q "\"exponent\":$2, \"worktype\":\"P-1\", \"b1\":$3, " || return 1
  echo "$r" | grep -q '"integrated":true' || return 1
  if [[ "$4" != 0 ]]; then echo "$r" | grep -q "\"b2\":$4, " || return 1; else ! echo "$r" | grep -q '"b2"' || return 1; fi
  if [[ "$5" = NF ]]; then
    echo "$r" | grep -q '"status":"NF"' && ! echo "$r" | grep -q factors
  else
    echo "$r" | grep -q "\"status\":\"F\".*\"factors\":\[\"$5\"\]"
  fi
}

# factorEnd <dir>:  the PRP test stopped for the factor:  no PRP result, the worktodo line is gone
factorEnd() { ! grep -q '"worktype":"PRP-3"' "$1/results-0.txt" && [[ ! -s "$1/worktodo-0.txt" ]] && grep -q "the PRP test is not needed" "$1/out"; }

# prpLines <dir> <p>:  the PRP residues logged, "k res64" per line (log and check lines)
prpLines() { grep -oE " $2 (OK|EE)? *[0-9]+ [0-9a-f]{16} " "$1/out" | awk '{print $(NF-1), $NF}' | sort -un; }

# sameResidues <dirA> <dirB> <p>:  at every iteration both logged, the same PRP res64 (and at least 2 in common)
sameResidues() {
  local n; n=$(join <(prpLines "$1" "$3") <(prpLines "$2" "$3") | awk '$2 != $3 {bad=1} END {print bad ? -1 : NR}')
  [[ "$n" -ge 2 ]]
}

REF=$WORK/ref-400033
reference() {
  # A PRP without P-1 (its line has tests saved 0):  the residues the PRP of the other tests must have
  if ! grep -q " 40000 " "$REF/out" 2>/dev/null; then
    setup "$REF" 400033
    run "$REF" "$PRPLL" -iters 40000
  fi
}

factor_tests() {
  local d=$WORK/factor
  setup "$d" 400381
  run "$d" "$PRPLL" -use PRPPM1_B1=1000
  if accOK "$d" 400381 1000 && [[ "$(cat "$d/gcd")" = 671839319 ]] && resultOK "$d" 400381 1000 0 671839319 && factorEnd "$d"; then
    pass "factor M400381 B1=1000: 671839319 = GMP gcd, PRP stopped ($(grep -o 'L=[0-9]*' "$d/out" | head -1), $(grep -o "res64 [0-9a-f]*" "$d/out" | head -1))"
  else
    fail "factor M400381 B1=1000 (see $d)"
  fi
}

nf_tests() {
  reference
  local d=$WORK/nf
  setup "$d" 400033
  run "$d" "$PRPLL" -use PRPPM1_B1=1000 -iters 40000
  if accOK "$d" 400033 1000 && [[ "$(cat "$d/gcd")" = 1 ]] && resultOK "$d" 400033 1000 0 NF && sameResidues "$d" "$REF" 400033 \
     && [[ -s "$d/worktodo-0.txt" ]]; then
    pass "no factor M400033 B1=1000: NF result, the PRP goes on with the residues of a PRP without P-1 ($(join <(prpLines "$d" 400033) <(prpLines "$REF" 400033) | wc -l) compared)"
  else
    fail "no factor M400033 (see $d)"
  fi
}

carry_tests() {
  # Without the long carry (at least 10 bits per word:  here FP64 256:1:256 at 2M):  the steps take u_k from carryFused's output
  # (LEAD_WIDTH) where they can, and the PRP residues must not change
  local p=2000003 B1=1000    # PoCL runs carryFused at 25-100 ms per iteration: a short test (stage 1 done at the check of 2000)
  local d=$WORK/carry r=$WORK/carry-ref
  setup "$r" "$p"
  run "$r" "$PRPLL" -fft 256:1:256 -iters 3000
  setup "$d" "$p"
  run "$d" "$PRPLL" -fft 256:1:256 -use PRPPM1_B1="$B1" -iters 3000
  local g; accOK "$d" "$p" "$B1" && g=$(cat "$d/gcd")
  if [[ -n "${g:-}" ]] && ! grep -q "Using long carry" "$d/out" && sameResidues "$d" "$r" "$p" \
     && { { [[ "$g" = 1 ]] && resultOK "$d" "$p" "$B1" 0 NF; } || { [[ "$g" != 1 ]] && resultOK "$d" "$p" "$B1" 0 "$g" && factorEnd "$d"; }; }; then
    pass "carryFused M$p B1=$B1 -fft 256:1:256: accumulator = GMP, gcd $g, PRP residues = the PRP alone"
  else
    fail "carryFused M$p B1=$B1 (see $d)"
  fi
}

# finish <dir> <bin> <args...>:  runs to the end (the factor of M400381 at B1=20000)
resume_tests() {
  # B1=20000:  28.9K bits, the first check at 2000, the forced last one after the last window (checkStep 40000 here).  L=7 puts the
  # grid off every block boundary, so every stop is mid-window.
  local d=$WORK/resume
  setup "$d" 400381
  PRPLL_PM1_PRP_L=7 run "$d" "$FAULT_PRPLL" -use PRPPM1_B1=20000 -iters 3000
  PRPLL_PM1_PRP_L=5 run "$d" "$FAULT_PRPLL" -use PRPPM1_B1=20000 -iters 4000
  (cd "$d" && { PRPLL_PM1_PRP_L=6 "$FAULT_PRPLL" -device "$DEVICE" -log 1000 -noclean -use PRPPM1_B1=20000 >> out 2>&1 & echo $! > pid; })
  until grep -q " 12000 " "$d/gpuowl-0.log" 2>/dev/null; do sleep 0.05; done     # (stdout is buffered: the log file; a fast GPU passes 12000 in under a second)
  kill -INT "$(cat "$d/pid")"
  while kill -0 "$(cat "$d/pid")" 2>/dev/null; do sleep 1; done
  run "$d" "$PRPLL" -use PRPPM1_B1=20000
  local loads; loads=$(grep -c "accumulator loaded at" "$d/out")
  # shellcheck disable=SC2012  # the names are prpll's own savefile names: no spaces or newlines
  if accOK "$d" 400381 20000 && [[ "$(cat "$d/gcd")" = 671839319 ]] && resultOK "$d" 400381 20000 0 671839319 && factorEnd "$d" \
     && [[ "$loads" = 3 ]] && [[ "$(ls "$d/p1prp-400381/"*.p1prp | wc -l)" -ge 4 ]]; then
    pass "resume M400381 B1=20000: -iters 3000 (L=7), -iters 4000 (L=5), SIGINT at ~12000 (L=6), then to the end: $loads loads, every savefile = GMP"
  else
    fail "resume (see $d)"
  fi
}

fault_tests() {
  # prp:  a PRP error at an accumulator step (the residue used by both);  bucket:  one bucket;  fold:  the fold's result.  Every one
  # must be caught and rolled back, and the result must still be GMP's.  Grid from 1 with L=6:  steps at 1 + 6m.
  local kind k
  for c in "prp 1201" "bucket 1201" "bucket 6001" "fold 2000" "prp 28351"; do
    read -r kind k <<< "$c"
    local d=$WORK/fault-$kind-$k
    setup "$d" 400381
    PRPLL_PM1_PRP_L=6 PRPLL_PM1_PRP_FAULT="$k:$kind" run "$d" "$FAULT_PRPLL" -use PRPPM1_B1=20000
    local caught
    case $kind in
      prp) caught=$(grep -cE " EE +[0-9]+ " "$d/out") ;;
      *)   caught=$(grep -c "check fails at\|running products differ" "$d/out") ;;
    esac
    if grep -q "TEST FAULT $kind" "$d/out" && [[ "$caught" -ge 1 ]] && accOK "$d" 400381 20000 && [[ "$(cat "$d/gcd")" = 671839319 ]] \
       && resultOK "$d" 400381 20000 0 671839319 && factorEnd "$d"; then
      pass "fault $kind at $k: caught ($(grep -oE 'accumulator errors[^)]*|[0-9]+ accumulator errors' "$d/out" | tail -1)), rolled back, factor and savefiles = GMP"
    else
      fail "fault $kind at $k (see $d)"
    fi
  done
}

onoff_tests() {
  reference
  # On mid-run:  a PRP started without the flag can't do P-1;  it goes on unchanged
  local d=$WORK/on-late
  setup "$d" 400033
  run "$d" "$PRPLL" -iters 4000
  run "$d" "$PRPLL" -use PRPPM1_B1=1000 -iters 6000
  # shellcheck disable=SC2010,SC2143  # the names are prpll's own savefile names: no spaces or newlines
  if grep -q "was started without P-1" "$d/out" && [[ ! -e "$d/results-0.txt" ]] && sameResidues "$d" "$REF" 400033 \
     && [[ -z "$(ls "$d/p1prp-400033/" 2>/dev/null | grep p1prp)" ]]; then
    pass "flag on at 4000: P-1 not possible (logged), no savefile, PRP residues unchanged"
  else
    fail "flag on mid-run (see $d)"
  fi
  # Off mid-run, then on again:  the PRP goes back to the accumulator's last savefile (within 2 check intervals)
  d=$WORK/off-on
  setup "$d" 400381
  run "$d" "$PRPLL" -use PRPPM1_B1=20000 -iters 4000
  run "$d" "$PRPLL" -use PRPPM1=0 -iters 4000
  local offOK=0
  # shellcheck disable=SC2012  # the names are prpll's own savefile names: no spaces or newlines
  [[ "$(ls "$d/p1prp-400381/"*.p1prp | wc -l)" = 3 ]] && offOK=1      # 0, 2000 and 4000, nothing from the run without the flag
  run "$d" "$PRPLL" -use PRPPM1_B1=20000
  if [[ "$offOK" = 1 ]] && grep -q "goes back there from 8000" "$d/out" && accOK "$d" 400381 20000 && resultOK "$d" 400381 20000 0 671839319 \
     && factorEnd "$d"; then
    pass "flag off from 4000 to 8000, then on: the PRP goes back to 4000, factor = GMP"
  else
    fail "flag off mid-run (see $d)"
  fi
}

crash_tests() {
  # kill -9 after unverified PRP savefiles (every 1000):  the PRP loads the unverified one and goes back to the accumulator's
  local d=$WORK/crash
  setup "$d" 400381
  (cd "$d" && { "$PRPLL" -device "$DEVICE" -log 1000 -noclean -use PRPPM1_B1=20000 >> out 2>&1 & echo $! > pid; })
  until grep -q " 9000 " "$d/gpuowl-0.log" 2>/dev/null; do sleep 0.05; done
  kill -KILL "$(cat "$d/pid")"
  while kill -0 "$(cat "$d/pid")" 2>/dev/null; do sleep 1; done
  run "$d" "$PRPLL" -use PRPPM1_B1=20000
  if grep -qE "goes back there from (8|9|10)000" "$d/out" && accOK "$d" 400381 20000 && resultOK "$d" 400381 20000 0 671839319 && factorEnd "$d"; then
    pass "kill -9 at ~9000: $(grep -o 'last savefile is at iteration [0-9]*: the PRP test goes back there from [0-9]*' "$d/out"), factor = GMP"
  else
    fail "crash (see $d)"
  fi
}

stage2_tests() {
  # B1=500 misses 671839319 (stage 1 at B1=500 gives gcd 1, GMP), B2=200000 finds it in stage 2;  a stop in stage 2 resumes there
  local d=$WORK/stage2
  setup "$d" 400381
  run "$d" "$PRPLL" -use PRPPM1_B1=500,PRPPM1_B2=200000
  if accOK "$d" 400381 500 && [[ "$(cat "$d/gcd")" = 1 ]] && resultOK "$d" 400381 500 200000 671839319 && factorEnd "$d"; then
    pass "stage 2 M400381 B1=500 B2=200000: stage 1 gcd 1 (GMP), stage 2 found 671839319, PRP stopped"
  else
    fail "stage 2 (see $d)"
  fi
  d=$WORK/stage2-stop
  setup "$d" 400381
  run "$d" "$PRPLL" -use PRPPM1_B1=500,PRPPM1_B2=200000 -iters 3000
  local stopped=0
  grep -q "stage 2 resumes\|Stage 2 B1=500" "$d/out" && [[ ! -e "$d/results-0.txt" ]] && [[ -s "$d/worktodo-0.txt" ]] && stopped=1
  run "$d" "$PRPLL" -use PRPPM1_B1=500,PRPPM1_B2=200000
  if [[ "$stopped" = 1 ]] && grep -q "stage 1 is done (at iteration [0-9]*), stage 2 resumes" "$d/out" && grep -q "Stage 2 resuming" "$d/out" \
     && resultOK "$d" 400381 500 200000 671839319 && factorEnd "$d"; then
    pass "stage 2 stopped by -iters 3000 and resumed: factor 671839319"
  else
    fail "stage 2 stop/resume (see $d)"
  fi
}

stage2fail_tests() {
  # Stage 2 that can't run (every FFT "fails to build", fault build):  the result for B1 alone, and the PRP test goes on;  a restart
  # does not try stage 2 again
  local d=$WORK/stage2-fail
  setup "$d" 400381
  PRPLL_STAGE2_BUILD_FAULT=99 run "$d" "$FAULT_PRPLL" -use PRPPM1_B1=500,PRPPM1_B2=200000 -iters 4000
  PRPLL_STAGE2_BUILD_FAULT=99 run "$d" "$FAULT_PRPLL" -use PRPPM1_B1=500,PRPPM1_B2=200000 -iters 2000
  if grep -q "P-1 stage 2 (B2=200000) failed" "$d/out" && resultOK "$d" 400381 500 0 NF && [[ -s "$d/worktodo-0.txt" ]] \
     && [[ "$(grep -c "stage 2 to B2" "$d/out")" = 1 ]] && grep -qE " OK +6000 [0-9a-f]{16}" "$d/out"; then
    pass "stage 2 failing: result for B1=500 only, the PRP went on to 6000, no second stage 2 on restart"
  else
    fail "stage 2 failing (see $d)"
  fi
}

watch_tests() {
  # The watch of mprime's results file (-mprimeDir), without mprime:  a PRP line that asks for no P-1 stops at the start on a factor line
  # for its exponent, ignores a "factor" that does not divide 2^p - 1 and lines of other exponents, and reads the file named in prime.txt
  local d=$WORK/watch-factor
  setup "$d" 400381; mkdir -p "$d/mp"
  echo '{"status":"F", "exponent":400381, "worktype":"P-1", "factors":["671839319"], "b1":500, "b2":1000, "fft-length":20480}' > "$d/mp/results.json.txt"
  run "$d" "$PRPLL" -mprimeDir mp -iters 2000
  if grep -q "400381 has the factor 671839319" "$d/out" && grep -q "the PRP test is not needed" "$d/out" && [[ ! -s "$d/worktodo-0.txt" ]] \
     && [[ ! -e "$d/results-0.txt" ]]; then
    pass "watch: a P-1 factor in mprime's results ends the PRP test (line removed, no PRP result)"
  else
    fail "watch: factor (see $d)"
  fi

  d=$WORK/watch-bogus
  setup "$d" 400381; mkdir -p "$d/mp"
  { echo '{"status":"F", "exponent":400381, "worktype":"P-1", "factors":["671839321"], "b1":500, "b2":1000}'
    echo '{"status":"F", "exponent":400033, "worktype":"P-1", "factors":["671839319"], "b1":500, "b2":1000}'
    echo '{"status":"NF", "exponent":400381, "worktype":"P-1", "b1":500, "b2":1000, "fft-length":20480}'
    printf '{"status":"F", "exponent":400381, "worktype":"P-1", "factors":["6718393'; } > "$d/mp/results.json.txt"
  run "$d" "$PRPLL" -mprimeDir mp -iters 2000
  if grep -q "gives 671839321 as a factor of M400381, which it is not: ignored" "$d/out" && ! grep -q "has the factor" "$d/out" \
     && grep -q "P-1 of 400381 found no factor, B1=500 B2=1000" "$d/out" && grep -qE " OK +2000 [0-9a-f]{16}" "$d/out" \
     && [[ -s "$d/worktodo-0.txt" ]]; then
    pass "watch: a wrong factor, another exponent's line and an unfinished line are ignored; the no-factor result is logged; the PRP goes on"
  else
    fail "watch: bogus lines (see $d)"
  fi
  # The rest of the unfinished line arrives:  the next check reads it
  echo '19"], "b1":500, "b2":1000}' >> "$d/mp/results.json.txt"
  run "$d" "$PRPLL" -mprimeDir mp -iters 2000
  if grep -q "400381 has the factor 671839319" "$d/out" && [[ ! -s "$d/worktodo-0.txt" ]]; then
    pass "watch: the factor line completed later is found on the next run"
  else
    fail "watch: completed line (see $d)"
  fi

  d=$WORK/watch-renamed
  setup "$d" 400381; mkdir -p "$d/mp"
  printf 'UsePrimenet=0\r\nresults.json.txt=other.json\r\n' > "$d/mp/prime.txt"
  echo '{"status":"F", "exponent":400381, "worktype":"ECM", "factors":["671839319"], "b1":50000}' > "$d/mp/other.json"
  run "$d" "$PRPLL" -mprimeDir mp -iters 2000
  if grep -q "other.json'): 400381 has the factor 671839319" "$d/out" && [[ ! -s "$d/worktodo-0.txt" ]]; then
    pass "watch: the results file named in prime.txt is read, and a factor from another work type counts"
  else
    fail "watch: renamed results file (see $d)"
  fi

  # A factor that appears while the test runs is found at the next verified check
  d=$WORK/watch-during
  setup "$d" 400381; mkdir -p "$d/mp"
  run "$d" "$PRPLL" -mprimeDir mp &
  local pid=$!
  for _ in $(seq 60); do grep -qE " OK +[0-9]+000 [0-9a-f]{16}" "$d/out" 2>/dev/null && break; sleep 1; done
  echo '{"status":"F", "exponent":400381, "worktype":"P-1", "factors":["671839319"], "b1":500, "b2":1000}' > "$d/mp/results.json.txt"
  wait "$pid"
  if grep -qE " OK +[0-9]+000 [0-9a-f]{16}" "$d/out" && grep -q "400381 has the factor 671839319" "$d/out" && [[ ! -s "$d/worktodo-0.txt" ]] \
     && ! grep -q '"worktype":"PRP' "$d/results-0.txt" 2>/dev/null; then
    pass "watch: a factor written while the test runs ends it at the next check (iteration $(grep -oE " OK +[0-9]+ " "$d/out" | tail -1 | tr -dc 0-9) of 400381)"
  else
    fail "watch: factor during the test (see $d)"
  fi
}

handoff_tests() {
  # -mprimeStage2 in the PRP test, without mprime:  after stage 1 and its GCD on the GPU the residue and the Pminus1 line are in the
  # mprime directory, no P-1 result is written here, stage 2 does not run on the GPU, and the PRP test goes on;  a restart does not
  # hand off again
  local d=$WORK/handoff
  setup "$d" 400381; mkdir -p "$d/mp"
  run "$d" "$PRPLL" -use PRPPM1_B1=500,PRPPM1_B2=1000 -mprimeDir mp -mprimeStage2 -iters 3000
  run "$d" "$PRPLL" -use PRPPM1_B1=500,PRPPM1_B2=1000 -mprimeDir mp -mprimeStage2 -iters 2000
  if grep -q "stage 1 residue handed to mprime for stage 2 (-mprimeStage2)" "$d/out" && grep -q "P-1 stage 1 GCD:" "$d/out" \
     && [[ "$(cat "$d/mp/worktodo.add")" = "Pminus1=0123456789ABCDEF0123456789ABCDEF,1,2,400381,-1,500,1000" ]] \
     && [[ -s "$d/mp/m0400381" ]] && ! grep -q "stage 2 to B2" "$d/out" && ! grep -q '"worktype":"P-1"' "$d/results-0.txt" 2>/dev/null \
     && grep -q "stage 2 (B2=1000) was handed to mprime" "$d/out" && grep -qE " OK +5000 [0-9a-f]{16}" "$d/out" && [[ -s "$d/worktodo-0.txt" ]]; then
    pass "hand-off: stage 2 of the P-1 in the PRP test goes to the mprime directory once, with the GPU GCD done, and the PRP went on to 5000"
  else
    fail "hand-off (see $d)"
  fi

  # Without -mprimeDir the option changes nothing but a log line:  stage 2 runs on the GPU and finds the factor
  d=$WORK/handoff-nodir
  setup "$d" 400381
  run "$d" "$PRPLL" -use PRPPM1_B1=500,PRPPM1_B2=1000 -mprimeStage2 -iters 3000
  if grep -q -- "-mprimeStage2 needs -mprimeDir" "$d/out" && resultOK "$d" 400381 500 1000 671839319; then
    pass "hand-off without -mprimeDir: logged, and stage 2 ran on the GPU (factor 671839319)"
  else
    fail "hand-off without -mprimeDir (see $d)"
  fi
}

case $WHAT in
  factor) factor_tests ;;
  nf) nf_tests ;;
  carry) carry_tests ;;
  resume) resume_tests ;;
  fault) fault_tests ;;
  onoff) onoff_tests ;;
  crash) crash_tests ;;
  stage2) stage2_tests ;;
  stage2fail) stage2fail_tests ;;
  watch) watch_tests ;;
  handoff) handoff_tests ;;
  all) factor_tests; nf_tests; carry_tests; resume_tests; fault_tests; onoff_tests; crash_tests; stage2_tests; stage2fail_tests; watch_tests; handoff_tests ;;
  *) echo "unknown test set '$WHAT'"; exit 2 ;;
esac
echo "$fails failure(s)"
[[ "$fails" = 0 ]]
