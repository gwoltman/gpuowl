#!/usr/bin/env bash
# prpll's P-1 against Mlucas (v21):  with B2 > B1 and stage 2 declined (-maxAlloc 100M), prpll's GCD finds nothing, it writes its
# residue (Mlucas savefile p<p>, and its stage 1 exponent as Mlucas's product file p<p>.s1_prod) and the "Pminus1=" line into an
# Mlucas directory, and Mlucas must take stage 1 as done and find the known factor in stage 2 (or none).  Each such case is also run
# by Mlucas from scratch, which must reach the same result, and whose own stage 1 residue must be prpll's residue raised to the
# ratio of the two stage 1 exponents (Mlucas's exponent is a multiple of prpll's; see docs/pm1.md).  Then -mlucasStage2, a line
# for the GCD alone, the B1 below Mlucas's minimum, leftover Mlucas files, and the choice between -mprimeDir and -mlucasDir.
#
#   MLUCAS=/path/to/Mlucas test/pm1/mlucas.sh
# Environment:  DEVICE (default 1), WORK (default test/pm1/work-mlucas), PRPLL (default build-release/prpll), MLUCAS_CPU (Mlucas's
# -cpu, default 0:3).  Mlucas is a test oracle only; prpll does not link it.  Mlucas needs an mlucas.cfg in its directory:  the
# script makes one with a short self-test at one FFT length, and Mlucas then self-tests any other length it needs by itself.

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DEVICE=${DEVICE:-1}
WORK=${WORK:-$HERE/work-mlucas}
PRPLL=${PRPLL:-$ROOT/build-release/prpll}
MLUCAS=${MLUCAS:?set MLUCAS to an Mlucas binary}
MLUCAS_CPU=${MLUCAS_CPU:-0:3}
FILES=(python3 "$HERE/mlucas_files.py")

fails=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }

mkdir -p "$WORK"
cfg=$WORK/mlucas.cfg
if [[ ! -s "$cfg" ]]; then
  mkdir -p "$WORK/cfg"
  (cd "$WORK/cfg" && "$MLUCAS" -fft 64 -iters 100 -cpu "$MLUCAS_CPU" < /dev/null > out 2>&1) && cp "$WORK/cfg/mlucas.cfg" "$cfg"
fi
[[ -s "$cfg" ]] || { echo "FAIL: Mlucas did not make an mlucas.cfg (see $WORK/cfg/out)"; exit 1; }
echo "Mlucas $(head -1 "$cfg") (the version line of its mlucas.cfg)"

mlucasDir() { mkdir -p "$1"; cp "$cfg" "$1/"; }
runMlucas() { (cd "$1" && timeout 3600 "$MLUCAS" -cpu "$MLUCAS_CPU" -maxalloc 10 < /dev/null > ml.out 2>&1); }

# outcome <Mlucas dir> <p>:  "F <factor> <stage>" or "NF", from its results.txt and its log
outcome() {
  local r; r=$(grep '"worktype":"P-1"' "$1/results.txt" 2>/dev/null | tail -1)
  if echo "$r" | grep -q '"status":"NF"'; then echo NF; return; fi
  local f; f=$(echo "$r" | grep -o '"factors":\["[0-9]*"' | grep -o '[0-9][0-9]*')
  local s; s=$(grep -o 'factor in Stage [12]' "$1/p$2.stat" 2>/dev/null | tail -1 | grep -o '[12]$')
  if [[ -n "$f" ]]; then echo "F $f $s"; else echo "none"; fi
}

# wanted <outcome> <expected factor or NF> <stage>:  the reported factor (a GCD, which can hold other factors too) is a multiple
wanted() {
  if [[ "$2" = NF ]]; then [[ "$1" = NF ]]; return; fi
  local f s; read -r _ f s <<< "$1"
  [[ -n "$f" ]] && [[ "$s" = "$3" ]] && python3 -c "import sys; sys.exit(int('$f') % $2 != 0)"
}

# own <dir> <p> <B1> <B2>:  Mlucas's own run from scratch;  sets own (its outcome)
own() {
  rm -rf "$1"; mlucasDir "$1"
  echo "Pminus1=1,2,$2,-1,$3,$4" > "$1/worktodo.txt"
  runMlucas "$1"
  own=$(outcome "$1" "$2")
}

# handoff <name> <p> <B1> <B2> <expected factor or NF> <stage>:  stage 2 declined on the GPU, Mlucas runs it from prpll's files
handoff() {
  local name=$1 p=$2 B1=$3 B2=$4 want=$5 stage=${6:-}
  local d=$WORK/$name
  rm -rf "$d"; mkdir -p "$d/run"; mlucasDir "$d/ml"
  echo "Pminus1=1,2,$p,-1,$B1,$B2" > "$d/run/worktodo-0.txt"
  (cd "$d/run" && "$PRPLL" -device "$DEVICE" -mlucasDir ../ml -maxAlloc 100M -log 10000 > out 2>&1)
  local ok=1 why=""
  grep -q "P-1 stage 1 found no factor" "$d/run/out" || { ok=0; why+=" no-stage1-GCD"; }
  grep -q "handed to Mlucas for stage 2: p$p in" "$d/run/out" || { ok=0; why+=" not-handed"; }
  [[ -e "$d/run/results-0.txt" ]] && { ok=0; why+=" prpll-result"; }                  # Mlucas reports this assignment, not prpll
  [[ -s "$d/run/worktodo-0.txt" ]] && { ok=0; why+=" line-kept"; }
  [[ "$(cat "$d/ml/worktodo.txt" 2>/dev/null)" = "Pminus1=1,2,$p,-1,$B1,$B2" ]] || { ok=0; why+=" mlucas-line"; }
  cp "$d/ml/p$p" "$d/handed-p$p" 2>/dev/null; cp "$d/ml/p$p.s1_prod" "$d/handed-p$p.s1_prod" 2>/dev/null
  "${FILES[@]}" save "$d/handed-p$p" "$p" > "$d/files" 2>&1 || { ok=0; why+=" savefile"; }
  "${FILES[@]}" ours "$d/handed-p$p.s1_prod" "$p" "$B1" >> "$d/files" 2>&1 || { ok=0; why+=" product"; }
  runMlucas "$d/ml"
  grep -q "p-1 stage 1 to b1 = $B1 already done -- proceeding to stage 2" "$d/ml/ml.out" || { ok=0; why+=" not-done"; }
  grep -q "S1 bit = " "$d/ml/p$p.stat" && { ok=0; why+=" ran-stage1"; }                 # not one iteration of stage 1 in Mlucas
  grep -q "Stage 1: No factor found" "$d/ml/p$p.stat" || { ok=0; why+=" mlucas-stage1-GCD"; }
  local got; got=$(outcome "$d/ml" "$p")
  wanted "$got" "$want" "$stage" || { ok=0; why+=" outcome"; }
  own "$d/own" "$p" "$B1" "$B2"
  [[ "$own" = "$got" ]] || { ok=0; why+=" own-outcome"; }
  "${FILES[@]}" relate "$d/handed-p$p" "$d/handed-p$p.s1_prod" "$d/own/p$p.s1" "$d/own/p$p.s1_prod" "$p" >> "$d/files" 2>&1 ||
    { ok=0; why+=" relation"; }
  if [[ "$ok" = 1 ]]; then
    pass "$name: p=$p B1=$B1 B2=$B2, prpll's GCD: no factor, handed to Mlucas; Mlucas from prpll's files: $got; Mlucas from scratch: $own; $(grep -o 'residue relation OK.*' "$d/files" | sed 's/ (Mlucas.s exponent.*//'))"
  else
    fail "$name:$why (see $d)"
  fi
}

# Factors of these Mersenne numbers, q = 2kp + 1 (checked by 2^p = 1 mod q); Mlucas runs no B1 below 10000:
#   M400417:   68012429119   = 2 * 3 * 28309 * 400417 + 1     (k = 3 * 28309)
#   M1000099:  1872347344039 = 2 * 3^2 * 104009 * 1000099 + 1 (k = 3^2 * 104009)
#   M400381:   671839319     = 2 * 839 * 400381 + 1           (k = 839:  stage 1 at any B1 >= 839)
handoff stage2-factor  400417  10000 30000  68012429119   2          # 28309 in (B1, B2]
handoff stage2-1M      1000099 50000 150000 1872347344039 2
handoff no-factor      400033  10000 30000  NF

# forced <name> <p> <B1> <B2> <expected factor> <stage>:  -mlucasStage2 with the GPU's memory free:  prpll runs stage 1 alone (no
# GCD, no stage 2) and Mlucas runs the GCD and stage 2
forced() {
  local name=$1 p=$2 B1=$3 B2=$4 want=$5 stage=$6
  local d=$WORK/$name
  rm -rf "$d"; mkdir -p "$d/run"; mlucasDir "$d/ml"
  echo "Pminus1=1,2,$p,-1,$B1,$B2" > "$d/run/worktodo-0.txt"
  (cd "$d/run" && "$PRPLL" -device "$DEVICE" -mlucasDir ../ml -mlucasStage2 -log 10000 > out 2>&1)
  cp -r "$d/ml" "$d/ml-handed"
  runMlucas "$d/ml"
  local ok=1 got; got=$(outcome "$d/ml" "$p")
  grep -q "handed to Mlucas for the GCD and stage 2 (-mlucasStage2)" "$d/run/out" || ok=0
  grep -q -E "stage 1 GCD|Stage 2 " "$d/run/out" && ok=0                  # neither the GPU's GCD nor its stage 2
  [[ -e "$d/run/results-0.txt" ]] && ok=0
  [[ -s "$d/run/worktodo-0.txt" ]] && ok=0
  grep -q "already done -- proceeding to stage 2" "$d/ml/ml.out" || ok=0
  wanted "$got" "$want" "$stage" || ok=0
  if [[ "$ok" = 1 ]]; then
    pass "$name: p=$p B1=$B1 B2=$B2 -mlucasStage2: prpll ran stage 1 only and handed $(grep -o 'Pminus1=.*' "$d/run/out" | head -1); Mlucas: $got"
  else
    fail "$name (see $d)"
  fi
}
forced  forced-stage2  400417  10000 30000  68012429119   2
forced  forced-stage1  400381  10000 20000  671839319     1           # 839 <= B1: the factor comes from Mlucas's stage 1 GCD

# The hand-off for the GCD alone (when the GPU GCD fails) asks for B2 = B1:  Mlucas runs the stage 1 GCD of a finished stage 1
# savefile whatever B2 is.  forced-stage1's files, with that line instead
d=$WORK/gcd-only
rm -rf "$d"; cp -r "$WORK/forced-stage1/ml-handed" "$d"
echo "Pminus1=1,2,400381,-1,10000,10000" > "$d/worktodo.txt"
runMlucas "$d"
got=$(outcome "$d" 400381)
if grep -q "already done" "$d/ml.out" && wanted "$got" 671839319 1 && ! grep -q '"B2"' "$d/results.txt"; then
  pass "a hand-off line with B2 = B1 makes Mlucas run the stage 1 GCD of prpll's residue alone: $got (no B2 in its result)"
else
  fail "B2 = B1 hand-off (see $d)"
fi

# -mlucasStage2 without -mlucasDir:  the line is not started
d=$WORK/forced-nodir
rm -rf "$d"; mkdir -p "$d"
echo "Pminus1=1,2,400417,-1,10000,30000" > "$d/worktodo-0.txt"
(cd "$d" && "$PRPLL" -device "$DEVICE" -mlucasStage2 -log 10000 > out 2>&1)
if grep -q -- "-mlucasStage2 needs -mlucasDir" "$d/out" && ! grep -q "P-1 stage 1 B1=" "$d/out"; then
  pass "-mlucasStage2 without -mlucasDir stops before stage 1"
else
  fail "-mlucasStage2 without -mlucasDir (see $d)"
fi

# B1 below Mlucas's minimum (Mlucas would raise it to 10000 and go on with stage 1 from our residue):  no hand-off, the result is
# for B1 alone, as without -mlucasDir
d=$WORK/low-b1
rm -rf "$d"; mkdir -p "$d/run"; mlucasDir "$d/ml"
echo "Pminus1=1,2,400033,-1,1000,2000" > "$d/run/worktodo-0.txt"
(cd "$d/run" && "$PRPLL" -device "$DEVICE" -mlucasDir ../ml -maxAlloc 100M -log 10000 > out 2>&1)
if grep -q "Mlucas can't take it: B1=1000" "$d/run/out" && [[ ! -e "$d/ml/p400033" ]] && [[ ! -s "$d/ml/worktodo.txt" ]] &&
   grep -q '"status":"NF"' "$d/run/results-0.txt" && ! grep -q '"B2"' "$d/run/results-0.txt"; then
  pass "B1=1000 with -mlucasDir: not handed to Mlucas, result for B1 alone"
else
  fail "B1 below Mlucas's minimum (see $d)"
fi

# Leftover Mlucas files of the exponent (here its log, whose "GCD" line would make Mlucas skip the stage 1 GCD) stop the hand-off;
# the worktodo line and the stage 1 savefile stay, and a second run after the files are moved hands off without a new stage 1
d=$WORK/stale
rm -rf "$d"; mkdir -p "$d/run"; mlucasDir "$d/ml"
echo "Pminus1=1,2,400033,-1,10000,20000" > "$d/run/worktodo-0.txt"
echo "Time for GCD = 00:00:00.050" > "$d/ml/p400033.stat"
(cd "$d/run" && "$PRPLL" -device "$DEVICE" -mlucasDir ../ml -maxAlloc 100M -log 10000 > out 2>&1)
ok=1
grep -q "not handed to Mlucas: '.*' already has p400033.stat" "$d/run/out" || ok=0
[[ -e "$d/ml/p400033" ]] && ok=0
grep -q "Pminus1=1,2,400033,-1,10000,20000" "$d/run/worktodo-0.txt" || ok=0
mv "$d/ml/p400033.stat" "$d/moved-p400033.stat"
(cd "$d/run" && "$PRPLL" -device "$DEVICE" -mlucasDir ../ml -maxAlloc 100M -log 10000 > out2 2>&1)
grep -q "P-1 stage 1 already done" "$d/run/out2" || ok=0
grep -q "stage 1 GCD skipped" "$d/run/out2" || ok=0
grep -q "handed to Mlucas for stage 2" "$d/run/out2" || ok=0
[[ -e "$d/ml/p400033" ]] || ok=0
if [[ "$ok" = 1 ]]; then
  pass "a leftover p400033.stat stops the hand-off and keeps the line; after moving it the next run hands off from the saved stage 1"
else
  fail "leftover Mlucas files (see $d)"
fi

# Both -mprimeDir and -mlucasDir:  an exponent within Prime95's range goes to mprime (mprime itself is not run here)
d=$WORK/both-dirs
rm -rf "$d"; mkdir -p "$d/run" "$d/mp"; mlucasDir "$d/ml"
echo "Pminus1=1,2,400033,-1,10000,20000" > "$d/run/worktodo-0.txt"
(cd "$d/run" && "$PRPLL" -device "$DEVICE" -mprimeDir ../mp -mlucasDir ../ml -maxAlloc 100M -log 10000 > out 2>&1)
if grep -q "handed to mprime for stage 2" "$d/run/out" && [[ -e "$d/mp/m0400033" ]] && [[ -s "$d/mp/worktodo.add" ]] &&
   [[ ! -e "$d/ml/p400033" ]] && [[ ! -e "$d/ml/worktodo.txt" ]]; then
  pass "with -mprimeDir and -mlucasDir, 400033 (within Prime95's range) goes to mprime"
else
  fail "both directories (see $d)"
fi

echo "$fails failure(s)"
[[ "$fails" = 0 ]]
