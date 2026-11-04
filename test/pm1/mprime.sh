#!/usr/bin/env bash
# prpll's P-1 against mprime:  with B2 <= B1, prpll runs stage 1 and its GCD alone and must report the factor that mprime finds from
# scratch with the same B1 (and that GMP finds, stage1_oracle);  with B2 > B1 and stage 2 declined (-maxAlloc 100M), prpll's GCD
# finds nothing, it writes its residue and the "Pminus1=" line into an mprime directory, and mprime runs stage 2 from it and must find
# the known factor.  (prpll's own stage 2 on the same lines:  stage2.sh.)  Also compares the
# stage 1 residue with GMP's and, when mprime logs its own ("stage 1 residue res64=...", see MPRIME_PRIME_TXT), with mprime's.
#
#   MPRIME=/path/to/mprime test/pm1/mprime.sh
# Environment:  DEVICE (default 1), WORK (default test/pm1/work-mprime), PRPLL (default build-release/prpll), MPRIME_PRIME_TXT
# (extra lines for mprime's prime.txt, e.g. a setting that makes it log its stage 1 res64, where it has one; without that log line
# the comparison with mprime's own residue is skipped and the rest of the checks still run).
# mprime and GMP are test oracles only; prpll links neither.

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DEVICE=${DEVICE:-1}
WORK=${WORK:-$HERE/work-mprime}
PRPLL=${PRPLL:-$ROOT/build-release/prpll}
MPRIME=${MPRIME:?set MPRIME to an mprime binary}
ORACLE=$HERE/stage1_oracle

fails=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }

make -s -C "$HERE" stage1_oracle || exit 1
mkdir -p "$WORK"

# An mprime directory that never talks to a server, exits when its work is done, keeps the given B2 and the save files
mprimeDir() {
  mkdir -p "$1"
  cat > "$1/prime.txt" <<EOF
AllowLowB1=1
StressTester=0
UsePrimenet=0
ExitWhenOutOfWork=1
Pminus1BestB2=0
KeepPminus1SaveFiles=1
${MPRIME_PRIME_TXT:-}
EOF
  printf 'WorkerThreads=1\nCoresPerTest=4\nMemory=2000\n' > "$1/local.txt"
}

runMprime() { (cd "$1" && timeout 1200 "$MPRIME" -d -w"$1" < /dev/null > mp.out 2>&1); }

# ownRun <dir> <p> <B1>:  mprime's own stage 1 (and GCD) from scratch, B2 = B1;  sets theirs (its res64) and theirF (its factor or NF)
ownRun() {
  mprimeDir "$1"
  echo "Pminus1=1,2,$2,-1,$3,$3" > "$1/worktodo.txt"
  runMprime "$1"
  theirs=$(grep -o -i 'stage 1 residue res64=[0-9a-f]*' "$1/mp.out" | cut -d= -f2 | tr 'A-F' 'a-f')
  theirF=$(grep -o '"factors":\["[0-9]*"' "$1/results.json.txt" 2>/dev/null | grep -o '[0-9]*' | tail -1)
  grep -q '"status":"NF"' "$1/results.json.txt" 2>/dev/null && theirF=NF
}

# alone <name> <p> <B1> <expected factor (a divisor of the gcd) or NF>:  B2 = 0, prpll on its own
alone() {
  local name=$1 p=$2 B1=$3 want=$4
  local d=$WORK/$name
  rm -rf "$d"; mkdir -p "$d/run"
  echo "Pminus1=1,2,$p,-1,$B1,0" > "$d/run/worktodo-0.txt"
  (cd "$d/run" && "$PRPLL" -device "$DEVICE" -log 10000 -noclean > out 2>&1)
  "$ORACLE" "$p" "$B1" "$(ls "$d/run/p1-$p/"*.p1 | sort | tail -1)" > "$d/run/oracle" 2>&1
  local ours gmp got; ours=$(grep -o 'res64 [0-9a-f]*)' "$d/run/oracle" | tr -d ')' | cut -d' ' -f2)
  gmp=$(grep '^gcd ' "$d/run/oracle" | cut -d' ' -f2); [ "$gmp" = 1 ] && gmp=NF
  local res; res=$(cat "$d/run/results-0.txt" 2>/dev/null)
  got=$(echo "$res" | grep -o '"factors":\["[0-9]*"' | grep -o '[0-9]*'); echo "$res" | grep -q '"status":"NF"' && got=NF
  ownRun "$d/own" "$p" "$B1"
  local ok=1
  grep -q "residue EQUAL" "$d/run/oracle" || ok=0
  if [ -n "$theirs" ]; then [ -n "$ours" ] && [ "$ours" = "$theirs" ] || ok=0; else echo "skipped: the residue comparison with mprime's own needs an mprime that can dump the stage 1 residue ($name)"; fi
  [ -n "$got" ] && [ "$got" = "$gmp" ] && [ "$got" = "$theirF" ] || ok=0
  if [ "$want" = NF ]; then [ "$got" = NF ] || ok=0; else python3 -c "import sys; sys.exit(int('$got') % $want != 0)" || ok=0; fi
  if [ $ok = 1 ]; then
    pass "$name: p=$p B1=$B1 B2=0, prpll alone: $got (GMP $gmp, mprime from scratch $theirF), residue $ours = GMP${theirs:+ = the one mprime logs}, GCD $(grep -o 'GCD: [0-9.]* s' "$d/run/out")"
  else
    fail "$name (see $d)"
  fi
}

# handoff <name> <p> <B1> <B2> <expected factor or NF> <expected stage of the factor>:  B2 > B1, stage 2 by mprime
handoff() {
  local name=$1 p=$2 B1=$3 B2=$4 want=$5 stage=${6:-}
  local d=$WORK/$name
  rm -rf "$d"; mkdir -p "$d/run"; mprimeDir "$d/mp"
  echo "Pminus1=1,2,$p,-1,$B1,$B2" > "$d/run/worktodo-0.txt"
  (cd "$d/run" && "$PRPLL" -device "$DEVICE" -mprimeDir ../mp -maxAlloc 100M -log 10000 > out 2>&1)
  local f; f=$(ls "$d/mp" | grep -v -E 'worktodo|prime.txt|local.txt')
  cp "$d/mp/$f" "$d/run/handed-$f"
  "$ORACLE" "$p" "$B1" "$d/run/handed-$f" > "$d/run/oracle" 2>&1
  local ours; ours=$(grep -o 'res64 [0-9a-f]*)' "$d/run/oracle" | tr -d ')' | cut -d' ' -f2)
  runMprime "$d/mp"
  ownRun "$d/own" "$p" "$B1"
  local res; res=$(cat "$d/mp/results.json.txt" 2>/dev/null)
  local ok=1
  grep -q "residue EQUAL" "$d/run/oracle" || ok=0
  grep -q '^gcd 1$' "$d/run/oracle" || ok=0                            # no stage 1 factor (else prpll reports it and hands nothing)
  grep -q "P-1 stage 1 found no factor" "$d/run/out" || ok=0
  [ -e "$d/run/results-0.txt" ] && ok=0                                 # mprime reports this assignment, not prpll
  if [ -n "$theirs" ]; then [ -n "$ours" ] && [ "$ours" = "$theirs" ] || ok=0; else echo "skipped: the residue comparison with mprime's own needs an mprime that can dump the stage 1 residue ($name)"; fi
  grep -i "error" "$d/mp/mp.out" | grep -v -i "error check" | grep -q . && ok=0
  if [ "$want" = NF ]; then
    echo "$res" | grep -q '"status":"NF"' || ok=0
  else
    # mprime reports the GCD, which can hold other factors of 2^p-1 as well:  it must be a multiple of the known one
    local got; got=$(echo "$res" | grep -o '"factors":\["[0-9]*"' | grep -o '[0-9]*' | tail -1)
    [ -n "$got" ] && python3 -c "import sys; sys.exit(int($got) % $want != 0)" || ok=0
    grep -q "P-1 found a factor in stage #$stage" "$d/mp/mp.out" || ok=0
  fi
  if [ $ok = 1 ]; then
    pass "$name: p=$p B1=$B1 B2=$B2, prpll's GCD: no factor, handed $(grep -o 'Pminus1=.*' "$d/run/out" | head -1), stage 1 residue $ours = GMP${theirs:+ = the one mprime logs}, mprime: $(grep -o 'P-1 found a factor in stage #[12].*\|M[0-9]* completed P-1.*' "$d/mp/mp.out" | head -1)$( [ "$want" != NF ] && echo " $got, a multiple of $want")"
  else
    fail "$name (see $d)"
  fi
}

# Factors of these Mersenne numbers, q = 2kp + 1 (found with test/pm1's search; checked by 2^p = 1 mod q):
#   M400381:   671839319     = 2 * 839 * 400381 + 1         (k = 839)
#   M1000253:  132975634327  = 2 * 3 * 22157 * 1000253 + 1   (k = 3 * 22157)
#   M1000099:  1872347344039 = 2 * 3^2 * 104009 * 1000099 + 1 (k = 3^2 * 104009)
alone   stage1-factor  400381  1000  671839319
alone   stage1-1M      1000253 50000 132975634327
alone   no-factor      400033  1000  NF
alone   stage1-B1-500  400381  500   NF                               # 839 > 500: stage 1 cannot find it
handoff stage2-factor  400381  500   1000   671839319     2           # 839 in (B1, B2]: mprime's stage 2 finds it
handoff stage2-1M      1000099 50000 150000 1872347344039 2

# Why a hand-off for the GCD alone (only when the GPU GCD fails) asks for B2 = B1 + 1:  with B2 <= the B1 in the savefile mprime
# reports "already tested" and runs no GCD
d=$WORK/b2-equals-b1
rm -rf "$d"; mkdir -p "$d/run"; mprimeDir "$d/mp"
echo "Pminus1=1,2,400033,-1,1000,2000" > "$d/run/worktodo-0.txt"
(cd "$d/run" && "$PRPLL" -device "$DEVICE" -mprimeDir ../mp -maxAlloc 100M -log 10000 > out 2>&1)
echo "Pminus1=1,2,400033,-1,1000,1000" > "$d/mp/worktodo.add"
runMprime "$d/mp"
if grep -q "already tested to B1=1000 and B2=1000" "$d/mp/mp.out" && ! grep -q "completed P-1" "$d/mp/mp.out"; then
  pass "a hand-off line with B2 = B1 makes mprime skip the GCD (\"already tested\"), hence B2 = B1 + 1 in the GCD fallback"
else
  fail "B2 = B1 behaviour (see $d)"
fi

echo "$fails failure(s)"
[ $fails = 0 ]
