#!/bin/bash
# Stage 2 v2 (table of odd n < M coprime to D, primes paired across D-sections) against GMP.
#   sets.sh plan                      host only:  check_stage2 plan + match over a grid of (B1, B2, D, M, windows)
#   sets.sh gpu <prpll> <device> [p] [quick]
#                                     every D of stage2Ds() with L = 2M/D = 1, 2.6 and 8 (1 and 2.6 for D > 2310), and the edge cases
#                                     (k = 0, B2 just above B1, primes of D just at B1, small windows, the planner's own choice), each
#                                     accumulator checked by "check_stage2 acc" (the product of the plan's items, and every prime covered).
#                                     <prpll> = a fault build (PRPLL_STAGE2_TEST).  quick:  one L per D and fewer edge cases.
#   sets.sh settings <prpll> <device> <p> <fft>...
#                                     each FFT with INPLACE 0/1/2 x L2_STRIPING 0/1/2 x MULTI_Q 0/1, D=210 M=1050
#   sets.sh planted <prpll> <device>  M400381's factor 671839319 (x of order q = 839 mod it after stage 1 with B1 = 500) found when q's
#                                     item is a pair across D-sections (D=30 M=120: 839 + 1021 at k=31; D=60 M=240: + 1201; D=420
#                                     M=1680: + 2521) and with the planner's own plan; the acc = GMP's product, and GMP's exact test of
#                                     the factor (check_stage2 factors) on the stage 1 residue
# Prints one PASS/FAIL line per case and a summary; exit status 0 when all passed.  The GPU runs share one kernel cache
# (work-sets/kernel-cache, -cache) unless KCACHE is set to another directory (or to '' for none).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
CHECK=$HERE/check_stage2
MODE=${1:-plan}
export KCACHE=${KCACHE-$HERE/work-sets/kernel-cache}
npass=0; nfail=0
pass() { echo "PASS $*"; npass=$((npass + 1)); }
fail() { echo "FAIL $*"; nfail=$((nfail + 1)); }

Ds=(30 60 90 120 150 180 210 330 420 462 630 660 840 990 1050 1260 1470 1680 1890 2310 2520 3150 4620 6930 9240)

plan_tests() {
  # (B1 B2 D M kW):  L = 1 (the old plan), fractional and large L, small windows (kW 1, 3, 7), k = 0 (B1 < D/2), tiny ranges
  local cases=(
    "1000 200000 210 105 0" "1000 200000 2310 1155 0" "13 1000 210 105 0" "1000 200000 30 300 0" "5 5000 30 15 0" "5 20000 30 451 0"
    "7 20000 210 2000 0" "1000 300000 2310 9001 5" "1000 100000 420 3000 7" "1000 100000 420 3000 1" "11 30000 4620 20000 0"
    "100 120 30 15 0" "1000 1001 210 105 0" "1000 1010 210 1050 0" "13 50000 9240 12000 0" "13 50000 6930 4000 3" "17 40000 60 700 0"
    "30000 400000 1050 7000 0" "30000 400000 1050 7000 2" "11 100000 462 1500 0" "11 100000 330 990 4" "11 100000 990 2000 0"
  )
  for c in "${cases[@]}"; do
    read -r B1 B2 D M KW <<< "$c"
    local r1 r2
    r1=$("$CHECK" plan "$B1" "$B2" "$D" "$M" "$KW")
    r2=$("$CHECK" match "$B1" "$B2" "$D" "$M" "$KW")
    if [[ $r1 == OK* && $r2 == OK* ]]; then pass "$r1 | ${r2#OK match B1=* M=*: }"; else fail "$r1 | $r2"; fi
  done
}

# gpu1 <name> <B1> <B2> <D> <M> [kWindow] [fft] [use]:  run_stage2.sh with this plan, acc = GMP and every prime covered
gpu1() {
  local name=$1 B1=$2 B2=$3 D=$4 M=$5 KW=${6:-} FFT=${7:--} USE=${8:--} r
  if [[ -n "$KW" ]] && [[ "$KW" != 0 ]]; then
    r=$(PRPLL_STAGE2_KWINDOW=$KW "$HERE/run_stage2.sh" "$PRPLL" "$WORK" "$DEVICE" "$name" "$P" "$B1" "$B2" "$D" "$FFT" "$USE" random:7 "$M")
  else
    r=$("$HERE/run_stage2.sh" "$PRPLL" "$WORK" "$DEVICE" "$name" "$P" "$B1" "$B2" "$D" "$FFT" "$USE" random:7 "$M")
  fi
  if [[ $r == *": OK acc"* ]]; then pass "$r"; else fail "$r"; fi
}

gpu_tests() {
  for D in "${Ds[@]}"; do
    local B2=$((D * 40 > 30000 ? D * 40 : 30000)) Ls
    if [[ "$QUICK" = 1 ]]; then Ls="2.6"; elif [[ "$D" -le 2310 ]]; then Ls="1 2.6 8"; else Ls="1 2.6"; fi
    for L in $Ls; do
      local M; M=$(awk -v d="$D" -v l="$L" 'BEGIN { m = int(l * d / 2); if (m < d / 2) m = d / 2; print m }')
      gpu1 "D$D-L$L" 13 "$B2" "$D" "$M"
    done
  done
  # B2 just above B1 (one prime 1009; no prime at all), primes of D at B1 (B1 = 5 with D = 30, 7 is the next prime; B1 = 7 with D = 210,
  # 11 next), small windows (no pair across them), the planner's own (D, M)
  gpu1 b2-one-prime 1000 1010 210 1050
  gpu1 b2-no-prime 1000 1001 210 105
  gpu1 b1-5-D30 5 20000 30 451
  gpu1 b1-7-D210 7 20000 210 2000
  gpu1 windows-3 1000 100000 420 3000 3
  gpu1 auto-30000 1000 30000 0 0
  [[ "$QUICK" = 1 ]] && return
  gpu1 windows-1 1000 60000 210 1050 1
  gpu1 auto-300000 30000 300000 0 0
  # A D with a prime above B1 must decline (11 > B1 = 7 for D = 2310)
  local d="$WORK/decline-D2310"
  mkdir -p "$d" && "$CHECK" random "$P" 7 "$d/x.bin" && (cd "$d" && rm -rf p2-* && PRPLL_STAGE2_TEST=$P,7,20000,x.bin,acc.bin,2310,1155 "$PRPLL" -device "$DEVICE" > log.txt 2>&1)
  if grep -q "Stage 2 declined: .*D=2310 has a prime factor above B1" "$d/log.txt" && [[ ! -f "$d/acc.bin" ]]; then
    pass "decline-D2310: B1=7 with D=2310 declined ($(grep -o 'D=2310 has a prime factor above B1' "$d/log.txt"))"
  else
    fail "decline-D2310: (see $d)"
  fi
}

settings_tests() {
  for fft in "$@"; do
    for ip in 0 1 2; do
      for l2 in 0 1 2; do
        for mq in 0 1; do
          gpu1 "$fft-ip$ip-l2$l2-mq$mq" 1000 30000 210 1050 0 "$fft" "INPLACE=$ip,L2_STRIPING=$l2,MULTI_Q=$mq"
        done
      done
    done
  done
}

planted_tests() {
  P=400381
  for DM in "30 120" "60 240" "420 1680" "0 0"; do
    read -r D M <<< "$DM"
    local name="planted-D$D-M$M" r f
    r=$("$HERE/run_stage2.sh" "$PRPLL" "$WORK" "$DEVICE" "$name" "$P" 500 200000 "$D" - - stage1 "$M")
    f=$("$CHECK" factors "$P" 500 200000 "$WORK/$name/x.bin" "$WORK/$name/acc.bin" 671839319)
    if [[ $r == *": OK acc"* && $f == OK* && $f == *found* && $f != *"not found"* ]]; then pass "$r | $f"; else fail "$r | $f"; fi
  done
}

case $MODE in
  plan) plan_tests ;;
  gpu) PRPLL=$2; DEVICE=$3; P=${4:-400009}; QUICK=${5:-0}; WORK=$HERE/work-sets; mkdir -p "$WORK"; gpu_tests ;;
  settings) PRPLL=$2; DEVICE=$3; P=$4; shift 4; WORK=$HERE/work-sets; mkdir -p "$WORK"; settings_tests "$@" ;;
  planted) PRPLL=$2; DEVICE=$3; WORK=$HERE/work-sets; mkdir -p "$WORK"; planted_tests ;;
  *) echo "usage: see the top of $0"; exit 2 ;;
esac
echo "sets.sh $MODE: $npass passed, $nfail failed"
[[ "$nfail" = 0 ]]
