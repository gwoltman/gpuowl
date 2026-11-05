#!/bin/bash
# Stage 2 against the GMP oracle (check_stage2), on any OpenCL device.
#   run_stage2.sh <prpll> <workdir> <device> <name> <p> <B1> <B2> <D> <fft spec or -> <extra -use or -> [x source: random:<seed> | stage1]
#                 [M]
# D = 0 lets the planner choose D and M; M (default 0) = the planner's M for D.
# Runs prpll's stage 2 (PRPLL_STAGE2_TEST) on x and checks the accumulator with "check_stage2 acc".  Prints one OK/FAIL line.
# PRPLL_STAGE2_TEST exists only in the fault build:  make BIN=build-pm1faults CPPFLAGS=-DPRPLL_PM1_FAULTS, then <prpll> =
# build-pm1faults/prpll.  (End-to-end runs from worktodo lines with the normal binary:  stage2.sh.)  Environment KCACHE (optional):  see
# below.
set -u
PRPLL=$1; WORK=$2; DEV=$3; NAME=$4; P=$5; B1=$6; B2=$7; D=$8; FFT=$9; USE=${10}; XSRC=${11:-random:1}; M=${12:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
grep -q PRPLL_STAGE2_TEST "$PRPLL" || { echo "FAIL $NAME: $PRPLL is not a fault build (no PRPLL_STAGE2_TEST)"; exit 2; }
mkdir -p "$WORK/$NAME" && cd "$WORK/$NAME" || exit 2
rm -rf p2-* acc.bin
case $XSRC in
  random:*) "$HERE/check_stage2" random "$P" "${XSRC#random:}" x.bin ;;
  stage1) "$HERE/check_stage2" stage1 "$P" "$B1" x.bin ;;
esac
OPTS=(-device "$DEV")
# KCACHE=<dir>:  a kernel cache shared by the runs (-cache), so that each run does not compile the kernels again
if [ -n "${KCACHE:-}" ]; then mkdir -p "$KCACHE"; rm -rf kernel-cache; ln -s "$KCACHE" kernel-cache; OPTS+=(-cache); fi
[ "$FFT" != - ] && OPTS+=(-fft "$FFT")
[ "$USE" != - ] && OPTS+=(-use "$USE")
PRPLL_STAGE2_TEST=$P,$B1,$B2,x.bin,acc.bin,$D,$M "$PRPLL" "${OPTS[@]}" > log.txt 2>&1
if [ ! -f acc.bin ]; then echo "FAIL $NAME: no acc ($(grep -E 'stage2Test|Exception|EE' log.txt | tail -1))"; exit 1; fi
read -r D M KW <<< "$(grep -o 'Stage 2 B1=[0-9]* B2=[0-9]*: D=[0-9]* M=[0-9]* .* windows of [0-9]*' log.txt | tail -1 \
  | sed 's/.*D=\([0-9]*\) M=\([0-9]*\) .* windows of \([0-9]*\).*/\1 \2 \3/')"
R=$("$HERE/check_stage2" acc "$P" "$B1" "$B2" "$D" "$M" x.bin acc.bin "$KW")
FFTLINE=$(grep -o 'FFT: .*' log.txt | head -1)
echo "$NAME: $R | $FFTLINE | $(grep -o 'max ROE [0-9.]*' log.txt | tail -n 1)"
[[ $R == OK* ]]
