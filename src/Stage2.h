// Copyright (C) Mihai Preda and George Woltman

#pragma once

// P-1 stage 2 on the GPU, paired "n^2" form (see pm1::Stage2Plan):  acc = the product over the needed (k, r) of x^((kD)^2) - x^(r^2)
// mod 2^E-1, where x is the stage 1 residue.  A factor f of 2^E-1 for which x has prime order q in (B1, B2] divides acc; the caller takes
// GCD(acc, 2^E-1).

#include "common.h"
#include "GpuCommon.h"
#include "FFTConfig.h"

class Args;

struct Stage2Result {
  enum Status {DONE, DECLINED, STOPPED};
  Status status{};
  string why{};             // DECLINED: the reason
  vector<u32> acc{};        // DONE: the accumulator
  u64 nMul{};               // accumulator and k-step multiplications of the plan, over all runs
};

class Stage2 {
public:
  // The FFT for stage 2 of exponent E:  as for a run without an error check, with 0.5 bits per word of margin over FFTConfig's limit
  // (the difference of two residues has words up to twice as large), and with the wide carry where the doubled products need it.
  static FFTConfig chooseFFT(const Args& args, u64 E, vector<KeyVal>* uses);

  // Runs stage 2 on its own Gpu, resuming from the savefile in p2-<E>/ when there is one for the same E, B1, B2 and x.  Holds the
  // device's memory lock (Memlock) while it runs.  D = 0 lets the memory planner choose D.  Stops (STOPPED, with a checkpoint) on a
  // stop request or, with -iters <n>, at the first block boundary after n multiplications of this run.  Throws on a GPU error it could
  // not recover from.
  static Stage2Result run(GpuCommon shared, u64 E, u64 B1, u64 B2, const vector<u32>& x, u32 D = 0);

  // Whether p2-<E>/ holds a savefile of the stage 2 for these E, B1, B2 and x (it was started, so the stage 1 GCD found no factor)
  static bool hasSavefile(u64 E, u64 B1, u64 B2, const vector<u32>& x);
};

#ifdef PRPLL_PM1_FAULTS
// TEST ONLY, in the fault build (test/pm1/run_stage2.sh):  spec "E,B1,B2,x.bin,acc.bin[,D]" runs stage 2 on x (little-endian u32 words,
// nWords(E) of them) and writes the accumulator to acc.bin (nothing when stage 2 declined or stopped).
void stage2Test(GpuCommon shared, const char* spec);
#endif
