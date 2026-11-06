// Copyright (C) Mihai Preda and George Woltman

#pragma once

// P-1 stage 2 on the GPU, paired "n^2" form (see pm1::Stage2Plan):  acc = the product over the items (k, n) of x^((kD)^2) - x^(n^2)
// mod 2^E-1, where x is the stage 1 residue.  A factor f of 2^E-1 for which x has prime order q in (B1, B2] divides acc; the caller takes
// GCD(acc, 2^E-1).

#include "common.h"
#include "GpuCommon.h"
#include "FFTConfig.h"

class Args;
class Gpu;

struct Stage2Result {
  enum Status {DONE, DECLINED, STOPPED};
  Status status{};
  string why{};             // DECLINED: the reason
  vector<u32> acc{};        // DONE: the accumulator
  u64 nMul{};               // accumulator and k-step multiplications of the plan, over all runs
};

class Stage2 {
public:
  // The FFTs for stage 2 of exponent E, in order (pm1Ffts):  as for a run without an error check, with 0.5 bits per word of margin over
  // FFTConfig's limit (the difference of two residues has words up to twice as large), and with the wide carry where the doubled
  // products need it.  Throws when an -fft is given and none of its FFTs can run stage 2.
  static vector<Pm1Fft> chooseFFTs(const Args& args, u64 E);

  // The FFT stage 2 of E runs on `fft`:  the wide carry where the doubled products need it.  keepCarry:  an explicit narrow carry (an
  // -fft spec's) is kept.
  static FFTConfig configFor(FFTConfig fft, u64 E, bool keepCarry = false);

  // Runs stage 2 on its own Gpu, resuming from the savefile in p2-<E>/ when there is one for the same E, B1, B2 and x.  Holds the
  // device's memory lock (Memlock) while it runs.  D = 0 lets the memory planner choose D, M = 0 lets it choose M (for D).  Stops (STOPPED, with a checkpoint) on a
  // stop request or, with -iters <n>, at the first block boundary after n multiplications of this run.  Throws on a GPU error it could
  // not recover from.
  // Without -fft, an FFT whose kernels can't be built is replaced by the next of chooseFFTs; with -fft only the FFTs given are tried,
  // and stage 2 throws when none of them can run.
  static Stage2Result run(GpuCommon shared, u64 E, u64 B1, u64 B2, const vector<u32>& x, u32 D = 0, u32 M = 0);

  // One timing of stage 2 multiplications for E on fft (as configFor gives it) with these -use settings (fftUses: the command line wins
  // over them):  us per multiplication as stage 2 counts them (blocks of 120 and their k step), the res64 of the accumulator (equal on
  // every FFT and setting for the same E:  the result of a candidate that differs is wrong), the max ROE and the settings the Gpu
  // resolved.  us is +inf when a check failed:  the table (as in stage 2), one multiplication by A_k - B_r against A_k and B_r multiplied
  // apart, the ROE limit, acc != 0.  Throws when the Gpu can't be built.  quick as for Gpu::timePRP (7: 1200 multiplications).
  struct MulTime {
    double us;
    u64 res64{};
    double maxROE{};
    u32 inPlace{}, tailKernels{}, multiQ{}, l2Striping{};
  };
  static MulTime timeMul(GpuCommon shared, u64 E, const FFTConfig& fft, const vector<KeyVal>& uses, int quick = 7);

  // The number of table buffers a stage 2 of E would have room for (pm1::Stage2Plan's table entries), estimated before stage 1 (for the bounds of a Pfactor line):  run()'s
  // memory budget now, with g (the stage 1 Gpu) holding about what stage 2's own Gpu will hold, in buffers of the size of chooseFFTs'
  // first FFT with g's layout.  0 when stage 2 can't run at all.
  static u64 tableCapacity(GpuCommon shared, const Gpu& g, u64 E);

  // Whether p2-<E>/ holds a savefile of the stage 2 for these E, B1, B2 and x (it was started, so the stage 1 GCD found no factor)
  static bool hasSavefile(u64 E, u64 B1, u64 B2, const vector<u32>& x);

  // Orders the untuned candidates ffts by their timeMul at E (for the rest of the run, the same candidates in the same order)
  static void rankUntuned(GpuCommon shared, u64 E, vector<Pm1Fft>& ffts);

private:

  // run on one FFT, with at most maxNCap table buffers
  static Stage2Result runOn(GpuCommon shared, u64 E, u64 B1, u64 B2, const vector<u32>& x, u32 D, u32 M, const FFTConfig& fft,
                            const vector<KeyVal>& uses, u64 maxNCap);
};

#ifdef PRPLL_PM1_FAULTS
// TEST ONLY, in the fault build (test/pm1/run_stage2.sh):  spec "E,B1,B2,x.bin,acc.bin[,D[,M]]" runs stage 2 on x (little-endian u32 words,
// nWords(E) of them) and writes the accumulator to acc.bin (nothing when stage 2 declined or stopped).
void stage2Test(GpuCommon shared, const char* spec);
// TEST ONLY, fault build:  spec "E,fft[,quick[,rounds]]" logs Stage2::timeMul of fft at E (PRPLL_TIMEMUL_SHAPE=<block>,<table> varies it)
void timeMulTest(GpuCommon shared, const char* spec);
#endif
