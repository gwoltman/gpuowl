// Copyright (C) Mihai Preda

#pragma once

#include "Primes.h"
#include "GpuCommon.h"
#include "FFTConfig.h"

#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

class GpuCommon;
class RoeInfo;
class Gpu;
class TuneEntry;
struct CudaSmLimits;
struct AmdSimdLimits;
#ifdef CUDA_BACKEND
using RegLimits = CudaSmLimits;
#else
using RegLimits = AmdSimdLimits;
#endif

using TuneConfig = vector<KeyVal>;

class Tune {
private:
  GpuCommon shared;
  Primes primes;
  // Whether WMUL=1 made carryFused faster, by a key of what determines carryFused (see regTuneEntry), for the rest of the tune
  std::map<std::string, bool> wmul1Results;
  // How the other TAIL_KERNELS choices did in regTuneEntry, by FFT type and TAIL_KERNELS value, for the rest of the tune
  // wins/losses/badLosses: against the entry's own choice.  beatSibling/lostToSibling: 3 vs 13 (or 1 vs 11), the same kernels with and
  // without a queue of their own.
  struct TailAltRecord { int wins = 0, losses = 0, badLosses = 0, beatSibling = 0, lostToSibling = 0; };
  std::map<std::string, TailAltRecord> tailAltResults;
  // -tune tk=: 0 (the default, until there is more data from users' GPUs) never tries the other TAIL_KERNELS choices, auto (-1) tries them,
  // N > 0 only times TAIL_KERNELS=N
  int tkOption = 0;

  float maxBpw(FFTConfig fft);
  float zForBpw(float bpw, FFTConfig fft, u32);

  // Whether this device's register usage can be tuned: CUDA, and AMD where the runtime reports the kernels' VGPRs
  bool canTuneRegs();

  // Find the best register limits for one tune.txt entry: returns the entry with its new cost and REGxxxx settings
  // tailOnly: only the tail kernels' register limits are searched, the line's other REGxxxx settings and WMUL are kept as they are,
  // and the other TAIL_KERNELS choices are not tried (regTuneEntry's own use, for those choices).
  // fwdOcc/fwdDir (tailOnly): each tail kernel tries only the occupancy of its best in the other TAIL_KERNELS choice (fwdOcc, by REGxxxx key)
  // and fwdDir more (two kernels: fewer registers) or fewer (one kernel), and twice that if the first step won.
  TuneEntry regTuneEntry(const TuneEntry& e, int quick, const RegLimits& sm, bool tailOnly = false,
                         const std::map<std::string, int>* fwdOcc = nullptr, int fwdDir = 0);

public:
  Tune(GpuCommon shared) : shared{shared} {}

  // Find the max-BPW for each FFT
  void ztune();

  // Find the best configuration for each FFT
  void ctune();

  // Considering the cost of each FFT and the max-BPW, work out the transition points between them
  void tune();

  void carryTune();

  // Find the best register limits (REGxxxx settings) for each tune.txt entry, or only those of onlyShapes if not empty,
  // and only those of onlySpecs (FFTConfig specs) if given
  void regTune(int quick, const vector<FFTShape>& onlyShapes, const std::set<std::string>* onlySpecs = nullptr,
               u64 minExp = 0, u64 maxExp = UINT64_MAX / 2);

  // Time the other variants of the FFT shapes in tune.txt (or only onlyShapes) as -tune times a shape, add those that earn an entry,
  // then tune their register usage.  variant0 is whether this device runs variant 0 (BCAST).  Variants with a max exponent outside
  // [minExp, maxExp] are skipped.
  void variantTune(int quick, bool variant0, const vector<FFTShape>& onlyShapes, u64 minExp, u64 maxExp);
};
