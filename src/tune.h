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

using TuneConfig = vector<KeyVal>;

class Tune {
private:
  GpuCommon shared;
  Primes primes;
  // Whether WMUL=1 made carryFused faster, by a key of what determines carryFused (see regTuneEntry), for the rest of the tune
  std::map<std::string, bool> wmul1Results;

  int workers = 1;              // -tune workers=N: time -use options with N concurrent workers

  double timeOption(u64 exponent, FFTConfig fft, int quick, u32* usedWmul = nullptr);

  float maxBpw(FFTConfig fft);
  float zForBpw(float bpw, FFTConfig fft, u32);

#ifdef CUDA_BACKEND
  // Find the best register limits for one tune.txt entry: returns the entry with its new cost and REGxxxx settings
  TuneEntry regTuneEntry(const TuneEntry& e, int quick, const CudaSmLimits& sm);
#endif

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
  void regTune(int quick, const vector<FFTShape>& onlyShapes, const std::set<std::string>* onlySpecs = nullptr);

  // Time the other variants of the FFT shapes in tune.txt (or only onlyShapes) as -tune times a shape, add those that earn an entry,
  // then tune their register usage.  variant0 is whether this device runs variant 0 (BCAST).  Variants with a max exponent outside
  // [minExp, maxExp] are skipped.
  void variantTune(int quick, bool variant0, const vector<FFTShape>& onlyShapes, u64 minExp, u64 maxExp);
};
