// Copyright (C) Mihai Preda

#pragma once

#include "Primes.h"
#include "GpuCommon.h"
#include "FFTConfig.h"

#include <array>
#include <vector>

class GpuCommon;
class RoeInfo;
class Gpu;

using TuneConfig = vector<KeyVal>;

class Tune {
private:
  GpuCommon shared;
  Primes primes;
  int workers = 1;              // -tune workers=N: time -use options with N concurrent workers

  double timeOption(u64 exponent, FFTConfig fft, int quick);

  float maxBpw(FFTConfig fft);
  float zForBpw(float bpw, FFTConfig fft, u32);

public:
  Tune(GpuCommon shared) : shared{shared} {}

  // Find the max-BPW for each FFT
  void ztune();

  // Find the best configuration for each FFT
  void ctune();

  // Considering the cost of each FFT and the max-BPW, work out the transition points between them
  void tune();

  void carryTune();
};
