// Copyright (C) Mihai Preda

#pragma once

#include "Args.h"
#include "common.h"
#include "GpuCommon.h"
#include "FFTConfig.h"

#include <memory>
#include <string>

class Args;
class Gpu;
class Result;
class Context;
class TrigBufCache;

class Task {
public:
  enum Kind {PRP, VERIFY, LL, CERT, PM1};

  Kind kind;
  u64 exponent{};
  string AID{};  // Assignment ID
  string line{}; // the verbatim worktodo line, used in deleteTask().
  u32 squarings{};  // For CERTs
  string verifyPath{}; // For Verify

  // For P-1 (Pminus1= and Pfactor= lines).  Pminus1 gives the bounds.  Pfactor gives the trial factoring depth and the number of
  // primality tests that a factor would save, and the bounds are chosen from those.
  u64 B1{}, B2{};
  double tfBits{}, testsSaved{};
  bool pfactor{};

  void execute(GpuCommon shared, u32 instance);
  void executeWith(GpuCommon shared, u32 instance, FFTConfig fft, const vector<KeyVal>& fftUses);   // execute's work on one FFT
  void executePM1(GpuCommon shared, u32 instance, FFTConfig fft, std::unique_ptr<Gpu> gpu);

  void writeResultPRP(FFTConfig fft, const Args&, u32 instance, bool isPrime, u64 res64, const std::string& res2048, u32 nErrors, const fs::path& proofPath) const;
  void writeResultLL(FFTConfig fft, const Args&, u32 instance, bool isPrime, u64 res64) const;
  void writeResultCERT(FFTConfig fft, const Args&, u32 instance, array <u64, 4> hash, u32 squarings) const;
  void writeResultPM1(FFTConfig fft, const Args&, u32 instance, u64 B1, u64 B2, const std::vector<std::string>& factors, u32 nErrors,
                      u32 nJacobi) const;
};
