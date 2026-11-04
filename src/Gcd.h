// Copyright (C) Mihai Preda and George Woltman

#pragma once

// GCD and Jacobi symbol against N = 2^p - 1 on the GPU, for P-1: gcd(x - 1, N) is the factor test of stage 1 and stage 2, and the
// Jacobi symbol (x | N) checks a residue.  Bernstein-Yang divsteps over 62-bit digits (see src/cl/gcd/gcd.cl); the kernels are a
// program of their own, so they leave the PRP kernels' compile cache alone.  There is no CPU fallback.

#include "Buffer.h"
#include "Kernel.h"
#include "KernelCompiler.h"
#include "Profile.h"
#include "Queue.h"
#include "common.h"
#include "GpuCommon.h"

#include <map>
#include <memory>

class Gpu;

class Gcd {
public:
  // skew > 0 (tests only) builds the kernels with -DSKEW=skew: the digits then often leave their canonical range (see gcd.cl)
  Gcd(GpuCommon shared, u32 skew = 0);

  // gcd(x - 1, 2^p - 1), where x holds the p-bit residue in 32-bit words (as Gpu::readAndCompress returns it; 2^p - 1 stands for
  // zero).  The result's words without the leading zero words: {1} when x - 1 and 2^p - 1 are coprime.
  Words gcdMinus1(const Words& x, u32 p);

  // gcd(x, 2^p - 1), as gcdMinus1 (for the accumulator of stage 2).  x = 0 gives 2^p - 1.
  Words gcd(const Words& x, u32 p);

  // The Jacobi symbol (x | 2^p - 1): -1, 0 or 1
  int jacobi(const Words& x, u32 p);

  struct Stats {
    u64 divsteps;       // of the last call, a multiple of 62 (it can overshoot the end by a few readback batches)
    u64 skewEvents;     // the -DSKEW events (non-canonical digits made) of the last call, 0 without skew
    double seconds;
  };
  [[nodiscard]] const Stats& lastStats() const { return stats; }

  // For tests: the leaf kernel's matrix of 62 nb divsteps from the low digits of f and g (62-bit digits, at least nb + 2 of each),
  // as U, V, Q, R (nb + 2 digits each) then delta and jac after the divsteps; and its time in seconds
  // For tests: the digits of a b (schoolbook kernel), and the subquadratic jump matrix of 62 nb divsteps (as leafMatrix)
  vector<i64> mulTest(const vector<i64>& a, const vector<i64>& b);
  vector<i64> jumpMatrix(const vector<i64>& f, const vector<i64>& g, u32 nb, i64 delta);
  u32 leafBatches = 250;      // the largest jump done by one leaf (at most LEAF_NMAX - 2 in gcd.cl)
  u32 fftDigits = 4000;       // products of numbers this long (in digits) or longer go through the PRPLL transforms
  u32 halfGcdDigits = 0;      // 0: quadratic only; else the subquadratic GCD runs while f or g has more digits than this

  vector<i64> leafMatrix(const vector<i64>& f, const vector<i64>& g, u32 nb, i64 delta, bool jacobi, double* seconds = nullptr);

private:
  GpuCommon shared;
  Profile profile;
  Queue queue;
  KernelCompiler compiler;
  u32 skew;
  Stats stats{};

  Kernel kToDigits, kInit, kInitJ, kApply[2], kApplyJ[2], kLengths, kSummary, kSummaryJ, kNotEqual, kBlockSigns, kBlockBorrows, kToWords, kLeaf, kLeafJ;

  Kernel kMulColumns, kMulDigits, kAddDigits;

  // The (f, g) pair of the last run, with f the result: GCD and Jacobi share the run loop.
  struct Run;
  // Digits [off, off + len) of a buffer
  struct Num {
    std::shared_ptr<Buffer<i64>> buf;
    u32 off = 0, len = 0;
    [[nodiscard]] Num slice(u32 from, u32 n) const { return {buf, off + from, std::min(n, len > from ? len - from : 0)}; }
  };
  struct Matrix { Num u, v, q, r; };
  Num newNum(u32 len);
  Words toBinary(const Num& x, bool& neg);
  Num fromBinary(const Words& w, bool neg, u32 n);
  struct Rung { u64 E; std::unique_ptr<Gpu> gpu; };
  std::map<u32, Rung> ladder;     // the Gpus of the FFT products, by FFT size
  Rung& gpuFor(u64 bits);
  Num mulLow(const Num& a, const Num& b, u32 n);                       // digits [0, n) of a b
  Num addShift(const Num& a, const Num& b, i64 sb, u32 drop, u32 n);   // digits [0, n) of (a + sb b) / 2^(62 drop)
  Num combine(const Num& x, const Num& a, const Num& y, const Num& b, u32 drop, u32 n);   // (x a + y b) / 2^(62 drop)
  Matrix jump(u32 nb, Buffer<i64>& delta, const Num& f, const Num& g, bool jacobi);
  std::map<u32, vector<std::unique_ptr<Buffer<i64>>>> pool;      // free digit buffers by size, see newNum
  Buffer<u32> scratchCount;
  std::pair<i64, i64> halfGcd(Run& r, bool jacobi);
  void run(Run& r, const Words& x, i64 sub, u32 p, bool jacobi);   // g = x - sub
  Words gcdOf(const Words& x, i64 sub, u32 p);
  Words toWords(Run& r, u32 nOut);
};
