// Copyright (C) Mihai Preda

#pragma once

// The P-1 bounds of "Pfactor=" lines:  B1 and B2 that maximise the expected saving,
//   P(a factor) * testsSaved * (one PRP test) - E[cost of the P-1],
// all in PRP iterations of the exponent.  The probability is the Dickman rho model of the old bounds calculator (pm1/pm1.cpp); the costs
// are prpll's measured ones (see Pm1Bounds.cpp).  Pure floating point:  no GPU, no bignum.

#include "common.h"

namespace pm1 {

// The probability that P-1 with B1, B2 finds a factor of 2^exponent - 1 that has no factor below 2^tfBits:  stage1 (a B1-smooth k, k of the
// factor 2kp + 1) and stage2 (B1-smooth but for one prime in (B1, B2], so not found by stage 1).
struct FactorProb {
  double stage1, stage2;
  [[nodiscard]] double total() const { return stage1 + stage2; }
};
FactorProb factorProbability(double exponent, double tfBits, double B1, double B2);

// Costs, in PRP iterations of the exponent:  stage 1 (one iteration per bit of its exponent, plus the Gerbicz check), one GCD, and stage
// 2 on D's table (its multiplications as estimated by stage2Muls, then each one's cost).
double stage1Cost(double B1);
double gcdCost(double exponent);
double stage2Muls(double B1, double B2, u32 D);
double stage2Cost(double B1, double B2, u32 D);

// The fraction of the primes in (B1, B2] that stage 2 pairs at D (pm1::Stage2Plan: 2 * nPairs / nPrimes), estimated
double pairedFraction(double B1, double B2, u32 D);

struct Bounds {
  u64 B1{}, B2{};     // B2 = B1: stage 1 only
  u32 D{};            // the D of stage 2 in the model (0:  no stage 2)
  FactorProb prob{};
  double cost{};      // the expected cost (stage 2 runs only when stage 1 found nothing)
  double savings{};   // prob * testsSaved * exponent - cost:  P-1 with these bounds pays for itself when > 0
};

// maxR:  the number of table buffers stage 2 can have (Stage2::tableCapacity), and so the D it can run; below the smallest table
// (stage2nR(210)) there is no stage 2.
Bounds evaluateBounds(u64 exponent, double tfBits, double testsSaved, u64 B1, u64 B2, u64 maxR);

// The best bounds (the largest saving, which can be negative).  B1 >= 11, B1 <= B2 <= MAX_B2; rounded to 3 significant digits.
Bounds chooseBounds(u64 exponent, double tfBits, double testsSaved, u64 maxR);

// The best B2 for a given B1 (a stage 1 that is already under way)
Bounds chooseB2(u64 exponent, double tfBits, double testsSaved, u64 B1, u64 maxR);

// The largest B2 stage 2 takes (Stage2::run)
inline constexpr u64 MAX_B2 = (u64(1) << 32) - 8192;

}
