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

// ---- P-1 stage 1 inside a PRP test (Pm1Prp) ----
// The window length L runs from INLINE_MIN_L to INLINE_MAX_L, and L needs 2^L buckets, u_s and the fold's INLINE_TEMP_BUFS buffers.
inline constexpr u32 INLINE_MIN_L = 2;
inline constexpr u32 INLINE_MAX_L = 16;
inline constexpr u32 INLINE_TEMP_BUFS = 6;
inline constexpr u64 inlineBuffers(u32 L) { return (u64(1) << L) + 1 + INLINE_TEMP_BUFS; }
// The cheapest L for `bits` bits of the stage 1 exponent over `iters` PRP iterations, with folds at the PRP's verified checks
// (`checkStep` apart) and at most maxBufs residue buffers, and its cost in PRP iterations:  an accumulator multiplication per L
// iterations and a fold of the 2^L buckets at every check.  L = 0 (cost infinite) when maxBufs is too few for INLINE_MIN_L.
struct InlineStage1 { u32 L{}; double cost{}; };
InlineStage1 inlineStage1(double bits, double iters, double checkStep, u64 maxBufs);

// How stage 1 is run, for its cost in the bounds:  on its own (maxBufs = 0), or inside a PRP test with room for maxBufs residue buffers
// and that test's verified-check interval checkStep.  Inside a PRP test, a factor also comes after the PRP has done the iterations of
// stage 1, so those are not saved.
struct Stage1Model { u64 maxBufs{}; double checkStep{}; };

// Costs, in PRP iterations of the exponent:  stage 1 (on its own, one iteration per bit of its exponent, plus the Gerbicz check;  inside
// a PRP test, inlineStage1) and one GCD
double stage1Cost(double B1, const Stage1Model& model = {});
double gcdCost(double exponent);

// ---- Stage 2 cost model, for the stage 2 planner (Stage2::run) and the bounds ----
// The fraction of the primes that the pairing of pm1::Stage2Plan pairs (2 * nPairs / nPrimes) where the primes of the classes coprime to
// D have the density rho = (D / phi(D)) / ln q and each prime has L = 2M / D candidate partners:  from a table of exact plans
double pairedFractionAt(double rho, double L);
// Estimates of Stage2Plan's nNeeded and of its paired fraction
double stage2Items(double B1, double B2, u32 D, u32 M);
double pairedFraction(double B1, double B2, u32 D, u32 M);

// A stage 2 plan (D, M) and its estimated cost in stage 2 multiplications (an accumulator multiplication by A_k - B_n is 1)
struct Stage2Choice {
  u32 D{}, M{};            // D = 0:  no stage 2 (no table fits)
  u32 tableSize{};
  double items{}, kSteps{}, tableSteps{};
  double cost{};
};
// The cost of (D, M) for (B1, B2]:  the items, the k steps, the table build and the duplicated blocks (Stage2.cpp)
Stage2Choice stage2Estimate(u64 B1, u64 B2, u32 D, u32 M);
// The cheapest (D, M) whose table has at most maxN entries, among the D of stage2Ds() that can run with B1 (or onlyD alone)
Stage2Choice chooseStage2(u64 B1, u64 B2, u64 maxN, u32 onlyD = 0);
// The cost of the plan stage 2 would make with maxN table buffers, in PRP iterations of the exponent (0 for B2 <= B1)
double stage2Cost(u64 B1, u64 B2, u64 maxN);
// The smallest table any stage 2 can run with (D = 30)
u64 stage2MinTable();

struct Bounds {
  u64 B1{}, B2{};     // B2 = B1: stage 1 only
  u32 D{}, M{};       // the stage 2 plan of the model (D = 0:  no stage 2)
  FactorProb prob{};
  double cost{};      // the expected cost (stage 2 runs only when stage 1 found nothing)
  double savings{};   // prob * (testsSaved * exponent - the iterations a PRP test has done when stage 1 ends) - cost:  P-1 with these bounds pays for itself when > 0
};

// maxN:  the number of table buffers stage 2 can have (Stage2::tableCapacity); below stage2MinTable() there is no stage 2.
Bounds evaluateBounds(u64 exponent, double tfBits, double testsSaved, u64 B1, u64 B2, u64 maxN, const Stage1Model& model = {});

// The best bounds (the largest saving, which can be negative).  B1 >= 11, B1 <= B2 <= MAX_B2; rounded to 3 significant digits.
Bounds chooseBounds(u64 exponent, double tfBits, double testsSaved, u64 maxN, const Stage1Model& model = {});

// The best B2 for a given B1 (a stage 1 that is already under way)
Bounds chooseB2(u64 exponent, double tfBits, double testsSaved, u64 B1, u64 maxN, const Stage1Model& model = {});

// The trial factoring depth assumed for an exponent whose PRP line does not give one:  GIMPS's GPU trial factoring target (Prime95's
// table of CPU limits plus 4, as AutoPrimeNet's factor_limit) less 1 bit, to err low (a lower depth raises the chance of a factor and
// the bounds a little).  E.g. 76 at 141M, 78 at 206M.
double assumedTfBits(u64 exponent);

// The largest B2 stage 2 takes (Stage2::run)
inline constexpr u64 MAX_B2 = (u64(1) << 32) - 8192;

}
