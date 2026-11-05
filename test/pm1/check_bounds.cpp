// Checks of the Pfactor bounds model (src/Pm1Bounds.cpp):  the probability against mprime's pm1prob (MPRIME_SRC, an independent copy of
// the same Dickman model, built here only as an oracle), monotonicity, the estimated stage 2 pairing against Stage2Plan's exact counts,
// and the choice of the bounds (B2 >= B1, memory limits, local optimality, no P-1 when it can't pay).
//   make check_bounds MPRIME_SRC=~/mprime && ./check_bounds
//   ./check_bounds p,tfBits,testsSaved,maxR ...      print the choices for these cases
//   ./check_bounds eval:p,tfBits,testsSaved,maxR,B1,B2 ...   print the model's chance, cost and saving at these bounds

#include "Pm1Bounds.h"
#include "Pm1.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <utility>

#ifdef HAVE_PM1PROB
extern "C" double pm1prob(double takeAwayBits, double factoredUpTo, uint64_t B1, uint64_t B2);
#endif

using namespace pm1;

static int bad = 0;
static void check(bool ok, const char *what) {
  if (!ok) { printf("FAIL: %s\n", what); ++bad; }
}

static void print(const char* tag, u64 p, double tf, double ts, u64 maxR, const Bounds& b, double ms) {
  printf("%s p=%" PRIu64 " tf=%g ts=%g maxR=%" PRIu64 ": B1=%" PRIu64 " B2=%" PRIu64 " D=%u  P=%.3f%% (%.3f%% + %.3f%%)  cost %.3f%%  "
         "saving %.3f%% of a PRP test  [%.0f ms]\n", tag, p, tf, ts, maxR, b.B1, b.B2, b.D, 100 * b.prob.total(), 100 * b.prob.stage1,
         100 * b.prob.stage2, 100 * b.cost / p, 100 * b.savings / p, ms);
}

static Bounds timedChoice(u64 p, double tf, double ts, u64 maxR, double* ms) {
  auto const t0 = std::chrono::steady_clock::now();
  Bounds const b = chooseBounds(p, tf, ts, maxR);
  *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return b;
}

static void testProbability() {
  char what[200];
  // Monotone:  up in B1 and in B2, down in the TF depth
  for (double p : {1e6, 1e8}) {
    double last = 0;
    for (double B1 = 1000; B1 <= 1e7; B1 *= 1.7) {
      double const q = factorProbability(p, 70, B1, B1).total();
      snprintf(what, sizeof(what), "P up in B1 (p=%g B1=%g)", p, B1);
      check(q > last, what);
      last = q;
    }
    last = 0;
    for (double B2 = 1e6; B2 <= 4e9; B2 *= 1.7) {
      double const q = factorProbability(p, 70, 1e6, B2).total();
      snprintf(what, sizeof(what), "P up in B2 (p=%g B2=%g)", p, B2);
      check(q >= last, what);
      last = q;
    }
    last = 1;
    for (double tf = 60; tf <= 90; tf += 1) {
      double const q = factorProbability(p, tf, 1e6, 3e7).total();
      snprintf(what, sizeof(what), "P down in tfBits (p=%g tf=%g)", p, tf);
      check(q < last, what);
      last = q;
    }
  }
  // B2 <= B1:  no stage 2 part
  check(factorProbability(1e8, 77, 1e6, 1e6).stage2 == 0, "B2 = B1: stage 2 probability 0");
  check(factorProbability(1e8, 77, 1e6, 5e5).stage2 == 0, "B2 < B1 counts as B2 = B1");
  // Tiny B1 (alpha beyond the rho table) and tfBits below 2p:  finite, in [0, 1]
  for (auto [p, tf, B1, B2] : {std::tuple{1e8, 77.0, 11.0, 11.0}, {1e8, 77.0, 11.0, 1e6}, {4e5, 0.0, 1000.0, 2e5}, {4e5, 19.0, 1e3, 1e3}}) {
    FactorProb const q = factorProbability(p, tf, B1, B2);
    snprintf(what, sizeof(what), "P in [0, 1] (p=%g tf=%g B1=%g B2=%g): %g + %g", p, tf, B1, B2, q.stage1, q.stage2);
    check(std::isfinite(q.total()) && q.stage1 >= 0 && q.stage2 >= 0 && q.total() <= 1, what);
  }

#ifdef HAVE_PM1PROB
  // Against mprime's pm1prob (rate model, slices of 1/4 bit, rho by 1/20):  the same model, different details; within 3% where the
  // probability is above 0.1%, within 8% in the far tail (B1 = B2 = 1000 with deep TF)
  double worst = 0, worstAbove = 0;
  for (double p : {4e5, 5e6, 3e7, 1.1e8, 3.32e8}) {
    for (double tf : {std::max(25.0, round(log2(p) + 6)), round(log2(p) + 24), round(log2(p) + 50)}) {
      for (double B1 : {1e3, 1e5, 1e6}) {
        for (double r : {1.0, 30.0, 300.0}) {
          double const ours = factorProbability(p, tf, B1, B1 * r).total();
          double const theirs = pm1prob(log2(p) + 1, tf, u64(B1), u64(B1 * r));
          double const rel = fabs(ours / theirs - 1);
          worst = std::max(worst, rel);
          if (theirs > 1e-3) { worstAbove = std::max(worstAbove, rel); }
          snprintf(what, sizeof(what), "P vs mprime pm1prob (p=%g tf=%g B1=%g B2=%g): %.5g%% vs %.5g%%", p, tf, B1, B1 * r, 100 * ours,
                   100 * theirs);
          check(rel < (theirs > 1e-3 ? 0.03 : 0.08), what);
        }
      }
    }
  }
  printf("probability vs mprime's pm1prob: worst relative difference %.2f%% where P > 0.1%%, %.2f%% overall (135 points)\n",
         100 * worstAbove, 100 * worst);
#else
  printf("probability vs mprime's pm1prob: not checked (build with MPRIME_SRC=<mprime sources>)\n");
#endif
}

static void testPairing() {
  // The estimated paired fraction and the multiplications against Stage2Plan's exact counts
  struct P { u64 B1, B2; u32 D; } ps[] = {{1000, 200000, 210}, {50000, 1500000, 2310}, {1000000, 30000000, 330}, {1000000, 30000000, 2310},
                                          {1000000, 30000000, 4620}, {100000, 30000000, 210}, {10000000, 300000000, 2310},
                                          {300000, 10000000, 1050}, {4370, 540000, 1470}};
  for (P p : ps) {
    Stage2Plan const plan = planStage2(p.B1, p.B2, p.D);
    double const exact = 2.0 * plan.nPairs / plan.nPrimes;
    double const est = pairedFraction(double(p.B1), double(p.B2), p.D);
    // stage2Muls adds the duplicated blocks (1/32) and the table build (D) to the loop's multiplications (mulsFrom)
    double const mulsExact = double(plan.mulsFrom(plan.kBegin)) * (1 + 1.0 / 32) + p.D;
    double const mulsEst = stage2Muls(double(p.B1), double(p.B2), p.D);
    char what[200];
    snprintf(what, sizeof(what), "pairing B1=%" PRIu64 " B2=%" PRIu64 " D=%u: paired %.4f est %.4f, multiplications %.0f est %.0f", p.B1,
             p.B2, p.D, exact, est, mulsExact, mulsEst);
    check(fabs(est - exact) < 0.01 && fabs(mulsEst / mulsExact - 1) < 0.01, what);
    printf("%s\n", what);
  }
}

static void testChoice() {
  char what[300];
  double ms = 0;
  struct C { u64 p; double tf, ts; } cs[] = {{400381, 25, 1}, {1000099, 30, 1}, {5000011, 66, 1}, {30000001, 72, 1}, {41000017, 74, 1},
                                           {60000011, 75, 1}, {110000017, 77, 1}, {110000017, 77, 2}, {332000027, 81, 1}};
  for (C c : cs) {
    Bounds prev{};
    for (u64 maxR : {0, 23, 24, 48, 240, 480, 960}) {
      Bounds const b = timedChoice(c.p, c.tf, c.ts, maxR, &ms);
      print("choice", c.p, c.tf, c.ts, maxR, b, ms);
      if (b.savings <= 0) {
        // Not worth it:  then no B1 on a coarse grid, with its best B2, saves anything either
        bool none = true;
        for (double B1 = 100; B1 < c.p / 10.0; B1 *= 1.5) { none = none && chooseB2(c.p, c.tf, c.ts, u64(B1), maxR).savings <= 0; }
        snprintf(what, sizeof(what), "p=%" PRIu64 " maxR=%" PRIu64 ": no saving, and no B1 saves anything", c.p, maxR);
        check(none, what);
        snprintf(what, sizeof(what), "p=%" PRIu64 " maxR=%" PRIu64 ": no saving, still valid bounds without stage 2", c.p, maxR);
        check(b.B1 >= 11 && (b.B2 == b.B1 || b.D), what);
        continue;
      }
      snprintf(what, sizeof(what), "p=%" PRIu64 " maxR=%" PRIu64 ": 11 <= B1 <= B2 <= MAX_B2, D fits (B1=%" PRIu64 " B2=%" PRIu64 " D=%u)", c.p,
               maxR, b.B1, b.B2, b.D);
      check(b.B1 >= 11 && b.B2 >= b.B1 && b.B2 <= MAX_B2 && (b.D == 0) == (b.B2 == b.B1) && (!b.D || stage2nR(b.D) <= maxR), what);
      snprintf(what, sizeof(what), "p=%" PRIu64 " maxR=%" PRIu64 ": no stage 2 below the smallest table", c.p, maxR);
      check(maxR >= stage2nR(210) || b.B2 == b.B1, what);
      // (Not exactly:  with more memory chooseStage2D can pick a larger D whose table costs more than this model's best D, up to 0.3%
      // of the saving at 41M)
      snprintf(what, sizeof(what), "p=%" PRIu64 " maxR=%" PRIu64 ": more memory, saving not lower (%.0f < %.0f)", c.p, maxR, b.savings,
               prev.savings);
      check(!prev.B1 || b.savings >= prev.savings * (1 - 1e-2), what);
      snprintf(what, sizeof(what), "p=%" PRIu64 " maxR=%" PRIu64 ": positive saving", c.p, maxR);
      check(b.savings > 0, what);
      // Locally optimal (the choice is rounded to 3 digits, so within a small tolerance)
      for (auto [f1, f2] : {std::pair{0.8, 1.0}, {1.25, 1.0}, {1.0, 0.8}, {1.0, 1.25}, {0.8, 0.8}, {1.25, 1.25}}) {
        u64 const B1 = u64(b.B1 * f1);
        Bounds const o = evaluateBounds(c.p, c.tf, c.ts, B1, std::max(B1, u64(b.B2 * f2)), maxR);
        snprintf(what, sizeof(what), "p=%" PRIu64 " maxR=%" PRIu64 ": B1 x%.2f B2 x%.2f saves no more (%.0f vs %.0f)", c.p, maxR, f1, f2,
                 o.savings, b.savings);
        check(o.savings <= b.savings + 1e-4 * c.p, what);
      }
      prev = b;
    }
  }

  // More tests saved:  larger bounds;  deeper TF:  a smaller chance and saving
  Bounds const one = chooseBounds(110000017, 77, 1, 480), two = chooseBounds(110000017, 77, 2, 480);
  check(two.B1 > one.B1 && two.B2 > one.B2 && two.prob.total() > one.prob.total(), "testsSaved 2 > 1: larger bounds");
  Bounds const deep = chooseBounds(110000017, 80, 1, 480);
  check(deep.prob.total() < one.prob.total() && deep.savings < one.savings, "tfBits 80 > 77: smaller chance and saving");

  // No saving when P-1 can't pay:  no test saved, or TF so deep that no factor is likely
  check(chooseBounds(110000017, 77, 0, 480).savings <= 0, "testsSaved = 0: no saving");
  check(chooseBounds(110000017, 99, 0.01, 480).savings <= 0, "tfBits 99, testsSaved 0.01: no saving");
  // Stage 1 alone not paying while a stage 2 would (prpll then runs stage 1 alone, Task.cpp)
  check(chooseBounds(30000001, 72, 1, 0).savings <= 0 && chooseBounds(30000001, 72, 1, 480).savings > 0,
        "30M tf 72: only stage 1 + stage 2 pays");

  // chooseB2 for a fixed B1 (a version 1 savefile)
  Bounds const b2 = chooseB2(110000017, 77, 1, 1000000, 480);
  check(b2.B1 == 1000000 && b2.B2 > b2.B1, "chooseB2 keeps B1 and adds a stage 2");
  check(chooseB2(110000017, 77, 1, 1000000, 0).B2 == 1000000, "chooseB2 without memory: B2 = B1");
}

int main(int argc, char** argv) {
  if (argc > 1) {
    for (int i = 1; i < argc; ++i) {
      u64 p{}, maxR{}, B1{}, B2{};
      double tf{}, ts{};
      if (!strncmp(argv[i], "eval:", 5)) {
        if (sscanf(argv[i] + 5, "%" SCNu64 ",%lf,%lf,%" SCNu64 ",%" SCNu64 ",%" SCNu64, &p, &tf, &ts, &maxR, &B1, &B2) != 6 || B1 < 2) {
          printf("bad case %s\n", argv[i]);
          return 2;
        }
        print("eval", p, tf, ts, maxR, evaluateBounds(p, tf, ts, B1, B2, maxR), 0);
        continue;
      }
      if (sscanf(argv[i], "%" SCNu64 ",%lf,%lf,%" SCNu64, &p, &tf, &ts, &maxR) != 4) { printf("bad case %s\n", argv[i]); return 2; }
      double ms = 0;
      Bounds const b = timedChoice(p, tf, ts, maxR, &ms);
      print("choice", p, tf, ts, maxR, b, ms);
    }
    return 0;
  }
  testProbability();
  testPairing();
  testChoice();
  printf("%s\n", bad ? "check_bounds: FAILED" : "check_bounds: all OK");
  return bad ? 1 : 0;
}
