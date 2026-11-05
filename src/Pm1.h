// Copyright (C) Mihai Preda and George Woltman

#pragma once

// P-1 helpers that need no bignum library:  a prime sieve, the exponent of P-1 stage 1, and the files by which a stage 1
// result is handed to mprime (a Prime95 P-1 savefile and a "Pminus1=" line in mprime's worktodo.add).

#include "common.h"

#include <algorithm>
#include <cmath>

namespace pm1 {

// Calls fn(q) for every prime q in [lo, hi], in increasing order.  A segmented sieve of the odd numbers: memory is about sqrt(hi)
// bits for the sieving primes plus one segment, however large hi is.
template<typename Fn>
void forEachPrime(u64 lo, u64 hi, Fn&& fn) {
  if (hi < 2 || hi < lo) { return; }
  if (lo < 2) { lo = 2; }

  u64 root = u64(std::sqrt(double(hi)));
  while (root * root > hi) { --root; }
  while ((root + 1) * (root + 1) <= hi) { ++root; }

  // The odd primes up to sqrt(hi)
  vector<u32> base;
  {
    vector<bool> composite(root + 1);
    for (u64 i = 2; i <= root; ++i) {
      if (composite[i]) { continue; }
      if (i > 2) { base.push_back(u32(i)); }
      for (u64 j = i * i; j <= root; j += i) { composite[j] = true; }
    }
  }

  if (lo <= 2) { fn(u64(2)); }

  constexpr u64 SEGMENT = u64(1) << 20;                // odd numbers per segment
  vector<char> composite(SEGMENT);
  u64 first = lo < 3 ? 3 : lo;
  if (first % 2 == 0) { ++first; }                      // the first odd number to look at

  for (u64 start = first; start <= hi; start += 2 * SEGMENT) {
    u64 const n = std::min(SEGMENT, (hi - start) / 2 + 1);      // the odd numbers start, start+2, ..., start+2(n-1)
    u64 const last = start + 2 * (n - 1);
    std::fill(composite.begin(), composite.begin() + long(n), char(0));
    for (u32 q : base) {
      u64 const qq = u64(q) * q;
      if (qq > last) { break; }
      u64 m = qq >= start ? qq : ((start + q - 1) / q) * q;      // the first multiple of q that is at least qq and at least start
      if (m % 2 == 0) { m += q; }                                // an odd multiple
      for (u64 j = (m - start) / 2; j < n; j += q) { composite[j] = 1; }
    }
    for (u64 i = 0; i < n; ++i) {
      if (!composite[i]) { fn(start + 2 * i); }
    }
  }
}

// The exponent of stage 1:  E = 2p * the product, over the primes q <= B1, of the largest power of q that is <= B1 (the same
// definition as Prime95's calc_exp, which includes 2p because the factors of 2^p-1 are 1 mod 2p).  Little-endian 32-bit words,
// no leading zero word.  Built with a balanced product tree, so it takes seconds even for B1 in the hundreds of millions.
vector<u32> stage1Exponent(u64 p, u64 B1);

// stage1Exponent(p, B1), kept in <dir>/B1-<B1>.exp so that a restart does not compute it again (seconds at B1 = 10M, about a
// minute at 50M).  A missing or damaged file is recomputed and written again.
vector<u32> stage1ExponentCached(const fs::path& dir, u64 p, u64 B1);

u64 bitLength(const vector<u32>& words);
bool testBit(const vector<u32>& words, u64 i);

// acc += the number formed by bits [lo, lo + len) of words.  acc is little-endian 32-bit words and grows as needed.
void addBits(vector<u32>& acc, const vector<u32>& words, u64 lo, u32 len);

// acc >> shift, which must fit in 64 bits.
u64 shiftRight64(const vector<u32>& acc, u32 shift);

// The decimal digits of a number given as little-endian 32-bit words (by repeated division by 10^9: quadratic, meant for factors).
string toDecimal(const vector<u32>& words);

// Whether the number in words (little-endian 32-bit words) is 1.
bool isOne(const vector<u32>& words);

// For a factor found by the GCD:  0 if it is certainly composite, 1 if it is prime or a strong probable prime to the bases 3 .. 53
// (Miller-Rabin; base 2 tells nothing for a divisor of 2^p - 1, where 2^p = 1), -1 if it has more than 126 bits (not tested).
int isProbablePrime(const vector<u32>& words);

// Prime95 file name of the P-1 savefile of exponent p (commonc.c, tempFileName):  "m<p>" from 80,000,000 up, an encoded name below.
string mprimeSaveName(u64 p);

// Writes <dir>/<mprimeSaveName(p)> as a Prime95 P-1 savefile (version 8) in the state "P-1 done" with B_done = C_done = B1, holding
// x, the stage 1 residue of 2^p-1 as little-endian 32-bit words.  Written to a ".new" file and renamed.
void writeMprimeStage1(const fs::path& dir, u64 p, u64 B1, const vector<u32>& x);

// "Pminus1=[AID,]1,2,p,-1,B1,B2\n"
string mprimePminus1Line(const string& AID, u64 p, u64 B1, u64 B2);

// Appends a line to <dir>/worktodo.add in one write.
void appendMprimeWork(const fs::path& dir, const string& line);

// ---- Stage 2 (paired "n^2" form, with a table of n^2 for every odd n < M coprime to D) ----
// The table holds B_n = x^(n^2) for the n in S = {n < M : gcd(n, D) = 1} (D even, so every n in S is odd), and the run steps
// A_k = x^((kD)^2).  Since (kD)^2 - n^2 = (kD - n)(kD + n), one accumulator multiplication by A_k - B_n ("an item (k, n)") covers both
// kD - n and kD + n.  For M = D/2 this is the old plan:  every prime q has exactly one (k, n), k = round(q / D).  A larger M lets a
// prime pair with any prime of the opposite class mod 2D at a distance below 2M:  two primes q1 > q2 share an item exactly when
//   q1 + q2 = 0 (mod 2D)  and  q1 - q2 < 2M      (then k = (q1 + q2) / 2D and n = (q1 - q2) / 2, and n is in S: gcd(n, D) =
//                                                  gcd(q1 - kD, D) = 1 as q1 is a prime above every prime of D)
// so the n of a pair may be above D/2:  its two primes are in different D-sections ((k + s)D + r and (k - s)D - r for n = sD + r).
// A prime without a partner is a "single":  item (round(q / D), |q - round(q / D) D|), whose n < D/2 is in S as M >= D/2.
//
// Pairing (planWindow):  the primes q with q = c or q = -c (mod 2D) form a "class pair"; a pair joins the two classes of one class pair.
// Each class pair is a bipartite graph whose edges join the primes of opposite classes at a distance below 2M, i.e. each prime's
// neighbours are an interval of the other class:  matching a new prime with the oldest pending prime of the other class that is still
// in reach (and making a pending prime a single once it is out of reach of every later prime) is a maximum matching (the
// earliest-deadline rule for convex bipartite graphs), so it gives the fewest items.  The sweep runs on windows of kWindow k's,
// independently of each other, so that any window's items can be made again (resume, rollback) without the windows before it; a pair is
// never formed across two windows (window edges cost under 0.1% of the pairs:  a window is at least 512 * 2M wide).
struct Stage2Plan {
  static constexpr u32 VERSION = 2;    // of the pairing:  a savefile of another version is not resumed (version 1 is M = D/2:  the same)

  u64 B1{}, B2{};
  u32 D{}, M{};
  vector<u32> ns;              // S, increasing
  vector<u32> indexOf;         // odd n < M -> its position in ns (at (n - 1) / 2), or u32(-1) when gcd(n, D) > 1
  u64 kBegin{}, kEnd{};        // every item has its k in [kBegin, kEnd)
  u64 kWindow{};               // k per window
  vector<u32> windowNeeded;    // the items of each window
  u64 nPrimes{};               // primes in (B1, B2]
  u64 nPairs{};                // items that cover two primes of (B1, B2]
  u64 nNeeded{};               // items (accumulator multiplications):  nPrimes - nPairs
  u64 nCross{};                // pairs with n > D/2 (their primes are in different D-sections)
  u64 nZero{};                 // items with k = 0 (A_0 = 1)

  [[nodiscard]] u64 nWindows() const { return windowNeeded.size(); }
  [[nodiscard]] u64 windowOf(u64 k) const { return (k - kBegin) / kWindow; }
  [[nodiscard]] u32 nTable() const { return u32(ns.size()); }
};

// The items of one window:  for k in [kLo, kHi) the table indices start[k - kLo] .. start[k - kLo + 1] - 1 of idx
struct Stage2Window {
  u64 w = u64(-1);
  u64 kLo{}, kHi{};
  vector<u32> start;
  vector<u32> idx;
  // Optional counts of planWindow
  u64 nPrimes{}, nPairs{}, nCross{}, nZero{};

  [[nodiscard]] u32 count(u64 k) const { return start[k - kLo + 1] - start[k - kLo]; }
  [[nodiscard]] const u32* items(u64 k) const { return idx.data() + start[k - kLo]; }
};

// The items of window w of the plan (deterministic:  depends on B1, B2, D, M and kWindow only)
void planWindow(const Stage2Plan& plan, u64 w, Stage2Window& out);

// The D for which stage 2 is planned (even, 11-smooth); D can run when every prime factor of D is at most B1 (the stage 1 exponent holds
// them, and a prime of (B1, B2] dividing D could not be covered)
const vector<u32>& stage2Ds();
bool stage2DFits(u32 D, u64 B1);
// |S| for (D, M);  phi(D) / 2 for M = D/2
u32 stage2TableSize(u32 D, u32 M);
// The largest M >= D/2 with stage2TableSize(D, M) <= n (M is then just above an element of S, or D/2), 0 if none
u32 stage2MaxM(u32 D, u64 n);

// Requires B1 >= 3, B1 < B2 <= 2^32 - 8192, D even and stage2DFits, D/2 <= M.  kWindow = 0:  the default (at least 2^22 / D, and
// at least 512 * 2M / D).  Sieves (B1, B2] once (one pass of planWindow over every window).
Stage2Plan planStage2(u64 B1, u64 B2, u32 D, u32 M, u64 kWindow = 0);

}
