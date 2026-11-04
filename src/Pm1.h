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

u64 bitLength(const vector<u32>& words);
bool testBit(const vector<u32>& words, u64 i);

// Prime95 file name of the P-1 savefile of exponent p (commonc.c, tempFileName):  "m<p>" from 80,000,000 up, an encoded name below.
string mprimeSaveName(u64 p);

// Writes <dir>/<mprimeSaveName(p)> as a Prime95 P-1 savefile (version 8) in the state "P-1 done" with B_done = C_done = B1, holding
// x, the stage 1 residue of 2^p-1 as little-endian 32-bit words.  Written to a ".new" file and renamed.
void writeMprimeStage1(const fs::path& dir, u64 p, u64 B1, const vector<u32>& x);

// "Pminus1=[AID,]1,2,p,-1,B1,B2\n"
string mprimePminus1Line(const string& AID, u64 p, u64 B1, u64 B2);

// Appends a line to <dir>/worktodo.add in one write.
void appendMprimeWork(const fs::path& dir, const string& line);

}
