// Copyright 2017 Mihai Preda.

#pragma once

#include "common.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <cfenv>

vector<u32> compactBits(const vector<Word> &dataVect, u64 E);
vector<Word> expandBits(const vector<u32> &compactBits, u32 N, u64 E);

// Rewrite balanced words (each in [-2^(b-1), 2^(b-1))) the way the kernels with ZEROMEAN do (carryutil.cl): the word at index p whose tie
// bit zmLb(p) is 1 is in (-2^(b-1), 2^(b-1)] instead.  The value mod 2^E - 1 does not change.
void zeroMeanWords(vector<Word>& words, u64 E);

constexpr u32 step(u32 N, u64 E) { return N - (E % N); }
constexpr u32 extra(u32 N, u64 E, u32 k) { return u64(step(N, E)) * k % N; }
constexpr bool isBigWord(u32 N, u64 E, u32 k) { return extra(N, E, k) + step(N, E) < N; }
constexpr u32 bitlen(u32 N, u64 E, u32 k) { return u32(E / N) + isBigWord(N, E, k); }
