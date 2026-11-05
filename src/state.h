// Copyright 2017 Mihai Preda.

#pragma once

#include "common.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <cfenv>

vector<u32> compactBits(const vector<Word> &dataVect, u64 E);
vector<Word> expandBits(const vector<u32> &compactBits, u32 N, u64 E);

// x * 2^-n mod 2^E - 1 (rotate right by n) and x * 2^n mod 2^E - 1 (rotate left), x in the compact form above.
// The all-zero empty vector (see Gpu::readChecked) stays empty.
vector<u32> rotateRight(const vector<u32>& x, u64 n, u64 E);
inline vector<u32> rotateLeft(const vector<u32>& x, u64 n, u64 E) { return rotateRight(x, E - n % E, E); }

constexpr u32 step(u32 N, u64 E) { return N - (E % N); }
constexpr u32 extra(u32 N, u64 E, u32 k) { return u64(step(N, E)) * k % N; }
constexpr bool isBigWord(u32 N, u64 E, u32 k) { return extra(N, E, k) + step(N, E) < N; }
constexpr u32 bitlen(u32 N, u64 E, u32 k) { return u32(E / N) + isBigWord(N, E, k); }
