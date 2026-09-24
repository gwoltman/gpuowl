// Copyright (C) Mihai Preda

#pragma once

#include <bitset>
#include "common.h"

class Primes {
  // Sieve of odd numbers 3..100001. The largest prime in it is 99991, so the
  // sieve alone gives O(1) trial division for n <= 99991^2 == 9,998,200,081;
  // isPrimeOdd() falls back to plain trial division beyond that bound.
  std::bitset<50000> sieve;
  [[nodiscard]] bool isPrimeOdd(u64 n) const;

public:
  Primes();

  [[nodiscard]] bool isPrime(u64 n) const;
  [[nodiscard]] u64 prevPrime(u64 n) const;
  [[nodiscard]] u64 nextPrime(u64 n) const;
  [[nodiscard]] u64 nearestPrime(u64 n) const;
};
