// Copyright (C) Mihai Preda

#pragma once

#include <bitset>
#include "common.h"

class Primes {
  std::bitset<50000> sieve;           // Allows for testing prims up to 10 billion
  [[nodiscard]] bool isPrimeOdd(u64 n) const;

public:
  Primes();

  [[nodiscard]] bool isPrime(u64 n) const;
  [[nodiscard]] u64 prevPrime(u64 n) const;
  [[nodiscard]] u64 nextPrime(u64 n) const;
  [[nodiscard]] u64 nearestPrime(u64 n) const;
};
