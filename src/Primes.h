// Copyright (C) Mihai Preda

#pragma once

#include <bitset>
#include "common.h"

class Primes {
  std::bitset<50000> sieve;           // Allows for testing prims up to 10 billion
  bool isPrimeOdd(u64 n) const;

public:
  Primes();

  bool isPrime(u64 n) const;
  u64 prevPrime(u64 n) const;
  u64 nextPrime(u64 n) const;
  u64 nearestPrime(u64 n) const;
};
