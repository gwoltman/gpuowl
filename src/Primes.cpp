// Copyright (C) Mihai Preda

#include "Primes.h"

#include <cassert>

Primes::Primes() {
  sieve.set();

  for (u32 i = 0; i < sieve.size(); ++i) {
    if (sieve[i]) {
      u32 const n = 2 * i + 3;
      for (u32 k = i + n; k < sieve.size(); k += n) { sieve.reset(k); }
    }
  }
}

bool Primes::isPrimeOdd(u64 n) const {
  assert(n % 2); // must be odd to call here

  if (n < 3) { return false; }
  u64 lastP = 0;
  for (u32 k = 0; k < sieve.size(); ++k) {
    if (sieve[k]) {
      u32 const p = k * 2 + 3;
      if (u64(p) * u64(p) > n) { return true; }
      if (n % p == 0) { return false; }
      lastP = p;
    }
  }

  // n is beyond the sieve's fast-path bound (lastP^2): none of the sieved
  // primes divide n, but we haven't reached sqrt(n) yet. This is rare (n
  // above ~9.998e9) and not a hot path, so fall back to plain trial
  // division by the remaining odd numbers instead of assuming n is prime.
  for (u64 p = lastP + 2; p <= n / p; p += 2) {
    if (n % p == 0) { return false; }
  }
  return true;
}

bool Primes::isPrime(u64 n) const {
  return (n%2 && isPrimeOdd(n)) || (n == 2);
}

u64 Primes::prevPrime(u64 n) const {
  --n;
  if (n % 2 == 0) { --n; }

  for (; n >= 2; n -= 2) { if (isPrimeOdd(n)) { return n; } }
  assert(false);
  return 0;
}

u64 Primes::nextPrime(u64 n) const {
  ++n;
  if (n % 2 == 0) { ++n; }
  for (; ; n += 2) { if (isPrimeOdd(n)) { return n; }}
  assert(false);
  return 0;
}

u64 Primes::nearestPrime(u64 n) const {
  if (isPrime(n)) { return n; }
  u64 const a = prevPrime(n);
  u64 const b = nextPrime(n);
  assert(a < n && n < b);
  return n-a < b-n ? a : b;
}
