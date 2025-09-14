// Copyright 2017 Mihai Preda.

#include "state.h"
#include "shared.h"
#include "log.h"
#include "timeutil.h"

#include <cassert>
#include <cmath>

static i64 lowBits(i64 u, int bits) { return (u << (64 - bits)) >> (64 - bits); }

static uWord unbalance(Word w, int nBits, int *carry) {
  assert(*carry == 0 || *carry == -1);
  w += *carry;
  *carry = 0;
  if (w < 0) {
    w += ((Word) 1 << nBits);
    *carry = -1;
  }
  if (!(0 <= w && w < ((Word) 1 << nBits))) { log("w=%lX, nBits=%d\n", (long) w, nBits); }
  assert(0 <= w && w < ((Word) 1 << nBits));
  return w;
}

std::vector<u32> compactBits(const vector<Word> &dataVect, u32 E) {
  if (dataVect.empty()) { return {}; } // Indicating all zero

  std::vector<u32> out;
  out.reserve((E - 1) / 32 + 1);

  u32 N = dataVect.size();
  const Word *data = dataVect.data();

  int carry = 0;
  u32 outWord = 0;
  int haveBits = 0;

  for (u32 p = 0; p < N; ++p) {
    int nBits = bitlen(N, E, p);
    uWord w = unbalance(data[p], nBits, &carry);

    assert(nBits > 0);
    assert(w < ((uWord) 1 << nBits));
    assert(haveBits < 32);

    while (nBits) {
      int topBits = 32 - haveBits;
      outWord |= w << haveBits;
      if (nBits >= topBits) {
        w >>= topBits;
        nBits -= topBits;
        out.push_back(outWord);
        outWord = 0;
        haveBits = 0;
      } else {
        haveBits += nBits;
        w >>= nBits;
	break;
      }
    }
  }

  assert(haveBits);
  out.push_back(outWord);

  for (int p = 0; carry; ++p) {
    i64 v = i64(out[p]) + carry;
    out[p] = v & 0xffffffff;
    carry = v >> 32;
  }

  assert(out.size() == (E - 1) / 32 + 1);
  return out;
}

struct BitBucket {
  u128 bits;
  u32 size;

  BitBucket() : bits(0), size(0) {}

  void put32(u32 b) {
    assert(size <= 96);
    bits += (u128(b) << size);
    size += 32;
  }
  
  i64 popSigned(u32 n) {
    assert(size >= n);
    i64 b = lowBits((i64) bits, n);
    size -= n;
    bits >>= n;
    bits += (b < 0); // carry fixup.
    return b;
  }
};

vector<Word> expandBits(const vector<u32> &compactBits, u32 N, u32 E) {
  assert(E % 32 != 0);

  std::vector<Word> out(N);
  Word *data = out.data();
  BitBucket bucket;
  
  auto it = compactBits.cbegin();
  [[maybe_unused]] auto itEnd = compactBits.cend();
  for (u32 p = 0; p < N; ++p) {
    u32 len = bitlen(N, E, p);
    while (bucket.size < len) {
      assert(it != itEnd);
      bucket.put32(*it++);
    }
    data[p] = (Word) bucket.popSigned(len);
  }
  assert(it == itEnd);
  assert(bucket.size == 32 - E % 32);
  assert(bucket.bits == 0 || bucket.bits == 1);
  data[0] += bucket.bits; // carry wrap-around.
  return out;
}
