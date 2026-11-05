// Copyright 2017 Mihai Preda.

#include "state.h"
#include <cassert>

static i64 lowBits(i64 u, int bits) { return (u << (64 - bits)) >> (64 - bits); }

std::vector<u32> compactBits(const vector<Word> &dataVect, u64 E) {
  if (dataVect.empty()) { return {}; } // Indicating all zero

  u32 const N = u32(dataVect.size());
  const Word *data = dataVect.data();

  std::vector<u32> out;
  out.reserve((E - 1) / 32 + 1);

  int carry = 0;
  u32 outWord = 0;
  int haveBits = 0;

  // Convert to compact form
  for (u32 p = 0; p < N; ++p) {
    int nBits = bitlen(N, E, p);
    assert(nBits > 0);

    //   Be careful adding in the carry -- it could overflow a 32-bit word.  Convert value into desired unsigned range.
    i64 const tmp = (i64) data[p] + carry;
    carry = (int) (tmp >> nBits);
    u64 w = (u64) (tmp - ((i64) carry << nBits));
    assert(w < (1ULL << nBits));

    assert(haveBits < 32);
    while (nBits) {
      int const needBits = 32 - haveBits;
      outWord |= w << haveBits;
      if (nBits >= needBits) {
        w >>= needBits;
        nBits -= needBits;
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
    i64 const v = i64(out[p]) + carry;
    out[p] = v & 0xffffffff;
    carry = v >> 32;
  }

  assert(out.size() == (E - 1) / 32 + 1);
  return out;
}

struct BitBucket {
  u128 bits{0};
  u32 size{0};

  BitBucket()  = default;

  void put32(u32 b) {
    assert(size <= 96);
    bits += (u128(b) << size);
    size += 32;
  }
  
  i64 popSigned(u32 n) {
    assert(size >= n);
    i64 const b = lowBits((i64) bits, n);
    size -= n;
    bits >>= n;
    bits += (b < 0); // carry fixup.
    return b;
  }
};

vector<Word> expandBits(const vector<u32> &compactBits, u32 N, u64 E) {
  assert(E % 32 != 0);

  std::vector<Word> out(N);
  Word *data = out.data();
  BitBucket bucket;
  
  auto it = compactBits.cbegin();
  [[maybe_unused]] auto itEnd = compactBits.cend();
  for (u32 p = 0; p < N; ++p) {
    u32 const len = bitlen(N, E, p);
    while (bucket.size < len) {
      assert(it != itEnd);
      bucket.put32(*it++);
    }
    data[p] = (Word) bucket.popSigned(len);
  }
  assert(it == itEnd);
  assert(bucket.size == 32 - E % 32);
  assert(bucket.bits == 0 || bucket.bits == 1);
  data[0] += u32(bucket.bits); // carry wrap-around.
  return out;
}

vector<u32> rotateRight(const vector<u32>& x, u64 n, u64 E) {
  if (x.empty()) { return {}; }
  assert(E % 32 != 0);
  u32 const nW = u32((E - 1) / 32 + 1);
  u32 const topBits = E % 32;

  // Reduce to E bits: compactBits() can leave a carry just above bit E-1, which is worth 1 (2^E == 1).
  vector<u32> v = x;
  v.resize(nW);
  while (u32 const over = v[nW - 1] >> topBits) {
    v[nW - 1] &= (1u << topBits) - 1;
    u64 carry = over;
    for (u32 i = 0; carry && i < nW; ++i) {
      u64 const t = u64(v[i]) + carry;
      v[i] = u32(t);
      carry = t >> 32;
    }
  }

  n %= E;
  if (n == 0) { return v; }

  // The 32 bits of v starting at bit pos, wrapping from bit E-1 to bit 0.
  auto bitsAt = [&](u64 pos) -> u32 {
    auto straight = [&](u64 p) -> u64 {   // 32 or more bits from p, may run past E-1 into the zero top bits
      u32 const w = u32(p / 32), b = u32(p % 32);
      u64 r = v[w] >> b;
      if (w + 1 < nW) { r |= u64(v[w + 1]) << (32 - b); }
      return r;
    };
    if (pos + 32 <= E) { return u32(straight(pos)); }
    u32 const lowLen = u32(E - pos);      // 1..31 bits up to the top, then wrap to bit 0
    u64 const low = straight(pos) & ((u64(1) << lowLen) - 1);
    return u32(low | (straight(0) << lowLen));
  };

  vector<u32> out(nW);
  u64 pos = n;
  for (u32 i = 0; i < nW; ++i) {
    out[i] = bitsAt(pos);
    pos += 32;
    if (pos >= E) { pos -= E; }
  }
  out[nW - 1] &= (1u << topBits) - 1;
  return out;
}
