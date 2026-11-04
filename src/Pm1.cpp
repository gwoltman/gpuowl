// Copyright (C) Mihai Preda and George Woltman

#include "Pm1.h"
#include "File.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace pm1 {

namespace {

// A natural number as little-endian 32-bit words with no leading zero word (zero is the empty vector).
using Nat = vector<u32>;

void trim(Nat& a) {
  while (!a.empty() && a.back() == 0) { a.pop_back(); }
}

Nat mulSchool(const Nat& a, const Nat& b) {
  if (a.empty() || b.empty()) { return {}; }
  Nat r(a.size() + b.size(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    u64 carry = 0;
    u64 const ai = a[i];
    for (size_t j = 0; j < b.size(); ++j) {
      u64 const t = ai * b[j] + r[i + j] + carry;
      r[i + j] = u32(t);
      carry = t >> 32;
    }
    r[i + b.size()] = u32(carry);
  }
  trim(r);
  return r;
}

// r += x << (32 * shift)
void addShifted(Nat& r, const Nat& x, size_t shift) {
  if (r.size() < x.size() + shift + 1) { r.resize(x.size() + shift + 1, 0); }
  u64 carry = 0;
  size_t i = 0;
  for (; i < x.size(); ++i) {
    u64 const t = u64(r[i + shift]) + x[i] + carry;
    r[i + shift] = u32(t);
    carry = t >> 32;
  }
  for (; carry; ++i) {
    u64 const t = u64(r[i + shift]) + carry;
    r[i + shift] = u32(t);
    carry = t >> 32;
  }
}

// r -= x; requires r >= x
void subtract(Nat& r, const Nat& x) {
  i64 borrow = 0;
  size_t i = 0;
  for (; i < x.size(); ++i) {
    i64 const t = i64(r[i]) - i64(x[i]) - borrow;
    borrow = t < 0;
    r[i] = u32(t < 0 ? t + (i64(1) << 32) : t);
  }
  for (; borrow; ++i) {
    i64 const t = i64(r[i]) - borrow;
    borrow = t < 0;
    r[i] = u32(t < 0 ? t + (i64(1) << 32) : t);
  }
  trim(r);
}

Nat mul(const Nat& a, const Nat& b) {
  constexpr size_t KARATSUBA = 40;
  if (a.size() < KARATSUBA || b.size() < KARATSUBA) { return mulSchool(a, b); }

  size_t const m = std::max(a.size(), b.size()) / 2;
  auto const low  = [m](const Nat& x) { Nat r(x.begin(), x.begin() + long(std::min(m, x.size()))); trim(r); return r; };
  auto const high = [m](const Nat& x) { return x.size() > m ? Nat(x.begin() + long(m), x.end()) : Nat{}; };

  Nat const a0 = low(a), a1 = high(a), b0 = low(b), b1 = high(b);
  Nat const z0 = mul(a0, b0);
  Nat const z2 = mul(a1, b1);

  Nat sa = a0;  addShifted(sa, a1, 0);  trim(sa);
  Nat sb = b0;  addShifted(sb, b1, 0);  trim(sb);
  Nat z1 = mul(sa, sb);
  subtract(z1, z0);
  subtract(z1, z2);

  Nat r;
  addShifted(r, z0, 0);
  addShifted(r, z1, m);
  addShifted(r, z2, 2 * m);
  trim(r);
  return r;
}

}  // namespace

vector<u32> stage1Exponent(u64 p, u64 B1) {
  // Leaves: 2p, then the prime powers packed into 64 bits each.  Merging the leaves pairwise (a balanced tree) keeps the operands
  // of every multiplication about equal in size.
  vector<Nat> level;
  auto const leaf = [](u64 v) { Nat r{u32(v), u32(v >> 32)}; trim(r); return r; };

  level.push_back(leaf(2 * p));

  u64 pack = 1;
  forEachPrime(2, B1, [&](u64 q) {
    u64 power = q;
    u64 const max = B1 / q;
    while (power <= max) { power *= q; }                // the largest power of q that is <= B1
    if (pack > UINT64_MAX / power) {
      level.push_back(leaf(pack));
      pack = power;
    } else {
      pack *= power;
    }
  });
  if (pack != 1) { level.push_back(leaf(pack)); }

  while (level.size() > 1) {
    vector<Nat> next;
    next.reserve((level.size() + 1) / 2);
    for (size_t i = 0; i + 1 < level.size(); i += 2) { next.push_back(mul(level[i], level[i + 1])); }
    if (level.size() % 2) { next.push_back(std::move(level.back())); }
    level = std::move(next);
  }
  return level.front();
}

u64 bitLength(const vector<u32>& words) {
  if (words.empty()) { return 0; }
  u64 bits = 32 * (words.size() - 1);
  for (u32 top = words.back(); top; top >>= 1) { ++bits; }
  return bits;
}

bool testBit(const vector<u32>& words, u64 i) {
  return i / 32 < words.size() && ((words[i / 32] >> (i % 32)) & 1);
}

string mprimeSaveName(u64 p) {
  char buf[48];
  if (p < 80000000) {
    snprintf(buf, sizeof(buf), "p%07" PRIu64, p % 10000000);
    if (p >= 10000000) { buf[1] = char('A' + (p / 1000000) - 10); }                           // buf[1] ranges from A-Y
    if (p >= 35000000) { char const c = buf[1]; buf[1] = buf[2]; buf[2] = char(c - 25); }     // buf[2] ranges from A-Z
    if (p >= 61000000) { char const c = buf[2]; buf[2] = buf[3]; buf[3] = char(c - 25); }     // buf[3] ranges from B-T
  } else {
    snprintf(buf, sizeof(buf), "p%" PRIu64, p);
  }
  buf[0] = 'm';                                                                               // a P-1 savefile
  return buf;
}

namespace {

// The bytes of a Prime95 save file (little-endian) and its running checksum, as write_uint32 / write_uint64 / write_giant sum them.
class SaveBytes {
  vector<char> bytes;
  u32 sum{};

  void raw(const void* p, size_t n) { bytes.insert(bytes.end(), static_cast<const char*>(p), static_cast<const char*>(p) + n); }

public:
  void headerU32(u32 x)       { raw(&x, 4); }       // header fields are not part of the checksum
  void headerDouble(double x) { raw(&x, 8); }
  void headerBytes(const void* p, size_t n) { raw(p, n); }

  void u32v(u32 x) { raw(&x, 4); sum += x; }
  void u64v(u64 x) { raw(&x, 8); sum += u32((x >> 32) + x); }

  // write_giant: the word count (summed by write_long, and again explicitly), then the words, each summed
  void giant(const vector<u32>& g) {
    u32 const len = u32(g.size());
    u32v(len);
    raw(g.data(), g.size() * 4);
    sum += len;
    for (u32 w : g) { sum += w; }
  }

  static constexpr size_t CHECKSUM_OFFSET = 48;
  void finish() { std::memcpy(bytes.data() + CHECKSUM_OFFSET, &sum, 4); }
  const vector<char>& data() const { return bytes; }
};

}  // namespace

void writeMprimeStage1(const fs::path& dir, u64 p, u64 B1, const vector<u32>& x) {
  constexpr u32 MAGIC = 0x317a394b;            // PM1_MAGICNUM
  constexpr u32 VERSION = 8;                   // PM1_VERSION of Prime95 30.10 to 31.06
  constexpr u32 STATE_DONE = 5;                // PM1_STATE_DONE

  vector<u32> g = x;
  trim(g);
  if (g.empty()) { throw "mprime savefile: the stage 1 residue is zero"; }

  SaveBytes f;
  // write_header: magic, version, K, B, N, C, 10 stage characters and 2 pad bytes, percent complete, checksum slot
  f.headerU32(MAGIC);
  f.headerU32(VERSION);
  f.headerDouble(1.0);                         // K
  f.headerU32(2);                              // B
  f.headerU32(u32(p));                         // N
  f.headerU32(u32(i32(-1)));                   // C
  char stage[12] = "S5";
  f.headerBytes(stage, 12);
  f.headerDouble(1.0);                         // percent complete
  f.headerU32(0);                              // checksum, filled in by finish()

  f.u32v(STATE_DONE);
  f.u64v(B1);                                  // B_done
  f.u64v(B1);                                  // C_done:  stage 2 has not started
  f.u32v(1);                                   // have_x
  f.giant(g);
  f.finish();

  fs::path const path = dir / mprimeSaveName(p);
  fs::path const tmp = fs::path(path.string() + ".new");
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    out.write(f.data().data(), std::streamsize(f.data().size()));
    out.flush();
    if (!out) { throw "mprime savefile: can't write " + tmp.string(); }
  }
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) { throw "mprime savefile: can't rename to " + path.string() + ": " + ec.message(); }
}

string mprimePminus1Line(const string& AID, u64 p, u64 B1, u64 B2) {
  string line = "Pminus1=";
  if (!AID.empty() && AID != "N/A") { line += AID + ','; }
  line += "1,2," + to_string(p) + ",-1," + to_string(B1) + ',' + to_string(B2) + '\n';
  return line;
}

void appendMprimeWork(const fs::path& dir, const string& line) {
  File::append(dir / "worktodo.add", line);
}

}  // namespace pm1
