// The P-1 stage 1 residue computed with GMP, as an independent oracle for prpll's:  x = 3^E mod 2^p-1 with E = 2p * the product over
// the primes q <= B1 of the largest power of q that is <= B1 (Prime95's calc_exp; E is built here with GMP alone, not with
// src/Pm1.cpp).  GMP is used only by this test program; prpll does not use it.
//
//   stage1_oracle <p> <B1> [<mprime savefile>]
// prints the res64 of x and, given the P-1 savefile that prpll wrote for mprime, compares the whole residue in it with x and checks
// the file's checksum.  Exit status 0 if they agree.

#include <gmp.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

static void exponent(mpz_t e, u64 p, u64 B1) {
  mpz_set_ui(e, 2 * p);
  mpz_t q;
  mpz_init(q);
  for (mpz_set_ui(q, 2); mpz_cmp_ui(q, B1) <= 0; mpz_nextprime(q, q)) {
    u64 const v = mpz_get_ui(q);
    u64 power = v;
    while (power <= B1 / v) { power *= v; }
    mpz_mul_ui(e, e, power);
  }
  mpz_clear(q);
}

// x mod 2^p-1, for 0 <= x < 2^(2p+2)
static void reduce(mpz_t x, mpz_t t, const mpz_t M, u64 p) {
  while (mpz_sizeinbase(x, 2) > p) {
    mpz_tdiv_q_2exp(t, x, p);
    mpz_tdiv_r_2exp(x, x, p);
    mpz_add(x, x, t);
  }
  if (mpz_cmp(x, M) == 0) { mpz_set_ui(x, 0); }
}

template<typename T> static T rd(const std::vector<unsigned char>& b, size_t o) { T v; memcpy(&v, b.data() + o, sizeof(T)); return v; }

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <p> <B1> [<mprime savefile>]\n", argv[0]); return 2; }
  u64 const p = strtoull(argv[1], nullptr, 10), B1 = strtoull(argv[2], nullptr, 10);

  mpz_t E, M, x, t;
  mpz_inits(E, M, x, t, nullptr);
  exponent(E, p, B1);
  mpz_set_ui(M, 1);
  mpz_mul_2exp(M, M, p);
  mpz_sub_ui(M, M, 1);

  // Left to right, one squaring per bit, as prpll does (any correct method gives the same x)
  size_t const nBits = mpz_sizeinbase(E, 2);
  mpz_set_ui(x, 1);
  for (size_t i = nBits; i-- > 0;) {
    mpz_mul(x, x, x);
    if (mpz_tstbit(E, i)) { mpz_mul_ui(x, x, 3); }
    reduce(x, t, M, p);
  }

  // Cross-check the exponent and the residue with GMP's own powm (cheap only for small p; skipped for the larger cases)
  if (double(p) * double(nBits) < 5e10) {
    mpz_t y, three;
    mpz_inits(y, three, nullptr);
    mpz_set_ui(three, 3);
    mpz_powm(y, three, E, M);
    if (mpz_cmp(x, y)) { printf("FAIL: the square-and-multiply residue differs from mpz_powm\n"); return 1; }
    mpz_clears(y, three, nullptr);
  }

  u64 const res64 = mpz_getlimbn(x, 0);
  printf("p=%" PRIu64 " B1=%" PRIu64 " E bits=%zu res64=%016" PRIx64 "\n", p, B1, nBits, res64);

  int status = 0;
  if (argc > 3) {
    std::ifstream in(argv[3], std::ios::binary);
    std::vector<unsigned char> b{std::istreambuf_iterator<char>(in), {}};
    // header (52 bytes, checksum at 48), state, B_done, C_done, have_x, then the giant:  count and words
    size_t o = 52;
    if (b.size() < o + 4 + 8 + 8 + 4 + 4) { printf("FAIL: savefile too short\n"); return 1; }
    u32 sum = 0;
    auto const add64 = [&](u64 v) { sum += u32(v) + u32(v >> 32); };
    u32 const state = rd<u32>(b, o); o += 4; sum += state;
    u64 const Bdone = rd<u64>(b, o); o += 8; add64(Bdone);
    u64 const Cdone = rd<u64>(b, o); o += 8; add64(Cdone);
    u32 const haveX = rd<u32>(b, o); o += 4; sum += haveX;
    u32 const len = rd<u32>(b, o); o += 4; sum += len;
    if (b.size() != o + 4 * size_t(len)) { printf("FAIL: savefile size %zu, expected %zu\n", b.size(), o + 4 * size_t(len)); return 1; }
    std::vector<u32> words(len);
    for (u32 i = 0; i < len; ++i) { words[i] = rd<u32>(b, o); o += 4; sum += words[i]; }
    sum += len;
    mpz_t y;
    mpz_init(y);
    mpz_import(y, len, -1, 4, 0, 0, words.data());
    bool const okHeader = rd<u32>(b, 0) == 0x317a394b && rd<u32>(b, 4) == 8 && rd<u32>(b, 20) == p;
    bool const okFields = state == 5 && Bdone == B1 && Cdone == B1 && haveX == 1;
    bool const okSum = rd<u32>(b, 48) == sum;
    bool const okX = mpz_cmp(x, y) == 0;
    printf("savefile %s: header %s, fields %s, checksum %s, residue %s (res64 %016" PRIx64 ")\n", argv[3], okHeader ? "ok" : "BAD",
           okFields ? "ok" : "BAD", okSum ? "ok" : "BAD", okX ? "EQUAL" : "DIFFERENT", u64(mpz_getlimbn(y, 0)));
    status = okHeader && okFields && okSum && okX ? 0 : 1;
    mpz_clear(y);
  }
  mpz_clears(E, M, x, t, nullptr);
  return status;
}
