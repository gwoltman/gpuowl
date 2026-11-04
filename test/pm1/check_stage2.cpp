// External oracle for P-1 stage 2 (src/Stage2.cpp), GMP only; not linked into prpll.
//
//   check_stage2 random p seed x.bin                 a random residue mod 2^p-1
//   check_stage2 stage1 p B1 x.bin                   x = 3^E mod 2^p-1, E = 2p * prod over primes q <= B1 of the largest q^e <= B1
//   check_stage2 acc p B1 B2 D x.bin acc.bin         acc.bin == prod over the (k, r) covering a prime of x^((kD)^2) - x^(r^2) mod 2^p-1?
//                                                    (the (k, r) set derived here from mpz_nextprime, independently of src/Pm1.cpp)
//   check_stage2 factors p B1 B2 x.bin acc.bin f...  for each factor f of 2^p-1:  f | gcd(acc, 2^p-1) exactly when x mod f has order 1
//                                                    or a prime order in (B1, B2]  (the exact condition, tested with GMP on x mod f)
//   check_stage2 search p kmax B1 B2                 factors f = 2kp+1 of 2^p-1 with k <= kmax, each with the order of 3^E mod f and
//                                                    whether stage 1 (B1) or stage 2 (B1, B2) finds it
//   check_stage2 plan B1 B2 D                        src/Pm1.cpp's planStage2 counts against a count made here
// Exit status 0 when every check passed.

#include "Pm1.h"

#include <gmp.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <set>
#include <string>
#include <vector>

using namespace std;

static void mersenne(mpz_t N, u64 p) { mpz_set_ui(N, 0); mpz_setbit(N, p); mpz_sub_ui(N, N, 1); }

static void readWords(mpz_t z, const char* name, u64 p) {
  size_t const n = (p - 1) / 32 + 1;
  vector<u32> w(n);
  FILE* f = fopen(name, "rb");
  if (!f || fread(w.data(), 4, n, f) != n) { fprintf(stderr, "can't read %s\n", name); exit(2); }
  fclose(f);
  mpz_import(z, n, -1, 4, 0, 0, w.data());
}

static void writeWords(const mpz_t z, const char* name, u64 p) {
  size_t const n = (p - 1) / 32 + 1;
  vector<u32> w(n, 0);
  size_t count = 0;
  mpz_export(w.data(), &count, -1, 4, 0, 0, z);
  FILE* f = fopen(name, "wb");
  if (!f || fwrite(w.data(), 4, n, f) != n) { fprintf(stderr, "can't write %s\n", name); exit(2); }
  fclose(f);
}

// E = 2p * prod q^floor(log_q B1)
static void stage1Exp(mpz_t E, u64 p, u64 B1) {
  mpz_set_ui(E, 2 * p);
  mpz_t q;
  mpz_init_set_ui(q, 1);
  while (true) {
    mpz_nextprime(q, q);
    u64 const qq = mpz_get_ui(q);
    if (qq > B1) { break; }
    u64 pw = qq;
    while (pw <= B1 / qq) { pw *= qq; }
    mpz_mul_ui(E, E, pw);
  }
  mpz_clear(q);
}

static vector<u64> primesIn(u64 lo, u64 hi) {      // (lo, hi]
  vector<u64> r;
  mpz_t q;
  mpz_init_set_ui(q, lo);
  while (true) {
    mpz_nextprime(q, q);
    if (mpz_cmp_ui(q, hi) > 0) { break; }
    r.push_back(mpz_get_ui(q));
  }
  mpz_clear(q);
  return r;
}

static u64 gcd64(u64 a, u64 b) { while (b) { u64 t = a % b; a = b; b = t; } return a; }

// The (k, r) covering the primes of (B1, B2]:  k = nearest multiple index, r = |q - kD|.  Also counts primes and pairs.
static set<pair<u64, u64>> pairsFor(u64 B1, u64 B2, u64 D, u64* nPrimes, u64* nPairs) {
  set<pair<u64, u64>> s;
  *nPrimes = *nPairs = 0;
  for (u64 q : primesIn(B1, B2)) {
    u64 const k = (2 * q + D) / (2 * D);        // round(q / D); q/D is never exactly half-integral for a prime q > the primes of D
    u64 const r = q > k * D ? q - k * D : k * D - q;
    if (!(r < D / 2 && gcd64(r, D) == 1)) { printf("FAIL: q=%" PRIu64 " gives r=%" PRIu64 "\n", q, r); exit(1); }
    ++*nPrimes;
    if (!s.insert({k, r}).second) { ++*nPairs; }
  }
  return s;
}

// The prime order of a mod f if it is a prime in (lo, hi], 1 if a = 1, else 0.
static u64 primeOrderIn(const mpz_t a, const mpz_t f, u64 lo, u64 hi) {
  mpz_t t;
  mpz_init(t);
  mpz_mod(t, a, f);
  u64 ret = 0;
  if (mpz_cmp_ui(t, 1) == 0) {
    ret = 1;
  } else {
    for (u64 q : primesIn(lo, hi)) {
      mpz_t e, y;
      mpz_init_set_ui(e, q);
      mpz_init(y);
      mpz_powm(y, t, e, f);
      bool const one = mpz_cmp_ui(y, 1) == 0;
      mpz_clears(e, y, nullptr);
      if (one) { ret = q; break; }
    }
  }
  mpz_clear(t);
  return ret;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "see the comment at the top of check_stage2.cpp\n"); return 2; }
  string const cmd = argv[1];
  mpz_t N, x, a, t, e;
  mpz_inits(N, x, a, t, e, nullptr);

  if (cmd == "random" && argc == 5) {
    u64 const p = strtoull(argv[2], 0, 10);
    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, strtoul(argv[3], 0, 10));
    mersenne(N, p);
    mpz_urandomm(x, rs, N);
    writeWords(x, argv[4], p);
    return 0;
  }

  if (cmd == "stage1" && argc == 5) {
    u64 const p = strtoull(argv[2], 0, 10), B1 = strtoull(argv[3], 0, 10);
    mersenne(N, p);
    stage1Exp(e, p, B1);
    mpz_set_ui(x, 3);
    mpz_powm(x, x, e, N);
    writeWords(x, argv[4], p);
    return 0;
  }

  if (cmd == "acc" && argc == 8) {
    u64 const p = strtoull(argv[2], 0, 10), B1 = strtoull(argv[3], 0, 10), B2 = strtoull(argv[4], 0, 10), D = strtoull(argv[5], 0, 10);
    mersenne(N, p);
    readWords(x, argv[6], p);
    readWords(a, argv[7], p);
    u64 nPrimes, nPairs;
    auto const pairs = pairsFor(B1, B2, D, &nPrimes, &nPairs);
    // x^(r^2) for each r used, x^((kD)^2) for each k used
    map<u64, vector<u64>> byK;
    set<u64> rs;
    for (auto [k, r] : pairs) { byK[k].push_back(r); rs.insert(r); }
    map<u64, mpz_t*> xr;
    for (u64 r : rs) {
      mpz_t* v = new mpz_t[1];
      mpz_init(*v);
      mpz_set_ui(e, r * r);
      mpz_powm(*v, x, e, N);
      xr[r] = v;
    }
    mpz_t acc, xk;
    mpz_init_set_ui(acc, 1);
    mpz_init(xk);
    for (auto& [k, list] : byK) {
      mpz_set_ui(e, k * D);
      mpz_mul(e, e, e);
      mpz_powm(xk, x, e, N);
      for (u64 r : list) {
        mpz_sub(t, xk, *xr[r]);
        mpz_mul(acc, acc, t);
        mpz_mod(acc, acc, N);
      }
    }
    mpz_mod(a, a, N);
    bool const ok = mpz_cmp(acc, a) == 0;
    printf("%s acc p=%" PRIu64 " B1=%" PRIu64 " B2=%" PRIu64 " D=%" PRIu64 ": %" PRIu64 " primes, %" PRIu64 " pairs, %zu multiplications\n",
           ok ? "OK" : "FAIL", p, B1, B2, D, nPrimes, nPairs, pairs.size());
    return ok ? 0 : 1;
  }

  if (cmd == "factors" && argc >= 8) {
    u64 const p = strtoull(argv[2], 0, 10), B1 = strtoull(argv[3], 0, 10), B2 = strtoull(argv[4], 0, 10);
    mersenne(N, p);
    readWords(x, argv[5], p);
    readWords(a, argv[6], p);
    mpz_gcd(t, a, N);
    int bad = 0;
    for (int i = 7; i < argc; ++i) {
      mpz_t f;
      mpz_init_set_str(f, argv[i], 10);
      mpz_mod(e, N, f);
      if (mpz_sgn(e) != 0) { printf("FAIL: %s does not divide 2^%" PRIu64 "-1\n", argv[i], p); ++bad; continue; }
      u64 const q = primeOrderIn(x, f, B1, B2);
      bool const expected = q != 0;      // order 1: every term x^a - x^b is 0 mod f
      mpz_mod(e, t, f);
      bool const found = mpz_sgn(e) == 0;
      printf("%s factor %s: order of x %s, %s by stage 2\n", expected == found ? "OK" : "FAIL", argv[i],
             q == 1 ? "1 (stage 1 finds it; every term is 0 mod f)" : q ? ("prime " + to_string(q) + " in (B1, B2]").c_str() : "not a prime in (B1, B2]",
             found ? "found" : "not found");
      bad += expected != found;
      mpz_clear(f);
    }
    return bad ? 1 : 0;
  }

  if (cmd == "search" && argc == 6) {
    u64 const p = strtoull(argv[2], 0, 10), kmax = strtoull(argv[3], 0, 10), B1 = strtoull(argv[4], 0, 10), B2 = strtoull(argv[5], 0, 10);
    stage1Exp(e, p, B1);
    for (u64 k = 1; k <= kmax; ++k) {
      unsigned __int128 const f = (unsigned __int128) 2 * k * p + 1;
      u64 const m8 = u64(f % 8);
      if (m8 != 1 && m8 != 7) { continue; }
      bool small = false;
      for (u64 s : {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}) {
        if (f % s == 0 && f != s) { small = true; break; }
      }
      if (small) { continue; }
      // 2^p mod f
      unsigned __int128 r = 1, b = 2;
      for (u64 pp = p; pp; pp >>= 1) {
        if (pp & 1) { r = r * b % f; }       // f < 2^64 assumed (k * p < 2^62)
        b = b * b % f;
      }
      if (r != 1) { continue; }
      mpz_t F, X;
      mpz_init(F);
      mpz_import(F, 1, -1, 8, 0, 0, &f);
      mpz_init_set_ui(X, 3);
      mpz_powm(X, X, e, F);
      u64 const q = primeOrderIn(X, F, B1, B2);
      gmp_printf("factor %Zd k=%" PRIu64 ": %s\n", F, k, q == 1 ? "stage 1" : q ? ("stage 2, q=" + to_string(q)).c_str() : "neither");
      mpz_clears(F, X, nullptr);
    }
    return 0;
  }

  if (cmd == "tf" && argc == 6) {
    // Trial factoring 2^p-1 for f = 2kp+1, kmin <= k < kmax, f < 2^63, with Montgomery arithmetic on threads.  Prints the factors.
    u64 const p = strtoull(argv[2], 0, 10), kmin = strtoull(argv[3], 0, 10), kmax = strtoull(argv[4], 0, 10);
    unsigned const nThreads = strtoul(argv[5], 0, 10);
    std::mutex mutex;
    auto work = [&](unsigned id) {
      for (u64 k = kmin + id; k < kmax; k += nThreads) {
        u64 const f = 2 * k * p + 1;
        if ((f & 7) != 1 && (f & 7) != 7) { continue; }
        if (f % 3 == 0 || f % 5 == 0 || f % 7 == 0 || f % 11 == 0 || f % 13 == 0 || f % 17 == 0 || f % 19 == 0 || f % 23 == 0) { continue; }
        // Montgomery:  ninv = -f^-1 mod 2^64
        u64 inv = f;
        for (int i = 0; i < 5; ++i) { inv *= 2 - f * inv; }
        u64 const ninv = -inv;
        auto redc = [&](unsigned __int128 t) {
          u64 const m = u64(t) * ninv;
          unsigned __int128 const u = (t + (unsigned __int128) m * f) >> 64;
          return u64(u >= f ? u - f : u);
        };
        u64 const r1 = u64(((unsigned __int128) 1 << 64) % f);      // 1 in Montgomery form
        u64 x = u64(((unsigned __int128) 2 << 64) % f);              // 2
        int top = 63 - __builtin_clzll(p);
        for (int b = top - 1; b >= 0; --b) {
          x = redc((unsigned __int128) x * x);
          if ((p >> b) & 1) { x = x + x; if (x >= f) { x -= f; } }
        }
        if (x == r1) {
          std::lock_guard lock{mutex};
          printf("factor %" PRIu64 " k=%" PRIu64 " =", f, k);
          u64 kk = k;
          for (u64 d = 2; d * d <= kk; ++d) { while (kk % d == 0) { printf(" %" PRIu64, d); kk /= d; } }
          if (kk > 1) { printf(" %" PRIu64, kk); }
          printf("\n");
          fflush(stdout);
        }
      }
    };
    vector<std::thread> threads;
    for (unsigned i = 0; i < nThreads; ++i) { threads.emplace_back(work, i); }
    for (auto& t : threads) { t.join(); }
    return 0;
  }

  if (cmd == "memplan" && argc == 6) {
    // memplan B1 B2 budgetMB bufferMB:  the D the memory planner picks (src/Pm1.cpp chooseStage2D, 6 buffers besides the table) and
    // its multiplication count
    u64 const B1 = strtoull(argv[2], 0, 10), B2 = strtoull(argv[3], 0, 10);
    double const budget = strtod(argv[4], 0), buf = strtod(argv[5], 0);
    u64 const maxR = budget > buf * 7 ? u64(budget / buf) - 6 : 0;
    u32 const D = pm1::chooseStage2D(B1, B2, maxR);
    if (!D) { printf("declined (%" PRIu64 " buffers)\n", maxR); return 0; }
    pm1::Stage2Plan const plan = pm1::planStage2(B1, B2, D);
    printf("D=%u nR=%zu (%.1f GB) muls=%" PRIu64 " (primes %" PRIu64 ", k steps %" PRIu64 ", table ~%u)\n", D, plan.rs.size(),
           (plan.rs.size() + 6) * buf / 1024, plan.nNeeded + 2 * (plan.kEnd - plan.kBegin), plan.nNeeded, 2 * (plan.kEnd - plan.kBegin), D / 2);
    return 0;
  }

  if (cmd == "plan" && argc == 5) {
    u64 const B1 = strtoull(argv[2], 0, 10), B2 = strtoull(argv[3], 0, 10), D = strtoull(argv[4], 0, 10);
    u64 nPrimes, nPairs;
    auto const pairs = pairsFor(B1, B2, D, &nPrimes, &nPairs);
    pm1::Stage2Plan const plan = pm1::planStage2(B1, B2, u32(D));
    u64 nBits = 0;
    bool same = true;
    for (u64 k = plan.kBegin; k < plan.kEnd; ++k) {
      for (u32 i = 0; i < plan.rs.size(); ++i) {
        if (plan.needed(k, i)) { ++nBits; same = same && pairs.count({k, plan.rs[i]}); }
      }
    }
    bool const ok = same && nBits == pairs.size() && plan.nNeeded == pairs.size() && plan.nPrimes == nPrimes && plan.nPairs == nPairs;
    printf("%s plan B1=%" PRIu64 " B2=%" PRIu64 " D=%" PRIu64 ": %" PRIu64 " primes, %" PRIu64 " pairs (%.1f%% paired), %zu multiplications + %" PRIu64
           " k steps\n", ok ? "OK" : "FAIL", B1, B2, D, nPrimes, nPairs, 200.0 * nPairs / nPrimes, pairs.size(), 2 * (plan.kEnd - plan.kBegin));
    return ok ? 0 : 1;
  }

  fprintf(stderr, "bad arguments\n");
  return 2;
}
