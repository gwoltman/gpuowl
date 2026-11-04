// Checks of the GPU GCD and Jacobi symbol (src/Gcd.cpp) against GMP's mpz_gcd and mpz_jacobi.  GMP is only the oracle here; prpll
// does not use it.
//
//   check_gcd [-device N] [-skew K] [-sizes p1,p2,...] [-seed S] [-quick]
// -sizes: the exponents p (N = 2^p - 1) of the general cases (default: a set of small and medium ones)
// -skew K: build the kernels with -DSKEW=K (non-canonical digits on purpose, see gcd.cl); every run must then report skew events

#include "Gcd.h"
#include "Gpu.h"
#include "FFTConfig.h"
#include "Args.h"
#include "Context.h"
#include "Background.h"
#include "TrigBufCache.h"
#include "clwrap.h"
#include "log.h"

#include <gmp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using std::string;
using std::vector;

static Words toWords(const mpz_t z, u32 p) {
  Words w((p - 1) / 32 + 1);
  size_t count = 0;
  mpz_export(w.data(), &count, -1, sizeof(u32), 0, 0, z);
  return w;
}

static void fromWords(mpz_t z, const Words& w) { mpz_import(z, w.size(), -1, sizeof(u32), 0, 0, w.data()); }

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Checker {
  Gcd& gcd;
  u32 skew;
  int fails = 0, runs = 0;
  u64 skewEvents = 0;
  bool timing = false;

  // sub = 1: gcdMinus1 (gcd(x - 1, N)), sub = 0: gcd (gcd(x, N), stage 2's test)
  void checkGcd(const char* what, const mpz_t x, u32 p, int sub = 1) {
    mpz_t N, xm1, want, got;
    mpz_inits(N, xm1, want, got, nullptr);
    mpz_set_ui(N, 1); mpz_mul_2exp(N, N, p); mpz_sub_ui(N, N, 1);
    mpz_sub_ui(xm1, x, sub);
    double t0 = now();
    mpz_gcd(want, xm1, N);
    double tGmp = now() - t0;
    Words w;
    try { w = sub ? gcd.gcdMinus1(toWords(x, p), p) : gcd.gcd(toWords(x, p), p); } catch (const std::exception& e) { printf("exception %s in gcd %s p=%u\n", e.what(), what, p); w = {0}; }
    auto st = gcd.lastStats();
    fromWords(got, w);
    bool ok = mpz_cmp(want, got) == 0;
    ++runs; skewEvents += st.skewEvents;
    if (!ok) { ++fails; }
    if (!ok || timing) {
      printf("%s %s p=%-9u %-24s %.3f s (GMP %.3f s) %.3f divsteps/bit, skew events %llu, result %zu bits\n", ok ? "OK  " : "FAIL",
             sub ? "gcd " : "gcd0", p, what, st.seconds, tGmp, double(st.divsteps) / p, (unsigned long long) st.skewEvents, mpz_sizeinbase(want, 2));
    }
    mpz_clears(N, xm1, want, got, nullptr);
  }

  void checkJacobi(const char* what, const mpz_t x, u32 p) {
    mpz_t N;
    mpz_init(N);
    mpz_set_ui(N, 1); mpz_mul_2exp(N, N, p); mpz_sub_ui(N, N, 1);
    double t0 = now();
    int want = mpz_jacobi(x, N);
    double tGmp = now() - t0;
    int got = 2;
    try { got = gcd.jacobi(toWords(x, p), p); } catch (const std::exception& e) { printf("exception %s in jacobi %s p=%u\n", e.what(), what, p); }
    auto st = gcd.lastStats();
    bool ok = want == got;
    ++runs; skewEvents += st.skewEvents;
    if (!ok) { ++fails; }
    if (!ok || timing) {
      printf("%s jac  p=%-9u %-24s %.3f s (GMP %.3f s) %.3f divsteps/bit, skew events %llu, symbol %d (GPU %d)\n", ok ? "OK  " : "FAIL",
             p, what, st.seconds, tGmp, double(st.divsteps) / p, (unsigned long long) st.skewEvents, want, got);
    }
    mpz_clear(N);
  }

  void both(const char* what, const mpz_t x, u32 p) {
    checkGcd(what, x, p);
    checkGcd(what, x, p, 0);
    checkJacobi(what, x, p);
  }
};

// The cases for one exponent p
static void cases(Checker& c, u32 p, gmp_randstate_t rs, bool quick) {
  mpz_t N, x, y, f;
  mpz_inits(N, x, y, f, nullptr);
  mpz_set_ui(N, 1); mpz_mul_2exp(N, N, p); mpz_sub_ui(N, N, 1);

  // Fixed values: 0, 1, 2, 3, N - 1, N (PRPLL's other zero), 2^k + 1
  for (unsigned long v : {0ul, 1ul, 2ul, 3ul}) {
    mpz_set_ui(x, v);
    c.both(("x=" + std::to_string(v)).c_str(), x, p);
  }
  mpz_sub_ui(x, N, 1); c.both("x=N-1", x, p);
  c.both("x=N (all ones)", N, p);
  mpz_set_ui(x, 1); mpz_mul_2exp(x, x, p / 2); mpz_add_ui(x, x, 1); c.both("x=2^(p/2)+1", x, p);

  // Random residues, x - 1 odd and even
  for (int i = 0; i < (quick ? 2 : 6); ++i) {
    mpz_urandomm(x, rs, N);
    mpz_setbit(x, 0);
    if (i & 1) { mpz_clrbit(x, 0); }
    c.both(i & 1 ? "random, x-1 odd" : "random, x-1 even", x, p);
  }
  // x - 1 with many factors of two
  mpz_urandomm(x, rs, N); mpz_fdiv_q_2exp(x, x, p / 3); mpz_mul_2exp(x, x, p / 3); mpz_add_ui(x, x, 1);
  c.both("x-1 = m 2^(p/3)", x, p);

  // A planted common factor: a proper factor f of N (if N is composite), and x = 1 + f y, so that f | gcd(x - 1, N)
  // (and x with f | x for the Jacobi symbol 0)
  mpz_set_ui(f, 0);
  for (u32 k = 1; k < 2000000 && u64(k) * 2 * p + 1 < (u64(1) << 40); ++k) {
    u64 q = u64(k) * 2 * p + 1;
    if ((q % 8 == 1 || q % 8 == 7) && mpz_fdiv_ui(N, q) == 0 && mpz_cmp_ui(N, q) != 0) { mpz_set_ui(f, q); break; }
  }
  for (u32 q = 3; !mpz_sgn(f) && q < 20000; q += 2) {
    if (mpz_fdiv_ui(N, q) == 0 && mpz_cmp_ui(N, q) != 0) { mpz_set_ui(f, q); }
  }
  if (mpz_sgn(f)) {
    mpz_urandomm(y, rs, N); mpz_mul(x, f, y); mpz_mod(x, x, N); mpz_add_ui(x, x, 1);
    c.both("planted factor of N", x, p);
    mpz_divexact(y, N, f);       // the cofactor: gcd(x - 1, N) = N / f
    mpz_mul_ui(x, y, 3); mpz_mod(x, x, N); mpz_add_ui(x, x, 1);
    c.both("planted cofactor N/f", x, p);
    // gcd(x, N) (stage 2's accumulator test):  x = f y, and x = (N / f) * 3
    mpz_urandomm(y, rs, N); mpz_mul(x, f, y); mpz_mod(x, x, N);
    c.checkGcd("x = f y", x, p, 0);
    mpz_divexact(y, N, f); mpz_mul_ui(x, y, 3); mpz_mod(x, x, N);
    c.checkGcd("x = 3 N/f", x, p, 0);
  }
  mpz_clears(N, x, y, f, nullptr);
}

// M4 gate: exact n-bit x n-bit products through Gpu::expMul on an NTT-only FFT (FFT3161, worst-case bits per word), timed with the
// host conversions (writeIn, readAndCompress); random and all-ones operands checked against GMP; and the Gpu's construction time.
static void gate(GpuCommon shared, u32 n) {
  u64 const E = 2 * u64(n) + 64;
  FFTShape shape;
  for (const FFTShape& s : FFTShape::allShapes()) {
    if (s.fft_type != FFT3161 || s.isPfa()) { continue; }
    double const b = std::ceil(double(E) / s.size());
    if (2 * b - 2 + std::log2(double(s.size())) <= 90) { shape = s; break; }
  }
  if (shape.fft_type != FFT3161) { printf("gate %u: no FFT\n", n); return; }
  FFTConfig fft{shape, 101, CARRY_AUTO};
  double t0 = now();
  auto gpu = Gpu::make(E, shared, fft, {}, false);
  double const tMake = now() - t0;
  gmp_randstate_t rs; gmp_randinit_mt(rs); gmp_randseed_ui(rs, n);
  mpz_t a, b, c, want;
  mpz_inits(a, b, c, want, nullptr);
  bool ok = true;
  double tMul = 0;
  for (int k = 0; k < 4; ++k) {
    if (k < 2) { mpz_urandomb(a, rs, n); mpz_urandomb(b, rs, n); }
    else { mpz_set_ui(a, 1); mpz_mul_2exp(a, a, n); mpz_sub_ui(a, a, 1); mpz_set(b, a); }   // all ones
    mpz_mul(want, a, b);
    Words wa(nWords(E)), wb(nWords(E));
    mpz_export(wa.data(), nullptr, -1, 4, 0, 0, a);
    mpz_export(wb.data(), nullptr, -1, 4, 0, 0, b);
    t0 = now();
    Words wc = gpu->expMul(wa, 1, wb, false);
    if (k) { tMul += now() - t0; }
    mpz_import(c, wc.size(), -1, 4, 0, 0, wc.data());
    ok = ok && mpz_cmp(c, want) == 0;
  }
  printf("gate %9u-bit product: FFT %s (E %llu, %.1f bits/word): %s, make %.2f s, %.2f ms per product with host conversions\n", n,
         fft.spec().c_str(), (unsigned long long) E, double(E) / shape.size(), ok ? "exact" : "WRONG", tMake, tMul / 3 * 1000);
  mpz_clears(a, b, c, want, nullptr);
}


// ---- Leaf check: the leaf kernel's matrix against divsteps on the exact numbers (same derive rules as gcd.cl) ----

struct RefMat { i64 u, v, q, r; };

static i64 refDerive(u64 f, u64 g, i64 delta, int& jac, bool jacobi, RefMat& m) {
  i64 u = 1, v = 0, q = 0, r = 1;
  for (int i = 0; i < 62; ++i) {
    if (jacobi) {
      if (g & 1) {
        if (delta > 0) { jac ^= int((f & g) >> 1) & 1; std::swap(f, g); std::swap(u, q); std::swap(v, r); delta = -delta; }
        g += f; q += u; r += v;
      }
      jac ^= int((f >> 1) ^ (f >> 2)) & 1;
      g >>= 1; u *= 2; v *= 2; delta += 1;
    } else if (g & 1) {
      if (delta > 0) { u64 nf = g, ng = (g - f) >> 1; i64 nu = 2 * q, nv = 2 * r, nq = q - u, nr = r - v; f = nf; g = ng; u = nu; v = nv; q = nq; r = nr; delta = 1 - delta; }
      else { g = (g + f) >> 1; q += u; r += v; u *= 2; v *= 2; delta = 1 + delta; }
    } else { g >>= 1; u *= 2; v *= 2; delta = 1 + delta; }
  }
  m = {u, v, q, r};
  return delta;
}

static u64 low64(const mpz_t z) { mpz_t t; mpz_init(t); mpz_fdiv_r_2exp(t, z, 64); u64 x = mpz_getlimbn(t, 0); mpz_clear(t); return x; }

static void setI64(mpz_t z, i64 x) { mpz_set_si(z, x); }

// Balanced 62-bit digits of z mod 2^(62 n)
static vector<i64> toDigits62(const mpz_t z, u32 n) {
  vector<i64> d(n);
  mpz_t t, r; mpz_inits(t, r, nullptr);
  mpz_fdiv_r_2exp(t, z, 62 * n);
  for (u32 i = 0; i < n; ++i) {
    mpz_fdiv_r_2exp(r, t, 62);
    i64 v = i64(mpz_get_ui(r));
    if (v >= (i64(1) << 61)) { v -= i64(1) << 62; }
    d[i] = v;
    setI64(r, v);
    mpz_sub(t, t, r);
    mpz_fdiv_q_2exp(t, t, 62);
  }
  mpz_clears(t, r, nullptr);
  return d;
}

static void fromDigits62(mpz_t z, const i64* d, u32 n) {
  mpz_t t; mpz_init(t);
  mpz_set_ui(z, 0);
  for (u32 i = n; i-- > 0;) { mpz_mul_2exp(z, z, 62); setI64(t, d[i]); mpz_add(z, z, t); }
  mpz_clear(t);
}

static bool checkLeaf(Gcd& gcd, u32 nb, u32 n, bool jacobi, gmp_randstate_t rs, bool print, bool useJump = false) {
  mpz_t f, g, F, G, U, V, Q, R, a, b, t, m;
  mpz_inits(f, g, F, G, U, V, Q, R, a, b, t, m, nullptr);
  mpz_urandomb(f, rs, 62 * n - 2); mpz_setbit(f, 0);
  mpz_urandomb(g, rs, 62 * n - 2);
  if (!jacobi && (nb & 1)) { mpz_neg(g, g); }
  i64 delta0 = jacobi ? 1 : 1 + i64(nb % 3) - 1;
  vector<i64> fd = toDigits62(f, n), gd = toDigits62(g, n);
  double sec = 0;
  vector<i64> got = useJump ? gcd.jumpMatrix(fd, gd, nb, delta0) : gcd.leafMatrix(fd, gd, nb, delta0, jacobi, &sec);
  // reference
  mpz_set(F, f); mpz_set(G, g);
  mpz_set_ui(U, 1); mpz_set_ui(V, 0); mpz_set_ui(Q, 0); mpz_set_ui(R, 1);
  i64 delta = delta0; int jac = 0;
  for (u32 k = 0; k < nb; ++k) {
    RefMat M;
    delta = refDerive(low64(F), low64(G), delta, jac, jacobi, M);
    mpz_t mu, mv, mq, mr; mpz_inits(mu, mv, mq, mr, nullptr);
    setI64(mu, M.u); setI64(mv, M.v); setI64(mq, M.q); setI64(mr, M.r);
    mpz_mul(a, mu, F); mpz_addmul(a, mv, G); mpz_mul(b, mq, F); mpz_addmul(b, mr, G);
    mpz_fdiv_q_2exp(F, a, 62); mpz_fdiv_q_2exp(G, b, 62);
    mpz_mul(a, mu, U); mpz_addmul(a, mv, Q); mpz_mul(b, mq, U); mpz_addmul(b, mr, Q); mpz_set(U, a); mpz_set(Q, b);
    mpz_mul(a, mu, V); mpz_addmul(a, mv, R); mpz_mul(b, mq, V); mpz_addmul(b, mr, R); mpz_set(V, a); mpz_set(R, b);
    mpz_clears(mu, mv, mq, mr, nullptr);
  }
  bool ok = true;
  mpz_t* want[4] = {&U, &V, &Q, &R};
  for (int k = 0; k < 4; ++k) {
    fromDigits62(m, &got[k * (nb + 2)], nb + 2);
    ok = ok && mpz_cmp(m, *want[k]) == 0;
  }
  ok = ok && got[4 * (nb + 2)] == delta && (!jacobi || got[4 * (nb + 2) + 1] == jac);
  if (!ok || print) {
    printf("%s %s %s nb=%u (%u divsteps) n=%u: %.1f us, %.1f ns per divstep\n", ok ? "OK  " : "FAIL", useJump ? "jump" : "leaf", jacobi ? "jacobi" : "gcd", nb, 62 * nb, n,
           sec * 1e6, sec * 1e9 / (62.0 * nb));
  }
  mpz_clears(f, g, F, G, U, V, Q, R, a, b, t, m, nullptr);
  return ok;
}

static bool checkMul(Gcd& gcd, u32 na, u32 nb, gmp_randstate_t rs) {
  mpz_t a, b, want, got;
  mpz_inits(a, b, want, got, nullptr);
  mpz_urandomb(a, rs, 62 * na - 1); mpz_urandomb(b, rs, 62 * nb - 1);
  if (na & 1) { mpz_neg(a, a); }
  if (nb % 3 == 0) { mpz_neg(b, b); }
  if (na % 5 == 0) {   // all ones
    mpz_set_ui(a, 1); mpz_mul_2exp(a, a, 62 * na - 2); mpz_sub_ui(a, a, 1);
    mpz_set_ui(b, 1); mpz_mul_2exp(b, b, 62 * nb - 2); mpz_sub_ui(b, b, 1); mpz_neg(b, b);
  }
  mpz_mul(want, a, b);
  vector<i64> d = gcd.mulTest(toDigits62(a, na + 1), toDigits62(b, nb + 1));
  fromDigits62(got, d.data(), u32(d.size()));
  bool ok = mpz_cmp(want, got) == 0;
  if (!ok) { printf("FAIL mul %u x %u digits\n", na, nb);}
  mpz_clears(a, b, want, got, nullptr);
  return ok;
}

int main(int argc, char** argv) {

  u32 device = 1, skew = 0;
  unsigned long seed = 1;
  bool quick = false;
  vector<u32> sizes{3, 5, 7, 11, 13, 29, 31, 61, 62, 63, 64, 67, 89, 113, 123, 124, 125, 127, 186, 187, 251, 509, 1279, 2203, 4423,
                    10007, 44497, 100003};
  vector<u32> timed, edges, gates, leaves, jumps;
  u32 hgcd = 0, leafNb = 0, fftDigits = 0;
  bool mulCheck = false;
  for (int i = 1; i < argc; ++i) {
    string a = argv[i];
    if (a == "-device") { device = atoi(argv[++i]); }
    else if (a == "-skew") { skew = atoi(argv[++i]); }
    else if (a == "-seed") { seed = strtoul(argv[++i], nullptr, 10); }
    else if (a == "-quick") { quick = true; }
    else if (a == "-hgcd") { hgcd = atoi(argv[++i]); }
    else if (a == "-leafnb") { leafNb = atoi(argv[++i]); }
    else if (a == "-fftdigits") { fftDigits = atoi(argv[++i]); }
    else if (a == "-mul") { mulCheck = true; }
    else if (a == "-sizes" || a == "-time" || a == "-edges" || a == "-gate" || a == "-leaf" || a == "-jump") {
      vector<u32>& v = a == "-sizes" ? sizes : a == "-time" ? timed : a == "-edges" ? edges : a == "-gate" ? gates : a == "-leaf" ? leaves : jumps;
      v.clear();
      for (char* s = strtok(argv[++i], ","); s; s = strtok(nullptr, ",")) { v.push_back(strtoul(s, nullptr, 10)); }
      if (a != "-sizes") { sizes.clear(); }
    } else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }

  Args args{true};
  args.device = device;
  args.setDefaults();
  Context context{getDevice(device)};
  Background background;
  TrigBufCache bufCache{&context};
  GpuCommon shared{&context, &args, &bufCache, &background};
  Gcd gcd{shared, skew};
  Checker c{gcd, skew};
  gmp_randstate_t rs;
  gmp_randinit_mt(rs);
  gmp_randseed_ui(rs, seed);

  if (hgcd) { gcd.halfGcdDigits = hgcd; }
  if (leafNb) { gcd.leafBatches = leafNb; }
  if (fftDigits) { gcd.fftDigits = fftDigits; }
  if (mulCheck) {
    bool ok = true;
    for (u32 na : {1u, 2u, 3u, 5u, 7u, 30u, 100u, 257u, 1000u}) {
      for (u32 nb : {1u, 2u, 4u, 9u, 64u, 300u, 1000u}) { ok = checkMul(gcd, na, nb, rs) && ok; ++c.runs; }
    }
    c.fails += !ok;
    printf("mul: %s\n", ok ? "ok" : "FAILED");
  }
  for (u32 nb : jumps) {
    bool ok = true;
    for (int i = 0; i < 4; ++i) { ok = checkLeaf(gcd, nb, nb + 2 + u32(i), false, rs, i == 3, true) && ok; }
    c.runs += 4; c.fails += !ok;
    fflush(stdout);
  }
  for (u32 nb : leaves) {
    for (bool jacobi : {false, true}) {
      bool ok = true;
      for (int i = 0; i < 5; ++i) { ok = checkLeaf(gcd, nb, std::min(nb + 2 + u32(i), 280u), jacobi, rs, i == 4) && ok; }
      c.runs += 5; c.fails += !ok;
    }
    fflush(stdout);
  }

  for (u32 n : gates) {
    try { gate(shared, n); } catch (const char* e) { printf("gate %u: %s\n", n, e); } catch (const string& e) { printf("gate %u: %s\n", n, e.c_str()); }
    fflush(stdout);
  }

  for (u32 p : sizes) {
    int before = c.fails;
    cases(c, p, rs, quick);
    printf("p=%u: %s\n", p, c.fails == before ? "ok" : "FAILED");
    fflush(stdout);
  }

  // x = 0, 1 (gcd = N: the conversion of a large result) and N, timed
  c.timing = true;
  for (u32 p : edges) {
    mpz_t N, x;
    mpz_inits(N, x, nullptr);
    mpz_set_ui(N, 1); mpz_mul_2exp(N, N, p); mpz_sub_ui(N, N, 1);
    mpz_set_ui(x, 1); c.both("x=1", x, p);
    mpz_set_ui(x, 0); c.both("x=0", x, p);
    c.both("x=N (all ones)", N, p);
    mpz_clears(N, x, nullptr);
    fflush(stdout);
  }

  // Timings: a random residue minus one (GCD) and a random residue (Jacobi), as in P-1
  c.timing = true;
  for (u32 p : timed) {
    mpz_t N, x;
    mpz_inits(N, x, nullptr);
    mpz_set_ui(N, 1); mpz_mul_2exp(N, N, p); mpz_sub_ui(N, N, 1);
    mpz_urandomm(x, rs, N);
    c.both("random", x, p);
    mpz_clears(N, x, nullptr);
    fflush(stdout);
  }

  printf("%d runs, %d failures, skew events %llu\n", c.runs, c.fails, (unsigned long long) c.skewEvents);
  if (skew && c.skewEvents == 0) { printf("FAIL: -skew %u made no skew events\n", skew); return 1; }
  return c.fails ? 1 : 0;
}
