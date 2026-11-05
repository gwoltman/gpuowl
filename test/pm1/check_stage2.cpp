// External oracle for P-1 stage 2 (src/Stage2.cpp), GMP only; not linked into prpll.
//
//   check_stage2 random p seed x.bin                 a random residue mod 2^p-1
//   check_stage2 stage1 p B1 x.bin                   x = 3^E mod 2^p-1, E = 2p * prod over primes q <= B1 of the largest q^e <= B1
//   check_stage2 acc p B1 B2 D M x.bin acc.bin [kW]  acc.bin == prod over the items (k, n) of src/Pm1.cpp's plan of x^((kD)^2) - x^(n^2)
//                                                    mod 2^p-1, and the items cover every prime of (B1, B2] (primes from mpz_nextprime):
//                                                    with both, every prime q in (B1, B2] divides the exponent of some factor of acc
//   check_stage2 factors p B1 B2 x.bin acc.bin f...  for each factor f of 2^p-1:  f | gcd(acc, 2^p-1) exactly when x mod f has order 1
//                                                    or a prime order in (B1, B2]  (the exact condition, tested with GMP on x mod f)
//   check_stage2 search p kmax B1 B2                 factors f = 2kp+1 of 2^p-1 with k <= kmax, each with the order of 3^E mod f and
//                                                    whether stage 1 (B1) or stage 2 (B1, B2) finds it
//   check_stage2 plan B1 B2 D M [kW]                 src/Pm1.cpp's plan against one made here:  the same items (an independent sweep of
//                                                    the same pairing rule on mpz_nextprime primes), every prime covered, the same counts
//   check_stage2 match B1 B2 D M [kW]                the plan's pairs against a maximum matching (Hopcroft-Karp) of each window's graph
//   check_stage2 memplan B1 B2 budgetMB bufferMB     the (D, M) the planner picks for a memory budget (see the code)
//   check_stage2 savefile p in.p1|in.p2 out.bin      the words of one of prpll's P-1 savefiles (header line, CRC, words) to out.bin,
//                                                    and its header line
//   check_stage2 gcd p acc.bin                       gcd(acc, 2^p-1) in decimal ("gcd 1": no factor)
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
#include <deque>
#include <functional>
#include <algorithm>
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

// An item (k, n):  one accumulator multiplication by x^((kD)^2) - x^(n^2)
using Item = pair<u64, u64>;

// The items of src/Pm1.cpp's plan, all windows
static vector<Item> planItems(const pm1::Stage2Plan& plan) {
  vector<Item> r;
  pm1::Stage2Window win;
  for (u64 w = 0; w < plan.nWindows(); ++w) {
    pm1::planWindow(plan, w, win);
    for (u64 k = win.kLo; k < win.kHi; ++k) {
      for (u32 i = 0; i < win.count(k); ++i) { r.push_back({k, plan.ns[win.items(k)[i]]}); }
    }
  }
  return r;
}

// Every prime of (B1, B2] is kD - n or kD + n for some item, and every item is valid (n < M coprime to D, k in the plan's range).
// Counts the items covering two primes and those with n > D/2.
// (An item counted with two primes may be a single:  its other prime paired elsewhere.)
static bool covers(const vector<Item>& items, const vector<u64>& primes, const pm1::Stage2Plan& plan, u64* nTwo, u64* nCross) {
  set<u64> covered;
  set<u64> const isPrime(primes.begin(), primes.end());
  *nTwo = *nCross = 0;
  bool ok = true;
  for (auto [k, n] : items) {
    if (!(n < plan.M && gcd64(n, plan.D) == 1 && k >= plan.kBegin && k < plan.kEnd)) {
      printf("FAIL: item (%" PRIu64 ", %" PRIu64 ") is not valid\n", k, n);
      ok = false;
    }
    u64 const kD = k * plan.D;
    int nIn = 0;
    for (u64 q : {kD + n, kD >= n ? kD - n : n - kD}) {
      if (isPrime.count(q)) { covered.insert(q); ++nIn; }
    }
    *nTwo += nIn == 2;
    *nCross += nIn == 2 && n > plan.D / 2;
    if (!nIn) { printf("FAIL: item (%" PRIu64 ", %" PRIu64 ") covers no prime\n", k, n); ok = false; }
  }
  if (covered.size() != primes.size()) {
    for (u64 q : primes) {
      if (!covered.count(q)) { printf("FAIL: prime %" PRIu64 " is not covered\n", q); ok = false; break; }
    }
  }
  return ok;
}

// The windows of the plan, made here:  window w holds the primes q with round(q / D) in [kBegin + w kW, kBegin + (w + 1) kW)
static u64 windowOfPrime(const pm1::Stage2Plan& plan, u64 q) { return ((q + plan.D / 2) / plan.D - plan.kBegin) / plan.kWindow; }

// The pairing rule again, on its own (not src/Pm1.cpp's code):  per window and class pair {c, 2D - c}, a two-pointer sweep of the sorted
// list in which a prime takes the oldest unpaired prime of the other class closer than 2M; pending primes that fall out of reach are
// singles at round(q / D)
static vector<Item> itemsHere(const pm1::Stage2Plan& plan, const vector<u64>& primes) {
  u64 const D = plan.D, M = plan.M;
  map<pair<u64, u64>, vector<u64>> groups;     // (window, class pair) -> primes, increasing
  for (u64 q : primes) {
    u64 const c = q % (2 * D);
    groups[{windowOfPrime(plan, q), std::min(c, 2 * D - c)}].push_back(q);
  }
  vector<Item> r;
  auto single = [&](u64 q) {
    u64 const k = (2 * q + D) / (2 * D);
    r.push_back({k, q > k * D ? q - k * D : k * D - q});
  };
  for (auto& [key, list] : groups) {
    std::deque<u64> wait;
    for (u64 q : list) {
      while (!wait.empty() && q - wait.front() >= 2 * M) { single(wait.front()); wait.pop_front(); }
      if (!wait.empty() && (wait.front() + q) % (2 * D) == 0) {
        r.push_back({(wait.front() + q) / (2 * D), (q - wait.front()) / 2});
        wait.pop_front();
      } else {
        wait.push_back(q);
      }
    }
    for (u64 q : wait) { single(q); }
  }
  return r;
}

// Maximum bipartite matching (Hopcroft-Karp) of the pairing graph:  vertices = primes, an edge between primes of one window with
// q1 + q2 = 0 mod 2D and |q1 - q2| < 2M (left = q mod 2D < D)
static u64 maxMatching(const pm1::Stage2Plan& plan, const vector<u64>& primes) {
  u64 const D = plan.D, M = plan.M;
  vector<u64> left, right;
  for (u64 q : primes) { (q % (2 * D) < D ? left : right).push_back(q); }
  map<u64, vector<u32>> rightByClass;      // class -> indices into right (increasing q)
  for (u32 j = 0; j < right.size(); ++j) { rightByClass[right[j] % (2 * D)].push_back(j); }
  vector<vector<u32>> adj(left.size());
  for (u32 i = 0; i < left.size(); ++i) {
    u64 const q = left[i];
    auto it = rightByClass.find((2 * D - q % (2 * D)) % (2 * D));
    if (it == rightByClass.end()) { continue; }
    for (u32 j : it->second) {
      u64 const q2 = right[j];
      if ((q > q2 ? q - q2 : q2 - q) < 2 * M && windowOfPrime(plan, q) == windowOfPrime(plan, q2)) { adj[i].push_back(j); }
    }
  }
  u32 const NIL = u32(-1);
  vector<u32> matchL(left.size(), NIL), matchR(right.size(), NIL), dist(left.size());
  auto bfs = [&]() {
    std::deque<u32> qu;
    bool found = false;
    for (u32 i = 0; i < left.size(); ++i) {
      if (matchL[i] == NIL) { dist[i] = 0; qu.push_back(i); } else { dist[i] = NIL; }
    }
    while (!qu.empty()) {
      u32 const i = qu.front(); qu.pop_front();
      for (u32 j : adj[i]) {
        u32 const i2 = matchR[j];
        if (i2 == NIL) { found = true; } else if (dist[i2] == NIL) { dist[i2] = dist[i] + 1; qu.push_back(i2); }
      }
    }
    return found;
  };
  std::function<bool(u32)> dfs = [&](u32 i) {
    for (u32 j : adj[i]) {
      u32 const i2 = matchR[j];
      if (i2 == NIL || (dist[i2] == dist[i] + 1 && dfs(i2))) { matchL[i] = j; matchR[j] = i; return true; }
    }
    dist[i] = NIL;
    return false;
  };
  u64 size = 0;
  while (bfs()) {
    for (u32 i = 0; i < left.size(); ++i) { if (matchL[i] == NIL && dfs(i)) { ++size; } }
  }
  return size;
}

// B1 B2 D M from argv[at..at+3], kWindow from argv[kwAt] if kwAt < argc
static pm1::Stage2Plan makePlan(int argc, char** argv, int at, int kwAt) {
  u64 const B1 = strtoull(argv[at], 0, 10), B2 = strtoull(argv[at + 1], 0, 10);
  u32 const D = u32(strtoul(argv[at + 2], 0, 10)), M = u32(strtoul(argv[at + 3], 0, 10));
  u64 const kW = kwAt < argc ? strtoull(argv[kwAt], 0, 10) : 0;
  if (!(D % 2 == 0 && pm1::stage2DFits(D, B1) && B1 < B2 && M >= D / 2)) { fprintf(stderr, "bad plan parameters\n"); exit(2); }
  return pm1::planStage2(B1, B2, D, M, kW);
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

  if (cmd == "acc" && (argc == 9 || argc == 10)) {
    u64 const p = strtoull(argv[2], 0, 10);
    pm1::Stage2Plan const plan = makePlan(argc, argv, 3, 9);
    u64 const D = plan.D;
    mersenne(N, p);
    readWords(x, argv[7], p);
    readWords(a, argv[8], p);
    vector<Item> const items = planItems(plan);
    vector<u64> const primes = primesIn(plan.B1, plan.B2);
    u64 nTwo, nCross;
    bool const cover = covers(items, primes, plan, &nTwo, &nCross);
    // x^(n^2) for each n used, and x^((kD)^2) for each k from the first item's to the last's:  by second differences in exact GMP
    // arithmetic (x^((n+2)^2) = x^(n^2) x^(4n+4);  x^(((k+1)D)^2) = x^((kD)^2) x^((2k+1)D^2)), each product reduced mod 2^p - 1
    auto modN = [&](mpz_t z) {           // z mod 2^p - 1 for 0 <= z < 2^(2p):  (z mod 2^p) + (z >> p), then once more
      for (int i = 0; i < 2; ++i) {
        mpz_tdiv_q_2exp(t, z, p);
        mpz_tdiv_r_2exp(z, z, p);
        mpz_add(z, z, t);
      }
      if (mpz_cmp(z, N) >= 0) { mpz_sub(z, z, N); }
    };
    auto mulN = [&](mpz_t z, const mpz_t y) { mpz_mul(z, z, y); modN(z); };
    map<u64, vector<u64>> byK;
    set<u64> ns;
    for (auto [k, n] : items) { byK[k].push_back(n); ns.insert(n); }
    map<u64, mpz_t*> xn;
    if (!ns.empty()) {
      mpz_t r, s8, sq;                   // r = x^(n^2), sq = x^(4n+4), s8 = x^8
      mpz_inits(r, s8, sq, nullptr);
      mpz_mod(r, x, N);
      mpz_powm_ui(s8, x, 8, N);
      mpz_set(sq, s8);
      for (u64 n = 1; n <= *ns.rbegin(); n += 2) {
        if (ns.count(n)) {
          mpz_t* v = new mpz_t[1];
          mpz_init_set(*v, r);
          xn[n] = v;
        }
        mulN(r, sq);
        mulN(sq, s8);
      }
      mpz_clears(r, s8, sq, nullptr);
    }
    mpz_t acc, xk, ck, gk;
    mpz_init_set_ui(acc, 1);
    mpz_inits(xk, ck, gk, nullptr);
    if (!byK.empty()) {
      u64 const k0 = byK.begin()->first;
      mpz_set_ui(e, k0 * D);
      mpz_mul(e, e, e);
      mpz_powm(xk, x, e, N);             // x^((k0 D)^2)
      mpz_set_ui(e, (2 * k0 + 1) * D);
      mpz_mul_ui(e, e, D);
      mpz_powm(ck, x, e, N);             // x^((2 k0 + 1) D^2)
      mpz_set_ui(e, 2 * D * D);
      mpz_powm(gk, x, e, N);             // x^(2 D^2)
      for (u64 k = k0; k <= byK.rbegin()->first; ++k) {
        if (auto it = byK.find(k); it != byK.end()) {
          for (u64 n : it->second) {
            mpz_sub(t, xk, *xn[n]);
            if (mpz_sgn(t) < 0) { mpz_add(t, t, N); }
            mulN(acc, t);
          }
        }
        mulN(xk, ck);
        mulN(ck, gk);
      }
    }
    mpz_mod(a, a, N);
    bool const ok = mpz_cmp(acc, a) == 0 && cover;
    u64 const nSingle = items.size() - (primes.size() - items.size());
    u64 const nZero = byK.count(0) ? byK[0].size() : 0;
    printf("%s acc p=%" PRIu64 " B1=%" PRIu64 " B2=%" PRIu64 " D=%" PRIu64 " M=%u (table %u): %zu primes, %zu multiplications (%" PRIu64
           " pairs, %" PRIu64 " across sections, %" PRIu64 " singles, %" PRIu64 " at k=0)%s\n", ok ? "OK" : "FAIL", p, plan.B1, plan.B2, D,
           plan.M, plan.nTable(), primes.size(), items.size(), primes.size() - items.size(), plan.nCross, nSingle, nZero,
           cover ? "" : " NOT COVERED");
    return ok ? 0 : 1;
  }

  if (cmd == "savefile" && argc == 5) {
    u64 const p = strtoull(argv[2], 0, 10);
    size_t const n = (p - 1) / 32 + 1;
    FILE* f = fopen(argv[3], "rb");
    if (!f) { fprintf(stderr, "can't read %s\n", argv[3]); return 2; }
    char header[512];
    u32 crc;
    vector<u32> w(n);
    if (!fgets(header, sizeof(header), f) || fread(&crc, 4, 1, f) != 1 || fread(w.data(), 4, n, f) != n) {
      fprintf(stderr, "short savefile %s\n", argv[3]);
      return 2;
    }
    fclose(f);
    FILE* fo = fopen(argv[4], "wb");
    if (!fo || fwrite(w.data(), 4, n, fo) != n) { fprintf(stderr, "can't write %s\n", argv[4]); return 2; }
    fclose(fo);
    printf("header %s", header);
    return 0;
  }

  if (cmd == "gcd" && argc == 4) {
    u64 const p = strtoull(argv[2], 0, 10);
    mersenne(N, p);
    readWords(a, argv[3], p);
    mpz_gcd(t, a, N);
    gmp_printf("gcd %Zd\n", t);
    return 0;
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

  if (cmd == "plan" && (argc == 6 || argc == 7)) {
    pm1::Stage2Plan const plan = makePlan(argc, argv, 2, 6);
    vector<Item> mine = planItems(plan), here = itemsHere(plan, primesIn(plan.B1, plan.B2));
    vector<u64> const primes = primesIn(plan.B1, plan.B2);
    u64 nTwo, nCross;
    bool const cover = covers(mine, primes, plan, &nTwo, &nCross);
    std::ranges::sort(mine);
    std::ranges::sort(here);
    bool const same = mine == here;
    // Singles have n < D/2:  the items with n > D/2 are the pairs across sections
    u64 const across = std::ranges::count_if(here, [&](const Item& it) { return it.second > plan.D / 2; });
    bool const counts = plan.nPrimes == primes.size() && plan.nNeeded == mine.size() && plan.nPairs == primes.size() - here.size() &&
                        plan.nCross == across && plan.nNeeded + plan.nPairs == plan.nPrimes;
    bool const ok = cover && same && counts;
    printf("%s plan B1=%" PRIu64 " B2=%" PRIu64 " D=%u M=%u (table %u, %" PRIu64 " windows of %" PRIu64 " k): %zu primes, %" PRIu64
           " pairs (%.1f%% paired, %" PRIu64 " across sections), %zu multiplications + %" PRIu64 " k steps%s%s%s\n", ok ? "OK" : "FAIL",
           plan.B1, plan.B2, plan.D, plan.M, plan.nTable(), plan.nWindows(), plan.kWindow, primes.size(), plan.nPairs,
           primes.size() ? 200.0 * plan.nPairs / primes.size() : 0.0, plan.nCross, mine.size(), 2 * (plan.kEnd - plan.kBegin),
           cover ? "" : " NOT COVERED", same ? "" : " ITEMS DIFFER", counts ? "" : " COUNTS DIFFER");
    return ok ? 0 : 1;
  }

  if (cmd == "match" && (argc == 6 || argc == 7)) {
    pm1::Stage2Plan const plan = makePlan(argc, argv, 2, 6);
    u64 const best = maxMatching(plan, primesIn(plan.B1, plan.B2));
    bool const ok = best == plan.nPairs;
    printf("%s match B1=%" PRIu64 " B2=%" PRIu64 " D=%u M=%u: plan %" PRIu64 " pairs, maximum matching %" PRIu64 "\n", ok ? "OK" : "FAIL",
           plan.B1, plan.B2, plan.D, plan.M, plan.nPairs, best);
    return ok ? 0 : 1;
  }

  fprintf(stderr, "bad arguments\n");
  return 2;
}
