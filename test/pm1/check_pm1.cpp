// Checks of src/Pm1.cpp: the sieve and the stage 1 exponent against GMP, the mprime save file name against Prime95's tempFileName,
// and the layout and checksum of the save file written for mprime.

#include "Pm1.h"

#include <gmp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

using namespace pm1;

static int bad = 0;
static void check(bool ok, const char *what) { if (!ok) { printf("FAIL: %s\n", what); ++bad; } }

static vector<u32> toWords(const mpz_t z) {
  vector<u32> w(mpz_sizeinbase(z, 2) / 32 + 2, 0);
  size_t count = 0;
  mpz_export(w.data(), &count, -1, 4, 0, 0, z);
  w.resize(count);
  return w;
}

static void testSieve() {
  struct R { u64 lo, hi; } ranges[] = {{0, 100}, {2, 2}, {3, 3}, {0, 1000000}, {999983, 1100000}, {2000000 - 5, 2000000 + (1u << 21)},
                                       {4294967000ull, 4294968000ull}, {10, 11}, {1, 2}, {5, 4}};
  for (R r : ranges) {
    vector<u64> a, b;
    forEachPrime(r.lo, r.hi, [&](u64 q) { a.push_back(q); });
    mpz_t z;
    mpz_init_set_ui(z, r.lo ? r.lo - 1 : 0);
    while (true) { mpz_nextprime(z, z); if (mpz_cmp_ui(z, r.hi) > 0) { break; } b.push_back(mpz_get_ui(z)); }
    mpz_clear(z);
    char what[100];
    snprintf(what, sizeof(what), "primes in [%llu, %llu]", (unsigned long long) r.lo, (unsigned long long) r.hi);
    check(a == b, what);
  }
  u64 n = 0;
  forEachPrime(0, 10000000, [&](u64) { ++n; });
  check(n == 664579, "pi(10^7)");
}

static void testExponent() {
  struct E { u64 p, B1; } es[] = {{3, 1}, {3, 2}, {7, 10}, {127, 100}, {4423, 1000}, {4423, 3000000}, {86243, 100000}, {9689, 1},
                                  {1000003, 250000}, {100000007ull, 5000000}, {1257787, 20000000}};
  for (E e : es) {
    mpz_t g;
    mpz_init_set_ui(g, 2 * e.p);          // Prime95's calc_exp: 2n times the largest power <= B1 of every prime <= B1
    mpz_t q;
    mpz_init(q);
    for (mpz_set_ui(q, 2); mpz_cmp_ui(q, e.B1) <= 0; mpz_nextprime(q, q)) {
      unsigned long v = mpz_get_ui(q), qq = v;
      while (v <= e.B1 / qq) { v *= qq; }
      mpz_mul_ui(g, g, v);
    }
    vector<u32> got = stage1Exponent(e.p, e.B1);
    char what[100];
    snprintf(what, sizeof(what), "exponent p=%llu B1=%llu", (unsigned long long) e.p, (unsigned long long) e.B1);
    check(got == toWords(g), what);
    check(bitLength(got) == mpz_sizeinbase(g, 2), "bitLength");
    u64 const n = bitLength(got);
    for (u64 i : {u64(0), u64(1), n / 2, n - 1, n, n + 40}) { check(testBit(got, i) == (i < n && mpz_tstbit(g, i)), "testBit"); }
    mpz_clear(q);
    mpz_clear(g);
  }
}

static void testNames() {
  // Output of Prime95's tempFileName (commonc.c) for a P-1 savefile of 2^p-1
  struct N { u64 p; const char *name; } names[] = {
    {9689, "m0009689"}, {11213, "m0011213"}, {10000019, "mA000019"}, {25000003, "mP000003"}, {35000017, "m0A00017"},
    {40000003, "m0F00003"}, {61000003, "m00B0003"}, {70000027, "m00K0027"}, {79999999, "m99T9999"}, {80000023, "m80000023"},
    {82997591, "m82997591"}, {124647911, "m124647911"}, {1257787, "m1257787"}, {61000000, "m00B0000"}, {34999999, "mY999999"},
    {35000000, "m0A00000"}, {9999999, "m9999999"}};
  for (N n : names) { check(mprimeSaveName(n.p) == n.name, n.name); }
#ifdef MPRIME_SRC
  if (strlen(MPRIME_SRC)) {
    // Compile the real tempFileName out of Prime95's source and compare (needs the sources named by MPRIME_SRC)
    string const script = string("awk 'NR>=4300' ") + MPRIME_SRC + "/commonc.c | awk '/^void tempFileName/{f=1} f{print} f&&/^}/{exit}' > savefile-test-ref_body.c";
    if (system(script.c_str()) == 0) {
      std::ofstream("savefile-test-ref.c") <<
        "#include <stdio.h>\n#include <string.h>\n#include <stdlib.h>\n#include <math.h>\n"
        "enum { WORK_FACTOR, WORK_ECM, WORK_PMINUS1, WORK_PFACTOR, WORK_PPLUS1, WORK_CERT };\n"
        "struct work_unit { unsigned long n; int work_type; unsigned skip_curves; long c; double k; char extension[32]; };\n"
        "#include \"savefile-test-ref_body.c\"\n"
        "int main(int argc, char** argv) { for (int i = 1; i < argc; ++i) { struct work_unit w = {strtoul(argv[i], 0, 10), WORK_PMINUS1, 0, -1, 1.0, {0}};\n"
        "  char buf[100]; tempFileName(&w, buf); printf(\"%s\\n\", buf); } }\n";
      string args;
      for (u64 p = 1000003; p < 130000000; p += 1377713) { args += " " + to_string(p); }
      if (system("gcc -w savefile-test-ref.c -o savefile-test-ref -lm") == 0) {
        FILE *f = popen(("./savefile-test-ref" + args).c_str(), "r");
        char line[100];
        u64 p = 1000003;
        int n = 0;
        while (f && fgets(line, sizeof(line), f)) { line[strcspn(line, "\n")] = 0; check(mprimeSaveName(p) == line, "name vs Prime95's tempFileName"); p += 1377713; ++n; }
        if (f) { pclose(f); }
        printf("compared %d save file names with Prime95's tempFileName\n", n);
      }
    }
    remove("savefile-test-ref_body.c"); remove("savefile-test-ref.c"); remove("savefile-test-ref");
  }
#endif
}

template<typename T> static T rd(const vector<char>& b, size_t off) { T x; memcpy(&x, b.data() + off, sizeof x); return x; }

static void testSaveFile() {
  fs::path const dir = "savefile-test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  u64 const p = 9689, B1 = 123456789012ull;
  vector<u32> const x = {0xdeadbeef, 1, 0x80000000u, 0, 0};      // the trailing zero words are dropped
  writeMprimeStage1(dir, p, B1, x);

  std::ifstream in(dir / mprimeSaveName(p), std::ios::binary);
  vector<char> b((std::istreambuf_iterator<char>(in)), {});
  check(rd<u32>(b, 0) == 0x317a394b, "magic");
  check(rd<u32>(b, 4) == 8, "version");
  check(rd<double>(b, 8) == 1.0, "K");
  check(rd<u32>(b, 16) == 2, "b");
  check(rd<u32>(b, 20) == p, "n");
  check(rd<i32>(b, 24) == -1, "c");
  check(!strcmp(&b[28], "S5"), "stage string");
  check(rd<double>(b, 40) == 1.0, "percent complete");
  check(b.size() == 52 + 4 + 8 + 8 + 4 + 4 + 3 * 4, "file size");

  size_t o = 52;
  u32 sum = 0;
  u32 const state = rd<u32>(b, o); o += 4; sum += state; check(state == 5, "state is done");
  u64 const bDone = rd<u64>(b, o); o += 8; sum += u32((bDone >> 32) + bDone); check(bDone == B1, "B_done");
  u64 const cDone = rd<u64>(b, o); o += 8; sum += u32((cDone >> 32) + cDone); check(cDone == B1, "C_done");
  u32 const haveX = rd<u32>(b, o); o += 4; sum += haveX; check(haveX == 1, "have_x");
  u32 const len = rd<u32>(b, o); o += 4; sum += len; check(len == 3, "giant length");
  for (u32 i = 0; i < len; ++i) { u32 const w = rd<u32>(b, o); o += 4; sum += w; check(w == x[i], "giant word"); }
  sum += len;                                                    // write_giant adds the word count a second time
  check(o == b.size(), "no trailing bytes");
  check(rd<u32>(b, 48) == sum, "checksum");
  check(!fs::exists(dir / (mprimeSaveName(p) + ".new")), "the .new file is renamed away");

  check(mprimePminus1Line("ABCDEF0123456789ABCDEF0123456789", p, B1, 2 * B1) == "Pminus1=ABCDEF0123456789ABCDEF0123456789,1,2,9689,-1,123456789012,246913578024\n", "line with AID");
  check(mprimePminus1Line("", p, 1000000, 30000000) == "Pminus1=1,2,9689,-1,1000000,30000000\n", "line without AID");
  check(mprimePminus1Line("N/A", p, 1000000, 30000000) == "Pminus1=1,2,9689,-1,1000000,30000000\n", "line with N/A");
  appendMprimeWork(dir, mprimePminus1Line("", p, 1000000, 30000000));
  appendMprimeWork(dir, mprimePminus1Line("", p + 2, 1000000, 30000000));
  std::ifstream add(dir / "worktodo.add");
  string const all((std::istreambuf_iterator<char>(add)), {});
  check(all == "Pminus1=1,2,9689,-1,1000000,30000000\nPminus1=1,2,9691,-1,1000000,30000000\n", "worktodo.add appended");

  bool threw = false;
  try { writeMprimeStage1(dir, p, B1, {0, 0}); } catch (const char *) { threw = true; }
  check(threw, "a zero residue is refused");
  fs::remove_all(dir);
}

// The Gerbicz check's sum of exponent chunks (addBits) and its high part (shiftRight64), against GMP
static void testChunks() {
  vector<u32> const E = stage1Exponent(1000003, 50000);
  u64 const nBits = bitLength(E);
  mpz_t e, c, chunk, h;
  mpz_inits(e, c, chunk, h, nullptr);
  mpz_import(e, E.size(), -1, 4, 0, 0, E.data());
  srand(1);
  for (u32 L : {1u, 2u, 31u, 32u, 33u, 64u, 268u, 3000u}) {
    vector<u32> acc;
    mpz_set_ui(c, 0);
    for (int j = 0; j < 40; ++j) {
      u64 const lo = u64(rand()) % (nBits - L);
      addBits(acc, E, lo, L);
      mpz_tdiv_q_2exp(chunk, e, lo);
      mpz_tdiv_r_2exp(chunk, chunk, L);
      mpz_add(c, c, chunk);
    }
    mpz_t got;
    mpz_init(got);
    mpz_import(got, acc.size(), -1, 4, 0, 0, acc.data());
    mpz_tdiv_q_2exp(h, c, L);
    char what[100];
    snprintf(what, sizeof(what), "sum of 40 chunks of %u bits", L);
    check(mpz_cmp(got, c) == 0, what);
    snprintf(what, sizeof(what), "high part of the sum, L = %u", L);
    check(mpz_fits_ulong_p(h) && shiftRight64(acc, L) == mpz_get_ui(h), what);
    mpz_clear(got);
  }
  mpz_clears(e, c, chunk, h, nullptr);
}

// The exponent cache:  written once, read back equal, rewritten when damaged
static void testExponentCache() {
  fs::path const dir = "exp-cache-test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  vector<u32> const want = stage1Exponent(1000003, 100000);
  check(stage1ExponentCached(dir, 1000003, 100000) == want, "exponent cache: computed");
  check(fs::exists(dir / "B1-100000.exp"), "exponent cache: file written");
  check(stage1ExponentCached(dir, 1000003, 100000) == want, "exponent cache: read back");
  {
    std::fstream f(dir / "B1-100000.exp", std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(200);
    f.put('x');
  }
  check(stage1ExponentCached(dir, 1000003, 100000) == want, "exponent cache: damaged file recomputed");
  check(stage1ExponentCached(dir, 1000003, 100000) == want, "exponent cache: rewritten");
  fs::remove_all(dir);
}

int main() {
  testChunks();
  testExponentCache();
  testSieve();
  testExponent();
  testNames();
  testSaveFile();
  printf(bad ? "%d FAILED\n" : "all P-1 helper checks passed\n", bad);
  return bad != 0;
}
