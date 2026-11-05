// Copyright (C) Mihai Preda and George Woltman

// P-1 stage 2, paired "n^2" form, as the old gpuowl doP2 (removed in 2f5ea959) on the current transforms.
//
// Every value is kept in words and, when it is an operand, also in "Low" form:  fully forward transformed (fftP, fftMidIn, fftHin),
// the form tailMulLow takes as its second operand.
//   B_r = x^(r^2), the table:  r odd, coprime to D, below D/2;  built by second differences, x^((r+2)^2) = x^(r^2) * x^(4r+4).
//   A_k = x^((kD)^2), C_k = x^((2k+1) D^2), G = x^(2 D^2):  A_{k+1} = A_k * C_k, C_{k+1} = C_k * G.
//   For each needed (k, r):  acc *= A_k - B_r, the difference taken in the transform domain by the tail kernel itself (tailMulDelta,
//   pm1tailmul.cl), the multiplication chained through carryFused as the PRP squarings are.
// Checks (no Gerbicz-type check exists for a product of unrelated factors; see stage2-design.md):  sums of the table buffers (built
// twice, re-verified at each checkpoint), the last table entry and A_k against Gpu::exponentiate, acc != 0, ROE of every carry, and
// one randomly chosen block in DUP_ONE_IN computed twice and compared.  A failed check goes back to the last checkpoint.

#include "Stage2.h"
#include "Gpu.h"
#include "Pm1.h"
#include "Gcd.h"
#include "Memlock.h"
#include "KernelCompiler.h"
#include "Saver.h"
#include "Signal.h"
#include "Args.h"
#include "AllocTrac.h"
#include "TuneEntry.h"
#include "timeutil.h"
#include "log.h"

#include "File.h"

#include <cinttypes>
#include <memory>
#include <cstring>
#include <random>
#include <optional>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace {

constexpr double CHECKPOINT_SECS = 600;
constexpr double LOG_SECS = 60;
constexpr u32 DUP_ONE_IN = 32;            // one block in this many (at random) is computed twice
constexpr u32 ROE_FIRST = 16;             // ROE is sampled in the first blocks of a run and then in one block in ROE_ONE_IN
constexpr u32 ROE_ONE_IN = 8;
constexpr u32 EXTRA_BUFS = 5;             // big buffers besides the table:  A, C, G and S (table build) in Low form, one spare
constexpr size_t RESERVE_BYTES = size_t(256) << 20;    // kept back when the free memory is not known, and under a -maxAlloc cap
constexpr size_t HEADROOM_BYTES = size_t(1) << 30;     // kept back from the free memory
constexpr float MAX_ROE = 0.4f;
constexpr int MAX_RETRIES = 3;
// The untuned FFT candidates are timed (Stage2::timeMul) before a stage 2 with at least this many primes, about B2 = 8M:  about a second
// per candidate (XTX:  7 candidates in 7-13 s at 30M-110M) against a stage 2 of minutes or more
constexpr double PROBE_MIN_PRIMES = 500'000;
constexpr int PROBE_QUICK = 9;            // 600 multiplications per candidate

// The primes in (B1, B2], about
double primesIn(u64 B1, u64 B2) { return B2 / log(double(B2)) - B1 / log(double(B1)); }

bool isZero(const vector<u32>& w) { return std::ranges::all_of(w, [](u32 x) { return x == 0; }); }

string formatETA(double secs) {
  u64 const s = u64(std::max(0.0, secs));
  char buf[64];
  snprintf(buf, sizeof(buf), "%" PRIu64 "d %02u:%02u", s / 86400, u32(s / 3600 % 24), u32(s / 60 % 60));
  return buf;
}

}

// The work of one stage 2 on one Gpu.  A friend of Gpu:  it drives the transform kernels directly.
class Stage2Run {
  Gpu& g;
  const pm1::Stage2Plan& plan;
  const vector<u32>& x;
  u32 const nR;
  size_t const bufSize;            // doubles
  size_t const wordSize;           // Words of a residue buffer

  vector<Buffer<double>> table;    // B_r, Low
  vector<u64> tableSums;
  Buffer<double> lowA, lowC, lowG, lowS;
  Buffer<Word> wX, wA, wC, wS, wR, acc, accDup;

  // ROE of every carry since the table was built:  the count, max, and sums for the mean and sd
  double roeN = 0, roeMax = 0, roeSum = 0, roeSum2 = 0;

  Buffer<double> big(const char* name) { return Buffer<double>{g.profile.make(name), &g.queue, bufSize}; }
  Buffer<Word> words(const char* name) { return Buffer<Word>{g.profile.make(name), &g.queue, wordSize}; }

  // out := in, fully forward transformed
  void toLow(Buffer<double>& out, Buffer<Word>& in) {
    g.fftP(g.buf1, in);
    g.fftMidIn(g.buf1);
    g.fftHin(out, g.buf1);
    g.replay();
  }

  // io *= low
  void mulLow(Buffer<Word>& io, Buffer<double>& low) {
    g.fftP(g.buf1, io);
    g.fftMidIn(g.buf1);
    g.tailMulLow(g.buf1, low);
    g.fftMidOut(g.buf1);
    g.fftW(g.buf3, g.buf1);
    g.carryA(io, g.buf3);
    g.carryB(io);
  }

  u64 sumOf(Buffer<double>& buf) {
    g.bufSumOut.zero();
    g.sum64(g.bufSumOut, u32(bufSize), reinterpret_cast<Buffer<Word>&>(buf));
    return g.bufSumOut.read()[0];
  }

  vector<u32> read(Buffer<Word>& buf) {
    vector<u32> w = g.readAndCompress(buf);
    if (w.empty()) { w = makeWords(g.E, 0); }    // readChecked marks an all-zero buffer by an empty vector
    return w;
  }

  // buf := x^e
  void power(Buffer<Word>& buf, u64 e) {
    g.writeIn(buf, x);
    g.exponentiate(buf, e);
  }

  // The ROE of the carries since the last call; false if too large
  bool roeOK() {
    auto [roeSq, roeMul] = g.readROE();
    for (const RoeInfo& r : {roeSq, roeMul}) {
      if (r.N) {
        roeN += r.N;
        roeMax = std::max(roeMax, r.max);
        roeSum += r.mean * r.N;
        roeSum2 += (r.sd * r.sd + r.mean * r.mean) * r.N;
      }
    }
    return std::max(roeSq.max, roeMul.max) <= MAX_ROE;
  }

public:
  static size_t bufDoubles(const Gpu& g) {
    return size_t(TOTAL_DATA_SIZE(g.fft, g.WIDTH, g.fft.shape.middle, g.SMALL_H, g.in_place, g.pad_size));
  }

  Stage2Run(Gpu& g, const pm1::Stage2Plan& plan, const vector<u32>& x) :
    g{g}, plan{plan}, x{x}, nR{u32(plan.rs.size())},
    bufSize{bufDoubles(g)},
    wordSize{g.N * g.fft.WordSize / sizeof(Word)},
    lowA{big("p2A")}, lowC{big("p2C")}, lowG{big("p2G")}, lowS{big("p2S")},
    wX{words("p2x")}, wA{words("p2wA")}, wC{words("p2wC")}, wS{words("p2wS")}, wR{words("p2wR")},
    acc{words("p2acc")}, accDup{words("p2accDup")}
  {
    table.reserve(nR);
    for (u32 i = 0; i < nR; ++i) { table.push_back(big("p2B")); }
    g.wantROE = ROE_SAMPLES;
  }

  // ROE samples taken before they are read (Gpu's ROE buffer holds 100'000).  The ROE variant of carryFused costs about 3% of a stage 2
  // multiplication (XTX, 1664K FFT64), so as in PRP not every carry is sampled:  the first ROE_FIRST blocks of a run and then one block
  // in ROE_ONE_IN.
  static constexpr u32 ROE_SAMPLES = 50'000;
  void sampleROE(bool on) { g.wantROE = on ? ROE_SAMPLES : 0; }
  bool roeFull() const { return g.roePos + 2 * plan.rs.size() + 16 >= ROE_SAMPLES; }

  // Builds the table B_r = x^(r^2) and its sums.  Returns false if the check of the last entry failed.
  bool buildTable(bool keepSums) {
    if (keepSums) { tableSums.clear(); }
    // R = x^(r^2), S = x^(4r+4), lowG temporarily = x^8 (Low)
    g.writeIn(wR, x);
    power(wS, 8);
    toLow(lowG, wS);
    u32 i = 0;
    u32 const last = plan.rs.back();
    for (u32 r = 1; ; r += 2) {
      if (i < nR && plan.rs[i] == r) {
        toLow(table[i], wR);
        u64 const sum = sumOf(table[i]);
        if (keepSums) {
          tableSums.push_back(sum);
        } else if (sum != tableSums[i]) {
          log("EE stage 2: table entry %u (r = %u) differs between two builds\n", i, r);
          return false;
        }
        ++i;
      }
      if (r == last) { break; }
      toLow(lowS, wS);
      mulLow(wR, lowS);            // x^((r+2)^2)
      mulLow(wS, lowG);            // x^(4(r+2)+4)
    }
    power(wS, u64(last) * last);
    if (read(wR) != read(wS)) {
      log("EE stage 2: table entry r = %u differs from x^(r^2)\n", last);
      return false;
    }
    return true;
  }

  bool tableOK() {
    for (u32 i = 0; i < nR; ++i) {
      if (sumOf(table[i]) != tableSums[i]) {
        log("EE stage 2: table entry %u (r = %u) changed in memory\n", i, plan.rs[i]);
        return false;
      }
    }
    return true;
  }

  // A, C and G for block k
  void startK(u64 k) {
    u64 const D = plan.D;
    power(wA, k * D * k * D);
    toLow(lowA, wA);
    power(wC, (2 * k + 1) * D * D);
    toLow(lowC, wC);
    power(wS, 2 * D * D);
    toLow(lowG, wS);
  }

  bool aOK(u64 k) {
    u64 const kD = k * plan.D;
    power(wS, kD * kD);
    return read(wS) == read(wA);
  }

  void setAcc(const vector<u32>& w) { g.writeIn(acc, w); }
  vector<u32> getAcc() { return read(acc); }

  // The table indices of the needed r of block k
  vector<u32> neededOf(u64 k) const {
    vector<u32> ret;
    for (u32 i = 0; i < nR; ++i) { if (plan.needed(k, i)) { ret.push_back(i); } }
    return ret;
  }

  // a *= A_k - B_r for the table entries `idx`, as a chain through carryFused.  Returns the number of multiplications.
  u32 block(const vector<u32>& idx, Buffer<Word>& a) {
    u32 const total = idx.size();
    for (u32 n = 0; n < total; ++n) {
      if (n == 0) { g.fftP(g.buf1, a); }
      g.fftMidIn(g.buf1);
      g.tailMulDelta(g.buf1, lowA, table[idx[n]]);
      g.fftMidOut(g.buf1);
      if (n + 1 == total) {
        g.fftW(g.buf3, g.buf1);
        g.carryA(a, g.buf3);
        g.carryB(a);
      } else {
        g.doCarry(g.buf1, a);
      }
    }
    return total;
  }

  // A_k, C_k -> A_{k+1}, C_{k+1}
  void stepK() {
    mulLow(wA, lowC);
    toLow(lowA, wA);
    mulLow(wC, lowG);
    toLow(lowC, wC);
  }

  // The kernels of stepK on scratch buffers (A and C unchanged), for timeMul's warm-up
  void stepKWarm() {
    wS << wA;
    mulLow(wS, lowC);
    toLow(lowS, wS);
  }

  // The block computed a second time from accBefore (in accDup) and compared with acc
  bool dupOK(u64 k) {
    block(neededOf(k), accDup);
    return g.isEqual(acc, accDup);
  }

  void saveAccForDup() { accDup << acc; }

  // One accumulator multiplication by A_k - B_i (tailMulDelta) against acc * A_k and acc * B_i (tailMulLow, which buildTable checked
  // against Gpu::exponentiate) subtracted mod 2^E - 1 here.  Exact:  for timeMul, whose chain has no other reference.
  bool deltaOK(u32 i) {
    accDup << acc;
    mulLow(accDup, lowA);
    vector<u32> const pa = read(accDup);
    accDup << acc;
    mulLow(accDup, table[i]);
    vector<u32> const pb = read(accDup);
#ifdef PRPLL_PM1_FAULTS
    if (faultTable) { table[i] << table[0]; }
#endif
    block({i}, acc);
    return canonical(subMod(pa, pb)) == canonical(read(acc));
  }

  // (a - b) mod 2^E - 1 on E-bit words
  vector<u32> subMod(const vector<u32>& a, const vector<u32>& b) const {
    vector<u32> r(a.size());
    u64 borrow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      u64 const d = u64(a[i]) - b[i] - borrow;
      r[i] = u32(d);
      borrow = d >> 63;
    }
    if (borrow) {      // r + 2^E - 1 = r - 1 + 2^E:  subtract 1, then drop the bits above E
      for (size_t i = 0; i < r.size() && r[i]-- == 0; ++i) {}
    }
    if (g.E % 32) { r.back() &= (1u << (g.E % 32)) - 1; }
    return r;
  }

  // 2^E - 1 (all ones) is 0
  vector<u32> canonical(vector<u32> w) const {
    u32 const top = g.E % 32 ? (1u << (g.E % 32)) - 1 : ~0u;
    bool const allOnes = w.back() == top && std::all_of(w.begin(), w.end() - 1, [](u32 v) { return v == ~0u; });
    if (allOnes) { std::ranges::fill(w, 0); }
    return w;
  }

  // The settings the Gpu resolved (clDefines):  INPLACE, TAIL_KERNELS, MULTI_Q, L2_STRIPING
  u32 inPlace() const { return g.in_place; }
  u32 tailKernels() const { return (g.tail_single_wide ? 0 : 2) + (g.tail_single_kernel ? 0 : 1) + (g.zero_q ? 10 : 0); }
  u32 multiQ() const { return g.multi_q; }
  u32 l2Striping() const { return g.l2_striping; }

#ifdef PRPLL_PM1_FAULTS
  // TEST ONLY (PRPLL_STAGE2_FAULT, fault build):  simulated faults
  void corruptTable() { table[0] << table[nR - 1]; }
  void corruptA() { wA << wC; }
  void corruptAcc() { acc << wA; }
  void corruptTableLast() { faultTable = true; }
  void corruptAccOnce() { acc << wA; }
  bool faultTable = false;
#endif
  u32 blockMain(u64 k) { return block(neededOf(k), acc); }
  u32 blockOn(const vector<u32>& idx) { return block(idx, acc); }
  u32 blockOnDup(const vector<u32>& idx) { return block(idx, accDup); }

  void finish() { g.queue.finish(); }

  bool roeCheck() { return roeOK(); }
  void roeReset() { roeOK(); roeN = roeMax = roeSum = roeSum2 = 0; }
  double maxROE() const { return roeMax; }
  RoeInfo roe() const {
    if (!roeN) { return {}; }
    double const mean = roeSum / roeN;
    return {u32(roeN), roeMax, mean, sqrt(std::max(0.0, roeSum2 / roeN - mean * mean))};
  }
};

FFTConfig Stage2::configFor(FFTConfig fft, u64 E, bool keepCarry) {
  // The products are up to twice as large, and so are the carries:  the narrow carry one bit per word sooner.  A tune file's narrow
  // carry (PRP's tune.txt times both widths) is PRP's choice:  stage 2 widens it too.
  bool const twoWidths = fft.shape.fft_type == FFT64 || fft.shape.fft_type == FFT3231 || fft.shape.fft_type == FFT323161;
  bool const mayWiden = fft.carry == CARRY_AUTO || (fft.carry == CARRY_32 && !keepCarry);
  if (twoWidths && mayWiden && E / double(fft.size()) > fft.shape.narrowCarryBPW() - 1) { fft.carry = CARRY_64; }
  return fft;
}

vector<Pm1Fft> Stage2::chooseFFTs(const Args& args, u64 E) {
  // The FFT has to handle E + 0.5 bits per word of its own size (pm1Ffts)
  vector<Pm1Fft> ffts = pm1Ffts(args, E, true);
  // FFT31 and FFT32 alone have no measured bpw limits (fftbpw.h) and failed stage 2's table check at 11.4 bpw:  not supported
  // (pm1Ffts offers them only from an -fft spec)
  std::erase_if(ffts, [](const Pm1Fft& f) { return f.fft.shape.fft_type == FFT31 || f.fft.shape.fft_type == FFT32; });
  if (ffts.empty()) {
    log("P-1 stage 2 of %" PRIu64 " can't run on the -fft given (%s), and P-1 does not replace it: stage 2 does not support M31 or FP32 "
        "alone\n", E, args.fftSpec.c_str());
    throw "P-1: the -fft given can't run stage 2";
  }
  for (Pm1Fft& f : ffts) { f.fft = configFor(f.fft, E, f.source == Pm1Fft::SPEC); }
  return ffts;
}

bool Stage2::hasSavefile(u64 E, u64 B1, u64 B2, const vector<u32>& x) {
  if (!fs::exists(fs::current_path() / (string(P2State::KIND) + '-' + to_string(E)))) { return false; }
  P2State const s = Saver<P2State>{E, 1000, 4}.load();
  return !s.acc.empty() && s.B1 == B1 && s.B2 == B2 && s.x64 == res64(x);
}

Stage2::MulTime Stage2::timeMul(GpuCommon shared, u64 E, const FFTConfig& fft, const vector<KeyVal>& uses, int quick) {
  // As many multiplications as Gpu::timePRP times squarings for this quick, in blocks of BLOCK (stage 2 at D = 2310 has ~120 per k)
  u32 BLOCK = 120, TABLE = 24;      // 24 entries (D = 210's, the smallest table):  more than a GPU cache holds
#ifdef PRPLL_PM1_FAULTS
  // TEST ONLY (fault build):  PRPLL_TIMEMUL_SHAPE=<block>,<table>
  if (const char* sh = getenv("PRPLL_TIMEMUL_SHAPE"); sh && strchr(sh, ',')) { BLOCK = atoi(sh); TABLE = atoi(strchr(sh, ',') + 1); }
#endif
  static constexpr u32 itersByQuick[] = {20000, 12000, 8000, 5000, 3000, 1800, 1200, 900, 600, 400};
  u32 const nBlocks = std::max(2u, itersByQuick[std::clamp(quick, 1, 10) - 1] / BLOCK);

  auto gpu = Gpu::make(E, shared, fft, {}, false, uses);
  Gpu& g = *gpu;

  // A fixed pseudo-random x (a stage 1 residue looks random), the r of D = 2310 for the table and a k as in a real stage 2, so that every
  // FFT computes the same accumulator for E
  vector<u32> x(nWords(E));
  std::mt19937 rng{u32(E)};
  for (u32& w : x) { w = rng(); }
  if (E % 32) { x.back() &= (1u << (E % 32)) - 1; }
  pm1::Stage2Plan plan = pm1::planStage2(1000, 2000, TABLE <= 240 ? 2310 : 4620);
  plan.rs.resize(TABLE);
  Stage2Run s2{g, plan, x};
  if (!s2.buildTable(true)) { return {std::numeric_limits<double>::infinity()}; }
  s2.startK(433);
  s2.setAcc(x);
  vector<u32> idx(BLOCK);
  for (u32 i = 0; i < BLOCK; ++i) { idx[i] = i % TABLE; }

  // Warm-up blocks for WARMUP_SECS, without ROE as the timed blocks (stage 2 samples it in 1 block in 8):  the first builds the kernels
  // the timed blocks use (a build inside the timed window cost 15-35% of a 600-multiplication timing on the XTX), and a GPU that idled
  // while the kernels compiled takes a while to reach its clocks.  They run on the spare accumulator and do not step k, so that the
  // result does not depend on their number.  Then the timed blocks, and one block with ROE.
  constexpr double WARMUP_SECS = 0.25;
  s2.roeReset();
  s2.sampleROE(false);
  s2.saveAccForDup();
  for (Timer warm; ; ) {
    s2.blockOnDup(idx);
    s2.stepKWarm();
    s2.finish();
    if (warm.at() >= WARMUP_SECS) { break; }
  }
  Timer timer;
  for (u32 b = 0; b < nBlocks; ++b) {
    s2.blockOn(idx);
    s2.stepK();
    if (Signal::stopRequested()) { throw "stop requested"; }
  }
  s2.finish();
  double const us = timer.at() * 1e6 / (nBlocks * (BLOCK + 2));      // counted as stage 2 counts them:  the k step as 2
  s2.sampleROE(true);
  s2.blockOn(idx);
#ifdef PRPLL_PM1_FAULTS
  // TEST ONLY (fault build):  PRPLL_TIMEMUL_FAULT=<delta|acc>[,<spec>] makes the multiplication by A_k - B_r wrong (a table entry
  // changed between its two checks) or the accumulator wrong (for the reference check), on every FFT or only on <spec>
  if (const char* f = getenv("PRPLL_TIMEMUL_FAULT"); f && (!strchr(f, ',') || fft.spec() == strchr(f, ',') + 1)) {
    string const kind = strchr(f, ',') ? string(f, strchr(f, ',')) : string(f);
    log("timeMul TEST FAULT: '%s' on %s\n", kind.c_str(), fft.spec().c_str());
    if (kind == "delta") { s2.corruptTableLast(); }
    if (kind == "acc") { s2.corruptAccOnce(); }
  }
#endif
  bool const deltaOK = s2.deltaOK(TABLE - 1);
  bool const roeOK = s2.roeCheck();
  vector<u32> const a = s2.getAcc();
  if (!roeOK) { log("%s: stage 2 ROE %.3f is too large\n", fft.spec().c_str(), s2.maxROE()); }
  if (!deltaOK) { log("%s: a stage 2 multiplication by A_k - B_r differs from A_k and B_r multiplied apart\n", fft.spec().c_str()); }
  bool const ok = roeOK && deltaOK && !isZero(a);
  return {ok ? us : std::numeric_limits<double>::infinity(), res64(a), s2.maxROE(), s2.inPlace(), s2.tailKernels(), s2.multiQ(),
          s2.l2Striping()};
}

void Stage2::rankUntuned(GpuCommon shared, u64 E, vector<Pm1Fft>& ffts) {
  // The ranking of the same candidates (nearby exponents have the same smallest fitting FFT of each type) is kept for the run
  static std::mutex mutex;
  static std::map<string, vector<string>> rankings;
  string key;
  for (const Pm1Fft& f : ffts) { key += f.fft.spec() + ' '; }
  std::unique_lock lock{mutex};
  auto it = rankings.find(key);
  if (it == rankings.end()) {
    struct Timed { double us; u64 res64; };
    vector<Timed> timed;
    Timer timer;
    for (const Pm1Fft& f : ffts) {
      Timed t{std::numeric_limits<double>::infinity(), 0};
      try {
        int quick = PROBE_QUICK;
#ifdef PRPLL_PM1_FAULTS
        // TEST ONLY (fault build):  PRPLL_STAGE2_PROBE_QUICK=<1..10> times the candidates longer or shorter
        if (const char* q = getenv("PRPLL_STAGE2_PROBE_QUICK")) { quick = atoi(q); }
#endif
        MulTime const m = timeMul(shared, E, f.fft, f.uses, quick);
        t = {m.us, m.res64};
      } catch (const string& mes) {
        log("Stage 2 FFT %s can't run: %s\n", f.fft.spec().c_str(), mes.c_str());
      } catch (const std::exception& e) {
        log("Stage 2 FFT %s can't run: %s\n", f.fft.spec().c_str(), e.what());
      }
      timed.push_back(t);
    }
    // Every FFT computes the same accumulator:  a candidate whose result differs from that of two or more others is wrong
    for (Timed& t : timed) {
      if (!std::isfinite(t.us)) { continue; }
      auto const same = std::ranges::count_if(timed, [&](const Timed& o) { return std::isfinite(o.us) && o.res64 == t.res64; });
      auto const other = std::ranges::count_if(timed, [&](const Timed& o) { return std::isfinite(o.us) && o.res64 != t.res64; });
      if (same == 1 && other >= 2) {
        log("EE stage 2 FFT %s computed a wrong accumulator (%016" PRIx64 "): not used\n",
            ffts[&t - timed.data()].fft.spec().c_str(), t.res64);
        t.us = std::numeric_limits<double>::infinity();
      }
    }
    vector<u32> order(ffts.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::stable_sort(order, [&](u32 a, u32 b) { return timed[a].us < timed[b].us; });
    string times;
    vector<string> specs;
    for (u32 i : order) {
      char buf[96];
      snprintf(buf, sizeof(buf), "%s%s %.0f", times.empty() ? "" : ", ", ffts[i].fft.spec().c_str(), timed[i].us);
      times += buf;
      specs.push_back(ffts[i].fft.spec());
    }
    log("Stage 2 untuned FFTs for %" PRIu64 ", us per multiplication: %s; timed in %.1f s (-tune pm1 stores a choice per exponent range)\n",
        E, times.c_str(), timer.at());
    it = rankings.emplace(key, std::move(specs)).first;
  }
  // The same order (an FFT that failed last, still a fallback)
  vector<Pm1Fft> ranked;
  for (const string& spec : it->second) {
    ranked.push_back(*std::ranges::find_if(ffts, [&](const Pm1Fft& f) { return f.fft.spec() == spec; }));
  }
  ffts = std::move(ranked);
}

Stage2Result Stage2::run(GpuCommon shared, u64 E, u64 B1, u64 B2, const vector<u32>& x, u32 D) {
  const Args& args = *shared.args;
  if (B1 < 11 || B2 <= B1 || B2 > (u64(1) << 32) - 8192) {
    return {.status = Stage2Result::DECLINED, .why = "stage 2 needs 11 <= B1 < B2 < 2^32 - 8192"};
  }
  if (x.size() != nWords(E) || isZero(x)) {
    return {.status = Stage2Result::DECLINED, .why = "bad stage 1 residue"};
  }

  // The FFTs to try.  An -fft that can't run stage 2 throws (the user may be testing that FFT); with none fitting otherwise, decline.
  vector<Pm1Fft> ffts;
  try {
    ffts = chooseFFTs(args, E);
  } catch (const char* mes) {
    if (strcmp(mes, "No FFT")) { throw; }
    return {.status = Stage2Result::DECLINED, .why = "no FFT handles the exponent with stage 2's margin"};
  }
  bool const userFft = !args.fftSpec.empty();

  // One stage 2 per device:  the lock is named by the device's unique id where it has one, so that it does not depend on the
  // OpenCL device numbering (ICD files, platforms) and two cards never share it
  Memlock memlock{args.masterDir.empty() ? fs::current_path() : args.masterDir, args.uid.empty() ? to_string(args.device) : args.uid};

  // Untuned (no -fft, no tune-pm1.txt or tune.txt line for E):  the order of the types depends on the size and the GPU, so time them
  if (ffts.size() > 1 && ffts.front().source == Pm1Fft::UNTUNED && primesIn(B1, B2) >= PROBE_MIN_PRIMES) { rankUntuned(shared, E, ffts); }

  // Kernels compile on first use, so a build failure can come at any point; the next FFT resumes from the last checkpoint (the
  // savefile holds words, whatever the FFT).  Only the FFTs of an -fft spec are tried when there is one.
#ifdef PRPLL_PM1_FAULTS
  // TEST ONLY (fault build):  PRPLL_STAGE2_BUILD_FAULT=<n> makes the first n FFTs fail as if their kernels did not compile
  int nBuildFaults = getenv("PRPLL_STAGE2_BUILD_FAULT") ? atoi(getenv("PRPLL_STAGE2_BUILD_FAULT")) : 0;
#endif
  for (size_t i = 0; i < ffts.size(); ++i) {
    bool const last = i + 1 == ffts.size();
    string err;
    try {
#ifdef PRPLL_PM1_FAULTS
      if (nBuildFaults > 0) {
        --nBuildFaults;
        throw "Can't compile (TEST FAULT) the stage 2 kernels of "s + ffts[i].fft.spec();
      }
#endif
      log("Stage 2 FFT %s (%s)\n", ffts[i].fft.spec().c_str(), ffts[i].sourceName());
      return runOn(shared, E, B1, B2, x, D, ffts[i].fft, ffts[i].uses);
    } catch (const string& mes) {
      if (!isBuildFailure(mes)) { throw; }
      err = mes;
    } catch (const std::exception& e) {
      if (!isBuildFailure(e.what())) { throw; }
      err = e.what();
    }
    if (userFft) {
      log("P-1 stage 2: the kernels of the -fft FFT %s can't be built (%s), and P-1 does not replace it%s\n", ffts[i].fft.spec().c_str(),
          err.c_str(), last ? "" : "; trying the next FFT of the -fft list");
    } else {
      log("Stage 2 can't build FFT %s (%s)%s\n", ffts[i].fft.spec().c_str(), err.c_str(), last ? "" : ", trying the next FFT");
    }
  }
  throw userFft ? "P-1 stage 2: the -fft FFT can't be built" : "P-1 stage 2: no FFT could be built";
}

Stage2Result Stage2::runOn(GpuCommon shared, u64 E, u64 B1, u64 B2, const vector<u32>& x, u32 D, const FFTConfig& fft,
                           const vector<KeyVal>& uses) {
  const Args& args = *shared.args;
  auto gpuPtr = Gpu::make(E, shared, fft, {}, true, uses);
  Gpu& g = *gpuPtr;
  u64 const x64 = res64(x);

  auto saverPtr = std::make_unique<Saver<P2State>>(E, 1000, 4);
  P2State saved = saverPtr->load();
  bool resume = !saved.acc.empty() && saved.B1 == B1 && saved.B2 == B2 && saved.x64 == x64;
  if (!saved.acc.empty() && !resume) {
    log("Stage 2 savefile is for B1=%" PRIu64 " B2=%" PRIu64 " x=%016" PRIx64 ", not this stage 2: starting over\n", saved.B1, saved.B2, saved.x64);
  }

  // Memory:  the table and EXTRA_BUFS more big buffers.  The budget is the memory free on the device now (after this Gpu's own buffers)
  // less HEADROOM_BYTES; where the free memory can't be queried (OpenCL without CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, e.g. Nvidia's or
  // PoCL), 90% of the device RAM less what prpll holds and RESERVE_BYTES.  -maxAlloc, when given, caps it (less what prpll holds and
  // RESERVE_BYTES).  Without -maxAlloc there is no cap:  AllocTrac's default limit (for PRP) is raised by the budget, for this device
  // only, while stage 2 runs.
  size_t const bufBytes = Stage2Run::bufDoubles(g) * sizeof(double);
  cl_device_id const dev = shared.context->deviceId();
  double const GB = double(1u << 30);
  const void* const devKey = shared.context->get();      // AllocTrac's accounting is per device
  double const held = double(AllocTrac::totalAllocBytes(devKey));
#ifdef CUDA_BACKEND
  bool const haveFree = true;      // cuMemGetInfo
#else
  bool const haveFree = hasFreeMemInfo(dev);
#endif
  double const freeBytes = haveFree ? double(getFreeMem(dev)) : -1;
  float const ramGB = getGpuRamGB(dev);
  double budget = haveFree ? freeBytes - HEADROOM_BYTES
                : ramGB > 0 ? ramGB * GB * 0.9 - held - RESERVE_BYTES : 0;
  double const cap = args.maxAlloc ? double(args.maxAlloc) - held - RESERVE_BYTES : -1;
  if (args.maxAlloc) { budget = std::min(budget, cap); }
  char memNote[256];
  if (haveFree) {
    snprintf(memNote, sizeof(memNote), "free %.2f GB - headroom %.2f GB", freeBytes / GB, HEADROOM_BYTES / GB);
  } else {
    snprintf(memNote, sizeof(memNote), "free unknown: 90%% of %.2f GB RAM - held %.2f GB - reserve %.2f GB", ramGB, held / GB,
             RESERVE_BYTES / GB);
  }
  char capNote[128] = "no cap";
  if (args.maxAlloc) {
    snprintf(capNote, sizeof(capNote), "-maxAlloc cap %.2f GB - held %.2f GB - reserve %.2f GB", args.maxAlloc / GB, held / GB,
             RESERVE_BYTES / GB);
  }
  log("Stage 2 memory: %s, %s => budget %.2f GB\n", memNote, capNote, std::max(0.0, budget) / GB);
  u64 const maxR = budget > double(bufBytes) * (EXTRA_BUFS + 1) ? u64(budget / bufBytes) - EXTRA_BUFS : 0;
  if (resume && !D && pm1::stage2nR(saved.D) <= maxR) { D = saved.D; }
  if (resume && D != saved.D) {
    log("Stage 2 savefile is for D=%u, which does not fit now (or another D was asked for): starting over\n", saved.D);
    resume = false;
  }
  if (!saved.acc.empty() && !resume) {
    // The savefiles of another stage 2 (other bounds, residue or D) would be loaded again before this run's own:  remove them
    saverPtr.reset();
    Saver<P2State>::clear(E);
    saverPtr = std::make_unique<Saver<P2State>>(E, 1000, 4);
  }
  Saver<P2State>& saver = *saverPtr;
  if (!D) { D = pm1::chooseStage2D(B1, B2, maxR); }
  if (!D || pm1::stage2nR(D) > maxR) {
    char why[256];
    snprintf(why, sizeof(why), "%.1f GB of GPU memory allow %" PRIu64 " buffers of %.0f MB, stage 2 needs at least %u + %u",
             std::max(0.0, budget) / double(1u << 30), maxR + (maxR ? EXTRA_BUFS : 0), bufBytes / double(1u << 20), pm1::stage2nR(210), EXTRA_BUFS);
    log("Stage 2 declined: %s\n", why);
    return {.status = Stage2Result::DECLINED, .why = why};
  }

  Timer timer;
  pm1::Stage2Plan const plan = pm1::planStage2(B1, B2, D);
  u32 const nR = plan.rs.size();
  u64 const mulsTotal = plan.mulsFrom(plan.kBegin);
  log("Stage 2 B1=%" PRIu64 " B2=%" PRIu64 ": D=%u, %u table buffers of %.0f MB, k %" PRIu64 "..%" PRIu64 ", %" PRIu64 " primes, %" PRIu64
      " pairs (%.1f%% of the primes paired), %" PRIu64 " multiplications (%" PRIu64 " for the primes, %" PRIu64 " for the k steps); plan %.1fs\n",
      B1, B2, D, nR, bufBytes / double(1u << 20), plan.kBegin, plan.kEnd - 1, plan.nPrimes, plan.nPairs,
      plan.nPrimes ? 200.0 * plan.nPairs / plan.nPrimes : 0.0, mulsTotal, plan.nNeeded, 2 * (plan.kEnd - plan.kBegin), timer.reset());

  // Without a -maxAlloc cap, AllocTrac's default limit must not refuse the planned buffers:  this device's limit is raised by the budget
  // (which the device's free memory bounds) while stage 2 runs
  std::optional<AllocTrac::Allowance> allowance;
  if (!args.maxAlloc) { allowance.emplace(devKey, size_t(budget) + RESERVE_BYTES); }

  std::unique_ptr<Stage2Run> s2Ptr;
  try {
    s2Ptr = std::make_unique<Stage2Run>(g, plan, x);
  } catch (const std::bad_alloc&) {
    log("Stage 2 declined: the buffers could not be allocated\n");
    return {.status = Stage2Result::DECLINED, .why = "the buffers could not be allocated"};
  }
  Stage2Run& s2 = *s2Ptr;

  P2State state = resume ? saved : P2State{.exponent = E, .k = plan.kBegin, .B1 = B1, .B2 = B2, .D = D, .x64 = x64, .nMul = 0,
                                           .acc = makeWords(E, 1), .elapsed = 0};
  if (resume) { log("Stage 2 resuming at k=%" PRIu64 " (%" PRIu64 " multiplications done)\n", state.k, state.nMul); }

  std::mt19937_64 rng{std::random_device{}()};

#ifdef PRPLL_PM1_FAULTS
  // TEST ONLY (fault build):  PRPLL_STAGE2_FAULT=<table|A|acc>,<k> simulates one fault after block k (acc: in a block that is then
  // duplicated)
  string faultKind;
  u64 faultK = 0;
  if (const char* f = getenv("PRPLL_STAGE2_FAULT"); f && strchr(f, ',')) {
    faultKind = string(f, strchr(f, ','));
    faultK = stoull(strchr(f, ',') + 1);
    log("Stage 2 TEST FAULT: a simulated '%s' fault after k=%" PRIu64 "\n", faultKind.c_str(), faultK);
  }
#endif
  u64 const mulsBefore = state.nMul;     // -iters counts the multiplications of this run
  Signal signal;
  int retries = 0;

  while (true) {
    // (Re)start from `state`, the last verified point
    bool ok = s2.buildTable(true) && s2.buildTable(false);
    if (ok) {
      log("Stage 2 table of %u built and checked in %.1fs\n", nR, timer.reset());
      s2.startK(state.k);
      s2.setAcc(state.acc);
      s2.roeReset();      // drop the ROE of the set-up
    }

    Timer sinceCheckpoint, sinceLog, sinceStart;
    u64 mulsAtLog = state.nMul;
    u64 const mulsAtStart = state.nMul;
    double elapsedAtStart = state.elapsed;
    u64 k = state.k;
    u64 nMul = state.nMul;
    auto const usPerMul = [&]() { return sinceStart.at() * 1e6 / double(std::max<u64>(1, nMul - mulsAtStart)); };

    while (ok) {
      bool const atEnd = k >= plan.kEnd;
      bool const stop = !atEnd && (signal.stopRequested() || (args.iters && nMul - mulsBefore >= u64(args.iters)));
      if (atEnd || stop || sinceCheckpoint.at() > CHECKPOINT_SECS) {
        // Checkpoint:  every long-lived value verified before the accumulator is saved
        vector<u32> a = s2.getAcc();
        if (isZero(a)) { log("EE stage 2: the accumulator is zero\n"); ok = false; break; }
        if (!s2.roeCheck()) { log("EE stage 2: ROE %.3f\n", s2.maxROE()); ok = false; break; }
        if (!s2.tableOK()) { ok = false; break; }
        if (!s2.aOK(k)) { log("EE stage 2: A_k differs from x^((kD)^2) at k=%" PRIu64 "\n", k); ok = false; break; }
        double const elapsed = elapsedAtStart + sinceCheckpoint.at();
        state = P2State{.exponent = E, .k = k, .B1 = B1, .B2 = B2, .D = D, .x64 = x64, .nMul = nMul, .acc = a, .elapsed = elapsed};
        saver.save(state);
        elapsedAtStart = elapsed;
        sinceCheckpoint.reset();
        retries = 0;
        if (atEnd) {
          log("Stage 2 done: %" PRIu64 " multiplications, %.0fs, %.1f us/mul, max ROE %.3f, ROE %s, acc %016" PRIx64 "\n",
              nMul, elapsed, usPerMul(), s2.maxROE(), s2.roe().toString().c_str(), res64(a));
          if (args.profile) { g.logTimeKernels(); }
          return {.status = Stage2Result::DONE, .acc = a, .nMul = nMul};
        }
        if (stop) {
          log("Stage 2 stopped at k=%" PRIu64 " (%" PRIu64 " of %" PRIu64 " multiplications), %.1f us/mul, max ROE %.3f, ROE %s\n", k, nMul,
              mulsTotal, usPerMul(), s2.maxROE(), s2.roe().toString().c_str());
          if (args.profile) { g.logTimeKernels(); }
          return {.status = Stage2Result::STOPPED, .nMul = nMul};
        }
      }

#ifdef PRPLL_PM1_FAULTS
      bool const fault = !faultKind.empty() && k == faultK;
      string const kind = faultKind;
      if (fault) { faultKind.clear(); }        // once
      bool const dup = rng() % DUP_ONE_IN == 0 || (fault && kind == "acc");
#else
      bool const dup = rng() % DUP_ONE_IN == 0;
#endif
      s2.sampleROE(k - state.k < ROE_FIRST || rng() % ROE_ONE_IN == 0);
      if (dup) { s2.saveAccForDup(); }
      nMul += s2.blockMain(k) + 2;
#ifdef PRPLL_PM1_FAULTS
      if (fault && kind == "acc") { s2.corruptAcc(); }
#endif
      if (dup && !s2.dupOK(k)) { log("EE stage 2: block k=%" PRIu64 " gave two different results\n", k); ok = false; break; }
      s2.stepK();
#ifdef PRPLL_PM1_FAULTS
      if (fault && kind == "table") { s2.corruptTable(); }
      if (fault && kind == "A") { s2.corruptA(); }
#endif
      ++k;
      if (s2.roeFull() && !s2.roeCheck()) { log("EE stage 2: ROE %.3f\n", s2.maxROE()); ok = false; break; }

      if (sinceLog.at() > LOG_SECS) {
        s2.finish();
        double const secs = sinceLog.reset();
        u64 const done = nMul - mulsAtLog;
        mulsAtLog = nMul;
        log("Stage 2 k=%" PRIu64 " %5.2f%% %.0f us/mul, ETA %s, max ROE %.3f\n", k, 100.0 * nMul / mulsTotal,
            secs / std::max<u64>(done, 1) * 1e6, formatETA(secs / std::max<u64>(done, 1) * double(mulsTotal - nMul)).c_str(), s2.maxROE());
      }
    }

    if (++retries > MAX_RETRIES) { throw "stage 2: too many errors"; }
    log("Stage 2 going back to k=%" PRIu64 " (retry %d)\n", state.k, retries);
  }
}

#ifdef PRPLL_PM1_FAULTS
// TEST ONLY (fault build):  spec "E,fft[+fft...][,quick[,rounds]]" logs Stage2::timeMul of fft at E
void timeMulTest(GpuCommon shared, const char* spec) {
  vector<string> f;
  for (const char* p = spec; ; ) {
    const char* comma = strchr(p, ',');
    f.push_back(comma ? string(p, comma) : string(p));
    if (!comma) { break; }
    p = comma + 1;
  }
  u64 const E = stoull(f[0]);
  if (f[1] == "probe") {     // "E,probe[,gcd]":  Stage2::rankUntuned as stage 2 runs it, with a Gcd object alive (as in a P-1 task) if gcd
    std::optional<Gcd> gcd;
    if (f.size() > 2 && f[2] == "gcd") { gcd.emplace(shared); gcd->gcdMinus1(makeWords(E, 3), u32(E)); }
    vector<Pm1Fft> ffts = Stage2::chooseFFTs(*shared.args, E);
    Stage2::rankUntuned(shared, E, ffts);
    return;
  }
  int const quick = f.size() > 2 ? stoi(f[2]) : 7;
  int const rounds = f.size() > 3 ? stoi(f[3]) : 1;
  for (int i = 0; i < rounds; ++i) {
    for (const string& spec : split(f[1], '+')) {      // several FFTs in one process, in this order
      FFTConfig const fft = Stage2::configFor(FFTConfig{spec}, E);
      Stage2::MulTime const m = Stage2::timeMul(shared, E, fft, TuneEntry::usesFor(*shared.args, fft), quick);
      log("timeMulTest %" PRIu64 " %s quick=%d: %.1f us/mul, res64 %016" PRIx64 ", ROE %.3f\n", E, fft.spec().c_str(), quick, m.us, m.res64, m.maxROE);
    }
  }
}

// TEST ONLY (fault build, test/pm1/run_stage2.sh)
void stage2Test(GpuCommon shared, const char* spec) {
  vector<string> f;
  for (string item; ;) {
    const char* comma = strchr(spec, ',');
    item = comma ? string(spec, comma) : string(spec);
    f.push_back(item);
    if (!comma) { break; }
    spec = comma + 1;
  }
  if (f.size() < 5) { throw "PRPLL_STAGE2_TEST=E,B1,B2,x.bin,acc.bin[,D]"; }
  u64 const E = stoull(f[0]), B1 = stoull(f[1]), B2 = stoull(f[2]);
  u32 const D = f.size() > 5 ? u32(stoul(f[5])) : 0;

  vector<u32> const x = File::openReadThrow(f[3]).read<u32>(nWords(E));
  LogContext const pushContext(std::to_string(E));
  Stage2Result const r = Stage2::run(shared, E, B1, B2, x, D);
  log("stage2Test: %s, %" PRIu64 " multiplications\n", r.status == Stage2Result::DONE ? "done" : r.status == Stage2Result::STOPPED ? "stopped" : r.why.c_str(), r.nMul);
  if (r.status == Stage2Result::DONE) {
    File fo = File::openWrite(f[4]);
    fo.write(r.acc.data(), r.acc.size() * sizeof(u32));
  }
}
#endif
