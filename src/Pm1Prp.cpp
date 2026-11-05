// Copyright (C) Mihai Preda and George Woltman

#include "Pm1Prp.h"
#include "Gpu.h"
#include "Gcd.h"
#include "Pm1.h"
#include "Stage2.h"
#include "AllocTrac.h"
#include "Args.h"
#include "Signal.h"
#include "timeutil.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>

namespace {

constexpr u32 TEMP_BUFS = 6;         // the fold's running products, its two results, and two for its checks
constexpr u32 MIN_L = 2;
constexpr u32 MAX_L = 16;
constexpr size_t RESERVE_BYTES = size_t(256) << 20;    // as stage 2:  kept back when the free memory is not known, and under -maxAlloc
constexpr size_t HEADROOM_BYTES = size_t(1) << 30;     // kept back from the free memory
// Cost in PRP squarings, for the choice of L (XTX, 40M and 100M):  an accumulator step (one multiplication with u_k already
// transformed, mostly from the fused carry's output), and a multiplication of the fold (both operands to transform)
constexpr double STEP_COST = 1.3;
constexpr double FOLD_MUL_COST = 1.7;
// A PRP that has to go back further than this many verified-check intervals for its accumulator does not:  the P-1 is given up
constexpr u64 MAX_REWIND_CHECKS = 2;

// The PRP's verified-check interval for a block size, without errors (as baseCheckStep in Gpu.cpp)
u64 checkInterval(u32 blockSize) { return blockSize == 200 ? 40'000 : blockSize == 400 ? 160'000 : blockSize == 500 ? 200'000 : 1'000'000; }

#ifdef PRPLL_PM1_FAULTS
// TEST ONLY (fault build):  PRPLL_PM1_PRP_FAULT="<k>:<kind>,..." once per process.  bucket:  flips a bit of the bucket stepped at
// iteration k;  prp:  flips a bit of the PRP residue at step k, before it is used (the PRP Gerbicz check must catch it);  fold:  flips a
// bit of the fold's result at the PRP check of iteration k.  PRPLL_PM1_PRP_L=<L> sets the window length.
struct Fault { u64 k; string kind; bool fired; };

vector<Fault>& faults() {
  static vector<Fault> v = [] {
    vector<Fault> r;
    if (const char* env = getenv("PRPLL_PM1_PRP_FAULT")) {
      for (const string& item : split(env, ',')) {
        auto colon = item.find(':');
        r.push_back({stoull(item.substr(0, colon)), colon == string::npos ? "bucket" : item.substr(colon + 1), false});
      }
    }
    return r;
  }();
  return v;
}

bool faultAt(u64 k, const char* kind) {
  for (Fault& f : faults()) {
    if (f.k == k && f.kind == kind && !f.fired) { f.fired = true; return true; }
  }
  return false;
}
#endif

}

// A residue that may be 1 without a buffer write:  the fold's running products start as 1
struct Pm1Prp::Val {
  Buffer<Word>& buf;
  bool one = true;
};

std::unique_ptr<Pm1Prp> Pm1Prp::make(Gpu& gpu, u64 B1, u64 B2, Pm1Report report) {
  u64 const E = gpu.E;
  if (B1 < 2) { return {}; }
  Saver<P1PrpState> dirOnly{E, 1000, 1};
  vector<u32> exp = pm1::stage1ExponentCached(dirOnly.dir(), E, B1);
  u64 const hiBit = pm1::bitLength(exp) - 1;
  // The last window ends at most MAX_L bits above hiBit, its PRP check at most a block later, and that must come before the PRP's end
  if (hiBit + MAX_L + 2 * gpu.args.blockSize >= E) {
    log("P-1 in the PRP test: B1=%" PRIu64 " needs the PRP residues up to iteration %" PRIu64 ", beyond the PRP test of %" PRIu64
        " itself; no P-1 in this test\n", B1, hiBit, E);
    return {};
  }
  return std::unique_ptr<Pm1Prp>{new Pm1Prp(gpu, B1, B2, std::move(report), std::move(exp), hiBit)};
}

Pm1Prp::Pm1Prp(Gpu& gpu, u64 B1, u64 B2, Pm1Report report, vector<u32> exp, u64 hiBit) :
  gpu{gpu},
  E{gpu.E},
  B1{B1},
  B2{B2},
  report{std::move(report)},
  exp{std::move(exp)},
  hiBit{hiBit},
  saver{E, 1000, gpu.args.nSavefiles + 2}
{
  log("P-1 in the PRP test: B1=%" PRIu64 "%s, %" PRIu64 " bits, done when the PRP passes iteration %" PRIu64 "\n", B1,
      B2 > B1 ? (" B2=" + to_string(B2)).c_str() : "", hiBit + 1, hiBit);
}

// The background thread may still have a savefile of this object's to write
Pm1Prp::~Pm1Prp() { gpu.background->waitEmpty(); }

void Pm1Prp::release() {
  bucket.clear();
  touched.clear();
  u0.reset();
  allowance.reset();
}

u32 Pm1Prp::chooseL(u64 bitsLeft, u64 itersLeft, u64 maxBufs) const {
  u64 const checkStep = std::min<u64>(checkInterval(gpu.args.blockSize), std::max<u64>(E / 8, gpu.args.logStep));
  double const folds = double(itersLeft) / checkStep + 2;
  u32 best = 0;
  double bestCost = 0;
  for (u32 l = MIN_L; l <= MAX_L && (u64(1) << l) + 1 + TEMP_BUFS <= maxBufs; ++l) {
    double const cost = double(bitsLeft) / l * STEP_COST + folds * (4.0 * (u64(1) << l) * FOLD_MUL_COST + 3.0 * l);
    if (!best || cost < bestCost) {
      best = l;
      bestCost = cost;
    }
  }
  return best;
}

// The buckets for this PRP:  as many as the memory budget allows (as for stage 2:  the free memory of the device less 1 GB, or 90% of
// its RAM less what prpll holds when the free memory can't be queried; -maxAlloc caps it), and no more than the cheapest L needs.
bool Pm1Prp::alloc(u64 fromK) {
  const Args& args = gpu.args;
  size_t const bufBytes = size_t(gpu.N) * gpu.fft.WordSize;
  cl_device_id const dev = gpu.shared.context->deviceId();
  const void* const devKey = gpu.shared.context->get();
  double const GB = double(1u << 30);
  double const held = double(AllocTrac::totalAllocBytes(devKey));
#ifdef CUDA_BACKEND
  bool const haveFree = true;
#else
  bool const haveFree = hasFreeMemInfo(dev);
#endif
  double const freeBytes = haveFree ? double(getFreeMem(dev)) : -1;
  float const ramGB = getGpuRamGB(dev);
  double budget = haveFree ? freeBytes - HEADROOM_BYTES : ramGB > 0 ? ramGB * GB * 0.9 - held - RESERVE_BYTES : 0;
  if (args.maxAlloc) { budget = std::min(budget, double(args.maxAlloc) - held - RESERVE_BYTES); }
  u64 const maxBufs = budget > 0 ? u64(budget / bufBytes) : 0;

  u64 const itersLeft = hiBit >= fromK ? hiBit + 1 - fromK : 0;
  L = chooseL(itersLeft, itersLeft, maxBufs);
#ifdef PRPLL_PM1_FAULTS
  if (const char* env = getenv("PRPLL_PM1_PRP_L")) { L = u32(atoi(env)); }
#endif
  if (!L) {
    log("P-1 in the PRP test: %.2f GB of GPU memory (%s) is too little for %u buffers of %.0f MB\n", std::max(0.0, budget) / GB,
        haveFree ? "free less 1 GB" : "90% of the RAM less what is held", 4 + 1 + TEMP_BUFS, bufBytes / double(1u << 20));
    return false;
  }
  u32 const nBuckets = 1u << L;
  log("P-1 in the PRP test: L=%u, %u buckets + %u buffers of %.0f MB (%.2f GB, budget %.2f GB), about %" PRIu64
      " accumulator multiplications for the %" PRIu64 " bits left\n", L, nBuckets, 1 + TEMP_BUFS, bufBytes / double(1u << 20),
      double(nBuckets + 1 + TEMP_BUFS) * bufBytes / GB, std::max(0.0, budget) / GB, itersLeft / L + 1, itersLeft);

  // Without -maxAlloc, AllocTrac's default limit (for PRP) is raised for these buffers
  if (!args.maxAlloc) { allowance.emplace(devKey, size_t(nBuckets + 1 + TEMP_BUFS) * bufBytes + RESERVE_BYTES); }
  u32 const wordBufSize = gpu.N * gpu.fft.WordSize / sizeof(Word);
  try {
    TimeInfo* const t = gpu.profile.make("pm1Bucket");
    bucket.reserve(nBuckets);
    for (u32 i = 0; i < nBuckets; ++i) { bucket.emplace_back(t, &gpu.queue, wordBufSize); }
    u0 = std::make_unique<Buffer<Word>>(gpu.profile.make("pm1U0"), &gpu.queue, wordBufSize);
    // The fold's buffers:  allocated now so that a fold can't run out of memory
    for (u32 i = 0; i < TEMP_BUFS; ++i) { temp.emplace_back(t, &gpu.queue, wordBufSize); }
  } catch (const std::bad_alloc&) {
    release();
    log("P-1 in the PRP test: the %u buffers could not be allocated\n", nBuckets + 1 + TEMP_BUFS);
    return false;
  }
  touched.assign(nBuckets, false);
  return true;
}

u32 Pm1Prp::windowAt(u64 w) const {
  u32 v = 0;
  for (u32 i = 0; i < L; ++i) { v |= u32(pm1::testBit(exp, w + i)) << i; }
  return v;
}

void Pm1Prp::startSegment(u64 at) {
  segStart = at;
  next = at;
  stepped = false;
  std::fill(touched.begin(), touched.end(), false);
  if (!X.empty()) {
    gpu.writeIn(bucket[1], X);
    touched[1] = true;
  }
}

// With a PRP check (k, prp64), the accumulator savefile records it (stage 3):  a restart does not try again
void Pm1Prp::giveUp(const string& why, u64 k, u64 prp64) {
  log("P-1 in the PRP test is given up: %s.  The PRP test goes on.\n", why.c_str());
  release();
  active = false;
  done = true;
  if (k != u64(-1)) { save(k, prp64, 3); }
}

void Pm1Prp::save(u64 k, u64 prp64, u32 stage) {
  P1PrpState state{.exponent=E, .B1=B1, .B2=B2, .k=k, .prp64=prp64, .next=next, .stage=stage, .nErrors=nErrors,
                   .data=X.empty() ? makeWords(E, 1) : X};
  // In the background thread, before the PRP savefile of the same check (queued after this):  a restart finds the PRP savefile of an
  // accumulator savefile, or goes back to the one before
  (*gpu.background)([this, state = std::move(state)] { saver.save(state); });
}

PRPState Pm1Prp::sync(PRPState state) {
  release();
  active = false;
  stage2Due = false;
  finalCheck = u64(-1);
  if (done) { return state; }

  vector<u64> const ks = saver.listK();

  // A P-1 that is finished, or waits for stage 2
  if (!ks.empty()) {
    std::optional<P1PrpState> last = saver.loadAt(ks.back());
    if (last && last->stage >= 1) {
      done = true;
      if (last->stage == 3) {
        log("P-1 in the PRP test was given up at iteration %" PRIu64 " (see the log of that run)\n", last->k);
      } else if (last->B1 != B1 || last->B2 != B2) {
        log("P-1 in the PRP test: already done with B1=%" PRIu64 " B2=%" PRIu64 " (not B1=%" PRIu64 " B2=%" PRIu64 ")\n",
            last->B1, last->B2, B1, B2);
      } else if (last->stage == 1) {
        X = last->data;
        nErrors = last->nErrors;
        doneK = last->k;
        done64 = last->prp64;
        next = last->next;
        log("P-1 in the PRP test: stage 1 is done (at iteration %" PRIu64 "), stage 2 resumes\n", last->k);
        stage2Due = true;
        afterSave();
      }
      return state;
    }
  }

  auto start = [&](u64 k) -> bool {
    if (!alloc(k)) {
      done = true;
      return false;
    }
    active = true;
    return true;
  };

  if (state.k == 0) {
    // A new test:  any accumulator savefiles are from an earlier one
    for (u64 k : ks) { saver.removeAt(k); }
    X.clear();
    nErrors = 0;
    if (start(1)) {
      startSegment(1);      // bit 0 of E is 0 (E is even):  the first window starts at u_1
      save(0, state.res64, 0);    // a PRP that has to go back to its start (no savefile before its first check) finds this
    }
    return state;
  }

  u64 const maxRewind = MAX_REWIND_CHECKS * checkInterval(state.blockSize);
  bool triedOther = false;
  for (auto it = ks.rbegin(); it != ks.rend(); ++it) {
    u64 const ka = *it;
    if (ka > state.k) { continue; }
    std::optional<P1PrpState> a = saver.loadAt(ka);
    if (!a || a->stage || a->B1 != B1 || a->next <= ka) { continue; }
    PRPState at = state;
    if (ka != state.k) {
      if (state.k - ka > maxRewind) { break; }
      std::optional<PRPState> s = gpu.loadPRPAt(ka);
      triedOther = true;
      if (!s) { continue; }
      at = std::move(*s);
    }
    if (at.res64 != a->prp64) { continue; }
    if (ka != state.k) {
      log("P-1 in the PRP test: its last savefile is at iteration %" PRIu64 ": the PRP test goes back there from %" PRIu64 "\n", ka,
          state.k);
    }
    X = a->data;
    if (pm1::isOne(X)) { X.clear(); }
    nErrors = std::max(nErrors, a->nErrors);
    if (start(a->next)) {
      startSegment(a->next);
      if (a->next > hiBit) { finalCheck = roundUp(at.k + 1, at.blockSize); }   // all bits are in: only the last fold is left
      log("P-1 in the PRP test: accumulator loaded at %" PRIu64 ", the next window starts at %" PRIu64 "\n", ka, a->next);
    }
    return at;
  }

  if (triedOther) {
    // The PRP state of state.k must be on the GPU again
    gpu.writeState(state.k, state.check, state.blockSize);
    if (gpu.dataResidue() != state.res64) { throw "Error on load"; }
  }
  done = true;
  if (ks.empty()) {
    log("P-1 in the PRP test: not possible here, the PRP test is at iteration %" PRIu64 " and was started without -prpPm1 (P-1 needs"
        " the PRP residues from iteration 1 on).  The PRP test goes on without P-1.\n", state.k);
  } else {
    log("P-1 in the PRP test: no accumulator savefile for the PRP test at iteration %" PRIu64 " (nor one it can go back to, within %"
        PRIu64 " iterations).  The PRP test goes on without P-1.\n", state.k, maxRewind);
  }
  return state;
}

bool Pm1Prp::wantsWords() const {
  if (!active) { return false; }
#ifdef PRPLL_PM1_FAULTS
  for (const Fault& f : faults()) { if (f.k == next && f.kind == "prp" && !f.fired) { return true; } }
#endif
  return next == segStart || !touched[windowAt(next)];
}

bool Pm1Prp::step(u64 k, bool haveWords) {
  assert(active && k == next);
  if (!haveWords) {
    // u_k is the fused carry's output in buf1 (LEAD_WIDTH), as the next squaring takes it:  a multiplication into a bucket
    u32 const v = windowAt(k);
    assert(touched[v] && k != segStart);
    gpu.modMul(bucket[v], gpu.bufData, Gpu::LEAD_WIDTH);
    ++nFused;
    stepped = true;
    lastStep = k;
    next = k + L;
    if (finalCheck == u64(-1) && next > hiBit) { finalCheck = roundUp(k + 1, gpu.args.blockSize); }
    return true;
  }
#ifdef PRPLL_PM1_FAULTS
  if (faultAt(k, "prp")) {
    Words x = gpu.readAndCompress(gpu.bufData);
    x[2] ^= 1u << 9;
    gpu.writeIn(gpu.bufData, x);
    log("TEST FAULT prp at %" PRIu64 "\n", k);
  }
#endif
  ++nWordSteps;
  if (k == segStart) { *u0 << gpu.bufData; }
  u32 const v = windowAt(k);
  bool mul = touched[v];
  if (mul) {
    gpu.modMul(bucket[v], gpu.bufData);
  } else {
    bucket[v] << gpu.bufData;
    touched[v] = true;
  }
#ifdef PRPLL_PM1_FAULTS
  if (faultAt(k, "bucket")) {
    Words x = gpu.readAndCompress(bucket[v]);
    x[1] ^= 1u << 5;
    gpu.writeIn(bucket[v], x);
    log("TEST FAULT bucket %u at %" PRIu64 "\n", v, k);
  }
#endif
  stepped = true;
  lastStep = k;
  next = k + L;
  if (finalCheck == u64(-1) && next > hiBit) { finalCheck = roundUp(k + 1, gpu.args.blockSize); }
  return mul;
}

// a := a * b, with 1 kept as a flag
void Pm1Prp::mulInto(Val& a, const Val& b) {
  if (b.one) { return; }
  if (a.one) {
    a.buf << b.buf;
    a.one = false;
  } else {
    gpu.modMul(a.buf, b.buf);
  }
}

bool Pm1Prp::same(Buffer<Word>& a, Buffer<Word>& b) {
  if (gpu.isEqual(a, b)) { return true; }
  // The word representation of a residue is not unique; compare the values before calling it an error
  Words const x = gpu.readAndCompress(a);
  Words const y = gpu.readAndCompress(b);
  return !x.empty() && x == y;
}

// The fold of the segment that ends at the PRP check of iteration k (see Pm1Prp.h), into out.  False when a check fails.
bool Pm1Prp::fold(u64 k, Words& out) {
  u32 const V = (1u << L) - 1;
  auto bv = [&](u32 v) { return Val{bucket[v], !touched[v]}; };

  // From the top:  r = prod_{v' >= v} B[v'], x = prod_{v >= 1} r_v = prod B[v]^v;  then r = T
  Val r{temp[0]}, x{temp[1]};
  for (u32 v = V; v >= 1; --v) {
    mulInto(r, bv(v));
    mulInto(x, r);
  }
  mulInto(r, bv(0));

  // From the bottom:  r2 = prod_{v' <= v} B[v'], x2 = prod_{v < V} r2_v = prod B[v]^(V - v);  then r2 = T
  Val r2{temp[2]}, x2{temp[3]};
  for (u32 v = 0; v < V; ++v) {
    mulInto(r2, bv(v));
    mulInto(x2, r2);
  }
  mulInto(r2, bv(V));

#ifdef PRPLL_PM1_FAULTS
  if (faultAt(k, "fold") && !x.one) {
    Words w = gpu.readAndCompress(x.buf);
    w[0] ^= 1u << 3;
    gpu.writeIn(x.buf, w);
    log("TEST FAULT fold at %" PRIu64 "\n", k);
  }
#endif

  if (r.one || r2.one) { return false; }      // a segment with steps has a bucket that is not 1
  if (!same(r.buf, r2.buf)) {
    log("P-1 in the PRP test: the two running products differ at %" PRIu64 "\n", k);
    return false;
  }

  // x * x2 * T = T^(2^L)
  Val z{temp[4]};
  mulInto(z, x);
  mulInto(z, x2);
  mulInto(z, r);
  Buffer<Word>& tPow = temp[5];
  tPow << r.buf;
  gpu.squareLoop(tPow, 0, L);
  if (z.one || !same(z.buf, tPow)) {
    log("P-1 in the PRP test: the fold check fails at %" PRIu64 "\n", k);
    return false;
  }

  // T^(2^L) * u_s * X = T * u_{s+nL} * X^(2^L), with u_{s+nL} = u_k^(2^(s+nL-k)) from the PRP residue u_k
  u64 const end = lastStep + L;
  assert(end > k && end - k <= L);
  Val lhs{tPow, false};
  mulInto(lhs, Val{*u0, false});
  Buffer<Word>& uEnd = temp[2];
  uEnd << gpu.bufData;
  gpu.squareLoop(uEnd, 0, end - k);
  Val rhs{uEnd, false};
  mulInto(rhs, r);
  if (!X.empty()) {
    Buffer<Word>& xp = temp[4];
    gpu.writeIn(xp, X);
    mulInto(lhs, Val{xp, false});
    Buffer<Word>& xPow = temp[3];
    xPow << xp;
    gpu.squareLoop(xPow, 0, L);
    mulInto(rhs, Val{xPow, false});
  }
  if (!same(lhs.buf, rhs.buf)) {
    log("P-1 in the PRP test: the accumulator check fails at %" PRIu64 " (segment %" PRIu64 "..%" PRIu64 ")\n", k, segStart, end);
    return false;
  }

  out = x.one ? makeWords(E, 1) : gpu.readAndCompress(x.buf);
  if (out.empty()) { return false; }      // read as zero
  if (pm1::isOne(out)) { out.clear(); }
  return true;
}

Pm1Prp::Outcome Pm1Prp::verified(u64 k, u64 prp64) {
  if (!active) { return CONTINUE; }
  if (stepped) {
    Timer timer;
    Words x;
    if (!fold(k, x)) {
      ++nErrors;
      if (++nSeqErrors > 2) {
        giveUp(to_string(nSeqErrors) + " accumulator check failures in a row", k, prp64);
        return CONTINUE;
      }
      log("P-1 in the PRP test: going back to the last savefile (%u accumulator errors)\n", nErrors);
      return RELOAD;
    }
    nSeqErrors = 0;
    X = std::move(x);
    log("P-1 in the PRP test: %" PRIu64 " of %" PRIu64 " bits in (fold %.1f s)\n", std::min(next, hiBit + 1), hiBit + 1, timer.at());
    startSegment(next);
  }
  if (next > hiBit) { return finish(k, prp64); }
  save(k, prp64, 0);
  return CONTINUE;
}

// The factor of a GCD, logged:  empty when the GCD is 1
static vector<string> factorOf(const Words& g, int stage) {
  if (pm1::isOne(g)) { return {}; }
  string const f = pm1::toDecimal(g);
  int const prime = pm1::isProbablePrime(g);
  log("P-1 found a factor in stage %d: %s (%" PRIu64 " bits, %s)\n", stage, f.c_str(), pm1::bitLength(g),
      prime == 1 ? "prime" : prime == 0 ? "composite: a product of factors, not split" : "too large to test for primality here");
  return {f};
}

Pm1Prp::Outcome Pm1Prp::finish(u64 k, u64 prp64) {
  release();
  active = false;
  done = true;
  doneK = k;
  done64 = prp64;
  Words const x = X.empty() ? makeWords(E, 1) : X;
  log("P-1 in the PRP test: stage 1 done at iteration %" PRIu64 ", %u accumulator errors; res64 %016" PRIx64 " (this process: %" PRIu64
      " steps from the fused carry, %" PRIu64 " from the words)\n", k, nErrors, res64(x), nFused, nWordSteps);
  if (pm1::isOne(x)) {
    giveUp("the stage 1 residue is 1", k, prp64);
    return CONTINUE;
  }

  Words g;
  try {
    Gcd gcd{gpu.shared};
    g = gcd.gcdMinus1(x, u32(E));
    log("P-1 stage 1 GCD: %.1f s\n", gcd.lastStats().seconds);
  } catch (const std::exception& e) {
    giveUp("the GCD failed on the GPU ("s + e.what() + ")", k, prp64);
    return CONTINUE;
  } catch (const string& mes) {
    giveUp("the GCD failed on the GPU (" + mes + ")", k, prp64);
    return CONTINUE;
  } catch (const char* mes) {
    giveUp("the GCD failed on the GPU ("s + mes + ")", k, prp64);
    return CONTINUE;
  }
  vector<string> const factors = factorOf(g, 1);
  if (!factors.empty()) {
    report(B1, 0, factors, nErrors);
    save(k, prp64, 2);
    factor = true;
    return FACTOR;
  }
  log("P-1 stage 1 found no factor (B1=%" PRIu64 ")\n", B1);
  if (B2 > B1) {
    save(k, prp64, 1);
    stage2Due = true;
  } else {
    report(B1, 0, {}, nErrors);
    save(k, prp64, 2);
  }
  return CONTINUE;
}

Pm1Prp::Outcome Pm1Prp::afterSave() {
  if (!stage2Due) { return CONTINUE; }
  stage2Due = false;
  // The PRP savefile of this check must be on disk:  a stop in stage 2 resumes the PRP from it, and stage 2 from its own savefile
  gpu.background->waitEmpty();
  Words const x = X.empty() ? makeWords(E, 1) : X;
  log("P-1 in the PRP test: stage 2 to B2=%" PRIu64 " (the PRP test waits)\n", B2);
  u64 doneB2 = 0;
  vector<string> factors;
  // Stage 2 must not stop the PRP test:  an error in it (an FFT that can't be built, an -fft it refuses, a GPU error) gives the result
  // for B1 alone, as a stage 2 that declines does
  string error;
  try {
    Stage2Result const r = Stage2::run(gpu.shared, E, B1, B2, x);
    if (r.status == Stage2Result::STOPPED) {
      gpu.queue.finish();
      throw "stop requested";
    }
    if (r.status == Stage2Result::DONE) {
      Gcd gcd{gpu.shared};
      Words const g = gcd.gcd(r.acc, u32(E));
      log("P-1 stage 2 GCD: %.1f s\n", gcd.lastStats().seconds);
      if (pm1::bitLength(g) >= E) { throw "the accumulator is zero (the GCD is 2^p - 1)"; }
      factors = factorOf(g, 2);
      if (factors.empty()) { log("P-1 stage 2 found no factor (B1=%" PRIu64 " B2=%" PRIu64 ")\n", B1, B2); }
      doneB2 = B2;
    } else {
      log("P-1 stage 2 (B2=%" PRIu64 ") is not run (%s); the result is for B1=%" PRIu64 " only\n", B2, r.why.c_str(), B1);
    }
  } catch (const char* mes) {
    if (!strcmp(mes, "stop requested") || Signal::stopRequested()) { throw; }
    error = mes;
  } catch (const string& mes) {
    if (Signal::stopRequested()) { throw; }
    error = mes;
  } catch (const std::exception& e) {
    if (Signal::stopRequested()) { throw; }
    error = e.what();
  }
  if (!error.empty()) {
    log("P-1 stage 2 (B2=%" PRIu64 ") failed (%s); the result is for B1=%" PRIu64 " only.  The PRP test goes on.\n", B2, error.c_str(), B1);
    factors.clear();
    doneB2 = 0;
  }
  report(B1, doneB2, factors, nErrors);
  save(doneK, done64, 2);
  gpu.background->waitEmpty();
  Saver<P2State>::clear(E);
  if (!factors.empty()) {
    factor = true;
    return FACTOR;
  }
  return CONTINUE;
}
