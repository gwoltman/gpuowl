// Copyright (C) Mihai Preda and George Woltman

#include "Gcd.h"
#include "Gpu.h"
#include "FFTConfig.h"
#include "Args.h"
#include "Context.h"
#include "timeutil.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <stdexcept>

// The kernels' work-group size (GW in gcd.cl)
static constexpr u32 GW = 256;
// The leaf's digit capacity (LEAF_NMAX in gcd.cl)
static constexpr u32 LEAF_NMAX = 280;
// The leaf's work-group size (LEAF_W in gcd.cl)
static constexpr u32 LEAF_W = 256;

static u32 roundUp(u32 n, u32 m) { return (n + m - 1) / m * m; }

// The number of work-groups of GW for n work-items (Kernel::setKernelsToExecute counts groups)
static u32 groups(u32 n) { return std::max((n + GW - 1) / GW, 1u); }

// The state of one run: (f, g) in two pairs of digit buffers and two matrix buffers, used alternately.
struct Gcd::Run {
  u32 n;        // digits allocated
  u32 limit;    // digits from limit up are zero; values stay below 2^(62 limit - 1), so the digit at limit stays zero too
  Buffer<i64> f[2], g[2], mat[2];
  Buffer<u32> lens, flag, skewCount;
  Buffer<i64> out;
  int cur = 0;

  Run(Queue* q, u32 n) :
    n{n},
    limit{},
    f{{nullptr, q, n}, {nullptr, q, n}}, g{{nullptr, q, n}, {nullptr, q, n}}, mat{{nullptr, q, 8}, {nullptr, q, 8}},
    lens{nullptr, q, 2}, flag{nullptr, q, 1}, skewCount{nullptr, q, 1}, out{nullptr, q, 8} {}

  Buffer<i64>& F() { return f[cur]; }
  Buffer<i64>& G() { return g[cur]; }
};

#define K(name, ...) name(#name, &compiler, profile.make(#name), &queue, __VA_ARGS__)
#define K1(name, inFile, size, defines) Kernel{name, &compiler, profile.make(name), &queue, "gcd.cl", inFile, size, defines}

Gcd::Gcd(GpuCommon shared, u32 skew) :
  shared{shared},
  queue{*shared.context, false},
  compiler{*shared.args, shared.context, skew ? "-DSKEW=" + std::to_string(skew) : "", true},
  skew{skew},
  K(kToDigits, "gcd.cl", "toDigits", GW),
  K(kInit, "gcd.cl", "init", 64),
  K(kInitJ, "gcd.cl", "init", 64, "-DJACOBI=1"),
  // two of each, one per direction of the ping-pong between the buffers, so that their buffer args stay set
  kApply{K1("kApply0", "apply", GW, ""), K1("kApply1", "apply", GW, "")},
  kApplyJ{K1("kApplyJ0", "apply", GW, "-DJACOBI=1"), K1("kApplyJ1", "apply", GW, "-DJACOBI=1")},
  K(kLengths, "gcd.cl", "lengths", GW),
  K(kSummary, "gcd.cl", "summary", 64),
  K(kSummaryJ, "gcd.cl", "summary", 64, "-DJACOBI=1"),
  K(kNotEqual, "gcd.cl", "notEqual", GW),
  K(kBlockSigns, "gcd.cl", "blockSigns", GW),
  K(kBlockBorrows, "gcd.cl", "blockBorrows", 64),
  K(kToWords, "gcd.cl", "toWords", GW),
  K(kLeaf, "gcd.cl", "leaf", LEAF_W),
  K(kLeafJ, "gcd.cl", "leaf", LEAF_W, "-DJACOBI=1"),
  K(kMulColumns, "gcd.cl", "mulColumns", GW),
  K(kMulDigits, "gcd.cl", "mulDigits", GW),
  K(kAddDigits, "gcd.cl", "addDigits", GW),
  scratchCount{nullptr, &queue, 1}
{
  if (skew > 59) { throw std::invalid_argument("Gcd: skew must be at most 59"); }
}

#undef K
#undef K1

// f = 2^p - 1, g = x - sub, then divsteps until g = 0 (GCD) or f = g (Jacobi).  Bernstein-Yang bound the GCD by
// about 2.88 divsteps per bit.  The Jacobi steps have no proven bound: random x take 2.97 per bit, but structured ones can take
// more (7.9 per bit for x = 3 (2^p - 1) / q + 1 with q a small factor), so their limit is only a guard against a hang.
void Gcd::run(Run& r, const Words& x, i64 sub, u32 p, bool jacobi) {
  if (p < 3) { throw std::invalid_argument("Gcd: p must be at least 3"); }
  Timer timer;
  stats = {};

  u32 const nWords = std::min(u32(x.size()), (p - 1) / 32 + 1);
  Buffer<u32> bufX{nullptr, &queue, std::max(nWords, 1u)};
  if (nWords) { bufX.write(Words(x.begin(), x.begin() + nWords)); }

  u32 const nDigits = (p + 61) / 62 + 1;
  r.limit = nDigits + 1;
  for (int i = 0; i < 2; ++i) { r.f[i].zero(); r.g[i].zero(); }
  r.skewCount.zero();
  kToDigits.setKernelsToExecute(groups(nDigits));
  kToDigits(bufX, nWords, p, 1u, i64(0), 0u, r.F(), nDigits);
  kToDigits(bufX, nWords, p, 0u, sub, 0u, r.G(), nDigits);

  Kernel* kApplyX = jacobi ? kApplyJ : kApply;
  Kernel& kSummaryX = jacobi ? kSummaryJ : kSummary;
  auto const [delta, jac] = halfGcdDigits ? halfGcd(r, jacobi) : std::pair<i64, i64>{1, 0};
  (jacobi ? kInitJ : kInit)(r.F(), r.G(), r.mat[r.cur], delta, jac);

  u64 const maxSteps = (u64(p) * (jacobi ? 64 : 3) + 4000) / 62;
  u64 steps = 0;
  for (;;) {
    r.lens.zero();
    kLengths.setKernelsToExecute(groups(r.limit + 1));
    kLengths(r.F(), r.G(), r.limit + 1, r.lens);
    kSummaryX(r.F(), r.lens, r.mat[r.cur], r.out);
    vector<i64> s = r.out.read(6);
    u32 const lenF = u32(s[0]), lenG = u32(s[1]);
    if (lenF > r.limit || lenG > r.limit) {
      log("GCD: digit %u is not zero\n", std::max(lenF, lenG) - 1);
      throw std::runtime_error("GCD: digits past the limit");
    }

    bool done = false;
    if (!jacobi) {
      done = lenG == 0;
    } else if (lenG == 0) {
      done = true;      // x = 0: (0 | N) = 0
    } else if (s[4]) {  // f = g in the low 64 bits: compare the whole numbers
      r.flag.zero();
      kNotEqual.setKernelsToExecute(groups(r.limit + 1));
      kNotEqual(r.F(), r.G(), r.limit + 1, r.flag);
      done = r.flag.read(1)[0] == 0;
    }
    if (done) { break; }
    if (steps >= maxSteps) {
      log("GCD: no end after %" PRIu64 " divsteps (p %u)\n", steps * 62, p);
      throw std::runtime_error("GCD: no end");
    }

    r.limit = std::max(lenF, lenG) + 1;
    u32 const batch = std::clamp(r.limit, 32u, 1024u);
    for (int k = 0; k < 2; ++k) {
      int const from = r.cur ^ k, to = from ^ 1;
      kApplyX[k].setKernelsToExecute(groups(r.limit + 2));
      kApplyX[k].setFixedArgs(1, r.f[from], r.g[from], r.f[to], r.g[to], r.mat[from], r.mat[to], r.skewCount);
    }
    for (u32 i = 0; i < batch; ++i) {
      kApplyX[i & 1](r.limit);
      r.cur ^= 1;
    }
    steps += batch;
  }

  stats.divsteps += steps * 62;
  stats.skewEvents = skew ? r.skewCount.read(1)[0] : 0;
  stats.seconds = timer.at();
}

// The words of |f|, f != 0
Words Gcd::toWords(Run& r, u32 nOut) {
  u32 const n = r.limit + 1;
  u32 const nBlocks = (n + GW - 1) / GW;
  vector<i64> s = r.out.read(4);
  u32 const neg = s[3] < 0;
  Buffer<i32> signs{nullptr, &queue, nBlocks}, borrows{nullptr, &queue, nBlocks};
  Buffer<u32> out{nullptr, &queue, nOut};
  kBlockSigns.setKernelsToExecute(groups(nBlocks));
  kBlockSigns(r.F(), n, neg, signs, nBlocks);
  kBlockBorrows(signs, nBlocks, borrows);
  kToWords.setKernelsToExecute(groups(nOut));
  kToWords(r.F(), n, neg, borrows, out, nOut);
  Words w = out.read();
  while (w.size() > 1 && w.back() == 0) { w.pop_back(); }
  return w;
}

Words Gcd::gcdMinus1(const Words& x, u32 p) { return gcdOf(x, 1, p); }

Words Gcd::gcd(const Words& x, u32 p) { return gcdOf(x, 0, p); }

Words Gcd::gcdOf(const Words& x, i64 sub, u32 p) {
  Run r{&queue, roundUp((p + 61) / 62 + 4, GW)};
  run(r, x, sub, p, false);
  // A result of 1 is told by one digit: f = +-1
  vector<i64> s = r.out.read(4);
  if (s[0] == 1 && (s[2] == 1 || s[2] == -1)) { return {1}; }
  return toWords(r, (p - 1) / 32 + 1);
}

int Gcd::jacobi(const Words& x, u32 p) {
  Run r{&queue, roundUp((p + 61) / 62 + 4, GW)};
  run(r, x, 0, p, true);
  vector<i64> s = r.out.read(6);
  if (s[1] == 0) { return 0; }                    // x = 0
  if (s[0] == 1 && s[2] == 1) { return s[5] ? -1 : 1; }    // f = g = 1
  return 0;                                       // f = g = gcd(x, N) > 1
}

vector<i64> Gcd::leafMatrix(const vector<i64>& f, const vector<i64>& g, u32 nb, i64 delta, bool jacobi, double* seconds) {
  u32 const n = u32(std::min(f.size(), g.size()));
  if (nb < 1 || n < nb + 2 || n > LEAF_NMAX) { throw std::invalid_argument("leafMatrix: bad sizes"); }
  Buffer<i64> bf{nullptr, &queue, n}, bg{nullptr, &queue, n}, matIn{nullptr, &queue, 8}, matOut{nullptr, &queue, 8};
  Buffer<i64> out{nullptr, &queue, 4 * (nb + 2)};
  Buffer<u32> skewCount{nullptr, &queue, 1};
  bf.write(vector<i64>(f.begin(), f.begin() + n));
  bg.write(vector<i64>(g.begin(), g.begin() + n));
  matIn.write(vector<i64>{0, 0, 0, 0, delta, 0, 0, 0});
  Kernel& k = jacobi ? kLeafJ : kLeaf;
  k(bf, bg, n, nb, matIn, out, nb + 2, matOut, skewCount);
  queue.finish();
  Timer timer;
  k(bf, bg, n, nb, matIn, out, nb + 2, matOut, skewCount);
  queue.finish();
  if (seconds) { *seconds = timer.at(); }
  vector<i64> ret = out.read();
  vector<i64> m = matOut.read();
  ret.push_back(m[4]);
  ret.push_back(m[5]);
  return ret;
}

// ---- The subquadratic GCD (Bernstein-Yang jump divsteps): see spike-gcd/halfgcd-design.md ----

// Buffers are kept for reuse, by size (a power of two): the queue is in order, so a buffer freed by the host can be reused at once.
Gcd::Num Gcd::newNum(u32 len) {
  u32 size = 64;
  while (size < len) { size *= 2; }
  auto& free = pool[size];
  Buffer<i64>* b;
  if (free.empty()) {
    b = new Buffer<i64>(nullptr, &queue, size);
  } else {
    b = free.back().release();
    free.pop_back();
  }
  return {std::shared_ptr<Buffer<i64>>(b, [this, size](Buffer<i64>* p) { pool[size].emplace_back(p); }), 0, len};
}

// The magnitude of x in binary words (as many as its digits need), and its sign
Words Gcd::toBinary(const Num& x, bool& neg) {
  Num c = x.off ? addShift(x, x, 0, 0, x.len + 1) : x;
  Buffer<u32> lens{nullptr, &queue, 2};
  Buffer<i64> mat{nullptr, &queue, 8}, out{nullptr, &queue, 8};
  lens.zero();
  kLengths.setKernelsToExecute(groups(c.len));
  kLengths(*c.buf, *c.buf, c.len, lens);
  kSummary(*c.buf, lens, mat, out);
  vector<i64> s = out.read(4);
  u32 const len = u32(s[0]);
  neg = s[3] < 0;
  if (len == 0) { return {}; }
  u32 const nBlocks = (len + GW - 1) / GW, nOut = (62 * len + 31) / 32;
  Buffer<i32> signs{nullptr, &queue, nBlocks}, borrows{nullptr, &queue, nBlocks};
  Buffer<u32> words{nullptr, &queue, nOut};
  kBlockSigns.setKernelsToExecute(groups(nBlocks));
  kBlockSigns(*c.buf, len, u32(neg), signs, nBlocks);
  kBlockBorrows(signs, nBlocks, borrows);
  kToWords.setKernelsToExecute(groups(nOut));
  kToWords(*c.buf, len, u32(neg), borrows, words, nOut);
  return words.read();
}

// n digits of the number in w (negated if neg)
Gcd::Num Gcd::fromBinary(const Words& w, bool neg, u32 n) {
  Num out = newNum(n);
  u32 const nW = u32(w.size());
  Buffer<u32> bufW{nullptr, &queue, std::max(nW, 1u)};
  if (nW) { bufW.write(w); }
  kToDigits.setKernelsToExecute(groups(n));
  kToDigits(bufW, nW, 32 * nW, 0u, i64(0), u32(neg), *out.buf, n);
  return out;
}

// An exact product through the PRPLL transforms: a Gpu for an exponent E above the product's size (so mod 2^E - 1 does not
// wrap), on the NTT type M31 x M61 only (FFT3161: the FP types are not exact for all data), with at most the worst-case bits per
// word (N words of b bits sum to N 2^(2b - 2) at most, which must stay below 2^91) and long carries.  One Gpu per FFT size.
Gcd::Rung& Gcd::gpuFor(u64 bits) {
  for (const FFTShape& s : FFTShape::allShapes()) {
    if (s.fft_type != FFT3161 || s.isPfa()) { continue; }
    u32 const b = u32((90 - std::log2(double(s.size()))) / 2 + 1);    // 2 b - 2 + log2 N <= 90
    u64 const E = u64(b) * s.size() - 1;    // the Gpu gives wrong products for an E that is a multiple of N (seen on PoCL)
    if (E < bits + 64) { continue; }
    auto it = ladder.find(s.size());
    if (it == ladder.end()) {
      it = ladder.emplace(s.size(), Rung{E, Gpu::make(E, shared, FFTConfig{s, 101, CARRY_64}, {}, false)}).first;
    }
    return it->second;
  }
  throw std::runtime_error("Gcd: no FFT for a product");
}

Gcd::Num Gcd::mulLow(const Num& a, const Num& b, u32 n) {
  if (a.len >= fftDigits && b.len >= fftDigits) {
    bool negA = false, negB = false;
    Words wa = toBinary(a, negA), wb = toBinary(b, negB);
    if (wa.empty() || wb.empty()) { Num out = newNum(n); out.buf->zero(); return out; }
    Rung& rung = gpuFor(32 * u64(wa.size() + wb.size()));
    u32 const nw = nWords(rung.E);
    wa.resize(nw, 0);
    wb.resize(nw, 0);
    Words c = rung.gpu->expMul(wa, 1, wb, false);
    c.resize(std::min(u32(c.size()), (62 * n + 31) / 32 + 1));
    return fromBinary(c, negA != negB, n);
  }
  Num out = newNum(n);
  if (a.len == 0 || b.len == 0) { out.buf->zero(); return out; }
  u32 const nCols = std::min(n, a.len + b.len - 1);
  Buffer<i64> cols{nullptr, &queue, 3 * nCols};
  kMulColumns.setKernelsToExecute(groups(nCols));
  kMulColumns(*a.buf, a.off, a.len, *b.buf, b.off, b.len, cols, nCols);
  kMulDigits.setKernelsToExecute(groups(n));
  kMulDigits(cols, nCols, *out.buf, n);
  return out;
}

Gcd::Num Gcd::addShift(const Num& a, const Num& b, i64 sb, u32 drop, u32 n) {
  Num out = newNum(n);
  kAddDigits.setKernelsToExecute(groups(n));
  kAddDigits(*a.buf, a.off, a.len, i64(1), *b.buf, b.off, b.len, sb, drop, *out.buf, n);
  return out;
}

Gcd::Num Gcd::combine(const Num& x, const Num& a, const Num& y, const Num& b, u32 drop, u32 n) {
  return addShift(mulLow(x, a, n + drop + 1), mulLow(y, b, n + drop + 1), 1, drop, n);
}

// The matrix T of 62 nb divsteps from the low digits of f and g (at least nb + 2 of each): entries of nb + 2 digits, with
// (f, g) after the divsteps = T (f, g) / 2^(62 nb).  delta (element 4 of a matrix buffer, on the device) is updated.
Gcd::Matrix Gcd::jump(u32 nb, Buffer<i64>& delta, const Num& f, const Num& g, bool jacobi) {
  if (nb <= leafBatches) {
    u32 const n = nb + 2;
    Num out = newNum(4 * n);
    // the leaf reads n digits from the start of its buffers: copy slices and short numbers (zero-padded)
    Num fc = f.off || f.len < n ? addShift(f, f, 0, 0, n) : f, gc = g.off || g.len < n ? addShift(g, g, 0, 0, n) : g;
    (jacobi ? kLeafJ : kLeaf)(*fc.buf, *gc.buf, n, nb, delta, *out.buf, n, delta, scratchCount);
    return {out.slice(0, n), out.slice(n, n), out.slice(2 * n, n), out.slice(3 * n, n)};
  }
  u32 const j = nb / 2;
  Matrix t1 = jump(j, delta, f.slice(0, j + 2), g.slice(0, j + 2), jacobi);
  // f, g after the first j batches, modulo 2^(62 (nb - j + 2))
  u32 const m = nb - j + 2;
  Num f1 = combine(t1.u, f, t1.v, g, j, m), g1 = combine(t1.q, f, t1.r, g, j, m);
  Matrix t2 = jump(nb - j, delta, f1, g1, jacobi);
  u32 const n = nb + 2;
  return {combine(t2.u, t1.u, t2.v, t1.q, 0, n), combine(t2.u, t1.v, t2.v, t1.r, 0, n),
          combine(t2.q, t1.u, t2.r, t1.q, 0, n), combine(t2.q, t1.v, t2.r, t1.r, 0, n)};
}

// Runs jumps on the whole (f, g) of r while they have more than halfGcdDigits digits; returns delta and jac
std::pair<i64, i64> Gcd::halfGcd(Run& r, bool jacobi) {
  Buffer<i64> delta{nullptr, &queue, 8};
  delta.write(vector<i64>{0, 0, 0, 0, 1, 0, 0, 0});
  u32 lastLen = ~0u;
  for (;;) {
    r.lens.zero();
    kLengths.setKernelsToExecute(groups(r.limit + 1));
    kLengths(r.F(), r.G(), r.limit + 1, r.lens);
    vector<u32> lens = r.lens.read();
    u32 const len = std::max(lens[0], lens[1]);
    if (len <= halfGcdDigits || lens[1] == 0 || len >= lastLen) { break; }     // small enough, g = 0, or no progress
    lastLen = len;
    r.limit = len + 1;
    Num f{std::shared_ptr<Buffer<i64>>(&r.F(), [](Buffer<i64>*) {}), 0, len + 1};
    Num g{std::shared_ptr<Buffer<i64>>(&r.G(), [](Buffer<i64>*) {}), 0, len + 1};
    u32 const nb = len;
    Matrix t = jump(nb, delta, f.slice(0, nb + 2), g.slice(0, nb + 2), jacobi);
    stats.divsteps += u64(62) * nb;
    Num f1 = combine(t.u, f, t.v, g, nb, r.limit + 1), g1 = combine(t.q, f, t.r, g, nb, r.limit + 1);
    int const next = r.cur ^ 1;
    r.f[next].zero();
    r.g[next].zero();
    queue.copyBuf(f1.buf->get(), r.f[next].get(), (r.limit + 1) * sizeof(i64), nullptr);
    queue.copyBuf(g1.buf->get(), r.g[next].get(), (r.limit + 1) * sizeof(i64), nullptr);
    r.cur = next;
  }
  vector<i64> d = delta.read();
  return {d[4], d[5]};
}

vector<i64> Gcd::mulTest(const vector<i64>& a, const vector<i64>& b) {
  Num na = newNum(u32(a.size())), nb = newNum(u32(b.size()));
  na.buf->write(a);
  nb.buf->write(b);
  u32 const n = u32(a.size() + b.size() + 1);
  return mulLow(na, nb, n).buf->read();
}

vector<i64> Gcd::jumpMatrix(const vector<i64>& f, const vector<i64>& g, u32 nb, i64 delta) {
  Num nf = newNum(u32(f.size())), ng = newNum(u32(g.size()));
  nf.buf->write(f);
  ng.buf->write(g);
  Buffer<i64> deltaBuf{nullptr, &queue, 8};
  deltaBuf.write(vector<i64>{0, 0, 0, 0, delta, 0, 0, 0});
  Matrix t = jump(nb, deltaBuf, nf, ng, false);
  delta = deltaBuf.read()[4];
  vector<i64> ret;
  for (const Num* x : {&t.u, &t.v, &t.q, &t.r}) {
    vector<i64> d = x->buf->read();
    ret.insert(ret.end(), d.begin() + x->off, d.begin() + x->off + nb + 2);
  }
  ret.push_back(delta);
  ret.push_back(0);
  return ret;
}
