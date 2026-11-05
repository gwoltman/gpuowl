// Copyright (C) Mihai Preda and George Woltman

// GCD and Jacobi symbol of huge numbers with Bernstein-Yang divsteps ("safegcd"), driven by src/Gcd.cpp.  These kernels are a
// program of their own (src/cl/gcd/ is not part of the main kernel bundle), so they do not touch the PRP kernels' compile cache.
//
// A number is a vector of signed 62-bit digits in i64, little-endian: value = sum d[j] 2^(62 j).  The digits are redundant:
// |d| <= DMAX, where DMAX = 2^61 + 4 (2^61 + 2^SKEW + 4 with -DSKEW, see carry2), and not necessarily balanced.  As DMAX < 2^62 - 1:
//  * zero has only the all-zero representation, and the sign of a number is that of its top nonzero digit (the digits below
//    digit k add up to less than 2^(62 k) in absolute value), so one digit tells whether a number is +-1;
//  * the low 64 bits are d[0] + d[1] 2^62 mod 2^64 in any representation.
//
// A step replaces (f, g) by T (f, g) / 2^124, where T is the matrix of 124 divsteps:  the product of the matrices of two batches of 62,
// each derived from the low 64 bits of f and g at its start (derive2, in the group that makes the low digits).  One launch per step:
// half the launches and half the memory passes per divstep of steps of 62, with the same multiplications per divstep.  How the digits
// of T (f, g) are formed and carried is at apply; no carry ever ripples:  every step is exact and there is nothing to detect or redo.
// (With -DSKEW=k, the carries round the other way whenever a digit's sum is within 2^k of a rounding boundary, to make non-canonical
// digits common; k <= 59 keeps DMAX < 2^62 - 1.  The tests use it to show that every consumer of the digits copes with them.)
//
// The divsteps (Bernstein & Yang, "Fast constant-time gcd computation and modular inversion", 2019):
//   GCD (f odd):    g odd and delta > 0: (delta, f, g) -> (1 - delta, g, (g - f) / 2); g odd: (1 + delta, f, (g + f) / 2);
//                   g even: (1 + delta, f, g / 2).  g reaches 0, and then |f| = gcd(f, g).
//   JACOBI (f odd, f > 0, g >= 0): the "posdivsteps" of M. Hamburg, "Computing the Jacobi symbol using Bernstein-Yang" (IACR ePrint
//                   2021/1271): g odd and delta > 0: swap f and g, delta -> -delta; then g odd: g += f; then g /= 2, delta += 1.
//                   f and g stay positive, so quadratic reciprocity needs only f, g mod 8: J = (-1)^jac (g | f) is invariant with
//                   jac ^= (f, g both 3 mod 4) at a swap and jac ^= (f = 3 or 5 mod 8) at a halving.  f and g meet at gcd(f, g)
//                   and stay there; the symbol is (-1)^jac when that is 1, and 0 otherwise.  It takes about 2.97 divsteps per bit
//                   against 2.08 for the GCD steps.
//
// The matrix buffers hold {U0, V0, Q0, R0, delta, jac, f == g in the low 64 bits (JACOBI), 0, U1, V1, Q1, R1, 0, 0, 0, 0} (see init).

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;

#define GW 256
#define MASK62 ((((u64) 1) << 62) - 1)
#define HALF62 (((i64) 1) << 61)

// floor((s + 2^61) / 2^62), without overflow for any s
i64 balancedCarry(i64 s) { return (s >> 62) + (i64) (((u64) (s & (i64) MASK62) + (u64) HALF62) >> 62); }

// The carry out of a digit whose sum is s; count != 0 counts the -DSKEW events in *stats
i64 carryOf(i64 s, global u32* stats, bool count) {
  i64 t = balancedCarry(s);
#if SKEW
  // s + 2^61 - t 2^62, in [0, 2^62)
  i64 pos = (i64) (((u64) (s & (i64) MASK62) + (u64) HALF62) & MASK62);
  if (pos < (((i64) 1) << SKEW)) {
    t -= 1;
    if (count) { atomic_add(stats, 1u); }
  } else if (pos >= (((i64) 1) << 62) - (((i64) 1) << SKEW)) {
    t += 1;
    if (count) { atomic_add(stats, 1u); }
  }
#else
  (void) stats; (void) count;
#endif
  return t;
}

// d = s - t 2^62 + tPrev, computed modulo 2^64 (t 2^62 alone may not fit, the result does)
i64 digitOf(i64 s, i64 t, i64 tPrev) { return (i64) ((u64) s - ((u64) t << 62) + (u64) tPrev); }

typedef struct { i64 lo, hi; } Split;

typedef struct { u64 lo, hi; } U128;
// a * b, 64 x 64 -> 128 unsigned, from four 32 x 32 partial products (math.cl's mul64:  mul_hi(a, b) and a * b would form the low
// cross products twice)
U128 mulWide(u64 a, u64 b) {
  u32 al = (u32) a, ah = (u32) (a >> 32), bl = (u32) b, bh = (u32) (b >> 32);
  u64 ll = (u64) al * bl;
  u64 lh = (u64) al * bh + (ll >> 32);
  u64 hl = (u64) ah * bl + (u32) lh;
  u64 hh = (u64) ah * bh + (lh >> 32) + (hl >> 32);
  U128 ret = {(hl << 32) | (u32) ll, hh};
  return ret;
}

// u f + v g as low 62 bits (lo) and the rest (hi, signed), for |u| + |v| <= 2^62 + 1 and |f|, |g| <= DMAX
Split combine(i64 u, i64 f, i64 v, i64 g) {
  U128 a = mulWide((u64) u, (u64) f), b = mulWide((u64) v, (u64) g);
  u64 aHi = a.hi - (u < 0 ? (u64) f : 0) - (f < 0 ? (u64) u : 0);
  u64 bHi = b.hi - (v < 0 ? (u64) g : 0) - (g < 0 ? (u64) v : 0);
  u64 xLo = a.lo + b.lo;
  u64 xHi = aHi + bHi + (xLo < a.lo ? 1 : 0);
  Split ret = {(i64) (xLo & MASK62), (i64) ((xHi << 2) | (xLo >> 62))};
  return ret;
}

// A step's matrix, delta and jac after its divsteps, and whether f == g in the low 64 bits before them (JACOBI)
typedef struct { i64 u, v, q, r, delta, jac, eq; } Mat;

// The number of trailing zero bits of x, x != 0
#ifdef __CUDA_ARCH__
u32 ctz64(u64 x) { return (u32) (__ffsll((long long) x) - 1); }
#else
u32 ctz64(u64 x) { return (u32) ctz(x); }
#endif

// The matrix of the next 62 divsteps from the low 64 bits of f and g.  The same divsteps as one at a time, with each run of steps on an
// even g (halve g, double u and v, delta + 1) done by one shift:  the same operations in the same order, so the same matrix.
Mat derive(u64 f, u64 g, i64 delta, i64 jac) {
  i64 u = 1, v = 0, q = 0, r = 1;
  i64 eq = 0;
  u32 left = 62;
#if JACOBI
  eq = (f == g);
  for (;;) {
    u32 z = ctz64(g | (((u64) 1) << left));
    g >>= z; u = (i64) ((u64) u << z); v = (i64) ((u64) v << z);
    delta += z; left -= z;
    jac ^= (i64) ((f >> 1) ^ (f >> 2)) & (i64) z & 1;
    if (left == 0) { break; }
    // g odd:  a swap if delta > 0, then g += f, then the halving
    if (delta > 0) {
      jac ^= (i64) ((f & g) >> 1) & 1;
      u64 t = f; f = g; g = t;
      i64 tu = u, tv = v; u = q; v = r; q = tu; r = tv;
      delta = -delta;
    }
    g += f; q += u; r += v;
    jac ^= (i64) ((f >> 1) ^ (f >> 2)) & 1;
    g >>= 1; u *= 2; v *= 2;
    delta += 1;
    if (--left == 0) { break; }
  }
#else
  for (;;) {
    u32 z = ctz64(g | (((u64) 1) << left));
    g >>= z; u = (i64) ((u64) u << z); v = (i64) ((u64) v << z);
    delta += z; left -= z;
    if (left == 0) { break; }
    // g odd
    if (delta > 0) {
      u64 nf = g, ng = (g - f) >> 1;
      i64 nu = 2 * q, nv = 2 * r, nq = q - u, nr = r - v;
      f = nf; g = ng; u = nu; v = nv; q = nq; r = nr;
      delta = 1 - delta;
    } else {
      g = (g + f) >> 1;
      q += u; r += v; u *= 2; v *= 2;
      delta = 1 + delta;
    }
    if (--left == 0) { break; }
  }
#endif
  Mat m = {u, v, q, r, delta, jac, eq};
  return m;
}

u64 low64(global const i64* x) { return (u64) x[0] + ((u64) x[1] << 62); }

// Bits [62 j, 62 j + 62) of the p-bit number in words (all ones when ones != 0); words past nWords are zero.
u64 chunk(global const u32* words, u32 nWords, u32 p, u32 ones, u32 j) {
  u32 lo = 62 * j;
  if (lo >= p) { return 0; }
  u32 nBits = min(p - lo, 62u);
  u64 mask = (((u64) 1) << nBits) - 1;
  if (ones) { return mask; }
  u32 w = lo / 32, sh = lo % 32;
  u64 w0 = w < nWords ? words[w] : 0, w1 = w + 1 < nWords ? words[w + 1] : 0, w2 = w + 2 < nWords ? words[w + 2] : 0;
  u64 x = ((w0 | (w1 << 32)) >> sh) | (sh ? (w2 << (64 - sh)) : 0);
  return x & mask;
}

// out = the p-bit number in words (or 2^p - 1 when ones != 0) minus sub, negated when neg != 0, as n digits, each in [-2^61, 2^61]
// (but digit 0, which is in [-2^61 - 1, 2^61] with sub = 1): chunk j less 2^62 when its top bit is set, plus that bit of chunk
// j - 1.  n >= p / 62 + 2.
KERNEL(GW) toDigits(global const u32* words, u32 nWords, u32 p, u32 ones, i64 sub, u32 neg, global i64* out, u32 n) {
  u32 j = get_global_id(0);
  if (j >= n) { return; }
  u64 c = chunk(words, nWords, p, ones, j);
  u64 cPrev = j ? chunk(words, nWords, p, ones, j - 1) : 0;
  i64 d = (i64) (c & (MASK62 >> 1)) - (i64) (c & (((u64) 1) << 61)) + (i64) (cPrev >> 61) - (j == 0 ? sub : 0);
  out[j] = neg ? -d : d;
}

// ---- A step:  124 divsteps, the matrices of two 62-divstep batches composed ----
// The matrix buffer of a step holds {U0, V0, Q0, R0, delta, jac, eq, 0, U1, V1, Q1, R1, 0, 0, 0, 0}:  T = [[U, V], [Q, R]] with entries
// X = X0 + X1 2^62, X0 in [-2^61, 2^61).  After the step (f, g) = T (f, g) / 2^124.  |U| + |V| <= 2^124 (a product of two matrices
// whose rows sum to at most 2^62), so |U0| + |V0| <= 2^62 and |U1| + |V1| <= 2^62 + 1.
// Digit j of U f + V g is the sum of X_j = U0 f_j + V0 g_j (at j) and Y_(j-1) = U1 f_(j-1) + V1 g_(j-1) (at j);  with the splits
// X = hiX 2^62 + loX the sum at j is A_j + B_j, A_j = loX_j + hiX_(j-1), B_j = loY_(j-1) + hiY_(j-2), each well inside an i64 but
// not their sum.  The carry out of j is t_j = floor(A_j / 2^62) + floor(B_j / 2^62) + c_j, c_j = the balanced carry of the sum of the
// low 62 bits of A_j and B_j, and the new digit is d_j = (that sum - c_j 2^62) + t_(j-1).  sum d_j 2^(62 j) telescopes to
// sum (A_j + B_j) 2^(62 j) whatever the carries are, as long as threads j and j + 1 compute t_j alike; the carries only keep the
// digits small:  |hi| <= DMAX + 1, so floor(A / 2^62) and floor(B / 2^62) are in [-1, 1] and c in [0, 2], and |d_j| <= 2^61 + 4.

typedef struct { u64 lo; i64 hi; } I128;

// a * b, signed 64 x 64 -> 128
I128 smul(i64 a, i64 b) {
  U128 p = mulWide((u64) a, (u64) b);
  I128 r = {p.lo, (i64) (p.hi - (a < 0 ? (u64) b : 0) - (b < 0 ? (u64) a : 0))};
  return r;
}

I128 add128(I128 a, I128 b) { I128 r = {a.lo + b.lo, a.hi + b.hi + (a.lo + b.lo < a.lo ? 1 : 0)}; return r; }

// The low 128 bits of x * F, x signed, F modulo 2^128
U128 mulLow128(i64 x, U128 F) {
  U128 p = mulWide((u64) x, F.lo);
  U128 r = {p.lo, p.hi + (u64) x * F.hi - (x < 0 ? F.lo : 0)};
  return r;
}

U128 addU128(U128 a, U128 b) { U128 r = {a.lo + b.lo, a.hi + b.hi + (a.lo + b.lo < a.lo ? 1 : 0)}; return r; }

// The low 128 bits of the number with digits d[0..2] (and d[3], which is above bit 128)
U128 low128(i64 d0, i64 d1, i64 d2) {
  U128 a = {(u64) d0, (u64) (d0 >> 63)};
  U128 b = {(u64) d1 << 62, (u64) (d1 >> 2)};
  U128 c = {0, (u64) d2 << 60};
  return addU128(addU128(a, b), c);
}

typedef struct { i64 x0, x1; } Pair;

// X = x0 + x1 2^62 with x0 in [-2^61, 2^61)
Pair splitEntry(I128 x) {
  i64 x0 = (i64) (x.lo << 2) >> 2;
  I128 negX0 = {(u64) -x0, x0 > 0 ? -1 : 0};
  I128 d = add128(x, negX0);
  Pair p = {x0, (i64) ((d.lo >> 62) | ((u64) d.hi << 2))};
  return p;
}

typedef struct { i64 m[12]; i64 delta, jac, eq; } Mat2;

// The matrix of the next 124 divsteps from the low digits of f and g
Mat2 derive2(i64 f0, i64 f1, i64 f2, i64 g0, i64 g1, i64 g2, i64 delta, i64 jac) {
  U128 F = low128(f0, f1, f2), G = low128(g0, g1, g2);
  Mat a = derive(F.lo, G.lo, delta, jac);
  // f and g after the first 62 divsteps, low 64 bits:  bits [62, 126) of u F + v G and q F + r G
  U128 sf = addU128(mulLow128(a.u, F), mulLow128(a.v, G)), sg = addU128(mulLow128(a.q, F), mulLow128(a.r, G));
  Mat b = derive((sf.lo >> 62) | (sf.hi << 2), (sg.lo >> 62) | (sg.hi << 2), a.delta, a.jac);
  Pair U = splitEntry(add128(smul(b.u, a.u), smul(b.v, a.q))), V = splitEntry(add128(smul(b.u, a.v), smul(b.v, a.r)));
  Pair Q = splitEntry(add128(smul(b.q, a.u), smul(b.r, a.q))), R = splitEntry(add128(smul(b.q, a.v), smul(b.r, a.r)));
  Mat2 m = {{U.x0, V.x0, Q.x0, R.x0, 0, 0, 0, 0, U.x1, V.x1, Q.x1, R.x1}, b.delta, b.jac, a.eq};
  return m;
}

void storeMat2(global i64* p, Mat2 m) {
  for (int i = 0; i < 4; ++i) { p[i] = m.m[i]; p[8 + i] = m.m[8 + i]; }
  p[4] = m.delta; p[5] = m.jac; p[6] = m.eq; p[7] = 0;
}

// The balanced carry c out of r (the sum of two 62-bit parts, below 2^63) and r - c 2^62 in *low, with -DSKEW's rounding the other
// way near a boundary
i64 carry2(u64 r, i64* low, global u32* stats, bool count) {
  i64 c = (i64) ((r + (u64) HALF62) >> 62);
#if SKEW
  i64 pos = (i64) ((r + (u64) HALF62) & MASK62);
  if (pos < (((i64) 1) << SKEW)) {
    c -= 1;
    if (count) { atomic_add(stats, 1u); }
  } else if (pos >= (((i64) 1) << 62) - (((i64) 1) << SKEW)) {
    c += 1;
    if (count) { atomic_add(stats, 1u); }
  }
#else
  (void) stats; (void) count;
#endif
  *low = (i64) (r - ((u64) c << 62));
  return c;
}

// The carry out of position j (A_j, B_j as above), and the digit's part below the carry in *low
i64 carryAB(i64 a, i64 b, i64* low, global u32* stats, bool count) {
  u64 r = (u64) (a & (i64) MASK62) + (u64) (b & (i64) MASK62);
  return (a >> 62) + (b >> 62) + carry2(r, low, stats, count);
}

i64 digit2(i64 a, i64 b, i64 aPrev, i64 bPrev, global u32* stats) {
  i64 low, lowPrev;
  carryAB(a, b, &low, stats, true);
  i64 tPrev = carryAB(aPrev, bPrev, &lowPrev, stats, false);
  return (i64) ((u64) low + (u64) tPrev);
}

// The first matrix
KERNEL(64) init(global const i64* f, global const i64* g, global i64* mat, i64 delta, i64 jac) {
  if (get_global_id(0) == 0) { storeMat2(mat, derive2(f[0], f[1], f[2], g[0], g[1], g[2], delta, jac)); }
}

// One step of 124 divsteps on digits [0, limit).  Groups overlap by three digits:  group b makes the sums of digits
// [b (GW - 3) - 3, b (GW - 3) + GW - 3), and its threads me >= 3 store the new digit j - 2 (j in [2, limit + 3)).  Group 0 then derives
// the next matrix (all its threads, on the same values:  on AMD the derive runs on the scalar unit).
KERNEL(GW) apply(u32 limit, global const i64* fIn, global const i64* gIn, global i64* fOut, global i64* gOut,
                 global const i64* matIn, global i64* matOut, global u32* stats) {
  local i64 lx[8][GW];
  local i64 low[8];

  i64 U0 = matIn[0], V0 = matIn[1], Q0 = matIn[2], R0 = matIn[3], U1 = matIn[8], V1 = matIn[9], Q1 = matIn[10], R1 = matIn[11];
  u32 me = get_local_id(0);
  i32 base = (i32) (get_group_id(0) * (GW - 3)) - 3;
  i32 j = base + (i32) me;

  {
    bool in = j >= 0 && j < (i32) limit;
    i64 f = in ? fIn[j] : 0, g = in ? gIn[j] : 0;
    Split xf = combine(U0, f, V0, g), yf = combine(U1, f, V1, g), xg = combine(Q0, f, R0, g), yg = combine(Q1, f, R1, g);
    lx[0][me] = xf.lo; lx[1][me] = xf.hi; lx[2][me] = yf.lo; lx[3][me] = yf.hi;
    lx[4][me] = xg.lo; lx[5][me] = xg.hi; lx[6][me] = yg.lo; lx[7][me] = yg.hi;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (me >= 3) {
    i64 dF = digit2(lx[0][me] + lx[1][me - 1], lx[2][me - 1] + lx[3][me - 2], lx[0][me - 1] + lx[1][me - 2], lx[2][me - 2] + lx[3][me - 3], stats);
    i64 dG = digit2(lx[4][me] + lx[5][me - 1], lx[6][me - 1] + lx[7][me - 2], lx[4][me - 1] + lx[5][me - 2], lx[6][me - 2] + lx[7][me - 3], stats);
    if (j >= 2 && j < (i32) limit + 3) {
      fOut[j - 2] = dF;
      gOut[j - 2] = dG;
    }
    if (base == -3 && me >= 5 && me <= 7) {
      low[me - 5] = dF;
      low[me - 1] = dG;
    }
  }
  if (base == -3) {
    barrier(CLK_LOCAL_MEM_FENCE);
    i64 v[6], d = matIn[4], jc = matIn[5];
    for (int i = 0; i < 3; ++i) { v[i] = low[i]; v[3 + i] = low[4 + i]; }
#if defined(__AMDGCN__)
    // The values are the same in every lane:  tell the compiler, so that the derive runs on the scalar unit
    for (int i = 0; i < 6; ++i) {
      v[i] = (i64) (((u64) (u32) __builtin_amdgcn_readfirstlane((i32) ((u64) v[i] >> 32)) << 32) | (u32) __builtin_amdgcn_readfirstlane((i32) v[i]));
    }
    d = (i64) (((u64) (u32) __builtin_amdgcn_readfirstlane((i32) ((u64) d >> 32)) << 32) | (u32) __builtin_amdgcn_readfirstlane((i32) d));
    jc = (i64) (u32) __builtin_amdgcn_readfirstlane((i32) jc);
#endif
    Mat2 m = derive2(v[0], v[1], v[2], v[3], v[4], v[5], d, jc);
    if (me == 0) { storeMat2(matOut, m); }
  }
}

// ---- The leaf of the subquadratic GCD: a whole transition matrix in one work-group ----

#ifndef LEAF_NMAX
#define LEAF_NMAX 280
#endif
#define LEAF_W 256

// Digit j of x a + y b, with a and b taken as zero from len up (the digit recomputes the two products below it rather than share
// them, which saves a barrier)
i64 combDigit(local const i64* a, local const i64* b, i64 x, i64 y, i32 j, i32 len, global u32* stats) {
  Split x0 = {0, 0}, x1 = {0, 0}, x2 = {0, 0};
  if (j < len) { x0 = combine(x, a[j], y, b[j]); }
  if (j >= 1 && j - 1 < len) { x1 = combine(x, a[j - 1], y, b[j - 1]); }
  if (j >= 2 && j - 2 < len) { x2 = combine(x, a[j - 2], y, b[j - 2]); }
  i64 s = x0.lo + x1.hi, sPrev = x1.lo + x2.hi;
  return digitOf(s, carryOf(s, stats, false), carryOf(sPrev, stats, false));
}

// (The body is not on the line of the declaration:  the CUDA shim takes a line with "local " and a '[' for a local array)
void storeLocalMat(local i64* p, Mat m) {
  p[0] = m.u; p[1] = m.v; p[2] = m.q; p[3] = m.r; p[4] = m.delta; p[5] = m.jac;
}

// The matrix T of 62 nb divsteps from the low n digits of f and g (n >= nb + 2, n <= LEAF_NMAX), with delta and jac from matIn.
// T goes to out as U, V, Q, R (nb + 2 digits each, stride apart): after the divsteps (f, g) = (U f + V g, Q f + R g) / 2^(62 nb).
// delta and jac after them go to matOut.  Per batch of 62 divsteps, thread 0 makes the low two digits of the next f and g itself
// and derives the next matrix while the other threads apply the current one to the rest, so a batch costs one barrier.
KERNEL(LEAF_W) leaf(global const i64* f, global const i64* g, u32 n, u32 nb, global const i64* matIn, global i64* out, u32 stride,
                    global i64* matOut, global u32* stats) {
  local i64 A[2][6][LEAF_NMAX];       // F, G, U, Q, V, R
  local i64 M[2][8];
  i32 me = (i32) get_local_id(0);

  for (i32 j = me; j < LEAF_NMAX; j += LEAF_W) {
    A[0][0][j] = j < (i32) n ? f[j] : 0;
    A[0][1][j] = j < (i32) n ? g[j] : 0;
    A[0][2][j] = j == 0;
    A[0][3][j] = 0;
    A[0][4][j] = 0;
    A[0][5][j] = j == 0;
  }
  if (me == 0) { storeLocalMat(M[0], derive(low64(f), low64(g), matIn[4], matIn[5])); }
  barrier(CLK_LOCAL_MEM_FENCE);

  u32 par = 0;
  for (i32 t = 0; t < (i32) nb; ++t) {
    i64 u = M[par][0], v = M[par][1], q = M[par][2], r = M[par][3];
    i32 lenFG = (i32) n - t;      // f and g are known modulo 2^(62 (n - t)) here
    i32 lenM = t + 2;             // the entries of T are below 2^(62 t)
    if (me == 0) {
      i64 f0 = combDigit(A[par][0], A[par][1], u, v, 1, lenFG, stats), f1 = combDigit(A[par][0], A[par][1], u, v, 2, lenFG, stats);
      i64 g0 = combDigit(A[par][0], A[par][1], q, r, 1, lenFG, stats), g1 = combDigit(A[par][0], A[par][1], q, r, 2, lenFG, stats);
      storeLocalMat(M[par ^ 1], derive((u64) f0 + ((u64) f1 << 62), (u64) g0 + ((u64) g1 << 62), M[par][4], M[par][5]));
    } else {
      for (i32 i = me - 1; i < lenFG - 1; i += LEAF_W - 1) {
        A[par ^ 1][0][i] = combDigit(A[par][0], A[par][1], u, v, i + 1, lenFG, stats);
        A[par ^ 1][1][i] = combDigit(A[par][0], A[par][1], q, r, i + 1, lenFG, stats);
      }
      for (i32 i = me - 1; i < lenM + 1; i += LEAF_W - 1) {
        A[par ^ 1][2][i] = combDigit(A[par][2], A[par][3], u, v, i, lenM, stats);
        A[par ^ 1][3][i] = combDigit(A[par][2], A[par][3], q, r, i, lenM, stats);
        A[par ^ 1][4][i] = combDigit(A[par][4], A[par][5], u, v, i, lenM, stats);
        A[par ^ 1][5][i] = combDigit(A[par][4], A[par][5], q, r, i, lenM, stats);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    par ^= 1;
  }

  for (i32 i = me; i < (i32) nb + 2; i += LEAF_W) {
    out[i] = A[par][2][i];
    out[stride + i] = A[par][4][i];
    out[2 * stride + i] = A[par][3][i];
    out[3 * stride + i] = A[par][5][i];
  }
  // M[par] is the matrix after the last batch: delta and jac after the nb batches are in the one before it
  if (me == 0) { matOut[4] = M[par ^ 1][4]; matOut[5] = M[par ^ 1][5]; }
}

// ---- Exact small products and sums of digit vectors (the subquadratic GCD's products below the FFT sizes) ----

// Column k of a * b (a: aLen digits from aOff, b: bLen digits from bOff), a 192-bit signed sum, split into balanced 62-bit pieces
// c0 + c1 2^62 + c2 2^124 (c0, c1 in [-2^61, 2^61)) and stored at cols[3 k ..].  A column has at most min(aLen, bLen) < 2^14
// terms of magnitude below 2^124, so it is below 2^138 and c2 is small.
KERNEL(GW) mulColumns(global const i64* a, u32 aOff, u32 aLen, global const i64* b, u32 bOff, u32 bLen, global i64* cols, u32 nCols) {
  u32 k = get_global_id(0);
  if (k >= nCols) { return; }
  u64 w0 = 0, w1 = 0, w2 = 0;     // the sum, 192-bit two's complement
  u32 iLo = k + 1 > bLen ? k + 1 - bLen : 0, iHi = min(k + 1, aLen);
  for (u32 i = iLo; i < iHi; ++i) {
    I128 xy = smul(a[aOff + i], b[bOff + k - i]);
    u64 lo = xy.lo, hi = (u64) xy.hi;
    u64 ext = (i64) hi < 0 ? ~(u64) 0 : 0;
    u64 n0 = w0 + lo;
    u64 c = n0 < lo;
    u64 n1 = w1 + hi + c;
    c = (n1 < w1) || (c && n1 == w1);
    w0 = n0; w1 = n1; w2 += ext + c;
  }
  // balanced split: c0 = w mod 2^62 in [-2^61, 2^61), then the same for (w - c0) / 2^62, and the rest
  i64 c0 = (i64) (w0 << 2) >> 2;
  // (w - c0) >> 62: subtract c0 (sign-extended) from the 192-bit w, then shift
  u64 e = c0 < 0 ? ~(u64) 0 : 0;
  u64 m0 = w0 - (u64) c0, b0 = w0 < (u64) c0;
  u64 m1 = w1 - e - b0, b1 = (w1 < e) || (w1 - e < b0);
  u64 m2 = w2 - e - b1;
  u64 q0 = (m0 >> 62) | (m1 << 2), q1 = (m1 >> 62) | (m2 << 2);   // (w - c0) / 2^62, 128-bit
  i64 c1 = (i64) (q0 << 2) >> 2;
  u64 f = c1 < 0 ? ~(u64) 0 : 0;
  u64 r0 = q0 - (u64) c1, bb = q0 < (u64) c1;
  u64 r1 = q1 - f - bb;
  i64 c2 = (i64) ((r0 >> 62) | (r1 << 2));
  cols[3 * k] = c0;
  cols[3 * k + 1] = c1;
  cols[3 * k + 2] = c2;
}

// out = the product's digits from the column pieces: digit k = c0_k + c1_(k-1) + c2_(k-2), rebalanced once
KERNEL(GW) mulDigits(global const i64* cols, u32 nCols, global i64* out, u32 n) {
  u32 k = get_global_id(0);
  if (k >= n) { return; }
  i64 s = (k < nCols ? cols[3 * k] : 0) + (k >= 1 && k - 1 < nCols ? cols[3 * (k - 1) + 1] : 0) + (k >= 2 && k - 2 < nCols ? cols[3 * (k - 2) + 2] : 0);
  i64 sPrev = k >= 1 ? (k - 1 < nCols ? cols[3 * (k - 1)] : 0) + (k >= 2 && k - 2 < nCols ? cols[3 * (k - 2) + 1] : 0)
                       + (k >= 3 && k - 3 < nCols ? cols[3 * (k - 3) + 2] : 0) : 0;
  out[k] = digitOf(s, balancedCarry(s), balancedCarry(sPrev));
}

// out[0, n) = digits of (sa a + sb b) / 2^(62 drop) (sa, sb = +-1; a, b taken as zero past their lengths), rebalanced once.  The
// caller knows the low drop digits of the sum are zero.
i64 sumAt(global const i64* a, u32 aOff, u32 aLen, i64 sa, global const i64* b, u32 bOff, u32 bLen, i64 sb, u32 k) {
  return (k < aLen ? sa * a[aOff + k] : 0) + (k < bLen ? sb * b[bOff + k] : 0);
}

KERNEL(GW) addDigits(global const i64* a, u32 aOff, u32 aLen, i64 sa, global const i64* b, u32 bOff, u32 bLen, i64 sb,
                     u32 drop, global i64* out, u32 n) {
  u32 i = get_global_id(0);
  if (i >= n) { return; }
  u32 k = i + drop;
  i64 s = sumAt(a, aOff, aLen, sa, b, bOff, bLen, sb, k);
  i64 sPrev = k ? sumAt(a, aOff, aLen, sa, b, bOff, bLen, sb, k - 1) : 0;
  out[i] = digitOf(s, balancedCarry(s), balancedCarry(sPrev));
}

// res[0], res[1] (zero on entry) = the number of digits of f, g up to the top nonzero one
KERNEL(GW) lengths(global const i64* f, global const i64* g, u32 n, global u32* res) {
  u32 j = get_global_id(0);
  if (j < n) {
    if (f[j]) { atomic_max(&res[0], j + 1); }
    if (g[j]) { atomic_max(&res[1], j + 1); }
  }
}

// out = {length of f, length of g, f digit 0, f top digit, f == g in the low 64 bits (JACOBI), jac}
KERNEL(64) summary(global const i64* f, global const u32* lens, global const i64* mat, global i64* out) {
  if (get_global_id(0) == 0) {
    u32 lenF = lens[0];
    out[0] = lenF;
    out[1] = lens[1];
    out[2] = f[0];
    out[3] = lenF ? f[lenF - 1] : 0;
    out[4] = mat[6];
    out[5] = mat[5];
  }
}

// res[0] (zero on entry) becomes nonzero unless f == g.  The digits of f - g, up to about 2^62 in size, are rebalanced once
// (a carry of at most 1 moves up one digit), which brings them under 2^61 + 2: then the difference is zero only if every digit is.
KERNEL(GW) notEqual(global const i64* f, global const i64* g, u32 n, global u32* res) {
  u32 j = get_global_id(0);
  if (j < n) {
    i64 e = f[j] - g[j];
    i64 ePrev = j ? f[j - 1] - g[j - 1] : 0;
    if (digitOf(e, balancedCarry(e), balancedCarry(ePrev))) { atomic_max(&res[0], 1u); }
  }
}

// Conversion of a nonzero number (negated when neg != 0, so that it is positive) to binary words.  With digits e_j, the borrow into
// digit j is -1 when the nearest nonzero digit below j is negative and 0 otherwise (|e_j| < 2^62 - 1), and the binary digit j is
// (e_j + borrow) mod 2^62.  blockSigns finds the sign of the top nonzero digit of each block of GW digits, blockBorrows turns those
// into the borrow into each block, toWords makes each output word from at most two digits.

i64 digitAt(global const i64* x, u32 neg, u32 j) { return neg ? -x[j] : x[j]; }

KERNEL(GW) blockSigns(global const i64* x, u32 n, u32 neg, global i32* signs, u32 nBlocks) {
  u32 b = get_global_id(0);
  if (b >= nBlocks) { return; }
  i32 sign = 0;
  for (u32 j = min(n, (b + 1) * GW); j > b * GW; --j) {
    i64 e = digitAt(x, neg, j - 1);
    if (e) { sign = e < 0 ? -1 : 1; break; }
  }
  signs[b] = sign;
}

KERNEL(64) blockBorrows(global const i32* signs, u32 nBlocks, global i32* borrows) {
  if (get_global_id(0) == 0) {
    i32 borrow = 0;
    for (u32 b = 0; b < nBlocks; ++b) {
      borrows[b] = borrow;
      if (signs[b]) { borrow = signs[b] < 0 ? -1 : 0; }
    }
  }
}

// Binary digit j
u64 binaryDigit(global const i64* x, u32 n, u32 neg, global const i32* borrows, u32 j) {
  if (j >= n) { return 0; }
  i64 borrow = borrows[j / GW];
  for (u32 k = j; k > j / GW * GW; --k) {
    i64 e = digitAt(x, neg, k - 1);
    if (e) { borrow = e < 0 ? -1 : 0; break; }
  }
  return (u64) (digitAt(x, neg, j) + borrow) & MASK62;
}

KERNEL(GW) toWords(global const i64* x, u32 n, u32 neg, global const i32* borrows, global u32* out, u32 nOut) {
  u32 w = get_global_id(0);
  if (w >= nOut) { return; }
  u32 j = w * 32 / 62, sh = w * 32 % 62;
  u64 bits = binaryDigit(x, n, neg, borrows, j) >> sh;
  if (sh > 30) { bits |= binaryDigit(x, n, neg, borrows, j + 1) << (62 - sh); }
  out[w] = (u32) bits;
}
