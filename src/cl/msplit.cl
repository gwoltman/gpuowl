// Copyright (C) George Woltman

// MSPLIT (experimental):  fftMiddleIn / fftMiddleOut of MIDDLE = MS * MS (9 or 16) split over MS threads per (x, y) point.  Thread j
// holds the elements j + MS * k, k < MS:  it applies their middleMul2 twiddles, does a radix-MS, the twiddles root(MIDDLE, j * k),
// exchanges through the LDS so that it holds the elements MS * n + j, does the second radix-MS and applies their middleMul twiddles
// (fftMiddleOut runs the same steps in the other order).  A point's threads are MS groups of G threads (G = TILE * TILE in place,
// IN_WG / OUT_WG out of place), so j is uniform over a wavefront when G >= the wavefront.  The split cuts the registers the middle
// needs by about MS and multiplies the threads by MS.
//
// MSPLIT=1 splits the FFT's 64-bit type, FP64 (MIDDLE 9 or 16) or GF61 (16:  its odd middles are always PFA).  Splitting the 32-bit
// types never paid (RTX 5080, TITAN V), so they don't.  The host (clDefines) turns it off for PFA, L2_STRIPING and MIDDLE_CHAIN61,
// and in place it lowers TILE64 to 8.  middleMul2's twiddles are as middleMul2's:  a base per thread and a chain of the step w^MS, read
// from a table of the split's own (see msplitTrig).  MSPLIT_TRIG=1 (FP64 only, default 0) instead has one slowTrig per element.
// MSPLIT_FFT=1 (the default):  the radix-MS steps and the twiddles between them per thread group
// (fft4_16a..d, fft3_9a..c and fft3_9delayed, see below).  Included by fftmiddlein.cl and fftmiddleout.cl after middle.cl.

#if !defined(MSPLIT)
#define MSPLIT 0
#endif

#define MSPLIT_FP64 (FFT_FP64 && MSPLIT)
#define MSPLIT_GF61 (NTT_GF61 && MSPLIT)

#if MSPLIT

#if MIDDLE == 9
#define MS 3
#elif MIDDLE == 16
#define MS 4
#else
#error MSPLIT needs MIDDLE 9 or 16
#endif
#if PFA || L2_STRIPING
#error MSPLIT needs no PFA and no L2_STRIPING
#endif
#if MSPLIT_GF61 && MIDDLE != 16
#error MSPLIT of GF61 needs MIDDLE 16
#endif
#if MSPLIT_GF61 && MIDDLE_CHAIN61
#error MSPLIT needs the GF61 middleMul table (MIDDLE_CHAIN61 off)
#endif
#if !defined(MSPLIT_TRIG)
#define MSPLIT_TRIG 0
#endif
#if !defined(MSPLIT_FFT)
#define MSPLIT_FFT 1
#endif

#define MS_LDS(G) (MS * MS * (G))         // Elements of a split kernel's LDS:  the exchange (and the transposes) of MS groups of G threads

#if MSPLIT_FP64
#define MFP2 T2
#define MFP T
// The twiddles root(MIDDLE, j * k) between the two radix-MS
void OVERLOAD msplitInnerTwiddle(MFP2 *u, u32 j) {
#if MS == 3
  const MFP2 W1 = U2((MFP) 0.7660444431189780352024, (MFP) 0.6427876096865393263226);    // e^(i*tau*1/9)
  const MFP2 W2 = U2((MFP) 0.1736481776669303488517, (MFP) 0.9848077530122080593667);    // e^(i*tau*2/9)
  const MFP2 W4 = U2((MFP) -0.9396926207859083840541, (MFP) 0.3420201433256687330441);   // e^(i*tau*4/9)
  if (j == 1) { u[1] = cmul(u[1], W1); u[2] = cmul(u[2], W2); }
  if (j == 2) { u[1] = cmul(u[1], W2); u[2] = cmul(u[2], W4); }
#else
  const MFP2 W1 = U2((MFP) 0.9238795325112867561282, (MFP) 0.3826834323650897717285);    // e^(i*tau*1/16)
  const MFP2 W3 = U2((MFP) 0.3826834323650897717285, (MFP) 0.9238795325112867561282);    // e^(i*tau*3/16)
  if (j == 1) { u[1] = cmul(u[1], W1); u[2] = mul_t8(u[2]);  u[3] = cmul(u[3], W3); }
  if (j == 2) { u[1] = mul_t8(u[1]);  u[2] = mul_t4(u[2]);  u[3] = mul_3t8(u[3]); }
  if (j == 3) { u[1] = cmul(u[1], W3); u[2] = mul_3t8(u[2]); u[3] = cmul(u[3], -W1); }
#endif
}

#if MS == 3
// MSPLIT_FFT=1:  as fft8_16b, the first radix-3's twiddles w^m (w = root(9, 1), m = j * k) are partial products w^m / f(m) by two FMAs,
// f(m) the larger of cos and sin:  f(1) = cos 40, f(2) = sin 80, f(4) = cos 160.  The second radix-3 of thread group j gets its inputs
// 1 and 2 with the delayed f(j), f(2j) and applies them in FMAs that were adds (fft3_9delayed), so it costs the same as fft3.
#define MSF1 0.766044443118978035202392650555        // cos 40
#define MSF2 0.984807753012208059366743024589        // sin 80
#define MSF4 -0.939692620785908384054109277325       // cos 160
MFP2 OVERLOAD msplitPartialCos(MFP2 u, MFP tn) { return U2(fma(-tn, u.y, u.x), fma(tn, u.x, u.y)); }    // u * (1 + i tn), tn = tan
MFP2 OVERLOAD msplitPartialSin(MFP2 u, MFP ct) { return U2(fma(ct, u.x, -u.y), fma(ct, u.y, u.x)); }     // u * (ct + i), ct = cot
void OVERLOAD fft3_9a(MFP2 *u) { fft3(u); }
void OVERLOAD fft3_9b(MFP2 *u) {                  // Twiddles 1, w, w^2, delaying f(1), f(2)
  fft3(u);
  u[1] = msplitPartialCos(u[1], (MFP) 0.839099631177280011763127298123);      // tan 40
  u[2] = msplitPartialSin(u[2], (MFP) 0.176326980708464973471090386869);      // cot 80
}
void OVERLOAD fft3_9c(MFP2 *u) {                  // Twiddles 1, w^2, w^4, delaying f(2), f(4)
  fft3(u);
  u[1] = msplitPartialSin(u[1], (MFP) 0.176326980708464973471090386869);
  u[2] = msplitPartialCos(u[2], (MFP) -0.363970234266202361351047882777);     // tan 160
}
// fft3 of u[0], f1 * u[1], f2 * u[2] (6 FMA + 6 ADD as fft3:  the sum and difference of u[1] and u[2] are FMAs by f2 / f1 that keep a
// delayed f1, applied in the FMAs that follow)
// r = f2 / f1, h = f1 / 2, sf = sin(tau/3) f1
void OVERLOAD fft3_9delayed(MFP2 *u, MFP f1, MFP r, MFP h, MFP sf) {
  MFP2 s = fmaT2(r, u[2], u[1]);
  MFP2 d = mul_t4(fmaT2(-r, u[2], u[1]));
  MFP2 t = fmaT2(-h, s, u[0]);
  u[0] = fmaT2(f1, s, u[0]);
  u[1] = fmaT2(sf, d, t);
  u[2] = fmaT2(-sf, d, t);
}
#define MSPLIT_DELAYED(u, F1, F2) fft3_9delayed(u, (MFP) (F1), (MFP) ((F2) / (F1)), (MFP) (0.5 * (F1)), (MFP) (0.8660254037844386467637231707529 * (F1)))
void OVERLOAD msplitStage1(MFP2 *u, u32 j) {
  if (MSPLIT_FFT) {
    if (j == 0) { fft3_9a(u); }
    if (j == 1) { fft3_9b(u); }
    if (j == 2) { fft3_9c(u); }
  } else {
    fft3(u);
    msplitInnerTwiddle(u, j);
  }
}
void OVERLOAD msplitStage2(MFP2 *u, u32 j) {
  if (MSPLIT_FFT && j == 1) { MSPLIT_DELAYED(u, MSF1, MSF2); }
  else if (MSPLIT_FFT && j == 2) { MSPLIT_DELAYED(u, MSF2, MSF4); }
  else { fft3(u); }
}
#undef MSF1
#undef MSF2
#undef MSF4
#undef MSPLIT_DELAYED
#else
// MSPLIT_FFT=1:  as fft8_16b, the first radix-4's twiddles w^m (w = root(16, 1), m = j * k) are partial products w^m / f(m), each two
// FMAs or two adds:  f(1) = f(3) = C1 = cos(tau/16) (w = C1 (1 + i T), w^3 = C1 (T + i), T = tan(tau/16)), f(2) = f(6) = sqrt(1/2)
// (w^2 = (1 + i) / sqrt2, w^6 = (-1 + i) / sqrt2), f(4) = 1 (w^4 = i), f(9) = -C1 (w^9 = -C1 (1 + i T)).  The second radix-4 of
// thread group j gets its inputs 1, 2, 3 with the delayed f(j), f(2j), f(3j) and applies them in FMAs that were adds (fft4_16delayed).
#define MST 0.4142135623730950488016887242097      // tan(tau/16)
#define MSC1 0.9238795325112867561281831893968     // cos(tau/16)
MFP2 OVERLOAD msplitPartialCos(MFP2 u, MFP tn) { return U2(fma(-tn, u.y, u.x), fma(tn, u.x, u.y)); }    // u * (1 + i tn)
MFP2 OVERLOAD msplitPartialSin(MFP2 u, MFP ct) { return U2(fma(ct, u.x, -u.y), fma(ct, u.y, u.x)); }     // u * (ct + i)
MFP2 OVERLOAD msplitT8(MFP2 u) { return U2(u.x - u.y, u.x + u.y); }                                      // u * (1 + i)
MFP2 OVERLOAD msplit3T8(MFP2 u) { return U2(-u.x - u.y, u.x - u.y); }                                    // u * (-1 + i)
void OVERLOAD fft4_16a(MFP2 *u) { fft4(u); }
void OVERLOAD fft4_16b(MFP2 *u) {                 // Twiddles 1, w, w^2, w^3, delaying C1, sqrt(1/2), C1
  fft4(u);
  u[1] = msplitPartialCos(u[1], (MFP) MST);
  u[2] = msplitT8(u[2]);
  u[3] = msplitPartialSin(u[3], (MFP) MST);
}
void OVERLOAD fft4_16c(MFP2 *u) {                 // Twiddles 1, w^2, w^4, w^6, delaying sqrt(1/2), 1, sqrt(1/2)
  fft4(u);
  u[1] = msplitT8(u[1]);
  u[2] = mul_t4(u[2]);
  u[3] = msplit3T8(u[3]);
}
void OVERLOAD fft4_16d(MFP2 *u) {                 // Twiddles 1, w^3, w^6, w^9, delaying C1, sqrt(1/2), -C1
  fft4(u);
  u[1] = msplitPartialSin(u[1], (MFP) MST);
  u[2] = msplit3T8(u[2]);
  u[3] = msplitPartialCos(u[3], (MFP) MST);
}
// fft4 of u[0], f1 * u[1], f2 * u[2], f1 * sign3 * u[3] (sign3 = +-1):  the same adds as fft4, those of u[2] and the second layer as FMAs
void OVERLOAD fft4_16delayed(MFP2 *u, MFP f1, MFP f2, MFP sign3) {
  MFP2 x0 = fmaT2(f2, u[2], u[0]);
  MFP2 x2 = fmaT2(-f2, u[2], u[0]);
  MFP2 s = fmaT2(sign3, u[3], u[1]);              // sign3 * u[3] is an add or a subtract
  MFP2 d = mul_t4(fmaT2(-sign3, u[3], u[1]));
  u[0] = fmaT2(f1, s, x0);
  u[2] = fmaT2(-f1, s, x0);
  u[1] = fmaT2(f1, d, x2);
  u[3] = fmaT2(-f1, d, x2);
}
void OVERLOAD msplitStage1(MFP2 *u, u32 j) {
  if (MSPLIT_FFT) {
    if (j == 0) { fft4_16a(u); }
    if (j == 1) { fft4_16b(u); }
    if (j == 2) { fft4_16c(u); }
    if (j == 3) { fft4_16d(u); }
  } else {
    fft4(u);
    msplitInnerTwiddle(u, j);
  }
}
void OVERLOAD msplitStage2(MFP2 *u, u32 j) {
  if (MSPLIT_FFT && j == 1) { fft4_16delayed(u, (MFP) MSC1, (MFP) M_SQRT1_2, (MFP) 1); }
  else if (MSPLIT_FFT && j == 2) { fft4_16delayed(u, (MFP) M_SQRT1_2, (MFP) 1, (MFP) 1); }
  else if (MSPLIT_FFT && j == 3) { fft4_16delayed(u, (MFP) MSC1, (MFP) M_SQRT1_2, (MFP) -1); }
  else { fft4(u); }
}
#undef MST
#undef MSC1
#endif
#undef MFP2
#undef MFP
#endif

#if MSPLIT_GF61
// root(16, 1), root(16, 3) and root(16, 9) = -root(16, 1) in GF(M61^2)
void OVERLOAD msplitInnerTwiddle(GF61 *u, u32 j) {
  const GF61 W1 = U2((Z61) 22027337052962166ull, (Z61) 1693317751237720973ull);
  const GF61 W3 = U2((Z61) 1693317751237720973ull, (Z61) 22027337052962166ull);
  const GF61 W9 = U2((Z61) (M61 - 22027337052962166ull), (Z61) (M61 - 1693317751237720973ull));
  if (j == 1) { u[1] = cmul(u[1], W1); u[2] = mul_t8(u[2]);  u[3] = cmul(u[3], W3); }
  if (j == 2) { u[1] = mul_t8(u[1]);  u[2] = mul_t4(u[2]);  u[3] = mul_3t8(u[3]); }
  if (j == 3) { u[1] = cmul(u[1], W3); u[2] = mul_3t8(u[2]); u[3] = cmul(u[3], W9); }
}

// MSPLIT_FFT=1:  the first radix-4 and its twiddles w^(j * k) (w = root(16, 1)) as one routine per thread group j, fft4_16a..d, in the
// style of fft8_16a/b:  quick adds, the twiddles folded into the reductions (mul_t16q, delayed mul_t8q / mul_3t8q with only one component
// made less negative, w^3 a = swap(w * conj(a)) and w^9 a = w * (-a)), outputs left 0..2+ for the second radix-4 (fft4CoreSpecial1).

// The quick radix-4 of inputs 0..1+:  u[0] = X0 = 0..4+, u[1] = X1, u[2] = X2, u[3] = X3 = -2-..2+
void OVERLOAD fft4q(GF61 *u) {
  GF61 x0 = u[0] + u[2], x2 = u[0] - u[2], s = u[1] + u[3], d = u[1] - u[3];
  u[0] = x0 + s;
  u[2] = x0 - s;
  u[1] = U2(x2.x - d.y, x2.y + d.x);           // x2 + i d
  u[3] = U2(x2.x + d.y, x2.y - d.x);           // x2 - i d
}
GF61 OVERLOAD swapxy(GF61 a) { return U2(a.y, a.x); }

void OVERLOAD fft4_16a(GF61 *u) { fft4(u); }   // Twiddles 1, 1, 1, 1.  Outputs 0..1+
void OVERLOAD fft4_16b(GF61 *u) {              // Twiddles 1, w, w^2, w^3.  Outputs 0..1.74+
  fft4q(u);
  u[0] = modM61(u[0]);
  u[1] = mul_t16q(modM61q(u[1], 2));                              // 0..1.28+, 0..1.74+
  u[2].x = lessneg(u[2].x, 2);                                    // u[2] = 0-..2+, -2-..2+
  u[2] = mul_t8q(u[2], 5);                                        // y - x and -(x + y) are -4-..2+
  u[3] = swapxy(mul_t16q(modM61q(U2(u[3].x, -u[3].y), 2)));       // w^3 X3 = swap(w * conj(X3)):  0..1.74+, 0..1.28+
}
void OVERLOAD fft4_16c(GF61 *u) {              // Twiddles 1, w^2, w^4, w^6.  Outputs 0..1+
  fft4q(u);
  u[0] = modM61(u[0]);
  u[1].x = lessneg(u[1].x, 2);
  u[1] = mul_t8q(u[1], 5);
  u[2] = modM61q(U2(-u[2].y, u[2].x), 2);                         // i X2
  u[3].y = lessneg(u[3].y, 2);                                    // u[3] = -2-..2+, 0-..2+
  u[3] = mul_3t8q(u[3], 3);                                       // x + y and y - x are -2-..4+
}
void OVERLOAD fft4_16d(GF61 *u) {              // Twiddles 1, w^3, w^6, w^9.  Outputs 0..1.74+
  fft4q(u);
  u[0] = modM61(u[0]);
  u[1] = swapxy(mul_t16q(modM61q(U2(u[1].x, -u[1].y), 2)));       // w^3 X1
  u[2].y = lessneg(u[2].y, 2);
  u[2] = mul_3t8q(u[2], 3);
  u[3] = mul_t16q(modM61q(U2(-u[3].x, -u[3].y), 2));                             // w^9 X3 = w * (-X3)
}

void OVERLOAD msplitStage1(GF61 *u, u32 j) {
  if (MSPLIT_FFT) {
    if (j == 0) { fft4_16a(u); }
    if (j == 1) { fft4_16b(u); }
    if (j == 2) { fft4_16c(u); }
    if (j == 3) { fft4_16d(u); }
  } else {
    fft4(u);
    msplitInnerTwiddle(u, j);
  }
}
// MSPLIT_FFT=1:  inputs 0..2+ (fft4_16b/d's outputs)
void OVERLOAD msplitStage2(GF61 *u, u32 j) {
  if (MSPLIT_FFT) {
    fft4CoreSpecial1(u);
    SWAP(u[1], u[2]);
  } else {
    fft4(u);
  }
}

// middleMul2 of the elements i = j + 4k, as middleMul2 does it:  base = root(ND, x * y + j * x * SMALL_HEIGHT) from trig2 and the split's
// table xt (root(MIDDLE * WIDTH, k), k < MS * WIDTH, stored as trig1, its Karatsuba terms at MS * WIDTH + k, see msplitTrig), then a
// chain of the step w^MS = xt[MS * x]
void OVERLOAD msplitMul2(GF61 *u, u32 x, u32 y, u32 j, u32 unused, TrigGF61 trig, TrigGF61 xt) {
  TrigGF61 trig2 = trig + SMALL_HEIGHT * (MIDDLE - 1) + WIDTH;
  u32 d = x * y + j * x * SMALL_HEIGHT;
  GF61 base = cmulTwid(TFLOAD(&trig2[d % SMALL_HEIGHT]), xt, d / SMALL_HEIGHT, MS * WIDTH, CMUL61_MID2);
  GF61 w = TFLOAD(&xt[MS * x]);
  GF61 wpre = CMUL61_MID2 ? TFLOAD(&xt[MS * WIDTH + MS * x]) : U2((Z61) 0, (Z61) 0);
  u[0] = cmul(u[0], base);
  for (u32 k = 1; k < MS; ++k) { base = CMUL61_MID2 ? cmulTwid(base, w, wpre) : cmulTwid(base, w); u[k] = cmul(u[k], base); }
}

// middleMul of the elements i = j + 4k:  the table's root(MIDDLE * SMALL_HEIGHT, i * s) at (i - 1) * SMALL_HEIGHT + s
void OVERLOAD msplitMul(GF61 *u, u32 s, u32 j, TrigGF61 trig, TrigGF61 unused) {
  for (u32 k = 0; k < MS; ++k) { u32 i = j + MS * k; if (i) { u[k] = cmulTwid(u[k], trig, (i - 1) * SMALL_HEIGHT + s, MIDDLE_PRE61, CMUL61_MID); } }
}
#endif

// The split kernels' bodies, once per type that splits
#if MSPLIT_FP64
#define MGF T2
#define MZ T
#define MTRIG Trig
#define MSPLIT_FP 1
#include "msplitbody.cl"
#undef MGF
#undef MZ
#undef MTRIG
#undef MSPLIT_FP
#endif

#if MSPLIT_GF61
#define MGF GF61
#define MZ Z61
#define MTRIG TrigGF61
#define MSPLIT_FP 0
#include "msplitbody.cl"
#undef MGF
#undef MZ
#undef MTRIG
#undef MSPLIT_FP
#endif

#endif
