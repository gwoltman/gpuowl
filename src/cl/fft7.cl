// Copyright (C) Mihai Preda & George Woltman

#pragma once

#define A(k) u[(base + k * step) % M]

#if FFT_FP64

// 38 FMA + 18 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 7-k, then each output pair j, 7-j is
// (u0 + sum_k cos(j*k*tau/7) sum_k) +- sum_k sin(j*k*tau/7) dif_k, with w = e^(2*pi*i/7).  cbase = u0 + cos(2*tau/7) * sum zeroes one
// term per cos row (the cosine nearest the middle of the range: the smallest coefficients), and the sin rows are scaled by sin(3*tau/7),
// the most accurately represented sine.
// Accumulation order (a double-precision study: a few percent less round-off than index order): cos terms by |coef| rank 1,0 (largest first), sin terms smallest first.
void OVERLOAD fft7by(T2 *u, u32 base, u32 step, u32 M) {
  const double COS2 = -0.2225209339563144042889;   // cos(2*tau/7): subtracted from every cos coefficient, it zeroes one term per cos row
  const double C1_C2 = 0.8460107358150479348139;   // cos(1*tau/7) - cos(2*tau/7)
  const double C3_C2 = -0.6784479339461047219472;  // cos(3*tau/7) - cos(2*tau/7)
  const double SIN3 = 0.4338837391175581204758;    // sin(3*tau/7): factored out of the sin rows, it makes one term per row a plain add
  const double S1_S3 = 1.8019377358048382524722;   // sin(1*tau/7) / sin(3*tau/7)
  const double S2_S3 = 2.2469796037174670610500;   // sin(2*tau/7) / sin(3*tau/7)

  X2_mul_t4(A(1), A(6));
  X2_mul_t4(A(2), A(5));
  X2_mul_t4(A(3), A(4));

  T2 sum = A(1) + A(2) + A(3);
  T2 cbase = fmaT2(COS2, sum, A(0));			// A(0) + COS2 * sum: each cos row below then has one term fewer
  A(0) = A(0) + sum;

  T2 tmp27a = fmaT2(C1_C2, A(1), cbase);
  T2 tmp36a = fmaT2(C1_C2, A(3), cbase);
  T2 tmp45a = fmaT2(C1_C2, A(2), cbase);

  tmp27a = fmaT2(C3_C2, A(3), tmp27a);
  tmp36a = fmaT2(C3_C2, A(2), tmp36a);
  tmp45a = fmaT2(C3_C2, A(1), tmp45a);

  T2 tmp27b = fmaT2(S1_S3, A(6), A(4));
  T2 tmp36b = fmaT2(S1_S3, -A(4), -A(5));
  T2 tmp45b = fmaT2(S1_S3, -A(5), A(6));

  tmp27b = fmaT2(S2_S3, A(5), tmp27b);
  tmp36b = fmaT2(S2_S3, A(6), tmp36b);
  tmp45b = fmaT2(S2_S3, A(4), tmp45b);

  fma_addsub(A(1), A(6), SIN3, tmp27a, tmp27b);
  fma_addsub(A(2), A(5), SIN3, tmp36a, tmp36b);
  fma_addsub(A(3), A(4), SIN3, tmp45a, tmp45b);
}

void OVERLOAD fft7(T2 *u) { return fft7by(u, 0, 1, 7); }

#endif

/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 38 FMA + 18 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 7-k, then each output pair j, 7-j is
// (u0 + sum_k cos(j*k*tau/7) sum_k) +- sum_k sin(j*k*tau/7) dif_k, with w = e^(2*pi*i/7).  cbase = u0 + cos(2*tau/7) * sum zeroes one
// term per cos row (the cosine nearest the middle of the range: the smallest coefficients), and the sin rows are scaled by sin(3*tau/7),
// the most accurately represented sine.
// Accumulation order (a double-precision study: a few percent less round-off than index order): cos terms by |coef| rank 1,0 (largest first), sin terms smallest first.
void OVERLOAD fft7by(F2 *u, u32 base, u32 step, u32 M) {
  const float COS2 = -0.2225209339563144042889f;   // cos(2*tau/7): subtracted from every cos coefficient, it zeroes one term per cos row
  const float C1_C2 = 0.8460107358150479348139f;   // cos(1*tau/7) - cos(2*tau/7)
  const float C3_C2 = -0.6784479339461047219472f;  // cos(3*tau/7) - cos(2*tau/7)
  const float SIN3 = 0.4338837391175581204758f;    // sin(3*tau/7): factored out of the sin rows, it makes one term per row a plain add
  const float S1_S3 = 1.8019377358048382524722f;   // sin(1*tau/7) / sin(3*tau/7)
  const float S2_S3 = 2.2469796037174670610500f;   // sin(2*tau/7) / sin(3*tau/7)

  X2_mul_t4(A(1), A(6));
  X2_mul_t4(A(2), A(5));
  X2_mul_t4(A(3), A(4));

  F2 sum = A(1) + A(2) + A(3);
  F2 cbase = fmaT2(COS2, sum, A(0));			// A(0) + COS2 * sum: each cos row below then has one term fewer
  A(0) = A(0) + sum;

  F2 tmp27a = fmaT2(C1_C2, A(1), cbase);
  F2 tmp36a = fmaT2(C1_C2, A(3), cbase);
  F2 tmp45a = fmaT2(C1_C2, A(2), cbase);

  tmp27a = fmaT2(C3_C2, A(3), tmp27a);
  tmp36a = fmaT2(C3_C2, A(2), tmp36a);
  tmp45a = fmaT2(C3_C2, A(1), tmp45a);

  F2 tmp27b = fmaT2(S1_S3, A(6), A(4));
  F2 tmp36b = fmaT2(S1_S3, -A(4), -A(5));
  F2 tmp45b = fmaT2(S1_S3, -A(5), A(6));

  tmp27b = fmaT2(S2_S3, A(5), tmp27b);
  tmp36b = fmaT2(S2_S3, A(6), tmp36b);
  tmp45b = fmaT2(S2_S3, A(4), tmp45b);

  fma_addsub(A(1), A(6), SIN3, tmp27a, tmp27b);
  fma_addsub(A(2), A(5), SIN3, tmp36a, tmp36b);
  fma_addsub(A(3), A(4), SIN3, tmp45a, tmp45b);
}

void OVERLOAD fft7(F2 *u) { return fft7by(u, 0, 1, 7); }

#endif

/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

#if PFA == 7 && PFA_W31 != 1205362885u
#error PFA_W31 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-7 DFT, y_j = sum_k u_k w^(jk) with w = 7^((p-1)/7) (pfaRootOfUnity), for the radix-PFA of a PFA middle.  The method of
// the FP version above: pair inputs k and 7-k (X2qs makes the sums and differences), then output pair j, 7-j is
// (u0 + sum_k c_jk sum_k) +- sum_k s_jk dif_k, with c_m = (w^m + w^-m)/2 and s_m = (w^m - w^-m)/2 in Z/pZ.  base = u0 + c3 * (the sum of
// the sums) zeroes one term per cos row (no roundoff in GF, so the base is chosen for the bounds:  c1 - c3 and c2 - c3 are small).  Each row
// is a GF31q (64-bit) dot product reduced once.
//
// The differences are signed (X2qs:  u_k - u_(7-k) as an i32, -M31..M31, one instruction, no fold), multiplied by the balanced sine constants
// (|s| <= M31/2, the sign of each row's term folded into the constant) with a signed multiply (mulqs / madqs).  Each row's sine sum b is then
// a signed value of magnitude at most sum_m |s_m| * M31.  K, a multiple of M31 at least that, is added once to the base, so y_j = a + b and
// y_(7-j) = a - b need no per-row offset.  Each row's first partial sums go through opaque(), which keeps NVVM from adding cbase to each row
// as a separate 64-bit add (see opaque in math.cl).
//
// Bounds (each rounded up):  inputs, and every add result, are 0..M31;  the differences -M31..M31.  The 64-bit values are given in M31^2
// (about 2^62):  a (the base and cos terms, K included) is 2.000..2.368, |b| at most 1.223, so a + b <= 3.591, below modM31(u64)'s limit
// 0xFFFFFFFF7FFFFFFF (4.000 M31^2), and a - b >= 0.777.  The cos products pass negAbove = M31, so no constant is folded.
void OVERLOAD fft7by(GF31 *u, u32 base, u32 step, u32 M) {
  const Z31 C3 = 141971251u;        // c3 = (w^3 + w^-3) / 2                 0.067 M31
  const Z31 C1_C3 = 263268255u;     // c1 - c3                               0.123 M31
  const Z31 C2_C3 = 384559815u;     // c2 - c3                               0.180 M31
  const i32 S1 = 800123379;         // s1 = (w - w^-1) / 2, balanced         0.373 M31
  const i32 S2 = 1073059520;        // s2                                    0.500 M31
  const i32 S3 = 752284155;         // s3                                    0.351 M31
  const u64 K = (u64) M31 << 32;    // 2.0 M31^2, low word 0 (u0 + K needs no add):  more than every row's |b| (1.223)

  X2qs(&A(1), &A(6));                                  // A(1) = 0..M31, A(6) = -M31..M31
  X2qs(&A(2), &A(5));                                  // A(2) = 0..M31, A(5) = -M31..M31
  X2qs(&A(3), &A(4));                                  // A(3) = 0..M31, A(4) = -M31..M31

  GF31 sum = add(add(A(1), A(2)), A(3));               // 0..M31
  GF31q u0K = makeGF31q(A(0).x + K, A(0).y + K);
  GF31q cbase = madq(sum, C3, opaque(u0K), M31);       // K + u0 + c3 * sum:  2.000..2.067 M31^2  (opaque:  K stays in the addend)
  A(0) = add(A(0), sum);                               // y0:  0..M31

  // One row at a time; the outputs are kept until the last row.  The comments give each partial sum's bound in M31^2.
  GF31q tmp27a = opaque(madq(A(1), C1_C3, cbase, M31));   // 2.000..2.189
  GF31q tmp27b = opaque(mulqs(A(6), S1));                 // +-0.373
  tmp27a = madq(A(2), C2_C3, tmp27a, M31);                // 2.000..2.368
  tmp27b = madqs(A(5), S2, tmp27b);                       // +-0.873
  tmp27b = madqs(A(4), S3, tmp27b);                       // +-1.223
  GF31 y1 = modM31(addq(tmp27a, tmp27b)), y6 = modM31(subqs(tmp27a, tmp27b));

  GF31q tmp36a = opaque(madq(A(1), C2_C3, cbase, M31));   // 2.000..2.246
  GF31q tmp36b = opaque(mulqs(A(6), S2));                 // +-0.500
  tmp36a = madq(A(3), C1_C3, tmp36a, M31);                // 2.000..2.368  (c6 = c1)
  tmp36b = madqs(A(5), -S3, tmp36b);                      // +-0.850  (s4 = -s3)
  tmp36b = madqs(A(4), -S1, tmp36b);                      // +-1.223  (s6 = -s1)
  GF31 y2 = modM31(addq(tmp36a, tmp36b)), y5 = modM31(subqs(tmp36a, tmp36b));

  GF31q tmp45a = opaque(madq(A(2), C1_C3, cbase, M31));   // 2.000..2.189  (c6 = c1)
  GF31q tmp45b = opaque(mulqs(A(6), S3));                 // +-0.351
  tmp45a = madq(A(3), C2_C3, tmp45a, M31);                // 2.000..2.368
  tmp45b = madqs(A(5), -S1, tmp45b);                      // +-0.723  (s6 = -s1)
  tmp45b = madqs(A(4), S2, tmp45b);                       // +-1.223
  GF31 y3 = modM31(addq(tmp45a, tmp45b)), y4 = modM31(subqs(tmp45a, tmp45b));

  A(1) = y1; A(6) = y6;
  A(2) = y2; A(5) = y5;
  A(3) = y3; A(4) = y4;
}

void OVERLOAD fft7(GF31 *u) { fft7by(u, 0, 1, 7); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

#if PFA == 7 && PFA_W61 != 1165310750493918737u
#error PFA_W61 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-7 DFT, y_j = sum_k u_k w^(jk) with w = 37^((p-1)/7) (pfaRootOfUnity), for the radix-PFA of a PFA middle.  The method of
// the FP version above: pair inputs k and 7-k (X2 makes the sums and differences), then output pair j, 7-j is
// (u0 + sum_k c_jk sum_k) +- sum_k s_jk dif_k, with c_m = (w^m + w^-m)/2 and s_m = (w^m - w^-m)/2 in Z/pZ.  base = u0 + c1 * (the sum of
// the sums) zeroes one term per cos row, and sbase = s1 * (the sum of the differences) one term per sin row.  Each row is a 128-bit dot
// product reduced once.
void OVERLOAD fft7by(GF61 *u, u32 base, u32 step, u32 M) {
  const Z61 C1 = 784002893271222362u;
  const Z61 C2_C1 = 989671222192671703u;            // c2 - c1
  const Z61 C3_C1 = 117084611814202137u;            // c3 - c1
  const Z61 S1 = 381307857222696375u;
  const Z61 S2_S1 = 1251050654277315072u;           // s2 - s1
  const Z61 S3_S1 = 450576194449256153u;            // s3 - s1
  const Z61 NS2_S1 = 292176640490986129u;           // -s2 - s1
  const Z61 NS3_S1 = 1092651100319045048u;          // -s3 - s1

  // Quick (unreduced) arithmetic, ranges in M61s as in math.cl.  Inputs 0..1+.
  X2qp(&A(1), &A(6));    // A(1) = 0..2+, A(6) = 1-..3+
  X2qp(&A(2), &A(5));    // A(2) = 0..2+, A(5) = 1-..3+
  X2qp(&A(3), &A(4));    // A(3) = 0..2+, A(4) = 1-..3+

  GF61 sum = modM61q(A(1) + A(2) + A(3), 0);     // 0..6+ before the mod
  GF61q cbase = madq(sum, C1, A(0));             // u0 + c1 * sum
  A(0) = modM61q(A(0) + sum, 0);                 // 0..2+ before the mod
  GF61 dsum = modM61q(A(6) + A(5) + A(4), -2);  // the sum of the differences:  3 (1-..3+) - 2 = 1-..7+ before the mod
  GF61q sbase = mulq(dsum, S1);                 // s1 * dsum

  // One row at a time, each k's cos and sin products together; the outputs are kept until the last row.  The sin rows start from
  // sbase = s1 * dsum:  each row's s1 term cancels (a row whose s1 term is negative computes its negated sines and swaps its outputs),
  // saving one multiply per row.  Each product of a quick value (sums below 2 M61, differences below 3 M61, dsum below 1+) and a constant
  // gives the n passed to weakModM61q, which keeps pc + qs and pc - qs + K M61 below 2^64 for modM61q.
  GF61q tmp27a = cbase;
  GF61q tmp27b = sbase;
  tmp27a = madq(A(2), C2_C1, tmp27a);
  tmp27b = madq(A(5), S2_S1, tmp27b);
  tmp27a = madq(A(3), C3_C1, tmp27a);
  tmp27b = madq(A(4), S3_S1, tmp27b);
  GF61 pc1 = weakModM61q(tmp27a, 2), qs1 = weakModM61q(tmp27b, 3);   // below 3 M61 and 4 M61
  GF61 y1 = modM61q(pc1 + qs1, 0), y6 = modM61q(pc1 - qs1, 4);   // below 7 M61 before the mods

  GF61q tmp36a = cbase;
  GF61q tmp36b = sbase;
  tmp36a = madq(A(1), C2_C1, tmp36a);
  tmp36b = madq(A(6), NS2_S1, tmp36b);
  tmp36a = madq(A(2), C3_C1, tmp36a);
  tmp36b = madq(A(5), S3_S1, tmp36b);
  GF61 pc2 = weakModM61q(tmp36a, 2), qs2 = weakModM61q(tmp36b, 2);   // below 3 M61 and 3 M61
  GF61 y5 = modM61q(pc2 + qs2, 0), y2 = modM61q(pc2 - qs2, 3);   // below 6 M61 before the mods  (sines negated)

  GF61q tmp45a = cbase;
  GF61q tmp45b = sbase;
  tmp45a = madq(A(1), C3_C1, tmp45a);
  tmp45b = madq(A(6), NS3_S1, tmp45b);
  tmp45a = madq(A(3), C2_C1, tmp45a);
  tmp45b = madq(A(4), NS2_S1, tmp45b);
  GF61 pc3 = weakModM61q(tmp45a, 2), qs3 = weakModM61q(tmp45b, 2);   // below 3 M61 and 3 M61
  GF61 y4 = modM61q(pc3 + qs3, 0), y3 = modM61q(pc3 - qs3, 3);   // below 6 M61 before the mods  (sines negated)

  A(1) = y1; A(6) = y6;
  A(2) = y2; A(5) = y5;
  A(3) = y3; A(4) = y4;
}

void OVERLOAD fft7(GF61 *u) { fft7by(u, 0, 1, 7); }

#endif


#undef A
