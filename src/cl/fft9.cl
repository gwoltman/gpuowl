// Copyright (C) Mihai Preda and George Woltman

#include "fft3.cl"

#if FFT_FP64

// 52 FMA + 28 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 9-k, then each output pair j, 9-j is
// (u0 + sum_k cos(j*k*tau/9) sum_k) +- sum_k sin(j*k*tau/9) dif_k, with w = e^(2*pi*i/9).  9 = 3*3 adds structure: rows 1, 2, 4 all hold
// cos(3*tau/9) = -1/2 on sum_3, which base = u0 - sum_3/2 absorbs, and their remaining cosines are a permutation of cos(tau/9),
// cos(2*tau/9), cos(4*tau/9), so base += cos(2*tau/9) * (sum_1 + sum_2 + sum_4) zeroes one more term per row (cos(2*tau/9) is the one
// nearest the middle of the range: the smallest coefficients).  Row 3 is u0 + sum_3 - (sum_1 + sum_2 + sum_4)/2 +- sqrt(3)/2 (dif_1 -
// dif_2 + dif_4).  A double-precision study put this version's round-off 18% below the former Nussbaumer one (12 FMA + 8 MUL + 72 ADD),
// with the sin rows 1, 2, 4 scaled by sin(4*tau/9), the best of the four choices.
// Accumulation order (a double-precision study: a few percent less round-off than index order): cos terms smallest first, sin terms smallest first.

void OVERLOAD fft9(T2 *u) {
  const double COS2 = 0.1736481776669303488517;    // cos(2*tau/9): subtracted from the cos coefficients of rows 1, 2, 4, it zeroes one term per row
  const double C1_C2 = 0.5923962654520476863507;   // cos(tau/9) - cos(2*tau/9)
  const double C4_C2 = -1.1133407984528387329058;  // cos(4*tau/9) - cos(2*tau/9)
  const double SIN3 = 0.8660254037844386467637;    // sin(3*tau/9) = sqrt(3)/2: the scale of row 3, whose sin terms are all +-1
  const double SIN4 = 0.3420201433256687330441;    // sin(4*tau/9): factored out of the sin rows 1, 2, 4, it makes one term per row a plain add
  const double S1_S4 = 1.8793852415718167681082;   // sin(tau/9) / sin(4*tau/9)
  const double S2_S4 = 2.8793852415718167681082;   // sin(2*tau/9) / sin(4*tau/9)
  const double S3_S4 = 2.5320888862379560704048;   // sin(3*tau/9) / sin(4*tau/9)

  X2_mul_t4(u[1], u[8]);
  X2_mul_t4(u[2], u[7]);
  X2_mul_t4(u[3], u[6]);
  X2_mul_t4(u[4], u[5]);

  T2 sum = u[1] + u[2] + u[4];
  T2 t = u[0] + u[3];
  T2 cbase = fmaT2(-0.5, u[3], u[0]);                   // u[0] - u[3]/2: rows 1, 2, 4 all have cos(3*tau/9) = -1/2 on u[3]
  cbase = fmaT2(COS2, sum, cbase);                      // + COS2 * sum: each of rows 1, 2, 4 then has one term fewer
  T2 tmp47a = fmaT2(-0.5, sum, t);			// row 3: u[0] + u[3] - (u[1] + u[2] + u[4]) / 2
  u[0] = t + sum;

  T2 tmp29a = fmaT2(C1_C2, u[1], cbase);
  T2 tmp38a = fmaT2(C1_C2, u[4], cbase);
  T2 tmp56a = fmaT2(C1_C2, u[2], cbase);

  tmp29a = fmaT2(C4_C2, u[4], tmp29a);
  tmp38a = fmaT2(C4_C2, u[2], tmp38a);
  tmp56a = fmaT2(C4_C2, u[1], tmp56a);

  T2 tmp29b = fmaT2(S1_S4, u[8], u[5]);
  T2 tmp38b = fmaT2(S1_S4, -u[5], u[7]);
  T2 tmp56b = fmaT2(S1_S4, -u[7], u[8]);

  tmp29b = fmaT2(S3_S4, u[6], tmp29b);
  tmp38b = fmaT2(S3_S4, -u[6], tmp38b);
  tmp56b = fmaT2(S3_S4, u[6], tmp56b);

  tmp29b = fmaT2(S2_S4, u[7], tmp29b);
  tmp38b = fmaT2(S2_S4, u[8], tmp38b);
  tmp56b = fmaT2(S2_S4, -u[5], tmp56b);

  T2 tmp47b = u[8] - u[7] + u[5];                               // row 3: sin(3*tau/9) * (dif1 - dif2 + dif4)

  fma_addsub(u[1], u[8], SIN4, tmp29a, tmp29b);
  fma_addsub(u[2], u[7], SIN4, tmp38a, tmp38b);
  fma_addsub(u[4], u[5], SIN4, tmp56a, tmp56b);
  fma_addsub(u[3], u[6], SIN3, tmp47a, tmp47b);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 52 FMA + 28 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 9-k, then each output pair j, 9-j is
// (u0 + sum_k cos(j*k*tau/9) sum_k) +- sum_k sin(j*k*tau/9) dif_k, with w = e^(2*pi*i/9).  9 = 3*3 adds structure: rows 1, 2, 4 all hold
// cos(3*tau/9) = -1/2 on sum_3, which base = u0 - sum_3/2 absorbs, and their remaining cosines are a permutation of cos(tau/9),
// cos(2*tau/9), cos(4*tau/9), so base += cos(2*tau/9) * (sum_1 + sum_2 + sum_4) zeroes one more term per row (cos(2*tau/9) is the one
// nearest the middle of the range: the smallest coefficients).  Row 3 is u0 + sum_3 - (sum_1 + sum_2 + sum_4)/2 +- sqrt(3)/2 (dif_1 -
// dif_2 + dif_4).  A double-precision study put this version's round-off 18% below the former Nussbaumer one (12 FMA + 8 MUL + 72 ADD),
// with the sin rows 1, 2, 4 scaled by sin(4*tau/9), the best of the four choices.
// Accumulation order (a double-precision study: a few percent less round-off than index order): cos terms smallest first, sin terms smallest first.

void OVERLOAD fft9(F2 *u) {
  const float COS2 = 0.1736481776669303488517f;    // cos(2*tau/9): subtracted from the cos coefficients of rows 1, 2, 4, it zeroes one term per row
  const float C1_C2 = 0.5923962654520476863507f;   // cos(tau/9) - cos(2*tau/9)
  const float C4_C2 = -1.1133407984528387329058f;  // cos(4*tau/9) - cos(2*tau/9)
  const float SIN3 = 0.8660254037844386467637f;    // sin(3*tau/9) = sqrt(3)/2: the scale of row 3, whose sin terms are all +-1
  const float SIN4 = 0.3420201433256687330441f;    // sin(4*tau/9): factored out of the sin rows 1, 2, 4, it makes one term per row a plain add
  const float S1_S4 = 1.8793852415718167681082f;   // sin(tau/9) / sin(4*tau/9)
  const float S2_S4 = 2.8793852415718167681082f;   // sin(2*tau/9) / sin(4*tau/9)
  const float S3_S4 = 2.5320888862379560704048f;   // sin(3*tau/9) / sin(4*tau/9)

  X2_mul_t4(u[1], u[8]);
  X2_mul_t4(u[2], u[7]);
  X2_mul_t4(u[3], u[6]);
  X2_mul_t4(u[4], u[5]);

  F2 sum = u[1] + u[2] + u[4];
  F2 t = u[0] + u[3];
  F2 cbase = fmaT2(-0.5f, u[3], u[0]);                   // u[0] - u[3]/2: rows 1, 2, 4 all have cos(3*tau/9) = -1/2 on u[3]
  cbase = fmaT2(COS2, sum, cbase);                      // + COS2 * sum: each of rows 1, 2, 4 then has one term fewer
  F2 tmp47a = fmaT2(-0.5f, sum, t);			// row 3: u[0] + u[3] - (u[1] + u[2] + u[4]) / 2
  u[0] = t + sum;

  F2 tmp29a = fmaT2(C1_C2, u[1], cbase);
  F2 tmp38a = fmaT2(C1_C2, u[4], cbase);
  F2 tmp56a = fmaT2(C1_C2, u[2], cbase);

  tmp29a = fmaT2(C4_C2, u[4], tmp29a);
  tmp38a = fmaT2(C4_C2, u[2], tmp38a);
  tmp56a = fmaT2(C4_C2, u[1], tmp56a);

  F2 tmp29b = fmaT2(S1_S4, u[8], u[5]);
  F2 tmp38b = fmaT2(S1_S4, -u[5], u[7]);
  F2 tmp56b = fmaT2(S1_S4, -u[7], u[8]);

  tmp29b = fmaT2(S3_S4, u[6], tmp29b);
  tmp38b = fmaT2(S3_S4, -u[6], tmp38b);
  tmp56b = fmaT2(S3_S4, u[6], tmp56b);

  tmp29b = fmaT2(S2_S4, u[7], tmp29b);
  tmp38b = fmaT2(S2_S4, u[8], tmp38b);
  tmp56b = fmaT2(S2_S4, -u[5], tmp56b);

  F2 tmp47b = u[8] - u[7] + u[5];                               // row 3: sin(3*tau/9) * (dif1 - dif2 + dif4)

  fma_addsub(u[1], u[8], SIN4, tmp29a, tmp29b);
  fma_addsub(u[2], u[7], SIN4, tmp38a, tmp38b);
  fma_addsub(u[4], u[5], SIN4, tmp56a, tmp56b);
  fma_addsub(u[3], u[6], SIN3, tmp47a, tmp47b);
}

#endif

/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

#if PFA == 9 && PFA_W31 != 309107220u
#error PFA_W31 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-9 DFT, y_j = sum_k u_k w^(jk) with w = 7^((p-1)/9) (pfaRootOfUnity).  Paired as in fft7/fft11 (64-bit dot products reduced
// once), with the structure 9 = 3 * 3 adds as in the FP fft9:  rows 1, 2, 4 share base = u0 + c3 sum3 + c4 (sum1 + sum2 + sum4).  A c4 base
// (c1 + c2 + c4 = 0) leaves the smallest cos constants, c1 - c4 and c2 - c4.  c3 = -1/2 and c4 are used as negative constants (balanced),
// with a signed multiply (madqs) of the reduced sums, so the base ranges from K - 0.712 to K:  that keeps a + |b| below 4.0 with K = 2.0.
//
// The differences are signed (X2qs:  u_k - u_(9-k) as an i32, -M31..M31, one instruction, no fold), multiplied by the balanced sine constants
// (|s| <= M31/2, the sign of each row's term folded into the constant) with a signed multiply (mulqs / madqs).  Each row's sine sum b is then
// a signed value of magnitude at most sum_m |s_m| * M31.  K, a multiple of M31 at least that plus the base's negative part, is added once to
// the base (and to row 3's t), so y_j = a + b and y_(9-j) = a - b need no per-row offset.  Each row's first partial sums go through opaque(),
// which keeps NVVM from adding cbase to each row as a separate 64-bit add (see opaque in math.cl).
//
// Bounds (each rounded outward):  inputs, and every add result, are 0..M31;  the differences -M31..M31.  The 64-bit values are given in M31^2
// (about 2^62):  a (the base and cos terms, K included) is 1.288..2.635 (row 3:  1.499..2.001), |b| at most 1.205 (row 3:  0.615), so
// a + b <= 3.840, below modM31(u64)'s limit 0xFFFFFFFF7FFFFFFF (4.000 M31^2), and a - b >= 0.083.  The unsigned cos products pass
// negAbove = M31, so no constant is folded.
void OVERLOAD fft9(GF31 *u) {
  const i32 C3 = -1073741824;       // c3 = -1/2, balanced                  -0.500 M31
  const i32 C4 = -454401419;        // c4 = (w^4 + w^-4) / 2, balanced      -0.212 M31
  const Z31 C1_C4 = 1041200310u;    // c1 - c4                               0.485 M31
  const Z31 C2_C4 = 322003947u;     // c2 - c4                               0.150 M31
  const i32 S1 = -277691671;        // s1 = (w - w^-1) / 2, balanced        -0.130 M31
  const i32 S2 = 942092970;         // s2                                    0.439 M31
  const i32 S3 = 439735912;         // s3                                    0.205 M31
  const i32 S4 = -927699006;        // s4                                   -0.432 M31
  const u64 K = (u64) M31 << 32;    // 2.0 M31^2, low word 0 (u0 + K needs no add):  more than every row's |b| plus 0.712 (1.917)

  X2qs(&u[1], &u[8]);                                  // u[1] = 0..M31, u[8] = -M31..M31
  X2qs(&u[2], &u[7]);                                  // u[2] = 0..M31, u[7] = -M31..M31
  X2qs(&u[3], &u[6]);                                  // u[3] = 0..M31, u[6] = -M31..M31
  X2qs(&u[4], &u[5]);                                  // u[4] = 0..M31, u[5] = -M31..M31

  GF31 s124 = add(add(u[1], u[2]), u[4]);              // 0..M31
  GF31 t = add(u[0], u[3]);                            // 0..M31
  GF31q u0K = makeGF31q(u[0].x + K, u[0].y + K);
  GF31q tK = makeGF31q(t.x + K, t.y + K);
  GF31q cbase = madqs(s124, C4, madqs(u[3], C3, opaque(u0K)));       // rows 1, 2, 4:  K + u0 + c3 sum3 + c4 (sum1 + sum2 + sum4):  1.499..2.001, 1.288..2.001
  u[0] = add(t, s124);                                 // y0:  0..M31

  // One row at a time; the outputs are kept until the last row.  The comments give each partial sum's bounds in M31^2.
  GF31q tmp29a = opaque(madq(u[1], C1_C4, cbase, M31));   // 1.288..2.485
  GF31q tmp29b = opaque(mulqs(u[8], S1));                 // +-0.130
  tmp29a = madq(u[2], C2_C4, tmp29a, M31);                // 1.288..2.635
  tmp29b = madqs(u[7], S2, tmp29b);                       // +-0.569
  tmp29b = madqs(u[6], S3, tmp29b);                       // +-0.773
  tmp29b = madqs(u[5], S4, tmp29b);                       // +-1.205
  GF31 y1 = modM31(addq(tmp29a, tmp29b)), y8 = modM31(subqs(tmp29a, tmp29b));

  GF31q tmp38a = opaque(madq(u[1], C2_C4, cbase, M31));   // 1.288..2.150
  GF31q tmp38b = opaque(mulqs(u[8], S2));                 // +-0.439
  tmp38a = madq(u[4], C1_C4, tmp38a, M31);                // 1.288..2.635  (c8 = c1)
  tmp38b = madqs(u[7], S4, tmp38b);                       // +-0.871
  tmp38b = madqs(u[6], -S3, tmp38b);                      // +-1.076  (s6 = -s3)
  tmp38b = madqs(u[5], -S1, tmp38b);                      // +-1.205  (s8 = -s1)
  GF31 y2 = modM31(addq(tmp38a, tmp38b)), y7 = modM31(subqs(tmp38a, tmp38b));

  GF31q tmp47a = madqs(s124, C3, opaque(tK));             // 1.499..2.001
  GF31q tmp47b = opaque(mulqs(u[8], S3));                 // +-0.205
  tmp47b = madqs(u[7], -S3, tmp47b);                      // +-0.410  (s6 = -s3)
  tmp47b = madqs(u[5], S3, tmp47b);                       // +-0.615
  GF31 y3 = modM31(addq(tmp47a, tmp47b)), y6 = modM31(subqs(tmp47a, tmp47b));

  GF31q tmp56a = opaque(madq(u[2], C1_C4, cbase, M31));   // 1.288..2.485  (c8 = c1)
  GF31q tmp56b = opaque(mulqs(u[8], S4));                 // +-0.432
  tmp56a = madq(u[4], C2_C4, tmp56a, M31);                // 1.288..2.635  (c7 = c2)
  tmp56b = madqs(u[7], -S1, tmp56b);                      // +-0.562  (s8 = -s1)
  tmp56b = madqs(u[6], S3, tmp56b);                       // +-0.767
  tmp56b = madqs(u[5], -S2, tmp56b);                      // +-1.205  (s7 = -s2)
  GF31 y4 = modM31(addq(tmp56a, tmp56b)), y5 = modM31(subqs(tmp56a, tmp56b));

  u[1] = y1; u[8] = y8;
  u[2] = y2; u[7] = y7;
  u[3] = y3; u[6] = y6;
  u[4] = y4; u[5] = y5;
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

#if PFA == 9 && PFA_W61 != 1102844585000305877u
#error PFA_W61 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-9 DFT, y_j = sum_k u_k w^(jk) with w = 37^((p-1)/9) (pfaRootOfUnity).
void OVERLOAD fft9(GF61 *u) {
  const Z61 W1 = 1102844585000305877u, W2 = 594418010121383343u, W4 = 569931187132395942u;
  // 3 x 3 Cooley-Tukey: radix-3 down the columns u[n1 + 3 n2], twiddles w^(n1 k2), radix-3 along the rows, then transpose.  (Paired
  // 64x64-bit dot products, as in fft7/fft11, need about 24 products per component here instead of 10.)
  fft3by(u, 0, 3, 9);
  fft3by(u, 1, 3, 9);
  fft3by(u, 2, 3, 9);
  u[4] = mul(u[4], W1);
  u[7] = mul(u[7], W2);
  u[5] = mul(u[5], W2);
  u[8] = mul(u[8], W4);
  fft3by(u, 0, 1, 9);
  fft3by(u, 3, 1, 9);
  fft3by(u, 6, 1, 9);
  SWAP(u[1], u[3]);
  SWAP(u[2], u[6]);
  SWAP(u[5], u[7]);
}

#endif
