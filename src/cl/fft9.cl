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
// once), with the structure 9 = 3 * 3 adds as in the FP fft9:  rows 1, 2, 4 share base = u0 + c3 sum3 + c1 (sum1 + sum2 + sum4).
void OVERLOAD fft9(GF31 *u) {
  const Z31 C1 = 586798891u;
  const Z31 C3 = 1073741823u;                          // c3 = -1/2
  const Z31 C2_C1 = 1428287284u;    // c2 - c1
  const Z31 C4_C1 = 1106283337u;    // c4 - c1
  const Z31 S1 = 1869791976u;
  const Z31 S2 = 942092970u;
  const Z31 S3 = 439735912u;
  const Z31 S4 = 1219784641u;

  X2(u[1], u[8]);
  X2(u[2], u[7]);
  X2(u[3], u[6]);
  X2(u[4], u[5]);

  GF31 s124 = add(add(u[1], u[2]), u[4]);
  GF31 t = add(u[0], u[3]);
  GF31q cbase = madq(s124, C1, madq(u[3], C3, u[0]));   // rows 1, 2, 4:  u0 + c3 sum3 + c1 (sum1 + sum2 + sum4)
  u[0] = add(t, s124);

  // One row at a time, each k's cos and sin products together; the outputs are kept until the last row.  The comments give each
  // row's maximum values (mulq keeps a product below M31 * M31/2), which keep addq and subq below modM31's limit of 0xFFFFFFFF7FFFFFFF.
  GF31q tmp29a = cbase;
  GF31q tmp29b = mulq(u[8], S1);
  tmp29a = madq(u[2], C2_C1, tmp29a);
  tmp29b = madq(u[7], S2, tmp29b);
  tmp29b = madq(u[6], S3, tmp29b);
  tmp29a = madq(u[4], C4_C1, tmp29a);
  tmp29b = madq(u[5], S4, tmp29b);
  GF31 y1 = modM31(addq(tmp29a, tmp29b)), y8 = modM31(subq(tmp29a, tmp29b));  // tmp29a < 3.19*2^61, tmp29b < 2.41*2^61

  GF31q tmp38a = cbase;
  tmp38a = madq(u[1], C2_C1, tmp38a);
  GF31q tmp38b = mulq(u[8], S2);
  tmp38a = madq(u[2], C4_C1, tmp38a);
  tmp38b = madq(u[7], S4, tmp38b);
  tmp38b = madq(u[6], (M31 - S3), tmp38b);
  tmp38b = madq(u[5], (M31 - S1), tmp38b);
  GF31 y2 = modM31(addq(tmp38a, tmp38b)), y7 = modM31(subq(tmp38a, tmp38b));  // tmp38a < 3.19*2^61, tmp38b < 2.41*2^61

  GF31q tmp47a = madq(s124, C3, t);                 // row 3:  u0 + sum3 + c3 (sum1 + sum2 + sum4)
  GF31q tmp47b = mulq(u[8], S3);
  tmp47b = madq(u[7], (M31 - S3), tmp47b);
  tmp47b = madq(u[5], S3, tmp47b);
  GF31 y3 = modM31(addq(tmp47a, tmp47b)), y6 = modM31(subq(tmp47a, tmp47b));  // tmp47a < 1.00*2^61, tmp47b < 1.23*2^61

  GF31q tmp56a = cbase;
  tmp56a = madq(u[1], C4_C1, tmp56a);
  GF31q tmp56b = mulq(u[8], S4);
  tmp56b = madq(u[7], (M31 - S1), tmp56b);
  tmp56b = madq(u[6], S3, tmp56b);
  tmp56a = madq(u[4], C2_C1, tmp56a);
  tmp56b = madq(u[5], (M31 - S2), tmp56b);
  GF31 y4 = modM31(addq(tmp56a, tmp56b)), y5 = modM31(subq(tmp56a, tmp56b));  // tmp56a < 3.19*2^61, tmp56b < 2.41*2^61

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

#if PFA == 9 && PFA_W61 != 1102844585000305877ull
#error PFA_W61 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-9 DFT, y_j = sum_k u_k w^(jk) with w = 37^((p-1)/9) (pfaRootOfUnity).
void OVERLOAD fft9(GF61 *u) {
  const Z61 W1 = 1102844585000305877ull, W2 = 594418010121383343ull, W4 = 569931187132395942ull;
  // 3 x 3 Cooley-Tukey: radix-3 down the columns u[n1 + 3 n2], twiddles w^(n1 k2), radix-3 along the rows, then transpose.  (Paired
  // 64x64-bit dot products, as in fft7/fft11, need about 24 products per component here instead of 10.)
  fft3by(u, 0, 3, 9);
  fft3by(u, 1, 3, 9);
  fft3by(u, 2, 3, 9);
  u[4] = pfaScale(u[4], W1);
  u[7] = pfaScale(u[7], W2);
  u[5] = pfaScale(u[5], W2);
  u[8] = pfaScale(u[8], W4);
  fft3by(u, 0, 1, 9);
  fft3by(u, 3, 1, 9);
  fft3by(u, 6, 1, 9);
  SWAP(u[1], u[3]);
  SWAP(u[2], u[6]);
  SWAP(u[5], u[7]);
}

#endif
