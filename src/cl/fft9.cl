// Copyright (C) Mihai Preda and George Woltman

#if FFT_FP64

// 52 FMA + 28 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 9-k, then each output pair j, 9-j is
// (u0 + sum_k cos(j*k*tau/9) sum_k) +- sum_k sin(j*k*tau/9) dif_k, with w = e^(2*pi*i/9).  9 = 3*3 adds structure: rows 1, 2, 4 all hold
// cos(3*tau/9) = -1/2 on sum_3, which base = u0 - sum_3/2 absorbs, and their remaining cosines are a permutation of cos(tau/9),
// cos(2*tau/9), cos(4*tau/9), so base += cos(2*tau/9) * (sum_1 + sum_2 + sum_4) zeroes one more term per row (cos(2*tau/9) is the one
// nearest the middle of the range: the smallest coefficients).  Row 3 is u0 + sum_3 - (sum_1 + sum_2 + sum_4)/2 +- sqrt(3)/2 (dif_1 -
// dif_2 + dif_4).  A double-precision study put this version's round-off 18% below the former Nussbaumer one (12 FMA + 8 MUL + 72 ADD),
// with the sin rows 1, 2, 4 scaled by sin(4*tau/9), the best of the four choices.
// Accumulation order (a double-precision study: a few percent less round-off than index order): cos terms smallest first, sin terms smallest first.

void fft9(T2 *u) {
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

void fft9(F2 *u) {
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
