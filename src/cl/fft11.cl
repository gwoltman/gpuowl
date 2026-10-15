// Copyright (C) Mihai Preda & George Woltman

#if FFT_FP64

// 102 FMA + 30 ADD.  See prime95's gwnum/zr11.mac file for the method: pair inputs k and 11-k, then each output pair j, 11-j is
// (u0 + sum_k cos(j*k*tau/11) sum_k) +- sum_k sin(j*k*tau/11) dif_k, the sin sum scaled by one sine.  Two refinements over zr11:
//  - every cos row holds the same five cosines, so with cbase = u0 + cos(3*tau/11) * sum the coefficients become cos - cos(3*tau/11) and
//    one term per row vanishes (4 fewer complex FMAs).  cos(3*tau/11) is the cosine nearest the middle of the range: it keeps every new
//    coefficient below 1 in magnitude, and a double-precision study found the round-off of the other choices 6-41% higher.
//  - the sin rows are scaled by 1/sin(4*tau/11) instead of 1/sin(tau/11): one term per row is again a plain add, and sin(4*tau/11),
//    the most accurately represented of the five sines, gave the lowest round-off in the same study.
// Overall the round-off matched the former 110 FMA version.
// Accumulation order (a double-precision study: a few percent less round-off than index order): cos terms by |coef| rank 1,0,3,2, sin terms smallest first.

void OVERLOAD fft11(T2 *u) {
  const double COS3 = -0.1423148382732851404438;   // cos(3*tau/11): subtracted from every cos coefficient, it zeroes one term per cos row
  const double C1_C3 = 0.9835683711044663093056;   // cos(tau/11) - cos(3*tau/11)
  const double C2_C3 = 0.5577298512751715659731;   // cos(2*tau/11) - cos(3*tau/11)
  const double C4_C3 = -0.5125458956719999236131;  // cos(4*tau/11) - cos(3*tau/11)
  const double C5_C3 = -0.8171781353412122494466;  // cos(5*tau/11) - cos(3*tau/11)
  const double SIN4 = 0.7557495743542582837740;    // sin(4*tau/11): factored out of the sin rows, it makes one term per row a plain add
  const double S1_S4 = 0.7153703234534297191124;   // sin(tau/11) / sin(4*tau/11)
  const double S2_S4 = 1.2036156237755650606683;   // sin(2*tau/11) / sin(4*tau/11)
  const double S3_S4 = 1.3097214678905701281139;   // sin(3*tau/11) / sin(4*tau/11)
  const double S5_S4 = 0.3727855977717922096098;   // sin(5*tau/11) / sin(4*tau/11)

  X2_mul_t4(u[1], u[10]);                               // (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[9]);
  X2_mul_t4(u[3], u[8]);
  X2_mul_t4(u[4], u[7]);
  X2_mul_t4(u[5], u[6]);

  T2 sum = u[1] + u[2] + u[3] + u[4] + u[5];
  T2 cbase = fmaT2(COS3, sum, u[0]);                    // u[0] + COS3 * sum: each cos row below then has one term fewer
  u[0] = u[0] + sum;

  T2 tmp211a = fmaT2(C2_C3, u[2], cbase);
  T2 tmp310a = fmaT2(C2_C3, u[1], cbase);
  T2 tmp49a = fmaT2(C2_C3, u[3], cbase);
  T2 tmp58a = fmaT2(C2_C3, u[5], cbase);
  T2 tmp67a = fmaT2(C2_C3, u[4], cbase);

  tmp211a = fmaT2(C4_C3, u[4], tmp211a);
  tmp310a = fmaT2(C4_C3, u[2], tmp310a);
  tmp49a = fmaT2(C4_C3, u[5], tmp49a);
  tmp58a = fmaT2(C4_C3, u[1], tmp58a);
  tmp67a = fmaT2(C4_C3, u[3], tmp67a);

  tmp211a = fmaT2(C1_C3, u[1], tmp211a);
  tmp310a = fmaT2(C1_C3, u[5], tmp310a);
  tmp49a = fmaT2(C1_C3, u[4], tmp49a);
  tmp58a = fmaT2(C1_C3, u[3], tmp58a);
  tmp67a = fmaT2(C1_C3, u[2], tmp67a);

  tmp211a = fmaT2(C5_C3, u[5], tmp211a);
  tmp310a = fmaT2(C5_C3, u[3], tmp310a);
  tmp49a = fmaT2(C5_C3, u[2], tmp49a);
  tmp58a = fmaT2(C5_C3, u[4], tmp58a);
  tmp67a = fmaT2(C5_C3, u[1], tmp67a);

  T2 tmp211b = fmaT2(S5_S4, u[6], u[7]);
  T2 tmp310b = fmaT2(S5_S4, -u[8], u[9]);
  T2 tmp49b = fmaT2(S5_S4, -u[9], u[6]);
  T2 tmp58b = fmaT2(S5_S4, u[7], u[10]);
  T2 tmp67b = fmaT2(S5_S4, u[10], u[8]);

  tmp211b = fmaT2(S1_S4, u[10], tmp211b);
  tmp310b = fmaT2(S1_S4, -u[6], tmp310b);
  tmp49b = fmaT2(S1_S4, u[7], tmp49b);
  tmp58b = fmaT2(S1_S4, u[8], tmp58b);
  tmp67b = fmaT2(S1_S4, -u[9], tmp67b);

  tmp211b = fmaT2(S2_S4, u[9], tmp211b);
  tmp310b = fmaT2(S2_S4, u[10], tmp310b);
  tmp49b = fmaT2(S2_S4, -u[8], tmp49b);
  tmp58b = fmaT2(S2_S4, -u[6], tmp58b);
  tmp67b = fmaT2(S2_S4, -u[7], tmp67b);

  tmp211b = fmaT2(S3_S4, u[8], tmp211b);
  tmp310b = fmaT2(S3_S4, -u[7], tmp310b);
  tmp49b = fmaT2(S3_S4, u[10], tmp49b);
  tmp58b = fmaT2(S3_S4, -u[9], tmp58b);
  tmp67b = fmaT2(S3_S4, u[6], tmp67b);

  fma_addsub(u[1], u[10], SIN4, tmp211a, tmp211b);
  fma_addsub(u[2], u[9], SIN4, tmp310a, tmp310b);
  fma_addsub(u[3], u[8], SIN4, tmp49a, tmp49b);
  fma_addsub(u[4], u[7], SIN4, tmp58a, tmp58b);
  fma_addsub(u[5], u[6], SIN4, tmp67a, tmp67b);
}

#endif

/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 102 FMA + 30 ADD.  See prime95's gwnum/zr11.mac file for the method: pair inputs k and 11-k, then each output pair j, 11-j is
// (u0 + sum_k cos(j*k*tau/11) sum_k) +- sum_k sin(j*k*tau/11) dif_k, the sin sum scaled by one sine.  Two refinements over zr11:
//  - every cos row holds the same five cosines, so with cbase = u0 + cos(3*tau/11) * sum the coefficients become cos - cos(3*tau/11) and
//    one term per row vanishes (4 fewer complex FMAs).  cos(3*tau/11) is the cosine nearest the middle of the range: it keeps every new
//    coefficient below 1 in magnitude, and a double-precision study found the round-off of the other choices 6-41% higher.
//  - the sin rows are scaled by 1/sin(4*tau/11) instead of 1/sin(tau/11): one term per row is again a plain add, and sin(4*tau/11),
//    the most accurately represented of the five sines, gave the lowest round-off in the same study.
// Overall the round-off matched the former 110 FMA version.
// Accumulation order (a double-precision study: a few percent less round-off than index order): cos terms by |coef| rank 1,0,3,2, sin terms smallest first.

void OVERLOAD fft11(F2 *u) {
  const float COS3 = -0.1423148382732851404438f;   // cos(3*tau/11): subtracted from every cos coefficient, it zeroes one term per cos row
  const float C1_C3 = 0.9835683711044663093056f;   // cos(tau/11) - cos(3*tau/11)
  const float C2_C3 = 0.5577298512751715659731f;   // cos(2*tau/11) - cos(3*tau/11)
  const float C4_C3 = -0.5125458956719999236131f;  // cos(4*tau/11) - cos(3*tau/11)
  const float C5_C3 = -0.8171781353412122494466f;  // cos(5*tau/11) - cos(3*tau/11)
  const float SIN4 = 0.7557495743542582837740f;    // sin(4*tau/11): factored out of the sin rows, it makes one term per row a plain add
  const float S1_S4 = 0.7153703234534297191124f;   // sin(tau/11) / sin(4*tau/11)
  const float S2_S4 = 1.2036156237755650606683f;   // sin(2*tau/11) / sin(4*tau/11)
  const float S3_S4 = 1.3097214678905701281139f;   // sin(3*tau/11) / sin(4*tau/11)
  const float S5_S4 = 0.3727855977717922096098f;   // sin(5*tau/11) / sin(4*tau/11)

  X2_mul_t4(u[1], u[10]);                               // (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[9]);
  X2_mul_t4(u[3], u[8]);
  X2_mul_t4(u[4], u[7]);
  X2_mul_t4(u[5], u[6]);

  F2 sum = u[1] + u[2] + u[3] + u[4] + u[5];
  F2 cbase = fmaT2(COS3, sum, u[0]);                    // u[0] + COS3 * sum: each cos row below then has one term fewer
  u[0] = u[0] + sum;

  F2 tmp211a = fmaT2(C2_C3, u[2], cbase);
  F2 tmp310a = fmaT2(C2_C3, u[1], cbase);
  F2 tmp49a = fmaT2(C2_C3, u[3], cbase);
  F2 tmp58a = fmaT2(C2_C3, u[5], cbase);
  F2 tmp67a = fmaT2(C2_C3, u[4], cbase);

  tmp211a = fmaT2(C4_C3, u[4], tmp211a);
  tmp310a = fmaT2(C4_C3, u[2], tmp310a);
  tmp49a = fmaT2(C4_C3, u[5], tmp49a);
  tmp58a = fmaT2(C4_C3, u[1], tmp58a);
  tmp67a = fmaT2(C4_C3, u[3], tmp67a);

  tmp211a = fmaT2(C1_C3, u[1], tmp211a);
  tmp310a = fmaT2(C1_C3, u[5], tmp310a);
  tmp49a = fmaT2(C1_C3, u[4], tmp49a);
  tmp58a = fmaT2(C1_C3, u[3], tmp58a);
  tmp67a = fmaT2(C1_C3, u[2], tmp67a);

  tmp211a = fmaT2(C5_C3, u[5], tmp211a);
  tmp310a = fmaT2(C5_C3, u[3], tmp310a);
  tmp49a = fmaT2(C5_C3, u[2], tmp49a);
  tmp58a = fmaT2(C5_C3, u[4], tmp58a);
  tmp67a = fmaT2(C5_C3, u[1], tmp67a);

  F2 tmp211b = fmaT2(S5_S4, u[6], u[7]);
  F2 tmp310b = fmaT2(S5_S4, -u[8], u[9]);
  F2 tmp49b = fmaT2(S5_S4, -u[9], u[6]);
  F2 tmp58b = fmaT2(S5_S4, u[7], u[10]);
  F2 tmp67b = fmaT2(S5_S4, u[10], u[8]);

  tmp211b = fmaT2(S1_S4, u[10], tmp211b);
  tmp310b = fmaT2(S1_S4, -u[6], tmp310b);
  tmp49b = fmaT2(S1_S4, u[7], tmp49b);
  tmp58b = fmaT2(S1_S4, u[8], tmp58b);
  tmp67b = fmaT2(S1_S4, -u[9], tmp67b);

  tmp211b = fmaT2(S2_S4, u[9], tmp211b);
  tmp310b = fmaT2(S2_S4, u[10], tmp310b);
  tmp49b = fmaT2(S2_S4, -u[8], tmp49b);
  tmp58b = fmaT2(S2_S4, -u[6], tmp58b);
  tmp67b = fmaT2(S2_S4, -u[7], tmp67b);

  tmp211b = fmaT2(S3_S4, u[8], tmp211b);
  tmp310b = fmaT2(S3_S4, -u[7], tmp310b);
  tmp49b = fmaT2(S3_S4, u[10], tmp49b);
  tmp58b = fmaT2(S3_S4, -u[9], tmp58b);
  tmp67b = fmaT2(S3_S4, u[6], tmp67b);

  fma_addsub(u[1], u[10], SIN4, tmp211a, tmp211b);
  fma_addsub(u[2], u[9], SIN4, tmp310a, tmp310b);
  fma_addsub(u[3], u[8], SIN4, tmp49a, tmp49b);
  fma_addsub(u[4], u[7], SIN4, tmp58a, tmp58b);
  fma_addsub(u[5], u[6], SIN4, tmp67a, tmp67b);
}

#endif

/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

#if PFA == 11 && PFA_W31 != 1969212174u
#error PFA_W31 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-11 DFT, y_j = sum_k u_k w^(jk) with w = 7^((p-1)/11) (pfaRootOfUnity), for the radix-PFA of a PFA middle.  The method of
// the FP version above: pair inputs k and 11-k (X2 makes the sums and differences), then output pair j, 11-j is
// (u0 + sum_k c_jk sum_k) +- sum_k s_jk dif_k, with c_m = (w^m + w^-m)/2 and s_m = (w^m - w^-m)/2 in Z/pZ.  base = u0 + c1 * (the sum of
// the sums) zeroes one term per cos row.  Each row is a GF31q (64-bit) dot product reduced once: mulq and madq keep every product below 2^61.
void OVERLOAD fft11(GF31 *u) {
  const Z31 C1 = 951289690u;
  const Z31 C2_C1 = 1482732644u;    // c2 - c1
  const Z31 C3_C1 = 336724628u;    // c3 - c1
  const Z31 C4_C1 = 1332637641u;    // c4 - c1
  const Z31 C5_C1 = 1755133048u;    // c5 - c1
  const Z31 S1 = 1017922484u;
  const Z31 S2 = 2080399339u;
  const Z31 S3 = 960443073u;
  const Z31 S4 = 143966213u;
  const Z31 S5 = 1886736629u;

  X2(u[1], u[10]);
  X2(u[2], u[9]);
  X2(u[3], u[8]);
  X2(u[4], u[7]);
  X2(u[5], u[6]);

  GF31 sall = add(add(add(add(u[1], u[2]), u[3]), u[4]), u[5]);
  GF31q cbase = madq(sall, C1, u[0]);                  // u0 + c1 * sall
  u[0] = add(u[0], sall);

  // One row at a time, each k's cos and sin products together; the outputs are kept until the last row.  The comments give each
  // row's maximum values (mulq keeps a product below M31 * M31/2), which keep addq and subq below modM31's limit of 0xFFFFFFFF7FFFFFFF.
  GF31q tmp211a = cbase;
  GF31q tmp211b = mulq(u[10], S1);
  tmp211a = madq(u[2], C2_C1, tmp211a);
  tmp211b = madq(u[9], S2, tmp211b);
  tmp211a = madq(u[3], C3_C1, tmp211a);
  tmp211b = madq(u[8], S3, tmp211b);
  tmp211a = madq(u[4], C4_C1, tmp211a);
  tmp211b = madq(u[7], S4, tmp211b);
  tmp211a = madq(u[5], C5_C1, tmp211a);
  tmp211b = madq(u[6], S5, tmp211b);
  GF31 y1 = modM31(addq(tmp211a, tmp211b)), y10 = modM31(subq(tmp211a, tmp211b));  // tmp211a < 2.94*2^61, tmp211b < 2.28*2^61

  GF31q tmp310a = cbase;
  tmp310a = madq(u[1], C2_C1, tmp310a);
  GF31q tmp310b = mulq(u[10], S2);
  tmp310a = madq(u[2], C4_C1, tmp310a);
  tmp310b = madq(u[9], S4, tmp310b);
  tmp310a = madq(u[3], C5_C1, tmp310a);
  tmp310b = madq(u[8], (M31 - S5), tmp310b);
  tmp310a = madq(u[4], C3_C1, tmp310a);
  tmp310b = madq(u[7], (M31 - S3), tmp310b);
  tmp310b = madq(u[6], (M31 - S1), tmp310b);
  GF31 y2 = modM31(addq(tmp310a, tmp310b)), y9 = modM31(subq(tmp310a, tmp310b));  // tmp310a < 2.94*2^61, tmp310b < 2.28*2^61

  GF31q tmp49a = cbase;
  tmp49a = madq(u[1], C3_C1, tmp49a);
  GF31q tmp49b = mulq(u[10], S3);
  tmp49a = madq(u[2], C5_C1, tmp49a);
  tmp49b = madq(u[9], (M31 - S5), tmp49b);
  tmp49a = madq(u[3], C2_C1, tmp49a);
  tmp49b = madq(u[8], (M31 - S2), tmp49b);
  tmp49b = madq(u[7], S1, tmp49b);
  tmp49a = madq(u[5], C4_C1, tmp49a);
  tmp49b = madq(u[6], S4, tmp49b);
  GF31 y3 = modM31(addq(tmp49a, tmp49b)), y8 = modM31(subq(tmp49a, tmp49b));  // tmp49a < 2.94*2^61, tmp49b < 2.28*2^61

  GF31q tmp58a = cbase;
  tmp58a = madq(u[1], C4_C1, tmp58a);
  GF31q tmp58b = mulq(u[10], S4);
  tmp58a = madq(u[2], C3_C1, tmp58a);
  tmp58b = madq(u[9], (M31 - S3), tmp58b);
  tmp58b = madq(u[8], S1, tmp58b);
  tmp58a = madq(u[4], C5_C1, tmp58a);
  tmp58b = madq(u[7], S5, tmp58b);
  tmp58a = madq(u[5], C2_C1, tmp58a);
  tmp58b = madq(u[6], (M31 - S2), tmp58b);
  GF31 y4 = modM31(addq(tmp58a, tmp58b)), y7 = modM31(subq(tmp58a, tmp58b));  // tmp58a < 2.94*2^61, tmp58b < 2.28*2^61

  GF31q tmp67a = cbase;
  tmp67a = madq(u[1], C5_C1, tmp67a);
  GF31q tmp67b = mulq(u[10], S5);
  tmp67b = madq(u[9], (M31 - S1), tmp67b);
  tmp67a = madq(u[3], C4_C1, tmp67a);
  tmp67b = madq(u[8], S4, tmp67b);
  tmp67a = madq(u[4], C2_C1, tmp67a);
  tmp67b = madq(u[7], (M31 - S2), tmp67b);
  tmp67a = madq(u[5], C3_C1, tmp67a);
  tmp67b = madq(u[6], S3, tmp67b);
  GF31 y5 = modM31(addq(tmp67a, tmp67b)), y6 = modM31(subq(tmp67a, tmp67b));  // tmp67a < 2.94*2^61, tmp67b < 2.28*2^61

  u[1] = y1; u[10] = y10;
  u[2] = y2; u[9] = y9;
  u[3] = y3; u[8] = y8;
  u[4] = y4; u[7] = y7;
  u[5] = y5; u[6] = y6;
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

#if PFA == 11 && PFA_W61 != 2266698407988980144ull
#error PFA_W61 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-11 DFT, y_j = sum_k u_k w^(jk) with w = 37^((p-1)/11) (pfaRootOfUnity), for the radix-PFA of a PFA middle.  The method of
// the FP version above: pair inputs k and 11-k (X2 makes the sums and differences), then output pair j, 11-j is
// (u0 + sum_k c_jk sum_k) +- sum_k s_jk dif_k, with c_m = (w^m + w^-m)/2 and s_m = (w^m - w^-m)/2 in Z/pZ.  base = u0 + c1 * (the sum of
// the sums) zeroes one term per cos row.  Each row is a 128-bit dot product reduced once.
void OVERLOAD fft11(GF61 *u) {
  const Z61 C1 = 1160353696041600296ull;
  const Z61 C2_C1 = 374271700540470474ull;    // c2 - c1
  const Z61 C3_C1 = 1401275570391910582ull;    // c3 - c1
  const Z61 C4_C1 = 1268713385799026254ull;    // c4 - c1
  const Z61 C5_C1 = 1530264404522213989ull;    // c5 - c1
  const Z61 S1 = 1106344711947379848ull;
  const Z61 S2 = 1389461816021368328ull;
  const Z61 S3 = 2075749902183963383ull;
  const Z61 S4 = 2286321371598826648ull;
  const Z61 S5 = 892586779545797283ull;

  // Quick (unreduced) arithmetic, ranges in M61s as in math.cl.  Inputs 0..1+.
  GF61 sum1 = u[1] + u[10];                    // 0..2+
  GF61 dif1 = u[1] + neg(u[10], 2);             // 1-..3+, positive for the multiplies
  GF61 sum2 = u[2] + u[9];                    // 0..2+
  GF61 dif2 = u[2] + neg(u[9], 2);             // 1-..3+, positive for the multiplies
  GF61 sum3 = u[3] + u[8];                    // 0..2+
  GF61 dif3 = u[3] + neg(u[8], 2);             // 1-..3+, positive for the multiplies
  GF61 sum4 = u[4] + u[7];                    // 0..2+
  GF61 dif4 = u[4] + neg(u[7], 2);             // 1-..3+, positive for the multiplies
  GF61 sum5 = u[5] + u[6];                    // 0..2+
  GF61 dif5 = u[5] + neg(u[6], 2);             // 1-..3+, positive for the multiplies
  GF61 sall = modM61q(sum1 + sum2 + sum3, 0);                  // 0..6+ before the mod
  sall = modM61q(sall + sum4 + sum5, 0);                      // 0..5+ before the mod
  // base = a0 + c1 * sall: each cos row then uses c_m - c1 and skips the c1 term
  u128 basex = mad64(sall.x, C1, u[0].x), basey = mad64(sall.y, C1, u[0].y);
  u[0] = modM61q(u[0] + sall, 0);                          // 0..2+ before the mod

  // One row at a time, each k's cos and sin terms together (the row's 128-bit accumulators are dead before the next row starts);
  // the outputs are kept until the last row
  u128 pc1x = basex, pc1y = basey;
  u128 qs1x = mul64(dif1.x, S1), qs1y = mul64(dif1.y, S1);
  pc1x = mad64(sum2.x, C2_C1, pc1x); pc1y = mad64(sum2.y, C2_C1, pc1y);
  qs1x = mad64(dif2.x, S2, qs1x); qs1y = mad64(dif2.y, S2, qs1y);
  pc1x = mad64(sum3.x, C3_C1, pc1x); pc1y = mad64(sum3.y, C3_C1, pc1y);
  qs1x = mad64(dif3.x, S3, qs1x); qs1y = mad64(dif3.y, S3, qs1y);
  pc1x = mad64(sum4.x, C4_C1, pc1x); pc1y = mad64(sum4.y, C4_C1, pc1y);
  qs1x = mad64(dif4.x, S4, qs1x); qs1y = mad64(dif4.y, S4, qs1y);
  pc1x = mad64(sum5.x, C5_C1, pc1x); pc1y = mad64(sum5.y, C5_C1, pc1y);
  qs1x = mad64(dif5.x, S5, qs1x); qs1y = mad64(dif5.y, S5, qs1y);
  GF61 pc1 = U2(weakModM61(pc1x, 128), weakModM61(pc1y, 128));        // 0..2+
  GF61 qs1 = U2(weakModM61(qs1x, 128), weakModM61(qs1y, 128));        // 0..2+
  GF61 y1 = modM61q(pc1 + qs1, 0), y10 = modM61q(pc1 - qs1, 3);  // 0..4+ and -2-..2+ before the mods

  u128 pc2x = basex, pc2y = basey;
  pc2x = mad64(sum1.x, C2_C1, pc2x); pc2y = mad64(sum1.y, C2_C1, pc2y);
  u128 qs2x = mul64(dif1.x, S2), qs2y = mul64(dif1.y, S2);
  pc2x = mad64(sum2.x, C4_C1, pc2x); pc2y = mad64(sum2.y, C4_C1, pc2y);
  qs2x = mad64(dif2.x, S4, qs2x); qs2y = mad64(dif2.y, S4, qs2y);
  pc2x = mad64(sum3.x, C5_C1, pc2x); pc2y = mad64(sum3.y, C5_C1, pc2y);
  qs2x = mad64(dif3.x, (M61 - S5), qs2x); qs2y = mad64(dif3.y, (M61 - S5), qs2y);
  pc2x = mad64(sum4.x, C3_C1, pc2x); pc2y = mad64(sum4.y, C3_C1, pc2y);
  qs2x = mad64(dif4.x, (M61 - S3), qs2x); qs2y = mad64(dif4.y, (M61 - S3), qs2y);
  qs2x = mad64(dif5.x, (M61 - S1), qs2x); qs2y = mad64(dif5.y, (M61 - S1), qs2y);
  GF61 pc2 = U2(weakModM61(pc2x, 128), weakModM61(pc2y, 128));        // 0..2+
  GF61 qs2 = U2(weakModM61(qs2x, 128), weakModM61(qs2y, 128));        // 0..2+
  GF61 y2 = modM61q(pc2 + qs2, 0), y9 = modM61q(pc2 - qs2, 3);  // 0..4+ and -2-..2+ before the mods

  u128 pc3x = basex, pc3y = basey;
  pc3x = mad64(sum1.x, C3_C1, pc3x); pc3y = mad64(sum1.y, C3_C1, pc3y);
  u128 qs3x = mul64(dif1.x, S3), qs3y = mul64(dif1.y, S3);
  pc3x = mad64(sum2.x, C5_C1, pc3x); pc3y = mad64(sum2.y, C5_C1, pc3y);
  qs3x = mad64(dif2.x, (M61 - S5), qs3x); qs3y = mad64(dif2.y, (M61 - S5), qs3y);
  pc3x = mad64(sum3.x, C2_C1, pc3x); pc3y = mad64(sum3.y, C2_C1, pc3y);
  qs3x = mad64(dif3.x, (M61 - S2), qs3x); qs3y = mad64(dif3.y, (M61 - S2), qs3y);
  qs3x = mad64(dif4.x, S1, qs3x); qs3y = mad64(dif4.y, S1, qs3y);
  pc3x = mad64(sum5.x, C4_C1, pc3x); pc3y = mad64(sum5.y, C4_C1, pc3y);
  qs3x = mad64(dif5.x, S4, qs3x); qs3y = mad64(dif5.y, S4, qs3y);
  GF61 pc3 = U2(weakModM61(pc3x, 128), weakModM61(pc3y, 128));        // 0..2+
  GF61 qs3 = U2(weakModM61(qs3x, 128), weakModM61(qs3y, 128));        // 0..2+
  GF61 y3 = modM61q(pc3 + qs3, 0), y8 = modM61q(pc3 - qs3, 3);  // 0..4+ and -2-..2+ before the mods

  u128 pc4x = basex, pc4y = basey;
  pc4x = mad64(sum1.x, C4_C1, pc4x); pc4y = mad64(sum1.y, C4_C1, pc4y);
  u128 qs4x = mul64(dif1.x, S4), qs4y = mul64(dif1.y, S4);
  pc4x = mad64(sum2.x, C3_C1, pc4x); pc4y = mad64(sum2.y, C3_C1, pc4y);
  qs4x = mad64(dif2.x, (M61 - S3), qs4x); qs4y = mad64(dif2.y, (M61 - S3), qs4y);
  qs4x = mad64(dif3.x, S1, qs4x); qs4y = mad64(dif3.y, S1, qs4y);
  pc4x = mad64(sum4.x, C5_C1, pc4x); pc4y = mad64(sum4.y, C5_C1, pc4y);
  qs4x = mad64(dif4.x, S5, qs4x); qs4y = mad64(dif4.y, S5, qs4y);
  pc4x = mad64(sum5.x, C2_C1, pc4x); pc4y = mad64(sum5.y, C2_C1, pc4y);
  qs4x = mad64(dif5.x, (M61 - S2), qs4x); qs4y = mad64(dif5.y, (M61 - S2), qs4y);
  GF61 pc4 = U2(weakModM61(pc4x, 128), weakModM61(pc4y, 128));        // 0..2+
  GF61 qs4 = U2(weakModM61(qs4x, 128), weakModM61(qs4y, 128));        // 0..2+
  GF61 y4 = modM61q(pc4 + qs4, 0), y7 = modM61q(pc4 - qs4, 3);  // 0..4+ and -2-..2+ before the mods

  u128 pc5x = basex, pc5y = basey;
  pc5x = mad64(sum1.x, C5_C1, pc5x); pc5y = mad64(sum1.y, C5_C1, pc5y);
  u128 qs5x = mul64(dif1.x, S5), qs5y = mul64(dif1.y, S5);
  qs5x = mad64(dif2.x, (M61 - S1), qs5x); qs5y = mad64(dif2.y, (M61 - S1), qs5y);
  pc5x = mad64(sum3.x, C4_C1, pc5x); pc5y = mad64(sum3.y, C4_C1, pc5y);
  qs5x = mad64(dif3.x, S4, qs5x); qs5y = mad64(dif3.y, S4, qs5y);
  pc5x = mad64(sum4.x, C2_C1, pc5x); pc5y = mad64(sum4.y, C2_C1, pc5y);
  qs5x = mad64(dif4.x, (M61 - S2), qs5x); qs5y = mad64(dif4.y, (M61 - S2), qs5y);
  pc5x = mad64(sum5.x, C3_C1, pc5x); pc5y = mad64(sum5.y, C3_C1, pc5y);
  qs5x = mad64(dif5.x, S3, qs5x); qs5y = mad64(dif5.y, S3, qs5y);
  GF61 pc5 = U2(weakModM61(pc5x, 128), weakModM61(pc5y, 128));        // 0..2+
  GF61 qs5 = U2(weakModM61(qs5x, 128), weakModM61(qs5y, 128));        // 0..2+
  GF61 y5 = modM61q(pc5 + qs5, 0), y6 = modM61q(pc5 - qs5, 3);  // 0..4+ and -2-..2+ before the mods

  u[1] = y1; u[10] = y10;
  u[2] = y2; u[9] = y9;
  u[3] = y3; u[8] = y8;
  u[4] = y4; u[7] = y7;
  u[5] = y5; u[6] = y6;
}

#endif
