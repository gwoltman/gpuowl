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
// (u0 + sum_k c_jk sum_k) +- sum_k s_jk dif_k, with c_m = (w^m + w^-m)/2 and s_m = (w^m - w^-m)/2 in Z/pZ.  base = u0 + c4 * (the sum of
// the sums) zeroes one term per cos row (no roundoff in GF, so the base is chosen for the bounds).  Each row is a GF31q (64-bit) dot product
// reduced once.
void OVERLOAD fft11(GF31 *u) {
  const Z31 C4 = 136443684u;
  const Z31 C1_C4 = 814846006u;    // c1 - c4
  const Z31 C2_C4 = 150095003u;    // c2 - c4
  const Z31 C3_C4 = 1151570634u;    // c3 - c4
  const Z31 C5_C4 = 422495407u;    // c5 - c4
  const Z31 S1 = 1017922484u;
  const Z31 S2 = 2080399339u;
  const Z31 S3 = 960443073u;
  const Z31 S4 = 143966213u;
  const Z31 S5 = 1886736629u;

  X2(u[1], u[10]);
  X2(u[9], u[2]); SWAP(u[2], u[9]);  // reversed difference
  X2(u[3], u[8]);
  X2(u[4], u[7]);
  X2(u[5], u[6]);

  GF31 sum = add(add(add(add(u[1], u[2]), u[3]), u[4]), u[5]);
  GF31q cbase = madq(sum, C4, u[0]);                   // u0 + c4 * sum
  u[0] = add(u[0], sum);

  // One row at a time, each k's cos and sin products together; the outputs are kept until the last row.  No product is folded
  // (negAbove = M31), so there are no M31 - x subtracts:  base c4 instead of c1, the k = 2 difference reversed (X2 then SWAP, free) and
  // row 3/10's sines negated (its outputs swapped, free) keep each row within modM31's limits.  The comments give each row's maximum
  // values; subq needs tmpNNa below about 4*2^61 and tmpNNb <= M31 * 2^32 (about 4*2^61), addq needs the sum below 8*2^61.
  GF31q tmp211a = cbase;
  tmp211a = madq(u[1], C1_C4, tmp211a);
  GF31q tmp211b = mulq(u[10], S1);
  tmp211a = madq(u[2], C2_C4, tmp211a);
  tmp211b = madq(u[9], (M31 - S2), tmp211b);
  tmp211a = madq(u[3], C3_C4, tmp211a, M31);
  tmp211b = madq(u[8], S3, tmp211b);
  tmp211b = madq(u[7], S4, tmp211b);
  tmp211a = madq(u[5], C5_C4, tmp211a);
  tmp211b = madq(u[6], S5, tmp211b, M31);
  GF31 y1 = modM31(addq(tmp211a, tmp211b)), y10 = modM31(subq(tmp211a, tmp211b));  // tmp211a < 2.49*2^61, tmp211b < 3.80*2^61

  GF31q tmp310a = cbase;
  tmp310a = madq(u[1], C2_C4, tmp310a);
  GF31q tmp310b = mulq(u[10], (M31 - S2));
  tmp310b = madq(u[9], S4, tmp310b);
  tmp310a = madq(u[3], C5_C4, tmp310a);
  tmp310b = madq(u[8], S5, tmp310b, M31);
  tmp310a = madq(u[4], C3_C4, tmp310a, M31);
  tmp310b = madq(u[7], S3, tmp310b);
  tmp310a = madq(u[5], C1_C4, tmp310a);
  tmp310b = madq(u[6], S1, tmp310b);
  GF31 y9 = modM31(addq(tmp310a, tmp310b)), y2 = modM31(subq(tmp310a, tmp310b));  // tmp310a < 2.49*2^61, tmp310b < 3.80*2^61  (sines negated)

  GF31q tmp49a = cbase;
  tmp49a = madq(u[1], C3_C4, tmp49a, M31);
  GF31q tmp49b = mulq(u[10], S3);
  tmp49a = madq(u[2], C5_C4, tmp49a);
  tmp49b = madq(u[9], S5, tmp49b, M31);
  tmp49a = madq(u[3], C2_C4, tmp49a);
  tmp49b = madq(u[8], (M31 - S2), tmp49b);
  tmp49a = madq(u[4], C1_C4, tmp49a);
  tmp49b = madq(u[7], S1, tmp49b);
  tmp49b = madq(u[6], S4, tmp49b);
  GF31 y3 = modM31(addq(tmp49a, tmp49b)), y8 = modM31(subq(tmp49a, tmp49b));  // tmp49a < 2.49*2^61, tmp49b < 3.80*2^61

  GF31q tmp58a = cbase;
  GF31q tmp58b = mulq(u[10], S4);
  tmp58a = madq(u[2], C3_C4, tmp58a, M31);
  tmp58b = madq(u[9], S3, tmp58b);
  tmp58a = madq(u[3], C1_C4, tmp58a);
  tmp58b = madq(u[8], S1, tmp58b);
  tmp58a = madq(u[4], C5_C4, tmp58a);
  tmp58b = madq(u[7], S5, tmp58b, M31);
  tmp58a = madq(u[5], C2_C4, tmp58a);
  tmp58b = madq(u[6], (M31 - S2), tmp58b);
  GF31 y4 = modM31(addq(tmp58a, tmp58b)), y7 = modM31(subq(tmp58a, tmp58b));  // tmp58a < 2.49*2^61, tmp58b < 3.80*2^61

  GF31q tmp67a = cbase;
  tmp67a = madq(u[1], C5_C4, tmp67a);
  GF31q tmp67b = mulq(u[10], S5, M31);
  tmp67a = madq(u[2], C1_C4, tmp67a);
  tmp67b = madq(u[9], S1, tmp67b);
  tmp67b = madq(u[8], S4, tmp67b);
  tmp67a = madq(u[4], C2_C4, tmp67a);
  tmp67b = madq(u[7], (M31 - S2), tmp67b);
  tmp67a = madq(u[5], C3_C4, tmp67a, M31);
  tmp67b = madq(u[6], S3, tmp67b);
  GF31 y5 = modM31(addq(tmp67a, tmp67b)), y6 = modM31(subq(tmp67a, tmp67b));  // tmp67a < 2.49*2^61, tmp67b < 3.80*2^61

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
// (u0 + sum_k c_jk sum_k) +- sum_k s_jk dif_k, with c_m = (w^m + w^-m)/2 and s_m = (w^m - w^-m)/2 in Z/pZ.  base = u0 + c4 * (the sum of
// the sums) zeroes one term per cos row, and sbase = s5 * (the sum of the differences) one term per sin row (no roundoff in GF, so the
// bases are chosen for the bounds).  Each row is a 128-bit dot product reduced once.
void OVERLOAD fft11(GF61 *u) {
  const Z61 C4 = 123224072626932599ull;
  const Z61 C1_C4 = 1037129623414667697ull;           // c1 - c4
  const Z61 C2_C4 = 1411401323955138171ull;           // c2 - c4
  const Z61 C3_C4 = 132562184592884328ull;            // c3 - c4
  const Z61 C5_C4 = 261551018723187735ull;            // c5 - c4
  const Z61 S5 = 892586779545797283ull;
  const Z61 S1_S5 = 213757932401582565ull;            // s1 - s5
  const Z61 S2_S5 = 496875036475571045ull;            // s2 - s5
  const Z61 S3_S5 = 1183163122638166100ull;           // s3 - s5
  const Z61 S4_S5 = 1393734592053029365ull;           // s4 - s5
  const Z61 NS1_S5 = 306911517720516820ull;           // -s1 - s5
  const Z61 NS2_S5 = 23794413646528340ull;            // -s2 - s5
  const Z61 NS3_S5 = 1643349336697627236ull;          // -s3 - s5
  const Z61 NS4_S5 = 1432777867282763971ull;          // -s4 - s5

  // Quick (unreduced) arithmetic, ranges in M61s as in math.cl.  Inputs 0..1+.
  X2qp(&u[1], &u[10]);   // u[1] = 0..2+, u[10] = 1-..3+
  X2qp(&u[2], &u[9]);    // u[2] = 0..2+, u[9] = 1-..3+
  X2qp(&u[3], &u[8]);    // u[3] = 0..2+, u[8] = 1-..3+
  X2qp(&u[4], &u[7]);    // u[4] = 0..2+, u[7] = 1-..3+
  X2qp(&u[5], &u[6]);    // u[5] = 0..2+, u[6] = 1-..3+

  GF61 sum = modM61q(u[1] + u[2] + u[3], 0);  // 0..6+ before the mod
  sum = modM61q(sum + u[4] + u[5], 0);        // 0..5+ before the mod
  GF61q cbase = madq(sum, C4, u[0]);          // u0 + c4 * sum
  u[0] = modM61q(u[0] + sum, 0);              // 0..2+ before the mod
  GF61 dsum = modM61q(u[10] + u[9] + u[8], -2);  // the sum of the differences:  3 (1-..3+) - 2 = 1-..7+ before the mod
  dsum = modM61q(dsum + u[7] + u[6], 0);         // 1-..7+ before the mod
  GF61q sbase = mulq(dsum, S5);                  // s5 * dsum

  // One row at a time, each k's cos and sin products together; the outputs are kept until the last row.  The sin rows start from
  // sbase = s5 * dsum:  each row's s5 term cancels (a row whose s5 term is negative computes its negated sines and swaps its outputs),
  // saving one multiply per row.  Each product of a quick value (sums below 2 M61, differences below 3 M61, dsum below 1+) and a constant
  // gives the n passed to weakModM61q, which keeps pc + qs and pc - qs + K M61 below 2^64 for modM61q.  The sin rows use the 128-bit
  // reduction (n >= 8).
  GF61q tmp211a = cbase;
  GF61q tmp211b = sbase;
  tmp211a = madq(u[1], C1_C4, tmp211a);
  tmp211b = madq(u[10], S1_S5, tmp211b);
  tmp211a = madq(u[2], C2_C4, tmp211a);
  tmp211b = madq(u[9], S2_S5, tmp211b);
  tmp211a = madq(u[3], C3_C4, tmp211a);
  tmp211b = madq(u[8], S3_S5, tmp211b);
  tmp211b = madq(u[7], S4_S5, tmp211b);
  tmp211a = madq(u[5], C5_C4, tmp211a);
  GF61 pc1 = weakModM61q(tmp211a, 3), qs1 = weakModM61q(tmp211b, 8);   // below 4 M61 and 3 M61
  GF61 y1 = modM61q(pc1 + qs1, 0), y10 = modM61q(pc1 - qs1, 3);   // below 7 M61 before the mods

  GF61q tmp310a = cbase;
  GF61q tmp310b = sbase;
  tmp310a = madq(u[1], C2_C4, tmp310a);
  tmp310b = madq(u[10], NS2_S5, tmp310b);
  tmp310b = madq(u[9], NS4_S5, tmp310b);
  tmp310a = madq(u[3], C5_C4, tmp310a);
  tmp310a = madq(u[4], C3_C4, tmp310a);
  tmp310b = madq(u[7], S3_S5, tmp310b);
  tmp310a = madq(u[5], C1_C4, tmp310a);
  tmp310b = madq(u[6], S1_S5, tmp310b);
  GF61 pc2 = weakModM61q(tmp310a, 3), qs2 = weakModM61q(tmp310b, 8);   // below 4 M61 and 3 M61
  GF61 y9 = modM61q(pc2 + qs2, 0), y2 = modM61q(pc2 - qs2, 3);   // below 7 M61 before the mods  (sines negated)

  GF61q tmp49a = cbase;
  GF61q tmp49b = sbase;
  tmp49a = madq(u[1], C3_C4, tmp49a);
  tmp49b = madq(u[10], NS3_S5, tmp49b);
  tmp49a = madq(u[2], C5_C4, tmp49a);
  tmp49a = madq(u[3], C2_C4, tmp49a);
  tmp49b = madq(u[8], S2_S5, tmp49b);
  tmp49a = madq(u[4], C1_C4, tmp49a);
  tmp49b = madq(u[7], NS1_S5, tmp49b);
  tmp49b = madq(u[6], NS4_S5, tmp49b);
  GF61 pc3 = weakModM61q(tmp49a, 3), qs3 = weakModM61q(tmp49b, 8);   // below 4 M61 and 3 M61
  GF61 y8 = modM61q(pc3 + qs3, 0), y3 = modM61q(pc3 - qs3, 3);   // below 7 M61 before the mods  (sines negated)

  GF61q tmp58a = cbase;
  GF61q tmp58b = sbase;
  tmp58b = madq(u[10], S4_S5, tmp58b);
  tmp58a = madq(u[2], C3_C4, tmp58a);
  tmp58b = madq(u[9], NS3_S5, tmp58b);
  tmp58a = madq(u[3], C1_C4, tmp58a);
  tmp58b = madq(u[8], S1_S5, tmp58b);
  tmp58a = madq(u[4], C5_C4, tmp58a);
  tmp58a = madq(u[5], C2_C4, tmp58a);
  tmp58b = madq(u[6], NS2_S5, tmp58b);
  GF61 pc4 = weakModM61q(tmp58a, 3), qs4 = weakModM61q(tmp58b, 8);   // below 4 M61 and 3 M61
  GF61 y4 = modM61q(pc4 + qs4, 0), y7 = modM61q(pc4 - qs4, 3);   // below 7 M61 before the mods

  GF61q tmp67a = cbase;
  GF61q tmp67b = sbase;
  tmp67a = madq(u[1], C5_C4, tmp67a);
  tmp67a = madq(u[2], C1_C4, tmp67a);
  tmp67b = madq(u[9], NS1_S5, tmp67b);
  tmp67b = madq(u[8], S4_S5, tmp67b);
  tmp67a = madq(u[4], C2_C4, tmp67a);
  tmp67b = madq(u[7], NS2_S5, tmp67b);
  tmp67a = madq(u[5], C3_C4, tmp67a);
  tmp67b = madq(u[6], S3_S5, tmp67b);
  GF61 pc5 = weakModM61q(tmp67a, 3), qs5 = weakModM61q(tmp67b, 8);   // below 4 M61 and 3 M61
  GF61 y5 = modM61q(pc5 + qs5, 0), y6 = modM61q(pc5 - qs5, 3);   // below 7 M61 before the mods

  u[1] = y1; u[10] = y10;
  u[2] = y2; u[9] = y9;
  u[3] = y3; u[8] = y8;
  u[4] = y4; u[7] = y7;
  u[5] = y5; u[6] = y6;
}

#endif
