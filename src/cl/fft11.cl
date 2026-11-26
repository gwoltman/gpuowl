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
// the FP version above: pair inputs k and 11-k (X2qs makes the sums and differences), then output pair j, 11-j is
// (u0 + sum_k c_jk sum_k) +- sum_k s_jk dif_k, with c_m = (w^m + w^-m)/2 and s_m = (w^m - w^-m)/2 in Z/pZ.  base = u0 + c4 * (the sum of
// the sums) zeroes one term per cos row (no roundoff in GF, so the base is chosen for the bounds).  Each row is a GF31q (64-bit) dot product
// reduced once.
//
// The differences are signed (X2qs:  u_k - u_(11-k) as an i32, -M31..M31, one instruction, no fold), multiplied by the balanced sine constants
// (|s| <= M31/2, the sign of each row's term folded into the constant) with a signed multiply (mulqs / madqs).  Each row's sine sum b is then
// a signed value of magnitude at most sum_m |s_m| * M31.  K, a multiple of M31 at least that plus the cos terms' negative part, is added once
// to the base, so y_j = a + b and y_(11-j) = a - b need no per-row offset.  K = M31 * 2^32 has a zero low word, so adding it costs nothing;
// for it to fit, c3 - c4 is used as a negative constant (balanced) with a signed multiply (madqs) of its reduced sum.  Each row's first
// partial sums go through opaque(), which keeps NVVM from adding cbase to each row as a separate 64-bit add (see opaque in math.cl).
//
// Bounds (each rounded outward):  inputs, and every add result, are 0..M31;  the differences -M31..M31.  The 64-bit values are given in M31^2
// (about 2^62):  a (the base and cos terms, K included) is 1.536..2.710, |b| at most 1.141, so a + b <= 3.851, below modM31(u64)'s limit
// 0xFFFFFFFF7FFFFFFF (4.000 M31^2), and a - b >= 0.395.  The unsigned cos products pass negAbove = M31, so no constant is folded.
void OVERLOAD fft11(GF31 *u) {
  const Z31 C4 = 136443684u;        // c4 = (w^4 + w^-4) / 2                 0.064 M31
  const Z31 C1_C4 = 814846006u;     // c1 - c4                               0.380 M31
  const Z31 C2_C4 = 150095003u;     // c2 - c4                               0.070 M31
  const i32 C3_C4 = -995913013;     // c3 - c4, balanced                    -0.464 M31
  const Z31 C5_C4 = 422495407u;     // c5 - c4                               0.197 M31
  const i32 S1 = 1017922484;        // s1 = (w - w^-1) / 2, balanced         0.474 M31
  const i32 S2 = -67084308;         // s2                                   -0.032 M31
  const i32 S3 = 960443073;         // s3                                    0.448 M31
  const i32 S4 = 143966213;         // s4                                    0.068 M31
  const i32 S5 = -260747018;        // s5                                   -0.122 M31
  const u64 K = (u64) M31 << 32;    // 2.0 M31^2, low word 0 (u0 + K needs no add):  more than every row's |b| plus 0.464 (1.605)

  X2qs(&u[1], &u[10]);                                 // u[1] = 0..M31, u[10] = -M31..M31
  X2qs(&u[2], &u[9]);                                  // u[2] = 0..M31, u[9] = -M31..M31
  X2qs(&u[3], &u[8]);                                  // u[3] = 0..M31, u[8] = -M31..M31
  X2qs(&u[4], &u[7]);                                  // u[4] = 0..M31, u[7] = -M31..M31
  X2qs(&u[5], &u[6]);                                  // u[5] = 0..M31, u[6] = -M31..M31

  GF31 sum = add(add(add(add(u[1], u[2]), u[3]), u[4]), u[5]);    // 0..M31
  GF31q u0K = makeGF31q(u[0].x + K, u[0].y + K);
  GF31q cbase = madq(sum, C4, opaque(u0K), M31);          // K + u0 + c4 * sum:  2.000..2.064 M31^2
  u[0] = add(u[0], sum);                                  // y0:  0..M31

  // One row at a time; the outputs are kept until the last row.  The comments give each partial sum's bound in M31^2.
  GF31q tmp211a = opaque(madq(u[1], C1_C4, cbase, M31));  // 2.000..2.443
  GF31q tmp211b = opaque(mulqs(u[10], S1));               // +-0.475
  tmp211a = madq(u[2], C2_C4, tmp211a, M31);              // 2.000..2.513
  tmp211b = madqs(u[9], S2, tmp211b);                     // +-0.506
  tmp211a = madqs(u[3], C3_C4, tmp211a);                  // 1.536..2.513
  tmp211b = madqs(u[8], S3, tmp211b);                     // +-0.953
  tmp211a = madq(u[5], C5_C4, tmp211a, M31);              // 1.536..2.710
  tmp211b = madqs(u[7], S4, tmp211b);                     // +-1.020
  tmp211b = madqs(u[6], S5, tmp211b);                     // +-1.141
  GF31 y1 = modM31(addq(tmp211a, tmp211b)), y10 = modM31(subqs(tmp211a, tmp211b));

  GF31q tmp310a = opaque(madq(u[1], C2_C4, cbase, M31));  // 2.000..2.134
  GF31q tmp310b = opaque(mulqs(u[10], S2));               // +-0.032
  tmp310a = madq(u[3], C5_C4, tmp310a, M31);              // 2.000..2.331  (c6 = c5)
  tmp310b = madqs(u[9], S4, tmp310b);                     // +-0.099
  tmp310a = madqs(u[4], C3_C4, tmp310a);                  // 1.536..2.331  (c8 = c3)
  tmp310b = madqs(u[8], -S5, tmp310b);                    // +-0.220  (s6 = -s5)
  tmp310a = madq(u[5], C1_C4, tmp310a, M31);              // 1.536..2.710  (c10 = c1)
  tmp310b = madqs(u[7], -S3, tmp310b);                    // +-0.667  (s8 = -s3)
  tmp310b = madqs(u[6], -S1, tmp310b);                    // +-1.141  (s10 = -s1)
  GF31 y2 = modM31(addq(tmp310a, tmp310b)), y9 = modM31(subqs(tmp310a, tmp310b));

  GF31q tmp49a = opaque(madqs(u[1], C3_C4, cbase));       // 1.536..2.064
  GF31q tmp49b = opaque(mulqs(u[10], S3));                // +-0.448
  tmp49a = madq(u[2], C5_C4, tmp49a, M31);                // 1.536..2.261  (c6 = c5)
  tmp49b = madqs(u[9], -S5, tmp49b);                      // +-0.569  (s6 = -s5)
  tmp49a = madq(u[3], C2_C4, tmp49a, M31);                // 1.536..2.331  (c9 = c2)
  tmp49b = madqs(u[8], -S2, tmp49b);                      // +-0.600  (s9 = -s2)
  tmp49a = madq(u[4], C1_C4, tmp49a, M31);                // 1.536..2.710
  tmp49b = madqs(u[7], S1, tmp49b);                       // +-1.074
  tmp49b = madqs(u[6], S4, tmp49b);                       // +-1.141
  GF31 y3 = modM31(addq(tmp49a, tmp49b)), y8 = modM31(subqs(tmp49a, tmp49b));

  GF31q tmp58a = opaque(madqs(u[2], C3_C4, cbase));       // 1.536..2.064  (c8 = c3)
  GF31q tmp58b = opaque(mulqs(u[10], S4));                // +-0.068
  tmp58a = madq(u[3], C1_C4, tmp58a, M31);                // 1.536..2.443
  tmp58b = madqs(u[9], -S3, tmp58b);                      // +-0.515  (s8 = -s3)
  tmp58a = madq(u[4], C5_C4, tmp58a, M31);                // 1.536..2.640
  tmp58b = madqs(u[8], S1, tmp58b);                       // +-0.989
  tmp58a = madq(u[5], C2_C4, tmp58a, M31);                // 1.536..2.710  (c9 = c2)
  tmp58b = madqs(u[7], S5, tmp58b);                       // +-1.110
  tmp58b = madqs(u[6], -S2, tmp58b);                      // +-1.141  (s9 = -s2)
  GF31 y4 = modM31(addq(tmp58a, tmp58b)), y7 = modM31(subqs(tmp58a, tmp58b));

  GF31q tmp67a = opaque(madq(u[1], C5_C4, cbase, M31));   // 2.000..2.261
  GF31q tmp67b = opaque(mulqs(u[10], S5));                // +-0.122
  tmp67a = madq(u[2], C1_C4, tmp67a, M31);                // 2.000..2.640  (c10 = c1)
  tmp67b = madqs(u[9], -S1, tmp67b);                      // +-0.596  (s10 = -s1)
  tmp67a = madq(u[4], C2_C4, tmp67a, M31);                // 2.000..2.710  (c9 = c2)
  tmp67b = madqs(u[8], S4, tmp67b);                       // +-0.663
  tmp67a = madqs(u[5], C3_C4, tmp67a);                    // 1.536..2.710
  tmp67b = madqs(u[7], -S2, tmp67b);                      // +-0.694  (s9 = -s2)
  tmp67b = madqs(u[6], S3, tmp67b);                       // +-1.141
  GF31 y5 = modM31(addq(tmp67a, tmp67b)), y6 = modM31(subqs(tmp67a, tmp67b));

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

#if PFA == 11 && PFA_W61 != 2266698407988980144u
#error PFA_W61 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-11 DFT, y_j = sum_k u_k w^(jk) with w = 37^((p-1)/11) (pfaRootOfUnity), for the radix-PFA of a PFA middle.  The method of
// the FP version above: pair inputs k and 11-k (X2 makes the sums and differences), then output pair j, 11-j is
// (u0 + sum_k c_jk sum_k) +- sum_k s_jk dif_k, with c_m = (w^m + w^-m)/2 and s_m = (w^m - w^-m)/2 in Z/pZ.  base = u0 + c4 * (the sum of
// the sums) zeroes one term per cos row, and sbase = s5 * (the sum of the differences) one term per sin row (no roundoff in GF, so the
// bases are chosen for the bounds).  Each row is a 128-bit dot product reduced once.
void OVERLOAD fft11(GF61 *u) {
  const Z61 C4 = 123224072626932599u;
  const Z61 C1_C4 = 1037129623414667697u;           // c1 - c4
  const Z61 C2_C4 = 1411401323955138171u;           // c2 - c4
  const Z61 C3_C4 = 132562184592884328u;            // c3 - c4
  const Z61 C5_C4 = 261551018723187735u;            // c5 - c4
  const Z61 S5 = 892586779545797283u;
  const Z61 S1_S5 = 213757932401582565u;            // s1 - s5
  const Z61 S2_S5 = 496875036475571045u;            // s2 - s5
  const Z61 S3_S5 = 1183163122638166100u;           // s3 - s5
  const Z61 S4_S5 = 1393734592053029365u;           // s4 - s5
  const Z61 NS1_S5 = 306911517720516820u;           // -s1 - s5
  const Z61 NS2_S5 = 23794413646528340u;            // -s2 - s5
  const Z61 NS3_S5 = 1643349336697627236u;          // -s3 - s5
  const Z61 NS4_S5 = 1432777867282763971u;          // -s4 - s5

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
  GF61q sbase = mulq(dsum, S5, false);           // s5 * dsum, unscaled as the sin rows (see Q61X8)

  // One row at a time, each k's cos and sin products together; the outputs are kept until the last row.  The sin rows start from
  // sbase = s5 * dsum:  each row's s5 term cancels (a row whose s5 term is negative computes its negated sines and swaps its outputs),
  // saving one multiply per row.  Each product of a quick value (sums below 2 M61, differences below 3 M61, dsum below 1+) and a constant
  // gives the n passed to weakModM61q, which keeps pc + qs and pc - qs + K M61 below 2^64 for modM61q.  The sin rows use the 128-bit
  // reduction (n >= 8), so they are never scaled (see Q61X8).
  GF61q tmp211a = cbase;
  GF61q tmp211b = sbase;
  tmp211a = madq(u[1], C1_C4, tmp211a);
  tmp211b = madq(u[10], S1_S5, tmp211b, false);
  tmp211a = madq(u[2], C2_C4, tmp211a);
  tmp211b = madq(u[9], S2_S5, tmp211b, false);
  tmp211a = madq(u[3], C3_C4, tmp211a);
  tmp211b = madq(u[8], S3_S5, tmp211b, false);
  tmp211b = madq(u[7], S4_S5, tmp211b, false);
  tmp211a = madq(u[5], C5_C4, tmp211a);
  GF61 pc1 = weakModM61q(tmp211a, 3), qs1 = weakModM61q(tmp211b, 8);   // below 4 M61 and 3 M61
  GF61 y1 = modM61q(pc1 + qs1, 0), y10 = modM61q(pc1 - qs1, 3);   // below 7 M61 before the mods

  GF61q tmp310a = cbase;
  GF61q tmp310b = sbase;
  tmp310a = madq(u[1], C2_C4, tmp310a);
  tmp310b = madq(u[10], NS2_S5, tmp310b, false);
  tmp310b = madq(u[9], NS4_S5, tmp310b, false);
  tmp310a = madq(u[3], C5_C4, tmp310a);
  tmp310a = madq(u[4], C3_C4, tmp310a);
  tmp310b = madq(u[7], S3_S5, tmp310b, false);
  tmp310a = madq(u[5], C1_C4, tmp310a);
  tmp310b = madq(u[6], S1_S5, tmp310b, false);
  GF61 pc2 = weakModM61q(tmp310a, 3), qs2 = weakModM61q(tmp310b, 8);   // below 4 M61 and 3 M61
  GF61 y9 = modM61q(pc2 + qs2, 0), y2 = modM61q(pc2 - qs2, 3);   // below 7 M61 before the mods  (sines negated)

  GF61q tmp49a = cbase;
  GF61q tmp49b = sbase;
  tmp49a = madq(u[1], C3_C4, tmp49a);
  tmp49b = madq(u[10], NS3_S5, tmp49b, false);
  tmp49a = madq(u[2], C5_C4, tmp49a);
  tmp49a = madq(u[3], C2_C4, tmp49a);
  tmp49b = madq(u[8], S2_S5, tmp49b, false);
  tmp49a = madq(u[4], C1_C4, tmp49a);
  tmp49b = madq(u[7], NS1_S5, tmp49b, false);
  tmp49b = madq(u[6], NS4_S5, tmp49b, false);
  GF61 pc3 = weakModM61q(tmp49a, 3), qs3 = weakModM61q(tmp49b, 8);   // below 4 M61 and 3 M61
  GF61 y8 = modM61q(pc3 + qs3, 0), y3 = modM61q(pc3 - qs3, 3);   // below 7 M61 before the mods  (sines negated)

  GF61q tmp58a = cbase;
  GF61q tmp58b = sbase;
  tmp58b = madq(u[10], S4_S5, tmp58b, false);
  tmp58a = madq(u[2], C3_C4, tmp58a);
  tmp58b = madq(u[9], NS3_S5, tmp58b, false);
  tmp58a = madq(u[3], C1_C4, tmp58a);
  tmp58b = madq(u[8], S1_S5, tmp58b, false);
  tmp58a = madq(u[4], C5_C4, tmp58a);
  tmp58a = madq(u[5], C2_C4, tmp58a);
  tmp58b = madq(u[6], NS2_S5, tmp58b, false);
  GF61 pc4 = weakModM61q(tmp58a, 3), qs4 = weakModM61q(tmp58b, 8);   // below 4 M61 and 3 M61
  GF61 y4 = modM61q(pc4 + qs4, 0), y7 = modM61q(pc4 - qs4, 3);   // below 7 M61 before the mods

  GF61q tmp67a = cbase;
  GF61q tmp67b = sbase;
  tmp67a = madq(u[1], C5_C4, tmp67a);
  tmp67a = madq(u[2], C1_C4, tmp67a);
  tmp67b = madq(u[9], NS1_S5, tmp67b, false);
  tmp67b = madq(u[8], S4_S5, tmp67b, false);
  tmp67a = madq(u[4], C2_C4, tmp67a);
  tmp67b = madq(u[7], NS2_S5, tmp67b, false);
  tmp67a = madq(u[5], C3_C4, tmp67a);
  tmp67b = madq(u[6], S3_S5, tmp67b, false);
  GF61 pc5 = weakModM61q(tmp67a, 3), qs5 = weakModM61q(tmp67b, 8);   // below 4 M61 and 3 M61
  GF61 y5 = modM61q(pc5 + qs5, 0), y6 = modM61q(pc5 - qs5, 3);   // below 7 M61 before the mods

  u[1] = y1; u[10] = y10;
  u[2] = y2; u[9] = y9;
  u[3] = y3; u[8] = y8;
  u[4] = y4; u[7] = y7;
  u[5] = y5; u[6] = y6;
}

#endif
