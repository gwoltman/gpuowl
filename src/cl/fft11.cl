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

void fft11(T2 *u) {
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

void fft11(F2 *u) {
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
