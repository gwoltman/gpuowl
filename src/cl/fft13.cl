// Copyright (C) Mihai Preda and George Woltman

#if FFT_FP64

// 146 FMA + 36 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 13-k, then each output pair j, 13-j is
// (u0 + sum_k cos(j*k*tau/13) sum_k) +- sum_k sin(j*k*tau/13) dif_k, with w = e^(2*pi*i/13).  As in fft11.cl, cbase = u0 + cos(3*tau/13) *
// sum zeroes one term per cos row (cos(3*tau/13) is the cosine nearest the middle of the range: the smallest coefficients), and the sin
// rows are scaled by sin(6*tau/13), the most accurately represented of the six sines.  A double-precision study put the round-off within
// 1.3% of the former 156 FMA version: the cos change alone costs 4% (its coefficients reach 1.09), the sin choice wins most of it back.
// Accumulation order (a double-precision study: a few percent less round-off than index order): cos terms by |coef| rank 1,0,3,2,4, sin terms smallest first.

void OVERLOAD fft13(T2 *u) {
  const double COS3 = 0.1205366802553230533491;    // cos(3*tau/13): subtracted from every cos coefficient, it zeroes one term per cos row
  const double C1_C3 = 0.7649193453978868425513;   // cos(tau/13) - cos(3*tau/13)
  const double C2_C3 = 0.4475280664758327491627;   // cos(2*tau/13) - cos(3*tau/13)
  const double C4_C3 = -0.4751415672978586793187;  // cos(4*tau/13) - cos(3*tau/13)
  const double C5_C3 = -0.8690474284264241519837;  // cos(5*tau/13) - cos(3*tau/13)
  const double C6_C3 = -1.0914784976813750805060;  // cos(6*tau/13) - cos(3*tau/13)
  const double SIN6 = 0.2393156642875577671488;    // sin(6*tau/13): factored out of the sin rows, it makes one term per row a plain add
  const double S1_S6 = 1.9418836348521040543140;   // sin(tau/13) / sin(6*tau/13)
  const double S2_S6 = 3.4389051311943062515832;   // sin(2*tau/13) / sin(6*tau/13)
  const double S3_S6 = 4.1481149052793775035225;   // sin(3*tau/13) / sin(6*tau/13)
  const double S4_S6 = 3.9070415447687313968244;   // sin(4*tau/13) / sin(6*tau/13)
  const double S5_S6 = 2.7709120513064197918008;   // sin(5*tau/13) / sin(6*tau/13)

  X2_mul_t4(u[1], u[12]);
  X2_mul_t4(u[2], u[11]);
  X2_mul_t4(u[3], u[10]);
  X2_mul_t4(u[4], u[9]);
  X2_mul_t4(u[5], u[8]);
  X2_mul_t4(u[6], u[7]);

  T2 sum = u[1] + u[2] + u[3] + u[4] + u[5] + u[6];
  T2 cbase = fmaT2(COS3, sum, u[0]);                    // u[0] + COS3 * sum: each cos row below then has one term fewer
  u[0] = u[0] + sum;

  T2 tmp213a = fmaT2(C4_C3, u[4], cbase);
  T2 tmp312a = fmaT2(C4_C3, u[2], cbase);
  T2 tmp411a = fmaT2(C4_C3, u[3], cbase);
  T2 tmp510a = fmaT2(C4_C3, u[1], cbase);
  T2 tmp69a = fmaT2(C4_C3, u[6], cbase);
  T2 tmp78a = fmaT2(C4_C3, u[5], cbase);

  tmp213a = fmaT2(C2_C3, u[2], tmp213a);
  tmp312a = fmaT2(C2_C3, u[1], tmp312a);
  tmp411a = fmaT2(C2_C3, u[5], tmp411a);
  tmp510a = fmaT2(C2_C3, u[6], tmp510a);
  tmp69a = fmaT2(C2_C3, u[3], tmp69a);
  tmp78a = fmaT2(C2_C3, u[4], tmp78a);

  tmp213a = fmaT2(C5_C3, u[5], tmp213a);
  tmp312a = fmaT2(C5_C3, u[4], tmp312a);
  tmp411a = fmaT2(C5_C3, u[6], tmp411a);
  tmp510a = fmaT2(C5_C3, u[2], tmp510a);
  tmp69a = fmaT2(C5_C3, u[1], tmp69a);
  tmp78a = fmaT2(C5_C3, u[3], tmp78a);

  tmp213a = fmaT2(C1_C3, u[1], tmp213a);
  tmp312a = fmaT2(C1_C3, u[6], tmp312a);
  tmp411a = fmaT2(C1_C3, u[4], tmp411a);
  tmp510a = fmaT2(C1_C3, u[3], tmp510a);
  tmp69a = fmaT2(C1_C3, u[5], tmp69a);
  tmp78a = fmaT2(C1_C3, u[2], tmp78a);

  tmp213a = fmaT2(C6_C3, u[6], tmp213a);
  tmp312a = fmaT2(C6_C3, u[3], tmp312a);
  tmp411a = fmaT2(C6_C3, u[2], tmp411a);
  tmp510a = fmaT2(C6_C3, u[5], tmp510a);
  tmp69a = fmaT2(C6_C3, u[4], tmp69a);
  tmp78a = fmaT2(C6_C3, u[1], tmp78a);

  T2 tmp213b = fmaT2(S1_S6, u[12], u[7]);
  T2 tmp312b = fmaT2(S1_S6, -u[7], u[10]);
  T2 tmp411b = fmaT2(S1_S6, -u[9], u[11]);
  T2 tmp510b = fmaT2(S1_S6, -u[10], -u[8]);
  T2 tmp69b = fmaT2(S1_S6, -u[8], -u[9]);
  T2 tmp78b = fmaT2(S1_S6, -u[11], u[12]);

  tmp213b = fmaT2(S5_S6, u[8], tmp213b);
  tmp312b = fmaT2(S5_S6, -u[9], tmp312b);
  tmp411b = fmaT2(S5_S6, u[7], tmp411b);
  tmp510b = fmaT2(S5_S6, -u[11], tmp510b);
  tmp69b = fmaT2(S5_S6, u[12], tmp69b);
  tmp78b = fmaT2(S5_S6, u[10], tmp78b);

  tmp213b = fmaT2(S2_S6, u[11], tmp213b);
  tmp312b = fmaT2(S2_S6, u[12], tmp312b);
  tmp411b = fmaT2(S2_S6, u[8], tmp411b);
  tmp510b = fmaT2(S2_S6, -u[7], tmp510b);
  tmp69b = fmaT2(S2_S6, u[10], tmp69b);
  tmp78b = fmaT2(S2_S6, -u[9], tmp78b);

  tmp213b = fmaT2(S4_S6, u[9], tmp213b);
  tmp312b = fmaT2(S4_S6, u[11], tmp312b);
  tmp411b = fmaT2(S4_S6, -u[10], tmp411b);
  tmp510b = fmaT2(S4_S6, u[12], tmp510b);
  tmp69b = fmaT2(S4_S6, u[7], tmp69b);
  tmp78b = fmaT2(S4_S6, u[8], tmp78b);

  tmp213b = fmaT2(S3_S6, u[10], tmp213b);
  tmp312b = fmaT2(S3_S6, -u[8], tmp312b);
  tmp411b = fmaT2(S3_S6, u[12], tmp411b);
  tmp510b = fmaT2(S3_S6, u[9], tmp510b);
  tmp69b = fmaT2(S3_S6, -u[11], tmp69b);
  tmp78b = fmaT2(S3_S6, -u[7], tmp78b);

  fma_addsub(u[1], u[12], SIN6, tmp213a, tmp213b);
  fma_addsub(u[2], u[11], SIN6, tmp312a, tmp312b);
  fma_addsub(u[3], u[10], SIN6, tmp411a, tmp411b);
  fma_addsub(u[4], u[9], SIN6, tmp510a, tmp510b);
  fma_addsub(u[5], u[8], SIN6, tmp69a, tmp69b);
  fma_addsub(u[6], u[7], SIN6, tmp78a, tmp78b);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 146 FMA + 36 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 13-k, then each output pair j, 13-j is
// (u0 + sum_k cos(j*k*tau/13) sum_k) +- sum_k sin(j*k*tau/13) dif_k, with w = e^(2*pi*i/13).  As in fft11.cl, cbase = u0 + cos(3*tau/13) *
// sum zeroes one term per cos row (cos(3*tau/13) is the cosine nearest the middle of the range: the smallest coefficients), and the sin
// rows are scaled by sin(6*tau/13), the most accurately represented of the six sines.  A double-precision study put the round-off within
// 1.3% of the former 156 FMA version: the cos change alone costs 4% (its coefficients reach 1.09), the sin choice wins most of it back.
// Accumulation order (a double-precision study: a few percent less round-off than index order): cos terms by |coef| rank 1,0,3,2,4, sin terms smallest first.

void OVERLOAD fft13(F2 *u) {
  const float COS3 = 0.1205366802553230533491f;    // cos(3*tau/13): subtracted from every cos coefficient, it zeroes one term per cos row
  const float C1_C3 = 0.7649193453978868425513f;   // cos(tau/13) - cos(3*tau/13)
  const float C2_C3 = 0.4475280664758327491627f;   // cos(2*tau/13) - cos(3*tau/13)
  const float C4_C3 = -0.4751415672978586793187f;  // cos(4*tau/13) - cos(3*tau/13)
  const float C5_C3 = -0.8690474284264241519837f;  // cos(5*tau/13) - cos(3*tau/13)
  const float C6_C3 = -1.0914784976813750805060f;  // cos(6*tau/13) - cos(3*tau/13)
  const float SIN6 = 0.2393156642875577671488f;    // sin(6*tau/13): factored out of the sin rows, it makes one term per row a plain add
  const float S1_S6 = 1.9418836348521040543140f;   // sin(tau/13) / sin(6*tau/13)
  const float S2_S6 = 3.4389051311943062515832f;   // sin(2*tau/13) / sin(6*tau/13)
  const float S3_S6 = 4.1481149052793775035225f;   // sin(3*tau/13) / sin(6*tau/13)
  const float S4_S6 = 3.9070415447687313968244f;   // sin(4*tau/13) / sin(6*tau/13)
  const float S5_S6 = 2.7709120513064197918008f;   // sin(5*tau/13) / sin(6*tau/13)

  X2_mul_t4(u[1], u[12]);
  X2_mul_t4(u[2], u[11]);
  X2_mul_t4(u[3], u[10]);
  X2_mul_t4(u[4], u[9]);
  X2_mul_t4(u[5], u[8]);
  X2_mul_t4(u[6], u[7]);

  F2 sum = u[1] + u[2] + u[3] + u[4] + u[5] + u[6];
  F2 cbase = fmaT2(COS3, sum, u[0]);                    // u[0] + COS3 * sum: each cos row below then has one term fewer
  u[0] = u[0] + sum;

  F2 tmp213a = fmaT2(C4_C3, u[4], cbase);
  F2 tmp312a = fmaT2(C4_C3, u[2], cbase);
  F2 tmp411a = fmaT2(C4_C3, u[3], cbase);
  F2 tmp510a = fmaT2(C4_C3, u[1], cbase);
  F2 tmp69a = fmaT2(C4_C3, u[6], cbase);
  F2 tmp78a = fmaT2(C4_C3, u[5], cbase);

  tmp213a = fmaT2(C2_C3, u[2], tmp213a);
  tmp312a = fmaT2(C2_C3, u[1], tmp312a);
  tmp411a = fmaT2(C2_C3, u[5], tmp411a);
  tmp510a = fmaT2(C2_C3, u[6], tmp510a);
  tmp69a = fmaT2(C2_C3, u[3], tmp69a);
  tmp78a = fmaT2(C2_C3, u[4], tmp78a);

  tmp213a = fmaT2(C5_C3, u[5], tmp213a);
  tmp312a = fmaT2(C5_C3, u[4], tmp312a);
  tmp411a = fmaT2(C5_C3, u[6], tmp411a);
  tmp510a = fmaT2(C5_C3, u[2], tmp510a);
  tmp69a = fmaT2(C5_C3, u[1], tmp69a);
  tmp78a = fmaT2(C5_C3, u[3], tmp78a);

  tmp213a = fmaT2(C1_C3, u[1], tmp213a);
  tmp312a = fmaT2(C1_C3, u[6], tmp312a);
  tmp411a = fmaT2(C1_C3, u[4], tmp411a);
  tmp510a = fmaT2(C1_C3, u[3], tmp510a);
  tmp69a = fmaT2(C1_C3, u[5], tmp69a);
  tmp78a = fmaT2(C1_C3, u[2], tmp78a);

  tmp213a = fmaT2(C6_C3, u[6], tmp213a);
  tmp312a = fmaT2(C6_C3, u[3], tmp312a);
  tmp411a = fmaT2(C6_C3, u[2], tmp411a);
  tmp510a = fmaT2(C6_C3, u[5], tmp510a);
  tmp69a = fmaT2(C6_C3, u[4], tmp69a);
  tmp78a = fmaT2(C6_C3, u[1], tmp78a);

  F2 tmp213b = fmaT2(S1_S6, u[12], u[7]);
  F2 tmp312b = fmaT2(S1_S6, -u[7], u[10]);
  F2 tmp411b = fmaT2(S1_S6, -u[9], u[11]);
  F2 tmp510b = fmaT2(S1_S6, -u[10], -u[8]);
  F2 tmp69b = fmaT2(S1_S6, -u[8], -u[9]);
  F2 tmp78b = fmaT2(S1_S6, -u[11], u[12]);

  tmp213b = fmaT2(S5_S6, u[8], tmp213b);
  tmp312b = fmaT2(S5_S6, -u[9], tmp312b);
  tmp411b = fmaT2(S5_S6, u[7], tmp411b);
  tmp510b = fmaT2(S5_S6, -u[11], tmp510b);
  tmp69b = fmaT2(S5_S6, u[12], tmp69b);
  tmp78b = fmaT2(S5_S6, u[10], tmp78b);

  tmp213b = fmaT2(S2_S6, u[11], tmp213b);
  tmp312b = fmaT2(S2_S6, u[12], tmp312b);
  tmp411b = fmaT2(S2_S6, u[8], tmp411b);
  tmp510b = fmaT2(S2_S6, -u[7], tmp510b);
  tmp69b = fmaT2(S2_S6, u[10], tmp69b);
  tmp78b = fmaT2(S2_S6, -u[9], tmp78b);

  tmp213b = fmaT2(S4_S6, u[9], tmp213b);
  tmp312b = fmaT2(S4_S6, u[11], tmp312b);
  tmp411b = fmaT2(S4_S6, -u[10], tmp411b);
  tmp510b = fmaT2(S4_S6, u[12], tmp510b);
  tmp69b = fmaT2(S4_S6, u[7], tmp69b);
  tmp78b = fmaT2(S4_S6, u[8], tmp78b);

  tmp213b = fmaT2(S3_S6, u[10], tmp213b);
  tmp312b = fmaT2(S3_S6, -u[8], tmp312b);
  tmp411b = fmaT2(S3_S6, u[12], tmp411b);
  tmp510b = fmaT2(S3_S6, u[9], tmp510b);
  tmp69b = fmaT2(S3_S6, -u[11], tmp69b);
  tmp78b = fmaT2(S3_S6, -u[7], tmp78b);

  fma_addsub(u[1], u[12], SIN6, tmp213a, tmp213b);
  fma_addsub(u[2], u[11], SIN6, tmp312a, tmp312b);
  fma_addsub(u[3], u[10], SIN6, tmp411a, tmp411b);
  fma_addsub(u[4], u[9], SIN6, tmp510a, tmp510b);
  fma_addsub(u[5], u[8], SIN6, tmp69a, tmp69b);
  fma_addsub(u[6], u[7], SIN6, tmp78a, tmp78b);
}

#endif
