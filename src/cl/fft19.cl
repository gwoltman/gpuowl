// Copyright (C) Mihai Preda and George Woltman

#if FFT_FP64

// 326 FMA + 54 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 19-k, then each
// output pair j, 19-j is (u0 + sum_k cos(j*k*tau/19) sum_k) +- sum_k sin(j*k*tau/19) dif_k, with w = e^(2*pi*i/19).  As in fft11.cl, cbase = u0 +
// cos(5*tau/19) * sum zeroes one term per cos row (cos(5*tau/19) = -0.083 is the cosine nearest the middle of the range: every coefficient
// stays below 1.03 in magnitude), and the sin rows are scaled by sin(2*tau/19), the most accurately represented of the 9 sines
// (relative rounding error 9.4e-19).
// Accumulation order as fft13.cl: cos terms by |coef| rank 1,0,3,2,5,4,7,6 (ranks from the smallest), sin terms smallest first.

void OVERLOAD fft19(T2 *u) {
  const double COS5 = -0.08257934547233232460034; // cos(5*tau/19): subtracted from every cos coefficient, it zeroes one term per cos row
  const double C1_C5 = 1.028396587172967003620;   // cos(1*tau/19) - cos(5*tau/19)
  const double C2_C5 = 0.8717198548687259238193;  // cos(2*tau/19) - cos(5*tau/19)
  const double C3_C5 = 0.6295275035947591993121;  // cos(3*tau/19) - cos(5*tau/19)
  const double C4_C5 = 0.3280648326131314735226;  // cos(4*tau/19) - cos(5*tau/19)
  const double C6_C5 = -0.3191160791806371329165; // cos(6*tau/19) - cos(5*tau/19)
  const double C7_C5 = -0.5947022261534087501618; // cos(7*tau/19) - cos(5*tau/19)
  const double C8_C5 = -0.7968944057341567467905; // cos(8*tau/19) - cos(5*tau/19)
  const double C9_C5 = -0.9037819579303900490022; // cos(9*tau/19) - cos(5*tau/19)
  const double SIN2 = 0.6142127126896678174443;   // sin(2*tau/19): factored out of the sin rows, it makes one term per row a plain add
  const double S1_S2 = 0.5286433551380082450379;  // sin(1*tau/19) / sin(2*tau/19)
  const double S3_S2 = 1.362991128263261113001;   // sin(3*tau/19) / sin(2*tau/19)
  const double S4_S2 = 1.578281018792787198438;   // sin(4*tau/19) / sin(2*tau/19)
  const double S5_S2 = 1.622539671382861994461;   // sin(5*tau/19) / sin(2*tau/19)
  const double S6_S2 = 1.490970974281598297845;   // sin(6*tau/19) / sin(2*tau/19)
  const double S7_S2 = 1.197832437318596463801;   // sin(7*tau/19) / sin(2*tau/19)
  const double S8_S2 = 0.7748901694868482834043;  // sin(8*tau/19) / sin(2*tau/19)
  const double S9_S2 = 0.2679765281313798449371;  // sin(9*tau/19) / sin(2*tau/19)

  X2_mul_t4(u[1], u[18]);
  X2_mul_t4(u[2], u[17]);
  X2_mul_t4(u[3], u[16]);
  X2_mul_t4(u[4], u[15]);
  X2_mul_t4(u[5], u[14]);
  X2_mul_t4(u[6], u[13]);
  X2_mul_t4(u[7], u[12]);
  X2_mul_t4(u[8], u[11]);
  X2_mul_t4(u[9], u[10]);

  T2 sum = u[1] + u[2] + u[3] + u[4] + u[5] + u[6] + u[7] + u[8] + u[9];
  T2 cbase = fmaT2(COS5, sum, u[0]);                    // u[0] + COS5 * sum: each cos row below then has one term fewer
  u[0] = u[0] + sum;

  T2 tmp118a = fmaT2(C4_C5, u[4], cbase);
  T2 tmp217a = fmaT2(C4_C5, u[2], cbase);
  T2 tmp316a = fmaT2(C4_C5, u[5], cbase);
  T2 tmp415a = fmaT2(C4_C5, u[1], cbase);
  T2 tmp514a = fmaT2(C4_C5, u[3], cbase);
  T2 tmp613a = fmaT2(C4_C5, u[7], cbase);
  T2 tmp712a = fmaT2(C4_C5, u[6], cbase);
  T2 tmp811a = fmaT2(C4_C5, u[9], cbase);
  T2 tmp910a = fmaT2(C4_C5, u[8], cbase);

  tmp118a = fmaT2(C6_C5, u[6], tmp118a);
  tmp217a = fmaT2(C6_C5, u[3], tmp217a);
  tmp316a = fmaT2(C6_C5, u[2], tmp316a);
  tmp415a = fmaT2(C6_C5, u[8], tmp415a);
  tmp514a = fmaT2(C6_C5, u[5], tmp514a);
  tmp613a = fmaT2(C6_C5, u[1], tmp613a);
  tmp712a = fmaT2(C6_C5, u[9], tmp712a);
  tmp811a = fmaT2(C6_C5, u[4], tmp811a);
  tmp910a = fmaT2(C6_C5, u[7], tmp910a);

  tmp118a = fmaT2(C3_C5, u[3], tmp118a);
  tmp217a = fmaT2(C3_C5, u[8], tmp217a);
  tmp316a = fmaT2(C3_C5, u[1], tmp316a);
  tmp415a = fmaT2(C3_C5, u[4], tmp415a);
  tmp514a = fmaT2(C3_C5, u[7], tmp514a);
  tmp613a = fmaT2(C3_C5, u[9], tmp613a);
  tmp712a = fmaT2(C3_C5, u[5], tmp712a);
  tmp811a = fmaT2(C3_C5, u[2], tmp811a);
  tmp910a = fmaT2(C3_C5, u[6], tmp910a);

  tmp118a = fmaT2(C7_C5, u[7], tmp118a);
  tmp217a = fmaT2(C7_C5, u[6], tmp217a);
  tmp316a = fmaT2(C7_C5, u[4], tmp316a);
  tmp415a = fmaT2(C7_C5, u[3], tmp415a);
  tmp514a = fmaT2(C7_C5, u[9], tmp514a);
  tmp613a = fmaT2(C7_C5, u[2], tmp613a);
  tmp712a = fmaT2(C7_C5, u[1], tmp712a);
  tmp811a = fmaT2(C7_C5, u[8], tmp811a);
  tmp910a = fmaT2(C7_C5, u[5], tmp910a);

  tmp118a = fmaT2(C2_C5, u[2], tmp118a);
  tmp217a = fmaT2(C2_C5, u[1], tmp217a);
  tmp316a = fmaT2(C2_C5, u[7], tmp316a);
  tmp415a = fmaT2(C2_C5, u[9], tmp415a);
  tmp514a = fmaT2(C2_C5, u[8], tmp514a);
  tmp613a = fmaT2(C2_C5, u[6], tmp613a);
  tmp712a = fmaT2(C2_C5, u[3], tmp712a);
  tmp811a = fmaT2(C2_C5, u[5], tmp811a);
  tmp910a = fmaT2(C2_C5, u[4], tmp910a);

  tmp118a = fmaT2(C8_C5, u[8], tmp118a);
  tmp217a = fmaT2(C8_C5, u[4], tmp217a);
  tmp316a = fmaT2(C8_C5, u[9], tmp316a);
  tmp415a = fmaT2(C8_C5, u[2], tmp415a);
  tmp514a = fmaT2(C8_C5, u[6], tmp514a);
  tmp613a = fmaT2(C8_C5, u[5], tmp613a);
  tmp712a = fmaT2(C8_C5, u[7], tmp712a);
  tmp811a = fmaT2(C8_C5, u[1], tmp811a);
  tmp910a = fmaT2(C8_C5, u[3], tmp910a);

  tmp118a = fmaT2(C1_C5, u[1], tmp118a);
  tmp217a = fmaT2(C1_C5, u[9], tmp217a);
  tmp316a = fmaT2(C1_C5, u[6], tmp316a);
  tmp415a = fmaT2(C1_C5, u[5], tmp415a);
  tmp514a = fmaT2(C1_C5, u[4], tmp514a);
  tmp613a = fmaT2(C1_C5, u[3], tmp613a);
  tmp712a = fmaT2(C1_C5, u[8], tmp712a);
  tmp811a = fmaT2(C1_C5, u[7], tmp811a);
  tmp910a = fmaT2(C1_C5, u[2], tmp910a);

  tmp118a = fmaT2(C9_C5, u[9], tmp118a);
  tmp217a = fmaT2(C9_C5, u[5], tmp217a);
  tmp316a = fmaT2(C9_C5, u[3], tmp316a);
  tmp415a = fmaT2(C9_C5, u[7], tmp415a);
  tmp514a = fmaT2(C9_C5, u[2], tmp514a);
  tmp613a = fmaT2(C9_C5, u[8], tmp613a);
  tmp712a = fmaT2(C9_C5, u[4], tmp712a);
  tmp811a = fmaT2(C9_C5, u[6], tmp811a);
  tmp910a = fmaT2(C9_C5, u[1], tmp910a);

  T2 tmp118b = fmaT2(S9_S2, u[10], u[17]);
  T2 tmp217b = fmaT2(S9_S2, -u[14], u[18]);
  T2 tmp316b = fmaT2(S9_S2, u[16], u[12]);
  T2 tmp415b = fmaT2(S9_S2, u[12], -u[10]);
  T2 tmp514b = fmaT2(S9_S2, -u[17], u[11]);
  T2 tmp613b = fmaT2(S9_S2, -u[11], -u[13]);
  T2 tmp712b = fmaT2(S9_S2, u[15], u[16]);
  T2 tmp811b = fmaT2(S9_S2, -u[13], u[14]);
  T2 tmp910b = fmaT2(S9_S2, u[18], -u[15]);

  tmp118b = fmaT2(S1_S2, u[18], tmp118b);
  tmp217b = fmaT2(S1_S2, -u[10], tmp217b);
  tmp316b = fmaT2(S1_S2, -u[13], tmp316b);
  tmp415b = fmaT2(S1_S2, u[14], tmp415b);
  tmp514b = fmaT2(S1_S2, u[15], tmp514b);
  tmp613b = fmaT2(S1_S2, -u[16], tmp613b);
  tmp712b = fmaT2(S1_S2, -u[11], tmp712b);
  tmp811b = fmaT2(S1_S2, -u[12], tmp811b);
  tmp910b = fmaT2(S1_S2, -u[17], tmp910b);

  tmp118b = fmaT2(S8_S2, u[11], tmp118b);
  tmp217b = fmaT2(S8_S2, u[15], tmp217b);
  tmp316b = fmaT2(S8_S2, u[10], tmp316b);
  tmp415b = fmaT2(S8_S2, u[17], tmp415b);
  tmp514b = fmaT2(S8_S2, -u[13], tmp514b);
  tmp613b = fmaT2(S8_S2, -u[14], tmp613b);
  tmp712b = fmaT2(S8_S2, -u[12], tmp712b);
  tmp811b = fmaT2(S8_S2, u[18], tmp811b);
  tmp910b = fmaT2(S8_S2, u[16], tmp910b);

  tmp118b = fmaT2(S7_S2, u[12], tmp118b);
  tmp217b = fmaT2(S7_S2, -u[13], tmp217b);
  tmp316b = fmaT2(S7_S2, -u[15], tmp316b);
  tmp415b = fmaT2(S7_S2, -u[16], tmp415b);
  tmp514b = fmaT2(S7_S2, u[10], tmp514b);
  tmp613b = fmaT2(S7_S2, -u[17], tmp613b);
  tmp712b = fmaT2(S7_S2, u[18], tmp712b);
  tmp811b = fmaT2(S7_S2, u[11], tmp811b);
  tmp910b = fmaT2(S7_S2, u[14], tmp910b);

  tmp118b = fmaT2(S3_S2, u[16], tmp118b);
  tmp217b = fmaT2(S3_S2, -u[11], tmp217b);
  tmp316b = fmaT2(S3_S2, u[18], tmp316b);
  tmp415b = fmaT2(S3_S2, -u[15], tmp415b);
  tmp514b = fmaT2(S3_S2, -u[12], tmp514b);
  tmp613b = fmaT2(S3_S2, -u[10], tmp613b);
  tmp712b = fmaT2(S3_S2, -u[14], tmp712b);
  tmp811b = fmaT2(S3_S2, -u[17], tmp811b);
  tmp910b = fmaT2(S3_S2, -u[13], tmp910b);

  tmp118b = fmaT2(S6_S2, u[13], tmp118b);
  tmp217b = fmaT2(S6_S2, u[16], tmp217b);
  tmp316b = fmaT2(S6_S2, u[17], tmp316b);
  tmp415b = fmaT2(S6_S2, -u[11], tmp415b);
  tmp514b = fmaT2(S6_S2, u[14], tmp514b);
  tmp613b = fmaT2(S6_S2, u[18], tmp613b);
  tmp712b = fmaT2(S6_S2, u[10], tmp712b);
  tmp811b = fmaT2(S6_S2, -u[15], tmp811b);
  tmp910b = fmaT2(S6_S2, u[12], tmp910b);

  tmp118b = fmaT2(S4_S2, u[15], tmp118b);
  tmp217b = fmaT2(S4_S2, u[17], tmp217b);
  tmp316b = fmaT2(S4_S2, -u[14], tmp316b);
  tmp415b = fmaT2(S4_S2, u[18], tmp415b);
  tmp514b = fmaT2(S4_S2, -u[16], tmp514b);
  tmp613b = fmaT2(S4_S2, u[12], tmp613b);
  tmp712b = fmaT2(S4_S2, u[13], tmp712b);
  tmp811b = fmaT2(S4_S2, -u[10], tmp811b);
  tmp910b = fmaT2(S4_S2, -u[11], tmp910b);

  tmp118b = fmaT2(S5_S2, u[14], tmp118b);
  tmp217b = fmaT2(S5_S2, -u[12], tmp217b);
  tmp316b = fmaT2(S5_S2, u[11], tmp316b);
  tmp415b = fmaT2(S5_S2, u[13], tmp415b);
  tmp514b = fmaT2(S5_S2, u[18], tmp514b);
  tmp613b = fmaT2(S5_S2, u[15], tmp613b);
  tmp712b = fmaT2(S5_S2, -u[17], tmp712b);
  tmp811b = fmaT2(S5_S2, u[16], tmp811b);
  tmp910b = fmaT2(S5_S2, u[10], tmp910b);

  fma_addsub(u[1], u[18], SIN2, tmp118a, tmp118b);
  fma_addsub(u[2], u[17], SIN2, tmp217a, tmp217b);
  fma_addsub(u[3], u[16], SIN2, tmp316a, tmp316b);
  fma_addsub(u[4], u[15], SIN2, tmp415a, tmp415b);
  fma_addsub(u[5], u[14], SIN2, tmp514a, tmp514b);
  fma_addsub(u[6], u[13], SIN2, tmp613a, tmp613b);
  fma_addsub(u[7], u[12], SIN2, tmp712a, tmp712b);
  fma_addsub(u[8], u[11], SIN2, tmp811a, tmp811b);
  fma_addsub(u[9], u[10], SIN2, tmp910a, tmp910b);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 326 FMA + 54 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 19-k, then each
// output pair j, 19-j is (u0 + sum_k cos(j*k*tau/19) sum_k) +- sum_k sin(j*k*tau/19) dif_k, with w = e^(2*pi*i/19).  As in fft11.cl, cbase = u0 +
// cos(5*tau/19) * sum zeroes one term per cos row (cos(5*tau/19) = -0.083 is the cosine nearest the middle of the range: every coefficient
// stays below 1.03 in magnitude), and the sin rows are scaled by sin(2*tau/19), the most accurately represented of the 9 sines
// (relative rounding error 9.4e-19).
// Accumulation order as fft13.cl: cos terms by |coef| rank 1,0,3,2,5,4,7,6 (ranks from the smallest), sin terms smallest first.

void OVERLOAD fft19(F2 *u) {
  const float COS5 = -0.08257934547233232460034f; // cos(5*tau/19): subtracted from every cos coefficient, it zeroes one term per cos row
  const float C1_C5 = 1.028396587172967003620f;   // cos(1*tau/19) - cos(5*tau/19)
  const float C2_C5 = 0.8717198548687259238193f;  // cos(2*tau/19) - cos(5*tau/19)
  const float C3_C5 = 0.6295275035947591993121f;  // cos(3*tau/19) - cos(5*tau/19)
  const float C4_C5 = 0.3280648326131314735226f;  // cos(4*tau/19) - cos(5*tau/19)
  const float C6_C5 = -0.3191160791806371329165f; // cos(6*tau/19) - cos(5*tau/19)
  const float C7_C5 = -0.5947022261534087501618f; // cos(7*tau/19) - cos(5*tau/19)
  const float C8_C5 = -0.7968944057341567467905f; // cos(8*tau/19) - cos(5*tau/19)
  const float C9_C5 = -0.9037819579303900490022f; // cos(9*tau/19) - cos(5*tau/19)
  const float SIN2 = 0.6142127126896678174443f;   // sin(2*tau/19): factored out of the sin rows, it makes one term per row a plain add
  const float S1_S2 = 0.5286433551380082450379f;  // sin(1*tau/19) / sin(2*tau/19)
  const float S3_S2 = 1.362991128263261113001f;   // sin(3*tau/19) / sin(2*tau/19)
  const float S4_S2 = 1.578281018792787198438f;   // sin(4*tau/19) / sin(2*tau/19)
  const float S5_S2 = 1.622539671382861994461f;   // sin(5*tau/19) / sin(2*tau/19)
  const float S6_S2 = 1.490970974281598297845f;   // sin(6*tau/19) / sin(2*tau/19)
  const float S7_S2 = 1.197832437318596463801f;   // sin(7*tau/19) / sin(2*tau/19)
  const float S8_S2 = 0.7748901694868482834043f;  // sin(8*tau/19) / sin(2*tau/19)
  const float S9_S2 = 0.2679765281313798449371f;  // sin(9*tau/19) / sin(2*tau/19)

  X2_mul_t4(u[1], u[18]);
  X2_mul_t4(u[2], u[17]);
  X2_mul_t4(u[3], u[16]);
  X2_mul_t4(u[4], u[15]);
  X2_mul_t4(u[5], u[14]);
  X2_mul_t4(u[6], u[13]);
  X2_mul_t4(u[7], u[12]);
  X2_mul_t4(u[8], u[11]);
  X2_mul_t4(u[9], u[10]);

  F2 sum = u[1] + u[2] + u[3] + u[4] + u[5] + u[6] + u[7] + u[8] + u[9];
  F2 cbase = fmaT2(COS5, sum, u[0]);                    // u[0] + COS5 * sum: each cos row below then has one term fewer
  u[0] = u[0] + sum;

  F2 tmp118a = fmaT2(C4_C5, u[4], cbase);
  F2 tmp217a = fmaT2(C4_C5, u[2], cbase);
  F2 tmp316a = fmaT2(C4_C5, u[5], cbase);
  F2 tmp415a = fmaT2(C4_C5, u[1], cbase);
  F2 tmp514a = fmaT2(C4_C5, u[3], cbase);
  F2 tmp613a = fmaT2(C4_C5, u[7], cbase);
  F2 tmp712a = fmaT2(C4_C5, u[6], cbase);
  F2 tmp811a = fmaT2(C4_C5, u[9], cbase);
  F2 tmp910a = fmaT2(C4_C5, u[8], cbase);

  tmp118a = fmaT2(C6_C5, u[6], tmp118a);
  tmp217a = fmaT2(C6_C5, u[3], tmp217a);
  tmp316a = fmaT2(C6_C5, u[2], tmp316a);
  tmp415a = fmaT2(C6_C5, u[8], tmp415a);
  tmp514a = fmaT2(C6_C5, u[5], tmp514a);
  tmp613a = fmaT2(C6_C5, u[1], tmp613a);
  tmp712a = fmaT2(C6_C5, u[9], tmp712a);
  tmp811a = fmaT2(C6_C5, u[4], tmp811a);
  tmp910a = fmaT2(C6_C5, u[7], tmp910a);

  tmp118a = fmaT2(C3_C5, u[3], tmp118a);
  tmp217a = fmaT2(C3_C5, u[8], tmp217a);
  tmp316a = fmaT2(C3_C5, u[1], tmp316a);
  tmp415a = fmaT2(C3_C5, u[4], tmp415a);
  tmp514a = fmaT2(C3_C5, u[7], tmp514a);
  tmp613a = fmaT2(C3_C5, u[9], tmp613a);
  tmp712a = fmaT2(C3_C5, u[5], tmp712a);
  tmp811a = fmaT2(C3_C5, u[2], tmp811a);
  tmp910a = fmaT2(C3_C5, u[6], tmp910a);

  tmp118a = fmaT2(C7_C5, u[7], tmp118a);
  tmp217a = fmaT2(C7_C5, u[6], tmp217a);
  tmp316a = fmaT2(C7_C5, u[4], tmp316a);
  tmp415a = fmaT2(C7_C5, u[3], tmp415a);
  tmp514a = fmaT2(C7_C5, u[9], tmp514a);
  tmp613a = fmaT2(C7_C5, u[2], tmp613a);
  tmp712a = fmaT2(C7_C5, u[1], tmp712a);
  tmp811a = fmaT2(C7_C5, u[8], tmp811a);
  tmp910a = fmaT2(C7_C5, u[5], tmp910a);

  tmp118a = fmaT2(C2_C5, u[2], tmp118a);
  tmp217a = fmaT2(C2_C5, u[1], tmp217a);
  tmp316a = fmaT2(C2_C5, u[7], tmp316a);
  tmp415a = fmaT2(C2_C5, u[9], tmp415a);
  tmp514a = fmaT2(C2_C5, u[8], tmp514a);
  tmp613a = fmaT2(C2_C5, u[6], tmp613a);
  tmp712a = fmaT2(C2_C5, u[3], tmp712a);
  tmp811a = fmaT2(C2_C5, u[5], tmp811a);
  tmp910a = fmaT2(C2_C5, u[4], tmp910a);

  tmp118a = fmaT2(C8_C5, u[8], tmp118a);
  tmp217a = fmaT2(C8_C5, u[4], tmp217a);
  tmp316a = fmaT2(C8_C5, u[9], tmp316a);
  tmp415a = fmaT2(C8_C5, u[2], tmp415a);
  tmp514a = fmaT2(C8_C5, u[6], tmp514a);
  tmp613a = fmaT2(C8_C5, u[5], tmp613a);
  tmp712a = fmaT2(C8_C5, u[7], tmp712a);
  tmp811a = fmaT2(C8_C5, u[1], tmp811a);
  tmp910a = fmaT2(C8_C5, u[3], tmp910a);

  tmp118a = fmaT2(C1_C5, u[1], tmp118a);
  tmp217a = fmaT2(C1_C5, u[9], tmp217a);
  tmp316a = fmaT2(C1_C5, u[6], tmp316a);
  tmp415a = fmaT2(C1_C5, u[5], tmp415a);
  tmp514a = fmaT2(C1_C5, u[4], tmp514a);
  tmp613a = fmaT2(C1_C5, u[3], tmp613a);
  tmp712a = fmaT2(C1_C5, u[8], tmp712a);
  tmp811a = fmaT2(C1_C5, u[7], tmp811a);
  tmp910a = fmaT2(C1_C5, u[2], tmp910a);

  tmp118a = fmaT2(C9_C5, u[9], tmp118a);
  tmp217a = fmaT2(C9_C5, u[5], tmp217a);
  tmp316a = fmaT2(C9_C5, u[3], tmp316a);
  tmp415a = fmaT2(C9_C5, u[7], tmp415a);
  tmp514a = fmaT2(C9_C5, u[2], tmp514a);
  tmp613a = fmaT2(C9_C5, u[8], tmp613a);
  tmp712a = fmaT2(C9_C5, u[4], tmp712a);
  tmp811a = fmaT2(C9_C5, u[6], tmp811a);
  tmp910a = fmaT2(C9_C5, u[1], tmp910a);

  F2 tmp118b = fmaT2(S9_S2, u[10], u[17]);
  F2 tmp217b = fmaT2(S9_S2, -u[14], u[18]);
  F2 tmp316b = fmaT2(S9_S2, u[16], u[12]);
  F2 tmp415b = fmaT2(S9_S2, u[12], -u[10]);
  F2 tmp514b = fmaT2(S9_S2, -u[17], u[11]);
  F2 tmp613b = fmaT2(S9_S2, -u[11], -u[13]);
  F2 tmp712b = fmaT2(S9_S2, u[15], u[16]);
  F2 tmp811b = fmaT2(S9_S2, -u[13], u[14]);
  F2 tmp910b = fmaT2(S9_S2, u[18], -u[15]);

  tmp118b = fmaT2(S1_S2, u[18], tmp118b);
  tmp217b = fmaT2(S1_S2, -u[10], tmp217b);
  tmp316b = fmaT2(S1_S2, -u[13], tmp316b);
  tmp415b = fmaT2(S1_S2, u[14], tmp415b);
  tmp514b = fmaT2(S1_S2, u[15], tmp514b);
  tmp613b = fmaT2(S1_S2, -u[16], tmp613b);
  tmp712b = fmaT2(S1_S2, -u[11], tmp712b);
  tmp811b = fmaT2(S1_S2, -u[12], tmp811b);
  tmp910b = fmaT2(S1_S2, -u[17], tmp910b);

  tmp118b = fmaT2(S8_S2, u[11], tmp118b);
  tmp217b = fmaT2(S8_S2, u[15], tmp217b);
  tmp316b = fmaT2(S8_S2, u[10], tmp316b);
  tmp415b = fmaT2(S8_S2, u[17], tmp415b);
  tmp514b = fmaT2(S8_S2, -u[13], tmp514b);
  tmp613b = fmaT2(S8_S2, -u[14], tmp613b);
  tmp712b = fmaT2(S8_S2, -u[12], tmp712b);
  tmp811b = fmaT2(S8_S2, u[18], tmp811b);
  tmp910b = fmaT2(S8_S2, u[16], tmp910b);

  tmp118b = fmaT2(S7_S2, u[12], tmp118b);
  tmp217b = fmaT2(S7_S2, -u[13], tmp217b);
  tmp316b = fmaT2(S7_S2, -u[15], tmp316b);
  tmp415b = fmaT2(S7_S2, -u[16], tmp415b);
  tmp514b = fmaT2(S7_S2, u[10], tmp514b);
  tmp613b = fmaT2(S7_S2, -u[17], tmp613b);
  tmp712b = fmaT2(S7_S2, u[18], tmp712b);
  tmp811b = fmaT2(S7_S2, u[11], tmp811b);
  tmp910b = fmaT2(S7_S2, u[14], tmp910b);

  tmp118b = fmaT2(S3_S2, u[16], tmp118b);
  tmp217b = fmaT2(S3_S2, -u[11], tmp217b);
  tmp316b = fmaT2(S3_S2, u[18], tmp316b);
  tmp415b = fmaT2(S3_S2, -u[15], tmp415b);
  tmp514b = fmaT2(S3_S2, -u[12], tmp514b);
  tmp613b = fmaT2(S3_S2, -u[10], tmp613b);
  tmp712b = fmaT2(S3_S2, -u[14], tmp712b);
  tmp811b = fmaT2(S3_S2, -u[17], tmp811b);
  tmp910b = fmaT2(S3_S2, -u[13], tmp910b);

  tmp118b = fmaT2(S6_S2, u[13], tmp118b);
  tmp217b = fmaT2(S6_S2, u[16], tmp217b);
  tmp316b = fmaT2(S6_S2, u[17], tmp316b);
  tmp415b = fmaT2(S6_S2, -u[11], tmp415b);
  tmp514b = fmaT2(S6_S2, u[14], tmp514b);
  tmp613b = fmaT2(S6_S2, u[18], tmp613b);
  tmp712b = fmaT2(S6_S2, u[10], tmp712b);
  tmp811b = fmaT2(S6_S2, -u[15], tmp811b);
  tmp910b = fmaT2(S6_S2, u[12], tmp910b);

  tmp118b = fmaT2(S4_S2, u[15], tmp118b);
  tmp217b = fmaT2(S4_S2, u[17], tmp217b);
  tmp316b = fmaT2(S4_S2, -u[14], tmp316b);
  tmp415b = fmaT2(S4_S2, u[18], tmp415b);
  tmp514b = fmaT2(S4_S2, -u[16], tmp514b);
  tmp613b = fmaT2(S4_S2, u[12], tmp613b);
  tmp712b = fmaT2(S4_S2, u[13], tmp712b);
  tmp811b = fmaT2(S4_S2, -u[10], tmp811b);
  tmp910b = fmaT2(S4_S2, -u[11], tmp910b);

  tmp118b = fmaT2(S5_S2, u[14], tmp118b);
  tmp217b = fmaT2(S5_S2, -u[12], tmp217b);
  tmp316b = fmaT2(S5_S2, u[11], tmp316b);
  tmp415b = fmaT2(S5_S2, u[13], tmp415b);
  tmp514b = fmaT2(S5_S2, u[18], tmp514b);
  tmp613b = fmaT2(S5_S2, u[15], tmp613b);
  tmp712b = fmaT2(S5_S2, -u[17], tmp712b);
  tmp811b = fmaT2(S5_S2, u[16], tmp811b);
  tmp910b = fmaT2(S5_S2, u[10], tmp910b);

  fma_addsub(u[1], u[18], SIN2, tmp118a, tmp118b);
  fma_addsub(u[2], u[17], SIN2, tmp217a, tmp217b);
  fma_addsub(u[3], u[16], SIN2, tmp316a, tmp316b);
  fma_addsub(u[4], u[15], SIN2, tmp415a, tmp415b);
  fma_addsub(u[5], u[14], SIN2, tmp514a, tmp514b);
  fma_addsub(u[6], u[13], SIN2, tmp613a, tmp613b);
  fma_addsub(u[7], u[12], SIN2, tmp712a, tmp712b);
  fma_addsub(u[8], u[11], SIN2, tmp811a, tmp811b);
  fma_addsub(u[9], u[10], SIN2, tmp910a, tmp910b);
}

#endif
