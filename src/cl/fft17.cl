// Copyright (C) Mihai Preda and George Woltman

#if FFT_FP64

// 258 FMA + 48 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 17-k, then each
// output pair j, 17-j is (u0 + sum_k cos(j*k*tau/17) sum_k) +- sum_k sin(j*k*tau/17) dif_k, with w = e^(2*pi*i/17).  As in fft11.cl, cbase = u0 +
// cos(4*tau/17) * sum zeroes one term per cos row (cos(4*tau/17) = 0.092 is the cosine nearest the middle of the range: every coefficient
// stays below 1.08 in magnitude), and the sin rows are scaled by sin(2*tau/17), the most accurately represented of the eight sines
// (relative rounding error 6.5e-18).
// Accumulation order as fft13.cl: cos terms by |coef| rank 1,0,3,2,5,4,6 (ranks from the smallest), sin terms smallest first.

void OVERLOAD fft17(T2 *u) {
  const double COS4 = 0.09226835946330199523965;  // cos(4*tau/17): subtracted from every cos coefficient, it zeroes one term per cos row
  const double C1_C4 = 0.8402038699410538093335;  // cos(1*tau/17) - cos(4*tau/17)
  const double C2_C4 = 0.6467405577573571206849;  // cos(2*tau/17) - cos(4*tau/17)
  const double C3_C4 = 0.3534699963132362721568;  // cos(3*tau/17) - cos(4*tau/17)
  const double C5_C4 = -0.3659313495353848587787; // cos(5*tau/17) - cos(4*tau/17)
  const double C6_C4 = -0.6949029958425583844182; // cos(6*tau/17) - cos(4*tau/17)
  const double C7_C4 = -0.9424854951929161473738; // cos(7*tau/17) - cos(4*tau/17)
  const double C8_C4 = -1.075241459147203773522;  // cos(8*tau/17) - cos(4*tau/17)
  const double SIN2 = 0.6736956436465572117127;   // sin(2*tau/17): factored out of the sin rows, it makes one term per row a plain add
  const double S1_S2 = 0.5362089982233463159835;  // sin(1*tau/17) / sin(2*tau/17)
  const double S3_S2 = 1.328735460585365293163;   // sin(3*tau/17) / sin(2*tau/17)
  const double S4_S2 = 1.478017834441318231849;   // sin(4*tau/17) / sin(2*tau/17)
  const double S5_S2 = 1.427685709776422850776;   // sin(5*tau/17) / sin(2*tau/17)
  const double S6_S2 = 1.184536718926603990479;   // sin(6*tau/17) / sin(2*tau/17)
  const double S7_S2 = 0.7814094804411995660846;  // sin(7*tau/17) / sin(2*tau/17)
  const double S8_S2 = 0.2727485616828054534919;  // sin(8*tau/17) / sin(2*tau/17)

  X2_mul_t4(u[1], u[16]);
  X2_mul_t4(u[2], u[15]);
  X2_mul_t4(u[3], u[14]);
  X2_mul_t4(u[4], u[13]);
  X2_mul_t4(u[5], u[12]);
  X2_mul_t4(u[6], u[11]);
  X2_mul_t4(u[7], u[10]);
  X2_mul_t4(u[8], u[9]);

  T2 sum = u[1] + u[2] + u[3] + u[4] + u[5] + u[6] + u[7] + u[8];
  T2 cbase = fmaT2(COS4, sum, u[0]);                    // u[0] + COS4 * sum: each cos row below then has one term fewer
  u[0] = u[0] + sum;

  T2 tmp116a = fmaT2(C5_C4, u[5], cbase);
  T2 tmp215a = fmaT2(C5_C4, u[6], cbase);
  T2 tmp314a = fmaT2(C5_C4, u[4], cbase);
  T2 tmp413a = fmaT2(C5_C4, u[3], cbase);
  T2 tmp512a = fmaT2(C5_C4, u[1], cbase);
  T2 tmp611a = fmaT2(C5_C4, u[2], cbase);
  T2 tmp710a = fmaT2(C5_C4, u[8], cbase);
  T2 tmp89a = fmaT2(C5_C4, u[7], cbase);

  tmp116a = fmaT2(C3_C4, u[3], tmp116a);
  tmp215a = fmaT2(C3_C4, u[7], tmp215a);
  tmp314a = fmaT2(C3_C4, u[1], tmp314a);
  tmp413a = fmaT2(C3_C4, u[5], tmp413a);
  tmp512a = fmaT2(C3_C4, u[4], tmp512a);
  tmp611a = fmaT2(C3_C4, u[8], tmp611a);
  tmp710a = fmaT2(C3_C4, u[2], tmp710a);
  tmp89a = fmaT2(C3_C4, u[6], tmp89a);

  tmp116a = fmaT2(C6_C4, u[6], tmp116a);
  tmp215a = fmaT2(C6_C4, u[3], tmp215a);
  tmp314a = fmaT2(C6_C4, u[2], tmp314a);
  tmp413a = fmaT2(C6_C4, u[7], tmp413a);
  tmp512a = fmaT2(C6_C4, u[8], tmp512a);
  tmp611a = fmaT2(C6_C4, u[1], tmp611a);
  tmp710a = fmaT2(C6_C4, u[4], tmp710a);
  tmp89a = fmaT2(C6_C4, u[5], tmp89a);

  tmp116a = fmaT2(C2_C4, u[2], tmp116a);
  tmp215a = fmaT2(C2_C4, u[1], tmp215a);
  tmp314a = fmaT2(C2_C4, u[5], tmp314a);
  tmp413a = fmaT2(C2_C4, u[8], tmp413a);
  tmp512a = fmaT2(C2_C4, u[3], tmp512a);
  tmp611a = fmaT2(C2_C4, u[6], tmp611a);
  tmp710a = fmaT2(C2_C4, u[7], tmp710a);
  tmp89a = fmaT2(C2_C4, u[4], tmp89a);

  tmp116a = fmaT2(C7_C4, u[7], tmp116a);
  tmp215a = fmaT2(C7_C4, u[5], tmp215a);
  tmp314a = fmaT2(C7_C4, u[8], tmp314a);
  tmp413a = fmaT2(C7_C4, u[6], tmp413a);
  tmp512a = fmaT2(C7_C4, u[2], tmp512a);
  tmp611a = fmaT2(C7_C4, u[4], tmp611a);
  tmp710a = fmaT2(C7_C4, u[1], tmp710a);
  tmp89a = fmaT2(C7_C4, u[3], tmp89a);

  tmp116a = fmaT2(C1_C4, u[1], tmp116a);
  tmp215a = fmaT2(C1_C4, u[8], tmp215a);
  tmp314a = fmaT2(C1_C4, u[6], tmp314a);
  tmp413a = fmaT2(C1_C4, u[4], tmp413a);
  tmp512a = fmaT2(C1_C4, u[7], tmp512a);
  tmp611a = fmaT2(C1_C4, u[3], tmp611a);
  tmp710a = fmaT2(C1_C4, u[5], tmp710a);
  tmp89a = fmaT2(C1_C4, u[2], tmp89a);

  tmp116a = fmaT2(C8_C4, u[8], tmp116a);
  tmp215a = fmaT2(C8_C4, u[4], tmp215a);
  tmp314a = fmaT2(C8_C4, u[3], tmp314a);
  tmp413a = fmaT2(C8_C4, u[2], tmp413a);
  tmp512a = fmaT2(C8_C4, u[5], tmp512a);
  tmp611a = fmaT2(C8_C4, u[7], tmp611a);
  tmp710a = fmaT2(C8_C4, u[6], tmp710a);
  tmp89a = fmaT2(C8_C4, u[1], tmp89a);

  T2 tmp116b = fmaT2(S8_S2, u[9], u[15]);
  T2 tmp215b = fmaT2(S8_S2, u[13], u[16]);
  T2 tmp314b = fmaT2(S8_S2, -u[14], -u[12]);
  T2 tmp413b = fmaT2(S8_S2, u[15], -u[9]);
  T2 tmp512b = fmaT2(S8_S2, u[12], -u[14]);
  T2 tmp611b = fmaT2(S8_S2, u[10], u[11]);
  T2 tmp710b = fmaT2(S8_S2, u[11], -u[10]);
  T2 tmp89b = fmaT2(S8_S2, u[16], -u[13]);

  tmp116b = fmaT2(S1_S2, u[16], tmp116b);
  tmp215b = fmaT2(S1_S2, -u[9], tmp215b);
  tmp314b = fmaT2(S1_S2, u[11], tmp314b);
  tmp413b = fmaT2(S1_S2, -u[13], tmp413b);
  tmp512b = fmaT2(S1_S2, u[10], tmp512b);
  tmp611b = fmaT2(S1_S2, u[14], tmp611b);
  tmp710b = fmaT2(S1_S2, u[12], tmp710b);
  tmp89b = fmaT2(S1_S2, -u[15], tmp89b);

  tmp116b = fmaT2(S7_S2, u[10], tmp116b);
  tmp215b = fmaT2(S7_S2, -u[12], tmp215b);
  tmp314b = fmaT2(S7_S2, u[9], tmp314b);
  tmp413b = fmaT2(S7_S2, u[11], tmp413b);
  tmp512b = fmaT2(S7_S2, -u[15], tmp512b);
  tmp611b = fmaT2(S7_S2, u[13], tmp611b);
  tmp710b = fmaT2(S7_S2, u[16], tmp710b);
  tmp89b = fmaT2(S7_S2, u[14], tmp89b);

  tmp116b = fmaT2(S6_S2, u[11], tmp116b);
  tmp215b = fmaT2(S6_S2, u[14], tmp215b);
  tmp314b = fmaT2(S6_S2, u[15], tmp314b);
  tmp413b = fmaT2(S6_S2, -u[10], tmp413b);
  tmp512b = fmaT2(S6_S2, u[9], tmp512b);
  tmp611b = fmaT2(S6_S2, u[16], tmp611b);
  tmp710b = fmaT2(S6_S2, -u[13], tmp710b);
  tmp89b = fmaT2(S6_S2, u[12], tmp89b);

  tmp116b = fmaT2(S3_S2, u[14], tmp116b);
  tmp215b = fmaT2(S3_S2, -u[10], tmp215b);
  tmp314b = fmaT2(S3_S2, u[16], tmp314b);
  tmp413b = fmaT2(S3_S2, u[12], tmp413b);
  tmp512b = fmaT2(S3_S2, u[13], tmp512b);
  tmp611b = fmaT2(S3_S2, -u[9], tmp611b);
  tmp710b = fmaT2(S3_S2, -u[15], tmp710b);
  tmp89b = fmaT2(S3_S2, -u[11], tmp89b);

  tmp116b = fmaT2(S5_S2, u[12], tmp116b);
  tmp215b = fmaT2(S5_S2, -u[11], tmp215b);
  tmp314b = fmaT2(S5_S2, -u[13], tmp314b);
  tmp413b = fmaT2(S5_S2, -u[14], tmp413b);
  tmp512b = fmaT2(S5_S2, u[16], tmp512b);
  tmp611b = fmaT2(S5_S2, -u[15], tmp611b);
  tmp710b = fmaT2(S5_S2, u[9], tmp710b);
  tmp89b = fmaT2(S5_S2, u[10], tmp89b);

  tmp116b = fmaT2(S4_S2, u[13], tmp116b);
  tmp215b = fmaT2(S4_S2, u[15], tmp215b);
  tmp314b = fmaT2(S4_S2, u[10], tmp314b);
  tmp413b = fmaT2(S4_S2, u[16], tmp413b);
  tmp512b = fmaT2(S4_S2, -u[11], tmp512b);
  tmp611b = fmaT2(S4_S2, -u[12], tmp611b);
  tmp710b = fmaT2(S4_S2, u[14], tmp710b);
  tmp89b = fmaT2(S4_S2, -u[9], tmp89b);

  fma_addsub(u[1], u[16], SIN2, tmp116a, tmp116b);
  fma_addsub(u[2], u[15], SIN2, tmp215a, tmp215b);
  fma_addsub(u[3], u[14], SIN2, tmp314a, tmp314b);
  fma_addsub(u[4], u[13], SIN2, tmp413a, tmp413b);
  fma_addsub(u[5], u[12], SIN2, tmp512a, tmp512b);
  fma_addsub(u[6], u[11], SIN2, tmp611a, tmp611b);
  fma_addsub(u[7], u[10], SIN2, tmp710a, tmp710b);
  fma_addsub(u[8], u[9], SIN2, tmp89a, tmp89b);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 258 FMA + 48 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 17-k, then each
// output pair j, 17-j is (u0 + sum_k cos(j*k*tau/17) sum_k) +- sum_k sin(j*k*tau/17) dif_k, with w = e^(2*pi*i/17).  As in fft11.cl, cbase = u0 +
// cos(4*tau/17) * sum zeroes one term per cos row (cos(4*tau/17) = 0.092 is the cosine nearest the middle of the range: every coefficient
// stays below 1.08 in magnitude), and the sin rows are scaled by sin(2*tau/17), the most accurately represented of the eight sines
// (relative rounding error 6.5e-18).
// Accumulation order as fft13.cl: cos terms by |coef| rank 1,0,3,2,5,4,6 (ranks from the smallest), sin terms smallest first.

void OVERLOAD fft17(F2 *u) {
  const float COS4 = 0.09226835946330199523965f;  // cos(4*tau/17): subtracted from every cos coefficient, it zeroes one term per cos row
  const float C1_C4 = 0.8402038699410538093335f;  // cos(1*tau/17) - cos(4*tau/17)
  const float C2_C4 = 0.6467405577573571206849f;  // cos(2*tau/17) - cos(4*tau/17)
  const float C3_C4 = 0.3534699963132362721568f;  // cos(3*tau/17) - cos(4*tau/17)
  const float C5_C4 = -0.3659313495353848587787f; // cos(5*tau/17) - cos(4*tau/17)
  const float C6_C4 = -0.6949029958425583844182f; // cos(6*tau/17) - cos(4*tau/17)
  const float C7_C4 = -0.9424854951929161473738f; // cos(7*tau/17) - cos(4*tau/17)
  const float C8_C4 = -1.075241459147203773522f;  // cos(8*tau/17) - cos(4*tau/17)
  const float SIN2 = 0.6736956436465572117127f;   // sin(2*tau/17): factored out of the sin rows, it makes one term per row a plain add
  const float S1_S2 = 0.5362089982233463159835f;  // sin(1*tau/17) / sin(2*tau/17)
  const float S3_S2 = 1.328735460585365293163f;   // sin(3*tau/17) / sin(2*tau/17)
  const float S4_S2 = 1.478017834441318231849f;   // sin(4*tau/17) / sin(2*tau/17)
  const float S5_S2 = 1.427685709776422850776f;   // sin(5*tau/17) / sin(2*tau/17)
  const float S6_S2 = 1.184536718926603990479f;   // sin(6*tau/17) / sin(2*tau/17)
  const float S7_S2 = 0.7814094804411995660846f;  // sin(7*tau/17) / sin(2*tau/17)
  const float S8_S2 = 0.2727485616828054534919f;  // sin(8*tau/17) / sin(2*tau/17)

  X2_mul_t4(u[1], u[16]);
  X2_mul_t4(u[2], u[15]);
  X2_mul_t4(u[3], u[14]);
  X2_mul_t4(u[4], u[13]);
  X2_mul_t4(u[5], u[12]);
  X2_mul_t4(u[6], u[11]);
  X2_mul_t4(u[7], u[10]);
  X2_mul_t4(u[8], u[9]);

  F2 sum = u[1] + u[2] + u[3] + u[4] + u[5] + u[6] + u[7] + u[8];
  F2 cbase = fmaT2(COS4, sum, u[0]);                    // u[0] + COS4 * sum: each cos row below then has one term fewer
  u[0] = u[0] + sum;

  F2 tmp116a = fmaT2(C5_C4, u[5], cbase);
  F2 tmp215a = fmaT2(C5_C4, u[6], cbase);
  F2 tmp314a = fmaT2(C5_C4, u[4], cbase);
  F2 tmp413a = fmaT2(C5_C4, u[3], cbase);
  F2 tmp512a = fmaT2(C5_C4, u[1], cbase);
  F2 tmp611a = fmaT2(C5_C4, u[2], cbase);
  F2 tmp710a = fmaT2(C5_C4, u[8], cbase);
  F2 tmp89a = fmaT2(C5_C4, u[7], cbase);

  tmp116a = fmaT2(C3_C4, u[3], tmp116a);
  tmp215a = fmaT2(C3_C4, u[7], tmp215a);
  tmp314a = fmaT2(C3_C4, u[1], tmp314a);
  tmp413a = fmaT2(C3_C4, u[5], tmp413a);
  tmp512a = fmaT2(C3_C4, u[4], tmp512a);
  tmp611a = fmaT2(C3_C4, u[8], tmp611a);
  tmp710a = fmaT2(C3_C4, u[2], tmp710a);
  tmp89a = fmaT2(C3_C4, u[6], tmp89a);

  tmp116a = fmaT2(C6_C4, u[6], tmp116a);
  tmp215a = fmaT2(C6_C4, u[3], tmp215a);
  tmp314a = fmaT2(C6_C4, u[2], tmp314a);
  tmp413a = fmaT2(C6_C4, u[7], tmp413a);
  tmp512a = fmaT2(C6_C4, u[8], tmp512a);
  tmp611a = fmaT2(C6_C4, u[1], tmp611a);
  tmp710a = fmaT2(C6_C4, u[4], tmp710a);
  tmp89a = fmaT2(C6_C4, u[5], tmp89a);

  tmp116a = fmaT2(C2_C4, u[2], tmp116a);
  tmp215a = fmaT2(C2_C4, u[1], tmp215a);
  tmp314a = fmaT2(C2_C4, u[5], tmp314a);
  tmp413a = fmaT2(C2_C4, u[8], tmp413a);
  tmp512a = fmaT2(C2_C4, u[3], tmp512a);
  tmp611a = fmaT2(C2_C4, u[6], tmp611a);
  tmp710a = fmaT2(C2_C4, u[7], tmp710a);
  tmp89a = fmaT2(C2_C4, u[4], tmp89a);

  tmp116a = fmaT2(C7_C4, u[7], tmp116a);
  tmp215a = fmaT2(C7_C4, u[5], tmp215a);
  tmp314a = fmaT2(C7_C4, u[8], tmp314a);
  tmp413a = fmaT2(C7_C4, u[6], tmp413a);
  tmp512a = fmaT2(C7_C4, u[2], tmp512a);
  tmp611a = fmaT2(C7_C4, u[4], tmp611a);
  tmp710a = fmaT2(C7_C4, u[1], tmp710a);
  tmp89a = fmaT2(C7_C4, u[3], tmp89a);

  tmp116a = fmaT2(C1_C4, u[1], tmp116a);
  tmp215a = fmaT2(C1_C4, u[8], tmp215a);
  tmp314a = fmaT2(C1_C4, u[6], tmp314a);
  tmp413a = fmaT2(C1_C4, u[4], tmp413a);
  tmp512a = fmaT2(C1_C4, u[7], tmp512a);
  tmp611a = fmaT2(C1_C4, u[3], tmp611a);
  tmp710a = fmaT2(C1_C4, u[5], tmp710a);
  tmp89a = fmaT2(C1_C4, u[2], tmp89a);

  tmp116a = fmaT2(C8_C4, u[8], tmp116a);
  tmp215a = fmaT2(C8_C4, u[4], tmp215a);
  tmp314a = fmaT2(C8_C4, u[3], tmp314a);
  tmp413a = fmaT2(C8_C4, u[2], tmp413a);
  tmp512a = fmaT2(C8_C4, u[5], tmp512a);
  tmp611a = fmaT2(C8_C4, u[7], tmp611a);
  tmp710a = fmaT2(C8_C4, u[6], tmp710a);
  tmp89a = fmaT2(C8_C4, u[1], tmp89a);

  F2 tmp116b = fmaT2(S8_S2, u[9], u[15]);
  F2 tmp215b = fmaT2(S8_S2, u[13], u[16]);
  F2 tmp314b = fmaT2(S8_S2, -u[14], -u[12]);
  F2 tmp413b = fmaT2(S8_S2, u[15], -u[9]);
  F2 tmp512b = fmaT2(S8_S2, u[12], -u[14]);
  F2 tmp611b = fmaT2(S8_S2, u[10], u[11]);
  F2 tmp710b = fmaT2(S8_S2, u[11], -u[10]);
  F2 tmp89b = fmaT2(S8_S2, u[16], -u[13]);

  tmp116b = fmaT2(S1_S2, u[16], tmp116b);
  tmp215b = fmaT2(S1_S2, -u[9], tmp215b);
  tmp314b = fmaT2(S1_S2, u[11], tmp314b);
  tmp413b = fmaT2(S1_S2, -u[13], tmp413b);
  tmp512b = fmaT2(S1_S2, u[10], tmp512b);
  tmp611b = fmaT2(S1_S2, u[14], tmp611b);
  tmp710b = fmaT2(S1_S2, u[12], tmp710b);
  tmp89b = fmaT2(S1_S2, -u[15], tmp89b);

  tmp116b = fmaT2(S7_S2, u[10], tmp116b);
  tmp215b = fmaT2(S7_S2, -u[12], tmp215b);
  tmp314b = fmaT2(S7_S2, u[9], tmp314b);
  tmp413b = fmaT2(S7_S2, u[11], tmp413b);
  tmp512b = fmaT2(S7_S2, -u[15], tmp512b);
  tmp611b = fmaT2(S7_S2, u[13], tmp611b);
  tmp710b = fmaT2(S7_S2, u[16], tmp710b);
  tmp89b = fmaT2(S7_S2, u[14], tmp89b);

  tmp116b = fmaT2(S6_S2, u[11], tmp116b);
  tmp215b = fmaT2(S6_S2, u[14], tmp215b);
  tmp314b = fmaT2(S6_S2, u[15], tmp314b);
  tmp413b = fmaT2(S6_S2, -u[10], tmp413b);
  tmp512b = fmaT2(S6_S2, u[9], tmp512b);
  tmp611b = fmaT2(S6_S2, u[16], tmp611b);
  tmp710b = fmaT2(S6_S2, -u[13], tmp710b);
  tmp89b = fmaT2(S6_S2, u[12], tmp89b);

  tmp116b = fmaT2(S3_S2, u[14], tmp116b);
  tmp215b = fmaT2(S3_S2, -u[10], tmp215b);
  tmp314b = fmaT2(S3_S2, u[16], tmp314b);
  tmp413b = fmaT2(S3_S2, u[12], tmp413b);
  tmp512b = fmaT2(S3_S2, u[13], tmp512b);
  tmp611b = fmaT2(S3_S2, -u[9], tmp611b);
  tmp710b = fmaT2(S3_S2, -u[15], tmp710b);
  tmp89b = fmaT2(S3_S2, -u[11], tmp89b);

  tmp116b = fmaT2(S5_S2, u[12], tmp116b);
  tmp215b = fmaT2(S5_S2, -u[11], tmp215b);
  tmp314b = fmaT2(S5_S2, -u[13], tmp314b);
  tmp413b = fmaT2(S5_S2, -u[14], tmp413b);
  tmp512b = fmaT2(S5_S2, u[16], tmp512b);
  tmp611b = fmaT2(S5_S2, -u[15], tmp611b);
  tmp710b = fmaT2(S5_S2, u[9], tmp710b);
  tmp89b = fmaT2(S5_S2, u[10], tmp89b);

  tmp116b = fmaT2(S4_S2, u[13], tmp116b);
  tmp215b = fmaT2(S4_S2, u[15], tmp215b);
  tmp314b = fmaT2(S4_S2, u[10], tmp314b);
  tmp413b = fmaT2(S4_S2, u[16], tmp413b);
  tmp512b = fmaT2(S4_S2, -u[11], tmp512b);
  tmp611b = fmaT2(S4_S2, -u[12], tmp611b);
  tmp710b = fmaT2(S4_S2, u[14], tmp710b);
  tmp89b = fmaT2(S4_S2, -u[9], tmp89b);

  fma_addsub(u[1], u[16], SIN2, tmp116a, tmp116b);
  fma_addsub(u[2], u[15], SIN2, tmp215a, tmp215b);
  fma_addsub(u[3], u[14], SIN2, tmp314a, tmp314b);
  fma_addsub(u[4], u[13], SIN2, tmp413a, tmp413b);
  fma_addsub(u[5], u[12], SIN2, tmp512a, tmp512b);
  fma_addsub(u[6], u[11], SIN2, tmp611a, tmp611b);
  fma_addsub(u[7], u[10], SIN2, tmp710a, tmp710b);
  fma_addsub(u[8], u[9], SIN2, tmp89a, tmp89b);
}

#endif
