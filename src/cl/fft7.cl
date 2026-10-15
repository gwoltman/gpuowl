// Copyright (C) Mihai Preda & George Woltman

#pragma once

#define A(k) u[(base + k * step) % M]

#if FFT_FP64

// 38 FMA + 18 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 7-k, then each output pair j, 7-j is
// (u0 + sum_k cos(j*k*tau/7) sum_k) +- sum_k sin(j*k*tau/7) dif_k, with w = e^(2*pi*i/7).  cbase = u0 + cos(2*tau/7) * sum zeroes one
// term per cos row (the cosine nearest the middle of the range: the smallest coefficients), and the sin rows are scaled by sin(3*tau/7),
// the most accurately represented sine.
// Accumulation order (a double-precision study: a few percent less round-off than index order): cos terms by |coef| rank 1,0 (largest first), sin terms smallest first.
void fft7by(T2 *u, u32 base, u32 step, u32 M) {
  const double COS2 = -0.2225209339563144042889;   // cos(2*tau/7): subtracted from every cos coefficient, it zeroes one term per cos row
  const double C1_C2 = 0.8460107358150479348139;   // cos(1*tau/7) - cos(2*tau/7)
  const double C3_C2 = -0.6784479339461047219472;  // cos(3*tau/7) - cos(2*tau/7)
  const double SIN3 = 0.4338837391175581204758;    // sin(3*tau/7): factored out of the sin rows, it makes one term per row a plain add
  const double S1_S3 = 1.8019377358048382524722;   // sin(1*tau/7) / sin(3*tau/7)
  const double S2_S3 = 2.2469796037174670610500;   // sin(2*tau/7) / sin(3*tau/7)

  X2_mul_t4(A(1), A(6));
  X2_mul_t4(A(2), A(5));
  X2_mul_t4(A(3), A(4));

  T2 sum = A(1) + A(2) + A(3);
  T2 cbase = fmaT2(COS2, sum, A(0));			// A(0) + COS2 * sum: each cos row below then has one term fewer
  A(0) = A(0) + sum;

  T2 tmp27a = fmaT2(C1_C2, A(1), cbase);
  T2 tmp36a = fmaT2(C1_C2, A(3), cbase);
  T2 tmp45a = fmaT2(C1_C2, A(2), cbase);

  tmp27a = fmaT2(C3_C2, A(3), tmp27a);
  tmp36a = fmaT2(C3_C2, A(2), tmp36a);
  tmp45a = fmaT2(C3_C2, A(1), tmp45a);

  T2 tmp27b = fmaT2(S1_S3, A(6), A(4));
  T2 tmp36b = fmaT2(S1_S3, -A(4), -A(5));
  T2 tmp45b = fmaT2(S1_S3, -A(5), A(6));

  tmp27b = fmaT2(S2_S3, A(5), tmp27b);
  tmp36b = fmaT2(S2_S3, A(6), tmp36b);
  tmp45b = fmaT2(S2_S3, A(4), tmp45b);

  fma_addsub(A(1), A(6), SIN3, tmp27a, tmp27b);
  fma_addsub(A(2), A(5), SIN3, tmp36a, tmp36b);
  fma_addsub(A(3), A(4), SIN3, tmp45a, tmp45b);
}

void fft7(T2 *u) { return fft7by(u, 0, 1, 7); }

#endif

/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 38 FMA + 18 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 7-k, then each output pair j, 7-j is
// (u0 + sum_k cos(j*k*tau/7) sum_k) +- sum_k sin(j*k*tau/7) dif_k, with w = e^(2*pi*i/7).  cbase = u0 + cos(2*tau/7) * sum zeroes one
// term per cos row (the cosine nearest the middle of the range: the smallest coefficients), and the sin rows are scaled by sin(3*tau/7),
// the most accurately represented sine.
// Accumulation order (a double-precision study: a few percent less round-off than index order): cos terms by |coef| rank 1,0 (largest first), sin terms smallest first.
void fft7by(F2 *u, u32 base, u32 step, u32 M) {
  const float COS2 = -0.2225209339563144042889f;   // cos(2*tau/7): subtracted from every cos coefficient, it zeroes one term per cos row
  const float C1_C2 = 0.8460107358150479348139f;   // cos(1*tau/7) - cos(2*tau/7)
  const float C3_C2 = -0.6784479339461047219472f;  // cos(3*tau/7) - cos(2*tau/7)
  const float SIN3 = 0.4338837391175581204758f;    // sin(3*tau/7): factored out of the sin rows, it makes one term per row a plain add
  const float S1_S3 = 1.8019377358048382524722f;   // sin(1*tau/7) / sin(3*tau/7)
  const float S2_S3 = 2.2469796037174670610500f;   // sin(2*tau/7) / sin(3*tau/7)

  X2_mul_t4(A(1), A(6));
  X2_mul_t4(A(2), A(5));
  X2_mul_t4(A(3), A(4));

  F2 sum = A(1) + A(2) + A(3);
  F2 cbase = fmaT2(COS2, sum, A(0));			// A(0) + COS2 * sum: each cos row below then has one term fewer
  A(0) = A(0) + sum;

  F2 tmp27a = fmaT2(C1_C2, A(1), cbase);
  F2 tmp36a = fmaT2(C1_C2, A(3), cbase);
  F2 tmp45a = fmaT2(C1_C2, A(2), cbase);

  tmp27a = fmaT2(C3_C2, A(3), tmp27a);
  tmp36a = fmaT2(C3_C2, A(2), tmp36a);
  tmp45a = fmaT2(C3_C2, A(1), tmp45a);

  F2 tmp27b = fmaT2(S1_S3, A(6), A(4));
  F2 tmp36b = fmaT2(S1_S3, -A(4), -A(5));
  F2 tmp45b = fmaT2(S1_S3, -A(5), A(6));

  tmp27b = fmaT2(S2_S3, A(5), tmp27b);
  tmp36b = fmaT2(S2_S3, A(6), tmp36b);
  tmp45b = fmaT2(S2_S3, A(4), tmp45b);

  fma_addsub(A(1), A(6), SIN3, tmp27a, tmp27b);
  fma_addsub(A(2), A(5), SIN3, tmp36a, tmp36b);
  fma_addsub(A(3), A(4), SIN3, tmp45a, tmp45b);
}

void fft7(F2 *u) { return fft7by(u, 0, 1, 7); }

#endif

#undef A
