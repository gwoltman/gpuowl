// Copyright (C) Mihai Preda and George Woltman

#pragma once

#define A(k) u[(base + k * step) % M]

#if FFT_FP64

// 18 FMA + 12 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 5-k, then each output pair j, 5-j is
// (u0 + sum_k cos(j*k*tau/5) sum_k) +- sum_k sin(j*k*tau/5) dif_k, with w = e^(2*pi*i/5).  cbase = u0 + cos(1*tau/5) * sum zeroes one
// term per cos row (the cosine nearest the middle of the range: the smallest coefficients), and the sin rows are scaled by sin(2*tau/5),
// the most accurately represented sine.
void fft5by(T2 *u, u32 base, u32 step, u32 M) {
  const double COS1 = 0.3090169943749474241023;    // cos(1*tau/5): subtracted from every cos coefficient, it zeroes one term per cos row
  const double C2_C1 = -1.1180339887498948482046;  // cos(2*tau/5) - cos(1*tau/5)
  const double SIN2 = 0.5877852522924731291687;    // sin(2*tau/5): factored out of the sin rows, it makes one term per row a plain add
  const double S1_S2 = 1.6180339887498948482046;   // sin(1*tau/5) / sin(2*tau/5)

  X2_mul_t4(A(1), A(4));
  X2_mul_t4(A(2), A(3));

  T2 sum = A(1) + A(2);
  T2 cbase = fmaT2(COS1, sum, A(0));                    // A(0) + COS1 * sum: each cos row below then has one term fewer
  A(0) = A(0) + sum;

  T2 tmp34a = fmaT2(C2_C1, A(1), cbase);
  T2 tmp25a = fmaT2(C2_C1, A(2), cbase);

  T2 tmp25b = fmaT2(S1_S2, A(4), A(3));
  T2 tmp34b = fmaT2(S1_S2, -A(3), A(4));

  fma_addsub(A(1), A(4), SIN2, tmp25a, tmp25b);
  fma_addsub(A(2), A(3), SIN2, tmp34a, tmp34b);
}

void fft5(T2 *u) { return fft5by(u, 0, 1, 5); }

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 18 FMA + 12 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 5-k, then each output pair j, 5-j is
// (u0 + sum_k cos(j*k*tau/5) sum_k) +- sum_k sin(j*k*tau/5) dif_k, with w = e^(2*pi*i/5).  cbase = u0 + cos(1*tau/5) * sum zeroes one
// term per cos row (the cosine nearest the middle of the range: the smallest coefficients), and the sin rows are scaled by sin(2*tau/5),
// the most accurately represented sine.
void fft5by(F2 *u, u32 base, u32 step, u32 M) {
  const float COS1 = 0.3090169943749474241023f;    // cos(1*tau/5): subtracted from every cos coefficient, it zeroes one term per cos row
  const float C2_C1 = -1.1180339887498948482046f;  // cos(2*tau/5) - cos(1*tau/5)
  const float SIN2 = 0.5877852522924731291687f;    // sin(2*tau/5): factored out of the sin rows, it makes one term per row a plain add
  const float S1_S2 = 1.6180339887498948482046f;   // sin(1*tau/5) / sin(2*tau/5)

  X2_mul_t4(A(1), A(4));
  X2_mul_t4(A(2), A(3));

  F2 sum = A(1) + A(2);
  F2 cbase = fmaT2(COS1, sum, A(0));                    // A(0) + COS1 * sum: each cos row below then has one term fewer
  A(0) = A(0) + sum;

  F2 tmp34a = fmaT2(C2_C1, A(1), cbase);
  F2 tmp25a = fmaT2(C2_C1, A(2), cbase);

  F2 tmp25b = fmaT2(S1_S2, A(4), A(3));
  F2 tmp34b = fmaT2(S1_S2, -A(3), A(4));

  fma_addsub(A(1), A(4), SIN2, tmp25a, tmp25b);
  fma_addsub(A(2), A(3), SIN2, tmp34a, tmp34b);
}

void fft5(F2 *u) { return fft5by(u, 0, 1, 5); }

#endif

#undef A
