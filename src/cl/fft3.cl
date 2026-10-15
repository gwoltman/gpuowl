// Copyright (C) Mihai Preda

#pragma once

#define A(k) u[(base + k * step) % M]

#if FFT_FP64

// 6 FMA + 6 ADD
void fft3by(T2 *u, u32 base, u32 step, u32 M) {
  const double SIN1 = 0.8660254037844386; // sin(tau/3) == sqrt(3)/2

  X2_mul_t4(A(1), A(2));

  T2 t = fmaT2(-0.5, A(1), A(0));

  A(0) = A(0) + A(1);
  A(1) = fmaT2( SIN1, A(2), t);
  A(2) = fmaT2(-SIN1, A(2), t);
}

void fft3(T2 *u) { fft3by(u, 0, 1, 3); }

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 6 FMA + 6 ADD
void fft3by(F2 *u, u32 base, u32 step, u32 M) {
  const float SIN1 = 0.8660254037844386f; // sin(tau/3) == sqrt(3)/2

  X2_mul_t4(A(1), A(2));

  F2 t = fmaT2(-0.5f, A(1), A(0));

  A(0) = A(0) + A(1);
  A(1) = fmaT2( SIN1, A(2), t);
  A(2) = fmaT2(-SIN1, A(2), t);
}

void fft3(F2 *u) { fft3by(u, 0, 1, 3); }

#endif

#undef A
