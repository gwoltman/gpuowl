// Copyright (C) Mihai Preda & George Woltman

#include "fft3.cl"

#if FFT_FP64

// 12 FMA + 24 ADD
void fft6(T2 *u) {
#if 1
  X2(u[0], u[3]);
  X2(u[2], u[5]);
  X2(u[4], u[1]);

  fft3by(u, 0, 2, 6);
  fft3by(u, 3, 2, 6);

  // Fix order: 0 5 4 3 2 1
  SWAP(u[1], u[5]);
  SWAP(u[2], u[4]);

#else

  const double COS1 = -0.5;               // cos(tau/3)
  const double SIN1 = 0.8660254037844386; // sin(tau/3) == sqrt(3)/2
  X2(u[0], u[3]);						// (r1+ i1+),  (r1-  i1-)
  X2_mul_t4(u[1], u[4]);					// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[5]);					// (r3+ i3+),  (i3- -r3-)

  X2_mul_t4(u[1], u[2]);					// (r2++  i2++),  (i2+- -r2+-)
  X2_mul_t4(u[4], u[5]);					// (i2-+ -r2-+), (-r2-- -i2--)

  T2 tmp35a = fmaT2(COS1, u[1], u[0]);
  u[0] = u[0] + u[1];
  T2 tmp26a = fmaT2(COS1, u[5], u[3]);
  u[3] = u[3] + u[5];

  fma_addsub(u[1], u[5], -SIN1, tmp26a, u[4]);
  fma_addsub(u[2], u[4], -SIN1, tmp35a, u[2]);
#endif
}

#endif
