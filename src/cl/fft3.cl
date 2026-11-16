// Copyright (C) Mihai Preda

#pragma once

#define A(k) u[(base + k * step) % M]

#if FFT_FP64

// 6 FMA + 6 ADD
void OVERLOAD fft3by(T2 *u, u32 base, u32 step, u32 M) {
  const double SIN1 = 0.8660254037844386; // sin(tau/3) == sqrt(3)/2

  X2_mul_t4(A(1), A(2));

  T2 t = fmaT2(-0.5, A(1), A(0));

  A(0) = A(0) + A(1);
  A(1) = fmaT2( SIN1, A(2), t);
  A(2) = fmaT2(-SIN1, A(2), t);
}

void OVERLOAD fft3(T2 *u) { fft3by(u, 0, 1, 3); }

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 6 FMA + 6 ADD
void OVERLOAD fft3by(F2 *u, u32 base, u32 step, u32 M) {
  const float SIN1 = 0.8660254037844386f; // sin(tau/3) == sqrt(3)/2

  X2_mul_t4(A(1), A(2));

  F2 t = fmaT2(-0.5f, A(1), A(0));

  A(0) = A(0) + A(1);
  A(1) = fmaT2( SIN1, A(2), t);
  A(2) = fmaT2(-SIN1, A(2), t);
}

void OVERLOAD fft3(F2 *u) { fft3by(u, 0, 1, 3); }

#endif

/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

#if PFA == 3 && PFA_W31 != 1513477735u
#error PFA_W31 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-3 DFT, y_j = sum_k u_k w^(jk) with w = 7^((p-1)/3) (pfaRootOfUnity), for the radix-PFA of a PFA middle:  w^2 + w + 1 = 0, so
// y1 = u0 - u2 + w (u1 - u2), y2 = u0 - u1 - w (u1 - u2).
void OVERLOAD fft3by(GF31 *u, u32 base, u32 step, u32 M) {
  const Z31 W = 1513477735u;                           // w, a cube root of unity:  w^2 + w + 1 = 0, so one scalar multiply
  GF31 wd = mul(sub(A(1), A(2)), W);
  GF31 y0 = add(add(A(0), A(1)), A(2));
  GF31 y1 = add(sub(A(0), A(2)), wd);
  GF31 y2 = sub(sub(A(0), A(1)), wd);
  A(0) = y0; A(1) = y1; A(2) = y2;
}

void OVERLOAD fft3(GF31 *u) { fft3by(u, 0, 1, 3); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

#if PFA == 3 && PFA_W61 != 1669582390241348315ull
#error PFA_W61 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-3 DFT, y_j = sum_k u_k w^(jk) with w = 37^((p-1)/3) (pfaRootOfUnity), for the radix-PFA of a PFA middle:  w^2 + w + 1 = 0, so
// y1 = u0 - u2 + w (u1 - u2), y2 = u0 - u1 - w (u1 - u2).
void OVERLOAD fft3by(GF61 *u, u32 base, u32 step, u32 M) {
  // MW = M61 - w = -w, w a cube root of unity (w^2 + w + 1 = 0, so one scalar multiply), stored times 8 for the cheaper weakModM61x8.  -w rather
  // than w:  8 MW + d stays below 2^64 (8 w + d does not), so mul64s applies, and the product's range is smaller.
  const Z61 MW = 636260618972345636ull;                  // 0.28 M61
  // Quick (unreduced) arithmetic, ranges in M61s as in math.cl.  Inputs 0..1+.
  GF61 d = A(1) + neg(A(2), 2);                                         // 1-..3+, positive for the multiply
  GF61 nwd = U2(weakModM61x8(mul64s(d.x, X8(MW))), weakModM61x8(mul64s(d.y, X8(MW))));         // -w (u1 - u2), 0..1.83+
  GF61 y0 = A(0) + A(1) + A(2);                                         // 0..3+
  GF61 a0 = addq(A(0), 3);                                              // 3..4+, so that y1 and y2 are positive
  GF61 y1 = a0 - A(2) - nwd;                                            // 0.17-..4+
  GF61 y2 = a0 - A(1) + nwd;                                            // 2-..5.83+
  A(0) = modM61q(y0, 0); A(1) = modM61q(y1, 0); A(2) = modM61q(y2, 0);
}

void OVERLOAD fft3(GF61 *u) { fft3by(u, 0, 1, 3); }

#endif


#undef A
