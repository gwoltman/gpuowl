// Copyright (C) Mihai Preda

#if FFT_FP64

#if 1

#if 1

#include "fft4.cl"

// 56 FMA + 96 ADD
void OVERLOAD fft16(T2 *u) {
  double
      C1 = 0.92387953251128674, // cos(tau/16)
      S1 = 0.38268343236508978, // sin(tau/16)
      S1_over_C1 = 0.4142135623730950488017,
      C1_over_S1 = 2.4142135623730950488017;

  for (int i = 0; i < 4; ++i) { fft4by(u, i, 4, 16); }

  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
  SWAP(u[1], u[2]);

  u[9]  = mul_t8_delayed(u[9]);              // delays a mul by M_SQRT1_2
  u[11] = mul_t8_delayed(u[11]);             // delays a mul by i*M_SQRT1_2 (a negation cheaper than mul_3t8_delayed)
  X2t4(u[8], u[10]);
  X2t4_mul_t4(u[9], u[11]);
  X2ad(u[8], u[9], M_SQRT1_2);
  X2ad(u[10], u[11], M_SQRT1_2);
  SWAP(u[9], u[10]);

  u[5]  = partial_cmul(u[5], S1_over_C1);    // delays a mul by C1
  u[6]  = mul_t8_delayed(u[6]);              // delays a mul by M_SQRT1_2
  u[7]  = partial_cmul(u[7], C1_over_S1);    // delays a mul by S1
  X2ad(u[4], u[6], M_SQRT1_2);
  X2ad_mul_t4(u[5], u[7], S1_over_C1);       // mul by S1/C1, now both are delaying a mul by C1
  X2ad(u[4], u[5], C1);                      // apply delayed mul by C1
  X2ad(u[6], u[7], C1);                      // apply delayed mul by C1
  SWAP(u[5], u[6]);

  u[13] = partial_cmul(u[13], C1_over_S1);   // delays a mul by S1
  u[14] = mul_t8_delayed(u[14]);             // delays a mul by i*M_SQRT1_2 (a negation cheaper than mul_3t8_delayed)
  u[15] = partial_cmul(u[15], S1_over_C1);   // delays a mul by -C1
  X2t4ad(u[12], u[14], M_SQRT1_2);
  X2ad_mul_t4(u[13], u[15], -C1_over_S1);    // mul by -C1/S1, now both are delaying a mul by S1
  X2ad(u[12], u[13], S1);                    // apply delayed mul by S1
  X2ad(u[14], u[15], S1);                    // apply delayed mul by S1
  SWAP(u[13], u[14]);

  SWAP(u[1], u[4]);
  SWAP(u[2], u[8]);
  SWAP(u[3], u[12]);
  SWAP(u[6], u[9]);
  SWAP(u[7], u[13]);
  SWAP(u[11], u[14]);
}

#else

#include "fft8.cl"

void OVERLOAD fft16(T2 *u) {
  double
      C1 = 0.92387953251128674, // cos(tau/16)
      S1 = 0.38268343236508978; // sin(tau/16)

  for (int i = 0; i < 8; ++i) { X2(u[i], u[i + 8]); }
  u[ 9] = cmul(u[ 9], U2( C1, S1)); // 1t16
  u[11] = cmul(u[11], U2( S1, C1)); // 3t16
  u[13] = cmul(u[13], U2(-S1, C1)); // 5t16
  u[15] = cmul(u[15], U2(-C1, S1)); // 7t16

  u[10] = mul_t8(u[10]);
  u[14] = mul_3t8(u[14]);

  u[12] = mul_t4(u[12]);

  fft8Core(u);
  fft8Core(u + 8);

  // revbin fix order
  // 0 8 4 12 2 10 6 14 1 9 5 13 3 11 7 15
  SWAP(u[1], u[8]);
  SWAP(u[2], u[4]);
  SWAP(u[3], u[12]);
  SWAP(u[5], u[10]);
  SWAP(u[7], u[14]);
  SWAP(u[11], u[13]);
}
#endif

#else

// FFT-16 Adapted from Nussbaumer, "Fast Fourier Transform and Convolution Algorithms"
// 28 FMA + 124 ADD
void OVERLOAD fft16(T2 *u) {
  double
      C1 = 0.70710678118654757,   // cos(2t/16)
      C2 = 0.38268343236508978,   // cos(3t/16)
      C3 = 1.3065629648763766,    // cos(3t/16) + cos(t/16)
      C4 = -0.54119610014619701,  // cos(3t/16) - cos(t/16)
      S2 = -0.076120467488713248, // sin(3t/16) - 1
      S3 = 0.54119610014619701,   // sin(3t/16) - sin(t/16)
      S4 = 1.3065629648763766;    // sin(3t/16) + sin(t/16)

  for (int i = 0; i < 8; ++i) { X2(u[i], u[i + 8]); }

  X2(u[0], u[4]);
  X2(u[1], u[5]);
  X2(u[2], u[6]);
  X2(u[3], u[7]);
  X2(u[10], u[14]);
  u[10] = mul_t4(u[10]);

  X2(u[0], u[2]);
  X2(u[1], u[3]);
  X2(u[5], u[7]);
  X2(u[0], u[1]);
  X2(u[9], u[15]);
  X2(u[13], u[11]);

  u[5] = mul_t4(u[5]);
  u[6] = mul_t4(u[6]);
  T2 s3 = fmaT2(C1, u[5],  u[6]);
  T2 s4 = fmaT2(C1, u[5], -u[6]);

  T2 s5 = fmaT2( C1, u[14], u[8]);
  u[5]  = fmaT2(-C1, u[14], u[8]);
  u[14] = s3;
  u[8] = u[1];
  u[1] = s5;

  T2 s1 = fmaT2(C1, u[7], u[4]);
  u[6] = fmaT2(-C1, u[7], u[4]);
  u[4] = u[2];
  u[2] = s1;

  T2 t25 = mul_t4(u[13]);
  T2 m7 = C2 * (u[15] + u[11]);
  T2 s7 = fmaT2(C3, u[15], -m7);
  u[13] = fmaT2(C4, u[11], -m7);

  u[12] = mul_t4(u[12]);
  u[15] = fmaT2( C1, u[10], u[12]);
  u[11] = fmaT2(-C1, u[10], u[12]);
  u[10] = s4;
  u[12] = mul_t4(u[3]);

  T2 t23 = mul_t4(u[9]);
  u[9] = s7;
  u[3] = t23 + t25;
  T2 m15 = fmaT2(S2, u[3], u[3]);
  u[3] = fmaT2(-S4, t25, m15);
  u[7] = fmaT2(-S3, t23, m15);

  X2(u[1], u[9]);
  X2(u[4], u[12]);
  X2(u[5], u[13]);
  X2(u[15], u[7]);
  X2(u[11], u[3]);

  X2(u[1], u[15]);
  X2(u[2], u[14]);
  X2(u[13], u[3]);
  X2(u[5], u[11]);
  X2(u[6], u[10]);
  X2(u[9], u[7]);
}

#endif

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

#include "fft8.cl"

// A radix-2 layer, then the two halves as in the 1K and 2K widths/heights: fft8_16a takes the sums and gives the even outputs, fft8_16b
// the differences, twiddles them by w^k and gives the odd outputs.
void OVERLOAD fft16(F2 *u) {
  for (u32 i = 0; i < 8; ++i) X2(u[i], u[i + 8]);
  fft8_16a(u);
  fft8_16b(u + 8);
  // Interleave: output 2k is fft8_16a's k, output 2k+1 is fft8_16b's k
  F2 t[16];
  for (u32 k = 0; k < 8; ++k) { t[2 * k] = u[k]; t[2 * k + 1] = u[k + 8]; }
  for (u32 k = 0; k < 16; ++k) { u[k] = t[k]; }
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

#include "fft8.cl"

// A radix-2 layer, then the two halves as in the 1K and 2K widths/heights: fft8_16a takes the sums and gives the even outputs, fft8_16b
// the differences, twiddles them by w^k and gives the odd outputs.
void OVERLOAD fft16(GF31 *u) {
  for (u32 i = 0; i < 8; ++i) X2(u[i], u[i + 8]);
  fft8_16a(u);
  fft8_16b(u + 8);
  // Interleave: output 2k is fft8_16a's k, output 2k+1 is fft8_16b's k
  GF31 t[16];
  for (u32 k = 0; k < 8; ++k) { t[2 * k] = u[k]; t[2 * k + 1] = u[k + 8]; }
  for (u32 k = 0; k < 16; ++k) { u[k] = t[k]; }
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

#include "fft8.cl"

// A radix-2 layer of quick adds and subtracts, then the two halves as in the 1K and 2K widths/heights: fft8_16a takes the sums (0..2+)
// and gives the even outputs, fft8_16b the differences (-1-..1+), twiddles them by w^k and gives the odd outputs.  Inputs 0..1+.
void OVERLOAD fft16(GF61 *u) {
  for (u32 i = 0; i < 8; ++i) X2q(&u[i], &u[i + 8]);     // u[0..7] = 0..2+, u[8..15] = -1-..1+
  fft8_16a(u);
  fft8_16b(u + 8);
  // Interleave: output 2k is fft8_16a's k, output 2k+1 is fft8_16b's k
  GF61 t[16];
  for (u32 k = 0; k < 8; ++k) { t[2 * k] = u[k]; t[2 * k + 1] = u[k + 8]; }
  for (u32 k = 0; k < 16; ++k) { u[k] = t[k]; }
}

#endif
