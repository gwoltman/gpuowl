// Copyright (C) Mihai Preda

#if FFT_FP64

#include "fft8.cl"

// A radix-2 layer, then the two halves as in the 1K and 2K widths/heights: fft8_16a takes the sums and gives the even outputs, fft8_16b
// the differences, twiddles them by w^k and gives the odd outputs.
void OVERLOAD fft16(T2 *u) {
  for (u32 i = 0; i < 8; ++i) X2(u[i], u[i + 8]);
  fft8_16a(u);
  fft8_16b(u + 8);
  // Interleave: output 2k is fft8_16a's k, output 2k+1 is fft8_16b's k
  T2 t[16];
  for (u32 k = 0; k < 8; ++k) { t[2 * k] = u[k]; t[2 * k + 1] = u[k + 8]; }
  for (u32 k = 0; k < 16; ++k) { u[k] = t[k]; }
}

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
