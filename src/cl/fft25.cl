// Copyright (C) Mihai Preda and George Woltman

#include "fft5.cl"

#if FFT_FP64

// Radix-25 DFT as 5 x 5 Cooley-Tukey (25 = 5 * 5 is not a coprime split, so unlike fft15/fft21/fft33 it needs twiddles):  inputs
// n = 5 n1 + n2, outputs k = k1 + 5 k2.  Five radix-5 DFTs over n1 (u[n2 + 5 n1]), the sixteen twiddles w^(n2 k1), five radix-5 DFTs
// over n2, whose output k2 lands at u[5 k1 + k2]: a 5 x 5 transpose puts it at k1 + 5 k2 (free in registers).
// 10 * (18 FMA + 12 ADD) + 16 cmul.  w = e^(2*pi*i/25).
void OVERLOAD fft25(T2 *u) {
  const T2 W1 = U2(0.9685831611286311194902, 0.2486898871648547882423);             // e^(2*pi*i*1/25)
  const T2 W2 = U2(0.8763066800438635873081, 0.4817536741017152749872);             // e^(2*pi*i*2/25)
  const T2 W3 = U2(0.7289686274214115231467, 0.6845471059286886737323);             // e^(2*pi*i*3/25)
  const T2 W4 = U2(0.5358267949789966182713, 0.8443279255020150785486);             // e^(2*pi*i*4/25)
  const T2 W6 = U2(0.06279051952931337607618, 0.9980267284282715619523);            // e^(2*pi*i*6/25)
  const T2 W8 = U2(-0.4257792915650726488625, 0.9048270524660195277137);            // e^(2*pi*i*8/25)
  const T2 W9 = U2(-0.6374239897486897101767, 0.7705132427757892308030);            // e^(2*pi*i*9/25)
  const T2 W12 = U2(-0.9921147013144778310498, 0.1253332335643042453731);           // e^(2*pi*i*12/25)
  const T2 W16 = U2(-0.6374239897486897101767, -0.7705132427757892308030);          // e^(2*pi*i*16/25)

  #pragma unroll
  for (u32 n2 = 0; n2 < 5; ++n2) { fft5by(u, n2, 5, 25); }                     // output k1 of the n2-th DFT at u[n2 + 5 k1]
  u[6] = cmul(u[6], W1);
  u[11] = cmul(u[11], W2);
  u[16] = cmul(u[16], W3);
  u[21] = cmul(u[21], W4);
  u[7] = cmul(u[7], W2);
  u[12] = cmul(u[12], W4);
  u[17] = cmul(u[17], W6);
  u[22] = cmul(u[22], W8);
  u[8] = cmul(u[8], W3);
  u[13] = cmul(u[13], W6);
  u[18] = cmul(u[18], W9);
  u[23] = cmul(u[23], W12);
  u[9] = cmul(u[9], W4);
  u[14] = cmul(u[14], W8);
  u[19] = cmul(u[19], W12);
  u[24] = cmul(u[24], W16);
  #pragma unroll
  for (u32 k1 = 0; k1 < 5; ++k1) { fft5by(u, 5 * k1, 1, 25); }                      // output k2 of the k1-th DFT at u[5 k1 + k2]
  SWAP(u[1], u[5]);
  SWAP(u[2], u[10]);
  SWAP(u[3], u[15]);
  SWAP(u[4], u[20]);
  SWAP(u[7], u[11]);
  SWAP(u[8], u[16]);
  SWAP(u[9], u[21]);
  SWAP(u[13], u[17]);
  SWAP(u[14], u[22]);
  SWAP(u[19], u[23]);
}

#endif

/**************************************************************************/
/*                  Similar to above, but for an FFT based on FP32        */
/**************************************************************************/

#if FFT_FP32

// Radix-25 DFT as 5 x 5 Cooley-Tukey (25 = 5 * 5 is not a coprime split, so unlike fft15/fft21/fft33 it needs twiddles):  inputs
// n = 5 n1 + n2, outputs k = k1 + 5 k2.  Five radix-5 DFTs over n1 (u[n2 + 5 n1]), the sixteen twiddles w^(n2 k1), five radix-5 DFTs
// over n2, whose output k2 lands at u[5 k1 + k2]: a 5 x 5 transpose puts it at k1 + 5 k2 (free in registers).
// 10 * (18 FMA + 12 ADD) + 16 cmul.  w = e^(2*pi*i/25).
void OVERLOAD fft25(F2 *u) {
  const F2 W1 = U2(0.9685831611286311194902f, 0.2486898871648547882423f);           // e^(2*pi*i*1/25)
  const F2 W2 = U2(0.8763066800438635873081f, 0.4817536741017152749872f);           // e^(2*pi*i*2/25)
  const F2 W3 = U2(0.7289686274214115231467f, 0.6845471059286886737323f);           // e^(2*pi*i*3/25)
  const F2 W4 = U2(0.5358267949789966182713f, 0.8443279255020150785486f);           // e^(2*pi*i*4/25)
  const F2 W6 = U2(0.06279051952931337607618f, 0.9980267284282715619523f);          // e^(2*pi*i*6/25)
  const F2 W8 = U2(-0.4257792915650726488625f, 0.9048270524660195277137f);          // e^(2*pi*i*8/25)
  const F2 W9 = U2(-0.6374239897486897101767f, 0.7705132427757892308030f);          // e^(2*pi*i*9/25)
  const F2 W12 = U2(-0.9921147013144778310498f, 0.1253332335643042453731f);         // e^(2*pi*i*12/25)
  const F2 W16 = U2(-0.6374239897486897101767f, -0.7705132427757892308030f);        // e^(2*pi*i*16/25)

  #pragma unroll
  for (u32 n2 = 0; n2 < 5; ++n2) { fft5by(u, n2, 5, 25); }                     // output k1 of the n2-th DFT at u[n2 + 5 k1]
  u[6] = cmul(u[6], W1);
  u[11] = cmul(u[11], W2);
  u[16] = cmul(u[16], W3);
  u[21] = cmul(u[21], W4);
  u[7] = cmul(u[7], W2);
  u[12] = cmul(u[12], W4);
  u[17] = cmul(u[17], W6);
  u[22] = cmul(u[22], W8);
  u[8] = cmul(u[8], W3);
  u[13] = cmul(u[13], W6);
  u[18] = cmul(u[18], W9);
  u[23] = cmul(u[23], W12);
  u[9] = cmul(u[9], W4);
  u[14] = cmul(u[14], W8);
  u[19] = cmul(u[19], W12);
  u[24] = cmul(u[24], W16);
  #pragma unroll
  for (u32 k1 = 0; k1 < 5; ++k1) { fft5by(u, 5 * k1, 1, 25); }                      // output k2 of the k1-th DFT at u[5 k1 + k2]
  SWAP(u[1], u[5]);
  SWAP(u[2], u[10]);
  SWAP(u[3], u[15]);
  SWAP(u[4], u[20]);
  SWAP(u[7], u[11]);
  SWAP(u[8], u[16]);
  SWAP(u[9], u[21]);
  SWAP(u[13], u[17]);
  SWAP(u[14], u[22]);
  SWAP(u[19], u[23]);
}

#endif

/**************************************************************************/
/*               Similar to above, but for an NTT based on GF(M61^2)      */
/**************************************************************************/

#if NTT_GF61

#if PFA == 25 && PFA_W61 != 777611702656095618ull
#error PFA_W61 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-25 DFT as 5 x 5 Cooley-Tukey (25 = 5 * 5 is not a coprime split, so unlike fft15/fft21/fft33 it needs twiddles):  inputs
// n = 5 n1 + n2, outputs k = k1 + 5 k2.  Five radix-5 DFTs over n1 (u[n2 + 5 n1]), the sixteen twiddles w^(n2 k1), five radix-5 DFTs
// over n2, whose output k2 lands at u[5 k1 + k2]: a 5 x 5 transpose puts it at k1 + 5 k2 (free in registers).
// w = 37^((p-1)/25) (pfaRootOfUnity) is in Z/pZ, so each twiddle is a scalar multiply.  fft5.cl hardwires w^5.
void OVERLOAD fft25(GF61 *u) {
  const Z61 W1 = 777611702656095618ull;                     // w^1
  const Z61 W2 = 1554350452820481594ull;                    // w^2
  const Z61 W3 = 2246822354163092846ull;                    // w^3
  const Z61 W4 = 765166422852948037ull;                     // w^4
  const Z61 W6 = 855356772850685687ull;                     // w^6
  const Z61 W8 = 614744475622347974ull;                     // w^8
  const Z61 W9 = 195375470834431163ull;                     // w^9
  const Z61 W12 = 1002938224617487650ull;                   // w^12
  const Z61 W16 = 2067802003810870481ull;                   // w^16

  #pragma unroll
  for (u32 n2 = 0; n2 < 5; ++n2) { fft5by(u, n2, 5, 25); }                          // output k1 of the n2-th DFT at u[n2 + 5 k1]
  u[6] = mul(u[6], W1);
  u[11] = mul(u[11], W2);
  u[16] = mul(u[16], W3);
  u[21] = mul(u[21], W4);
  u[7] = mul(u[7], W2);
  u[12] = mul(u[12], W4);
  u[17] = mul(u[17], W6);
  u[22] = mul(u[22], W8);
  u[8] = mul(u[8], W3);
  u[13] = mul(u[13], W6);
  u[18] = mul(u[18], W9);
  u[23] = mul(u[23], W12);
  u[9] = mul(u[9], W4);
  u[14] = mul(u[14], W8);
  u[19] = mul(u[19], W12);
  u[24] = mul(u[24], W16);
  #pragma unroll
  for (u32 k1 = 0; k1 < 5; ++k1) { fft5by(u, 5 * k1, 1, 25); }                      // output k2 of the k1-th DFT at u[5 k1 + k2]
  SWAP(u[1], u[5]);
  SWAP(u[2], u[10]);
  SWAP(u[3], u[15]);
  SWAP(u[4], u[20]);
  SWAP(u[7], u[11]);
  SWAP(u[8], u[16]);
  SWAP(u[9], u[21]);
  SWAP(u[13], u[17]);
  SWAP(u[14], u[22]);
  SWAP(u[19], u[23]);
}

#endif
