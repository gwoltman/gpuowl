// Copyright (C) Mihai Preda and George Woltman

#include "fft3.cl"
#include "fft11.cl"

#if FFT_FP64

// Radix-33 DFT, y_j = sum_k u_k w^(jk) with w = e^(2*pi*i/33).
void OVERLOAD fft33(T2 *u) {
  // 3 x 11 Good-Thomas, which needs no twiddles:  inputs n = 11 n1 + 3 n2, outputs k = 22 k1 + 12 k2 (mod 33).  The radix-3 DFTs use
  // the root w^11 and the radix-11 ones w^3:  the roots hardwired in fft3.cl and fft11.cl.  Each DFT runs on a small gathered array, as
  // the GF61 fft15 (fft11 has no in-place "by" form for the in-order PFA of fft15.cl/fft21.cl).
  T2 b[33];
  #pragma unroll
  for (u32 n2 = 0; n2 < 11; ++n2) {
    T2 t[3] = { u[3 * n2 % 33], u[(11 + 3 * n2) % 33], u[(22 + 3 * n2) % 33] };
    fft3(t);
    for (u32 k1 = 0; k1 < 3; ++k1) { b[3 * n2 + k1] = t[k1]; }
  }
  #pragma unroll
  for (u32 k1 = 0; k1 < 3; ++k1) {
    T2 t[11];
    for (u32 n2 = 0; n2 < 11; ++n2) { t[n2] = b[3 * n2 + k1]; }
    fft11(t);
    for (u32 k2 = 0; k2 < 11; ++k2) { u[(22 * k1 + 12 * k2) % 33] = t[k2]; }
  }
}

#endif


/**************************************************************************/
/*                  Similar to above, but for an FFT based on FP32        */
/**************************************************************************/

#if FFT_FP32

// Radix-33 DFT, y_j = sum_k u_k w^(jk) with w = e^(2*pi*i/33).
void OVERLOAD fft33(F2 *u) {
  // 3 x 11 Good-Thomas, which needs no twiddles:  inputs n = 11 n1 + 3 n2, outputs k = 22 k1 + 12 k2 (mod 33).  The radix-3 DFTs use
  // the root w^11 and the radix-11 ones w^3:  the roots hardwired in fft3.cl and fft11.cl.  Each DFT runs on a small gathered array, as
  // the GF61 fft15 (fft11 has no in-place "by" form for the in-order PFA of fft15.cl/fft21.cl).
  F2 b[33];
  #pragma unroll
  for (u32 n2 = 0; n2 < 11; ++n2) {
    F2 t[3] = { u[3 * n2 % 33], u[(11 + 3 * n2) % 33], u[(22 + 3 * n2) % 33] };
    fft3(t);
    for (u32 k1 = 0; k1 < 3; ++k1) { b[3 * n2 + k1] = t[k1]; }
  }
  #pragma unroll
  for (u32 k1 = 0; k1 < 3; ++k1) {
    F2 t[11];
    for (u32 n2 = 0; n2 < 11; ++n2) { t[n2] = b[3 * n2 + k1]; }
    fft11(t);
    for (u32 k2 = 0; k2 < 11; ++k2) { u[(22 * k1 + 12 * k2) % 33] = t[k2]; }
  }
}

#endif

/**************************************************************************/
/*               Similar to above, but for an NTT based on GF(M31^2)      */
/**************************************************************************/

#if NTT_GF31

#if PFA == 33 && PFA_W31 != 392483048u
#error PFA_W31 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-33 DFT, y_j = sum_k u_k w^(jk) with w = 7^((p-1)/33) (pfaRootOfUnity).
void OVERLOAD fft33(GF31 *u) {
  // 3 x 11 Good-Thomas, which needs no twiddles:  inputs n = 11 n1 + 3 n2, outputs k = 22 k1 + 12 k2 (mod 33).  The radix-3 DFTs use
  // the root w^11 and the radix-11 ones w^3:  the roots hardwired in fft3.cl and fft11.cl.  Each DFT runs on a small gathered array, as
  // the GF61 fft15 (fft11 has no in-place "by" form for the in-order PFA of fft15.cl/fft21.cl).
  GF31 b[33];
  #pragma unroll
  for (u32 n2 = 0; n2 < 11; ++n2) {
    GF31 t[3] = { u[3 * n2 % 33], u[(11 + 3 * n2) % 33], u[(22 + 3 * n2) % 33] };
    fft3(t);
    for (u32 k1 = 0; k1 < 3; ++k1) { b[3 * n2 + k1] = t[k1]; }
  }
  #pragma unroll
  for (u32 k1 = 0; k1 < 3; ++k1) {
    GF31 t[11];
    for (u32 n2 = 0; n2 < 11; ++n2) { t[n2] = b[3 * n2 + k1]; }
    fft11(t);
    for (u32 k2 = 0; k2 < 11; ++k2) { u[(22 * k1 + 12 * k2) % 33] = t[k2]; }
  }
}

#endif

/**************************************************************************/
/*               Similar to above, but for an NTT based on GF(M61^2)      */
/**************************************************************************/

#if NTT_GF61

#if PFA == 33 && PFA_W61 != 148397494762984903ull
#error PFA_W61 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-33 DFT, y_j = sum_k u_k w^(jk) with w = 37^((p-1)/33) (pfaRootOfUnity).
void OVERLOAD fft33(GF61 *u) {
  // 3 x 11 Good-Thomas, which needs no twiddles:  inputs n = 11 n1 + 3 n2, outputs k = 22 k1 + 12 k2 (mod 33).  The radix-3 DFTs use
  // the root w^11 and the radix-11 ones w^3:  the roots hardwired in fft3.cl and fft11.cl.  Each DFT runs on a small gathered array, as
  // the GF61 fft15 (fft11 has no in-place "by" form for the in-order PFA of fft15.cl/fft21.cl).
  GF61 b[33];
  #pragma unroll
  for (u32 n2 = 0; n2 < 11; ++n2) {
    GF61 t[3] = { u[3 * n2 % 33], u[(11 + 3 * n2) % 33], u[(22 + 3 * n2) % 33] };
    fft3(t);
    for (u32 k1 = 0; k1 < 3; ++k1) { b[3 * n2 + k1] = t[k1]; }
  }
  #pragma unroll
  for (u32 k1 = 0; k1 < 3; ++k1) {
    GF61 t[11];
    for (u32 n2 = 0; n2 < 11; ++n2) { t[n2] = b[3 * n2 + k1]; }
    fft11(t);
    for (u32 k2 = 0; k2 < 11; ++k2) { u[(22 * k1 + 12 * k2) % 33] = t[k2]; }
  }
}

#endif
