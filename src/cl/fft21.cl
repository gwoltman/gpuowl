// Copyright (C) Mihai Preda and George Woltman

#include "fft3.cl"
#include "fft7.cl"

#if FFT_FP64

// FFT 21 using PFA ("Prime Factor Algorithm") in place and in order, as fft15.cl (Burrus & Eschenbacher 1981): seven radix-3 DFTs on
// u[b + 7k] then three radix-7 DFTs on u[b + 3k], with no twiddles and no final reordering.  The radix-3 outputs land in order (7 = 1 mod 3);
// radix-7 output k is X(5k mod 7), a 6-cycle of its outputs 1..6.  7 * (6 FMA + 6 ADD) + 3 * (38 FMA + 18 ADD) = 156 FMA + 96 ADD.
void OVERLOAD fft21(T2 *u) {
  #pragma unroll
  for (u32 b = 0; b < 21; b += 3) { fft3by(u, b, 7, 21); }
  #pragma unroll
  for (u32 b = 0; b < 21; b += 7) {
    fft7by(u, b, 3, 21);
    T2 t = u[(b + 3) % 21];                                    // outputs 1..6 become X(3), X(6), X(2), X(5), X(1), X(4)
    u[(b + 3) % 21] = u[(b + 9) % 21];
    u[(b + 9) % 21] = u[(b + 6) % 21];
    u[(b + 6) % 21] = u[(b + 18) % 21];
    u[(b + 18) % 21] = u[(b + 12) % 21];
    u[(b + 12) % 21] = u[(b + 15) % 21];
    u[(b + 15) % 21] = t;
  }
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// FFT 21 using PFA ("Prime Factor Algorithm") in place and in order, as fft15.cl (Burrus & Eschenbacher 1981): seven radix-3 DFTs on
// u[b + 7k] then three radix-7 DFTs on u[b + 3k], with no twiddles and no final reordering.  The radix-3 outputs land in order (7 = 1 mod 3);
// radix-7 output k is X(5k mod 7), a 6-cycle of its outputs 1..6.  7 * (6 FMA + 6 ADD) + 3 * (38 FMA + 18 ADD) = 156 FMA + 96 ADD.
void OVERLOAD fft21(F2 *u) {
  #pragma unroll
  for (u32 b = 0; b < 21; b += 3) { fft3by(u, b, 7, 21); }
  #pragma unroll
  for (u32 b = 0; b < 21; b += 7) {
    fft7by(u, b, 3, 21);
    F2 t = u[(b + 3) % 21];                                    // outputs 1..6 become X(3), X(6), X(2), X(5), X(1), X(4)
    u[(b + 3) % 21] = u[(b + 9) % 21];
    u[(b + 9) % 21] = u[(b + 6) % 21];
    u[(b + 6) % 21] = u[(b + 18) % 21];
    u[(b + 18) % 21] = u[(b + 12) % 21];
    u[(b + 12) % 21] = u[(b + 15) % 21];
    u[(b + 15) % 21] = t;
  }
}

#endif

/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

#if PFA == 21 && PFA_W31 != 1295308677u
#error PFA_W31 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-21 DFT, y_j = sum_k u_k w^(jk) with w = 7^((p-1)/21) (pfaRootOfUnity).
void OVERLOAD fft21(GF31 *u) {
  // 3 x 7 Good-Thomas, which needs no twiddles:  inputs n = 7 n1 + 3 n2, outputs k = 7 k1 + 15 k2 (mod 21).  The radix-3 DFTs use
  // the root w^7 and the radix-7 ones w^3:  the roots hardwired in fft3.cl and fft7.cl.  Each DFT runs on a small gathered array, as
  // the GF61 fft15.
  GF31 b[21];
  #pragma unroll
  for (u32 n2 = 0; n2 < 7; ++n2) {
    GF31 t[3] = { u[3 * n2 % 21], u[(7 + 3 * n2) % 21], u[(14 + 3 * n2) % 21] };
    fft3(t);
    for (u32 k1 = 0; k1 < 3; ++k1) { b[3 * n2 + k1] = t[k1]; }
  }
  #pragma unroll
  for (u32 k1 = 0; k1 < 3; ++k1) {
    GF31 t[7] = { b[k1], b[3 + k1], b[6 + k1], b[9 + k1], b[12 + k1], b[15 + k1], b[18 + k1] };
    fft7(t);
    for (u32 k2 = 0; k2 < 7; ++k2) { u[(7 * k1 + 15 * k2) % 21] = t[k2]; }
  }
}

#endif

/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

#if PFA == 21 && PFA_W61 != 712608377804628304ull
#error PFA_W61 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-21 DFT, y_j = sum_k u_k w^(jk) with w = 37^((p-1)/21) (pfaRootOfUnity).
void OVERLOAD fft21(GF61 *u) {
  // 3 x 7 Good-Thomas, which needs no twiddles:  inputs n = 7 n1 + 3 n2, outputs k = 7 k1 + 15 k2 (mod 21).  The radix-3 DFTs use
  // the root w^7 and the radix-7 ones w^3:  the roots hardwired in fft3.cl and fft7.cl.  Each DFT runs on a small gathered array, which
  // keeps fewer values live than working in place on u[] (the FP versions' in-order form).
  GF61 b[21];
  #pragma unroll
  for (u32 n2 = 0; n2 < 7; ++n2) {
    GF61 t[3] = { u[3 * n2 % 21], u[(7 + 3 * n2) % 21], u[(14 + 3 * n2) % 21] };
    fft3(t);
    for (u32 k1 = 0; k1 < 3; ++k1) { b[3 * n2 + k1] = t[k1]; }
  }
  #pragma unroll
  for (u32 k1 = 0; k1 < 3; ++k1) {
    GF61 t[7] = { b[k1], b[3 + k1], b[6 + k1], b[9 + k1], b[12 + k1], b[15 + k1], b[18 + k1] };
    fft7(t);
    for (u32 k2 = 0; k2 < 7; ++k2) { u[(7 * k1 + 15 * k2) % 21] = t[k2]; }
  }
}

#endif
