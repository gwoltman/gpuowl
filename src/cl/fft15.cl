// Copyright (C) Mihai Preda and George Woltman

#include "fft3.cl"
#include "fft5.cl"

#if FFT_FP64

// FFT 15 using PFA ("Prime Factor Algorithm") in place and in order, see "An In-Place, In-Order Prime Factor Algorithm" by Burrus &
// Eschenbacher (1981): five radix-3 DFTs on u[b + 5k] then three radix-5 DFTs on u[b + 3k], with no twiddles and no final reordering.
// Their output maps differ from the standard DFT's -- radix-3 output k is X(2k mod 3), radix-5 output k is X(3k mod 5) -- but that is only
// a permutation of the outputs of fft3by / fft5by, free in registers.  30 FMA + 30 ADD + 3 * (18 FMA + 12 ADD) = 84 FMA + 66 ADD.
void OVERLOAD fft15(T2 *u) {
  #pragma unroll
  for (u32 b = 0; b < 15; b += 3) {
    fft3by(u, b, 5, 15);
    SWAP(u[(b + 5) % 15], u[(b + 10) % 15]);                 // outputs 1 and 2 exchanged
  }
  #pragma unroll
  for (u32 b = 0; b < 15; b += 5) {
    fft5by(u, b, 3, 15);
    T2 t = u[(b + 3) % 15];                                    // outputs 1, 2, 3, 4 become X(3), X(1), X(4), X(2)
    u[(b + 3) % 15] = u[(b + 9) % 15];
    u[(b + 9) % 15] = u[(b + 12) % 15];
    u[(b + 12) % 15] = u[(b + 6) % 15];
    u[(b + 6) % 15] = t;
  }
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// FFT 15 using PFA ("Prime Factor Algorithm") in place and in order, see "An In-Place, In-Order Prime Factor Algorithm" by Burrus &
// Eschenbacher (1981): five radix-3 DFTs on u[b + 5k] then three radix-5 DFTs on u[b + 3k], with no twiddles and no final reordering.
// Their output maps differ from the standard DFT's -- radix-3 output k is X(2k mod 3), radix-5 output k is X(3k mod 5) -- but that is only
// a permutation of the outputs of fft3by / fft5by, free in registers.  30 FMA + 30 ADD + 3 * (18 FMA + 12 ADD) = 84 FMA + 66 ADD.
void OVERLOAD fft15(F2 *u) {
  #pragma unroll
  for (u32 b = 0; b < 15; b += 3) {
    fft3by(u, b, 5, 15);
    SWAP(u[(b + 5) % 15], u[(b + 10) % 15]);                 // outputs 1 and 2 exchanged
  }
  #pragma unroll
  for (u32 b = 0; b < 15; b += 5) {
    fft5by(u, b, 3, 15);
    F2 t = u[(b + 3) % 15];                                    // outputs 1, 2, 3, 4 become X(3), X(1), X(4), X(2)
    u[(b + 3) % 15] = u[(b + 9) % 15];
    u[(b + 9) % 15] = u[(b + 12) % 15];
    u[(b + 12) % 15] = u[(b + 6) % 15];
    u[(b + 6) % 15] = t;
  }
}

#endif

/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

#if PFA == 15 && PFA_W61 != 1276162394648261926ull
#error PFA_W61 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-15 DFT, y_j = sum_k u_k w^(jk) with w = 37^((p-1)/15) (pfaRootOfUnity).
void OVERLOAD fft15(GF61 *u) {
  // 3 x 5 Good-Thomas, which needs no twiddles:  inputs n = 5 n1 + 3 n2, outputs k = 10 k1 + 6 k2 (mod 15).  The radix-3 DFTs use
  // the root w^5 and the radix-5 ones w^3:  the roots hardwired in fft3.cl and fft5.cl.  Each DFT runs on a small gathered array, which
  // keeps fewer values live than working in place on u[] (the FP versions' in-order form).
  GF61 b[15];
  #pragma unroll
  for (u32 n2 = 0; n2 < 5; ++n2) {
    GF61 t[3] = { u[3 * n2 % 15], u[(5 + 3 * n2) % 15], u[(10 + 3 * n2) % 15] };
    fft3(t);
    for (u32 k1 = 0; k1 < 3; ++k1) { b[3 * n2 + k1] = t[k1]; }
  }
  #pragma unroll
  for (u32 k1 = 0; k1 < 3; ++k1) {
    GF61 t[5] = { b[k1], b[3 + k1], b[6 + k1], b[9 + k1], b[12 + k1] };
    fft5(t);
    for (u32 k2 = 0; k2 < 5; ++k2) { u[(10 * k1 + 6 * k2) % 15] = t[k2]; }
  }
}

#endif
