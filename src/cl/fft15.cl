// Copyright (C) Mihai Preda and George Woltman

#if FFT_FP64

#include "fft3.cl"
#include "fft5.cl"

// FFT 15 using PFA ("Prime Factor Algorithm") in place and in order, see "An In-Place, In-Order Prime Factor Algorithm" by Burrus &
// Eschenbacher (1981): five radix-3 DFTs on u[b + 5k] then three radix-5 DFTs on u[b + 3k], with no twiddles and no final reordering.
// Their output maps differ from the standard DFT's -- radix-3 output k is X(2k mod 3), radix-5 output k is X(3k mod 5) -- but that is only
// a permutation of the outputs of fft3by / fft5by, free in registers.  30 FMA + 30 ADD + 3 * (18 FMA + 12 ADD) = 84 FMA + 66 ADD.
void fft15(T2 *u) {
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

#include "fft3.cl"
#include "fft5.cl"

// FFT 15 using PFA ("Prime Factor Algorithm") in place and in order, see "An In-Place, In-Order Prime Factor Algorithm" by Burrus &
// Eschenbacher (1981): five radix-3 DFTs on u[b + 5k] then three radix-5 DFTs on u[b + 3k], with no twiddles and no final reordering.
// Their output maps differ from the standard DFT's -- radix-3 output k is X(2k mod 3), radix-5 output k is X(3k mod 5) -- but that is only
// a permutation of the outputs of fft3by / fft5by, free in registers.  30 FMA + 30 ADD + 3 * (18 FMA + 12 ADD) = 84 FMA + 66 ADD.
void fft15(F2 *u) {
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
