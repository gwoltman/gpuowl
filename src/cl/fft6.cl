// Copyright (C) Mihai Preda & George Woltman

#include "fft3.cl"

#if FFT_FP64

// 12 FMA + 24 ADD
void OVERLOAD fft6(T2 *u) {
  X2(u[0], u[3]);
  X2(u[2], u[5]);
  X2(u[4], u[1]);

  fft3by(u, 0, 2, 6);
  fft3by(u, 3, 2, 6);

  // Fix order: 0 5 4 3 2 1
  SWAP(u[1], u[5]);
  SWAP(u[2], u[4]);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 12 FMA + 24 ADD
void OVERLOAD fft6(F2 *u) {
  X2(u[0], u[3]);
  X2(u[2], u[5]);
  X2(u[4], u[1]);

  fft3by(u, 0, 2, 6);
  fft3by(u, 3, 2, 6);

  // Fix order: 0 5 4 3 2 1
  SWAP(u[1], u[5]);
  SWAP(u[2], u[4]);
}

#endif
