// Copyright (C) Mihai Preda and George Woltman

#if FFT_FP64

#include "fft3.cl"
#include "fft4.cl"

// 24 FMA + 72 ADD
void fft12(T2 *u) {
  fft3by(u, 0, 4, 12);
  fft3by(u, 3, 4, 12);
  fft3by(u, 6, 4, 12);
  fft3by(u, 9, 4, 12);

  fft4by(u, 0, 3, 12);
  fft4by(u, 4, 3, 12);
  fft4by(u, 8, 3, 12);

  // Fix order: 0 7 2 9 4 11 6 1 8 3 10 5
  SWAP(u[1], u[7]);
  SWAP(u[3], u[9]);
  SWAP(u[5], u[11]);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

#include "fft3.cl"
#include "fft4.cl"

// 24 FMA + 72 ADD
void fft12(F2 *u) {
  fft3by(u, 0, 4, 12);
  fft3by(u, 3, 4, 12);
  fft3by(u, 6, 4, 12);
  fft3by(u, 9, 4, 12);

  fft4by(u, 0, 3, 12);
  fft4by(u, 4, 3, 12);
  fft4by(u, 8, 3, 12);

  // Fix order: 0 7 2 9 4 11 6 1 8 3 10 5
  SWAP(u[1], u[7]);
  SWAP(u[3], u[9]);
  SWAP(u[5], u[11]);
}

#endif
