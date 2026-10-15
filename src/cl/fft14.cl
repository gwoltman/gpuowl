// Copyright (C) Mihai Preda and George Woltman

#include "fft7.cl"

#if FFT_FP64

// PFA(7*2): 76 FMA + 64 ADD
void OVERLOAD fft14(T2 *u) {
  fft7by(u, 0, 2, 14);
  fft7by(u, 7, 2, 14);
  for (int i = 0; i < 14; i += 2) { X2(u[i], u[(i + 7) % 14]); }

  // Fix order: 0 11 8 5 2 13 10 7 4 1 12 9 6 3

#define CYCLE(a, b, c) { T2 t = u[a]; u[a] = u[b]; u[b] = u[c]; u[c] = t; }
  CYCLE(1, 9, 11);
  CYCLE(2, 4, 8);
  CYCLE(3, 13, 5);
  CYCLE(6, 12, 10);
#undef CYCLE
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// PFA(7*2): 76 FMA + 64 ADD
void OVERLOAD fft14(F2 *u) {
  fft7by(u, 0, 2, 14);
  fft7by(u, 7, 2, 14);
  for (int i = 0; i < 14; i += 2) { X2(u[i], u[(i + 7) % 14]); }

  // Fix order: 0 11 8 5 2 13 10 7 4 1 12 9 6 3

#define CYCLE(a, b, c) { F2 t = u[a]; u[a] = u[b]; u[b] = u[c]; u[c] = t; }
  CYCLE(1, 9, 11);
  CYCLE(2, 4, 8);
  CYCLE(3, 13, 5);
  CYCLE(6, 12, 10);
#undef CYCLE
}

#endif
