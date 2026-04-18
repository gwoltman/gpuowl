// Copyright (C) Mihai Preda

#pragma once

#include "fft4.cl"

#if FFT_FP64

T2 mul_t8_delayed(T2 a) { return U2(a.x - a.y, a.x + a.y); }
T2 mul_3t8_delayed(T2 a) { return U2(-(a.x + a.y), a.x - a.y); }
//#define X2_apply_delay(a, b) { T2 t = a; a = t + M_SQRT1_2 * b; b = t - M_SQRT1_2 * b; }
#define X2_apply_delay(a, b) { T2 t = a; a.x = fma(b.x, M_SQRT1_2, a.x); a.y = fma(b.y, M_SQRT1_2, a.y); b.x = fma(-M_SQRT1_2, b.x, t.x); b.y = fma(-M_SQRT1_2, b.y, t.y); }

void OVERLOAD fft4CoreSpecial(T2 *u) {
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);                                        // X2(u[1], u[3]); u[3] = mul_t4(u[3]);
  X2_apply_delay(u[0], u[1]);
  X2_apply_delay(u[2], u[3]);
}

void OVERLOAD fft8Core(T2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);   u[5] = mul_t8_delayed(u[5]);
  X2_mul_t4(u[2], u[6]);                                        // X2(u[2], u[6]);   u[6] = mul_t4(u[6]);
  X2(u[3], u[7]);   u[7] = mul_3t8_delayed(u[7]);
  fft4Core(u);
  fft4CoreSpecial(u + 4);
}

// 4 MUL + 52 ADD
void OVERLOAD fft8(T2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

F2 mul_t8_delayed(F2 a) { return U2(a.x - a.y, a.x + a.y); }
F2 mul_3t8_delayed(F2 a) { return U2(-(a.x + a.y), a.x - a.y); }
//#define X2_apply_delay(a, b) { F2 t = a; a = t + M_SQRT1_2 * b; b = t - M_SQRT1_2 * b; }
#define X2_apply_delay(a, b) { F2 t = a; a.x = fma(b.x, (float) M_SQRT1_2, a.x); a.y = fma(b.y, (float) M_SQRT1_2, a.y); b.x = fma((float) -M_SQRT1_2, b.x, t.x); b.y = fma((float) -M_SQRT1_2, b.y, t.y); }

void OVERLOAD fft4CoreSpecial(F2 *u) {
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);                                        // X2(u[1], u[3]); u[3] = mul_t4(u[3]);
  X2_apply_delay(u[0], u[1]);
  X2_apply_delay(u[2], u[3]);
}

void OVERLOAD fft8Core(F2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);   u[5] = mul_t8_delayed(u[5]);
  X2_mul_t4(u[2], u[6]);                                        // X2(u[2], u[6]);   u[6] = mul_t4(u[6]);
  X2(u[3], u[7]);   u[7] = mul_3t8_delayed(u[7]);
  fft4Core(u);
  fft4CoreSpecial(u + 4);
}

// 4 MUL + 52 ADD
void OVERLOAD fft8(F2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD fft8Core(GF31 *u) {
  X2(u[0], u[4]);
  X2_mul_t8(u[1], u[5]);
  X2_mul_t4(u[2], u[6]);
  X2_mul_3t8(u[3], u[7]);
  fft4Core(u);
  fft4Core(u + 4);
}

// 4 MUL + 52 ADD
void OVERLOAD fft8(GF31 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD fft4CoreSpecial1(GF61 *u) {         // Starts with u[0,1,2,3] in range of 0..2*M61+epsilon.
  X2q(&u[0], &u[2]);                              // X2(u[0], u[2]);  No reductions mod M61.  u[0,2] range is 0..4+, -2-..2+
  X2q_mul_t4(&u[1], &u[3]);                       // X2(u[1], u[3]);  u[3] = mul_t4(u[3]);    u[1,3] range is 0..4+, -2-..2+
  u[1] = optsubqu(u[1], 2, 2);                    // Partially reduce.  If u[1] > 2*M61, sub 2*M61.  u[1] now has range 0..2+
  u[3] = optsubqs(u[3], 0, 2);                    // Partially reduce.  If u[3] > 0*M61, sub 2*M61.  u[3] now has range -2-..0+
  X2q(&u[0], &u[1]);                              // X2(u[0], u[1]);  u[0,1] range is 0..6+, -2-..4+
  X2q(&u[2], &u[3]);                              // X2(u[2], u[3]);  u[2,3] range is -4-..2+, -2-..4+
  u[0] = modM61q(u[0], 0);
  u[1] = modM61q(u[1], 3);
  u[2] = modM61q(u[2], 5);
  u[3] = modM61q(u[3], 3);
}

void OVERLOAD fft4CoreSpecial2(GF61 *u) {         // Bottom half of an fft8.  Starts with u[0,1,2,3] in range of -1*M61-epsilon..1*M61+epsilon
  X2q(&u[0], &u[2]);                              // X2(u[0], u[2]);  No reductions mod M61.  u[0,2] range is -2-..2+, -2-..2+
  u[1] = mul_t8q(u[1], 3);                        // Perform delayed mul_t8.  u[1] range is 0..1+
  u[3] = mul_t8q(u[3], 3);                        // Perform delayed mul_t8.  u[3] range is 0..1+
  X2q_mul_t4(&u[1], &u[3]);                       // X2(u[1], u[3]);  u[3] = mul_t4(u[3]);    u[1,3] range is 0..2+, -1-..1+
  X2q(&u[0], &u[1]);                              // X2(u[0], u[1]);  u[0,1] range is -2-..4+, -4-..2+
  X2q(&u[2], &u[3]);                              // X2(u[2], u[3]);  u[2,3] range is -3-..3+, -3-..3+
  u[0] = modM61q(u[0], 3);
  u[1] = modM61q(u[1], 5);
  u[2] = modM61q(u[2], 4);
  u[3] = modM61q(u[3], 4);
}

void OVERLOAD fft8Core(GF61 *u) {                 // Starts with all u[i] values in range of 0..M61+epsilon (shorthand notation is 0..1+)
  X2q(&u[0], &u[4]);                              // X2(u[0], u[4]);  No reductions mod M61.  u[0,4] range is 0..2+, -1-..1+
  X2q(&u[1], &u[5]);                              // X2(u[1], u[5]);  Delay mul_t8 on u[5].   u[1,5] range is 0..2+, -1-..1+
  X2q_mul_t4(&u[2], &u[6]);                       // X2(u[2], u[6]);  u[6] = mul_t4(u[6]);    u[2,6] range is 0..2+, -1-..1+
  X2q_mul_t4(&u[3], &u[7]);                       // X2(u[3], u[7]);  u[7] = mul_t4(u[7]);    u[3,7] range is 0..2+, -1-..1+   Delay mul_t8 on u[7].
  fft4CoreSpecial1(u);
  fft4CoreSpecial2(u + 4);
}

void OVERLOAD fft8(GF61 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif
