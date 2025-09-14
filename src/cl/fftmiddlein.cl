// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "math.cl"
#include "fft-middle.cl"
#include "middle.cl"

#if FFT_FP64

KERNEL(IN_WG) fftMiddleIn(P(T2) out, CP(T2) in, Trig trig) {
  T2 u[MIDDLE];

  u32 SIZEY = IN_WG / IN_SIZEX;

  u32 N = WIDTH / IN_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;

  u32 startx = gx * IN_SIZEX;
  u32 starty = gy * SIZEY;

  u32 x = startx + mx;
  u32 y = starty + my;

  readMiddleInLine(u, in, y, x);

  middleMul2(u, x, y, 1, trig);

  fft_MIDDLE(u);

  middleMul(u, y, trig);

#if MIDDLE_IN_LDS_TRANSPOSE
  // Transpose the x and y values
  local T lds[IN_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);
  out += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out += mx * SIZEY + my;
#endif

  writeMiddleInLine(out, u, gy, gx);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

KERNEL(IN_WG) fftMiddleIn(P(T2) out, CP(T2) in, Trig trig) {
  F2 u[MIDDLE];

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 trigF2 = (TrigFP32) trig;

  u32 SIZEY = IN_WG / IN_SIZEX;

  u32 N = WIDTH / IN_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;

  u32 startx = gx * IN_SIZEX;
  u32 starty = gy * SIZEY;

  u32 x = startx + mx;
  u32 y = starty + my;

  readMiddleInLine(u, inF2, y, x);

  middleMul2(u, x, y, 1, trigF2);

  fft_MIDDLE(u);

  middleMul(u, y, trigF2);

#if MIDDLE_IN_LDS_TRANSPOSE
  // Transpose the x and y values
  local F lds[IN_WG / 2 * (MIDDLE <= 16 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);
  outF2 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  outF2 += mx * SIZEY + my;
#endif

  writeMiddleInLine(outF2, u, gy, gx);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

KERNEL(IN_WG) fftMiddleInGF31(P(T2) out, CP(T2) in, Trig trig) {
  GF31 u[MIDDLE];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 trig31 = (TrigGF31) (trig + DISTMTRIGGF31);

  u32 SIZEY = IN_WG / IN_SIZEX;

  u32 N = WIDTH / IN_SIZEX;
  
  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;

  u32 startx = gx * IN_SIZEX;
  u32 starty = gy * SIZEY;

  u32 x = startx + mx;
  u32 y = starty + my;

  readMiddleInLine(u, in31, y, x);

  middleMul2(u, x, y, trig31);

  fft_MIDDLE(u);

  middleMul(u, y, trig31);

#if MIDDLE_IN_LDS_TRANSPOSE
  // Transpose the x and y values
  local Z31 lds[IN_WG / 2 * (MIDDLE <= 16 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);
  out31 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out31 += mx * SIZEY + my;
#endif

  writeMiddleInLine(out31, u, gy, gx);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

KERNEL(IN_WG) fftMiddleInGF61(P(T2) out, CP(T2) in, Trig trig) {
  GF61 u[MIDDLE];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);

  u32 SIZEY = IN_WG / IN_SIZEX;

  u32 N = WIDTH / IN_SIZEX;
  
  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;

  u32 startx = gx * IN_SIZEX;
  u32 starty = gy * SIZEY;

  u32 x = startx + mx;
  u32 y = starty + my;

  readMiddleInLine(u, in61, y, x);

  middleMul2(u, x, y, trig61);

  fft_MIDDLE(u);

  middleMul(u, y, trig61);

#if MIDDLE_IN_LDS_TRANSPOSE
  // Transpose the x and y values
  local Z61 lds[IN_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);
  out61 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out61 += mx * SIZEY + my;
#endif

  writeMiddleInLine(out61, u, gy, gx);
}

#endif
