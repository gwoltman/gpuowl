// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "fft-middle.cl"
#include "middle.cl"

#if !INPLACE                  // Original implementation (not in place)

#if FFT_FP64

KERNEL(IN_WG) fftMiddleIn(P(T2) out, CP(T2) in, u32 base, Trig trig) {
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

#if FFT_TYPE == FFT64
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing FP64 data
#endif

  readMiddleInLine(u, in, y, x);

  middleMul2(u, x, y, 1, trig);

  fft_MIDDLE(u);

  middleMul(u, y, trig);

  dependentLaunch();       // Next kernel will be tailSquareFP64 which must dependentLaunchWait before reading data

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

KERNEL(IN_WG) fftMiddleIn(P(T2) out, CP(T2) in, u32 base, Trig trig) {
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

#if FFT_TYPE == FFT32
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing FP32 data
#endif

  readMiddleInLine(u, inF2, y, x);

  middleMul2(u, x, y, 1, trigF2);

  fft_MIDDLE(u);

  middleMul(u, y, trigF2);

  dependentLaunch();       // Next kernel will be tailSquareFP32 which must dependentLaunchWait before reading data

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

KERNEL(IN_WG) fftMiddleInGF31(P(T2) out, CP(T2) in, u32 base, Trig trig) {
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

#if FFT_TYPE == FFT31
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing GF31 data
#endif

  readMiddleInLine(u, in31, y, x);

  middleMul2(u, x, y, trig31);

  fft_MIDDLE(u);

  middleMul(u, y, trig31);

  dependentLaunch();       // Next kernel will be tailSquareGF31 which must dependentLaunchWait before reading data

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

KERNEL(IN_WG) fftMiddleInGF61(P(T2) out, CP(T2) in, u32 base, Trig trig) {
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

#if FFT_TYPE == FFT61
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing GF31 data
#endif

  readMiddleInLine(u, in61, y, x);

  middleMul2(u, x, y, trig61);

  fft_MIDDLE(u);

  middleMul(u, y, trig61);

  dependentLaunch();       // Next kernel will be tailSquareGF61 which must dependentLaunchWait before reading data

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




// fftMiddleIn processes lines output by fftP or carryFused.  Call this the x coordinate with range 0..WIDTH-1.  The y coordinate ranges from 0..MIDDLE*SMALL_HEIGHT-1.
// In place transpose processeses blocks of 16 x coordinates by 16 y coordinates.  fftMiddleIn processes processes MIDDLE blocks at a time.
// fftMiddleIn outputs lines for tailSquare.  The y coordinate from 0..SMALL_HEIGHT-1 is transposed into the x coordinate for tailSquare.
//
// fftMiddleIn can work on all the FFT data, in which case we can process blocks in any order.  Sequentially through memory by increasing x coordinates first might be best.
// More interestingly, fftMiddleIn can be configured to work on smaller amounts of FFT data in hopes that the data will stay in the L2 cache during the tailSquare and
// fftMiddleOut kernels.  I call this L2 striping.  In this case we process "columns of FFT data" by processing all the y coordinates to create complete lines for tailSquare.
// We must also output lines x and N-x for tailSquare handling of Hermetian symmetry.

#else           // in place transpose

// L2 striping processes both base_lo and base_hi (see Gpu.cpp) in one kernel call to reduce kernel launch overhead.  There is also some special handling required for the
// first and last stripe groups.  This results in a more complicated map group_id to compute the startx and starty coordinates.
#if L2_STRIPING
void map_striping_group_id(u32 base_lo, u32 g, u32 *startx, u32 *starty) {
  // Old, simple L2 striping code
  // u32 N = SMALL_HEIGHT / 16;
  // u32 starty = g % N * 16;
  // u32 startx = base + g / N * 16;

  // If MIDDLE is odd, the first fftMiddleIn must process the special N/2 tailSquare line from the WIDTH/2 stripe.
  // If MULTI_Q, the first fftMiddleIn in the second queue must also process the 3*WIDTH/4 stripe.
  u32 oneStripeKernelsToExecute = SMALL_HEIGHT / 16;
  if ((base_lo == 0 && (MIDDLE & 1)) || (MULTI_Q && base_lo == WIDTH / 4)) {
    if (g < oneStripeKernelsToExecute) {
      *startx = base_lo + WIDTH / 2;
      *starty = g * 16;
      return;
    }
    g -= oneStripeKernelsToExecute;
  }

  // Process stripe group base_lo.
  u32 stripe_group_size = L2_STRIPING;
  u32 kernelsToExecute = stripe_group_size * oneStripeKernelsToExecute;
  if (g < kernelsToExecute) {
    *startx = base_lo + g / oneStripeKernelsToExecute * 16;
    *starty = g % oneStripeKernelsToExecute * 16;
    return;
  }
  g -= kernelsToExecute;

  // Process stripe group base_hi.  The last group must take into account that the first stripe may have already been processed.
  u32 base_hi = WIDTH - stripe_group_size * 16 - base_lo;
  if ((base_hi == WIDTH / 2 && (MIDDLE & 1)) || (MULTI_Q && base_hi == 3 * WIDTH / 4)) base_hi += 16;
  *startx = base_hi + g / oneStripeKernelsToExecute * 16;
  *starty = g % oneStripeKernelsToExecute * 16;
}
#endif

#if FFT_FP64

KERNEL(256) fftMiddleIn(P(T2) out, P(T2) in, u32 base, Trig trig) {
  assert(out == in);
  T2 u[MIDDLE];

  u32 g = get_group_id(0);
#if L2_STRIPING
  u32 startx, starty;
  map_striping_group_id(base, g, &startx, &starty);
  u32 zerohack = (MIDDLE >= 16) ? 0 : g / 131072;  // Rocm optimizer goes bonkers if zerohack used when MIDDLE=16
#elif INPLACE == 1                                 // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
  u32 zerohack = g / 131072;                       // A super tiny benefit (much smaller than margin of error) on TitanV
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
  u32 zerohack = (MIDDLE >= 16) ? 0 : g / 131072;  // Rocm optimizer goes bonkers if zerohack used when MIDDLE=16
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

#if FFT_TYPE == FFT64
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing FP64 data
#endif

  readMiddleInLine(u, in, y, x);

  middleMul2(u, x, y, 1, trig);

  fft_MIDDLE(u);

  middleMul(u, y, trig);

  dependentLaunch();       // Next kernel will be tailSquareFP64 which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local T2 lds[256];
  middleShuffle(lds, u);

  writeMiddleInLine(in + zerohack, u, y, x);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

KERNEL(256) fftMiddleIn(P(T2) out, P(T2) in, u32 base, Trig trig) {
  assert(out == in);
  F2 u[MIDDLE];

  P(F2) inF2 = (P(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 trigF2 = (TrigFP32) trig;

  u32 g = get_group_id(0);
#if L2_STRIPING
  u32 startx, starty;
  map_striping_group_id(base, g, &startx, &starty);
  u32 zerohack = 0;                                // Need to test if g / 131072 is of any benefit
#elif INPLACE == 1                                 // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
  u32 zerohack = 0;                                // Need to test if g / 131072 is of any benefit
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
  u32 zerohack = (MIDDLE >= 16) ? 0 : g / 131072;  // Rocm optimizer goes bonkers if zerohack used when MIDDLE=16 (for FP64, FP32 untimed)
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

#if FFT_TYPE == FFT32
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing FP32 data
#endif

  readMiddleInLine(u, inF2, y, x);

  middleMul2(u, x, y, 1, trigF2);

  fft_MIDDLE(u);

  middleMul(u, y, trigF2);

  dependentLaunch();       // Next kernel will be tailSquareFP32 which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local F2 lds[256];
  middleShuffle(lds, u);

  writeMiddleInLine(inF2 + zerohack, u, y, x);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

KERNEL(256) fftMiddleInGF31(P(T2) out, P(T2) in, u32 base, Trig trig) {
  assert(out == in);
  GF31 u[MIDDLE];

  P(GF31) in31 = (P(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 trig31 = (TrigGF31) (trig + DISTMTRIGGF31);

  u32 g = get_group_id(0);
#if L2_STRIPING
  u32 startx, starty;
  map_striping_group_id(base, g, &startx, &starty);
  u32 zerohack = 0;                                // Need to test if g / 131072 is of any benefit
#elif INPLACE == 1                                 // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
  u32 zerohack = 0;                                // Need to test if g / 131072 is of any benefit
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
  u32 zerohack = (MIDDLE >= 16) ? 0 : g / 131072;  // Rocm optimizer goes bonkers if zerohack used when MIDDLE=16 (for FP64, GF31 untimed)
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

#if FFT_TYPE == FFT31
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing GF31 data
#endif

  readMiddleInLine(u, in31, y, x);

  middleMul2(u, x, y, trig31);

  fft_MIDDLE(u);

  middleMul(u, y, trig31);

  dependentLaunch();       // Next kernel will be tailSquareGF31 which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local GF31 lds[256];
  middleShuffle(lds, u);

  writeMiddleInLine(in31 + zerohack, u, y, x);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

KERNEL(256) fftMiddleInGF61(P(T2) out, P(T2) in, u32 base, Trig trig) {
  assert(out == in);
  GF61 u[MIDDLE];

  P(GF61) in61 = (P(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);

  u32 g = get_group_id(0);
#if L2_STRIPING
  u32 startx, starty;
  map_striping_group_id(base, g, &startx, &starty);
  u32 zerohack = 0;                                // Need to test if g / 131072 is of any benefit
#elif INPLACE == 1                                 // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
  u32 zerohack = 0;                                // Need to test if g / 131072 is of any benefit
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
  u32 zerohack = (MIDDLE >= 16) ? 0 : g / 131072;  // Rocm optimizer goes bonkers if zerohack used when MIDDLE=16 (for FP64, GF61 untimed)
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

#if FFT_TYPE == FFT61
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing GF31 data
#endif

  readMiddleInLine(u, in61, y, x);

  middleMul2(u, x, y, trig61);

  fft_MIDDLE(u);

  middleMul(u, y, trig61);

  dependentLaunch();       // Next kernel will be tailSquareGF61 which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local GF61 lds[256];
  middleShuffle(lds, u);

  writeMiddleInLine(in61 + zerohack, u, y, x);
}

#endif

#endif
