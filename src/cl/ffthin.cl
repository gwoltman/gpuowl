// Copyright (C) Mihai Preda

#include "base.cl"
#include "fftheight.cl"
#include "middle.cl"

// If not doing L2 stripes, process the lines in any order.
// If L2 striping, process lines output by fftMiddleIn.  fftMiddleIn outputs 2 * stripe_group_size * 16 * MIDDLE tailSquare lines.
u32 get_line_number(u32 base_lo) {
  u32 g = get_group_id(0);
#if L2_STRIPING
  // Old, simple L2 striping code
  // return g / (L2_STRIPING * 16) * WIDTH + base + g % (L2_STRIPING * 16);

  // Process stripe group base_lo or base_hi
  u32 base_hi = WIDTH - stripe_group_size * 16 - base_lo;
  u32 linesInOneStripe = 16 * MIDDLE;
  u32 stripe_group_size = L2_STRIPING;
  u32 linesInOneStripeGroup = stripe_group_size * linesInOneStripe;
  u32 base;
  if (g  linesInOneStripeGroup) base = base_lo;
  else base = base_hi, g -= linesInOneStripeGroup;
  return g / (L2_STRIPING * 16) * WIDTH + base + g % (L2_STRIPING * 16);
#else
  return g;
#endif
}

#if FFT_FP64

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHin(P(T2) out, CP(T2) in, u32 base, Trig smallTrig) {
  local T2 lds[LDS_BYTES / sizeof(T2)];
  const u32 H = ND / SMALL_HEIGHT;

  T2 u[NH];
  u32 line = get_line_number(base);
  u32 me = get_local_id(0);

  readTailFusedLine(in, u, line, me);

#if FFT_VARIANT_H != 0
  T2 w;
#elif NH == 8
  T2 w = fancyTrig_N(H * me);
#else
  T2 w = slowTrig_N(H * me, ND / NH);
#endif

  fft_HEIGHT(lds, u, smallTrig, w, 1, me);

  write(G_H, NH, u, out, SMALL_HEIGHT * transPos(line, MIDDLE, WIDTH));
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHin(P(T2) out, CP(T2) in, u32 base, Trig smallTrig) {
  local F2 lds[LDS_BYTES / sizeof(F2)];
  const u32 H = ND / SMALL_HEIGHT;

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;

  F2 u[NH];
  u32 line = get_line_number(base);
  u32 me = get_local_id(0);

  readTailFusedLine(inF2, u, line, me);

#if FFT_VARIANT_H != 0
  T2 w;
#elif NH == 8
  F2 w = fancyTrig_N(H * me);
#else
  F2 w = slowTrig_N(H * me, ND / NH);
#endif

  fft_HEIGHT(lds, u, smallTrigF2, 1, me);

  write(G_H, NH, u, outF2, SMALL_HEIGHT * transPos(line, MIDDLE, WIDTH));
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHinGF31(P(T2) out, CP(T2) in, u32 base, Trig smallTrig) {
  local GF31 lds[LDS_BYTES / sizeof(GF31)];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTHTRIGGF31);

  GF31 u[NH];
  u32 line = get_line_number(base);
  u32 me = get_local_id(0);

  readTailFusedLine(in31, u, line, me);

  fft_HEIGHT(lds, u, smallTrig31, 1, me);

  write(G_H, NH, u, out31, SMALL_HEIGHT * transPos(line, MIDDLE, WIDTH));
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHinGF61(P(T2) out, CP(T2) in, u32 base, Trig smallTrig) {
  local GF61 lds[LDS_BYTES / sizeof(GF61)];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTHTRIGGF61);

  GF61 u[NH];
  u32 line = get_line_number(base);
  u32 me = get_local_id(0);

  readTailFusedLine(in61, u, line, me);

  fft_HEIGHT(lds, u, smallTrig61, 1, me);

  write(G_H, NH, u, out61, SMALL_HEIGHT * transPos(line, MIDDLE, WIDTH));
}

#endif
