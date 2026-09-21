// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "fftheight.cl"

#define INCLUDE_FILE "tailutil.cl"
#include "expand.cl"
#define INCLUDE_FILE "middle.cl"
#include "expand.cl"

// If not doing L2 stripes, process the lines in any order.
// If L2 striping, process lines output by fftMiddleIn.  fftMiddleIn outputs 16 * MIDDLE tailSquare lines.
u32 get_line_number(u32 base) {
  u32 g = get_group_id(0);
#if !SINGLE_KERNEL
#if L2_STRIPING
  if (base == 0) g = g + 1;
#else
  g = g + 1;
#endif
#endif
#if L2_STRIPING
  // Old, simple L2 striping code
  // return get_group_id(1) * WIDTH + base + g;

  // Process all lines from low half of base_lo stripe group.  One stripe group is stripe_group_size * 16 * MIDDLE lines.
  u32 base_lo = base;
  u32 stripe_group_size = L2_STRIPING;
  u32 half_size = (MIDDLE + 1) / 2;  // For base_lo, round odd middles up.
  u32 kernelsToExecute = half_size * stripe_group_size * 16;
  if (g < kernelsToExecute) return g / (stripe_group_size * 16) * WIDTH + base_lo + g % (stripe_group_size * 16);
  g -= kernelsToExecute;

  // Process lines from low half of base_hi stripe group.  One stripe group is stripe_group_size * 16 * MIDDLE lines.
  // The first line in the base_hi stripe group is not ready for processing (except for the last group).
  u32 base_hi = WIDTH - stripe_group_size * 16 - base_lo;
  if (base_hi != WIDTH / 2) base_hi++;   // Skip first line in base_hi (usually)
  half_size = MIDDLE / 2;                // For base_hi, round odd middles up.
  return g % half_size * WIDTH + base_hi + g / half_size;
#else
  return g;
#endif
}

#if FFT_FP64

// Handle the final multiplication step on a pair of complex numbers.  Swap real and imaginary results for the inverse FFT.
// We used to conjugate the results, but swapping real and imaginary can save some negations in carry propagation.

void OVERLOAD onePairMul(T2* pa, T2* pb, T2* pc, T2* pd, T2 t_squared) {
  T2 a = *pa, b = *pb, c = *pc, d = *pd;

  X2conjb(a, b);
  X2conjb(c, d);

  *pa = cfma(a, c, cmul(cmul(b, d), -t_squared));
  *pb = cfma(b, c, cmul(a, d));

  X2_conjb(*pa, *pb);

  *pa = SWAP_XY(*pa), *pb = SWAP_XY(*pb);
}

void OVERLOAD pairMul(u32 N, T2 *u, T2 *v, T2 *p, T2 *q, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(2 * foo2(u[i], p[i]));
      v[i] = SWAP_XY(4 * cmul(v[i], q[i]));
    } else {
      onePairMul(&u[i], &v[i], &p[i], &q[i], base_squared);
    }

    if (N == NH) {
      onePairMul(&u[i+NH/2], &v[i+NH/2], &p[i+NH/2], &q[i+NH/2], -base_squared);
    }

    T2 new_base_squared = mul_t4(base_squared);
    onePairMul(&u[i+NH/4], &v[i+NH/4], &p[i+NH/4], &q[i+NH/4], new_base_squared);

    if (N == NH) {
      onePairMul(&u[i+3*NH/4], &v[i+3*NH/4], &p[i+3*NH/4], &q[i+3*NH/4], -new_base_squared);
    }
  }
}

#if !SINGLE_KERNEL
// The kernel tailMulZero handles the special cases in tailMul, i.e. the lines 0 and H/2
// This kernel is launched with 2 workgroups (handling line 0, resp. H/2)
KERNEL(G_H) tailMulZero(P(T2) out, CP(T2) in, CP(T2) a, Trig smallTrig) {
  local T2 lds[LDS_BYTES(1) / sizeof(T2)];
  LDSinit(lds, 1);

  T2 u[NH], p[NH];
  const u32 H = ND / SMALL_HEIGHT;

  // This kernel in executed in two workgroups.
  u32 which = get_group_id(0);
  assert(which < 2);

  u32 line = which ? (H/2) : 0;
  u32 memline = transPos(line, MIDDLE, WIDTH);
  u32 me = get_local_id(0);

  dependentLaunch();       // Next kernel will be tailMulFP64 which must dependentLaunchWait before reading data from fftMiddleInFP64
  dependentLaunchWait();   // Previous kernel was fftMiddleInFP64 that launched dependents before writing FP64 data

  readTailFusedLine(in, u, line, me);

#if FFT_VARIANT_H != 0
  T2 w;
#elif NH == 8
  T2 w = fancyTrig_N(H * me);
#else
  T2 w = slowTrig_N(H * me, ND / NH);
#endif

#if MUL_LOW
  read(G_H, NH, p, a, memline * SMALL_HEIGHT);
  fft_HEIGHT1(lds, u, smallTrig, w, 1, me);
#else
  readTailFusedLine(a, p, line, me);
  fft_HEIGHT1(lds, u, smallTrig, w, 1, me);
  fft_HEIGHT1(lds, p, smallTrig, w, 1, me);
#endif

  T2 trig = slowTrig_N(line + me * H, ND / NH);

  reverse(lds, u + NH/2, !which);
  reverse(lds, p + NH/2, !which);
  pairMul(NH/2, u,   u + NH/2, p, p + NH/2, trig, !which);
  reverse(lds, u + NH/2, !which);

  fft_HEIGHT1(lds, u, smallTrig, w, 1, me);
  writeTailFusedLine(u, out, memline, me);
}
#endif

#if SINGLE_WIDE

KERNEL_CAP(G_H) tailMul(P(T2) out, CP(T2) in, CP(T2) a, u32 base, Trig smallTrig) {
  local T2 lds[LDS_BYTES(1) / sizeof(T2)];
  LDSinit(lds, 1);

  const u32 H = ND / SMALL_HEIGHT;

  T2 u[NH], v[NH];
  T2 p[NH], q[NH];

  u32 line1 = get_line_number(base);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  dependentLaunchWait();   // Previous kernel was fftMiddleInFP64 that launched dependents before writing FP64 data

  u32 me = get_local_id(0);
  readTailFusedLine(in, u, line1, me);
  readTailFusedLine(in, v, line2, me);

#if FFT_VARIANT_H != 0
  T2 w;
#elif NH == 8
  T2 w = fancyTrig_N(H * me);
#else
  T2 w = slowTrig_N(H * me, ND / NH);
#endif

#if MUL_LOW
  read(G_H, NH, p, a, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a, memline2 * SMALL_HEIGHT);
  fft_HEIGHT1(lds, u, smallTrig, w, 1, me);
  fft_HEIGHT1(lds, v, smallTrig, w, 1, me);
#else
  readTailFusedLine(a, p, line1, me);
  readTailFusedLine(a, q, line2, me);
  fft_HEIGHT1(lds, u, smallTrig, w, 1, me);
  fft_HEIGHT1(lds, v, smallTrig, w, 1, me);
  fft_HEIGHT1(lds, p, smallTrig, w, 1, me);
  fft_HEIGHT1(lds, q, smallTrig, w, 1, me);
#endif

  T2 trig = slowTrig_N(line1 + me * H, ND / NH);

#if SINGLE_KERNEL
  if (line1 == 0) {
    reverse(lds, u + NH/2, true);
    reverse(lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig, true);
    reverse(lds, u + NH/2, true);

    T2 trig2 = cmulFancy(trig, TAILT);
    reverse(lds, v + NH/2, false);
    reverse(lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, trig2, false);
    reverse(lds, v + NH/2, false);
  } else {
#else
  if (1) {
#endif
    reverseLine(lds, v);
    reverseLine(lds, q);
    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(lds, v);
  }

  dependentLaunch();       // Next kernel will be fftMiddleOutFP64 which must dependentLaunchWait before reading data

  fft_HEIGHT2(lds, v, smallTrig, w, 1, me);
  fft_HEIGHT2(lds, u, smallTrig, w, 1, me);
  writeTailFusedLine(v, out, memline2, me);
  writeTailFusedLine(u, out, memline1, me);
}


//
// Create a kernel that uses a double-wide workgroup (u in half the workgroup, v in the other half)
// We hope to get better occupancy with the reduced register usage
//

#else

// Special pairMul for double-wide line 0: both halves compute their own self-pairing (u with p),
// there is no cross-half data since line_u == 0 and line_v == H/2 both pair with themselves.
void OVERLOAD pairMul2_special(T2 *u, T2 *p, T2 base_squared) {
  u32 me = get_local_id(0);
  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (i == 0 && me == 0) {
      u[0] = SWAP_XY(2 * foo2(u[0], p[0]));
      u[NH/2] = SWAP_XY(4 * cmul(u[NH/2], p[NH/2]));
    } else {
      onePairMul(&u[i], &u[NH/2+i], &p[i], &p[NH/2+i], base_squared);
    }
    T2 new_base_squared = mul_t4(base_squared);
    onePairMul(&u[i+NH/4], &u[NH/2+i+NH/4], &p[i+NH/4], &p[NH/2+i+NH/4], new_base_squared);
  }
}

KERNEL_CAP(G_H * 2) tailMul(P(T2) out, CP(T2) in, CP(T2) a, u32 base, Trig smallTrig) {
  local T2 lds[LDS_BYTES(2) / sizeof(T2)];
  LDSinit(lds, 2);

  const u32 H = ND / SMALL_HEIGHT;

  T2 u[NH], p[NH];

  u32 line_u = get_line_number(base);
  u32 line_v = line_u ? H - line_u : (H / 2);
  u32 me = get_local_id(0);
  u32 lowMe = me % G_H;  // lane-id in one of the two halves (half-workgroups).

  // We're going to call the halves "first-half" and "second-half".
  bool isSecondHalf = me >= G_H;

  u32 line = !isSecondHalf ? line_u : line_v;
  u32 memline = transPos(line, MIDDLE, WIDTH);

  dependentLaunchWait();   // Previous kernel was fftMiddleInFP64 that launched dependents before writing FP64 data

  // Read line u (own half's line) and p (own half's multiplier line)
  readTailFusedLine(in, u, line, lowMe);

#if FFT_VARIANT_H != 0
  T2 w;
#elif NH == 8
  T2 w = fancyTrig_N(H * lowMe);
#else
  T2 w = slowTrig_N(H * lowMe, ND / NH);
#endif

#if MUL_LOW
  read(G_H, NH, p, a, memline * SMALL_HEIGHT, lowMe);
  fft_HEIGHT1(lds, u, smallTrig, w, 2, lowMe);
#else
  readTailFusedLine(a, p, line, lowMe);
  fft_HEIGHT1(lds, u, smallTrig, w, 2, lowMe);
  fft_HEIGHT1(lds, p, smallTrig, w, 2, lowMe);
#endif

  T2 trig = slowTrig_N(line + H * lowMe, ND / NH * 2);

#if SINGLE_KERNEL
  // Line 0 and H/2 are special: they pair with themselves, line 0 is offseted by 1.
  if (line_u == 0) {
    reverse2(lds, u);
    reverse2(lds, p);
    pairMul2_special(u, p, trig);
    reverse2(lds, u);
  }
  else {
#else
  if (1) {
#endif
    revCrossLine(lds, u);
    revCrossLine(lds, p);
    pairMul(NH/2, u, u + NH/2, p, p + NH/2, trig, false);
    revCrossLine(lds, u);
  }

  dependentLaunch();       // Next kernel will be fftMiddleOutFP64 which must dependentLaunchWait before reading data

  fft_HEIGHT2(lds, u, smallTrig, w, 2, lowMe);

  // Write line u (own half's line)
  writeTailFusedLine(u, out, memline, lowMe);
}

#endif

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// Handle the final multiplication step on a pair of complex numbers.  Swap real and imaginary results for the inverse FFT.
// We used to conjugate the results, but swapping real and imaginary can save some negations in carry propagation.

void OVERLOAD onePairMul(F2* pa, F2* pb, F2* pc, F2* pd, F2 t_squared) {
  F2 a = *pa, b = *pb, c = *pc, d = *pd;
  X2conjb(a, b);
  X2conjb(c, d);
  *pa = cfma(a, c, cmul(cmul(b, d), -t_squared));
  *pb = cfma(b, c, cmul(a, d));
  X2_conjb(*pa, *pb);
  *pa = SWAP_XY(*pa), *pb = SWAP_XY(*pb);
}

void OVERLOAD pairMul(u32 N, F2 *u, F2 *v, F2 *p, F2 *q, F2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(2 * foo2(u[i], p[i]));
      v[i] = SWAP_XY(4 * cmul(v[i], q[i]));
    } else {
      onePairMul(&u[i], &v[i], &p[i], &q[i], base_squared);
    }

    if (N == NH) {
      onePairMul(&u[i+NH/2], &v[i+NH/2], &p[i+NH/2], &q[i+NH/2], -base_squared);
    }

    F2 new_base_squared = mul_t4(base_squared);
    onePairMul(&u[i+NH/4], &v[i+NH/4], &p[i+NH/4], &q[i+NH/4], new_base_squared);

    if (N == NH) {
      onePairMul(&u[i+3*NH/4], &v[i+3*NH/4], &p[i+3*NH/4], &q[i+3*NH/4], -new_base_squared);
    }
  }
}

#if !SINGLE_KERNEL
// The kernel tailMulZero handles the special cases in tailMul, i.e. the lines 0 and H/2
// This kernel is launched with 2 workgroups (handling line 0, resp. H/2)
KERNEL(G_H) tailMulZero(P(T2) out, CP(T2) in, CP(T2) a, Trig smallTrig) {
  local F2 lds[LDS_BYTES(1) / sizeof(F2)];
  LDSinit(lds, 1);

  F2 u[NH], p[NH];
  const u32 H = ND / SMALL_HEIGHT;

  CP(F2) inF2 = (CP(F2)) in;
  CP(F2) aF2 = (CP(F2)) a;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;

  // This kernel in executed in two workgroups.
  u32 which = get_group_id(0);
  assert(which < 2);

  u32 line = which ? (H/2) : 0;
  u32 memline = transPos(line, MIDDLE, WIDTH);
  u32 me = get_local_id(0);

  dependentLaunch();       // Next kernel will be tailMulFP32 which must dependentLaunchWait before reading data from fftMiddleInFP32
  dependentLaunchWait();   // Previous kernel was fftMiddleInFP32 that launched dependents before writing FP32 data

  readTailFusedLine(inF2, u, line, me);

#if MUL_LOW
  read(G_H, NH, p, aF2, memline * SMALL_HEIGHT);
  fft_HEIGHT1(lds, u, smallTrigF2, 1, me);
#else
  readTailFusedLine(aF2, p, line, me);
  fft_HEIGHT1(lds, u, smallTrigF2, 1, me);
  fft_HEIGHT1(lds, p, smallTrigF2, 1, me);
#endif

  F2 trig = slowTrig_N(line + me * H, ND / NH);

  reverse(lds, u + NH/2, !which);
  reverse(lds, p + NH/2, !which);
  pairMul(NH/2, u,   u + NH/2, p, p + NH/2, trig, !which);
  reverse(lds, u + NH/2, !which);

  fft_HEIGHT1(lds, u, smallTrigF2, 1, me);
  writeTailFusedLine(u, outF2, memline, me);
}
#endif

#if SINGLE_WIDE

KERNEL_CAP(G_H) tailMul(P(T2) out, CP(T2) in, CP(T2) a, u32 base, Trig smallTrig) {
  local F2 lds[LDS_BYTES(1) / sizeof(F2)];
  LDSinit(lds, 1);

  const u32 H = ND / SMALL_HEIGHT;

  CP(F2) inF2 = (CP(F2)) in;
  CP(F2) aF2 = (CP(F2)) a;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;

  F2 u[NH], v[NH];
  F2 p[NH], q[NH];

  u32 line1 = get_line_number(base);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  dependentLaunchWait();   // Previous kernel was fftMiddleInFP32 that launched dependents before writing FP32 data

  u32 me = get_local_id(0);
  readTailFusedLine(inF2, u, line1, me);
  readTailFusedLine(inF2, v, line2, me);

#if MUL_LOW
  read(G_H, NH, p, aF2, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, aF2, memline2 * SMALL_HEIGHT);
  fft_HEIGHT1(lds, u, smallTrigF2, 1, me);
  fft_HEIGHT1(lds, v, smallTrigF2, 1, me);
#else
  readTailFusedLine(aF2, p, line1, me);
  readTailFusedLine(aF2, q, line2, me);
  fft_HEIGHT1(lds, u, smallTrigF2, 1, me);
  fft_HEIGHT1(lds, v, smallTrigF2, 1, me);
  fft_HEIGHT1(lds, p, smallTrigF2, 1, me);
  fft_HEIGHT1(lds, q, smallTrigF2, 1, me);
#endif

  F2 trig = slowTrig_N(line1 + me * H, ND / NH);

#if SINGLE_KERNEL
  if (line1 == 0) {
    reverse(lds, u + NH/2, true);
    reverse(lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig, true);
    reverse(lds, u + NH/2, true);

    F2 trig2 = cmulFancy(trig, TAILT);
    reverse(lds, v + NH/2, false);
    reverse(lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, trig2, false);
    reverse(lds, v + NH/2, false);
  } else {
#else
  if (1) {
#endif
    reverseLine(lds, v);
    reverseLine(lds, q);
    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(lds, v);
  }

  dependentLaunch();       // Next kernel will be fftMiddleOutFP32 which must dependentLaunchWait before reading data

  fft_HEIGHT2(lds, v, smallTrigF2, 1, me);
  fft_HEIGHT2(lds, u, smallTrigF2, 1, me);
  writeTailFusedLine(v, outF2, memline2, me);
  writeTailFusedLine(u, outF2, memline1, me);
}


//
// Create a kernel that uses a double-wide workgroup (u in half the workgroup, v in the other half)
// We hope to get better occupancy with the reduced register usage
//

#else

// Special pairMul for double-wide line 0: both halves compute their own self-pairing (u with p),
// there is no cross-half data since line_u == 0 and line_v == H/2 both pair with themselves.
void OVERLOAD pairMul2_special(F2 *u, F2 *p, F2 base_squared) {
  u32 me = get_local_id(0);
  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (i == 0 && me == 0) {
      u[0] = SWAP_XY(2 * foo2(u[0], p[0]));
      u[NH/2] = SWAP_XY(4 * cmul(u[NH/2], p[NH/2]));
    } else {
      onePairMul(&u[i], &u[NH/2+i], &p[i], &p[NH/2+i], base_squared);
    }
    F2 new_base_squared = mul_t4(base_squared);
    onePairMul(&u[i+NH/4], &u[NH/2+i+NH/4], &p[i+NH/4], &p[NH/2+i+NH/4], new_base_squared);
  }
}

KERNEL_CAP(G_H * 2) tailMul(P(T2) out, CP(T2) in, CP(T2) a, u32 base, Trig smallTrig) {
  local F2 lds[LDS_BYTES(2) / sizeof(F2)];
  LDSinit(lds, 2);

  const u32 H = ND / SMALL_HEIGHT;

  CP(F2) inF2 = (CP(F2)) in;
  CP(F2) aF2 = (CP(F2)) a;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;

  F2 u[NH], p[NH];

  u32 line_u = get_line_number(base);
  u32 line_v = line_u ? H - line_u : (H / 2);
  u32 me = get_local_id(0);
  u32 lowMe = me % G_H;  // lane-id in one of the two halves (half-workgroups).

  // We're going to call the halves "first-half" and "second-half".
  bool isSecondHalf = me >= G_H;

  u32 line = !isSecondHalf ? line_u : line_v;
  u32 memline = transPos(line, MIDDLE, WIDTH);

  dependentLaunchWait();   // Previous kernel was fftMiddleInFP32 that launched dependents before writing FP32 data

  // Read line u (own half's line) and p (own half's multiplier line)
  readTailFusedLine(inF2, u, line, lowMe);

#if MUL_LOW
  read(G_H, NH, p, aF2, memline * SMALL_HEIGHT, lowMe);
  fft_HEIGHT1(lds, u, smallTrigF2, 2, lowMe);
#else
  readTailFusedLine(aF2, p, line, lowMe);
  fft_HEIGHT1(lds, u, smallTrigF2, 2, lowMe);
  fft_HEIGHT1(lds, p, smallTrigF2, 2, lowMe);
#endif

  F2 trig = slowTrig_N(line + H * lowMe, ND / NH * 2);

#if SINGLE_KERNEL
  // Line 0 and H/2 are special: they pair with themselves, line 0 is offseted by 1.
  if (line_u == 0) {
    reverse2(lds, u);
    reverse2(lds, p);
    pairMul2_special(u, p, trig);
    reverse2(lds, u);
  }
  else {
#else
  if (1) {
#endif
    revCrossLine(lds, u);
    revCrossLine(lds, p);
    pairMul(NH/2, u, u + NH/2, p, p + NH/2, trig, false);
    revCrossLine(lds, u);
  }

  dependentLaunch();       // Next kernel will be fftMiddleOutFP32 which must dependentLaunchWait before reading data

  fft_HEIGHT2(lds, u, smallTrigF2, 2, lowMe);

  // Write line u (own half's line)
  writeTailFusedLine(u, outF2, memline, lowMe);
}

#endif

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD onePairMul(GF31* pa, GF31* pb, GF31* pc, GF31* pd, GF31 t_squared) {
  GF31 a = *pa, b = *pb, c = *pc, d = *pd;
  X2conjb(a, b);
  X2conjb(c, d);
  GF31 ac = cmul(a, c);
  GF31 bd = cmul(b, d);
  *pa = sub(ac, cmul(bd, t_squared));
  *pb = sub(sub(cmul(add(a, b), add(c, d)), ac), bd);
  X2_conjb(*pa, *pb);
  *pa = SWAP_XY(*pa), *pb = SWAP_XY(*pb);
}

void OVERLOAD pairMul(u32 N, GF31 *u, GF31 *v, GF31 *p, GF31 *q, GF31 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(mul2(foo2(u[i], p[i])));
      v[i] = SWAP_XY(shl(cmul(v[i], q[i]), 2));
   } else {
      onePairMul(&u[i], &v[i], &p[i], &q[i], base_squared);
    }

    if (N == NH) {
      onePairMul(&u[i+NH/2], &v[i+NH/2], &p[i+NH/2], &q[i+NH/2], neg(base_squared));
    }

    GF31 new_base_squared = mul_t4(base_squared);
    onePairMul(&u[i+NH/4], &v[i+NH/4], &p[i+NH/4], &q[i+NH/4], new_base_squared);

    if (N == NH) {
      onePairMul(&u[i+3*NH/4], &v[i+3*NH/4], &p[i+3*NH/4], &q[i+3*NH/4], neg(new_base_squared));
    }
  }
}

#if !SINGLE_KERNEL
// The kernel tailMulZeroGF31 handles the special cases in tailMulGF31, i.e. the lines 0 and H/2
// This kernel is launched with 2 workgroups (handling line 0, resp. H/2)
KERNEL(G_H) tailMulZeroGF31(P(T2) out, CP(T2) in, CP(T2) a, Trig smallTrig) {
  local GF31 lds[LDS_BYTES(1) / sizeof(GF31)];
  LDSinit(lds, 1);

  GF31 u[NH], p[NH];
  const u32 H = ND / SMALL_HEIGHT;

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF31) a31 = (CP(GF31)) (a + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTHTRIGGF31);

  // This kernel in executed in two workgroups.
  u32 which = get_group_id(0);
  assert(which < 2);

  u32 line = which ? (H/2) : 0;
  u32 memline = transPos(line, MIDDLE, WIDTH);
  u32 me = get_local_id(0);

  dependentLaunch();       // Next kernel will be tailMulGF31 which must dependentLaunchWait before reading data from fftMiddleInGF31
  dependentLaunchWait();   // Previous kernel was fftMiddleInGF31 that launched dependents before writing GF31 data

  readTailFusedLine(in31, u, line, me);

#if MUL_LOW
  read(G_H, NH, p, a31, memline * SMALL_HEIGHT);
  fft_HEIGHT1(lds, u, smallTrig31, 1, me);
#else
  readTailFusedLine(a31, p, line, me);
  fft_HEIGHT1(lds, u, smallTrig31, 1, me);
  fft_HEIGHT1(lds, p, smallTrig31, 1, me);
#endif

  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
#if TAIL_TRIGS31 >= 1
  GF31 trig = TFLOAD(&smallTrig31[height_trigs + me]);
#if SINGLE_WIDE
  GF31 mult = TSLOAD(&smallTrig31[height_trigs + G_H + line]);
#else
  GF31 mult = TSLOAD(&smallTrig31[height_trigs + G_H + which]);
#endif
  trig = cmul(trig, mult);
#else
#if SINGLE_WIDE
  GF31 trig = TOLOAD(&smallTrig31[height_trigs + line*G_H + me]);
#else
  GF31 trig = TOLOAD(&smallTrig31[height_trigs + which*G_H + me]);
#endif
#endif

  reverse(lds, u + NH/2, !which);
  reverse(lds, p + NH/2, !which);
  pairMul(NH/2, u,   u + NH/2, p, p + NH/2, trig, !which);
  reverse(lds, u + NH/2, !which);

  fft_HEIGHT2(lds, u, smallTrig31, 1, me);
  writeTailFusedLine(u, out31, memline, me);
}
#endif

#if SINGLE_WIDE

KERNEL_CAP(G_H) tailMulGF31(P(T2) out, CP(T2) in, CP(T2) a, u32 base, Trig smallTrig) {
  local GF31 lds[LDS_BYTES(1) / sizeof(GF31)];
  LDSinit(lds, 1);

  const u32 H = ND / SMALL_HEIGHT;

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF31) a31 = (CP(GF31)) (a + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTHTRIGGF31);

  GF31 u[NH], v[NH];
  GF31 p[NH], q[NH];

  u32 line1 = get_line_number(base);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  dependentLaunchWait();   // Previous kernel was fftMiddleInGF31 that launched dependents before writing GF31 data

  u32 me = get_local_id(0);
  readTailFusedLine(in31, u, line1, me);
  readTailFusedLine(in31, v, line2, me);

#if MUL_LOW
  read(G_H, NH, p, a31, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a31, memline2 * SMALL_HEIGHT);
  fft_HEIGHT1(lds, u, smallTrig31, 1, me);
  fft_HEIGHT1(lds, v, smallTrig31, 1, me);
#else
  readTailFusedLine(a31, p, line1, me);
  readTailFusedLine(a31, q, line2, me);
  fft_HEIGHT1(lds, u, smallTrig31, 1, me);
  fft_HEIGHT1(lds, v, smallTrig31, 1, me);
  fft_HEIGHT1(lds, p, smallTrig31, 1, me);
  fft_HEIGHT1(lds, q, smallTrig31, 1, me);
#endif

  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
#if TAIL_TRIGS31 >= 1
  GF31 trig = TFLOAD(&smallTrig31[height_trigs + me]);                    // Trig values for line zero, should be cached
  GF31 mult = TSLOAD(&smallTrig31[height_trigs + G_H + line1]);
  trig = cmul(trig, mult);
#else
  GF31 trig = TOLOAD(&smallTrig31[height_trigs + line1*G_H + me]);
#endif

#if SINGLE_KERNEL
  if (line1 == 0) {
    reverse(lds, u + NH/2, true);
    reverse(lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig, true);
    reverse(lds, u + NH/2, true);

    GF31 trig2 = cmul(trig, TAILTGF31);
    reverse(lds, v + NH/2, false);
    reverse(lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, trig2, false);
    reverse(lds, v + NH/2, false);
  } else {
#else
  if (1) {
#endif
    reverseLine(lds, v);
    reverseLine(lds, q);
    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(lds, v);
  }

  dependentLaunch();       // Next kernel will be fftMiddleOutGF31 which must dependentLaunchWait before reading data

  fft_HEIGHT2(lds, v, smallTrig31, 1, me);
  fft_HEIGHT2(lds, u, smallTrig31, 1, me);
  writeTailFusedLine(v, out31, memline2, me);
  writeTailFusedLine(u, out31, memline1, me);
}


//
// Create a kernel that uses a double-wide workgroup (u in half the workgroup, v in the other half)
// We hope to get better occupancy with the reduced register usage
//

#else

// Special pairMul for double-wide line 0: both halves compute their own self-pairing (u with p),
// there is no cross-half data since line_u == 0 and line_v == H/2 both pair with themselves.
void OVERLOAD pairMul2_special(GF31 *u, GF31 *p, GF31 base_squared) {
  u32 me = get_local_id(0);
  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (i == 0 && me == 0) {
      u[0] = SWAP_XY(mul2(foo2(u[0], p[0])));
      u[NH/2] = SWAP_XY(shl(cmul(u[NH/2], p[NH/2]), 2));
    } else {
      onePairMul(&u[i], &u[NH/2+i], &p[i], &p[NH/2+i], base_squared);
    }
    onePairMul(&u[i+NH/4], &u[NH/2+i+NH/4], &p[i+NH/4], &p[NH/2+i+NH/4], mul_t4(base_squared));
  }
}

KERNEL_CAP(G_H * 2) tailMulGF31(P(T2) out, CP(T2) in, CP(T2) a, u32 base, Trig smallTrig) {
  local GF31 lds[LDS_BYTES(2) / sizeof(GF31)];
  LDSinit(lds, 2);

  const u32 H = ND / SMALL_HEIGHT;

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF31) a31 = (CP(GF31)) (a + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTHTRIGGF31);

  GF31 u[NH], p[NH];

  u32 line_u = get_line_number(base);
  u32 line_v = line_u ? H - line_u : (H / 2);
  u32 me = get_local_id(0);
  u32 lowMe = me % G_H;  // lane-id in one of the two halves (half-workgroups).

  // We're going to call the halves "first-half" and "second-half".
  bool isSecondHalf = me >= G_H;

  u32 line = !isSecondHalf ? line_u : line_v;
  u32 memline = transPos(line, MIDDLE, WIDTH);

  dependentLaunchWait();   // Previous kernel was fftMiddleInGF31 that launched dependents before writing GF31 data

  // Read line u (own half's line) and p (own half's multiplier line)
  readTailFusedLine(in31, u, line, lowMe);

#if MUL_LOW
  read(G_H, NH, p, a31, memline * SMALL_HEIGHT, lowMe);
  fft_HEIGHT1(lds, u, smallTrig31, 2, lowMe);
#else
  readTailFusedLine(a31, p, line, lowMe);
  fft_HEIGHT1(lds, u, smallTrig31, 2, lowMe);
  fft_HEIGHT1(lds, p, smallTrig31, 2, lowMe);
#endif

  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
#if TAIL_TRIGS31 >= 1
  GF31 trig = TFLOAD(&smallTrig31[height_trigs + lowMe]);                                 // Trig values for line zero, should be cached
  GF31 mult = TSLOAD(&smallTrig31[height_trigs + G_H + line_u*2 + isSecondHalf]);         // Two multipliers.  One for line u, one for line v.
  trig = cmul(trig, mult);
#else
  GF31 trig = TOLOAD(&smallTrig31[height_trigs + line_u*G_H*2 + me]);
#endif

#if SINGLE_KERNEL
  // Line 0 and H/2 are special: they pair with themselves, line 0 is offseted by 1.
  if (line_u == 0) {
    reverse2(lds, u);
    reverse2(lds, p);
    pairMul2_special(u, p, trig);
    reverse2(lds, u);
  }
  else {
#else
  if (1) {
#endif
    revCrossLine(lds, u);
    revCrossLine(lds, p);
    pairMul(NH/2, u, u + NH/2, p, p + NH/2, trig, false);
    revCrossLine(lds, u);
  }

  dependentLaunch();       // Next kernel will be fftMiddleOutGF31 which must dependentLaunchWait before reading data

  fft_HEIGHT2(lds, u, smallTrig31, 2, lowMe);

  // Write line u (own half's line)
  writeTailFusedLine(u, out31, memline, lowMe);
}

#endif

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD onePairMul(GF61* pa, GF61* pb, GF61* pc, GF61* pd, GF61 t_squared) {
  GF61 a = *pa, b = *pb, c = *pc, d = *pd;
  X2conjb(a, b);
  X2conjb(c, d);
  GF61 ac = cmul(a, c);
  GF61 bd = cmul(b, d);
  GF61 e = subq(ac, cmul(bd, t_squared));                    // Range is -1-..1+
  GF61 f = subq(subq(cmul(add(a, b), add(c, d)), ac), bd);   // Compute bc + ad.  Range is -2-..1+
  X2q_conjb(&e, &f);                                         // e range is -3-..2+,  f.x range is -2-..3+, f.y range is -3-..2+
  e = modM61q(e, 4);
  f = modM61q(f, 3, 4);
  *pa = SWAP_XY(e), *pb = SWAP_XY(f);
}

void OVERLOAD pairMul(u32 N, GF61 *u, GF61 *v, GF61 *p, GF61 *q, GF61 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(mul2(foo2(u[i], p[i])));
      v[i] = SWAP_XY(shl(cmul(v[i], q[i]), 2));
   } else {
      onePairMul(&u[i], &v[i], &p[i], &q[i], base_squared);
    }

    if (N == NH) {
      onePairMul(&u[i+NH/2], &v[i+NH/2], &p[i+NH/2], &q[i+NH/2], neg(base_squared));
    }

    GF61 new_base_squared = mul_t4(base_squared);
    onePairMul(&u[i+NH/4], &v[i+NH/4], &p[i+NH/4], &q[i+NH/4], new_base_squared);

    if (N == NH) {
      onePairMul(&u[i+3*NH/4], &v[i+3*NH/4], &p[i+3*NH/4], &q[i+3*NH/4], neg(new_base_squared));
    }
  }
}

#if !SINGLE_KERNEL
// The kernel tailMulZeroGF61 handles the special cases in tailMulGF61, i.e. the lines 0 and H/2
// This kernel is launched with 2 workgroups (handling line 0, resp. H/2)
KERNEL(G_H) tailMulZeroGF61(P(T2) out, CP(T2) in, CP(T2) a, Trig smallTrig) {
  local GF61 lds[LDS_BYTES(1) / sizeof(GF61)];
  LDSinit(lds, 1);

  GF61 u[NH], p[NH];
  const u32 H = ND / SMALL_HEIGHT;

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  CP(GF61) a61 = (CP(GF61)) (a + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTHTRIGGF61);

  // This kernel in executed in two workgroups.
  u32 which = get_group_id(0);
  assert(which < 2);

  u32 line = which ? (H/2) : 0;
  u32 memline = transPos(line, MIDDLE, WIDTH);
  u32 me = get_local_id(0);

  dependentLaunch();       // Next kernel will be tailMulGF61 which must dependentLaunchWait before reading data from fftMiddleInGF61
  dependentLaunchWait();   // Previous kernel was fftMiddleInGF61 that launched dependents before writing GF61 data

  readTailFusedLine(in61, u, line, me);

#if MUL_LOW
  read(G_H, NH, p, a61, memline * SMALL_HEIGHT);
  fft_HEIGHT1(lds, u, smallTrig61, 1, me);
#else
  readTailFusedLine(a61, p, line, me);
  fft_HEIGHT1(lds, u, smallTrig61, 1, me);
  fft_HEIGHT1(lds, p, smallTrig61, 1, me);
#endif

  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
#if TAIL_TRIGS61 >= 1
  GF61 trig = TFLOAD(&smallTrig61[height_trigs + me]);
#if SINGLE_WIDE
  GF61 mult = TSLOAD(&smallTrig61[height_trigs + G_H + line]);
#else
  GF61 mult = TSLOAD(&smallTrig61[height_trigs + G_H + which]);
#endif
  trig = cmul(trig, mult);
#else
#if SINGLE_WIDE
  GF61 trig = TOLOAD(&smallTrig61[height_trigs + line*G_H + me]);
#else
  GF61 trig = TOLOAD(&smallTrig61[height_trigs + which*G_H + me]);
#endif
#endif

  reverse(lds, u + NH/2, !which);
  reverse(lds, p + NH/2, !which);
  pairMul(NH/2, u,   u + NH/2, p, p + NH/2, trig, !which);
  reverse(lds, u + NH/2, !which);

  fft_HEIGHT2(lds, u, smallTrig61, 1, me);
  writeTailFusedLine(u, out61, memline, me);
}
#endif

#if SINGLE_WIDE

KERNEL_CAP(G_H) tailMulGF61(P(T2) out, CP(T2) in, CP(T2) a, u32 base, Trig smallTrig) {
  local GF61 lds[LDS_BYTES(1) / sizeof(GF61)];
  LDSinit(lds, 1);

  const u32 H = ND / SMALL_HEIGHT;

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  CP(GF61) a61 = (CP(GF61)) (a + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTHTRIGGF61);

  GF61 u[NH], v[NH];
  GF61 p[NH], q[NH];

  u32 line1 = get_line_number(base);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  dependentLaunchWait();   // Previous kernel was fftMiddleInGF61 that launched dependents before writing GF61 data

  u32 me = get_local_id(0);
  readTailFusedLine(in61, u, line1, me);
  readTailFusedLine(in61, v, line2, me);

#if MUL_LOW
  read(G_H, NH, p, a61, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a61, memline2 * SMALL_HEIGHT);
  fft_HEIGHT1(lds, u, smallTrig61, 1, me);
  fft_HEIGHT1(lds, v, smallTrig61, 1, me);
#else
  readTailFusedLine(a61, p, line1, me);
  readTailFusedLine(a61, q, line2, me);
  fft_HEIGHT1(lds, u, smallTrig61, 1, me);
  fft_HEIGHT1(lds, v, smallTrig61, 1, me);
  fft_HEIGHT1(lds, p, smallTrig61, 1, me);
  fft_HEIGHT1(lds, q, smallTrig61, 1, me);
#endif

  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
#if TAIL_TRIGS61 >= 1
  GF61 trig = TFLOAD(&smallTrig61[height_trigs + me]);                    // Trig values for line zero, should be cached
  GF61 mult = TSLOAD(&smallTrig61[height_trigs + G_H + line1]);
  trig = cmul(trig, mult);
#else
  GF61 trig = TOLOAD(&smallTrig61[height_trigs + line1*G_H + me]);
#endif

#if SINGLE_KERNEL
  if (line1 == 0) {
    reverse(lds, u + NH/2, true);
    reverse(lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig, true);
    reverse(lds, u + NH/2, true);

    GF61 trig2 = cmul(trig, TAILTGF61);
    reverse(lds, v + NH/2, false);
    reverse(lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, trig2, false);
    reverse(lds, v + NH/2, false);
  } else {
#else
  if (1) {
#endif
    reverseLine(lds, v);
    reverseLine(lds, q);
    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(lds, v);
  }

  dependentLaunch();       // Next kernel will be fftMiddleOutGF61 which must dependentLaunchWait before reading data

  fft_HEIGHT2(lds, v, smallTrig61, 1, me);
  fft_HEIGHT2(lds, u, smallTrig61, 1, me);
  writeTailFusedLine(v, out61, memline2, me);
  writeTailFusedLine(u, out61, memline1, me);
}


//
// Create a kernel that uses a double-wide workgroup (u in half the workgroup, v in the other half)
// We hope to get better occupancy with the reduced register usage
//

#else

// Special pairMul for double-wide line 0: both halves compute their own self-pairing (u with p),
// there is no cross-half data since line_u == 0 and line_v == H/2 both pair with themselves.
void OVERLOAD pairMul2_special(GF61 *u, GF61 *p, GF61 base_squared) {
  u32 me = get_local_id(0);
  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (i == 0 && me == 0) {
      u[0] = SWAP_XY(mul2(foo2(u[0], p[0])));
      u[NH/2] = SWAP_XY(shl(cmul(u[NH/2], p[NH/2]), 2));
    } else {
      onePairMul(&u[i], &u[NH/2+i], &p[i], &p[NH/2+i], base_squared);
    }
    onePairMul(&u[i+NH/4], &u[NH/2+i+NH/4], &p[i+NH/4], &p[NH/2+i+NH/4], mul_t4(base_squared));
  }
}

KERNEL_CAP(G_H * 2) tailMulGF61(P(T2) out, CP(T2) in, CP(T2) a, u32 base, Trig smallTrig) {
  local GF61 lds[LDS_BYTES(2) / sizeof(GF61)];
  LDSinit(lds, 2);

  const u32 H = ND / SMALL_HEIGHT;

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  CP(GF61) a61 = (CP(GF61)) (a + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTHTRIGGF61);

  GF61 u[NH], p[NH];

  u32 line_u = get_line_number(base);
  u32 line_v = line_u ? H - line_u : (H / 2);
  u32 me = get_local_id(0);
  u32 lowMe = me % G_H;  // lane-id in one of the two halves (half-workgroups).

  // We're going to call the halves "first-half" and "second-half".
  bool isSecondHalf = me >= G_H;

  u32 line = !isSecondHalf ? line_u : line_v;
  u32 memline = transPos(line, MIDDLE, WIDTH);

  dependentLaunchWait();   // Previous kernel was fftMiddleInGF61 that launched dependents before writing GF61 data

  // Read line u (own half's line) and p (own half's multiplier line)
  readTailFusedLine(in61, u, line, lowMe);

#if MUL_LOW
  read(G_H, NH, p, a61, memline * SMALL_HEIGHT, lowMe);
  fft_HEIGHT1(lds, u, smallTrig61, 2, lowMe);
#else
  readTailFusedLine(a61, p, line, lowMe);
  fft_HEIGHT1(lds, u, smallTrig61, 2, lowMe);
  fft_HEIGHT1(lds, p, smallTrig61, 2, lowMe);
#endif

  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
#if TAIL_TRIGS61 >= 1
  GF61 trig = TFLOAD(&smallTrig61[height_trigs + lowMe]);                                 // Trig values for line zero, should be cached
  GF61 mult = TSLOAD(&smallTrig61[height_trigs + G_H + line_u*2 + isSecondHalf]);         // Two multipliers.  One for line u, one for line v.
  trig = cmul(trig, mult);
#else
  GF61 trig = TOLOAD(&smallTrig61[height_trigs + line_u*G_H*2 + me]);
#endif

#if SINGLE_KERNEL
  // Line 0 and H/2 are special: they pair with themselves, line 0 is offseted by 1.
  if (line_u == 0) {
    reverse2(lds, u);
    reverse2(lds, p);
    pairMul2_special(u, p, trig);
    reverse2(lds, u);
  }
  else {
#else
  if (1) {
#endif
    revCrossLine(lds, u);
    revCrossLine(lds, p);
    pairMul(NH/2, u, u + NH/2, p, p + NH/2, trig, false);
    revCrossLine(lds, u);
  }

  dependentLaunch();       // Next kernel will be fftMiddleOutGF61 which must dependentLaunchWait before reading data

  fft_HEIGHT2(lds, u, smallTrig61, 2, lowMe);

  // Write line u (own half's line)
  writeTailFusedLine(u, out61, memline, lowMe);
}

#endif

#endif
