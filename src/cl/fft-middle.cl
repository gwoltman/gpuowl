// Copyright (C) Mihai Preda

#include "trig.cl"

void fft2(T2* u) { X2(u[0], u[1]); }

#if MIDDLE == 3
#include "fft3.cl"
#elif MIDDLE == 4
#include "fft4.cl"
#elif MIDDLE == 5
#include "fft5.cl"
#elif MIDDLE == 6
#include "fft6.cl"
#elif MIDDLE == 7
#include "fft7.cl"
#elif MIDDLE == 8
#include "fft8.cl"
#elif MIDDLE == 9
#include "fft9.cl"
#elif MIDDLE == 10
#include "fft10.cl"
#elif MIDDLE == 11
#include "fft11.cl"
#elif MIDDLE == 12
#include "fft12.cl"
#elif MIDDLE == 13
#include "fft13.cl"
#elif MIDDLE == 14
#include "fft14.cl"
#elif MIDDLE == 15
#include "fft15.cl"
#elif MIDDLE == 16
#include "fft16.cl"
#endif

void fft_MIDDLE(T2 *u) {
#if MIDDLE == 1
  // Do nothing
#elif MIDDLE == 2
  fft2(u);
#elif MIDDLE == 3
  fft3(u);
#elif MIDDLE == 4
  fft4(u);
#elif MIDDLE == 5
  fft5(u);
#elif MIDDLE == 6
  fft6(u);
#elif MIDDLE == 7
  fft7(u);
#elif MIDDLE == 8
  fft8(u);
#elif MIDDLE == 9
  fft9(u);
#elif MIDDLE == 10
  fft10(u);
#elif MIDDLE == 11
  fft11(u);
#elif MIDDLE == 12
  fft12(u);
#elif MIDDLE == 13
  fft13(u);
#elif MIDDLE == 14
  fft14(u);
#elif MIDDLE == 15
  fft15(u);
#elif MIDDLE == 16
  fft16(u);
#else
#error UNRECOGNIZED MIDDLE
#endif
}

// Apply the twiddles needed after fft_MIDDLE and before fft_HEIGHT in forward FFT.
// Also used after fft_HEIGHT and before fft_MIDDLE in inverse FFT.

#define WADD(i, w) u[i] = cmul(u[i], w)
#define WSUB(i, w) u[i] = cmul_by_conjugate(u[i], w)

#define WADDF(i, w) u[i] = cmulFancy(u[i], w)
#define WSUBF(i, w) u[i] = cmulFancy(u[i], conjugate(w))

// Keep in sync with TrigBufCache.cpp, see comment there.
#define SHARP_MIDDLE 5

#if !defined(MM_CHAIN) && !defined(MM2_CHAIN) && TRIG_HI
#define MM_CHAIN 1
#define MM2_CHAIN 2
#endif

void middleMul(T2 *u, u32 s, Trig trig) {
  assert(s < SMALL_HEIGHT);
  assert(MIDDLE > 1);

  T2 w = trig[s]; // s / BIG_HEIGHT

  if (MIDDLE < SHARP_MIDDLE) {
    WADD(1, w);
#if MM_CHAIN == 0
    T2 base = csqTrig(w);
    for (u32 k = 2; k < MIDDLE; ++k) {
      WADD(k, base);
      base = cmul(base, w);
    }
#elif MM_CHAIN == 1
    for (u32 k = 2; k < MIDDLE; ++k) { WADD(k, slowTrig_N(WIDTH * k * s, WIDTH * k * SMALL_HEIGHT)); }
#else
#error MM_CHAIN must be 0 or 1
#endif

  } else { // MIDDLE >= 5

#if MM_CHAIN == 0
    WADDF(1, w);
    T2 base;
    if (MIDDLE >= 10) {
      base = csqFancyUpdate(w);
      WADDF(2, base);
      base.x += 1;
    } else {
      base = w;
      base.x += 1;
      base = cmulFancy(base, w);
      WADD(2, base);
    }

    for (u32 k = 3; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }

#elif MM_CHAIN == 1
    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      T2 base = slowTrig_N(WIDTH * k * s, WIDTH * SMALL_HEIGHT * k);
      WADD(k-1, base);
      WADD(k,   base);
      WADD(k+1, base);
    }

    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      WSUBF(k-1, w);
      WADDF(k+1, w);
    }

    WADDF(1, w);

    if ((MIDDLE - 2) % 3 > 0) {
      WADDF(2, w);
      WADDF(2, w);
    }

    if ((MIDDLE - 2) % 3 == 2) {
      WADDF(3, w);
      WADDF(3, csqFancyUpdate(w));
    }
#else
#error MM_CHAIN must be 0 or 1.
#endif
  }
}

void middleMul2(T2 *u, u32 x, u32 y, double factor, Trig trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);
  assert(MIDDLE > 1);

  T2 w = trig[SMALL_HEIGHT + x]; // x / (MIDDLE * WIDTH)

  if (MIDDLE < SHARP_MIDDLE) {
    T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, (WIDTH-1) * (SMALL_HEIGHT-1) + (WIDTH-1) * SMALL_HEIGHT) * factor;
    for (u32 k = 0; k < MIDDLE; ++k) { WADD(k, base); }
    WSUB(0, w);
    if (MIDDLE > 2) { WADD(2, w); }
    if (MIDDLE > 3) { WADD(3, w); WADD(3, w); }

  } else { // MIDDLE >= 5
    // T2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE);

#if MM2_CHAIN == 0

    T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, (WIDTH-1) * (SMALL_HEIGHT-1) + (WIDTH-1) * SMALL_HEIGHT) * factor;
    WADD(0, base);
    WADD(1, base);

    for (u32 k = 2; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }
    WSUBF(0, w);

#elif MM2_CHAIN == 1
    u32 cnt = 1;
    for (u32 start = 0, sz = (MIDDLE - start + cnt - 1) / cnt; cnt > 0; --cnt, start += sz) {
      if (start + sz > MIDDLE) { --sz; }
      u32 n = (sz - 1) / 2;
      u32 mid = start + n;

      T2 base1 = slowTrig_N(x * y + x * SMALL_HEIGHT * mid, (WIDTH-1) * (SMALL_HEIGHT-1) + (WIDTH-1) * SMALL_HEIGHT * mid) * factor;
      WADD(mid, base1);

      T2 base2 = base1;
      for (u32 i = 1; i <= n; ++i) {
        base1 = cmulFancy(base1, conjugate(w));
        WADD(mid - i, base1);

        base2 = cmulFancy(base2, w);
        WADD(mid + i, base2);
      }
      if (!(sz & 1)) {
        base2 = cmulFancy(base2, w);
        WADD(mid + n + 1, base2);
      }
    }

#elif MM2_CHAIN == 2
    for (u32 i = 1; i < MIDDLE; i += 3) {
      T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT * i, (WIDTH-1) * (SMALL_HEIGHT-1) + (WIDTH-1) * SMALL_HEIGHT * i) * factor;
      T2 tmp = cmulFancyDual_setup(base, w);
      WADD(i-1, cmulFancyDual_conj(base, w, tmp));
      WADD(i, base);
      if (i + 1 < MIDDLE) { WADD(i+1, cmulFancyDual_plain(base, w, tmp)); }
    }
    if (MIDDLE % 3 == 1) {
      WADD(MIDDLE-1, slowTrig_N(x * y + x * SMALL_HEIGHT * (MIDDLE-1), (WIDTH-1) * (SMALL_HEIGHT-1) + (WIDTH-1) * SMALL_HEIGHT * (MIDDLE-1)) * factor);
    }
#else
#error MM2_CHAIN must be 0, 1 or 2.
#endif
  }
}




#define PERMUTE 0

#define NUM_LANES 64
// Get the T2 value from the specified lane
T2 read_from_lane(T2 src, u32 lane) {
  int4 s = as_int4(src);
  int4 dest;
  for (int i = 0; i < 4; ++i) { dest[i] = __builtin_amdgcn_ds_bpermute (lane * 4, s[i]); }
  return as_double2(dest);
}

//******************************************************************************
// Special versions of MiddleMul and MiddleMul2 for AMD GPUs
// Should improve accuracy for modest speed cost over standard versions
//******************************************************************************

void middleMulOut(T2 *u, u32 y, u32 x, Trig trig) {
  assert(y < WIDTH);
  assert(x < SMALL_HEIGHT);
  if (MIDDLE == 1) { return; }

#if !PERMUTE
  // Use old permute-less version
  middleMul(u, x, trig);
  return;
#endif

  // Compute all the trig values that will be needed by this 64-thread wavefront.
  // A fftMiddleOut wavefront has OUT_SIZEX x values and 64/OUT_SIZEX y values.
  // We need MIDDLE-1 trig values for each x value.
  // If OUT_SIZEX * (MIDDLE-1) exceeds 64, there are somewhat slower methods
  // available using fewer MIDDLE trig values.
  u32 TRIGVALS_PER_X = NUM_LANES / OUT_SIZEX;
  u32 lane = get_local_id(0) % NUM_LANES;
  // Compute trig (x * j / BIGHEIGHT) for j = 1...TRIGVALS_PER_X
  T2 trigvals = slowTrig_N(x * (lane / OUT_SIZEX + 1) * WIDTH, (SMALL_HEIGHT-1) * TRIGVALS_PER_X * WIDTH);

  T2 w;
  for (u32 j = 1; j < TRIGVALS_PER_X; ++j) {
    // Get j-th trigval for x
    w = read_from_lane(trigvals, x % OUT_SIZEX + (j-1) * OUT_SIZEX);
    // Apply j-th trig value to as many u[i] as necessary
    for (u32 i = j; i < MIDDLE; i += TRIGVALS_PER_X) { WADD(i, w); }
  }
  // Get last trigval for x
  w = read_from_lane(trigvals, x % OUT_SIZEX + (TRIGVALS_PER_X-1) * OUT_SIZEX);
  // Apply last trig value to as many u[i] as necessary
  T2 base = w;
  for (u32 k = TRIGVALS_PER_X; k < MIDDLE; k += TRIGVALS_PER_X) {
    for (u32 i = 0; i < TRIGVALS_PER_X && k + i < MIDDLE; ++i) { WADD(k + i, base); }
    if (k == TRIGVALS_PER_X) base = csq(w);
    else base = cmul(base, w);
  }
}

void middleMulIn(T2 *u, u32 y, u32 x, Trig trig) {
  assert(y < SMALL_HEIGHT);
  assert(x < WIDTH);
  if (MIDDLE == 1) { return; }

#if !PERMUTE
  // Use old permute-less version
  middleMul(u, y, trig);
  return;
#endif

  // Compute all the trig values that will be needed by this 64-thread wavefront.
  // A fftMiddleIn wavefront has IN_SIZEX x values and 64/IN_SIZEX y values.
  // We need MIDDLE-1 trig values for each y value.
  // If 64/IN_SIZEX * (MIDDLE-1) exceeds 64, there are slightly slower methods
  // available using fewer MIDDLE trig values.
  u32 TRIGVALS_PER_Y = IN_SIZEX;
  u32 SIZEY = NUM_LANES / IN_SIZEX;
  u32 lane = get_local_id(0) % NUM_LANES;
  // Compute trig (y * j / BIGHEIGHT) for j = 1...TRIGVALS_PER_Y 
  T2 trigvals = slowTrig_N(y * (lane % IN_SIZEX + 1) * WIDTH, (SMALL_HEIGHT-1) * TRIGVALS_PER_Y * WIDTH);

  T2 w;
  for (u32 j = 1; j < TRIGVALS_PER_Y; ++j) {
    // Get j-th trigval for y
    w = read_from_lane(trigvals, (y % SIZEY) * TRIGVALS_PER_Y + (j-1));
    // Apply j-th trig value to as many u[i] as necessary
    for (u32 i = j; i < MIDDLE; i += TRIGVALS_PER_Y) { WADD(i, w); }
  }
  // Get last trigval for y
  w = read_from_lane(trigvals, (y % SIZEY) * TRIGVALS_PER_Y + (TRIGVALS_PER_Y-1));
  // Apply last trig value to as many u[i] as necessary
  T2 base = w;
  for (u32 k = TRIGVALS_PER_Y; k < MIDDLE; k += TRIGVALS_PER_Y) {
    for (u32 i = 0; i < TRIGVALS_PER_Y && k + i < MIDDLE; ++i) { WADD(k + i, base); }
    if (k == TRIGVALS_PER_Y) base = csq(w);
    else base = cmul(base, w);
  }
}


void middleMul2Out(T2 *u, u32 y, u32 x, double factor, Trig trig) {
  assert(y < WIDTH);
  assert(x < SMALL_HEIGHT);
  if (MIDDLE == 1) { WADD(0, slowTrig_N(x * y, ND) * factor); return; }

#if !PERMUTE
  // Use old permute-less version
  middleMul2(u, y, x, factor, trig);
  return;
#endif

  // Compute all the trig values that will be needed by this 64-thread wavefront.
  // A fftMiddleOut wavefront has OUT_SIZEX x values and 64/OUT_SIZEX y values.
  // We need (MIDDLE+1)/2 trig values for each y value.
  // If OUT_SIZEX * (MIDDLE+1)/2 exceeds 64, there are somewhat less accurate
  // methods available using fewer MIDDLE trig values.
  u32 TRIGVALS_PER_Y = OUT_SIZEX;
  u32 SIZEY = NUM_LANES / OUT_SIZEX;
  u32 lane = get_local_id(0) % NUM_LANES;
  // Compute trig (y * j / BIGWIDTH) for j = 1...TRIGVALS_PER_Y 
  T2 trigvals = slowTrig_N(y * (lane % OUT_SIZEX + 1) * SMALL_HEIGHT, (WIDTH-1) * TRIGVALS_PER_Y * SMALL_HEIGHT);

  u32 midpt = (MIDDLE + 1) / 2;
  T2 basemid = slowTrig_N(y * x + midpt * y * SMALL_HEIGHT, (WIDTH-1) * (SMALL_HEIGHT-1) + midpt * (WIDTH-1) * SMALL_HEIGHT) * factor;
  WADD(midpt, basemid);

  T2 base1, base2;
  for (u32 j = 1; j <= TRIGVALS_PER_Y && j <= midpt; ++j) {
    // Get j-th trigval for y
    T2 w = read_from_lane(trigvals, (y % SIZEY) * TRIGVALS_PER_Y + (j-1));
    T2 tmp = cmulDual_setup(basemid, w);

    // Work on midpt - j
    base1 = cmulDual_conj(basemid, w, tmp);
    WADD(midpt - j, base1);

    // Work on midpt + j
    if (midpt + j < MIDDLE) {
      base2 = cmulDual_plain(basemid, w, tmp);
      WADD(midpt + j, base2);
    }
  }

  // If there weren't enough trig values, do some more processing
  for (u32 processed = TRIGVALS_PER_Y; processed < midpt; ) {
    for (u32 j = 1; j <= TRIGVALS_PER_Y && processed < midpt; ++j) {
      // Get j-th trigval for y
      T2 w = read_from_lane(trigvals, (y % SIZEY) * TRIGVALS_PER_Y + (j-1));

      processed++;

      // Work on decreasing u[i]
      if (j < TRIGVALS_PER_Y) WADD(midpt - processed, cmul(base1, conjugate(w)));
      else WADD(midpt - processed, base1 = cmul(base1, conjugate(w)));

      // Work on increasing u[i]
      if (midpt + processed < MIDDLE) {
        if (j < TRIGVALS_PER_Y) WADD(midpt + processed, cmul(base2, w));
        else WADD(midpt + processed, base2 = cmul(base2, w));
      }
    }
  }
}


void middleMul2In(T2 *u, u32 y, u32 x, double factor, Trig trig) {
  assert(y < SMALL_HEIGHT);
  assert(x < WIDTH);
  if (MIDDLE == 1) { WADD(0, slowTrig_N(x * y, ND) * factor); return; }

#if !PERMUTE
  // Use old permute-less version
  middleMul2(u, x, y, factor, trig);
  return;
#endif

  // Compute all the trig values that will be needed by this 64-thread wavefront.
  // A fftMiddleIn wavefront has IN_SIZEX x values and 64/IN_SIZEX y values.
  // We need (MIDDLE+1)/2 trig values for each x value.
  // If IN_SIZEX * ((MIDDLE+1)/2) exceeds 64, there are somewhat less accurate
  // methods available using fewer MIDDLE trig values.
  u32 TRIGVALS_PER_X = NUM_LANES / IN_SIZEX;
  u32 lane = get_local_id(0) % NUM_LANES;
  // Compute trig (x * j / BIGWIDTH) for j = 1...TRIGVALS_PER_X
  T2 trigvals = slowTrig_N(x * (lane / IN_SIZEX + 1) * SMALL_HEIGHT, (WIDTH-1) * TRIGVALS_PER_X * SMALL_HEIGHT);

  u32 midpt = (MIDDLE + 1) / 2;
  T2 basemid = slowTrig_N(x * y + midpt * x * SMALL_HEIGHT, (WIDTH-1) * (SMALL_HEIGHT-1) + midpt * (WIDTH-1) * SMALL_HEIGHT) * factor;
  WADD(midpt, basemid);

  T2 base1, base2;
  for (u32 j = 1; j <= TRIGVALS_PER_X && j <= midpt; ++j) {
    // Get j-th trigval for x
    T2 w = read_from_lane(trigvals, x % IN_SIZEX + (j-1) * IN_SIZEX);
    T2 tmp = cmulDual_setup(basemid, w);

    // Work on midpt - j
    base1 = cmulDual_conj(basemid, w, tmp);
    WADD(midpt - j, base1);

    // Work on midpt + j
    if (midpt + j < MIDDLE) {
      base2 = cmulDual_plain(basemid, w, tmp);
      WADD(midpt + j, base2);
    }
  }

  // If there weren't enough trig values, do some more processing
  for (u32 processed = TRIGVALS_PER_X; processed < midpt; ) {
    for (u32 j = 1; j <= TRIGVALS_PER_X && processed < midpt; ++j) {
      // Get j-th trigval for x
      T2 w = read_from_lane(trigvals, x % IN_SIZEX + (j-1) * IN_SIZEX);

      processed++;

      // Work on decreasing u[i]
      if (j < TRIGVALS_PER_X) WADD(midpt - processed, cmul(base1, conjugate(w)));
      else WADD(midpt - processed, base1 = cmul(base1, conjugate(w)));

      // Work on increasing u[i]
      if (midpt + processed < MIDDLE) {
        if (j < TRIGVALS_PER_X) WADD(midpt + processed, cmul(base2, w));
        else WADD(midpt + processed, base2 = cmul(base2, w));
      }
    }
  }
}

#undef WADD
#undef WADDF
#undef WSUB
#undef WSUBF

// Do a partial transpose during fftMiddleIn/Out
void middleShuffle(local T *lds, T2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 8) {
    local T *p1 = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local T *p2 = lds + me;
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = p2[workgroupSize * i]; }
  } else {
    local int *p1 = ((local int*) lds) + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local int *p2 = (local int*) lds + me;
    int4 *pu = (int4 *)u;

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].y = p2[workgroupSize * i]; }
    bar();

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].z; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].z = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].w; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].w = p2[workgroupSize * i]; }
  }
}
