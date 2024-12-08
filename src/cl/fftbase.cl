// Copyright (C) Mihai Preda

#include "fft4.cl"
#include "fft8.cl"
#include "trig.cl"
// #include "math.cl"

#if BCAST

int bcast4(int x)  { return __builtin_amdgcn_mov_dpp(x, 0, 0xf, 0xf, false); }
int bcast8(int x)  { return __builtin_amdgcn_ds_swizzle(x, 0x0018); }
int bcast16(int x) { return __builtin_amdgcn_ds_swizzle(x, 0x0010); }
int bcast64(int x) { return __builtin_amdgcn_readfirstlane(x); }

int bcastAux(int x, u32 span) {
  return span == 4 ? bcast4(x) : span == 8 ? bcast8(x) : span == 16 ? bcast16(x) : span == 64 ? bcast64(x) : x;
}

T2 bcast(T2 src, u32 span) {
  int4 s = as_int4(src);
  for (int i = 0; i < 4; ++i) { s[i] = bcastAux(s[i], span); }
  return as_double2(s);
}

void chainMul4(T2 *u, T2 w) {
  u[1] = cmul(u[1], w);

  T2 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  double a = 2 * base.y;
  base = U2(fma(a, -w.y, w.x), fma(a, w.x, -w.y));
  u[3] = cmul(u[3], base);
}

void chainMul8(T2 *u, T2 w) {
  u[1] = cmulFancy(u[1], w);

#if 1
  T2 base = 2 * U2(- w.y * w.y, fma(w.x, w.y, w.y));
  u[2] = cmulFancy(u[2], base);
#else
  T2 base = U2(fma(-2 * w.y, w.y, 1), 2 * fma(w.x, w.y, w.y));
  u[2] = cmul(u[2], base);
#endif

  double a = 2 * base.y;
  // base = U2(fma(a, -w.y, w.x + 1), fma(a, w.x, a - w.y));
  base = U2(fma(a, -w.y, w.x + 1), fma(a, w.x, a - w.y));

  for (int i = 3; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmulFancy(base, w);
  }
}

#endif

void shuflBigLDS(u32 WG, local T2 *lds, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i]; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i] = lds[i * WG + me]; }
}

void shufl(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  local T* lds = (local T*) lds2;

#if 0
  // This also works for *f* that is not a power of two.
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me / f) * f * n + me % f] = u[i].x; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].x = lds[i * WG + me]; }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me / f) * f * n + me % f] = u[i].y; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].y = lds[i * WG + me]; }
#else
  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].x; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].x = lds[i * WG + me]; }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].y; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].y = lds[i * WG + me]; }
#endif
}

// Shufl two simultaneous FFT_HEIGHTs.  Needed for tailSquared where u and v are computed simultaneously in different threads.
// NOTE:  It is very important for this routine to use lds memory in coordination with reverseLine2 and unreverseLine2.
// Failure to do so would result in the need for more bar() calls.  Specifically, the u values are stored in the upper half
// of lds memory (first SMALL_HEIGHT T2 values).  The v values are stored in the lower half of lds memory (next SMALL_HEIGHT T2 values).
void shufl2(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);

  // Partition lds memory into upper and lower halves
  assert(WG == G_H);
  lds2 += (me / WG) * SMALL_HEIGHT;

  // Accessing lds memory as doubles is faster than T2 accesses
  local T* lds = (local T*) lds2;

  me = me % WG;
#if 0
  // This also works for *f* that is not a power of two.
  for (u32 i = 0; i < n; ++i) {
    u32 idx = i * f + (me / f) * f * n + me % f;
    lds[idx] = u[i].x;
    lds[SMALL_HEIGHT + idx] = u[i].y;
  }
  if (WG > WAVEFRONT) bar();
  for (u32 i = 0; i < n; ++i) { u[i].x = lds[i * WG + me]; u[i].y = lds[SMALL_HEIGHT + i * WG + me]; }
#else
  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);
  for (u32 i = 0; i < n; ++i) {
    u32 idx = i * f + (me & ~mask) * n + (me & mask);
    lds[idx] = u[i].x;
    lds[SMALL_HEIGHT + idx] = u[i].y;
  }
  if (WG > WAVEFRONT) bar();
  for (u32 i = 0; i < n; ++i) { u[i].x = lds[i * WG + me]; u[i].y = lds[SMALL_HEIGHT + i * WG + me]; }
#endif
}


void tabMul(u32 WG, Trig trig, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);

#if 0
  u32 p = me / f * f;
#else
  u32 p = me & ~(f - 1);
#endif

#if 0
  T2 w = slowTrig_N(ND / n / WG * p, ND / n);
  T2 base = w;
  for (int i = 1; i < n; ++i) {
    u[i] = cmul(u[i], w);
    w = cmul(w, base);
  }
#endif

  T2 w = trig[p];

  if (n >= 8) {
    u[1] = cmulFancy(u[1], w);
  } else {
    u[1] = cmul(u[1], w);
  }

#if 0//CLEAN == 2                          // Titan V likes this case -- only one read.  But Z not helped much.  Needs more investigation.

  u32 midpt = (n + 1) / 2;
  T2 base = trig[(midpt-1)*WG + p];
  T2 tmp = cmulFancyDual_setup(base, w);
  u[midpt] = cmul(u[midpt], base);
  T2 base1 = cmulFancyDual_conj(base, w, tmp);
  u[midpt - 1] = cmul(u[midpt - 1], base1);
  T2 base2 = cmulFancyDual_plain(base, w, tmp);
  u[midpt + 1] = cmul(u[midpt + 1], base2);
  for (u32 i = 2; midpt + i < MIDDLE; ++i) {
    if (midpt - i > 1) u[midpt - i] = cmul(u[midpt - i], base1 = cmulFancy(base1, conjugate(w)));
    u[midpt + i] = cmul(u[midpt + i], base2 = cmulFancy(base2, w));
  }

#elif CLEAN == 1                        // Radeon VII loves this case

  for (u32 i = 2; i < n; ++i) {
    T2 base = trig[(i-1)*WG + p];
    u[i] = cmul(u[i], base);
  }

#elif CLEAN == 0
  if (n >= 8) {
    T2 base = csqTrigFancyFancy(w);
    u[2] = cmulFancy(u[2], base);
    base = ccubeTrigFancy(base, w);
    for (u32 i = 3; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmulFancy(base, w);
    }
  } else {
    T2 base = csqTrig(w);
    u[2] = cmul(u[2], base);
    base = ccubeTrig(base, w);
    for (u32 i = 3; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }
#else
#error CLEAN must be 0 or 1
#endif
}

// Tabmul two simultaneous FFT_HEIGHTs.  Needed for the new tailSquared where u and v are computed simultaneously in different threads.

void tabMul2(u32 WG, Trig trig, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);

  me = me % WG;
#if 0
  u32 p = me / f * f;
#else
  u32 p = me & ~(f - 1);
#endif

#if 0
  T2 w = slowTrig_N(ND / n / WG * p, ND / n);
  T2 base = w;
  for (int i = 1; i < n; ++i) {
    u[i] = cmul(u[i], w);
    w = cmul(w, base);
  }
#endif

  T2 w = trig[p];

  if (n >= 8) {
    u[1] = cmulFancy(u[1], w);
  } else {
    u[1] = cmul(u[1], w);
  }

#if 0//CLEAN == 2                          // Titan V likes this case -- only one read.  But Z not helped much.  Needs more investigation.

  u32 midpt = (n + 1) / 2;
  T2 base = trig[(midpt-1)*WG + p];
  T2 tmp = cmulFancyDual_setup(base, w);
  u[midpt] = cmul(u[midpt], base);
  T2 base1 = cmulFancyDual_conj(base, w, tmp);
  u[midpt - 1] = cmul(u[midpt - 1], base1);
  T2 base2 = cmulFancyDual_plain(base, w, tmp);
  u[midpt + 1] = cmul(u[midpt + 1], base2);
  for (u32 i = 2; midpt + i < MIDDLE; ++i) {
    if (midpt - i > 1) u[midpt - i] = cmul(u[midpt - i], base1 = cmulFancy(base1, conjugate(w)));
    u[midpt + i] = cmul(u[midpt + i], base2 = cmulFancy(base2, w));
  }

#elif CLEAN == 1                        // Radeon VII loves this case

  for (u32 i = 2; i < n; ++i) {
    T2 base = trig[(i-1)*WG + p];
    u[i] = cmul(u[i], base);
  }

#elif CLEAN == 0
  if (n >= 8) {
    T2 base = csqTrigFancyFancy(w);
    u[2] = cmulFancy(u[2], base);
    base = ccubeTrigFancy(base, w);
    for (u32 i = 3; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmulFancy(base, w);
    }
  } else {
    T2 base = csqTrig(w);
    u[2] = cmul(u[2], base);
    base = ccubeTrig(base, w);
    for (u32 i = 3; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }
#else
#error CLEAN must be 0 or 1
#endif
}
