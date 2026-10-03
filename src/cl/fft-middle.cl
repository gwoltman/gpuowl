// Copyright (C) Mihai Preda

#include "math.cl"
#include "trig.cl"

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
#include "fft4.cl"           // PFA uses the GF fft4 (fft12.cl only includes it for FP64)
#elif MIDDLE == 13
#include "fft13.cl"
#elif MIDDLE == 14
#include "fft14.cl"
#elif MIDDLE == 15
#include "fft15.cl"
#elif MIDDLE == 16
#include "fft16.cl"
#endif

#if !defined(MM_CHAIN) && !defined(MM2_CHAIN) && FFT_VARIANT_M == 0
#define MM_CHAIN 0
#define MM2_CHAIN 0
#endif

#if !defined(MM_CHAIN) && !defined(MM2_CHAIN) && FFT_VARIANT_M == 1
#define MM_CHAIN 1
#define MM2_CHAIN 2
#endif

// Apply the twiddles needed after fft_MIDDLE and before fft_HEIGHT in forward FFT.
// Also used after fft_HEIGHT and before fft_MIDDLE in inverse FFT.

#define WADD(i, w) u[i] = cmul(u[i], w)
#define WSUB(i, w) u[i] = cmul_by_conjugate(u[i], w)
#define WADDF(i, w) u[i] = cmulFancy(u[i], w)
#define WSUBF(i, w) u[i] = cmulFancy(u[i], conjugate(w))

#if FFT_FP64

void OVERLOAD fft2(T2* u) { X2(u[0], u[1]); }

void OVERLOAD fft_MIDDLE(T2 *u) {
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

// Keep in sync with TrigBufCache.cpp, see comment there.
#define SHARP_MIDDLE 5

void OVERLOAD middleMul(T2 *u, u32 s, Trig trig) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) return;

  if (WIDTH == SMALL_HEIGHT) trig += SMALL_HEIGHT;     // In this case we can share the MiddleMul2 trig table.  Skip over the MiddleMul trig table.
  T2 w = TFLOAD(&trig[s]);           // s / BIG_HEIGHT

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
    base = csqTrigFancy(w);
    WADDF(2, base);
    base = ccubeTrigFancy(base, w);
    WADDF(3, base);
    base.x += 1;

    for (u32 k = 4; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }

#elif 0 && MM_CHAIN == 1        // This is fewer F64 ops, but may be slower on Radeon 7 -- probably the optimizer being weird.  It also has somewhat worse Z.
    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      T2 base, base_minus1, base_plus1;
      base = slowTrig_N(WIDTH * k * s, WIDTH * SMALL_HEIGHT * k);
      cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
      WADD(k-1, base_minus1);
      WADD(k,   base);
      WADD(k+1, base_plus1);
    }

    WADDF(1, w);

    T2 w2;
    if ((MIDDLE - 2) % 3 > 0) {
      w2 = csqTrigFancy(w);
      WADDF(2, w2);
    }

    if ((MIDDLE - 2) % 3 == 2) {
      T2 w3 = ccubeTrigFancy(w2, w);
      WADDF(3, w3);
    }

#elif MM_CHAIN == 1
    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      T2 base, base_minus1, base_plus1;
      base = slowTrig_N(WIDTH * k * s, WIDTH * SMALL_HEIGHT * k);
      cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
      WADD(k-1, base_minus1);
      WADD(k,   base);
      WADD(k+1, base_plus1);
    }

    WADDF(1, w);

    if ((MIDDLE - 2) % 3 > 0) {
      WADDF(2, w);
      WADDF(2, w);
    }

    if ((MIDDLE - 2) % 3 == 2) {
      WADDF(3, w);
      WADDF(3, csqTrigFancy(w));
    }
#else
#error MM_CHAIN must be 0 or 1.
#endif
  }
}

void OVERLOAD middleMul2(T2 *u, u32 x, u32 y, double factor, Trig trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  if (MIDDLE == 1) {
    WADD(0, slowTrig_N(x * y, ND) * factor);
    return;
  }

  trig += SMALL_HEIGHT;          // Skip over the MiddleMul trig table
  T2 w = TFLOAD(&trig[x]);       // x / (MIDDLE * WIDTH)

  if (MIDDLE < SHARP_MIDDLE) {
    T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, ND / MIDDLE * 2) * factor;
    for (u32 k = 0; k < MIDDLE; ++k) { WADD(k, base); }
    WSUB(0, w);
    if (MIDDLE > 2) { WADD(2, w); }
    if (MIDDLE > 3) { WADD(3, w); WADD(3, w); }

  } else { // MIDDLE >= 5
    // T2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE);

#if 0                                   // Slower on Radeon 7, but proves the concept for use in GF61.  Might be worthwhile on poor FP64 GPUs

    Trig trig2 = trig + WIDTH;          // Skip over the fist MiddleMul2 trig table
    u32 desired_root = x * y;
    T2 base = cmulFancy(TFLOAD(&trig2[desired_root % SMALL_HEIGHT]), TFLOAD(&trig[desired_root / SMALL_HEIGHT])) * factor;   //Optimization to do: put multiply by factor in trig2 table

    WADD(0, base);
    for (u32 k = 1; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }

#elif AMDGPU && MM2_CHAIN == 0          // Oddly, Radeon 7 is faster with this version that uses more F64 ops

    T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, ND / MIDDLE * 2) * factor;
    WADD(0, base);
    WADD(1, base);

    for (u32 k = 2; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }
    WSUBF(0, w);

#elif MM2_CHAIN == 0

    u32 mid = MIDDLE / 2;
    T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT * mid, ND / MIDDLE * (mid + 1)) * factor;
    WADD(mid, base);

    T2 basehi, baselo;
    cmul_a_by_fancyb_and_conjfancyb(&basehi, &baselo, base, w);
    WADD(mid-1, baselo);
    WADD(mid+1, basehi);

    for (int i = mid-2; i >= 0; --i) {
      baselo = cmulFancy(baselo, conjugate(w));
      WADD(i, baselo);
    }

    for (int i = mid+2; i < MIDDLE; ++i) {
      basehi = cmulFancy(basehi, w);
      WADD(i, basehi);
    }

#elif MM2_CHAIN == 1
    u32 cnt = 1;
    for (u32 start = 0, sz = (MIDDLE - start + cnt - 1) / cnt; cnt > 0; --cnt, start += sz) {
      if (start + sz > MIDDLE) { --sz; }
      u32 n = (sz - 1) / 2;
      u32 mid = start + n;

      T2 base1 = slowTrig_N(x * y + x * SMALL_HEIGHT * mid, ND / MIDDLE * (mid + 1)) * factor;
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
    T2 base, base_minus1, base_plus1;
    for (u32 i = 1; ; i += 3) {
      if (i-1 == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        WADD(i-1, base_minus1);
        break;
      } else if (i == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        base = cmulFancy(base_minus1, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
        break;
      } else {
        base = slowTrig_N(x * y + x * SMALL_HEIGHT * i, ND / MIDDLE * (i + 1)) * factor;
        cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
        WADD(i+1, base_plus1);
        if (i+1 == MIDDLE-1) break;
      }
    }
#else
#error MM2_CHAIN must be 0, 1 or 2.
#endif
  }
}

// Do a partial transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local T *lds, T2 *u, u32 workgroupSize, u32 blockSize) {
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
    // The plain write index below -- (me%blockSize)*(workgroupSize/blockSize)+me/blockSize -- is a classic
    // shared-memory transpose and conflicts badly: verified via simulation, every 32-lane phase group collides
    // (workgroupSize=128, blockSize=16, the default IN_SIZEX/OUT_SIZEX). Replace it with a "diagonal" address,
    // addr(outer,inner) = outer*BIG + (inner + k*outer) % BIG, where BIG/ROWS are blockSize and
    // workgroupSize/blockSize in whichever order is larger, "outer" is whichever of row/col has the smaller
    // range, and k is 1 (square: blockSize==ROWS) or 2 (2:1 ratio, e.g. 16 vs 8, or 8 vs 16) -- the only ratios
    // verified conflict-free (on both the write and the read below) by exhaustive simulation, and also verified
    // to reproduce the exact same per-thread values as the plain formula (a correctness round trip, not just a
    // conflict count -- see the shufl_and_fft2 fix earlier for why that distinction matters). Falls back to the
    // plain, possibly-conflicting formula for any other ratio so an unusual -use combination still computes the
    // right answer, just without the guaranteed fix.
    u32 ROWS = workgroupSize / blockSize;
    u32 row = me / blockSize, col = me % blockSize;
    u32 row_w = me % ROWS, col_w = me / ROWS;
    u32 p1idx, p2idx;
    if (blockSize == 2 * ROWS) {
      p1idx = row * blockSize + (col + 2 * row) % blockSize;
      p2idx = row_w * blockSize + (col_w + 2 * row_w) % blockSize;
    } else if (ROWS == 2 * blockSize) {
      p1idx = col * ROWS + (row + 2 * col) % ROWS;
      p2idx = col_w * ROWS + (row_w + 2 * col_w) % ROWS;
    } else if (blockSize == ROWS) {
      p1idx = row * blockSize + (col + row) % blockSize;
      p2idx = row_w * blockSize + (col_w + row_w) % blockSize;
    } else {
      p1idx = col * ROWS + row;
      p2idx = me;
    }
    local int *p1 = ((local int*) lds) + p1idx;
    local int *p2 = ((local int*) lds) + p2idx;
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

// Do a partial transpose during fftMiddleIn/Out and write the results to global memory
void OVERLOAD middleShuffleWrite(global T2 *out, T2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  out += (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
  for (int i = 0; i < MIDDLE; ++i) { out[i * workgroupSize] = u[i]; }
}

// Do an in-place 16x16 transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local T2 *lds, T2 *u) {
  u32 me = get_local_id(0);
  u32 y = me / 16;
  u32 x = me % 16;

  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * 16 + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * 16 + y] = u[i];"
    bar();
    u[i] = lds[y * 16 + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * 16 + x ^ y] = u[i];
    bar();
    u[i] = lds[x * 16 + y ^ x];
  }
}


#if PFA

// The FP side of a PFA middle step (hybrid FFT/NTT types), as pfaMiddleIn / pfaMiddleOut for GF61 below.  The PFA-th root of unity
// w = e^(2*pi*i/PFA) is complex and on the unit circle, so the inverse radix-PFA is the forward one on SWAP_XY'd data, as for the
// binary parts of the transform.
// Forward PFA-point DFT, pairing the inputs k and PFA - k:  y_j = a0 + sum_k cos(2*pi*jk/PFA) (a_k + a_-k) + i sin(2*pi*jk/PFA) (a_k - a_-k)
void OVERLOAD pfaDftR(T2 *a) {
  const T c[PFA] = { PFA_COS }, s[PFA] = { PFA_SIN };
  const u32 h = (PFA - 1) / 2;
  T2 sum[(PFA - 1) / 2], dif[(PFA - 1) / 2], y[PFA];
  y[0] = a[0];
  for (u32 k = 1; k <= h; ++k) { sum[k - 1] = a[k] + a[PFA - k]; dif[k - 1] = a[k] - a[PFA - k]; y[0] += sum[k - 1]; }
  for (u32 j = 1; j <= h; ++j) {
    T2 pc = sum[0] * c[j % PFA], qs = dif[0] * s[j % PFA];
    for (u32 k = 2; k <= h; ++k) { pc += sum[k - 1] * c[j * k % PFA]; qs += dif[k - 1] * s[j * k % PFA]; }
    pc += a[0];
    T2 iqs = U2(-qs.y, qs.x);
    y[j] = pc + iqs;
    y[PFA - j] = pc - iqs;
  }
  for (u32 k = 0; k < PFA; ++k) { a[k] = y[k]; }
}

// Rotate a[0..PFA-1] by a run-time amount t < PFA: right (a[i] = a[i - t]) or left (a[i] = a[i + t]), one select per bit of t
void OVERLOAD pfaRotate(T2 *a, u32 t, bool right) {
  for (u32 bit = 1; bit < PFA; bit *= 2) {
    T2 r[PFA];
    for (u32 i = 0; i < PFA; ++i) { r[i] = (t & bit) ? a[(i + (right ? PFA - bit : bit)) % PFA] : a[i]; }
    for (u32 i = 0; i < PFA; ++i) { a[i] = r[i]; }
  }
}

// The twiddle w^(x*b) of a row, w a root of order PFA_L, from the PFA middle trig table (see genMiddleTrigFP64 and pfaTwiddle for GF61)
T2 OVERLOAD pfaTwiddle(u32 x, u32 b, Trig trig) {
  Trig trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  u32 desired_root = x * b;
  return cmul(TFLOAD(&trig[desired_root % PFA_BH]), TFLOAD(&trig1[desired_root / PFA_BH]));
}

// The middle step of a row: radix-PFA_M2 and the twiddles w^(WIDTH*y*k), see pfaRowMiddle for GF61
void OVERLOAD pfaRowMiddle(T2 *u, u32 y, Trig trig, bool inverse) {
#if PFA_M2 > 1
  Trig mm = trig + PFA_BH;
  if (inverse) { for (u32 k = 1; k < PFA_M2; ++k) { u[k] = cmul(u[k], TFLOAD(&mm[(k - 1) * SMALL_HEIGHT + y])); } }
#if PFA_M2 == 2
  X2(u[0], u[1]);
#else
  fft4(u);
#endif
  if (!inverse) { for (u32 k = 1; k < PFA_M2; ++k) { u[k] = cmul(u[k], TFLOAD(&mm[(k - 1) * SMALL_HEIGHT + y])); } }
#endif
}

// fftMiddleIn for PFA, see pfaMiddleIn for GF61
void OVERLOAD pfaMiddleIn(T2 *u, u32 x, u32 y, Trig trig) {
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    T2 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    T2 a[PFA];
    for (u32 j = 0; j < PFA; ++j) { a[j * (PFA_BH % PFA) % PFA] = u[m2 + PFA_M2 * j]; }
    pfaRotate(a, t, true);
    pfaDftR(a);
    for (u32 k3 = 0; k3 < PFA; ++k3) { u[m2 + PFA_M2 * k3] = a[k3]; }
  }
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, false); }
}

// fftMiddleOut for PFA on the SWAP_XY'd data of the inverse transform, see pfaMiddleOut for GF61.  factor is the stock
// normalization (NWORDS includes the factor PFA of the radix-PFA).
void OVERLOAD pfaMiddleOut(T2 *u, u32 x, u32 y, T factor, Trig trig) {
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, true); }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    T2 a[PFA];
    for (u32 k3 = 0; k3 < PFA; ++k3) { a[k3] = u[m2 + PFA_M2 * k3]; }
    pfaDftR(a);
    pfaRotate(a, t, false);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = a[j * (PFA_BH % PFA) % PFA]; }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    T2 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig) * factor;
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
}

#endif

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

void OVERLOAD fft2(F2* u) { X2(u[0], u[1]); }

void OVERLOAD fft_MIDDLE(F2 *u) {
#if MIDDLE == 1
  // Do nothing
#elif MIDDLE == 2
  fft2(u);
#elif MIDDLE == 4
  fft4(u);
#elif MIDDLE == 8
  fft8(u);
#elif MIDDLE == 16
  fft16(u);
#elif PFA
  // Not used: PFA's middle step is done by pfaMiddleIn / pfaMiddleOut
#else
#error UNRECOGNIZED MIDDLE
#endif
}

// Keep in sync with TrigBufCache.cpp, see comment there.
#define SHARP_MIDDLE 5

void OVERLOAD middleMul(F2 *u, u32 s, TrigFP32 trig) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) return;

  if (WIDTH == SMALL_HEIGHT) trig += SMALL_HEIGHT;     // In this case we can share the MiddleMul2 trig table.  Skip over the MiddleMul trig table.
  F2 w = TFLOAD(&trig[s]);           // s / BIG_HEIGHT

  if (MIDDLE < SHARP_MIDDLE) {
    WADD(1, w);
#if MM_CHAIN == 0
    F2 base = csqTrig(w);
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
    F2 base;
    base = csqTrigFancy(w);
    WADDF(2, base);
    base = ccubeTrigFancy(base, w);
    WADDF(3, base);
    base.x += 1;

    for (u32 k = 4; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }

#elif 0 && MM_CHAIN == 1        // This is fewer F64 ops, but may be slower on Radeon 7 -- probably the optimizer being weird.  It also has somewhat worse Z.
    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      F2 base, base_minus1, base_plus1;
      base = slowTrig_N(WIDTH * k * s, WIDTH * SMALL_HEIGHT * k);
      cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
      WADD(k-1, base_minus1);
      WADD(k,   base);
      WADD(k+1, base_plus1);
    }

    WADDF(1, w);

    F2 w2;
    if ((MIDDLE - 2) % 3 > 0) {
      w2 = csqTrigFancy(w);
      WADDF(2, w2);
    }

    if ((MIDDLE - 2) % 3 == 2) {
      F2 w3 = ccubeTrigFancy(w2, w);
      WADDF(3, w3);
    }

#elif MM_CHAIN == 1
    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      F2 base, base_minus1, base_plus1;
      base = slowTrig_N(WIDTH * k * s, WIDTH * SMALL_HEIGHT * k);
      cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
      WADD(k-1, base_minus1);
      WADD(k,   base);
      WADD(k+1, base_plus1);
    }

    WADDF(1, w);

    if ((MIDDLE - 2) % 3 > 0) {
      WADDF(2, w);
      WADDF(2, w);
    }

    if ((MIDDLE - 2) % 3 == 2) {
      WADDF(3, w);
      WADDF(3, csqTrigFancy(w));
    }
#else
#error MM_CHAIN must be 0 or 1.
#endif
  }
}

void OVERLOAD middleMul2(F2 *u, u32 x, u32 y, float factor, TrigFP32 trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  if (MIDDLE == 1) {
    WADD(0, slowTrig_N(x * y, ND) * factor);
    return;
  }

  trig += SMALL_HEIGHT;     // Skip over the MiddleMul trig table
  F2 w = TFLOAD(&trig[x]);           // x / (MIDDLE * WIDTH)

  if (MIDDLE < SHARP_MIDDLE) {
    F2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, ND / MIDDLE * 2) * factor;
    for (u32 k = 0; k < MIDDLE; ++k) { WADD(k, base); }
    WSUB(0, w);
    if (MIDDLE > 2) { WADD(2, w); }
    if (MIDDLE > 3) { WADD(3, w); WADD(3, w); }

  } else { // MIDDLE >= 5
    // F2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE);

#if 0                                   // Slower on Radeon 7, but proves the concept for use in GF61.  Might be worthwhile on poor FP64 GPUs

    TrigFP32 trig2 = trig + WIDTH;          // Skip over the fist MiddleMul2 trig table
    u32 desired_root = x * y;
    F2 base = cmulFancy(TFLOAD(&trig2[desired_root % SMALL_HEIGHT]), TFLOAD(&trig[desired_root / SMALL_HEIGHT])) * factor;   //Optimization to do: put multiply by factor in trig2 table

    WADD(0, base);
    for (u32 k = 1; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }

#elif AMDGPU && MM2_CHAIN == 0          // Oddly, Radeon 7 is faster with this version that uses more F64 ops

    F2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, ND / MIDDLE * 2) * factor;
    WADD(0, base);
    WADD(1, base);

    for (u32 k = 2; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }
    WSUBF(0, w);

#elif MM2_CHAIN == 0

    u32 mid = MIDDLE / 2;
    F2 base = slowTrig_N(x * y + x * SMALL_HEIGHT * mid, ND / MIDDLE * (mid + 1)) * factor;
    WADD(mid, base);

    F2 basehi, baselo;
    cmul_a_by_fancyb_and_conjfancyb(&basehi, &baselo, base, w);
    WADD(mid-1, baselo);
    WADD(mid+1, basehi);

    for (int i = mid-2; i >= 0; --i) {
      baselo = cmulFancy(baselo, conjugate(w));
      WADD(i, baselo);
    }

    for (int i = mid+2; i < MIDDLE; ++i) {
      basehi = cmulFancy(basehi, w);
      WADD(i, basehi);
    }

#elif MM2_CHAIN == 1
    u32 cnt = 1;
    for (u32 start = 0, sz = (MIDDLE - start + cnt - 1) / cnt; cnt > 0; --cnt, start += sz) {
      if (start + sz > MIDDLE) { --sz; }
      u32 n = (sz - 1) / 2;
      u32 mid = start + n;

      F2 base1 = slowTrig_N(x * y + x * SMALL_HEIGHT * mid, ND / MIDDLE * (mid + 1)) * factor;
      WADD(mid, base1);

      F2 base2 = base1;
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
    F2 base, base_minus1, base_plus1;
    for (u32 i = 1; ; i += 3) {
      if (i-1 == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        WADD(i-1, base_minus1);
        break;
      } else if (i == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        base = cmulFancy(base_minus1, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
        break;
      } else {
        base = slowTrig_N(x * y + x * SMALL_HEIGHT * i, ND / MIDDLE * (i + 1)) * factor;
        cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
        WADD(i+1, base_plus1);
        if (i+1 == MIDDLE-1) break;
      }
    }
#else
#error MM2_CHAIN must be 0, 1 or 2.
#endif
  }
}

// Do a partial transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local F *lds, F2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 16) {
    // See the T2 overload of middleShuffle (in the T2_GF61 section above) for the derivation of this
    // diagonal addressing, which replaces the plain (me%blockSize)*(workgroupSize/blockSize)+me/blockSize
    // write index -- a classic shared-memory transpose that conflicts badly at the default IN_SIZEX/OUT_SIZEX.
    u32 ROWS = workgroupSize / blockSize;
    u32 row = me / blockSize, col = me % blockSize;
    u32 row_w = me % ROWS, col_w = me / ROWS;
    u32 p1idx, p2idx;
    if (blockSize == 2 * ROWS) {
      p1idx = row * blockSize + (col + 2 * row) % blockSize;
      p2idx = row_w * blockSize + (col_w + 2 * row_w) % blockSize;
    } else if (ROWS == 2 * blockSize) {
      p1idx = col * ROWS + (row + 2 * col) % ROWS;
      p2idx = col_w * ROWS + (row_w + 2 * col_w) % ROWS;
    } else if (blockSize == ROWS) {
      p1idx = row * blockSize + (col + row) % blockSize;
      p2idx = row_w * blockSize + (col_w + row_w) % blockSize;
    } else {
      p1idx = col * ROWS + row;
      p2idx = me;
    }
    local F *p1 = lds + p1idx;
    local F *p2 = lds + p2idx;
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = p2[workgroupSize * i]; }
  }
}

// Do a partial transpose during fftMiddleIn/Out and write the results to global memory
void OVERLOAD middleShuffleWrite(global F2 *out, F2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  out += (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
  for (int i = 0; i < MIDDLE; ++i) { out[i * workgroupSize] = u[i]; }
}

// Do an in-place 16x16 transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local F2 *lds, F2 *u) {
  u32 me = get_local_id(0);
  u32 y = me / 16;
  u32 x = me % 16;
  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * 16 + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * 16 + y] = u[i];"
    bar();
    u[i] = lds[y * 16 + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * 16 + x ^ y] = u[i];
    bar();
    u[i] = lds[x * 16 + y ^ x];
  }
}

#if PFA

// The FP side of a PFA middle step (hybrid FFT/NTT types), as pfaMiddleIn / pfaMiddleOut for GF61 below.  The PFA-th root of unity
// w = e^(2*pi*i/PFA) is complex and on the unit circle, so the inverse radix-PFA is the forward one on SWAP_XY'd data, as for the
// binary parts of the transform.
// Forward PFA-point DFT, pairing the inputs k and PFA - k:  y_j = a0 + sum_k cos(2*pi*jk/PFA) (a_k + a_-k) + i sin(2*pi*jk/PFA) (a_k - a_-k)
void OVERLOAD pfaDftR(F2 *a) {
  const F c[PFA] = { PFA_COSF }, s[PFA] = { PFA_SINF };
  const u32 h = (PFA - 1) / 2;
  F2 sum[(PFA - 1) / 2], dif[(PFA - 1) / 2], y[PFA];
  y[0] = a[0];
  for (u32 k = 1; k <= h; ++k) { sum[k - 1] = a[k] + a[PFA - k]; dif[k - 1] = a[k] - a[PFA - k]; y[0] += sum[k - 1]; }
  for (u32 j = 1; j <= h; ++j) {
    F2 pc = sum[0] * c[j % PFA], qs = dif[0] * s[j % PFA];
    for (u32 k = 2; k <= h; ++k) { pc += sum[k - 1] * c[j * k % PFA]; qs += dif[k - 1] * s[j * k % PFA]; }
    pc += a[0];
    F2 iqs = U2(-qs.y, qs.x);
    y[j] = pc + iqs;
    y[PFA - j] = pc - iqs;
  }
  for (u32 k = 0; k < PFA; ++k) { a[k] = y[k]; }
}

// Rotate a[0..PFA-1] by a run-time amount t < PFA: right (a[i] = a[i - t]) or left (a[i] = a[i + t]), one select per bit of t
void OVERLOAD pfaRotate(F2 *a, u32 t, bool right) {
  for (u32 bit = 1; bit < PFA; bit *= 2) {
    F2 r[PFA];
    for (u32 i = 0; i < PFA; ++i) { r[i] = (t & bit) ? a[(i + (right ? PFA - bit : bit)) % PFA] : a[i]; }
    for (u32 i = 0; i < PFA; ++i) { a[i] = r[i]; }
  }
}

// The twiddle w^(x*b) of a row, w a root of order PFA_L, from the PFA middle trig table (see genMiddleTrigFP32 and pfaTwiddle for GF61)
F2 OVERLOAD pfaTwiddle(u32 x, u32 b, TrigFP32 trig) {
  TrigFP32 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  u32 desired_root = x * b;
  return cmul(TFLOAD(&trig[desired_root % PFA_BH]), TFLOAD(&trig1[desired_root / PFA_BH]));
}

// The middle step of a row: radix-PFA_M2 and the twiddles w^(WIDTH*y*k), see pfaRowMiddle for GF61
void OVERLOAD pfaRowMiddle(F2 *u, u32 y, TrigFP32 trig, bool inverse) {
#if PFA_M2 > 1
  TrigFP32 mm = trig + PFA_BH;
  if (inverse) { for (u32 k = 1; k < PFA_M2; ++k) { u[k] = cmul(u[k], TFLOAD(&mm[(k - 1) * SMALL_HEIGHT + y])); } }
#if PFA_M2 == 2
  X2(u[0], u[1]);
#else
  fft4(u);
#endif
  if (!inverse) { for (u32 k = 1; k < PFA_M2; ++k) { u[k] = cmul(u[k], TFLOAD(&mm[(k - 1) * SMALL_HEIGHT + y])); } }
#endif
}

// fftMiddleIn for PFA, see pfaMiddleIn for GF61
void OVERLOAD pfaMiddleIn(F2 *u, u32 x, u32 y, TrigFP32 trig) {
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    F2 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    F2 a[PFA];
    for (u32 j = 0; j < PFA; ++j) { a[j * (PFA_BH % PFA) % PFA] = u[m2 + PFA_M2 * j]; }
    pfaRotate(a, t, true);
    pfaDftR(a);
    for (u32 k3 = 0; k3 < PFA; ++k3) { u[m2 + PFA_M2 * k3] = a[k3]; }
  }
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, false); }
}

// fftMiddleOut for PFA on the SWAP_XY'd data of the inverse transform, see pfaMiddleOut for GF61.  factor is the stock
// normalization (NWORDS includes the factor PFA of the radix-PFA).
void OVERLOAD pfaMiddleOut(F2 *u, u32 x, u32 y, F factor, TrigFP32 trig) {
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, true); }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    F2 a[PFA];
    for (u32 k3 = 0; k3 < PFA; ++k3) { a[k3] = u[m2 + PFA_M2 * k3]; }
    pfaDftR(a);
    pfaRotate(a, t, false);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = a[j * (PFA_BH % PFA) % PFA]; }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    F2 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig) * factor;
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
}

#endif

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD fft2(GF31* u) { X2(u[0], u[1]); }

void OVERLOAD fft_MIDDLE(GF31 *u) {
#if MIDDLE == 1
  // Do nothing
#elif MIDDLE == 2
  fft2(u);
#elif MIDDLE == 4
  fft4(u);
#elif MIDDLE == 8
  fft8(u);
#elif MIDDLE == 16
  fft16(u);
#elif PFA
  // Not used: PFA's radix-PFA is done by pfaMiddleIn / pfaMiddleOut
#else
#error UNRECOGNIZED MIDDLE
#endif
}

void OVERLOAD middleMul(GF31 *u, u32 s, TrigGF31 trig) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) return;

#if !MIDDLE_CHAIN           // Read all trig values from memory

  for (u32 k = 1; k < MIDDLE; ++k) {
    WADD(k, TFLOAD(&trig[s]));
    s += SMALL_HEIGHT;
  }

#else

  GF31 w = TFLOAD(&trig[s]);         // s / BIG_HEIGHT
  WADD(1, w);
  if (MIDDLE == 2) return;

#if SHOULD_BE_FASTER
  GF31 sq = csqTrig(w);
  WADD(2, sq);
  GF31 base = ccubeTrig(sq, w);                         // GWBUG: compute w^4 as csqTriq(sq), w^6 as ccubeTrig(w2, w4), and w^5 and w^7 as cmul_a_by_b_and_conjb
  for (u32 k = 3; k < MIDDLE; ++k) {
#else
  GF31 base = csq(w);
  for (u32 k = 2; k < MIDDLE; ++k) {
#endif
    WADD(k, base);
    base = cmul(base, w);
  }

#endif

}

#if PFA

GF31 OVERLOAD pfaScale(GF31 a, Z31 s) { return U2(mul(a.x, s), mul(a.y, s)); }

// Three-point DFT using w^2 + w + 1 = 0, so it needs a single scalar multiply:  y1 = a0 - a2 + w(a1 - a2), y2 = a0 - a1 - w(a1 - a2)
void OVERLOAD pfaDft3(GF31 *a, Z31 w) {
  GF31 wd = pfaScale(sub(a[1], a[2]), w);
  GF31 y0 = add(add(a[0], a[1]), a[2]);
  GF31 y1 = add(sub(a[0], a[2]), wd);
  GF31 y2 = sub(sub(a[0], a[1]), wd);
  a[0] = y0; a[1] = y1; a[2] = y2;
}

// Forward PFA-point DFT with the root w = PFA_WPOW31[1] (the host passes PFA_WPOW31 = w^0..w^(PFA-1) and, for 7 and 11,
// PFA_C31[m] = (w^m + w^-m) / 2 and PFA_S31[m] = (w^m - w^-m) / 2).  All the constants are in Z/pZ.
void OVERLOAD pfaDftR(GF31 *a) {
  const Z31 wp[PFA] = { PFA_WPOW31 };
#if PFA == 3
  pfaDft3(a, wp[1]);
#elif PFA == 9
  // 3 x 3 Cooley-Tukey with w^3 a cube root of unity: inputs n1 + 3*n2, outputs k2 + 3*k1, twiddles w^(n1*k2)
  GF31 b[9];
  for (u32 n1 = 0; n1 < 3; ++n1) {
    GF31 t[3] = { a[n1], a[n1 + 3], a[n1 + 6] };
    pfaDft3(t, wp[3]);
    for (u32 k2 = 0; k2 < 3; ++k2) { b[3 * n1 + k2] = t[k2]; }
  }
  b[4] = pfaScale(b[4], wp[1]); b[5] = pfaScale(b[5], wp[2]); b[7] = pfaScale(b[7], wp[2]); b[8] = pfaScale(b[8], wp[4]);
  for (u32 k2 = 0; k2 < 3; ++k2) {
    GF31 t[3] = { b[k2], b[3 + k2], b[6 + k2] };
    pfaDft3(t, wp[3]);
    for (u32 k1 = 0; k1 < 3; ++k1) { a[k2 + 3 * k1] = t[k1]; }
  }
#else
  // Pair the inputs k and PFA - k:  y_j = a0 + sum_k c_jk (a_k + a_-k) + s_jk (a_k - a_-k),  y_-j = the same with -s
  const Z31 c[PFA] = { PFA_C31 }, s[PFA] = { PFA_S31 };
  const u32 h = (PFA - 1) / 2;
  GF31 sum[(PFA - 1) / 2], dif[(PFA - 1) / 2], y[PFA];
  y[0] = a[0];
  // Unrolled so that c[] and s[] are indexed by constants.  Otherwise NVIDIA's compiler kept the GF61 arrays on the stack
  // for PFA = 11, making fftMiddleIn/Out 30% slower.
  #pragma unroll
  for (u32 k = 1; k <= h; ++k) { sum[k - 1] = add(a[k], a[PFA - k]); dif[k - 1] = sub(a[k], a[PFA - k]); y[0] = add(y[0], sum[k - 1]); }
  #pragma unroll
  for (u32 j = 1; j <= h; ++j) {
    GF31 pc = pfaScale(sum[0], c[j % PFA]), qs = pfaScale(dif[0], s[j % PFA]);
    #pragma unroll
    for (u32 k = 2; k <= h; ++k) { pc = add(pc, pfaScale(sum[k - 1], c[j * k % PFA])); qs = add(qs, pfaScale(dif[k - 1], s[j * k % PFA])); }
    pc = add(a[0], pc);
    y[j] = add(pc, qs);
    y[PFA - j] = sub(pc, qs);
  }
  for (u32 k = 0; k < PFA; ++k) { a[k] = y[k]; }
#endif
}

// The inverse DFT (up to a factor PFA) is the forward one with its outputs 1..PFA-1 reversed
void OVERLOAD pfaIdftR(GF31 *a) {
  pfaDftR(a);
  for (u32 k = 1; k <= (PFA - 1) / 2; ++k) { SWAP(a[k], a[PFA - k]); }
}

// Rotate a[0..PFA-1] by a run-time amount t < PFA: right (a[i] = a[i - t]) or left (a[i] = a[i + t]), one select per bit of t
void OVERLOAD pfaRotate(GF31 *a, u32 t, bool right) {
  for (u32 bit = 1; bit < PFA; bit *= 2) {
    GF31 r[PFA];
    for (u32 i = 0; i < PFA; ++i) { r[i] = (t & bit) ? a[(i + (right ? PFA - bit : bit)) % PFA] : a[i]; }
    for (u32 i = 0; i < PFA; ++i) { a[i] = r[i]; }
  }
}

// The middle trig table for PFA, see genMiddleTrigGF31.  With w a root of order PFA_L (a row) and b < PFA_BH a binary line:
//   at 0:                            trig2[k] = w^k for k < PFA_BH
//   at PFA_BH:                       the middleMul twiddles of a row, w^(WIDTH*y*k) at (k - 1) * SMALL_HEIGHT + y for k < PFA_M2
//   at SMALL_HEIGHT * (MIDDLE - 1):  trig1[k] = w^(PFA_BH*k) for k < WIDTH
// The twiddle w^(x*b) between the width step and the middle step of a row:
GF31 OVERLOAD pfaTwiddle(u32 x, u32 b, TrigGF31 trig) {
  TrigGF31 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  u32 desired_root = x * b;
  return cmul(TFLOAD(&trig[desired_root % PFA_BH]), TFLOAD(&trig1[desired_root / PFA_BH]));
}

// The middle step of a row: radix-PFA_M2 on u[0..PFA_M2-1], then the twiddles w^(WIDTH*y*k), as fft_MIDDLE + middleMul of a
// stock MIDDLE=PFA_M2 transform.  On SWAP_XY'd data (fftMiddleOut) the same code computes the inverse, in reverse order.
void OVERLOAD pfaRowMiddle(GF31 *u, u32 y, TrigGF31 trig, bool inverse) {
#if PFA_M2 > 1
  TrigGF31 mm = trig + PFA_BH;
  if (inverse) { for (u32 k = 1; k < PFA_M2; ++k) { u[k] = cmul(u[k], TFLOAD(&mm[(k - 1) * SMALL_HEIGHT + y])); } }
#if PFA_M2 == 2
  X2(u[0], u[1]);
#else
  fft4(u);
#endif
  if (!inverse) { for (u32 k = 1; k < PFA_M2; ++k) { u[k] = cmul(u[k], TFLOAD(&mm[(k - 1) * SMALL_HEIGHT + y])); } }
#endif
}

// fftMiddleIn for PFA: x is the width frequency, y the height position.  u[m] holds line m*SMALL_HEIGHT + y, which is row
// (m*SMALL_HEIGHT + y) % PFA at binary middle index m % PFA_M2.  On output u[km + PFA_M2*k3] is row frequency k3, middle frequency km.
void OVERLOAD pfaMiddleIn(GF31 *u, u32 x, u32 y, TrigGF31 trig) {
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    GF31 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig);                 // The same for all rows
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    // u[m2 + PFA_M2*j] is row (t + j*s) % PFA.  Put the rows in order: a fixed permutation, then a rotation by t.
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    GF31 a[PFA];
    for (u32 j = 0; j < PFA; ++j) { a[j * (PFA_BH % PFA) % PFA] = u[m2 + PFA_M2 * j]; }
    pfaRotate(a, t, true);
    pfaDftR(a);
    for (u32 k3 = 0; k3 < PFA; ++k3) { u[m2 + PFA_M2 * k3] = a[k3]; }
  }
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, false); }
}

// fftMiddleOut for PFA, the inverse of pfaMiddleIn on the SWAP_XY'd data of the inverse transform.  The binary parts of the
// inverse are forward transforms (conjugating a unit-circle root inverts it), but the root of unity of the radix-PFA is in Z/pZ
// and is its own conjugate, so the inverse radix-PFA is an explicit inverse DFT.  1/PFA is folded into the row twiddle
// (multiplying by an element of Z/pZ commutes with SWAP_XY), which leaves a power-of-two scale for the carry step to shift out.
void OVERLOAD pfaMiddleOut(GF31 *u, u32 x, u32 y, TrigGF31 trig) {
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, true); }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    GF31 a[PFA];
    for (u32 k3 = 0; k3 < PFA; ++k3) { a[k3] = u[m2 + PFA_M2 * k3]; }
    pfaIdftR(a);
    pfaRotate(a, t, false);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = a[j * (PFA_BH % PFA) % PFA]; }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    GF31 w = pfaScale(pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig), PFA_INVR_31);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
}

#endif

void OVERLOAD middleMul2(GF31 *u, u32 x, u32 y, TrigGF31 trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  // First trig table comes after the MiddleMul trig table.  Second trig table comes after the first MiddleMul2 trig table.
  TrigGF31 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  TrigGF31 trig2 = trig1 + WIDTH;
  // The first trig table can be shared with MiddleMul trig table if WIDTH = HEIGHT.
  if (WIDTH == SMALL_HEIGHT) trig1 = trig;

  GF31 w = TFLOAD(&trig1[x]);         // x / (MIDDLE * WIDTH)
  u32 desired_root = x * y;
  GF31 base = cmul(TFLOAD(&trig2[desired_root % SMALL_HEIGHT]), TFLOAD(&trig1[desired_root / SMALL_HEIGHT]));

  WADD(0, base);
  for (u32 k = 1; k < MIDDLE; ++k) {
    base = cmul(base, w);
    WADD(k, base);
  }
}

// Do a partial transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local Z31 *lds, GF31 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 16) {
    // See the T2 overload of middleShuffle (in the T2_GF61 section above) for the derivation of this
    // diagonal addressing, which replaces the plain (me%blockSize)*(workgroupSize/blockSize)+me/blockSize
    // write index -- a classic shared-memory transpose that conflicts badly at the default IN_SIZEX/OUT_SIZEX.
    u32 ROWS = workgroupSize / blockSize;
    u32 row = me / blockSize, col = me % blockSize;
    u32 row_w = me % ROWS, col_w = me / ROWS;
    u32 p1idx, p2idx;
    if (blockSize == 2 * ROWS) {
      p1idx = row * blockSize + (col + 2 * row) % blockSize;
      p2idx = row_w * blockSize + (col_w + 2 * row_w) % blockSize;
    } else if (ROWS == 2 * blockSize) {
      p1idx = col * ROWS + (row + 2 * col) % ROWS;
      p2idx = col_w * ROWS + (row_w + 2 * col_w) % ROWS;
    } else if (blockSize == ROWS) {
      p1idx = row * blockSize + (col + row) % blockSize;
      p2idx = row_w * blockSize + (col_w + row_w) % blockSize;
    } else {
      p1idx = col * ROWS + row;
      p2idx = me;
    }
    local Z31 *p1 = lds + p1idx;
    local Z31 *p2 = lds + p2idx;
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = p2[workgroupSize * i]; }
  }
}

// Do a partial transpose during fftMiddleIn/Out and write the results to global memory
void OVERLOAD middleShuffleWrite(global GF31 *out, GF31 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  out += (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
  for (int i = 0; i < MIDDLE; ++i) { out[i * workgroupSize] = u[i]; }
}

// Do an in-place 16x16 transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local GF31 *lds, GF31 *u) {
  u32 me = get_local_id(0);
  u32 y = me / 16;
  u32 x = me % 16;
  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * 16 + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * 16 + y] = u[i];"
    bar();
    u[i] = lds[y * 16 + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * 16 + x ^ y] = u[i];
    bar();
    u[i] = lds[x * 16 + y ^ x];
  }
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD fft2(GF61* u) { X2(u[0], u[1]); }

void OVERLOAD fft_MIDDLE(GF61 *u) {
#if MIDDLE == 1
  // Do nothing
#elif MIDDLE == 2
  fft2(u);
#elif MIDDLE == 4
  fft4(u);
#elif MIDDLE == 8
  fft8(u);
#elif MIDDLE == 16
  fft16(u);
#elif PFA
  // Not used: PFA's radix-PFA is done by pfaMiddleIn / pfaMiddleOut
#else
#error UNRECOGNIZED MIDDLE
#endif
}

void OVERLOAD middleMul(GF61 *u, u32 s, TrigGF61 trig) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) return;

#if !MIDDLE_CHAIN           // Read all trig values from memory

  for (u32 k = 1; k < MIDDLE; ++k) {
    WADD(k, TFLOAD(&trig[s]));
    s += SMALL_HEIGHT;
  }

#else

  GF61 w = TFLOAD(&trig[s]);         // s / BIG_HEIGHT
  WADD(1, w);
  if (MIDDLE == 2) return;

#if SHOULD_BE_FASTER
  GF61 sq = csqTrig(w);
  WADD(2, sq);
  GF61 base = ccubeTrig(sq, w);                         // GWBUG: compute w^4 as csqTriq(sq), w^6 as ccubeTrig(w2, w4), and w^5 and w^7 as cmul_a_by_b_and_conjb
  for (u32 k = 3; k < MIDDLE; ++k) {
#else
  GF61 base = csq(w);
  for (u32 k = 2; k < MIDDLE; ++k) {
#endif
    WADD(k, base);
    base = cmul(base, w);
  }

#endif

}

#if PFA

GF61 OVERLOAD pfaScale(GF61 a, Z61 s) { return U2(mul(a.x, s), mul(a.y, s)); }

// Three-point DFT using w^2 + w + 1 = 0, so it needs a single scalar multiply:  y1 = a0 - a2 + w(a1 - a2), y2 = a0 - a1 - w(a1 - a2)
void OVERLOAD pfaDft3(GF61 *a, Z61 w) {
  GF61 wd = pfaScale(sub(a[1], a[2]), w);
  GF61 y0 = add(add(a[0], a[1]), a[2]);
  GF61 y1 = add(sub(a[0], a[2]), wd);
  GF61 y2 = sub(sub(a[0], a[1]), wd);
  a[0] = y0; a[1] = y1; a[2] = y2;
}

// Forward PFA-point DFT with the root w = PFA_WPOW61[1] (the host passes PFA_WPOW61 = w^0..w^(PFA-1) and, for 7 and 11,
// PFA_C61[m] = (w^m + w^-m) / 2 and PFA_S61[m] = (w^m - w^-m) / 2).  All the constants are in Z/pZ.
void OVERLOAD pfaDftR(GF61 *a) {
  const Z61 wp[PFA] = { PFA_WPOW61 };
#if PFA == 3
  pfaDft3(a, wp[1]);
#elif PFA == 9
  // 3 x 3 Cooley-Tukey with w^3 a cube root of unity: inputs n1 + 3*n2, outputs k2 + 3*k1, twiddles w^(n1*k2)
  GF61 b[9];
  for (u32 n1 = 0; n1 < 3; ++n1) {
    GF61 t[3] = { a[n1], a[n1 + 3], a[n1 + 6] };
    pfaDft3(t, wp[3]);
    for (u32 k2 = 0; k2 < 3; ++k2) { b[3 * n1 + k2] = t[k2]; }
  }
  b[4] = pfaScale(b[4], wp[1]); b[5] = pfaScale(b[5], wp[2]); b[7] = pfaScale(b[7], wp[2]); b[8] = pfaScale(b[8], wp[4]);
  for (u32 k2 = 0; k2 < 3; ++k2) {
    GF61 t[3] = { b[k2], b[3 + k2], b[6 + k2] };
    pfaDft3(t, wp[3]);
    for (u32 k1 = 0; k1 < 3; ++k1) { a[k2 + 3 * k1] = t[k1]; }
  }
#else
  // Pair the inputs k and PFA - k:  y_j = a0 + sum_k c_jk (a_k + a_-k) + s_jk (a_k - a_-k),  y_-j = the same with -s
  const Z61 c[PFA] = { PFA_C61 }, s[PFA] = { PFA_S61 };
  const u32 h = (PFA - 1) / 2;
  GF61 sum[(PFA - 1) / 2], dif[(PFA - 1) / 2], y[PFA];
  y[0] = a[0];
  // Unrolled so that c[] and s[] are indexed by constants.  Otherwise NVIDIA's compiler kept the GF61 arrays on the stack
  // for PFA = 11, making fftMiddleIn/Out 30% slower.
  #pragma unroll
  for (u32 k = 1; k <= h; ++k) { sum[k - 1] = add(a[k], a[PFA - k]); dif[k - 1] = sub(a[k], a[PFA - k]); y[0] = add(y[0], sum[k - 1]); }
  #pragma unroll
  for (u32 j = 1; j <= h; ++j) {
    GF61 pc = pfaScale(sum[0], c[j % PFA]), qs = pfaScale(dif[0], s[j % PFA]);
    #pragma unroll
    for (u32 k = 2; k <= h; ++k) { pc = add(pc, pfaScale(sum[k - 1], c[j * k % PFA])); qs = add(qs, pfaScale(dif[k - 1], s[j * k % PFA])); }
    pc = add(a[0], pc);
    y[j] = add(pc, qs);
    y[PFA - j] = sub(pc, qs);
  }
  for (u32 k = 0; k < PFA; ++k) { a[k] = y[k]; }
#endif
}

// The inverse DFT (up to a factor PFA) is the forward one with its outputs 1..PFA-1 reversed
void OVERLOAD pfaIdftR(GF61 *a) {
  pfaDftR(a);
  for (u32 k = 1; k <= (PFA - 1) / 2; ++k) { SWAP(a[k], a[PFA - k]); }
}

// Rotate a[0..PFA-1] by a run-time amount t < PFA: right (a[i] = a[i - t]) or left (a[i] = a[i + t]), one select per bit of t
void OVERLOAD pfaRotate(GF61 *a, u32 t, bool right) {
  for (u32 bit = 1; bit < PFA; bit *= 2) {
    GF61 r[PFA];
    for (u32 i = 0; i < PFA; ++i) { r[i] = (t & bit) ? a[(i + (right ? PFA - bit : bit)) % PFA] : a[i]; }
    for (u32 i = 0; i < PFA; ++i) { a[i] = r[i]; }
  }
}

// The middle trig table for PFA, see genMiddleTrigGF61.  With w a root of order PFA_L (a row) and b < PFA_BH a binary line:
//   at 0:                            trig2[k] = w^k for k < PFA_BH
//   at PFA_BH:                       the middleMul twiddles of a row, w^(WIDTH*y*k) at (k - 1) * SMALL_HEIGHT + y for k < PFA_M2
//   at SMALL_HEIGHT * (MIDDLE - 1):  trig1[k] = w^(PFA_BH*k) for k < WIDTH
// The twiddle w^(x*b) between the width step and the middle step of a row:
GF61 OVERLOAD pfaTwiddle(u32 x, u32 b, TrigGF61 trig) {
  TrigGF61 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  u32 desired_root = x * b;
  return cmul(TFLOAD(&trig[desired_root % PFA_BH]), TFLOAD(&trig1[desired_root / PFA_BH]));
}

// The middle step of a row: radix-PFA_M2 on u[0..PFA_M2-1], then the twiddles w^(WIDTH*y*k), as fft_MIDDLE + middleMul of a
// stock MIDDLE=PFA_M2 transform.  On SWAP_XY'd data (fftMiddleOut) the same code computes the inverse, in reverse order.
void OVERLOAD pfaRowMiddle(GF61 *u, u32 y, TrigGF61 trig, bool inverse) {
#if PFA_M2 > 1
  TrigGF61 mm = trig + PFA_BH;
  if (inverse) { for (u32 k = 1; k < PFA_M2; ++k) { u[k] = cmul(u[k], TFLOAD(&mm[(k - 1) * SMALL_HEIGHT + y])); } }
#if PFA_M2 == 2
  X2(u[0], u[1]);
#else
  fft4(u);
#endif
  if (!inverse) { for (u32 k = 1; k < PFA_M2; ++k) { u[k] = cmul(u[k], TFLOAD(&mm[(k - 1) * SMALL_HEIGHT + y])); } }
#endif
}

// fftMiddleIn for PFA: x is the width frequency, y the height position.  u[m] holds line m*SMALL_HEIGHT + y, which is row
// (m*SMALL_HEIGHT + y) % PFA at binary middle index m % PFA_M2.  On output u[km + PFA_M2*k3] is row frequency k3, middle frequency km.
void OVERLOAD pfaMiddleIn(GF61 *u, u32 x, u32 y, TrigGF61 trig) {
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    GF61 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig);                 // The same for all rows
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    // u[m2 + PFA_M2*j] is row (t + j*s) % PFA.  Put the rows in order: a fixed permutation, then a rotation by t.
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    GF61 a[PFA];
    for (u32 j = 0; j < PFA; ++j) { a[j * (PFA_BH % PFA) % PFA] = u[m2 + PFA_M2 * j]; }
    pfaRotate(a, t, true);
    pfaDftR(a);
    for (u32 k3 = 0; k3 < PFA; ++k3) { u[m2 + PFA_M2 * k3] = a[k3]; }
  }
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, false); }
}

// fftMiddleOut for PFA, the inverse of pfaMiddleIn on the SWAP_XY'd data of the inverse transform.  The binary parts of the
// inverse are forward transforms (conjugating a unit-circle root inverts it), but the root of unity of the radix-PFA is in Z/pZ
// and is its own conjugate, so the inverse radix-PFA is an explicit inverse DFT.  1/PFA is folded into the row twiddle
// (multiplying by an element of Z/pZ commutes with SWAP_XY), which leaves a power-of-two scale for the carry step to shift out.
void OVERLOAD pfaMiddleOut(GF61 *u, u32 x, u32 y, TrigGF61 trig) {
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, true); }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    GF61 a[PFA];
    for (u32 k3 = 0; k3 < PFA; ++k3) { a[k3] = u[m2 + PFA_M2 * k3]; }
    pfaIdftR(a);
    pfaRotate(a, t, false);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = a[j * (PFA_BH % PFA) % PFA]; }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    GF61 w = pfaScale(pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig), PFA_INVR_61);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
}

#endif

void OVERLOAD middleMul2(GF61 *u, u32 x, u32 y, TrigGF61 trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  // First trig table comes after the MiddleMul trig table.  Second trig table comes after the first MiddleMul2 trig table.
  TrigGF61 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  TrigGF61 trig2 = trig1 + WIDTH;
  // The first trig table can be shared with MiddleMul trig table if WIDTH = HEIGHT.
  if (WIDTH == SMALL_HEIGHT) trig1 = trig;

  GF61 w = TFLOAD(&trig1[x]);                      // x / (MIDDLE * WIDTH)
  u32 desired_root = x * y;
  GF61 base = cmul(TFLOAD(&trig2[desired_root % SMALL_HEIGHT]), TFLOAD(&trig1[desired_root / SMALL_HEIGHT]));

  WADD(0, base);
  for (u32 k = 1; k < MIDDLE; ++k) {
    base = cmul(base, w);
    WADD(k, base);
  }
}

// Do a partial transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local Z61 *lds, GF61 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 8) {
    local Z61 *p1 = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local Z61 *p2 = lds + me;
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = p2[workgroupSize * i]; }
  } else {
    // See the T2 overload of middleShuffle above for the derivation of this diagonal addressing.
    u32 ROWS = workgroupSize / blockSize;
    u32 row = me / blockSize, col = me % blockSize;
    u32 row_w = me % ROWS, col_w = me / ROWS;
    u32 p1idx, p2idx;
    if (blockSize == 2 * ROWS) {
      p1idx = row * blockSize + (col + 2 * row) % blockSize;
      p2idx = row_w * blockSize + (col_w + 2 * row_w) % blockSize;
    } else if (ROWS == 2 * blockSize) {
      p1idx = col * ROWS + (row + 2 * col) % ROWS;
      p2idx = col_w * ROWS + (row_w + 2 * col_w) % ROWS;
    } else if (blockSize == ROWS) {
      p1idx = row * blockSize + (col + row) % blockSize;
      p2idx = row_w * blockSize + (col_w + row_w) % blockSize;
    } else {
      p1idx = col * ROWS + row;
      p2idx = me;
    }
    local int *p1 = ((local int*) lds) + p1idx;
    local int *p2 = ((local int*) lds) + p2idx;
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

// Do a partial transpose during fftMiddleIn/Out and write the results to global memory
void OVERLOAD middleShuffleWrite(global GF61 *out, GF61 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  out += (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
  for (int i = 0; i < MIDDLE; ++i) { out[i * workgroupSize] = u[i]; }
}

// Do an in-place 16x16 transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local GF61 *lds, GF61 *u) {
  u32 me = get_local_id(0);
  u32 y = me / 16;
  u32 x = me % 16;
  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * 16 + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * 16 + y] = u[i];"
    bar();
    u[i] = lds[y * 16 + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * 16 + x ^ y] = u[i];
    bar();
    u[i] = lds[x * 16 + y ^ x];
  }
}

#endif


#undef WADD
#undef WADDF
#undef WSUB
#undef WSUBF
