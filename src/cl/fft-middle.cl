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

// The LDS write (p1idx) and read (p2idx) element indices of this thread for middleShuffle's partial transpose: the plain transpose
// writes at (me % blockSize) * (workgroupSize / blockSize) + me / blockSize, which conflicts badly at the default IN_SIZEX/OUT_SIZEX.
// The "diagonal" address outer*BIG + (inner + k*outer) % BIG, where BIG/ROWS are blockSize and workgroupSize/blockSize in whichever
// order is larger, "outer" is whichever of row/col has the smaller range, and k is 1 (square) or 2 (2:1 ratio), moves the same values
// without bank conflicts, for 4-byte as well as whole 8-byte elements (verified by simulation).  Other ratios fall back to the plain
// transpose.
void middleShuffleIdx(u32 workgroupSize, u32 blockSize, u32 *p1idx, u32 *p2idx) {
  u32 me = get_local_id(0);
  u32 ROWS = workgroupSize / blockSize;
  u32 row = me / blockSize, col = me % blockSize;
  u32 row_w = me % ROWS, col_w = me / ROWS;
  if (blockSize == 2 * ROWS) {
    *p1idx = row * blockSize + (col + 2 * row) % blockSize;
    *p2idx = row_w * blockSize + (col_w + 2 * row_w) % blockSize;
  } else if (ROWS == 2 * blockSize) {
    *p1idx = col * ROWS + (row + 2 * col) % ROWS;
    *p2idx = col_w * ROWS + (row_w + 2 * col_w) % ROWS;
  } else if (blockSize == ROWS) {
    *p1idx = row * blockSize + (col + row) % blockSize;
    *p2idx = row_w * blockSize + (col_w + row_w) % blockSize;
  } else {
    *p1idx = col * ROWS + row;
    *p2idx = me;
  }
}

// The LDS row that middleShuffle writes u[i] to.  PFA fftMiddleOut passes pfaRows: pfaMiddleOut leaves out its rotation of the rows
// by t, and writing u[i] to row pfaInIndex(i, y) puts each value on the line the rotation would have given it (see pfaInIndex).
#define SHUFFLE_ROW(i) (pfaRows ? MIDDLE_IN_INDEX(i, y) : (i))

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

    Trig trig2 = trig + WIDTH;          // Skip over the first MiddleMul2 trig table
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
    // A loop with a compile-time trip count (not an endless loop left with break): the compiler must unroll it fully to index u[]
    // with constants.  Otherwise u[] goes to local memory -- at MIDDLE=8 that made fftMiddleIn/Out 2.8x slower on a TITAN V.
    T2 base, base_minus1, base_plus1;
    for (u32 i = 1; i <= MIDDLE; i += 3) {
      if (i-1 == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        WADD(i-1, base_minus1);
      } else if (i == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        base = cmulFancy(base_minus1, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
      } else {
        base = slowTrig_N(x * y + x * SMALL_HEIGHT * i, ND / MIDDLE * (i + 1)) * factor;
        cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
        WADD(i+1, base_plus1);
      }
    }
#else
#error MM2_CHAIN must be 0, 1 or 2.
#endif
  }
}

// Do a partial transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local T *lds, T2 *u, u32 workgroupSize, u32 blockSize, bool pfaRows, u32 y) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 8) {
    u32 p1idx, p2idx;
    middleShuffleIdx(workgroupSize, blockSize, &p1idx, &p2idx);
    local T *p1 = lds + p1idx;
    local T *p2 = lds + p2idx;
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = p2[workgroupSize * i]; }
  } else {
    // Half the LDS of the MIDDLE <= 8 case: move 4-byte pieces.  middleShuffleIdx's diagonal addressing is conflict-free for both.
    u32 p1idx, p2idx;
    middleShuffleIdx(workgroupSize, blockSize, &p1idx, &p2idx);
    local int *p1 = ((local int*) lds) + p1idx;
    local int *p2 = ((local int*) lds) + p2idx;
    int4 *pu = (int4 *)u;

    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = pu[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = pu[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].y = p2[workgroupSize * i]; }
    bar();

    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = pu[i].z; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].z = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = pu[i].w; }
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
  u32 y = me / INPLACE_TILE;
  u32 x = me % INPLACE_TILE;

  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * INPLACE_TILE + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * INPLACE_TILE + y] = u[i];"
    bar();
    u[i] = lds[y * INPLACE_TILE + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * INPLACE_TILE + x ^ y] = u[i];
    bar();
    u[i] = lds[x * INPLACE_TILE + y ^ x];
  }
}


#if PFA

// The FP side of a PFA middle step (hybrid FFT/NTT types), as pfaMiddleIn / pfaMiddleOut for GF61 below.  The PFA-th root of unity
// w = e^(2*pi*i/PFA) is complex and on the unit circle, so the inverse radix-PFA is the forward one on SWAP_XY'd data, as for the
// binary parts of the transform.
// Forward PFA-point DFT, y_j = sum_k a_k w^(jk) in natural order: the hand-tuned fftN, which all use w = e^(2*pi*i/N)
void OVERLOAD pfaDftR(T2 *a) {
#if PFA == 3
  fft3(a);
#elif PFA == 5
  fft5(a);
#elif PFA == 7
  fft7(a);
#elif PFA == 9
  fft9(a);
#elif PFA == 11
  fft11(a);
#elif PFA == 15
  fft15(a);
#else
#error PFA
#endif
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
  return slowTrig_N(x * b * PFA, ND);              // On the fly as in non-PFA middleMul2 (faster than the two table loads):  w has order ND / PFA
}

// The middle step of a row: radix-PFA_M2 and the twiddles w^(WIDTH*y*k), see pfaRowMiddle for GF61
// The row twiddles w^(WIDTH*y*k) for 0 < k < PFA_M2, each made just before its multiply (fewer live values than computing them all
// first).  PFA_M2 > 2 chains them from one load, t, t^2, t^3..., as non-PFA middleMul.
void OVERLOAD pfaRowTwiddles(T2 *u, u32 y, Trig mm) {
#if PFA_M2 > 2
  T2 t = TFLOAD(&mm[y]);
  u[1] = cmul(u[1], t);
  T2 tk = csqTrig(t);
  for (u32 k = 2; k < PFA_M2; ++k) {
    u[k] = cmul(u[k], tk);
    if (k + 1 < PFA_M2) { tk = cmul(tk, t); }
  }
#else
  for (u32 k = 1; k < PFA_M2; ++k) { u[k] = cmul(u[k], TFLOAD(&mm[(k - 1) * SMALL_HEIGHT + y])); }
#endif
}

void OVERLOAD pfaRowMiddle(T2 *u, u32 y, Trig trig, bool inverse) {
#if PFA_M2 > 1
  Trig mm = trig + PFA_BH;
  if (inverse) { pfaRowTwiddles(u, y, mm); }
#if PFA_M2 == 2
  X2(u[0], u[1]);
#else
  fft4(u);
#endif
  if (!inverse) { pfaRowTwiddles(u, y, mm); }
#endif
}

// fftMiddleIn for PFA, see pfaMiddleIn for GF61
void OVERLOAD pfaMiddleIn(T2 *u, u32 x, u32 y, Trig trig) {
#if PFA_M2 > 2                                   // Chained, w0 * s^m2 with s = w^(x * SMALL_HEIGHT):  2 slowTrig_N instead of PFA_M2
  T2 w = pfaTwiddle(x, y, trig);
  for (u32 j = 0; j < PFA; ++j) { u[PFA_M2 * j] = cmul(u[PFA_M2 * j], w); }
  T2 s = slowTrig_N(x * SMALL_HEIGHT * PFA, ND);   // made here, just before its first use
  for (u32 m2 = 1; m2 < PFA_M2; ++m2) {
    w = cmul(w, s);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
#else
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    T2 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
#endif
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    T2 a[PFA];                                   // readMiddleInLine did the rotation by t (see pfaInIndex)
    for (u32 j = 0; j < PFA; ++j) { a[j * (PFA_BH % PFA) % PFA] = u[m2 + PFA_M2 * j]; }
    pfaDftR(a);
    for (u32 k3 = 0; k3 < PFA; ++k3) { u[m2 + PFA_M2 * k3] = a[k3]; }
  }
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, false); }
}

// fftMiddleOut for PFA on the SWAP_XY'd data of the inverse transform, see pfaMiddleOut for GF61.  factor is the stock
// normalization (NWORDS includes the factor PFA of the radix-PFA).
void OVERLOAD pfaMiddleOut(T2 *u, u32 x, u32 y, T factor, Trig trig, bool rotate) {
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, true); }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    T2 a[PFA];
    for (u32 k3 = 0; k3 < PFA; ++k3) { a[k3] = u[m2 + PFA_M2 * k3]; }
    pfaDftR(a);
    if (rotate) { pfaRotate(a, t, false); }       // Else middleShuffle does it (see SHUFFLE_ROW)
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = a[j * (PFA_BH % PFA) % PFA]; }
  }
#if PFA_M2 > 2                                   // Chained, as pfaMiddleIn
  T2 w = pfaTwiddle(x, y, trig) * factor;
  for (u32 j = 0; j < PFA; ++j) { u[PFA_M2 * j] = cmul(u[PFA_M2 * j], w); }
  T2 s = slowTrig_N(x * SMALL_HEIGHT * PFA, ND);   // made here, just before its first use
  for (u32 m2 = 1; m2 < PFA_M2; ++m2) {
    w = cmul(w, s);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
#else
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    T2 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig) * factor;
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
#endif
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

    TrigFP32 trig2 = trig + WIDTH;          // Skip over the first MiddleMul2 trig table
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
    // A compile-time trip count so that the loop is fully unrolled and u[] stays in registers, see the FP64 middleMul2
    F2 base, base_minus1, base_plus1;
    for (u32 i = 1; i <= MIDDLE; i += 3) {
      if (i-1 == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        WADD(i-1, base_minus1);
      } else if (i == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        base = cmulFancy(base_minus1, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
      } else {
        base = slowTrig_N(x * y + x * SMALL_HEIGHT * i, ND / MIDDLE * (i + 1)) * factor;
        cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
        WADD(i+1, base_plus1);
      }
    }
#else
#error MM2_CHAIN must be 0, 1 or 2.
#endif
  }
}

// Do a partial transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local F *lds, F2 *u, u32 workgroupSize, u32 blockSize, bool pfaRows, u32 y) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 16) {
    // Diagonal addressing, conflict-free at the default IN_SIZEX/OUT_SIZEX (see middleShuffleIdx)
    u32 p1idx, p2idx;
    middleShuffleIdx(workgroupSize, blockSize, &p1idx, &p2idx);
    local F *p1 = lds + p1idx;
    local F *p2 = lds + p2idx;
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = u[i].y; }
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
  u32 y = me / INPLACE_TILE;
  u32 x = me % INPLACE_TILE;
  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * INPLACE_TILE + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * INPLACE_TILE + y] = u[i];"
    bar();
    u[i] = lds[y * INPLACE_TILE + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * INPLACE_TILE + x ^ y] = u[i];
    bar();
    u[i] = lds[x * INPLACE_TILE + y ^ x];
  }
}

#if PFA

// The FP side of a PFA middle step (hybrid FFT/NTT types), as pfaMiddleIn / pfaMiddleOut for GF61 below.  The PFA-th root of unity
// w = e^(2*pi*i/PFA) is complex and on the unit circle, so the inverse radix-PFA is the forward one on SWAP_XY'd data, as for the
// binary parts of the transform.
// Forward PFA-point DFT, y_j = sum_k a_k w^(jk) in natural order: the hand-tuned fftN, which all use w = e^(2*pi*i/N)
void OVERLOAD pfaDftR(F2 *a) {
#if PFA == 3
  fft3(a);
#elif PFA == 5
  fft5(a);
#elif PFA == 7
  fft7(a);
#elif PFA == 9
  fft9(a);
#elif PFA == 11
  fft11(a);
#elif PFA == 15
  fft15(a);
#else
#error PFA
#endif
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
  return slowTrig_N(x * b * PFA, ND);              // On the fly as in non-PFA middleMul2 (faster than the two table loads):  w has order ND / PFA
}

// The middle step of a row: radix-PFA_M2 and the twiddles w^(WIDTH*y*k), see pfaRowMiddle for GF61
// The row twiddles w^(WIDTH*y*k) for 0 < k < PFA_M2, each made just before its multiply (fewer live values than computing them all
// first).  PFA_M2 > 2 chains them from one load, t, t^2, t^3..., as non-PFA middleMul.
void OVERLOAD pfaRowTwiddles(F2 *u, u32 y, TrigFP32 mm) {
#if PFA_M2 > 2
  F2 t = TFLOAD(&mm[y]);
  u[1] = cmul(u[1], t);
  F2 tk = csqTrig(t);
  for (u32 k = 2; k < PFA_M2; ++k) {
    u[k] = cmul(u[k], tk);
    if (k + 1 < PFA_M2) { tk = cmul(tk, t); }
  }
#else
  for (u32 k = 1; k < PFA_M2; ++k) { u[k] = cmul(u[k], TFLOAD(&mm[(k - 1) * SMALL_HEIGHT + y])); }
#endif
}

void OVERLOAD pfaRowMiddle(F2 *u, u32 y, TrigFP32 trig, bool inverse) {
#if PFA_M2 > 1
  TrigFP32 mm = trig + PFA_BH;
  if (inverse) { pfaRowTwiddles(u, y, mm); }
#if PFA_M2 == 2
  X2(u[0], u[1]);
#else
  fft4(u);
#endif
  if (!inverse) { pfaRowTwiddles(u, y, mm); }
#endif
}

// fftMiddleIn for PFA, see pfaMiddleIn for GF61
void OVERLOAD pfaMiddleIn(F2 *u, u32 x, u32 y, TrigFP32 trig) {
#if PFA_M2 > 2                                   // Chained, w0 * s^m2 with s = w^(x * SMALL_HEIGHT):  2 slowTrig_N instead of PFA_M2
  F2 w = pfaTwiddle(x, y, trig);
  for (u32 j = 0; j < PFA; ++j) { u[PFA_M2 * j] = cmul(u[PFA_M2 * j], w); }
  F2 s = slowTrig_N(x * SMALL_HEIGHT * PFA, ND);   // made here, just before its first use
  for (u32 m2 = 1; m2 < PFA_M2; ++m2) {
    w = cmul(w, s);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
#else
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    F2 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
#endif
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    F2 a[PFA];                                   // readMiddleInLine did the rotation by t (see pfaInIndex)
    for (u32 j = 0; j < PFA; ++j) { a[j * (PFA_BH % PFA) % PFA] = u[m2 + PFA_M2 * j]; }
    pfaDftR(a);
    for (u32 k3 = 0; k3 < PFA; ++k3) { u[m2 + PFA_M2 * k3] = a[k3]; }
  }
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, false); }
}

// fftMiddleOut for PFA on the SWAP_XY'd data of the inverse transform, see pfaMiddleOut for GF61.  factor is the stock
// normalization (NWORDS includes the factor PFA of the radix-PFA).
void OVERLOAD pfaMiddleOut(F2 *u, u32 x, u32 y, F factor, TrigFP32 trig, bool rotate) {
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, true); }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    F2 a[PFA];
    for (u32 k3 = 0; k3 < PFA; ++k3) { a[k3] = u[m2 + PFA_M2 * k3]; }
    pfaDftR(a);
    if (rotate) { pfaRotate(a, t, false); }       // Else middleShuffle does it (see SHUFFLE_ROW)
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = a[j * (PFA_BH % PFA) % PFA]; }
  }
#if PFA_M2 > 2                                   // Chained, as pfaMiddleIn
  F2 w = pfaTwiddle(x, y, trig) * factor;
  for (u32 j = 0; j < PFA; ++j) { u[PFA_M2 * j] = cmul(u[PFA_M2 * j], w); }
  F2 s = slowTrig_N(x * SMALL_HEIGHT * PFA, ND);   // made here, just before its first use
  for (u32 m2 = 1; m2 < PFA_M2; ++m2) {
    w = cmul(w, s);
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
#else
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    F2 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig) * factor;
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
#endif
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

#if !MIDDLE_CHAIN31           // Read all trig values from memory

  for (u32 k = 1; k < MIDDLE; ++k) {
    WADD(k, TFLOAD(&trig[s]));
    s += SMALL_HEIGHT;
  }

#else

  // The powers of w (MIDDLE is 2, 4, 8 or 16): w^2 (and w^4, w^6 for MIDDLE 8) squared, w^3 cubed, w^5, w^7 as the pair w^6 * w, w^6 / w
  GF31 w = TFLOAD(&trig[s]);         // s / BIG_HEIGHT
  WADD(1, w);
  if (MIDDLE == 2) return;
  GF31 w2 = csqTrig(w), w3 = ccubeTrig(w2, w);
  WADD(2, w2);
  WADD(3, w3);
  if (MIDDLE == 4) return;
  if (MIDDLE == 8) {
    GF31 w4 = csqTrig(w2), w6 = csqTrig(w3), w5, w7;
    cmul_a_by_b_and_conjb(&w7, &w5, w6, w);
    WADD(4, w4);
    WADD(5, w5);
    WADD(6, w6);
    WADD(7, w7);
    return;
  }
  // MIDDLE == 16: a serial chain.  With u[16] live, computing the powers by squaring and pairing holds several at once and costs dozens of
  // registers (fftMiddleIn 100 -> 136 on sm_120).
  GF31 base = w3;
  for (u32 k = 4; k < MIDDLE; ++k) {
    base = cmul(base, w);
    WADD(k, base);
  }

#endif

}

#if PFA

// Forward PFA-point DFT, y_j = sum_k a_k w^(jk) in natural order: the hand-tuned fftN (GF31 section), w = the root of pfaRootOfUnity
void OVERLOAD pfaDftR(GF31 *a) {
#if PFA == 3
  fft3(a);
#elif PFA == 7
  fft7(a);
#elif PFA == 9
  fft9(a);
#elif PFA == 11
  fft11(a);
#else
#error PFA
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
//   at SMALL_HEIGHT * (2*PFA_M2 - 1): trig2[k] / PFA for k < PFA_BH (GF31, GF61:  fftMiddleOut's normalization, see pfaTwiddleInv)
//   at SMALL_HEIGHT * (MIDDLE - 1):  trig1[k] = w^(PFA_BH*k) for k < WIDTH
// The twiddle w^(x*b) between the width step and the middle step of a row:
GF31 OVERLOAD pfaTwiddle(u32 x, u32 b, TrigGF31 trig) {
  TrigGF31 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  u32 desired_root = x * b;
  return cmul(TFLOAD(&trig[desired_root % PFA_BH]), TFLOAD(&trig1[desired_root / PFA_BH]));
}
// pfaTwiddle times 1/PFA (fftMiddleOut's normalization) from the copy of trig2 times 1/PFA at SMALL_HEIGHT * (2 * PFA_M2 - 1)
GF31 OVERLOAD pfaTwiddleInv(u32 x, u32 b, TrigGF31 trig) {
  TrigGF31 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  TrigGF31 trig2inv = trig + SMALL_HEIGHT * (2 * PFA_M2 - 1);
  u32 desired_root = x * b;
  return cmul(TFLOAD(&trig2inv[desired_root % PFA_BH]), TFLOAD(&trig1[desired_root / PFA_BH]));
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

// fftMiddleIn for PFA: x is the width frequency, y the height position.  u[m] holds line pfaInIndex(m, y)*SMALL_HEIGHT + y
// (binary middle index m % PFA_M2), see pfaInIndex.  On output u[km + PFA_M2*k3] is row frequency k3, middle frequency km.
void OVERLOAD pfaMiddleIn(GF31 *u, u32 x, u32 y, TrigGF31 trig) {
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    GF31 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig);                 // The same for all rows
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    // Put the rows in order: a fixed permutation (readMiddleInLine did the rotation by t = (m2*SMALL_HEIGHT + y) % PFA, see pfaInIndex)
    GF31 a[PFA];
    for (u32 j = 0; j < PFA; ++j) { a[j * (PFA_BH % PFA) % PFA] = u[m2 + PFA_M2 * j]; }
    pfaDftR(a);
    for (u32 k3 = 0; k3 < PFA; ++k3) { u[m2 + PFA_M2 * k3] = a[k3]; }
  }
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, false); }
}

// fftMiddleOut for PFA, the inverse of pfaMiddleIn on the SWAP_XY'd data of the inverse transform.  The binary parts of the
// inverse are forward transforms (conjugating a unit-circle root inverts it), but the root of unity of the radix-PFA is in Z/pZ
// and is its own conjugate, so the inverse radix-PFA is an explicit inverse DFT.  1/PFA is folded into the row twiddle
// (multiplying by an element of Z/pZ commutes with SWAP_XY), which leaves a power-of-two scale for the carry step to shift out.
void OVERLOAD pfaMiddleOut(GF31 *u, u32 x, u32 y, TrigGF31 trig, bool rotate) {
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, true); }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    GF31 a[PFA];
    for (u32 k3 = 0; k3 < PFA; ++k3) { a[k3] = u[m2 + PFA_M2 * k3]; }
    pfaIdftR(a);
    if (rotate) { pfaRotate(a, t, false); }       // Else middleShuffle does it (see SHUFFLE_ROW)
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = a[j * (PFA_BH % PFA) % PFA]; }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    GF31 w = pfaTwiddleInv(x, y + SMALL_HEIGHT * m2, trig);
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
void OVERLOAD middleShuffle(local Z31 *lds, GF31 *u, u32 workgroupSize, u32 blockSize, bool pfaRows, u32 y) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 16) {
    // Diagonal addressing, conflict-free at the default IN_SIZEX/OUT_SIZEX (see middleShuffleIdx)
    u32 p1idx, p2idx;
    middleShuffleIdx(workgroupSize, blockSize, &p1idx, &p2idx);
    local Z31 *p1 = lds + p1idx;
    local Z31 *p2 = lds + p2idx;
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = u[i].y; }
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

// All seven rows are written before the barrier, so a row permutation
// can be encoded in LDS addresses instead of a register select network.
void middleShuffleTile31(local GF31 *lds, GF31 *u, bool rotate, u32 heightPos) {
  u32 p1,p2; middleShuffleIdx((INPLACE_TILE31*INPLACE_TILE31),INPLACE_TILE31,&p1,&p2);
  for (u32 i=0;i<MIDDLE;++i) {
    u32 row=rotate ? MIDDLE_IN_INDEX(i,heightPos) : i;
    lds[row*(INPLACE_TILE31*INPLACE_TILE31)+p1]=u[i];
  }
  bar();
  for (u32 i=0;i<MIDDLE;++i) u[i]=lds[i*(INPLACE_TILE31*INPLACE_TILE31)+p2];
}

void OVERLOAD middleShuffle(local GF31 *lds, GF31 *u) {
  u32 me = get_local_id(0);
  u32 y = me / INPLACE_TILE31;
  u32 x = me % INPLACE_TILE31;
  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * INPLACE_TILE31 + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * INPLACE_TILE31 + y] = u[i];"
    bar();
    u[i] = lds[y * INPLACE_TILE31 + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * INPLACE_TILE31 + x ^ y] = u[i];
    bar();
    u[i] = lds[x * INPLACE_TILE31 + y ^ x];
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

#if !MIDDLE_CHAIN61           // Read all trig values from memory

  // Left to itself ROCm keeps this loop and the one in middleMul2 rolled at MIDDLE=16, which puts u[] in scratch
  #pragma unroll
  for (u32 k = 1; k < MIDDLE; ++k) {
    WADD(k, TFLOAD(&trig[s]));
    s += SMALL_HEIGHT;
  }

#else

  // The powers of w (MIDDLE is 2, 4, 8 or 16): w^2 (and w^4, w^6 for MIDDLE 8) squared, w^3 cubed, w^5, w^7 as the pair w^6 * w, w^6 / w
  GF61 w = TFLOAD(&trig[s]);         // s / BIG_HEIGHT
  WADD(1, w);
  if (MIDDLE == 2) return;
  GF61 w2 = csqTrig(w), w3 = ccubeTrig(w2, w);
  WADD(2, w2);
  WADD(3, w3);
  if (MIDDLE == 4) return;
  if (MIDDLE == 8) {
    GF61 w4 = csqTrig(w2), w6 = csqTrig(w3), w5, w7;
    cmul_a_by_b_and_conjb(&w7, &w5, w6, w);
    WADD(4, w4);
    WADD(5, w5);
    WADD(6, w6);
    WADD(7, w7);
    return;
  }
  // MIDDLE == 16: a serial chain.  With u[16] live, computing the powers by squaring and pairing holds several at once and costs dozens of
  // registers (fftMiddleIn 100 -> 136 on sm_120).
  GF61 base = w3;
  for (u32 k = 4; k < MIDDLE; ++k) {
    base = cmul(base, w);
    WADD(k, base);
  }

#endif

}

#if PFA

// Forward PFA-point DFT, y_j = sum_k a_k w^(jk) in natural order: the hand-tuned fftN (GF61 section), w = the root of pfaRootOfUnity
void OVERLOAD pfaDftR(GF61 *a) {
#if PFA == 3
  fft3(a);
#elif PFA == 5
  fft5(a);
#elif PFA == 7
  fft7(a);
#elif PFA == 9
  fft9(a);
#elif PFA == 11
  fft11(a);
#elif PFA == 15
  fft15(a);
#else
#error PFA
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
//   at SMALL_HEIGHT * (2*PFA_M2 - 1): trig2[k] / PFA for k < PFA_BH (GF31, GF61:  fftMiddleOut's normalization, see pfaTwiddleInv)
//   at SMALL_HEIGHT * (MIDDLE - 1):  trig1[k] = w^(PFA_BH*k) for k < WIDTH
// The twiddle w^(x*b) between the width step and the middle step of a row:
GF61 OVERLOAD pfaTwiddle(u32 x, u32 b, TrigGF61 trig) {
  TrigGF61 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  u32 desired_root = x * b;
  return cmul(TFLOAD(&trig[desired_root % PFA_BH]), TFLOAD(&trig1[desired_root / PFA_BH]));
}
// pfaTwiddle times 1/PFA (fftMiddleOut's normalization) from the copy of trig2 times 1/PFA at SMALL_HEIGHT * (2 * PFA_M2 - 1)
GF61 OVERLOAD pfaTwiddleInv(u32 x, u32 b, TrigGF61 trig) {
  TrigGF61 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  TrigGF61 trig2inv = trig + SMALL_HEIGHT * (2 * PFA_M2 - 1);
  u32 desired_root = x * b;
  return cmul(TFLOAD(&trig2inv[desired_root % PFA_BH]), TFLOAD(&trig1[desired_root / PFA_BH]));
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

// fftMiddleIn for PFA: x is the width frequency, y the height position.  u[m] holds line pfaInIndex(m, y)*SMALL_HEIGHT + y
// (binary middle index m % PFA_M2), see pfaInIndex.  On output u[km + PFA_M2*k3] is row frequency k3, middle frequency km.
void OVERLOAD pfaMiddleIn(GF61 *u, u32 x, u32 y, TrigGF61 trig) {
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    GF61 w = pfaTwiddle(x, y + SMALL_HEIGHT * m2, trig);                 // The same for all rows
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = cmul(u[m2 + PFA_M2 * j], w); }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    // Put the rows in order: a fixed permutation (readMiddleInLine did the rotation by t = (m2*SMALL_HEIGHT + y) % PFA, see pfaInIndex)
    GF61 a[PFA];
    for (u32 j = 0; j < PFA; ++j) { a[j * (PFA_BH % PFA) % PFA] = u[m2 + PFA_M2 * j]; }
    pfaDftR(a);
    for (u32 k3 = 0; k3 < PFA; ++k3) { u[m2 + PFA_M2 * k3] = a[k3]; }
  }
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, false); }
}

// fftMiddleOut for PFA, the inverse of pfaMiddleIn on the SWAP_XY'd data of the inverse transform.  The binary parts of the
// inverse are forward transforms (conjugating a unit-circle root inverts it), but the root of unity of the radix-PFA is in Z/pZ
// and is its own conjugate, so the inverse radix-PFA is an explicit inverse DFT.  1/PFA is folded into the row twiddle
// (multiplying by an element of Z/pZ commutes with SWAP_XY), which leaves a power-of-two scale for the carry step to shift out.
void OVERLOAD pfaMiddleOut(GF61 *u, u32 x, u32 y, TrigGF61 trig, bool rotate) {
  for (u32 k3 = 0; k3 < PFA; ++k3) { pfaRowMiddle(u + PFA_M2 * k3, y, trig, true); }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    u32 t = (m2 * SMALL_HEIGHT + y) % PFA;
    GF61 a[PFA];
    for (u32 k3 = 0; k3 < PFA; ++k3) { a[k3] = u[m2 + PFA_M2 * k3]; }
    pfaIdftR(a);
    if (rotate) { pfaRotate(a, t, false); }       // Else middleShuffle does it (see SHUFFLE_ROW)
    for (u32 j = 0; j < PFA; ++j) { u[m2 + PFA_M2 * j] = a[j * (PFA_BH % PFA) % PFA]; }
  }
  for (u32 m2 = 0; m2 < PFA_M2; ++m2) {
    GF61 w = pfaTwiddleInv(x, y + SMALL_HEIGHT * m2, trig);
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
  #pragma unroll
  for (u32 k = 1; k < MIDDLE; ++k) {
    base = cmul(base, w);
    WADD(k, base);
  }
}

// Do a partial transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local Z61 *lds, GF61 *u, u32 workgroupSize, u32 blockSize, bool pfaRows, u32 y) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 8) {
    u32 p1idx, p2idx;
    middleShuffleIdx(workgroupSize, blockSize, &p1idx, &p2idx);
    local Z61 *p1 = lds + p1idx;
    local Z61 *p2 = lds + p2idx;
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = p2[workgroupSize * i]; }
  } else {
    // Half the LDS of the MIDDLE <= 8 case: move 4-byte pieces, see middleShuffleIdx.
    u32 p1idx, p2idx;
    middleShuffleIdx(workgroupSize, blockSize, &p1idx, &p2idx);
    local int *p1 = ((local int*) lds) + p1idx;
    local int *p2 = ((local int*) lds) + p2idx;
    int4 *pu = (int4 *)u;

    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = pu[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = pu[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].y = p2[workgroupSize * i]; }
    bar();

    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = pu[i].z; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].z = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[SHUFFLE_ROW(i) * workgroupSize] = pu[i].w; }
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

// All seven rows are written before the barrier, so a row permutation
// can be encoded in LDS addresses instead of a register select network.
void middleShuffleTile61(local GF61 *lds, GF61 *u, bool rotate, u32 heightPos) {
  u32 p1,p2; middleShuffleIdx((INPLACE_TILE61*INPLACE_TILE61),INPLACE_TILE61,&p1,&p2);
  for (u32 i=0;i<MIDDLE;++i) {
    u32 row=rotate ? MIDDLE_IN_INDEX(i,heightPos) : i;
    lds[row*(INPLACE_TILE61*INPLACE_TILE61)+p1]=u[i];
  }
  bar();
  for (u32 i=0;i<MIDDLE;++i) u[i]=lds[i*(INPLACE_TILE61*INPLACE_TILE61)+p2];
}

void OVERLOAD middleShuffle(local GF61 *lds, GF61 *u) {
  u32 me = get_local_id(0);
  u32 y = me / INPLACE_TILE61;
  u32 x = me % INPLACE_TILE61;
  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * INPLACE_TILE61 + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * INPLACE_TILE61 + y] = u[i];"
    bar();
    u[i] = lds[y * INPLACE_TILE61 + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * INPLACE_TILE61 + x ^ y] = u[i];
    bar();
    u[i] = lds[x * INPLACE_TILE61 + y ^ x];
  }
}

#endif


#undef WADD
#undef WADDF
#undef WSUB
#undef WSUBF
