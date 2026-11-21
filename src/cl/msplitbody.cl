// Copyright (C) George Woltman

// The split middle kernels' bodies (see msplit.cl), included once per type that splits with
//   MGF, MZ, MTRIG    the element, component and middle trig table types
//   MSPLIT_FP         1 for FP64 (the twiddles below), 0 for GF61 (msplitMul2 and msplitMul in msplit.cl)

#define MTT (TILE64 * TILE64)

#if MSPLIT_FP
// The chain of the elements i = j + MS * k from base (element j + MS) and the step wm (fancy):  u[0] by base / wm, u[1] by base, u[2] by
// base * wm (and u[3] by base * wm^2)
void OVERLOAD msplitChain(MGF *u, MGF base, MGF wm) {
  MGF hi, lo;
  cmul_a_by_fancyb_and_conjfancyb(&hi, &lo, base, wm);
  u[0] = cmul(u[0], lo);
  u[1] = cmul(u[1], base);
  u[2] = cmul(u[2], hi);
#if MS == 4
  u[3] = cmul(u[3], cmulFancy(hi, wm));
#endif
}

// middleMul2 of the elements i = j + MS * k:  root(ND, x * y + x * SMALL_HEIGHT * i) times factor.  As middleMul2:  one slowTrig, of
// element j + MS, and a chain of the step w^MS = root(MS * WIDTH, x), read from the split's table xt (fancy, see msplitTrig).
// MSPLIT_TRIG=1:  one slowTrig per element.
void OVERLOAD msplitMul2(MGF *u, u32 x, u32 y, u32 j, MZ factor, MTRIG trig, MTRIG xt) {
  if (MSPLIT_TRIG) {
    for (u32 k = 0; k < MS; ++k) { u32 i = j + MS * k; u[k] = cmul(u[k], slowTrig_N(x * y + x * SMALL_HEIGHT * i, ND / MIDDLE * (i + 1)) * factor); }
  } else {
    msplitChain(u, slowTrig_N(x * y + x * SMALL_HEIGHT * (j + MS), ND / MIDDLE * (j + MS + 1)) * factor, TFLOAD(&xt[x]));
  }
}

// middleMul of the elements i = j + MS * k:  root(ND, WIDTH * i * s), s < SMALL_HEIGHT.  As msplitMul2:  one slowTrig, of element
// j + MS, and a chain of the step root(MS * SMALL_HEIGHT, s) from the split's table (fancy).  MSPLIT_TRIG=1:  one slowTrig per element.
void OVERLOAD msplitMul(MGF *u, u32 s, u32 j, MTRIG trig, MTRIG xt) {
  if (MSPLIT_TRIG) {
    for (u32 k = 0; k < MS; ++k) { u32 i = j + MS * k; if (i) { u[k] = cmul(u[k], slowTrig_N(WIDTH * i * s, WIDTH * SMALL_HEIGHT * i)); } }
  } else {
    msplitChain(u, slowTrig_N(WIDTH * (j + MS) * s, WIDTH * SMALL_HEIGHT * (j + MS)), TFLOAD(&xt[WIDTH + s]));
  }
}
#endif

// fft_MIDDLE as MS x MS Cooley-Tukey over the MS threads of a point (t of G in group j).  In: u[k] = x[j + MS*k].  Out: u[k] = X[j + MS*k].
// msplitStage1 (the first radix-MS and the twiddles root(MIDDLE, j * k)) and msplitStage2 (the second radix-MS) are in msplit.cl
void OVERLOAD msplitFft(local MGF *lds, MGF *u, u32 j, u32 t, u32 G) {
  msplitStage1(u, j);
  for (u32 k = 0; k < MS; ++k) { lds[(k * MS + j) * G + t] = u[k]; }
  bar();
  for (u32 n = 0; n < MS; ++n) { u[n] = lds[(j * MS + n) * G + t]; }
  msplitStage2(u, j);
  bar();                                                  // Before the LDS is reused
}

#if !INPLACE

// middleShuffle's partial transpose (diagonal addressing, component pieces) of each u[k], thread t of group j.  Each of the MIDDLE rows
// is a transpose of its own, row j + MS*k in its own part of the LDS.
void OVERLOAD msplitShuffle(local MZ *lds, MGF *u, u32 workgroupSize, u32 blockSize, u32 j, u32 t) {
  u32 p1idx, p2idx;
  middleShuffleIdxMe(t, workgroupSize, blockSize, &p1idx, &p2idx);
  local MZ *p1 = lds + j * MS * workgroupSize + p1idx;
  local MZ *p2 = lds + j * MS * workgroupSize + p2idx;
  for (u32 k = 0; k < MS; ++k) { p1[k * workgroupSize] = u[k].x; }
  bar();
  for (u32 k = 0; k < MS; ++k) { u[k].x = p2[k * workgroupSize]; }
  bar();
  for (u32 k = 0; k < MS; ++k) { p1[k * workgroupSize] = u[k].y; }
  bar();
  for (u32 k = 0; k < MS; ++k) { u[k].y = p2[k * workgroupSize]; }
}

// fftMiddleIn, workgroups of MS * IN_WG threads and an LDS of MS_LDS(IN_WG) elements.  wait:  this type is the first in the iteration.
void OVERLOAD msplitIn(P(MGF) out, CP(MGF) in, MTRIG trig, MTRIG xt, local MGF *lds, bool wait) {
  MGF u[MS];
  u32 SIZEY = IN_WG / IN_SIZEX;
  u32 N = WIDTH / IN_SIZEX;
  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;
  u32 me = get_local_id(0);
  u32 j = me / IN_WG;
  u32 t = me % IN_WG;
  u32 mx = t % IN_SIZEX;
  u32 my = t / IN_SIZEX;
  u32 x = gx * IN_SIZEX + mx;
  u32 y = gy * SIZEY + my;

  if (wait) { dependentLaunchWait(); }

  readMiddleInLines(u, in, y, x, j, MS);
  msplitMul2(u, x, y, j, 1, trig, xt);
  msplitFft(lds, u, j, t, IN_WG);
  msplitMul(u, y, j, trig, xt);

  dependentLaunch();

#if MIDDLE_IN_LDS_TRANSPOSE
  msplitShuffle((local MZ *) lds, u, IN_WG, IN_SIZEX, j, t);
  out += t;
#else
  out += mx * SIZEY + my;
#endif
  writeMiddleInLines(out, u, gy, gx, j, MS);
}

// fftMiddleOut, workgroups of MS * OUT_WG threads and an LDS of MS_LDS(OUT_WG) elements
void OVERLOAD msplitOut(P(MGF) out, CP(MGF) in, MTRIG trig, MTRIG xt, local MGF *lds, MZ factor) {
  MGF u[MS];
  u32 SIZEY = OUT_WG / OUT_SIZEX;
  u32 N = SMALL_HEIGHT / OUT_SIZEX;
  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;
  u32 me = get_local_id(0);
  u32 j = me / OUT_WG;
  u32 t = me % OUT_WG;
  u32 mx = t % OUT_SIZEX;
  u32 my = t / OUT_SIZEX;
  u32 x = gx * OUT_SIZEX + mx;
  u32 y = gy * SIZEY + my;

  dependentLaunchWait();

  readMiddleOutLines(u, in, y, x, j, MS);
  msplitMul(u, x, j, trig, xt);
  msplitFft(lds, u, j, t, OUT_WG);
  msplitMul2(u, y, x, j, factor, trig, xt);

  dependentLaunch();

#if MIDDLE_OUT_LDS_TRANSPOSE
  msplitShuffle((local MZ *) lds, u, OUT_WG, OUT_SIZEX, j, t);
  out += t;
#else
  out += mx * SIZEY + my;
#endif
  writeMiddleOutLines(out, u, gy, gx, j, MS);
}

#else

// The TILE64 x TILE64 transpose of each u[k] (middleShuffle), each of the MS thread groups in its own part of the LDS
void OVERLOAD msplitShuffle(local MGF *lds, MGF *u, u32 j, u32 t) {
  u32 x = t % TILE64, y = t / TILE64;
  local MGF *l = lds + j * MS * MTT;
  for (u32 k = 0; k < MS; ++k) { l[k * MTT + x * TILE64 + (y ^ x)] = u[k]; }
  bar();
  for (u32 k = 0; k < MS; ++k) { u[k] = l[k * MTT + y * TILE64 + (x ^ y)]; }
}

// fftMiddleIn in place, workgroups of MS * MTT threads and an LDS of MS_LDS(MTT) elements
void OVERLOAD msplitIn(P(MGF) out, CP(MGF) in, MTRIG trig, MTRIG xt, local MGF *lds, bool wait) {
  MGF u[MS];
  u32 g = get_group_id(0);
#if INPLACE64 == 1 || INPLACE64 == 3
  u32 N = SMALL_HEIGHT / TILE64;
  u32 starty = g % N * TILE64;
  u32 startx = g / N * TILE64;
#else
  u32 N = WIDTH / TILE64;
  u32 startx = g % N * TILE64;
  u32 starty = g / N * TILE64;
#endif
  u32 me = get_local_id(0);
  u32 j = me / MTT;
  u32 t = me % MTT;
  u32 x = startx + t % TILE64;
  u32 y = starty + t / TILE64;

  if (wait) { dependentLaunchWait(); }

  readMiddleInLines(u, in, y, x, j, MS);
  msplitMul2(u, x, y, j, 1, trig, xt);
  msplitFft(lds, u, j, t, MTT);
  msplitMul(u, y, j, trig, xt);

  dependentLaunch();

  msplitShuffle(lds, u, j, t);
  writeMiddleInLines(out, u, y, x, j, MS);
}

// fftMiddleOut in place, workgroups of MS * MTT threads and an LDS of MS_LDS(MTT) elements
void OVERLOAD msplitOut(P(MGF) out, CP(MGF) in, MTRIG trig, MTRIG xt, local MGF *lds, MZ factor) {
  MGF u[MS];
  u32 g = get_group_id(0);
#if INPLACE64 == 1 || INPLACE64 == 3
  u32 N = SMALL_HEIGHT / TILE64;
  u32 startx = g % N * TILE64;
  u32 starty = g / N * TILE64;
#else
  u32 N = WIDTH / TILE64;
  u32 starty = g % N * TILE64;
  u32 startx = g / N * TILE64;
#endif
  u32 me = get_local_id(0);
  u32 j = me / MTT;
  u32 t = me % MTT;
  u32 x = startx + t % TILE64;
  u32 y = starty + t / TILE64;

  dependentLaunchWait();

  readMiddleOutLines(u, in, y, x, j, MS);
  msplitMul(u, x, j, trig, xt);
  msplitFft(lds, u, j, t, MTT);
  msplitMul2(u, y, x, j, factor, trig, xt);

  dependentLaunch();

  msplitShuffle(lds, u, j, t);
  writeMiddleOutLines(out, u, y, x, j, MS);
}

#endif

#undef MTT
