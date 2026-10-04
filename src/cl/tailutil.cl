// Copyright (C) Mihai Preda

// TAIL_TRIGS setting:
//      2 = No memory accesses, trig values computed from scratch.  Good for excellent DP GPUs such as Titan V or Radeon VII Pro.
//      1 = Limited memory accesses and some DP computation.  Tuned for Radeon VII a GPU with good DP performance.
//      0 = No DP computation.  Trig vaules read from memory.  Good for GPUs with poor DP performance (a typical consumer grade GPU).
#if !defined(TAIL_TRIGS)
#define TAIL_TRIGS      2                         // Default is compute trig values from scratch for FP64
#endif
#if !defined(TAIL_TRIGS31)
#define TAIL_TRIGS31    0                         // Default is read all trig values from memory for GF31
#endif
#if !defined(TAIL_TRIGS32)
#define TAIL_TRIGS32    2                         // Default is compute trig values from scratch for FP32
#endif
#if !defined(TAIL_TRIGS61)
#define TAIL_TRIGS61    0                         // Default is read all trig values from memory for GF61
#endif

// TAIL_KERNELS setting:
//      0 = single wide, single kernel
//      1 = single wide, two kernels
//      2 = double wide, single kernel
//      3 = double wide, two kernels
#if !defined(TAIL_KERNELS)
#define TAIL_KERNELS    2                         // Default is double-wide tailSquare with a single kernel
#endif
#if TAIL_KERNELS < 0 || TAIL_KERNELS > 3
#error TAIL_KERNELS must be 0..3
#endif
#define SINGLE_WIDE    (TAIL_KERNELS < 2)         // Old single-wide tailSquare vs. new double-wide tailSquare
#define SINGLE_KERNEL  ((TAIL_KERNELS & 1) == 0)  // TailSquare uses a single kernel vs. two kernels

// 64-bit implementations of reverse routines

#if FFT_FP64 || NTT_GF61

void OVERLOAD reverse(local T2_GF61 *lds2, T2_GF61 *u, bool bump) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me + bump;

  if (SHUFL_BYTES_H >= 8) {
    local T2_GF61 *lds = lds2;
    bar(WG);
#if NH == 8
    lds[revMe + 0 * WG] = u[3];
    lds[revMe + 1 * WG] = u[2];
    lds[revMe + 2 * WG] = u[1];
    lds[bump ? ((revMe + 3 * WG) % (4 * WG)) : (revMe + 3 * WG)] = u[0];
#elif NH == 4
    lds[revMe + 0 * WG] = u[1];
    lds[bump ? ((revMe + WG) % (2 * WG)) : (revMe + WG)] = u[0];
#endif
    bar(WG);
    for (i32 i = 0; i < NH/2; ++i) { u[i] = lds[i * WG + me]; }
  }

  else if (SHUFL_BYTES_H == 4) {
    local T_Z61 *lds = (local T_Z61 *) lds2;
    bar(WG);
#if NH == 8
    lds[revMe + 0 * WG] = u[3].x;
    lds[revMe + 1 * WG] = u[2].x;
    lds[revMe + 2 * WG] = u[1].x;
    lds[bump ? ((revMe + 3 * WG) % (4 * WG)) : (revMe + 3 * WG)] = u[0].x;
#elif NH == 4
    lds[revMe + 0 * WG] = u[1].x;
    lds[bump ? ((revMe + WG) % (2 * WG)) : (revMe + WG)] = u[0].x;
#endif
    bar(WG);
    for (i32 i = 0; i < NH/2; ++i) { u[i].x = lds[i * WG + me]; }
    bar(WG);
#if NH == 8
    lds[revMe + 0 * WG] = u[3].y;
    lds[revMe + 1 * WG] = u[2].y;
    lds[revMe + 2 * WG] = u[1].y;
    lds[bump ? ((revMe + 3 * WG) % (4 * WG)) : (revMe + 3 * WG)] = u[0].y;
#elif NH == 4
    lds[revMe + 0 * WG] = u[1].y;
    lds[bump ? ((revMe + WG) % (2 * WG)) : (revMe + WG)] = u[0].y;
#endif
    bar(WG);
    for (i32 i = 0; i < NH/2; ++i) { u[i].y = lds[i * WG + me]; }
  }
}

void OVERLOAD reverseLine(local T2_GF61 *lds, T2_GF61 *u) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me;

  if (SHUFL_BYTES_H == 16) {
    local T2_GF61 *ldsOut = lds + revMe;
    local T2_GF61 *ldsIn = lds + me;
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsOut[WG * (NH - 1 - i)] = u[i]; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i] = ldsIn[WG * i]; }
  }

  else if (SHUFL_BYTES_H == 8) {
    local T_Z61 *ldsOut = (local T_Z61 *) lds + revMe;
    local T_Z61 *ldsIn = (local T_Z61 *) lds + me;
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsOut[WG * (NH - 1 - i)] = u[i].x; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i].x = ldsIn[WG * i]; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsOut[WG * (NH - 1 - i)] = u[i].y; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i].y = ldsIn[WG * i]; }
  }

  else if (SHUFL_BYTES_H == 4) {
    local int *ldsOut = (local int *) lds + revMe;
    local int *ldsIn = (local int *) lds + me;
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsOut[WG * (NH - 1 - i)] = as_int4(u[i]).x; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { int4 tmp = as_int4(u[i]); tmp.x = ldsIn[WG * i]; u[i] = as_T2_GF61(tmp); }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsOut[WG * (NH - 1 - i)] = as_int4(u[i]).y; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { int4 tmp = as_int4(u[i]); tmp.y = ldsIn[WG * i]; u[i] = as_T2_GF61(tmp); }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsOut[WG * (NH - 1 - i)] = as_int4(u[i]).z; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { int4 tmp = as_int4(u[i]); tmp.z = ldsIn[WG * i]; u[i] = as_T2_GF61(tmp); }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsOut[WG * (NH - 1 - i)] = as_int4(u[i]).w; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { int4 tmp = as_int4(u[i]); tmp.w = ldsIn[WG * i]; u[i] = as_T2_GF61(tmp); }
  }
}

#if PFA
// reverseLine, optionally offset by one: u[p] = u[(SMALL_HEIGHT - p) % SMALL_HEIGHT] for p = i * WG + me.  The PFA FP tail pairs
// the kx = 0 lines of rows k3 and PFA - k3 this way.  Only the LDS index differs from reverseLine.
void OVERLOAD reverseLineMaybeBump(local T2_GF61 *lds, T2_GF61 *u, bool bump) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me + bump;
#define REV_IDX(i) ((WG * (NH - 1 - (i)) + revMe) % (NH * WG))

  if (SHUFL_BYTES_H == 16) {
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { lds[REV_IDX(i)] = u[i]; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i] = lds[WG * i + me]; }
  }

  else if (SHUFL_BYTES_H == 8) {
    local T_Z61 *ldsZ = (local T_Z61 *) lds;
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsZ[REV_IDX(i)] = u[i].x; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i].x = ldsZ[WG * i + me]; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsZ[REV_IDX(i)] = u[i].y; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i].y = ldsZ[WG * i + me]; }
  }

  else if (SHUFL_BYTES_H == 4) {
    local int *ldsI = (local int *) lds;
    for (u32 c = 0; c < 4; ++c) {
      bar(WG);
      for (u32 i = 0; i < NH; ++i) { int4 t = as_int4(u[i]); ldsI[REV_IDX(i)] = c == 0 ? t.x : c == 1 ? t.y : c == 2 ? t.z : t.w; }
      bar(WG);
      for (u32 i = 0; i < NH; ++i) {
        int4 t = as_int4(u[i]); int v = ldsI[WG * i + me];
        if (c == 0) { t.x = v; } else if (c == 1) { t.y = v; } else if (c == 2) { t.z = v; } else { t.w = v; }
        u[i] = as_T2_GF61(t);
      }
    }
  }
#undef REV_IDX
}
#endif

//
// These versions are for the kernel(s) that use a double-wide workgroup (u in half the workgroup, v in the other half)
//

void OVERLOAD reverse2(local T2_GF61 *lds2, T2_GF61 *u) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;

  if (SBMUL(2) * SHUFL_BYTES_H >= 8) {
    local T2_GF61 *lds = LDSsharing_ptr(lds2, 2);
    // For NH=8, u[0] to u[3] are left unchanged.  Write to lds:
    //  u[7]rev   u[6]rev   u[5]rev   u[4]rev
    //  v[7]rev   v[6]rev   v[5]rev   v[4]rev
    LDStx_start(lds2, 2);
    for (u32 i = 0; i < NH/2; ++i) { lds[((NH/2 - i) * WG - (me >= WG ? 1 : 0) - lowMe) % (NH/2 * WG)] = u[NH/2 + i]; }
    // For NH=8, read from lds into u[i]:
    //  u[4] =   u[7]rev   v[7]rev
    //  u[5] =   u[6]rev   v[6]rev
    //  u[6] =   u[5]rev   v[5]rev
    //  u[7] =   u[4]rev   v[4]rev
    LDSbar(2);
    for (u32 i = 0; i < NH/2; ++i) { u[NH/2 + i] = lds[i * WG + lowMe]; }
    LDStx_end(lds2, 2);
  }

  else if (SBMUL(2) * SHUFL_BYTES_H == 4) {
    local T_Z61 *lds = LDSsharing_ptr((local T_Z61 *)lds2, 2);
    LDStx_start(lds2, 2);
    for (u32 i = 0; i < NH/2; ++i) { lds[((NH/2 - i) * WG - (me >= WG ? 1 : 0) - lowMe) % (NH/2 * WG)] = u[NH/2 + i].x; }
    LDSbar(2);
    for (u32 i = 0; i < NH/2; ++i) { u[NH/2 + i].x = lds[i * WG + lowMe]; }
    LDSbar(2);
    for (u32 i = 0; i < NH/2; ++i) { lds[((NH/2 - i) * WG - (me >= WG ? 1 : 0) - lowMe) % (NH/2 * WG)] = u[NH/2 + i].y; }
    LDSbar(2);
    for (u32 i = 0; i < NH/2; ++i) { u[NH/2 + i].y = lds[i * WG + lowMe]; }
    LDStx_end(lds2, 2);
  }
}

// Double-wide tailSquare: reverse u[NH/2..NH-1] for either kind of line pair with a single code path.  Normal pairs cross the
// reversed parts between the two half-workgroups (revCrossLine); the pair of lines 0 and H/2 pairs each line with itself, line 0
// offset by one (reverse2).  Only the LDS addresses differ, so they are selected rather than branched on.  A separate line-0 code
// path (run by a single workgroup) made the ROCm compiler size the whole kernel's VGPRs for it.
void OVERLOAD revLineOrSelf(local T2_GF61 *lds2, T2_GF61 *u, bool line0) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;
  u32 myHalf = me / WG;
  u32 outHalf = line0 ? myHalf : myHalf ^ 1;
  u32 off = line0 ? myHalf : 1;

  if (SHUFL_BYTES_H >= 8) {
    local T2_GF61 *ldsOut = lds2 + outHalf * (LDS_SHUFL_BYTES(2) / sizeof(T2_GF61));
    local T2_GF61 *ldsIn  = lds2 + myHalf * (LDS_SHUFL_BYTES(2) / sizeof(T2_GF61));
    bar();
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[(WG * (NH/2 - 1 - i) + WG - lowMe - off) % (NH/2 * WG)] = u[i + NH/2]; }
    bar();
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2] = ldsIn[WG * i + lowMe]; }
  }

  else if (SHUFL_BYTES_H == 4) {
    local T_Z61 *ldsOut = (local T_Z61 *) lds2 + outHalf * (LDS_SHUFL_BYTES(2) / sizeof(T_Z61));
    local T_Z61 *ldsIn  = (local T_Z61 *) lds2 + myHalf * (LDS_SHUFL_BYTES(2) / sizeof(T_Z61));
    bar();
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[(WG * (NH/2 - 1 - i) + WG - lowMe - off) % (NH/2 * WG)] = u[i + NH/2].x; }
    bar();
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2].x = ldsIn[WG * i + lowMe]; }
    bar();
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[(WG * (NH/2 - 1 - i) + WG - lowMe - off) % (NH/2 * WG)] = u[i + NH/2].y; }
    bar();
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2].y = ldsIn[WG * i + lowMe]; }
  }

  // One last bar() is needed when sharing LDS memory, as in revCrossLine.
  if (SHARING_LDS(2)) bar();
}

#if PFA
// revLineOrSelf for the double-wide PFA FP tail, which has a third kind of line pair: the kx = 0 lines of rows k3 and PFA - k3 (bump)
// pair element ky with element -ky of the other line, i.e. cross the halves offset by one.  That leaves element 0 of the first half's
// line next to element SMALL_HEIGHT/2 of the second's (and vice versa) instead of next to element 0.  So on the way in (fwd) the
// second half's lane 0 sends its element 0 in place of its element SMALL_HEIGHT/2, which it keeps in u[0]: the first half then pairs
// the two elements 0, the second half the two elements SMALL_HEIGHT/2 (with -t^2, see pairSq).  On the way back it swaps u[0] and
// u[NH/2] to restore its line.  The same code path handles all three kinds of pair; only selects differ.
void OVERLOAD revLinePfa(local T2_GF61 *lds2, T2_GF61 *u, bool line0, bool bump, bool fwd) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;
  u32 myHalf = me / WG;
  u32 outHalf = line0 ? myHalf : myHalf ^ 1;
  u32 off = line0 ? myHalf : !bump;
  bool fix = !line0 && bump && myHalf && lowMe == 0;
  T2_GF61 keep = u[NH/2];
  u[NH/2] = (fix && fwd) ? u[0] : keep;     // Sent in place of element SMALL_HEIGHT/2.  Unconditional assignments, no stack.

  if (SHUFL_BYTES_H >= 8) {
    local T2_GF61 *ldsOut = lds2 + outHalf * (LDS_SHUFL_BYTES(2) / sizeof(T2_GF61));
    local T2_GF61 *ldsIn  = lds2 + myHalf * (LDS_SHUFL_BYTES(2) / sizeof(T2_GF61));
    bar();
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[(WG * (NH/2 - 1 - i) + WG - lowMe - off) % (NH/2 * WG)] = u[i + NH/2]; }
    bar();
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2] = ldsIn[WG * i + lowMe]; }
  }

  else if (SHUFL_BYTES_H == 4) {
    local T_Z61 *ldsOut = (local T_Z61 *) lds2 + outHalf * (LDS_SHUFL_BYTES(2) / sizeof(T_Z61));
    local T_Z61 *ldsIn  = (local T_Z61 *) lds2 + myHalf * (LDS_SHUFL_BYTES(2) / sizeof(T_Z61));
    bar();
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[(WG * (NH/2 - 1 - i) + WG - lowMe - off) % (NH/2 * WG)] = u[i + NH/2].x; }
    bar();
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2].x = ldsIn[WG * i + lowMe]; }
    bar();
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[(WG * (NH/2 - 1 - i) + WG - lowMe - off) % (NH/2 * WG)] = u[i + NH/2].y; }
    bar();
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2].y = ldsIn[WG * i + lowMe]; }
  }

  T2_GF61 a0 = u[0], h0 = u[NH/2];
  u[0] = fix ? (fwd ? keep : h0) : a0;
  u[NH/2] = (fix && !fwd) ? a0 : h0;

  // One last bar() is needed when sharing LDS memory, as in revCrossLine.
  if (SHARING_LDS(2)) bar();
}
#endif

// This is used to reverse the second part of a line, and cross the reversed parts between the halves.
void OVERLOAD revCrossLine(local T2_GF61 *lds2, T2_GF61 *u) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;
  u32 revLowMe = WG - 1 - lowMe;

  if (SHUFL_BYTES_H >= 8) {
    local T2_GF61 *ldsOut = lds2;
    local T2_GF61 *ldsIn = lds2;
    if (me < WG) ldsOut += LDS_SHUFL_BYTES(2) / sizeof(T2_GF61); // Crossing LDS halves
    else ldsIn += LDS_SHUFL_BYTES(2) / sizeof(T2_GF61);          // Staying within LDS halves (just like shufl)
    bar();   // we need a full bar because we're crossing halves
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[WG * (NH/2 - 1 - i) + revLowMe] = u[i + NH/2]; }
    bar();   // we need a full bar because we just crossed halves.  LDS reads are compatible with future shufl calls.
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2] = ldsIn[WG * i + lowMe]; }
    // One last bar() is needed when sharing LDS memory.  This is because when sharing a workgroup will write to more than its own LDS area.
    if (SHARING_LDS(2)) bar();
  }

  else if (SHUFL_BYTES_H == 4) {
    local T_Z61 *ldsOut = (local T_Z61 *) lds2;
    local T_Z61 *ldsIn = (local T_Z61 *) lds2;
    if (me < WG) ldsOut += LDS_SHUFL_BYTES(2) / sizeof(T_Z61);
    else ldsIn += LDS_SHUFL_BYTES(2) / sizeof(T_Z61);
    bar();   // we need a full bar because we're crossing halves
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[WG * (NH/2 - 1 - i) + revLowMe] = u[i + NH/2].x; }
    bar();   // we need a full bar because we just crossed halves
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2].x = ldsIn[WG * i + lowMe]; }
    bar();   // we need a full bar because we're crossing halves
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[WG * (NH/2 - 1 - i) + revLowMe] = u[i + NH/2].y; }
    bar();   // we need a full bar because we just crossed halves.  LDS reads are compatible with future shufl calls.
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2].y = ldsIn[WG * i + lowMe]; }
    // One last bar() is needed when sharing LDS memory.  This is because when sharing a workgroup will write to more than its own LDS area.
    if (SHARING_LDS(2)) bar();
  }
}

#endif


/**************************************************************************/
/*        Similar to above, but for an FFT based on FP32 or GF31          */
/**************************************************************************/

#if FFT_FP32 || NTT_GF31

void OVERLOAD reverse(local F2_GF31 *lds, F2_GF31 *u, bool bump) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me + bump;

  if (SHUFL_BYTES_H >= 4) {
    bar(WG);
#if NH == 8
    lds[revMe + 0 * WG] = u[3];
    lds[revMe + 1 * WG] = u[2];
    lds[revMe + 2 * WG] = u[1];
    lds[bump ? ((revMe + 3 * WG) % (4 * WG)) : (revMe + 3 * WG)] = u[0];
#elif NH == 4
    lds[revMe + 0 * WG] = u[1];
    lds[bump ? ((revMe + WG) % (2 * WG)) : (revMe + WG)] = u[0];
#endif
    bar(WG);
    for (i32 i = 0; i < NH/2; ++i) { u[i] = lds[i * WG + me]; }
  }
}

void OVERLOAD reverseLine(local F2_GF31 *lds, F2_GF31 *u) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me;

  if (SHUFL_BYTES_H >= 8) {
    local F2_GF31 *ldsOut = lds + revMe;
    local F2_GF31 *ldsIn = lds + me;
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsOut[WG * (NH - 1 - i)] = u[i]; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i] = ldsIn[WG * i]; }
  }

  else if (SHUFL_BYTES_H == 4) {
    local F_Z31 *ldsOut = (local F_Z31 *) lds + revMe;
    local F_Z31 *ldsIn = (local F_Z31 *) lds + me;
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsOut[WG * (NH - 1 - i)] = u[i].x; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i].x = ldsIn[WG * i]; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsOut[WG * (NH - 1 - i)] = u[i].y; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i].y = ldsIn[WG * i]; }
  }
}

#if PFA
// 32-bit version of reverseLineMaybeBump above
void OVERLOAD reverseLineMaybeBump(local F2_GF31 *lds, F2_GF31 *u, bool bump) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me + bump;
#define REV_IDX(i) ((WG * (NH - 1 - (i)) + revMe) % (NH * WG))

  if (SHUFL_BYTES_H >= 8) {
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { lds[REV_IDX(i)] = u[i]; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i] = lds[WG * i + me]; }
  }

  else if (SHUFL_BYTES_H == 4) {
    local F_Z31 *ldsZ = (local F_Z31 *) lds;
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsZ[REV_IDX(i)] = u[i].x; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i].x = ldsZ[WG * i + me]; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { ldsZ[REV_IDX(i)] = u[i].y; }
    bar(WG);
    for (u32 i = 0; i < NH; ++i) { u[i].y = ldsZ[WG * i + me]; }
  }
#undef REV_IDX
}
#endif

//
// These versions are for the kernel(s) that use a double-wide workgroup (u in half the workgroup, v in the other half)
//

void OVERLOAD reverse2(local F2_GF31 *lds2, F2_GF31 *u) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;

  if (SBMUL(2) * SHUFL_BYTES_H >= 4) {
    local F2_GF31 *lds = LDSsharing_ptr(lds2, 2);
    // For NH=8, u[0] to u[3] are left unchanged.  Write to lds:
    //  u[7]rev   u[6]rev   u[5]rev   u[4]rev
    //  v[7]rev   v[6]rev   v[5]rev   v[4]rev
    LDStx_start(lds2, 2);
    for (u32 i = 0; i < NH/2; ++i) { lds[((NH/2 - i) * WG - (me >= WG ? 1 : 0) - lowMe) % (NH/2 * WG)] = u[NH/2 + i]; }
    // For NH=8, read from lds into u[i]:
    //  u[4] =   u[7]rev   v[7]rev
    //  u[5] =   u[6]rev   v[6]rev
    //  u[6] =   u[5]rev   v[5]rev
    //  u[7] =   u[4]rev   v[4]rev
    LDSbar(2);
    for (u32 i = 0; i < NH/2; ++i) { u[NH/2 + i] = lds[i * WG + lowMe]; }
    LDStx_end(lds2, 2);
  }
}

// 32-bit version of revLineOrSelf above: one code path for both kinds of line pair in the double-wide tailSquare/tailMul.
void OVERLOAD revLineOrSelf(local F2_GF31 *lds2, F2_GF31 *u, bool line0) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;
  u32 myHalf = me / WG;
  u32 outHalf = line0 ? myHalf : myHalf ^ 1;
  u32 off = line0 ? myHalf : 1;

  if (SHUFL_BYTES_H >= 4) {
    local F2_GF31 *ldsOut = lds2 + outHalf * (LDS_SHUFL_BYTES(2) / sizeof(F2_GF31));
    local F2_GF31 *ldsIn  = lds2 + myHalf * (LDS_SHUFL_BYTES(2) / sizeof(F2_GF31));
    bar();
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[(WG * (NH/2 - 1 - i) + WG - lowMe - off) % (NH/2 * WG)] = u[i + NH/2]; }
    bar();
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2] = ldsIn[WG * i + lowMe]; }
  }

  // One last bar() is needed when sharing LDS memory, as in revCrossLine.
  if (SHARING_LDS(2)) bar();
}

#if PFA
// 32-bit version of revLinePfa above
void OVERLOAD revLinePfa(local F2_GF31 *lds2, F2_GF31 *u, bool line0, bool bump, bool fwd) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;
  u32 myHalf = me / WG;
  u32 outHalf = line0 ? myHalf : myHalf ^ 1;
  u32 off = line0 ? myHalf : !bump;
  bool fix = !line0 && bump && myHalf && lowMe == 0;
  F2_GF31 keep = u[NH/2];
  u[NH/2] = (fix && fwd) ? u[0] : keep;     // Sent in place of element SMALL_HEIGHT/2.  Unconditional assignments, no stack.

  if (SHUFL_BYTES_H >= 4) {
    local F2_GF31 *ldsOut = lds2 + outHalf * (LDS_SHUFL_BYTES(2) / sizeof(F2_GF31));
    local F2_GF31 *ldsIn  = lds2 + myHalf * (LDS_SHUFL_BYTES(2) / sizeof(F2_GF31));
    bar();
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[(WG * (NH/2 - 1 - i) + WG - lowMe - off) % (NH/2 * WG)] = u[i + NH/2]; }
    bar();
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2] = ldsIn[WG * i + lowMe]; }
  }

  F2_GF31 a0 = u[0], h0 = u[NH/2];
  u[0] = fix ? (fwd ? keep : h0) : a0;
  u[NH/2] = (fix && !fwd) ? a0 : h0;

  // One last bar() is needed when sharing LDS memory, as in revCrossLine.
  if (SHARING_LDS(2)) bar();
}
#endif

// This is used to reverse the second part of a line, and cross the reversed parts between the halves.
void OVERLOAD revCrossLine(local F2_GF31 *lds2, F2_GF31 *u) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;
  u32 revLowMe = WG - 1 - lowMe;

  if (SHUFL_BYTES_H >= 4) {
    local F2_GF31 *ldsOut = lds2;
    local F2_GF31 *ldsIn = lds2;
    if (me < WG) ldsOut += LDS_SHUFL_BYTES(2) / sizeof(F2);
    else ldsIn += LDS_SHUFL_BYTES(2) / sizeof(F2);
    bar();   // we need a full bar because we're crossing halves
    for (u32 i = 0; i < NH/2; ++i) { ldsOut[WG * (NH/2 - 1 - i) + revLowMe] = u[i + NH/2]; }
    bar();   // we need a full bar because we just crossed halves.  LDS reads are compatible with future shufl calls.
    for (u32 i = 0; i < NH/2; ++i) { u[i + NH/2] = ldsIn[WG * i + lowMe]; }
    // One last bar() is needed when sharing LDS memory.  This is because when sharing a workgroup will write to more than its own LDS area.
    if (SHARING_LDS(2)) bar();
  }
}

#endif
