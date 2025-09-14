// Copyright (C) Mihai Preda

#include "math.cl"

#if FFT_FP64

void OVERLOAD reverse(u32 WG, local T2 *lds, T2 *u, bool bump) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me + bump;

  bar();

#if NH == 8
  lds[revMe + 0 * WG] = u[3];
  lds[revMe + 1 * WG] = u[2];
  lds[revMe + 2 * WG] = u[1];
  lds[bump ? ((revMe + 3 * WG) % (4 * WG)) : (revMe + 3 * WG)] = u[0];
#elif NH == 4
  lds[revMe + 0 * WG] = u[1];
  lds[bump ? ((revMe + WG) % (2 * WG)) : (revMe + WG)] = u[0];
#else
#error
#endif

  bar();
  for (i32 i = 0; i < NH/2; ++i) { u[i] = lds[i * WG + me]; }
}

void OVERLOAD reverseLine(u32 WG, local T2 *lds2, T2 *u) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me;

  local T2 *lds = lds2 + revMe;
  bar();
  for (u32 i = 0; i < NH; ++i) { lds[WG * (NH - 1 - i)] = u[i]; }

  lds = lds2 + me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i] = lds[WG * i]; }
}

// This is used to reverse the second part of a line, and cross the reversed parts between the halves.
void OVERLOAD revCrossLine(u32 WG, local T2* lds2, T2 *u, u32 n, bool writeSecondHalf) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;
  
  u32 revLowMe = WG - 1 - lowMe;

  for (u32 i = 0; i < n; ++i) { lds2[WG * n * writeSecondHalf + WG * (n - 1 - i) + revLowMe] = u[i]; }

  bar();   // we need a full bar because we're crossing halves

  for (u32 i = 0; i < n; ++i) { u[i] = lds2[WG * n * !writeSecondHalf + WG * i + lowMe]; }
}

//
// These versions are for the kernel(s) that uses a double-wide workgroup (u in half the workgroup, v in the other half)
//

void OVERLOAD reverse2(local T2 *lds, T2 *u) {
  u32 me = get_local_id(0);
  
  // For NH=8, u[0] to u[3] are left unchanged.  Write to lds:
  //	u[7]rev   u[6]rev
  //	u[5]rev   u[4]rev
  //	v[7]rev   v[6]rev
  //	v[5]rev   v[4]rev
  bar();
  for (u32 i = 0; i < NH / 2; ++i) {
    u32 j = (i * G_H + me % G_H);
    lds[me < G_H ? ((NH/2)*G_H - j) % ((NH/2)*G_H) : NH*G_H-1 - j] = u[NH/2 + i];
  }
  // For NH=8, read from lds into u[i]:
  //	u[4] =   u[7]rev   v[7]rev
  //	u[5] =   u[6]rev   v[6]rev
  //	u[6] =   u[5]rev   v[5]rev
  //	u[7] =   u[4]rev   v[4]rev
  bar();
  lds += me % G_H + (me / G_H) * NH/2 * G_H;
  for (u32 i = 0; i < NH / 2; ++i) { u[NH/2 + i] = lds[i * G_H]; }
}

// Somewhat similar to reverseLine.
// The u values are in threads < G_H, the v values to reverse in threads >= G_H.
// Whereas reverseLine leaves u values alone.  This reverseLine moves u values around
// so that pairSq2 can easily operate on pairs.  This means for NH = 4, web output:
//      u[0]    u[1]            // Returned in u[0]
//      u[2]    u[3]            // Returned in u[1]
//      v[3]rev v[2]rev         // Returned in u[2]
//      v[1]rev v[0]rev         // Returned in u[3]
void OVERLOAD reverseLine2(local T2 *lds, T2 *u) {
  u32 me = get_local_id(0);

// NOTE:  It is important that this routine use lds memory in coordination with shufl2.  Failure to do so would require an
// unqualified bar() call here.  Specifically, the u values are stored in the upper half of lds memory (SMALL_HEIGHT T2 values).
// The v values are stored in the lower half of lds memory (the next SMALL_HEIGHT T2 values).

  if (G_H > WAVEFRONT) bar();

// For NH=4, the lds indices (where to write each incoming u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
// That means saving to lds using index: me < G_H ? me % G_H + i * G_H : 8*G_H-1 - me % G_H - i * G_H

#if 1
  local T2 *ldsOut = lds + (me < G_H ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsOutInc = (me < G_H) ? G_H : -G_H;
  for (u32 i = 0; i < NH; ++i, ldsOut += ldsOutInc) { *ldsOut = u[i]; }

  lds += me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i] = lds[i * 2*G_H]; }
#else
  local T *ldsOut = (local T *) lds + (me < G_H ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsOutInc = (me < G_H) ? G_H : -G_H;
  for (u32 i = 0; i < NH; ++i, ldsOut += ldsOutInc) { ldsOut[0] = u[i].x; ldsOut[NH*2*G_H] = u[i].y; }

  local T *ldsIn = (local T *) lds + me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i].x = ldsIn[i * 2*G_H]; u[i].y = ldsIn[NH*2*G_H + i * 2*G_H]; }
#endif
}

// Undo a reverseLine2
void OVERLOAD unreverseLine2(local T2 *lds, T2 *u) {
  u32 me = get_local_id(0);

// NOTE:  It is important that this routine use lds memory in coordination with reverseLine2 and shufl2.  By initially
// writing to the lds locations that reverseLine2 read from we do not need an initial bar() call here.  Also, by reading
// from the lds locations that shufl2 will use (u values in the upper half of lds memory, v values in the lower half of
// lds memory) we can issue a qualified bar() call before calling FFT_HEIGHT2.

#if 1
  local T2 *ldsOut = lds + me;
  for (u32 i = 0; i < NH; ++i) { ldsOut[i * 2*G_H] = u[i]; }

// For NH=4, the lds indices (where to read each outgoing u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
  lds += (me < G_H) ? me % G_H : (NH*2)*G_H-1 - me % G_H;
  i32 ldsInc = (me < G_H) ? G_H : -G_H;
  bar();
  for (u32 i = 0; i < NH; ++i, lds += ldsInc) { u[i] = *lds; }
#else
  local T *ldsOut = (local T *) lds + me;
  for (u32 i = 0; i < NH; ++i) { ldsOut[i * 2*G_H] = u[i].x; ldsOut[NH*2*G_H + i * 2*G_H] = u[i].y; }

// For NH=4, the lds indices (where to read each outgoing u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
  local T *ldsIn = (local T *) lds + ((me < G_H) ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsInc = (me < G_H) ? G_H : -G_H;
  bar();
  for (u32 i = 0; i < NH; ++i, ldsIn += ldsInc) { u[i].x = ldsIn[0]; u[i].y = ldsIn[NH*2*G_H]; }
#endif
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

void OVERLOAD reverse(u32 WG, local F2 *lds, F2 *u, bool bump) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me + bump;

  bar();

#if NH == 8
  lds[revMe + 0 * WG] = u[3];
  lds[revMe + 1 * WG] = u[2];
  lds[revMe + 2 * WG] = u[1];
  lds[bump ? ((revMe + 3 * WG) % (4 * WG)) : (revMe + 3 * WG)] = u[0];
#elif NH == 4
  lds[revMe + 0 * WG] = u[1];
  lds[bump ? ((revMe + WG) % (2 * WG)) : (revMe + WG)] = u[0];
#else
#error
#endif

  bar();
  for (i32 i = 0; i < NH/2; ++i) { u[i] = lds[i * WG + me]; }
}

void OVERLOAD reverseLine(u32 WG, local F2 *lds2, F2 *u) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me;

  local F2 *lds = lds2 + revMe;
  bar();
  for (u32 i = 0; i < NH; ++i) { lds[WG * (NH - 1 - i)] = u[i]; }

  lds = lds2 + me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i] = lds[WG * i]; }
}

// This is used to reverse the second part of a line, and cross the reversed parts between the halves.
void OVERLOAD revCrossLine(u32 WG, local F2* lds2, F2 *u, u32 n, bool writeSecondHalf) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;

  u32 revLowMe = WG - 1 - lowMe;

  for (u32 i = 0; i < n; ++i) { lds2[WG * n * writeSecondHalf + WG * (n - 1 - i) + revLowMe] = u[i]; }

  bar();   // we need a full bar because we're crossing halves

  for (u32 i = 0; i < n; ++i) { u[i] = lds2[WG * n * !writeSecondHalf + WG * i + lowMe]; }
}

//
// These versions are for the kernel(s) that uses a double-wide workgroup (u in half the workgroup, v in the other half)
//

void OVERLOAD reverse2(local F2 *lds, F2 *u) {
  u32 me = get_local_id(0);

  // For NH=8, u[0] to u[3] are left unchanged.  Write to lds:
  //	u[7]rev   u[6]rev
  //	u[5]rev   u[4]rev
  //	v[7]rev   v[6]rev
  //	v[5]rev   v[4]rev
  bar();
  for (u32 i = 0; i < NH / 2; ++i) {
    u32 j = (i * G_H + me % G_H);
    lds[me < G_H ? ((NH/2)*G_H - j) % ((NH/2)*G_H) : NH*G_H-1 - j] = u[NH/2 + i];
  }
  // For NH=8, read from lds into u[i]:
  //	u[4] =   u[7]rev   v[7]rev
  //	u[5] =   u[6]rev   v[6]rev
  //	u[6] =   u[5]rev   v[5]rev
  //	u[7] =   u[4]rev   v[4]rev
  bar();
  lds += me % G_H + (me / G_H) * NH/2 * G_H;
  for (u32 i = 0; i < NH / 2; ++i) { u[NH/2 + i] = lds[i * G_H]; }
}

// Somewhat similar to reverseLine.
// The u values are in threads < G_H, the v values to reverse in threads >= G_H.
// Whereas reverseLine leaves u values alone.  This reverseLine moves u values around
// so that pairSq2 can easily operate on pairs.  This means for NH = 4, web output:
//      u[0]    u[1]            // Returned in u[0]
//      u[2]    u[3]            // Returned in u[1]
//      v[3]rev v[2]rev         // Returned in u[2]
//      v[1]rev v[0]rev         // Returned in u[3]
void OVERLOAD reverseLine2(local F2 *lds, F2 *u) {
  u32 me = get_local_id(0);

// NOTE:  It is important that this routine use lds memory in coordination with shufl2.  Failure to do so would require an
// unqualified bar() call here.  Specifically, the u values are stored in the upper half of lds memory (SMALL_HEIGHT F2 values).
// The v values are stored in the lower half of lds memory (the next SMALL_HEIGHT F2 values).

  if (G_H > WAVEFRONT) bar();

// For NH=4, the lds indices (where to write each incoming u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
// That means saving to lds using index: me < G_H ? me % G_H + i * G_H : 8*G_H-1 - me % G_H - i * G_H

#if 1
  local F2 *ldsOut = lds + (me < G_H ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsOutInc = (me < G_H) ? G_H : -G_H;
  for (u32 i = 0; i < NH; ++i, ldsOut += ldsOutInc) { *ldsOut = u[i]; }

  lds += me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i] = lds[i * 2*G_H]; }
#else
  local F *ldsOut = (local F *) lds + (me < G_H ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsOutInc = (me < G_H) ? G_H : -G_H;
  for (u32 i = 0; i < NH; ++i, ldsOut += ldsOutInc) { ldsOut[0] = u[i].x; ldsOut[NH*2*G_H] = u[i].y; }

  local F *ldsIn = (local F *) lds + me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i].x = ldsIn[i * 2*G_H]; u[i].y = ldsIn[NH*2*G_H + i * 2*G_H]; }
#endif
}

// Undo a reverseLine2
void OVERLOAD unreverseLine2(local F2 *lds, F2 *u) {
  u32 me = get_local_id(0);

// NOTE:  It is important that this routine use lds memory in coordination with reverseLine2 and shufl2.  By initially
// writing to the lds locations that reverseLine2 read from we do not need an initial bar() call here.  Also, by reading
// from the lds locations that shufl2 will use (u values in the upper half of lds memory, v values in the lower half of
// lds memory) we can issue a qualified bar() call before calling FFT_HEIGHT2.

#if 1
  local F2 *ldsOut = lds + me;
  for (u32 i = 0; i < NH; ++i) { ldsOut[i * 2*G_H] = u[i]; }

// For NH=4, the lds indices (where to read each outgoing u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
  lds += (me < G_H) ? me % G_H : (NH*2)*G_H-1 - me % G_H;
  i32 ldsInc = (me < G_H) ? G_H : -G_H;
  bar();
  for (u32 i = 0; i < NH; ++i, lds += ldsInc) { u[i] = *lds; }
#else
  local F *ldsOut = (local F *) lds + me;
  for (u32 i = 0; i < NH; ++i) { ldsOut[i * 2*G_H] = u[i].x; ldsOut[NH*2*G_H + i * 2*G_H] = u[i].y; }

// For NH=4, the lds indices (where to read each outgoing u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
  local F *ldsIn = (local F *) lds + ((me < G_H) ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsInc = (me < G_H) ? G_H : -G_H;
  bar();
  for (u32 i = 0; i < NH; ++i, ldsIn += ldsInc) { u[i].x = ldsIn[0]; u[i].y = ldsIn[NH*2*G_H]; }
#endif
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD reverse(u32 WG, local GF31 *lds, GF31 *u, bool bump) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me + bump;

  bar();

#if NH == 8
  lds[revMe + 0 * WG] = u[3];
  lds[revMe + 1 * WG] = u[2];
  lds[revMe + 2 * WG] = u[1];
  lds[bump ? ((revMe + 3 * WG) % (4 * WG)) : (revMe + 3 * WG)] = u[0];
#elif NH == 4
  lds[revMe + 0 * WG] = u[1];
  lds[bump ? ((revMe + WG) % (2 * WG)) : (revMe + WG)] = u[0];
#else
#error
#endif

  bar();
  for (i32 i = 0; i < NH/2; ++i) { u[i] = lds[i * WG + me]; }
}

void OVERLOAD reverseLine(u32 WG, local GF31 *lds2, GF31 *u) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me;

  local GF31 *lds = lds2 + revMe;
  bar();
  for (u32 i = 0; i < NH; ++i) { lds[WG * (NH - 1 - i)] = u[i]; }

  lds = lds2 + me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i] = lds[WG * i]; }
}

// This is used to reverse the second part of a line, and cross the reversed parts between the halves.
void OVERLOAD revCrossLine(u32 WG, local GF31* lds2, GF31 *u, u32 n, bool writeSecondHalf) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;

  u32 revLowMe = WG - 1 - lowMe;

  for (u32 i = 0; i < n; ++i) { lds2[WG * n * writeSecondHalf + WG * (n - 1 - i) + revLowMe] = u[i]; }

  bar();   // we need a full bar because we're crossing halves

  for (u32 i = 0; i < n; ++i) { u[i] = lds2[WG * n * !writeSecondHalf + WG * i + lowMe]; }
}

//
// These versions are for the kernel(s) that uses a double-wide workgroup (u in half the workgroup, v in the other half)
//

void OVERLOAD reverse2(local GF31 *lds, GF31 *u) {
  u32 me = get_local_id(0);
  
  // For NH=8, u[0] to u[3] are left unchanged.  Write to lds:
  //	u[7]rev   u[6]rev
  //	u[5]rev   u[4]rev
  //	v[7]rev   v[6]rev
  //	v[5]rev   v[4]rev
  bar();
  for (u32 i = 0; i < NH / 2; ++i) {
    u32 j = (i * G_H + me % G_H);
    lds[me < G_H ? ((NH/2)*G_H - j) % ((NH/2)*G_H) : NH*G_H-1 - j] = u[NH/2 + i];
  }
  // For NH=8, read from lds into u[i]:
  //	u[4] =   u[7]rev   v[7]rev
  //	u[5] =   u[6]rev   v[6]rev
  //	u[6] =   u[5]rev   v[5]rev
  //	u[7] =   u[4]rev   v[4]rev
  bar();
  lds += me % G_H + (me / G_H) * NH/2 * G_H;
  for (u32 i = 0; i < NH / 2; ++i) { u[NH/2 + i] = lds[i * G_H]; }
}

// Somewhat similar to reverseLine.
// The u values are in threads < G_H, the v values to reverse in threads >= G_H.
// Whereas reverseLine leaves u values alone.  This reverseLine moves u values around
// so that pairSq2 can easily operate on pairs.  This means for NH = 4, web output:
//      u[0]    u[1]            // Returned in u[0]
//      u[2]    u[3]            // Returned in u[1]
//      v[3]rev v[2]rev         // Returned in u[2]
//      v[1]rev v[0]rev         // Returned in u[3]
void OVERLOAD reverseLine2(local GF31 *lds, GF31 *u) {
  u32 me = get_local_id(0);

// NOTE:  It is important that this routine use lds memory in coordination with shufl2.  Failure to do so would require an
// unqualified bar() call here.  Specifically, the u values are stored in the upper half of lds memory (SMALL_HEIGHT GF31 values).
// The v values are stored in the lower half of lds memory (the next SMALL_HEIGHT GF31 values).

  if (G_H > WAVEFRONT) bar();

// For NH=4, the lds indices (where to write each incoming u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
// That means saving to lds using index: me < G_H ? me % G_H + i * G_H : 8*G_H-1 - me % G_H - i * G_H

#if 1
  local GF31 *ldsOut = lds + (me < G_H ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsOutInc = (me < G_H) ? G_H : -G_H;
  for (u32 i = 0; i < NH; ++i, ldsOut += ldsOutInc) { *ldsOut = u[i]; }

  lds += me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i] = lds[i * 2*G_H]; }
#else
  local Z61 *ldsOut = (local Z61 *) lds + (me < G_H ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsOutInc = (me < G_H) ? G_H : -G_H;
  for (u32 i = 0; i < NH; ++i, ldsOut += ldsOutInc) { ldsOut[0] = u[i].x; ldsOut[NH*2*G_H] = u[i].y; }

  local ZF61 *ldsIn = (local T *) lds + me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i].x = ldsIn[i * 2*G_H]; u[i].y = ldsIn[NH*2*G_H + i * 2*G_H]; }
#endif
}

// Undo a reverseLine2
void OVERLOAD unreverseLine2(local GF31 *lds, GF31 *u) {
  u32 me = get_local_id(0);

// NOTE:  It is important that this routine use lds memory in coordination with reverseLine2 and shufl2.  By initially
// writing to the lds locations that reverseLine2 read from we do not need an initial bar() call here.  Also, by reading
// from the lds locations that shufl2 will use (u values in the upper half of lds memory, v values in the lower half of
// lds memory) we can issue a qualified bar() call before calling FFT_HEIGHT2.

#if 1
  local GF31 *ldsOut = lds + me;
  for (u32 i = 0; i < NH; ++i) { ldsOut[i * 2*G_H] = u[i]; }

// For NH=4, the lds indices (where to read each outgoing u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
  lds += (me < G_H) ? me % G_H : (NH*2)*G_H-1 - me % G_H;
  i32 ldsInc = (me < G_H) ? G_H : -G_H;
  bar();
  for (u32 i = 0; i < NH; ++i, lds += ldsInc) { u[i] = *lds; }
#else
  local Z61 *ldsOut = (local T *) lds + me;
  for (u32 i = 0; i < NH; ++i) { ldsOut[i * 2*G_H] = u[i].x; ldsOut[NH*2*G_H + i * 2*G_H] = u[i].y; }

// For NH=4, the lds indices (where to read each outgoing u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
  local Z61 *ldsIn = (local T *) lds + ((me < G_H) ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsInc = (me < G_H) ? G_H : -G_H;
  bar();
  for (u32 i = 0; i < NH; ++i, ldsIn += ldsInc) { u[i].x = ldsIn[0]; u[i].y = ldsIn[NH*2*G_H]; }
#endif
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD reverse(u32 WG, local GF61 *lds, GF61 *u, bool bump) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me + bump;

  bar();

#if NH == 8
  lds[revMe + 0 * WG] = u[3];
  lds[revMe + 1 * WG] = u[2];
  lds[revMe + 2 * WG] = u[1];
  lds[bump ? ((revMe + 3 * WG) % (4 * WG)) : (revMe + 3 * WG)] = u[0];
#elif NH == 4
  lds[revMe + 0 * WG] = u[1];
  lds[bump ? ((revMe + WG) % (2 * WG)) : (revMe + WG)] = u[0];
#else
#error
#endif

  bar();
  for (i32 i = 0; i < NH/2; ++i) { u[i] = lds[i * WG + me]; }
}

void OVERLOAD reverseLine(u32 WG, local GF61 *lds2, GF61 *u) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me;

  local GF61 *lds = lds2 + revMe;
  bar();
  for (u32 i = 0; i < NH; ++i) { lds[WG * (NH - 1 - i)] = u[i]; }

  lds = lds2 + me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i] = lds[WG * i]; }
}

// This is used to reverse the second part of a line, and cross the reversed parts between the halves.
void OVERLOAD revCrossLine(u32 WG, local GF61* lds2, GF61 *u, u32 n, bool writeSecondHalf) {
  u32 me = get_local_id(0);
  u32 lowMe = me % WG;

  u32 revLowMe = WG - 1 - lowMe;

  for (u32 i = 0; i < n; ++i) { lds2[WG * n * writeSecondHalf + WG * (n - 1 - i) + revLowMe] = u[i]; }

  bar();   // we need a full bar because we're crossing halves

  for (u32 i = 0; i < n; ++i) { u[i] = lds2[WG * n * !writeSecondHalf + WG * i + lowMe]; }
}

//
// These versions are for the kernel(s) that uses a double-wide workgroup (u in half the workgroup, v in the other half)
//

void OVERLOAD reverse2(local GF61 *lds, GF61 *u) {
  u32 me = get_local_id(0);
  
  // For NH=8, u[0] to u[3] are left unchanged.  Write to lds:
  //	u[7]rev   u[6]rev
  //	u[5]rev   u[4]rev
  //	v[7]rev   v[6]rev
  //	v[5]rev   v[4]rev
  bar();
  for (u32 i = 0; i < NH / 2; ++i) {
    u32 j = (i * G_H + me % G_H);
    lds[me < G_H ? ((NH/2)*G_H - j) % ((NH/2)*G_H) : NH*G_H-1 - j] = u[NH/2 + i];
  }
  // For NH=8, read from lds into u[i]:
  //	u[4] =   u[7]rev   v[7]rev
  //	u[5] =   u[6]rev   v[6]rev
  //	u[6] =   u[5]rev   v[5]rev
  //	u[7] =   u[4]rev   v[4]rev
  bar();
  lds += me % G_H + (me / G_H) * NH/2 * G_H;
  for (u32 i = 0; i < NH / 2; ++i) { u[NH/2 + i] = lds[i * G_H]; }
}

// Somewhat similar to reverseLine.
// The u values are in threads < G_H, the v values to reverse in threads >= G_H.
// Whereas reverseLine leaves u values alone.  This reverseLine moves u values around
// so that pairSq2 can easily operate on pairs.  This means for NH = 4, web output:
//      u[0]    u[1]            // Returned in u[0]
//      u[2]    u[3]            // Returned in u[1]
//      v[3]rev v[2]rev         // Returned in u[2]
//      v[1]rev v[0]rev         // Returned in u[3]
void OVERLOAD reverseLine2(local GF61 *lds, GF61 *u) {
  u32 me = get_local_id(0);

// NOTE:  It is important that this routine use lds memory in coordination with shufl2.  Failure to do so would require an
// unqualified bar() call here.  Specifically, the u values are stored in the upper half of lds memory (SMALL_HEIGHT GF61 values).
// The v values are stored in the lower half of lds memory (the next SMALL_HEIGHT GF61 values).

  if (G_H > WAVEFRONT) bar();

// For NH=4, the lds indices (where to write each incoming u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
// That means saving to lds using index: me < G_H ? me % G_H + i * G_H : 8*G_H-1 - me % G_H - i * G_H

#if 1
  local GF61 *ldsOut = lds + (me < G_H ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsOutInc = (me < G_H) ? G_H : -G_H;
  for (u32 i = 0; i < NH; ++i, ldsOut += ldsOutInc) { *ldsOut = u[i]; }

  lds += me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i] = lds[i * 2*G_H]; }
#else
  local Z61 *ldsOut = (local Z61 *) lds + (me < G_H ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsOutInc = (me < G_H) ? G_H : -G_H;
  for (u32 i = 0; i < NH; ++i, ldsOut += ldsOutInc) { ldsOut[0] = u[i].x; ldsOut[NH*2*G_H] = u[i].y; }

  local ZF61 *ldsIn = (local T *) lds + me;
  bar();
  for (u32 i = 0; i < NH; ++i) { u[i].x = ldsIn[i * 2*G_H]; u[i].y = ldsIn[NH*2*G_H + i * 2*G_H]; }
#endif
}

// Undo a reverseLine2
void OVERLOAD unreverseLine2(local GF61 *lds, GF61 *u) {
  u32 me = get_local_id(0);

// NOTE:  It is important that this routine use lds memory in coordination with reverseLine2 and shufl2.  By initially
// writing to the lds locations that reverseLine2 read from we do not need an initial bar() call here.  Also, by reading
// from the lds locations that shufl2 will use (u values in the upper half of lds memory, v values in the lower half of
// lds memory) we can issue a qualified bar() call before calling FFT_HEIGHT2.

#if 1
  local GF61 *ldsOut = lds + me;
  for (u32 i = 0; i < NH; ++i) { ldsOut[i * 2*G_H] = u[i]; }

// For NH=4, the lds indices (where to read each outgoing u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
  lds += (me < G_H) ? me % G_H : (NH*2)*G_H-1 - me % G_H;
  i32 ldsInc = (me < G_H) ? G_H : -G_H;
  bar();
  for (u32 i = 0; i < NH; ++i, lds += ldsInc) { u[i] = *lds; }
#else
  local Z61 *ldsOut = (local T *) lds + me;
  for (u32 i = 0; i < NH; ++i) { ldsOut[i * 2*G_H] = u[i].x; ldsOut[NH*2*G_H + i * 2*G_H] = u[i].y; }

// For NH=4, the lds indices (where to read each outgoing u[i] which has v[i] in the upper threads) looks like this:
// 0..GH-1 +0*G_H    GH-1..0 +7*G_H
// 0..GH-1 +1*G_H    GH-1..0 +6*G_H
// 0..GH-1 +2*G_H    GH-1..0 +5*G_H
// 0..GH-1 +3*G_H    GH-1..0 +4*G_H
  local Z61 *ldsIn = (local T *) lds + ((me < G_H) ? me % G_H : (NH*2)*G_H-1 - me % G_H);
  i32 ldsInc = (me < G_H) ? G_H : -G_H;
  bar();
  for (u32 i = 0; i < NH; ++i, ldsIn += ldsInc) { u[i].x = ldsIn[0]; u[i].y = ldsIn[NH*2*G_H]; }
#endif
}

#endif
