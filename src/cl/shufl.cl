// Copyright (C) Mihai Preda and George Woltman

// The LDSSWIZ swizzle masks below are sized for a workgroup of at least 64: at WG 32 the XOR patterns
// (lowMe & 7), (lowMe & 15), ((lowMe / 8) & 15) and friends fold several rows onto each other, and five
// of the cases then return the wrong data -- 64 to 192 elements of 256, depending on the case.  No
// dispatch reaches them there today (the WG == 32 branch of fft_common asks for f=1,r=4 at RADIX 8 and
// f=4,r=8, and no swizzle case matches either), so like the padded cases above this is a constraint to
// record rather than code to rewrite.
// The RADIX == 8 LDSSWIZ cases address LDS as byte offsets from lds2 XORed with small constants.  That relies on each
// workgroup's LDS region starting at a multiple of WG * RADIX * SHUFL_BYTES, i.e. on there being no LDSPAD padding.  Every
// such LDSSWIZ case therefore has a matching LDSPAD case earlier in the same function, which returns first when LDSPAD is on.
#if LDSSWIZ && WG < 64
#error LDSSWIZ needs a workgroup of at least 64: its swizzle masks fold rows together below that
#endif

// SWIZ_RECOMPUTE: with SHUFL_BYTES == 4 and LDSSWIZ, the RADIX == 8, f == 8 swizzle needs four different XORed LDS
// addresses on both the write and the read side.  Left alone, the compiler computes all eight once and keeps them live
// across the four int-sized passes, which on gfx906 costs 9 VGPRs (carryFused 81 -> 90, dropping from 3 to 2 waves/SIMD).
// SWIZ_RECOMPUTE makes the compiler recompute them in each pass instead, trading those VGPRs for extra integer
// instructions.  Default is on for AMD only.  On nVidia (TitanV) the addresses fit within carryFused's register cap and
// recomputing them is slightly slower.
#if !defined(SWIZ_RECOMPUTE)
#define SWIZ_RECOMPUTE AMDGPU
#endif

// The LDSPAD RADIX == 4 reads in this file choose between a WG == 64 form and an else arm whose
// i * 64 and (lowMe / 64) * 16 terms only balance at WG == 256.  Both workgroup sizes RADIX 4 can
// have today are therefore handled -- a 256-wide/high shape gives WG 64, and a 1024 one would give
// WG 256 if the commented-out clause in FFTConfig::nW()/nH() were re-enabled -- so the code is
// correct as it stands, and generalising it would add index arithmetic for no present benefit.
// Any other WG would read slots that were never written, silently and with the wrong residue as the
// only symptom, so refuse to build it instead.  Generalising is easy when it is needed: i * (WG / 4)
// in place of i * 64 is index-identical at both 64 and 256.
#if LDSPAD && RADIX == 4 && WG != 64 && WG != 256
#error RADIX == 4 with this workgroup size needs the LDSPAD reads in shufl.cl generalised first (they assume WG is 64 or 256)
#endif

// Strongly typed versions of LDSptr and LDSsharing_ptr.  On TitanV, CUDA 12.9, this is 1% faster.
local T_F_Z31_Z61 * OVERLOAD LDSptr(local T_F_Z31_Z61 *lds, const u32 numWG) {
  return lds + ((u32)get_local_id(0) / WG) * LDS_SHUFL_BYTES(numWG) / sizeof(T_F_Z31_Z61);
}
local T2_F2_GF31_GF61 * OVERLOAD LDSptr(local T2_F2_GF31_GF61 *lds, const u32 numWG) {
  return lds + ((u32)get_local_id(0) / WG) * LDS_SHUFL_BYTES(numWG) / sizeof(T2_F2_GF31_GF61);
}
local T_F_Z31_Z61 * OVERLOAD LDSsharing_ptr(local T_F_Z31_Z61 *lds, const u32 numWG) {
  if (!SHARING_LDS(numWG)) return LDSptr(lds, numWG);
  return lds + ((u32)get_local_id(0) / WG / SBMUL(numWG)) * SBMUL(numWG) * LDS_SHUFL_BYTES(numWG) / sizeof(T_F_Z31_Z61);
}
local T2_F2_GF31_GF61 * OVERLOAD LDSsharing_ptr(local T2_F2_GF31_GF61 *lds, const u32 numWG) {
  if (!SHARING_LDS(numWG)) return LDSptr(lds, numWG);
  return lds + ((u32)get_local_id(0) / WG / SBMUL(numWG)) * SBMUL(numWG) * LDS_SHUFL_BYTES(numWG) / sizeof(T2_F2_GF31_GF61);
}


#ifdef T2_GF61

// Shufl two or more fft_WIDTHs or FFT_HEIGHTs operating on 64-bit values using LDS_BYTES of LDS memory.
// Care is taken that each simultaneous workgroup does not interfere with the LDS memory of other simultaneous workgroups --
// even when operating on differernt sized data elements as can happen in an M31+M61 NTT.
// WG = workgroup size of a single fft_WIDTH or fft_HEIGHT
// n = sizeof array u (nW or nH).  n * WG = WIDTH or HEIGHT
// r usually equals RADIX if a full fft_RADIX step was just performed.  On occasion u[8] values may do less than an fft8 step.
// numWG = number of fft_WIDTHs or fft_HEIGHTs being processed simultaneously
// lowMe = me % WG
int4 int4_of(int a, int b, int c, int d) { int4 v; v.x = a; v.y = b; v.z = c; v.w = d; return v; }

void OVERLOAD shufl(local T2_GF61 *lds2, T2_GF61 *u, u32 f, u32 r, u32 numWG, u32 lowMe) {

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  // If SHUFL_BYTES is 16 we can write the complete T2 value to LDS memory with one instruction.
  // We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
  if (SBMUL(numWG) * SHUFL_BYTES >= 16) {
    local T2_GF61* lds = LDSsharing_ptr(lds2, numWG);

#if LDSPAD
    // Special case first RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...448, 8, 72..., 16...   lds[64..127] = +1
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 448, 1, 65...   output[64..127] = +8
    // Pad 1 value every row to eliminate bank conflicts.
    if (0 && f == 1 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe & 7) * (WG + 1) + (lowMe / 8) * 8 + i] = u[i]; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * (WG / 64) * 8                    + ((lowMe / 8) & 7) * (WG + 1) + (lowMe & 7)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 1) + (lowMe & 7)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case first RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are in order and written straight to LDS memory.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that uses a little padding.  Pad one value after every row to eliminate bank conflicts.
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 448, 1, 65...   output[64..127] = +8
    if (f == 1 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 1) + lowMe] = u[i]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * (WG / 8) + (lowMe / 8) + (lowMe & 7) * (WG + 1)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...192, 1, 65..., 16...   lds[64..127] = +2
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 1, 65...   output[64..127] = +16
    // Pad 1 value every row to eliminate bank conflicts.
    if (f == 1 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 2) & 3) * (WG + 1) + (lowMe / 8) * 8 + (lowMe & 1) * 4 + i] = u[i]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * WG / 4 + (lowMe / 32) * 8 + ((lowMe / 8) & 3) * (WG + 1) + (lowMe & 7)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16... 4..  lds[64..127] = +1
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 16... 1..   output[64..127] = +4
    // Pad 4 values after every row to eliminate bank conflicts.
    if (f == 4 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 4) + (lowMe / 16) * 16 + i * 4 + (lowMe & 3)] = u[i]; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

#if LDSSWIZ
    // Special case first RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 1, 65...   lds[64..127] = +8
    // Swizzle LDS blocks to eliminate bank conflicts.  Swizzle on the first 8 threads written to LDS (multiples of 1) and the first 8 threads read from LDS (multiples of 64).
    if (f == 1 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 8 + i) ^ (lowMe & 7)] = u[i]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 8) & 7)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // No swizzle of LDS blocks is needed to eliminate bank conflicts.  The first 8 threads written to LDS (multiples of 64) and
    // the first 8 threads read from LDS (multiples of 64) are already in separate LDS banks.
    // The read index must be the inverse of the natural write for every WG, not just WG == 64; at WG == 64 it
    // reduces to lowMe / 8 * 64 + i * 8 + (lowMe & 7).
    // We can however save a bar() by writing to same locations that previous shufl wrote to.
    if (f == 8 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);             //GRRR.... LDStx_start will do the bar we are trying to save
      for (u32 i = 0; i < RADIX; ++i) { lds[i * WG + lowMe] = u[i]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[((lowMe / 8) & 7) * WG + i * (WG / 8) + (lowMe / 64) * 8 + (lowMe & 7)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 1, 65...   lds[64..127] = +16
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 8 threads written to LDS (4 multiples of 1 and 2 multiples of 4) and the first 8 threads read from LDS (4 multiples of 64 and 2 multiples of 1).
    if (f == 1 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 7)] = u[i]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 7)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16, 80...   lds[64..127] = +4
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 8 threads written to LDS (4 multiples of 64 and 2 multiples of 1) and the first 8 threads read from LDS (4 multiples of 64 and 2 multiples of 4).
    if (f == 4 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 4)] = u[i]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 4)]; }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

    // Otherwise, execute the original shufl code modified to handle case where a full RADIX fft was not done
    LDStx_start(lds2, numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i / (RADIX / r) * f + i % (RADIX / r) * WG * r + (lowMe & ~mask) * r + (lowMe & mask)] = u[i]; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * WG + lowMe]; }
    LDStx_end(lds2, numWG);
    return;
  }

  // If SHUFL_BYTES is 8 we split the T2 values into two T values.  These are written to LDS memory with two instructions.
  // We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
  else if (SBMUL(numWG) * SHUFL_BYTES == 8) {
    local T_Z61* lds = LDSsharing_ptr((local T_Z61 *)lds2, numWG);

#if LDSPAD
    // Special case first RADIX == 8 code to eliminate LDS bank conflicts.
    // Input values are in order and written straight to LDS memory.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that uses a little padding.  Pad two values after every row to eliminate bank conflicts.
    // Read from LDS in the desired output order.  In the example:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    if (f == 1 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 2) + lowMe] = u[i].x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * WG / 8 + (lowMe / 8) + (lowMe & 7) * (WG + 2)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 2) + lowMe] = u[i].y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * WG / 8 + (lowMe / 8) + (lowMe & 7) * (WG + 2)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS with 8 pads after each row.
    // Read from LDS in output order.  In the example:  u[0] = 0, 64, ... 448, 8, 72...   u[1] = +1
    if (f == 8 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = u[i].x; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * (WG / 64) * 8                    +  (lowMe / 8)      * (WG + 8) + (lowMe & 7)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = u[i].y; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * (WG / 64) * 8                    +  (lowMe / 8)      * (WG + 8) + (lowMe & 7)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case alternate first RADIX == 8 code to eliminate LDS bank conflicts (only a radix-4 step was performed).
    // Input values are in order and written straight to LDS memory.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +32...
    // Output to LDS that uses a little padding.  Pad four values after every other row to eliminate bank conflicts.
    // Read from LDS in the desired output order.  In the example:  u[0] = 0, 64, 128, 192, 1, 65...   u[1] = +8
    if (f == 1 && r == 4 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i / 2 * (2 * WG + 4) + (i % 2) * WG + lowMe] = u[i].x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * WG / 4 + (lowMe / 4) + (lowMe & 3) * (2 * WG + 4)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i / 2 * (2 * WG + 4) + (i % 2) * WG + lowMe] = u[i].y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * WG / 4 + (lowMe / 4) + (lowMe & 3) * (2 * WG + 4)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case alternate second RADIX == 8 to eliminate LDS bank conflicts (first shufl was partial after a radix-4 step).
    // Input values are the output from a previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, 128, 192, 1, 65...   u[1] = +8
    // Output to LDS with 4 pads after each row.
    // Read from LDS in output order.  In the example:  u[0] = 0, 64, 128, 192, 8, 72...   u[1] = +1
    if (f == 4 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = u[i].x; }
      LDSbar(numWG);
      if (WG == 32) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * (WG / 32) * 4                    +  (lowMe / 4)      * (WG + 4) + (lowMe & 3)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * (WG / 32) * 4 + (lowMe / 32) * 4 + ((lowMe / 4) & 7) * (WG + 4) + (lowMe & 3)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = u[i].y; }
      LDSbar(numWG);
      if (WG == 32) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * (WG / 32) * 4                    +  (lowMe / 4)      * (WG + 4) + (lowMe & 3)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * (WG / 32) * 4 + (lowMe / 32) * 4 + ((lowMe / 4) & 7) * (WG + 4) + (lowMe & 3)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...192, 1.., 2.., 3.., 16...   lds[64..127] = +4
    // Read from LDS in the desired output order.  In the example:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Pad one value after every row to eliminate bank conflicts.
    if (1 && f == 1 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = u[i].x; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 16                     + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = u[i].y; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 16                     + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are in order and written straight to LDS memory.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS using a little padding.  Pad four values after every row to eliminate bank conflicts.
    // Read from LDS in the desired output order.  In the example:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    if (0 && f == 1 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = u[i].x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * WG / 4 + (lowMe / 4) + (lowMe & 3) * (WG + 4)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = u[i].y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * WG / 4 + (lowMe / 4) + (lowMe & 3) * (WG + 4)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0...192, 16..., 32..., 48..., 4...   lds[64..127] = +1
    // Output to LDS in the order we expect to read.  In the example:  u[0] = 0...192, 16... 32.. 48.. 1...  u[1] = +4
    // Pad 4 values after every row to eliminate bank conflicts.
    if (0 && f == 4 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 4) + (lowMe / 16) * 16 + i * 4 + (lowMe & 3)] = u[i].x; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 4) + (lowMe / 16) * 16 + i * 4 + (lowMe & 3)] = u[i].y; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS with 4 pads after each row.
    // Read from LDS in output order.  In the example:  u[0] = 0...192, 16... 32.. 48.. 1...  u[1] = +4
    if (1 && f == 4 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = u[i].x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * (WG / 16) * 4 + (lowMe / 16) * 4 + ((lowMe / 4) & 3) * (WG + 4) + (lowMe & 3)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = u[i].y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * (WG / 16) * 4 + (lowMe / 16) * 4 + ((lowMe / 4) & 3) * (WG + 4) + (lowMe & 3)]; }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

#if LDSSWIZ
    // The swizzles below are expressed as two byte offsets from lds2 (wa for the write, ra for the read) that are XORed with a
    // constant for each i.  This workgroup's LDS region starts at a multiple of WG * RADIX * 8 bytes and the XORs only touch
    // bits 3 to 6, so they stay within the region.  Each additional address then costs at most one XOR, and accesses a constant
    // distance apart can be merged by the compiler (ds_write2_b64/ds_read2st64_b64).  See SWIZ_RECOMPUTE.

    // Special case first RADIX == 8 code to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 1, 65...   lds[64..127] = +8
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 1 and 2 multiples of 8) and the first 16 threads read from LDS (8 multiples of 64 and 2 multiples of 1).
    // The writes cannot be paired without padding (a conflict-free write must XOR all three bits of i), the reads are.
    // Works for any WG >= 64 (the read XOR ((i * WG / 8) & 15) is (i & 1) * 8 at WG == 64 and 0 for larger WGs).
    if (f == 1 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      local char* root = (local char*)lds2;
      u32 region = (u32)((local char*)lds - root);
      u32 wa = region + ((lowMe * 8) ^ (lowMe & 15)) * 8;
      u32 ra = region + (lowMe ^ ((lowMe / 8) & 15)) * 8;
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX; ++i) { *(local T_Z61*)(root + (wa ^ (i * 8))) = u[i].x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = *(local T_Z61*)(root + i * WG * 8 + (ra ^ (((i * WG / 8) & 15) * 8))); }
      LDSbar(numWG);
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX; ++i) { *(local T_Z61*)(root + (wa ^ (i * 8))) = u[i].y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = *(local T_Z61*)(root + i * WG * 8 + (ra ^ (((i * WG / 8) & 15) * 8))); }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 64 and 2 multiples of 1) and the first 16 threads read from LDS (8 multiples of 64 and 2 multiples of 8).
    // Only bit 0 of i is XORed, so u[i] and u[i + 2] are a constant distance apart for both the writes and the reads.
    // Works for any WG >= 64 (the read XOR ((i * WG / 8) & 8) is (i & 1) * 8 at WG == 64 and 0 for larger WGs).
    if (f == 8 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      local char* root = (local char*)lds2;
      u32 region = (u32)((local char*)lds - root);
      u32 wa = region + (((lowMe / 8) * 64 + (lowMe & 7)) ^ (lowMe & 8)) * 8;
      u32 ra = region + (lowMe ^ ((lowMe / 8) & 8)) * 8;
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX; i += 2) { *(local T_Z61*)(root + wa + i * 64) = u[i].x; }             // even i first so that same-base stores are adjacent
      for (u32 i = 1; i < RADIX; i += 2) { *(local T_Z61*)(root + (wa ^ 64) + (i - 1) * 64) = u[i].x; }  // and can be merged into ds_write2
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = *(local T_Z61*)(root + i * WG * 8 + (ra ^ (((i * WG / 8) & 8) * 8))); }
      LDSbar(numWG);
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX; i += 2) { *(local T_Z61*)(root + wa + i * 64) = u[i].y; }             // even i first so that same-base stores are adjacent
      for (u32 i = 1; i < RADIX; i += 2) { *(local T_Z61*)(root + (wa ^ 64) + (i - 1) * 64) = u[i].y; }  // and can be merged into ds_write2
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = *(local T_Z61*)(root + i * WG * 8 + (ra ^ (((i * WG / 8) & 8) * 8))); }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 1, 65...   lds[64..127] = +16
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 1 and 4 multiples of 4) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 1).
    if (f == 1 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = u[i].x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = u[i].y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16, 80...   lds[64..127] = +4
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 64 and 4 multiples of 1) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 16).
    if (f == 4 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = u[i].x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = u[i].y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

    // Otherwise, execute the original shufl code modified to handle case where a full RADIX fft was not done
    LDStx_start(lds2, numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i / (RADIX / r) * f + i % (RADIX / r) * WG * r + (lowMe & ~mask) * r + (lowMe & mask)] = u[i].x; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * WG + lowMe]; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i / (RADIX / r) * f + i % (RADIX / r) * WG * r + (lowMe & ~mask) * r + (lowMe & mask)] = u[i].y; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * WG + lowMe]; }
    LDStx_end(lds2, numWG);
    return;
  }

  // If SHUFL_BYTES is 4 we split the T2 values into 4 int values.  These are written to LDS memory using four instructions.
  // We're writing 4 bytes at a time, which means all 32 banks must be in play for a conflict-free pass.
  // NOTE: every LDStx_start below uses the _with_fence variant, not just the generic fallback's.  Radeon VII was
  // found to need the fence for the plain fallback case (see the comment on the generic path below), and the
  // mechanism is not understood well enough to trust that the LDSPAD/LDSSWIZ cases -- which share the same
  // 4-byte-at-a-time LDS traffic -- are exempt.
  else if (SBMUL(numWG) * SHUFL_BYTES == 4) {
    local int* lds = (local int*)LDSsharing_ptr(lds2, numWG);

#if LDSPAD
    // Input values are in order and written straight to LDS memory.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that uses a little padding.  Pad four values after every row to eliminate bank conflicts.
    // Read from LDS in the desired output order.  In the example:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    if (f == 1 && r == 8 && RADIX == 8) {
      LDStx_start_with_fence(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * WG / 8 + (lowMe / 8) + (lowMe & 7) * (WG + 4)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * WG / 8 + (lowMe / 8) + (lowMe & 7) * (WG + 4)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).z; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * WG / 8 + (lowMe / 8) + (lowMe & 7) * (WG + 4)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).w; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * WG / 8 + (lowMe / 8) + (lowMe & 7) * (WG + 4)]; u[i] = as_T2_GF61(tmp); }
      LDStx_end(lds2, numWG);
      return;
    }

    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS with 8 pads after each row.
    // Read from LDS in output order.  In the example:  u[0] = 0, 64, ... 448, 8, 72...   u[1] = +1
    if (f == 8 && r == 8 && RADIX == 8) {
      LDStx_start_with_fence(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = as_int4(u[i]).x; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * (WG / 64) * 8                    +  (lowMe / 8)      * (WG + 8) + (lowMe & 7)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = as_int4(u[i]).y; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * (WG / 64) * 8                    +  (lowMe / 8)      * (WG + 8) + (lowMe & 7)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = as_int4(u[i]).z; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * (WG / 64) * 8                    +  (lowMe / 8)      * (WG + 8) + (lowMe & 7)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = as_int4(u[i]).w; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * (WG / 64) * 8                    +  (lowMe / 8)      * (WG + 8) + (lowMe & 7)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]; u[i] = as_T2_GF61(tmp); }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case alternate first RADIX == 8 code to eliminate LDS bank conflicts (only a radix-4 step was performed).
    // Input values are in order and written straight to LDS memory.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +32...
    // Output to LDS that uses a little padding.  Pad eight values after every other row to eliminate bank conflicts.
    // Read from LDS in the desired output order.  In the example:  u[0] = 0, 64, 128, 192, 1, 65...   u[1] = +8
    if (f == 1 && r == 4 && RADIX == 8) {
      LDStx_start_with_fence(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i / 2 * (2 * WG + 8) + (i % 2) * WG + lowMe] = as_int4(u[i]).x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * WG / 4 + (lowMe / 4) + (lowMe & 3) * (2 * WG + 8)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i / 2 * (2 * WG + 8) + (i % 2) * WG + lowMe] = as_int4(u[i]).y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * WG / 4 + (lowMe / 4) + (lowMe & 3) * (2 * WG + 8)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i / 2 * (2 * WG + 8) + (i % 2) * WG + lowMe] = as_int4(u[i]).z; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * WG / 4 + (lowMe / 4) + (lowMe & 3) * (2 * WG + 8)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i / 2 * (2 * WG + 8) + (i % 2) * WG + lowMe] = as_int4(u[i]).w; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * WG / 4 + (lowMe / 4) + (lowMe & 3) * (2 * WG + 8)]; u[i] = as_T2_GF61(tmp); }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case alternate second RADIX == 8 to eliminate LDS bank conflicts (first shufl was partial after a radix-4 step).
    // Input values are the output from a previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, 128, 192, 1, 65...   u[1] = +8
    // Output to LDS with 4 pads after each row.
    // Read from LDS in output order.  In the example:  u[0] = 0, 64, 128, 192, 8, 72...   u[1] = +1
    if (f == 4 && r == 8 && RADIX == 8) {
      LDStx_start_with_fence(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).x; }
      LDSbar(numWG);
      if (WG == 32) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * (WG / 32) * 4                    +  (lowMe / 4)      * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * (WG / 32) * 4 + (lowMe / 32) * 4 + ((lowMe / 4) & 7) * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).y; }
      LDSbar(numWG);
      if (WG == 32) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * (WG / 32) * 4                    +  (lowMe / 4)      * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * (WG / 32) * 4 + (lowMe / 32) * 4 + ((lowMe / 4) & 7) * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).z; }
      LDSbar(numWG);
      if (WG == 32) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * (WG / 32) * 4                    +  (lowMe / 4)      * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * (WG / 32) * 4 + (lowMe / 32) * 4 + ((lowMe / 4) & 7) * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).w; }
      LDSbar(numWG);
      if (WG == 32) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * (WG / 32) * 4                    +  (lowMe / 4)      * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * (WG / 32) * 4 + (lowMe / 32) * 4 + ((lowMe / 4) & 7) * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      LDStx_end(lds2, numWG);
      return;
    }

    // Same permutation as the 8-byte path's active "Special case first RADIX == 4" above, done in 4 int-sized passes.
    if (f == 1 && r == 4 && RADIX == 4) {
      LDStx_start_with_fence(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = as_int4(u[i]).x; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * 16                     + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = as_int4(u[i]).y; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * 16                     + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = as_int4(u[i]).z; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * 16                     + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = as_int4(u[i]).w; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * 16                     + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; u[i] = as_T2_GF61(tmp); }
      else          for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; u[i] = as_T2_GF61(tmp); }
      LDStx_end(lds2, numWG);
      return;
    }

    // Same permutation as the 8-byte path's active "Special case second RADIX == 4" above, done in 4 int-sized passes.
    if (f == 4 && r == 4 && RADIX == 4) {
      LDStx_start_with_fence(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * (WG / 16) * 4 + (lowMe / 16) * 4 + ((lowMe / 4) & 3) * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * (WG / 16) * 4 + (lowMe / 16) * 4 + ((lowMe / 4) & 3) * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).z; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * (WG / 16) * 4 + (lowMe / 16) * 4 + ((lowMe / 4) & 3) * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 4) + lowMe] = as_int4(u[i]).w; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * (WG / 16) * 4 + (lowMe / 16) * 4 + ((lowMe / 4) & 3) * (WG + 4) + (lowMe & 3)]; u[i] = as_T2_GF61(tmp); }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

#if LDSSWIZ
    // XORing the low bits of i into the address stops the compiler from merging accesses (every access becomes a separate ds_write_b32/ds_read_b32
    // with its own address VGPR).  This swizzle algorithm only XOR address bits the compiler can still merge across.
    // First RADIX == 8: each thread writes u[0..3] and u[4..7] as two contiguous 16-byte chunks (ds_write_b128), swapping
    // the two chunks when lowMe & 4.  The 8 lanes of a b128 access then cover all 32 banks.  The read XOR does not depend on i,
    // so all reads share one base address (ds_read2st64_b32).
    if (f == 1 && r == 8 && RADIX == 8) {
      LDStx_start_with_fence(lds2, numWG);
      ((local int4*)lds)[lowMe * 2 +  ((lowMe / 4) & 1)     ] = int4_of(as_int4(u[0]).x, as_int4(u[1]).x, as_int4(u[2]).x, as_int4(u[3]).x);
      ((local int4*)lds)[lowMe * 2 + (((lowMe / 4) & 1) ^ 1)] = int4_of(as_int4(u[4]).x, as_int4(u[5]).x, as_int4(u[6]).x, as_int4(u[7]).x);
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * WG + (lowMe ^ ((lowMe / 8) & 4))]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      ((local int4*)lds)[lowMe * 2 +  ((lowMe / 4) & 1)     ] = int4_of(as_int4(u[0]).y, as_int4(u[1]).y, as_int4(u[2]).y, as_int4(u[3]).y);
      ((local int4*)lds)[lowMe * 2 + (((lowMe / 4) & 1) ^ 1)] = int4_of(as_int4(u[4]).y, as_int4(u[5]).y, as_int4(u[6]).y, as_int4(u[7]).y);
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * WG + (lowMe ^ ((lowMe / 8) & 4))]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      ((local int4*)lds)[lowMe * 2 +  ((lowMe / 4) & 1)     ] = int4_of(as_int4(u[0]).z, as_int4(u[1]).z, as_int4(u[2]).z, as_int4(u[3]).z);
      ((local int4*)lds)[lowMe * 2 + (((lowMe / 4) & 1) ^ 1)] = int4_of(as_int4(u[4]).z, as_int4(u[5]).z, as_int4(u[6]).z, as_int4(u[7]).z);
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * WG + (lowMe ^ ((lowMe / 8) & 4))]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      ((local int4*)lds)[lowMe * 2 +  ((lowMe / 4) & 1)     ] = int4_of(as_int4(u[0]).w, as_int4(u[1]).w, as_int4(u[2]).w, as_int4(u[3]).w);
      ((local int4*)lds)[lowMe * 2 + (((lowMe / 4) & 1) ^ 1)] = int4_of(as_int4(u[4]).w, as_int4(u[5]).w, as_int4(u[6]).w, as_int4(u[7]).w);
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * WG + (lowMe ^ ((lowMe / 8) & 4))]; u[i] = as_T2_GF61(tmp); }
      LDStx_end(lds2, numWG);
      return;
    }

    // Second RADIX == 8: XOR only bits 0-1 of i with bits 3-4 of lowMe, so u[i] and u[i + 4] stay a constant 32 ints apart
    // (ds_write2_b32).  The read XOR depends only on i (and lowMe / 64 for WG > 64), so pairs are again a constant distance apart.
    // Each side needs four XORed addresses.  Only the two i == 0 addresses are kept across the passes, see SWIZ_RECOMPUTE.
    // They are byte offsets from lds2 so that each of the other addresses costs a single XOR (this workgroup's LDS region
    // starts at a multiple of WG * RADIX * 4 bytes, so XORing bits 5 and 6 of the offset stays within the region).
    if (f == 8 && r == 8 && RADIX == 8) {
      LDStx_start_with_fence(lds2, numWG);
      local char* root = (local char*)lds2;
      u32 wa = (u32)((local char*)lds - root) + ((lowMe / 8) * 64 + ((lowMe / 8) & 3) * 8 + (lowMe & 7)) * 4;
      u32 ra = (u32)((local char*)lds - root) + (lowMe ^ (((lowMe / 64) & 3) * 8)) * 4;
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX / 2; ++i) { *(local int*)(root + (wa ^ (i * 32))) = as_int4(u[i]).x; *(local int*)(root + (wa ^ (i * 32)) + 128) = as_int4(u[i + 4]).x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = *(local int*)(root + i * WG * 4 + (ra ^ (((i * (WG / 64)) & 3) * 32))); u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX / 2; ++i) { *(local int*)(root + (wa ^ (i * 32))) = as_int4(u[i]).y; *(local int*)(root + (wa ^ (i * 32)) + 128) = as_int4(u[i + 4]).y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = *(local int*)(root + i * WG * 4 + (ra ^ (((i * (WG / 64)) & 3) * 32))); u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX / 2; ++i) { *(local int*)(root + (wa ^ (i * 32))) = as_int4(u[i]).z; *(local int*)(root + (wa ^ (i * 32)) + 128) = as_int4(u[i + 4]).z; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = *(local int*)(root + i * WG * 4 + (ra ^ (((i * (WG / 64)) & 3) * 32))); u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX / 2; ++i) { *(local int*)(root + (wa ^ (i * 32))) = as_int4(u[i]).w; *(local int*)(root + (wa ^ (i * 32)) + 128) = as_int4(u[i + 4]).w; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = *(local int*)(root + i * WG * 4 + (ra ^ (((i * (WG / 64)) & 3) * 32))); u[i] = as_T2_GF61(tmp); }
      LDStx_end(lds2, numWG);
      return;
    }

    // Same permutation as the 8-byte path's "Special case first RADIX == 4" swizzle above, done in 4 int-sized passes.
    if (f == 1 && r == 4 && RADIX == 4) {
      LDStx_start_with_fence(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = as_int4(u[i]).x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = as_int4(u[i]).y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = as_int4(u[i]).z; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = as_int4(u[i]).w; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; u[i] = as_T2_GF61(tmp); }
      LDStx_end(lds2, numWG);
      return;
    }

    // Same permutation as the 8-byte path's "Special case second RADIX == 4" swizzle above, done in 4 int-sized passes.
    if (f == 4 && r == 4 && RADIX == 4) {
      LDStx_start_with_fence(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = as_int4(u[i]).x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = as_int4(u[i]).y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = as_int4(u[i]).z; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; u[i] = as_T2_GF61(tmp); }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = as_int4(u[i]).w; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; u[i] = as_T2_GF61(tmp); }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

    // Otherwise (no LDSPAD/LDSSWIZ case above matched, or neither is enabled): execute the original shufl code
    // modified to handle the case where a full RADIX fft was not done.  NOT OPTIMIZED TO REDUCE LDS BANK CONFLICTS!!
    // Use the same write index as the 8- and 16-byte paths: it honours r, which is smaller than RADIX when
    // the caller has done only a partial fft_RADIX step (fft8_4 on the SIZE=256/RADIX=8 path).  For r == RADIX
    // this is identical to the i * f + (lowMe & ~mask) * RADIX + (lowMe & mask) it replaces.

    // NOTE: This is the one known case where Radeon VIIs require a local memory fence!  The reason is not known, I would think each LDSbar call
    // would need the local memory fence, but only the transaction start does!?

    LDStx_start_with_fence(lds2, numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i / (RADIX / r) * f + i % (RADIX / r) * WG * r + (lowMe & ~mask) * r + (lowMe & mask)] = as_int4(u[i]).x; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * WG + lowMe]; u[i] = as_T2_GF61(tmp); }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i / (RADIX / r) * f + i % (RADIX / r) * WG * r + (lowMe & ~mask) * r + (lowMe & mask)] = as_int4(u[i]).y; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * WG + lowMe]; u[i] = as_T2_GF61(tmp); }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i / (RADIX / r) * f + i % (RADIX / r) * WG * r + (lowMe & ~mask) * r + (lowMe & mask)] = as_int4(u[i]).z; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * WG + lowMe]; u[i] = as_T2_GF61(tmp); }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i / (RADIX / r) * f + i % (RADIX / r) * WG * r + (lowMe & ~mask) * r + (lowMe & mask)] = as_int4(u[i]).w; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * WG + lowMe]; u[i] = as_T2_GF61(tmp); }
    LDStx_end(lds2, numWG);
    return;
  }
}

// Shortcut for the most common case where caller did a full RADIX step (as opposed to the oddball cases where we have a u[8] but only did a radix 2 or 4 step).
void OVERLOAD shufl(local T2_GF61 *lds2, T2_GF61 *u, u32 f, u32 numWG, u32 lowMe) {
  shufl(lds2, u, f, RADIX, numWG, lowMe);
}


// Shufl two or more fft_WIDTHs or fft_HEIGHTs operating on 64-bit values using LDS_BYTES of LDS memory.  An fft2 is also performed.
// At present, this is only used by WIDTH or HEIGHT = 1K with RADIX=8 and f=8.
void OVERLOAD shufl_and_fft2(local T2_GF61 *lds2, T2_GF61 *u, u32 f, u32 numWG, u32 lowMe) {
  assert(RADIX == 8);

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  // Start by doing the writes of a standard shufl.
  // Next, each thread reads a pair of values.  The lower threads add the two values, the higher threads subtract the two values.
  // val1 is read from          i * WG/2
  // val2 is read from 4 * WG + i * WG/2

  // If SHUFL_BYTES is 16 we can write the complete T2 value to LDS memory with one instruction.
  if (SBMUL(numWG) * SHUFL_BYTES >= 16) {
    local T2_GF61* lds = LDSsharing_ptr(lds2, numWG);

    // Execute the original shufl code with an fft2 add-on.
    LDStx_start(lds2, numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i]; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) {
      T2_GF61 val1 = lds[         i * (WG / 2) + lowMe % (WG / 2)];
      T2_GF61 val2 = lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)];
      if (lowMe < WG / 2) u[i] = addq(val1, val2);
      else u[i] = subq(val1, val2);
    }
    LDStx_end(lds2, numWG);
    return;
  }

  // If SHUFL_BYTES is 8 we split the T2 values into two T values.  These are written to LDS memory with two instructions.
  else if (SBMUL(numWG) * SHUFL_BYTES == 8) {
    local T_Z61* lds = LDSsharing_ptr((local T_Z61 *)lds2, numWG);

#if LDSPAD
    // Special case second RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS with 8 pads after each row.
    // Read from LDS in output order.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    if (f == 8 && RADIX == 8) {
      local T_Z61 *ldsIn;
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = u[i].x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        // Read val1 from the standard shufl's i = i/2,     lowMe = lowMe % WG/2 + (i&1) * WG/2
        // Read val2 from the standard shufl's i = i/2 + 4, lowMe = lowMe % WG/2 + (i&1) * WG/2
        T_Z61 val1 = lds[(i / 2)     * (WG / 64) * 8 + (((i & 1) * (WG / 2)) / 64) * 8 + ((lowMe % (WG / 2)) / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)];
        T_Z61 val2 = lds[(i / 2 + 4) * (WG / 64) * 8 + (((i & 1) * (WG / 2)) / 64) * 8 + ((lowMe % (WG / 2)) / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)];
        if (lowMe < WG / 2) u[i].x = addq(val1, val2); 
        else u[i].x = subq(val1, val2);
      }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = u[i].y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        // Read val1 from the standard shufl's i = i/2,     lowMe = lowMe % WG/2 + (i&1) * WG/2
        // Read val2 from the standard shufl's i = i/2 + 4, lowMe = lowMe % WG/2 + (i&1) * WG/2
        T_Z61 val1 = lds[(i / 2)     * (WG / 64) * 8 + (((i & 1) * (WG / 2)) / 64) * 8 + ((lowMe % (WG / 2)) / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)];
        T_Z61 val2 = lds[(i / 2 + 4) * (WG / 64) * 8 + (((i & 1) * (WG / 2)) / 64) * 8 + ((lowMe % (WG / 2)) / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)];
        if (lowMe < WG / 2) u[i].y = addq(val1, val2);
        else u[i].y = subq(val1, val2);
      }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

#if LDSSWIZ
    // Special case second RADIX == 8 to eliminate LDS bank conflicts, with an fft2 add-on.  Works for any WG >= 128.
    // The write is plain shufl's f == 8 swizzle.  val1 is plain shufl's read of i / 2 at lane lowMe % (WG / 2) + (i & 1) * (WG / 2),
    // which is index i * (WG / 2) + lowMe % (WG / 2) under the same swizzle; val2 is 4 * WG further (ds_read2st64_b64 pairs).
    // Addresses are byte offsets from lds2 XORed with a constant for each i, see plain shufl's LDSSWIZ cases.
    if (f == 8 && RADIX == 8 && WG >= 128) {
      LDStx_start(lds2, numWG);
      local char* root = (local char*)lds2;
      u32 region = (u32)((local char*)lds - root);
      u32 lowHalf = lowMe % (WG / 2);
      u32 wa = region + (((lowMe / 8) * 64 + (lowMe & 7)) ^ (lowMe & 8)) * 8;
      u32 ra = region + (lowHalf ^ ((lowHalf / 8) & 8)) * 8;
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX; i += 2) { *(local T_Z61*)(root + wa + i * 64) = u[i].x; }             // even i first so that same-base stores are adjacent
      for (u32 i = 1; i < RADIX; i += 2) { *(local T_Z61*)(root + (wa ^ 64) + (i - 1) * 64) = u[i].x; }  // and can be merged into ds_write2
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        u32 a1 = i * (WG / 2) * 8 + (ra ^ (((i * (WG / 2) / 8) & 8) * 8));
        T_Z61 val1 = *(local T_Z61*)(root + a1);
        T_Z61 val2 = *(local T_Z61*)(root + a1 + 4 * WG * 8);
        if (lowMe < WG / 2) u[i].x = addq(val1, val2);
        else u[i].x = subq(val1, val2);
      }
      LDSbar(numWG);
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX; i += 2) { *(local T_Z61*)(root + wa + i * 64) = u[i].y; }             // even i first so that same-base stores are adjacent
      for (u32 i = 1; i < RADIX; i += 2) { *(local T_Z61*)(root + (wa ^ 64) + (i - 1) * 64) = u[i].y; }  // and can be merged into ds_write2
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        u32 a1 = i * (WG / 2) * 8 + (ra ^ (((i * (WG / 2) / 8) & 8) * 8));
        T_Z61 val1 = *(local T_Z61*)(root + a1);
        T_Z61 val2 = *(local T_Z61*)(root + a1 + 4 * WG * 8);
        if (lowMe < WG / 2) u[i].y = addq(val1, val2);
        else u[i].y = subq(val1, val2);
      }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

    // Execute the original shufl code with an fft2 add-on.
    LDStx_start(lds2, numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].x; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) {
      T_Z61 val1 = lds[         i * (WG / 2) + lowMe % (WG / 2)];
      T_Z61 val2 = lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)];
      if (lowMe < WG / 2) u[i].x = addq(val1, val2);
      else u[i].x = subq(val1, val2);
    }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].y; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) {
      T_Z61 val1 = lds[         i * (WG / 2) + lowMe % (WG / 2)];
      T_Z61 val2 = lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)];
      if (lowMe < WG / 2) u[i].y = addq(val1, val2);
      else u[i].y = subq(val1, val2);
    }
    LDStx_end(lds2, numWG);
    return;
  }

  // If SHUFL_BYTES is 4 we split the T2 values into 4 int values.  These are written to LDS memory using four instructions.
  else if (SBMUL(numWG) * SHUFL_BYTES == 4) {

    // Lower LDS requirements may let the optimizer use fewer VGPRs and increase occupancy for WIDTHs >= 1024.
    // Alas, the increased occupancy does not offset extra code needed for shufl_int (the assembly
    // code generated is not pretty).  This might not be true for nVidia or future ROCm optimizers.
    local int* lds = (local int*)LDSsharing_ptr(lds2, numWG);

#if LDSPAD
    // Same permutation as the 8-byte LDSPAD case above, done in 4 int-sized passes.  As in the fallback below, all four 32-bit
    // pieces of val1 and val2 are gathered before the fft2.  val2 is a constant 32 * (WG / 64) ints after val1.
    if (f == 8 && RADIX == 8) {
      int4 v1[RADIX], v2[RADIX];
      LDStx_start_with_fence(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = as_int4(u[i]).x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        u32 k = (i / 2) * (WG / 64) * 8 + (((i & 1) * (WG / 2)) / 64) * 8 + ((lowMe % (WG / 2)) / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7);
        v1[i].x = lds[k]; v2[i].x = lds[k + 4 * (WG / 64) * 8];
      }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = as_int4(u[i]).y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        u32 k = (i / 2) * (WG / 64) * 8 + (((i & 1) * (WG / 2)) / 64) * 8 + ((lowMe % (WG / 2)) / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7);
        v1[i].y = lds[k]; v2[i].y = lds[k + 4 * (WG / 64) * 8];
      }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = as_int4(u[i]).z; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        u32 k = (i / 2) * (WG / 64) * 8 + (((i & 1) * (WG / 2)) / 64) * 8 + ((lowMe % (WG / 2)) / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7);
        v1[i].z = lds[k]; v2[i].z = lds[k + 4 * (WG / 64) * 8];
      }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = as_int4(u[i]).w; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        u32 k = (i / 2) * (WG / 64) * 8 + (((i & 1) * (WG / 2)) / 64) * 8 + ((lowMe % (WG / 2)) / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7);
        v1[i].w = lds[k]; v2[i].w = lds[k + 4 * (WG / 64) * 8];
      }
      LDStx_end(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        T2_GF61 val1 = as_T2_GF61(v1[i]);
        T2_GF61 val2 = as_T2_GF61(v2[i]);
        if (lowMe < WG / 2) u[i] = addq(val1, val2);
        else u[i] = subq(val1, val2);
      }
      return;
    }
#endif

#if LDSSWIZ
    // Same swizzle as plain shufl's 4-byte f == 8 case (only bits 0-1 of i XORed, so u[i] and u[i + 4] are 32 ints apart), with
    // val1 at index i * (WG / 2) + lowMe % (WG / 2) under that swizzle and val2 4 * WG ints further.  Works for any WG >= 128.
    // Addresses are byte offsets from lds2 XORed with a constant for each i, see plain shufl's LDSSWIZ cases and SWIZ_RECOMPUTE.
    if (f == 8 && RADIX == 8 && WG >= 128) {
      int4 v1[RADIX], v2[RADIX];
      LDStx_start_with_fence(lds2, numWG);
      local char* root = (local char*)lds2;
      u32 region = (u32)((local char*)lds - root);
      u32 lowHalf = lowMe % (WG / 2);
      u32 wa = region + ((lowMe / 8) * 64 + ((lowMe / 8) & 3) * 8 + (lowMe & 7)) * 4;
      u32 ra = region + (lowHalf ^ (((lowHalf / 64) & 3) * 8)) * 4;
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX / 2; ++i) { *(local int*)(root + (wa ^ (i * 32))) = as_int4(u[i]).x; *(local int*)(root + (wa ^ (i * 32)) + 128) = as_int4(u[i + 4]).x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        u32 a1 = i * (WG / 2) * 4 + (ra ^ ((((i * (WG / 2)) / 64) & 3) * 32));
        v1[i].x = *(local int*)(root + a1); v2[i].x = *(local int*)(root + a1 + 4 * WG * 4);
      }
      LDSbar(numWG);
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX / 2; ++i) { *(local int*)(root + (wa ^ (i * 32))) = as_int4(u[i]).y; *(local int*)(root + (wa ^ (i * 32)) + 128) = as_int4(u[i + 4]).y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        u32 a1 = i * (WG / 2) * 4 + (ra ^ ((((i * (WG / 2)) / 64) & 3) * 32));
        v1[i].y = *(local int*)(root + a1); v2[i].y = *(local int*)(root + a1 + 4 * WG * 4);
      }
      LDSbar(numWG);
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX / 2; ++i) { *(local int*)(root + (wa ^ (i * 32))) = as_int4(u[i]).z; *(local int*)(root + (wa ^ (i * 32)) + 128) = as_int4(u[i + 4]).z; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        u32 a1 = i * (WG / 2) * 4 + (ra ^ ((((i * (WG / 2)) / 64) & 3) * 32));
        v1[i].z = *(local int*)(root + a1); v2[i].z = *(local int*)(root + a1 + 4 * WG * 4);
      }
      LDSbar(numWG);
      if (SWIZ_RECOMPUTE) { OPAQUE(wa); OPAQUE(ra); }
      for (u32 i = 0; i < RADIX / 2; ++i) { *(local int*)(root + (wa ^ (i * 32))) = as_int4(u[i]).w; *(local int*)(root + (wa ^ (i * 32)) + 128) = as_int4(u[i + 4]).w; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        u32 a1 = i * (WG / 2) * 4 + (ra ^ ((((i * (WG / 2)) / 64) & 3) * 32));
        v1[i].w = *(local int*)(root + a1); v2[i].w = *(local int*)(root + a1 + 4 * WG * 4);
      }
      LDStx_end(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        T2_GF61 val1 = as_T2_GF61(v1[i]);
        T2_GF61 val2 = as_T2_GF61(v2[i]);
        if (lowMe < WG / 2) u[i] = addq(val1, val2);
        else u[i] = subq(val1, val2);
      }
      return;
    }
#endif

    // Otherwise (no LDSPAD/LDSSWIZ case above): the original shufl code, NOT OPTIMIZED TO REDUCE LDS BANK CONFLICTS.
    // The fft2 has to add and subtract whole 64-bit values, so first gather all four 32-bit pieces of val1 and val2
    // (same LDS locations as the 16- and 8-byte paths above), then combine.  u[] stays intact as the source of the
    // four write passes.
    int4 v1[RADIX], v2[RADIX];
    LDStx_start(lds2, numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).x; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { v1[i].x = lds[i * (WG / 2) + lowMe % (WG / 2)]; v2[i].x = lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)]; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).y; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { v1[i].y = lds[i * (WG / 2) + lowMe % (WG / 2)]; v2[i].y = lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)]; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).z; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { v1[i].z = lds[i * (WG / 2) + lowMe % (WG / 2)]; v2[i].z = lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)]; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).w; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { v1[i].w = lds[i * (WG / 2) + lowMe % (WG / 2)]; v2[i].w = lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)]; }
    LDStx_end(lds2, numWG);
    for (u32 i = 0; i < RADIX; ++i) {
      T2_GF61 val1 = as_T2_GF61(v1[i]);
      T2_GF61 val2 = as_T2_GF61(v2[i]);
      if (lowMe < WG / 2) u[i] = addq(val1, val2);
      else u[i] = subq(val1, val2);
    }
    return;
  }
}

#endif


#ifdef F2_GF31

// Shufl two or more fft_WIDTHs or FFT_HEIGHTs using two 4-byte floats or Z31s.
void OVERLOAD shufl(local F2_GF31 *lds2, F2_GF31 *u, u32 f, u32 r, u32 numWG, u32 lowMe) {

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  //GW - would a 16 byte implementation be useful?  Less LDS conflict work?

  // If SHUFL_BYTES is 8 or more we can write the complete F2 value to LDS memory with one instruction.
  // We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
  if (SBMUL(numWG) * SHUFL_BYTES >= 8) {
    local F2_GF31* lds = LDSsharing_ptr(lds2, numWG);

#if LDSPAD
    // Special case first RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are in order and written straight to LDS memory.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that uses a little padding.  Pad two values after every row to eliminate bank conflicts.
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 448, 1, 65...   output[64..127] = +8
    if (f == 1 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 2) + lowMe] = u[i]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * WG / 8 + (lowMe / 8) + (lowMe & 7) * (WG + 2)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Pad 8 values after every 64 values to eliminate bank conflicts.
    if (1 && f == 8 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
    // One expression for every WG.  The old pair of arms laid the data out as exactly eight padded rows,
    // which only inverts when WG/64 == 8, and read a row of 64 regardless of WG: correct at WG 64 and 512,
    // wrong everywhere between (896 of 1024 elements at WG 128, 1792 of 2048 at WG 256, some of them
    // reading slots nothing had written).  This is the form the 64-bit sibling above already uses.
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = u[i]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS with 8 pads after each row.
    // Read from LDS in output order.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    if (0 && f == 8 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = u[i]; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * (WG / 64) * 8                    + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...192, 1.., 2.., 3.., 16...   lds[64..127] = +4
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 1, 65...   output[64..127] = +16
    // Pad one value after every row to eliminate bank conflicts.
    if (f == 1 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = u[i]; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 16                     +  (lowMe / 16)      * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0...192, 16..., 32..., 48..., 4...   lds[64..127] = +1
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0...192, 16... 32.. 48.. 1...  lds[64..127] = +4
    // Pad 4 values after every row to eliminate bank conflicts.
    if (f == 4 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 4) + (lowMe / 16) * 16 + i * 4 + (lowMe & 3)] = u[i]; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

#if LDSSWIZ
    // Special case first RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 1, 65...   lds[64..127] = +8
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 1 and 2 multiples of 8) and the first 26 threads read from LDS (multiples of 64 and two multiples of 1).
    if (f == 1 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 8 + i) ^ (lowMe & 15)] = u[i]; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ (((i & 1) * 8) + ((lowMe / 8) & 7))]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ (((lowMe / 8) & 15))]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 64 and 2 multiples of 1) and the first 16 threads read from LDS (8 multiples of 64 and two multiples of 8).
    if (f == 8 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 8 * 64 + i * 8 + (lowMe & 7)) ^ (lowMe & 8)] = u[i]; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((i & 1) * 8)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 8) & 8)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 1, 65...   lds[64..127] = +16
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 1 and 4 multiples of 4) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 1).
    if (f == 1 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = u[i]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16, 80 ...   lds[64..127] = +4
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 64 and 4 multiples of 1) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 16).
    if (f == 4 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = u[i]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

    // Otherwise, execute the original shufl code modified to handle case where a full RADIX fft was not done
    LDStx_start(lds2, numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i / (RADIX / r) * f + i % (RADIX / r) * WG * r + (lowMe & ~mask) * r + (lowMe & mask)] = u[i]; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * WG + lowMe]; }
    LDStx_end(lds2, numWG);
    return;
  }

  // If SHUFL_BYTES is 4 we split the F2 values into two F values.  These are written to LDS memory using two instructions.
  // We're writing 4 bytes at a time, which means groups of 32 must have unique LDS banks.
  else if (SBMUL(numWG) * SHUFL_BYTES == 4) {
    local F_Z31* lds = LDSsharing_ptr((local F_Z31 *)lds2, numWG);

#if LDSPAD
    // Special case first RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...448, 1, 65..., 2, 66..., 3, 67..., 32, 96...   lds[64..127] = +4
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 448, 1, 65...   output[64..127] = +8
    // Pad one value after every row to eliminate bank conflicts.
    if (f == 1 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 7) * (WG + 1) + (lowMe / 32) * 32 + (lowMe & 3) * 8 + i] = u[i].x; }
      LDSbar(numWG);
      // Read back in the generic shufl's output order.  The write above stores (i', me') at
      // ((me'/4)&7)*(WG+1) + (me'/32)*32 + (me'&3)*8 + i', and output (i, lowMe) needs i' = lowMe & 7,
      // me' = i*WG/8 + lowMe/8; the per-WG forms below are that inverse with the constants folded.
      if      (WG == 64)  for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i / 4) * 32 + (i & 3) * (2 * (WG + 1)) +  (lowMe / 32)      * (WG + 1) + (lowMe & 31)]; }
      else if (WG == 128) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i / 2) * 32 + (4 * (i & 1) + lowMe / 32) * (WG + 1) + (lowMe & 31)]; }
      else if (WG == 512) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + (lowMe / 256) * 32             + ((lowMe / 32) & 7) * (WG + 1) + (lowMe & 31)]; }
      else                for (u32 i = 0; i < RADIX; ++i) { u32 mep = i * (WG / 8) + lowMe / 8; u[i].x = lds[((mep / 4) & 7) * (WG + 1) + (mep / 32) * 32 + (mep & 3) * 8 + (lowMe & 7)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 7) * (WG + 1) + (lowMe / 32) * 32 + (lowMe & 3) * 8 + i] = u[i].y; }
      LDSbar(numWG);
      if      (WG == 64)  for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i / 4) * 32 + (i & 3) * (2 * (WG + 1)) +  (lowMe / 32)      * (WG + 1) + (lowMe & 31)]; }
      else if (WG == 128) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i / 2) * 32 + (4 * (i & 1) + lowMe / 32) * (WG + 1) + (lowMe & 31)]; }
      else if (WG == 512) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + (lowMe / 256) * 32             + ((lowMe / 32) & 7) * (WG + 1) + (lowMe & 31)]; }
      else                for (u32 i = 0; i < RADIX; ++i) { u32 mep = i * (WG / 8) + lowMe / 8; u[i].y = lds[((mep / 4) & 7) * (WG + 1) + (mep / 32) * 32 + (mep & 3) * 8 + (lowMe & 7)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Pad 8 values after every 64 values to eliminate bank conflicts.
    if (f == 8 && r == 8 && RADIX == 8) {
      LDStx_start(lds2, numWG);
    // One expression for every WG.  The old pair of arms laid the data out as exactly eight padded rows,
    // which only inverts when WG/64 == 8, and read a row of 64 regardless of WG: correct at WG 64 and 512,
    // wrong everywhere between (896 of 1024 elements at WG 128, 1792 of 2048 at WG 256, some of them
    // reading slots nothing had written).  This is the form the 64-bit sibling above already uses.
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = u[i].x; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = u[i].y; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...192, 1.., 2.., 7.., 32...   lds[64..127] = +8
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 1, 65...   output[64..127] = +16
    // Pad one value after every row to eliminate bank conflicts.
    if (f == 1 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 3) * (WG + 1) + (lowMe / 32) * 32 + (lowMe & 7) * 4 + i] = u[i].x; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i / 2) * 32 + (i & 1) * (2 * (WG + 1)) +  (lowMe / 32)      * (WG + 1) + (lowMe & 31)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 +             (lowMe / 128) * 32 + ((lowMe / 32) & 3) * (WG + 1) + (lowMe & 31)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 3) * (WG + 1) + (lowMe / 32) * 32 + (lowMe & 7) * 4 + i] = u[i].y; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i / 2) * 32 + (i & 1) * (2 * (WG + 1)) +  (lowMe / 32)      * (WG + 1) + (lowMe & 31)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 +             (lowMe / 128) * 32 + ((lowMe / 32) & 3) * (WG + 1) + (lowMe & 31)]; }
      LDStx_end(lds2, numWG);
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0...192, 16..., 32..., 48..., 1.... ... 8...   lds[64..127] = +2
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0...192, 16... 32.. 48.. 1...  lds[64..127] = +4
    // Pad 4 values after every row to eliminate bank conflicts.
    if (f == 4 && r == 4 && RADIX == 4) {
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 3) * (WG + 4) + (lowMe / 32) * 32 + ((lowMe / 4) & 1) * 16 + i * 4 + (lowMe & 3)] = u[i].x; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i / 2) * 32 + (i & 1) * (2 * (WG + 4)) +  (lowMe / 32)      * (WG + 4) + (lowMe & 31)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64             + (lowMe / 128) * 32 + ((lowMe / 32) & 3) * (WG + 4) + (lowMe & 31)]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 3) * (WG + 4) + (lowMe / 32) * 32 + ((lowMe / 4) & 1) * 16 + i * 4 + (lowMe & 3)] = u[i].y; }
      LDSbar(numWG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i / 2) * 32 + (i & 1) * (2 * (WG + 4)) +  (lowMe / 32)      * (WG + 4) + (lowMe & 31)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64             + (lowMe / 128) * 32 + ((lowMe / 32) & 3) * (WG + 4) + (lowMe & 31)]; }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

    // Otherwise, execute the original shufl code modified to handle case where a full RADIX fft was not done
    LDStx_start(lds2, numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i / (RADIX / r) * f + i % (RADIX / r) * WG * r + (lowMe & ~mask) * r + (lowMe & mask)] = u[i].x; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * WG + lowMe]; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i / (RADIX / r) * f + i % (RADIX / r) * WG * r + (lowMe & ~mask) * r + (lowMe & mask)] = u[i].y; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * WG + lowMe]; }
    LDStx_end(lds2, numWG);
    return;
  }
}

// Shortcut for the most common case where caller did a full RADIX step (as opposed to the oddball cases where we have a u[8] but only did a radix 2 or 4 step).
void OVERLOAD shufl(local F2_GF31 *lds2, F2_GF31 *u, u32 f, u32 numWG, u32 lowMe) {
  shufl(lds2, u, f, RADIX, numWG, lowMe);
}


// NEEDS TONS OF WORK!!!  SWIZ NOT CODED, MOST PAD CASES NOT CODED.
// At present, this is only used by WIDTH or HEIGHT = 1K with RADIX=8 and f=8.

// Shufl two or more fft_WIDTHs or fft_HEIGHTs operating on 32-bit values using LDS_BYTES of LDS memory.  An fft2 is also performed.
void OVERLOAD shufl_and_fft2(local F2_GF31 *lds2, F2_GF31 *u, u32 f, u32 numWG, u32 lowMe) {
  assert(RADIX == 8);

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  // Start by doing the writes of a standard shufl.
  // Next, each thread reads a pair of values.  The lower threads add the two values, the higher threads subtract the two values.
  // val1 is read from          i * WG/2
  // val2 is read from 4 * WG + i * WG/2

  // If SHUFL_BYTES is 8 or more we can write the complete F2 value to LDS memory with one instruction.
  if (SBMUL(numWG) * SHUFL_BYTES >= 8) {
    local F2_GF31* lds = LDSsharing_ptr(lds2, numWG);

#if LDSPAD
    // Special case second RADIX == 8 to eliminate LDS bank conflicts.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS with 8 pads after each row.
    // Read from LDS in output order.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    if (f == 8 && RADIX == 8) {
      local F2_GF31 *ldsIn;
      LDStx_start(lds2, numWG);
      for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 8) + lowMe] = u[i]; }
      LDSbar(numWG);
      for (u32 i = 0; i < RADIX; ++i) {
        // Read val1 from the standard shufl's i = i/2,     lowMe = lowMe % WG/2 + (i&1) * WG/2
        // Read val2 from the standard shufl's i = i/2 + 4, lowMe = lowMe % WG/2 + (i&1) * WG/2
        F2_GF31 val1 = lds[(i / 2)     * (WG / 64) * 8 + (((i & 1) * (WG / 2)) / 64) * 8 + ((lowMe % (WG / 2)) / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)];
        F2_GF31 val2 = lds[(i / 2 + 4) * (WG / 64) * 8 + (((i & 1) * (WG / 2)) / 64) * 8 + ((lowMe % (WG / 2)) / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)];
        if (lowMe < WG / 2) u[i] = addq(val1, val2);
        else u[i] = subq(val1, val2);
      }
      LDStx_end(lds2, numWG);
      return;
    }
#endif

    // Execute the original shufl code with an fft2 add-on.
    LDStx_start(lds2, numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i]; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) {
      F2_GF31 val1 = lds[         i * (WG / 2) + lowMe % (WG / 2)];
      F2_GF31 val2 = lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)];
      if (lowMe < WG / 2) u[i] = addq(val1, val2);
      else u[i] = subq(val1, val2);
    }
    LDStx_end(lds2, numWG);
    return;
  }

  // If SHUFL_BYTES is 4 we split the F2 values into two F values.  These are written to LDS memory using two instructions.
  else if (SBMUL(numWG) * SHUFL_BYTES == 4) {
    local F_Z31* lds = LDSsharing_ptr((local F_Z31 *)lds2, numWG);

    // Execute the original shufl code with an fft2 add-on.
    LDStx_start(lds2, numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].x; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) {
      F_Z31 val1 = lds[         i * (WG / 2) + lowMe % (WG / 2)];
      F_Z31 val2 = lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)];
      if (lowMe < WG / 2) u[i].x = addq(val1, val2);
      else u[i].x = subq(val1, val2);
    }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].y; }
    LDSbar(numWG);
    for (u32 i = 0; i < RADIX; ++i) {
      F_Z31 val1 = lds[         i * (WG / 2) + lowMe % (WG / 2)];
      F_Z31 val2 = lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)];
      if (lowMe < WG / 2) u[i].y = addq(val1, val2);
      else u[i].y = subq(val1, val2);
    }
    LDStx_end(lds2, numWG);
    return;
  }
}

#endif
