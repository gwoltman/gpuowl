// Copyright (C) Mihai Preda

#ifdef T2_GF61

// Shufl two or more fft_WIDTHs or FFT_HEIGHTs operating on 64-bit values using LDS_BYTES of LDS memory.
// Care is taken that each simultaneous workgroup does not interfere with the LDS memory of other simultaneous workgroups --
// even when operating on differernt sized data elements as can happen in an M31+M61 NTT.
// WG = workgroup size of a single fft_WIDTH or fft_HEIGHT
// n = sizeof array u (nW or nH).  n * WG = WIDTH or HEIGHT
// numWG = number of fft_WIDTHs or fft_HEIGHTs being processed simultaneously
// lowMe = me % WG
// NOTE: shufl routines perform a bar(WG) at the start but not at the end.  After calling shufl, a bar(WG) is required
// before next LDS memory usage.  All routines that use LDS memory MUST OBEY THIS PROTOCOL of bar() before LDS use and
// only bar(WG) required before next use.  ALSO NOTE: the first shufl call does not need to do bar(WG).  A relatively
// minor optimization would be to special case the first shufl call.
void OVERLOAD shufl(local T2_GF61 *lds2, T2_GF61 *u, u32 f, u32 numWG, u32 lowMe) {

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  int force_default = 0;
#if NOWG2                       // For timing tests only.  Option to not turn off LDS bank conflict code when numWG > 1.  I've not found a GPU where this is beneficial.
  if (numWG > 1) force_default = 1;
#endif
#if NOLDS2                      // For timing tests only.  Option to not turn off LDS bank for second shufl calls.  I've not found a GPU where this is beneficial.
  if (f != 1) force_default = 1;
#endif

  // If SHUFL_BYTES is 16 we can write the complete T2 value to LDS memory with one instruction.
  if (SHUFL_BYTES == 16) {
    local T2_GF61* lds = lds2;
    if (numWG > 1) lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(T2_GF61);

#if LDSPAD
    // Special case first RADIX == 8 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...448, 8, 72..., 16...   lds[64..127] = +1
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 448, 1, 65...   output[64..127] = +8
    // Pad 1 value every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe & 7) * (WG + 1) + (lowMe / 8) * 8 + i] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 8                     + ((lowMe / 8) & 7) * (WG + 1) + (lowMe & 7)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 32 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 1) + (lowMe & 7)]; }
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // No padding of LDS blocks is needed to eliminate bank conflicts!  Groups of 8 threads are already in separate LDS banks.
    // We could however save a bar() by writing to same locations that the previous shufl wrote to.
    if (0 && f == 8 && RADIX == 8) {
      // for (u32 i = 0; i < RADIX; ++i) { lds[something] = u[i]; }
      bar(WG);
      //for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[something]; }
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...192, 1, 65..., 16...   lds[64..127] = +2
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 1, 65...   output[64..127] = +16
    // Pad 1 value every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 2) & 3) * (WG + 1) + (lowMe / 8) * 8 + (lowMe & 1) * 4 + i] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * WG / 4 + (lowMe / 32) * 8 + ((lowMe / 8) & 3) * (WG + 1) + (lowMe & 7)]; }
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16... 4..  lds[64..127] = +1
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 16... 1..   output[64..127] = +4
    // Pad 4 values after every row to eliminate bank conflicts.
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[lowMe / 4 * (WG + 4) + i * 4 + (lowMe & 3)] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      return;
    }
#endif

#if LDSSWIZ
    // Special case first RADIX == 8 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 1, 65...   lds[64..127] = +8
    // Swizzle LDS blocks to eliminate bank conflicts.  Swizzle on the first 8 threads written to LDS (multiples of 1) and the first 8 threads read from LDS (multiples of 64).
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 8 + i) ^ (lowMe & 7)] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 8) & 7)]; }
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // No swizzle of LDS blocks is needed to eliminate bank conflicts.  The first 8 threads written to LDS (multiples of 64) and
    // the first 8 threads read from LDS (multiples of 64) are already in separate LDS banks.
    // We can however save a bar() by writing to same locations that previous shufl wrote to.
    if (!force_default && f == 8 && RADIX == 8) {
      for (u32 i = 0; i < RADIX; ++i) { lds[i * WG + lowMe] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[lowMe / 8 * 64 + i * 8 + (lowMe & 7)]; }
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 1, 65...   lds[64..127] = +16
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 8 threads written to LDS (4 multiples of 1 and 2 multiples of 4) and the first 8 threads read from LDS (4 multiples of 64 and 2 multiples of 1).
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 7)] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 7)]; }
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16, 80...   lds[64..127] = +4
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 8 threads written to LDS (4 multiples of 64 and 2 multiples of 1) and the first 8 threads read from LDS (4 multiples of 64 and 2 multiples of 4).
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 4)] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 4)]; }
      return;
    }
#endif

    // Otherwise, execute the original shufl code
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i]; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * WG + lowMe]; }
  }

  // If SHUFL_BYTES is 8 we split the T2 values into two T values.  These are written to LDS memory with two instructions.
  else if (SHUFL_BYTES == 8) {
    local T_Z61* lds = ((local T_Z61*) lds2);
    if (numWG > 1) lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(T_Z61);

#if LDSPAD
    // Special case first RADIX == 8 code to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...448, 1, 65..., 16, 80...   lds[64..127] = +2
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 448, 1, 65...   output[64..127] = +8
    // Pad one value after every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 2) & 7) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 1) * 8 + i] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i / 2) * 16 + (i & 1) * (4 * (WG + 1)) + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + (lowMe / 128) * 16             + ((lowMe / 16) & 7) * (WG + 1) + (lowMe & 15)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 2) & 7) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 1) * 8 + i] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i / 2) * 16 + (i & 1) * (4 * (WG + 1)) + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + (lowMe / 128) * 16             + ((lowMe / 16) & 7) * (WG + 1) + (lowMe & 15)]; }
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Pad 8 values after every row to eliminate bank conflicts.
    if (!force_default && f == 8 && RADIX == 8) {
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { lds[ (lowMe / 8)      * (WG + 8)                     + i * 8 + (lowMe & 7)] = u[i].x; }
      else          for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 7) * (WG + 8) + (lowMe / 64) * 64 + i * 8 + (lowMe & 7)] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * (WG + 8) + lowMe]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + lowMe / 64 * (WG + 8) + (lowMe & 63)]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { lds[ (lowMe / 8)      * (WG + 8)                     + i * 8 + (lowMe & 7)] = u[i].y; }
      else          for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 7) * (WG + 8) + (lowMe / 64) * 64 + i * 8 + (lowMe & 7)] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * (WG + 8) + lowMe]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + lowMe / 64 * (WG + 8) + (lowMe & 63)]; }
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...192, 1.., 2.., 3.., 16...   lds[64..127] = +4
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 1, 65...   output[64..127] = +16
    // Pad one value after every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 16                     + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 16                     + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0...192, 16..., 32..., 48..., 4...   lds[64..127] = +1
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0...192, 16... 32.. 48.. 1...  lds[64..127] = +4
    // Pad 4 values after every row to eliminate bank conflicts.
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 4) + (lowMe / 16) * 16 + i * 4 + (lowMe & 3)] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 4) + (lowMe / 16) * 16 + i * 4 + (lowMe & 3)] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      return;
    }
#endif

#if LDSSWIZ
    // Special case first RADIX == 8 code to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 1, 65...   lds[64..127] = +8
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 1 and 2 multiples of 8) and the first 16 threads read from LDS (8 multiples of 64 and 2 multiples of 1).
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 8 + i) ^ (lowMe & 15)] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ (((i & 1) * 8) + ((lowMe / 8) & 7))]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ (((lowMe / 8) & 15))]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 8 + i) ^ (lowMe & 15)] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ (((i & 1) * 8) + ((lowMe / 8) & 7))]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ (((lowMe / 8) & 15))]; }
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 64 and 2 multiples of 1) and the first 16 threads read from LDS (8 multiples of 64 and 2 multiples of 8).
    if (!force_default && f == 8 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 8 * 64 + i * 8 + (lowMe & 7)) ^ (lowMe & 8)] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ ((i & 1) * 8)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ ((lowMe / 8) & 8)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 8 * 64 + i * 8 + (lowMe & 7)) ^ (lowMe & 8)] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ ((i & 1) * 8)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ ((lowMe / 8) & 8)]; }
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 1, 65...   lds[64..127] = +16
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 1 and 4 multiples of 4) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 1).
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = u[i].x; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = u[i].y; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; }
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16, 80...   lds[64..127] = +4
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 64 and 4 multiples of 1) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 16).
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = u[i].x; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = u[i].y; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; }
      return;
    }
#endif

    // Execute the original shufl code
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].x; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * WG + lowMe]; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].y; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * WG + lowMe]; }
  }

  // If SHUFL_BYTES is 4 we split the T2 values into 4 int values.  These are written to LDS memory using four instructions.
  // NOT OPTIMIZED TO REDUCE LDS BANK CONFLICTS!!
  else if (SHUFL_BYTES == 4) {
    // Lower LDS requirements may let the optimizer use fewer VGPRs and increase occupancy for WIDTHs >= 1024.
    // Alas, the increased occupancy does not offset extra code needed for shufl_int (the assembly
    // code generated is not pretty).  This might not be true for nVidia or future ROCm optimizers.
    local int* lds = (local int*) lds2;
    if (numWG > 1) lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(int);

    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).x; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * WG + lowMe]; u[i] = as_T2_GF61(tmp); }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).y; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * WG + lowMe]; u[i] = as_T2_GF61(tmp); }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).z; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * WG + lowMe]; u[i] = as_T2_GF61(tmp); }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).w; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * WG + lowMe]; u[i] = as_T2_GF61(tmp); }
  }
}

#endif


#ifdef F2_GF31

// Shufl two or more fft_WIDTHs or FFT_HEIGHTs using two 4-byte floats or Z31s.
void OVERLOAD shufl(local F2_GF31 *lds2, F2_GF31 *u, u32 f, u32 numWG, u32 lowMe) {

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  //GW - would a 16 byte implementation be useful?  Less LDS conflict work?

  int force_default = 0;
#if NOWG2
  if (numWG > 1) force_default = 1;
#endif
#if NOLDS2
  if (f != 1) force_default = 1;
#endif

  // If SHUFL_BYTES is 8 or more we can write the complete F2 value to LDS memory with one instruction.
  if (SHUFL_BYTES >= 8) {
    local F2_GF31* lds = lds2;
    if (numWG > 1) lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(F2_GF31);

#if LDSPAD
    // Special case first RADIX == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...448, 1, 65..., 16, 80...   lds[64..127] = +2
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 448, 1, 65...   output[64..127] = +8
    // Pad one value after every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 2) & 7) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 1) * 8 + i] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i / 2) * 16 + (i & 1) * (4 * (WG + 1)) + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + (lowMe / 128) * 16             + ((lowMe / 16) & 7) * (WG + 1) + (lowMe & 15)]; }
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Pad 8 values after every 64 values to eliminate bank conflicts.
    if (!force_default && f == 8 && RADIX == 8) {
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { lds[ (lowMe / 8)      * (WG + 8)                     + i * 8 + (lowMe & 7)] = u[i]; }
      else          for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 7) * (WG + 8) + (lowMe / 64) * 64 + i * 8 + (lowMe & 7)] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * (WG + 8) + lowMe]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + lowMe / 64 * (WG + 8) + (lowMe & 63)]; }
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...192, 1.., 2.., 3.., 16...   lds[64..127] = +4
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 1, 65...   output[64..127] = +16
    // Pad one value after every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 16                     +  (lowMe / 16)      * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0...192, 16..., 32..., 48..., 4...   lds[64..127] = +1
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0...192, 16... 32.. 48.. 1...  lds[64..127] = +4
    // Pad 4 values after every row to eliminate bank conflicts.
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 4) + (lowMe / 16) * 16 + i * 4 + (lowMe & 3)] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      return;
    }
#endif

#if LDSSWIZ
    // Special case first RADIX == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 1, 65...   lds[64..127] = +8
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 1 and 2 multiples of 8) and the first 26 threads read from LDS (multiples of 64 and two multiples of 1).
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 8 + i) ^ (lowMe & 15)] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ (((i & 1) * 8) + ((lowMe / 8) & 7))]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ (((lowMe / 8) & 15))]; }
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 64 and 2 multiples of 1) and the first 16 threads read from LDS (8 multiples of 64 and two multiples of 8).
    if (!force_default && f == 8 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 8 * 64 + i * 8 + (lowMe & 7)) ^ (lowMe & 8)] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((i & 1) * 8)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 8) & 8)]; }
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 1, 65...   lds[64..127] = +16
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 1 and 4 multiples of 4) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 1).
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; }
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16, 80 ...   lds[64..127] = +4
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 64 and 4 multiples of 1) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 16).
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; }
      return;
    }
#endif

    // Execute the original shufl code
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i]; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * WG + lowMe]; }
  }

  // If SHUFL_BYTES is 4 we split the F2 values into two F values.  These are written to LDS memory using two instructions.
  else if (SHUFL_BYTES == 4) {
    local F_Z31* lds = ((local F_Z31*) lds2);
    if (numWG > 1) lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(F_Z31);

#if LDSPAD
    // Special case first RADIX == 8 to eliminate LDS bank conflicts.  We're writing 4 bytes at a time, which means groups of 32 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...448, 1, 65..., 2, 66..., 3, 67..., 32, 96...   lds[64..127] = +4
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 448, 1, 65...   output[64..127] = +8
    // Pad one value after every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 7) * (WG + 1) + (lowMe / 32) * 32 + (lowMe & 3) * 8 + i] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i / 4) * 32 + (i & 3) * (2 * (WG + 1)) +  (lowMe / 32)      * (WG + 1) + (lowMe & 31)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + (lowMe / 256) * 32             + ((lowMe / 32) & 7) * (WG + 1) + (lowMe & 31)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 7) * (WG + 1) + (lowMe / 32) * 32 + (lowMe & 3) * 8 + i] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i / 4) * 32 + (i & 3) * (2 * (WG + 1)) +  (lowMe / 32)      * (WG + 1) + (lowMe & 31)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + (lowMe / 256) * 32             + ((lowMe / 32) & 7) * (WG + 1) + (lowMe & 31)]; }
      return;
    }

    // Special case second RADIX == 8 to eliminate LDS bank conflicts.  We're writing 4 bytes at a time, which means groups of 32 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Pad 8 values after every 64 values to eliminate bank conflicts.
    if (!force_default && f == 8 && RADIX == 8) {
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { lds[ (lowMe / 8)      * (WG + 8)                     + i * 8 + (lowMe & 7)] = u[i].x; }
      else          for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 7) * (WG + 8) + (lowMe / 64) * 64 + i * 8 + (lowMe & 7)] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * (WG + 8) + lowMe]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + lowMe / 64 * (WG + 8) + (lowMe & 63)]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { lds[ (lowMe / 8)      * (WG + 8)                     + i * 8 + (lowMe & 7)] = u[i].y; }
      else          for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 7) * (WG + 8) + (lowMe / 64) * 64 + i * 8 + (lowMe & 7)] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * (WG + 8) + lowMe]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + lowMe / 64 * (WG + 8) + (lowMe & 63)]; }
      return;
    }

    // Special case first RADIX == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 32 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...192, 1.., 2.., 7.., 32...   lds[64..127] = +8
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 1, 65...   output[64..127] = +16
    // Pad one value after every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 3) * (WG + 1) + (lowMe / 32) * 32 + (lowMe & 7) * 4 + i] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i / 2) * 32 + (i & 1) * (2 * (WG + 1)) +  (lowMe / 32)      * (WG + 1) + (lowMe & 31)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 +             (lowMe / 128) * 32 + ((lowMe / 32) & 3) * (WG + 1) + (lowMe & 31)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 3) * (WG + 1) + (lowMe / 32) * 32 + (lowMe & 7) * 4 + i] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i / 2) * 32 + (i & 1) * (2 * (WG + 1)) +  (lowMe / 32)      * (WG + 1) + (lowMe & 31)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 +             (lowMe / 128) * 32 + ((lowMe / 32) & 3) * (WG + 1) + (lowMe & 31)]; }
      return;
    }

    // Special case second RADIX == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0...192, 16..., 32..., 48..., 1.... ... 8...   lds[64..127] = +2
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0...192, 16... 32.. 48.. 1...  lds[64..127] = +4
    // Pad 4 values after every row to eliminate bank conflicts.
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 3) * (WG + 4) + (lowMe / 32) * 32 + ((lowMe / 4) & 1) * 16 + i * 4 + (lowMe & 3)] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i / 2) * 32 + (i & 1) * (2 * (WG + 4)) +  (lowMe / 32)      * (WG + 4) + (lowMe & 31)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64             + (lowMe / 128) * 32 + ((lowMe / 32) & 3) * (WG + 4) + (lowMe & 31)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 3) * (WG + 4) + (lowMe / 32) * 32 + ((lowMe / 4) & 1) * 16 + i * 4 + (lowMe & 3)] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i / 2) * 32 + (i & 1) * (2 * (WG + 4)) +  (lowMe / 32)      * (WG + 4) + (lowMe & 31)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64             + (lowMe / 128) * 32 + ((lowMe / 32) & 3) * (WG + 4) + (lowMe & 31)]; }
      return;
    }
#endif

    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].x; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * WG + lowMe]; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].y; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * WG + lowMe]; }
  }
}

#endif


