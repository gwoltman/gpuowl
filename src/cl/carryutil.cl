// Copyright (C) Mihai Preda

// carryFused's carry.  MUL3 triples the carries, so the MUL3 kernels switch to 64-bit carries at a lower bpw (MUL3_CARRY64).
#if CARRY64 || (MUL3 && MUL3_CARRY64)
typedef i64 CFcarry;
#else
typedef i32 CFcarry;
#endif

// The carry for the non-fused CarryA, CarryB, CarryM kernels.
// Simply use largest possible carry always as the split kernels are slow anyway (and seldomly used normally).
#if FFT_TYPE != FFT32 && FFT_TYPE != FFT31
typedef i64 CarryABM;
#else
typedef i32 CarryABM;
#endif

/********************************/
/*       Helper routines        */
/********************************/

// Return unsigned low bits (number of bits must be between 1 and 31)
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_ubfe)
u32 OVERLOAD ulowBits(i32 u, u32 bits) { return __builtin_amdgcn_ubfe(u, 0, bits); }
#elif HAS_PTX >= 700        // szext instruction requires sm_70 support or higher
u32 OVERLOAD ulowBits(i32 u, u32 bits) { u32 res; __asm("szext.clamp.u32 %0, %1, %2;" : "=r"(res) : "r"(u), "r"(bits)); return res; }
#else
u32 OVERLOAD ulowBits(i32 u, u32 bits) { return (((u32) u << (32 - bits)) >> (32 - bits)); }
#endif
u32 OVERLOAD ulowBits(u32 u, u32 bits) { return ulowBits((i32) u, bits); }
// Return unsigned low bits (number of bits must be between 1 and 63)
u64 OVERLOAD ulowBits(i64 u, u32 bits) { return (((u64) u << (64 - bits)) >> (64 - bits)); }
u64 OVERLOAD ulowBits(u64 u, u32 bits) { return ulowBits((i64) u, bits); }

// Return unsigned low bits where number of bits is known at compile time (number of bits can be 0 to 32)
u32 OVERLOAD ulowFixedBits(i32 u, const u32 bits) { if (bits == 32) return u; return u & ((1 << bits) - 1); }
u32 OVERLOAD ulowFixedBits(u32 u, const u32 bits) { return ulowFixedBits((i32) u, bits); }
// Return unsigned low bits where number of bits is known at compile time (number of bits can be 0 to 64)
u64 OVERLOAD ulowFixedBits(i64 u, const u32 bits) { return u & ((1LL << bits) - 1); }
u64 OVERLOAD ulowFixedBits(u64 u, const u32 bits) { return ulowFixedBits((i64) u, bits); }

// Return signed low bits (number of bits must be between 1 and 31)
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_sbfe)
i32 OVERLOAD lowBits(i32 u, u32 bits) { return __builtin_amdgcn_sbfe(u, 0, bits); }
#elif HAS_PTX >= 700        // szext instruction requires sm_70 support or higher
i32 OVERLOAD lowBits(i32 u, u32 bits) { i32 res; __asm("szext.clamp.s32 %0, %1, %2;" : "=r"(res) : "r"(u), "r"(bits)); return res; }
#else
i32 OVERLOAD lowBits(i32 u, u32 bits) { return ((u << (32 - bits)) >> (32 - bits)); }
#endif
i32 OVERLOAD lowBits(u32 u, u32 bits) { return lowBits((i32)u, bits); }
// Return signed low bits (number of bits must be between 1 and 63)
i64 OVERLOAD lowBits(i64 u, u32 bits) { return ((u << (64 - bits)) >> (64 - bits)); }
i64 OVERLOAD lowBits(u64 u, u32 bits) { return lowBits((i64)u, bits); }

// Return signed low bits (number of bits must be between 1 and 32)
#if HAS_PTX                 // szext does not return result we are looking for if bits = 32
i32 OVERLOAD lowBitsSafe32(i32 u, u32 bits) { return lowBits(u, bits); }
#else
i32 OVERLOAD lowBitsSafe32(i32 u, u32 bits) { return lowBits((u64)u, bits); }
#endif
i32 OVERLOAD lowBitsSafe32(u32 u, u32 bits) { return lowBitsSafe32((i32)u, bits); }

// Return signed low bits where number of bits is known at compile time (number of bits can be 0 to 32)
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_sbfe)
i32 OVERLOAD lowFixedBits(i32 u, const u32 bits) { if (bits == 32) return u; return __builtin_amdgcn_sbfe(u, 0, bits); }
#elif HAS_PTX >= 700        // szext instruction requires sm_70 support or higher
i32 OVERLOAD lowFixedBits(i32 u, const u32 bits) { if (bits == 32) return u; i32 res; __asm("szext.clamp.s32 %0, %1, %2;" : "=r"(res) : "r"(u), "r"(bits)); return res; }
#else
i32 OVERLOAD lowFixedBits(i32 u, const u32 bits) { if (bits == 32) return u; return (u << (32 - bits)) >> (32 - bits); }
#endif
i32 OVERLOAD lowFixedBits(u32 u, const u32 bits) { return lowFixedBits((i32)u, bits); }
// Return signed low bits where number of bits is known at compile time (number of bits can be 1 to 63).  The two versions are the same speed on TitanV.
i64 OVERLOAD lowFixedBits(i64 u, const u32 bits) { if (bits <= 32) return lowFixedBits((i32) u, bits); return ((u << (64 - bits)) >> (64 - bits)); }
//i64 OVERLOAD lowFixedBits(i64 u, const u32 bits) { if (bits <= 32) return lowFixedBits((i32) u, bits); return (i64) ulowFixedBits(u, bits - 1) - (u & (1LL << (bits - 1))); }
i64 OVERLOAD lowFixedBits(u64 u, const u32 bits) { return lowFixedBits((i64)u, bits); }

// Extract 32 bits from a 64-bit value (starting bit offset can be 0 to 31)
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_alignbit)
i32 xtract32(i64 x, u32 bits) { return __builtin_amdgcn_alignbit(as_int2(x).y, as_int2(x).x, bits); }
#elif HAS_PTX >= 320        // shf instruction requires sm_32 support or higher
i32 xtract32(i64 x, u32 bits) { i32 res; __asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(res) : "r"(as_uint2(x).x), "r"(as_uint2(x).y), "r"(bits)); return res; }
#else
i32 xtract32(i64 x, u32 bits) { return x >> bits; }
#endif

// Extract 32 bits from a 64-bit value (starting bit offset can be 0 to 32)
#if HAS_PTX >= 320        // shf instruction requires sm_32 support or higher
i32 xtractSafe32(i64 x, u32 bits) { i32 res; __asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(res) : "r"(as_uint2(x).x), "r"(as_uint2(x).y), "r"(bits)); return res; }
#else
i32 xtractSafe32(i64 x, u32 bits) { return x >> bits; }
#endif

u32 bitlen(bool b) { return EXP / NWORDS + b; }
bool test(u32 bits, u32 pos) { return (bits >> pos) & 1; }

#if FFT_FP64
// Rounding constant: 3 * 2^51, See https://stackoverflow.com/questions/17035464
#define RNDVAL (3.0 * (1ull << 51))

// Convert a double to long efficiently.  Double must be in RNDVAL+integer format.
i64 RNDVALdoubleToLong(double d) {
  int2 words = as_int2(d);
#if EXP / NWORDS >= 19
  // We extend the range to 52 bits instead of 51 by taking the sign from the negation of bit 51
  words.y ^= 0x00080000u;
  words.y = lowBits(words.y, 20);
#else
  // Take the sign from bit 50 (i.e. use lower 51 bits).
  words.y = lowBits(words.y, 19);
#endif
  return as_long(words);
}

#elif FFT_FP32
// Rounding constant: 3 * 2^22
#define RNDVAL (3.0f * (1 << 22))

// Convert a float to int efficiently.  Float must be in RNDVAL+integer format.
i32 RNDVALfloatToInt(float d) {
  int w = as_int(d);
//#if 0
// We extend the range to 23 bits instead of 22 by taking the sign from the negation of bit 22
//  w ^= 0x00800000u;
//  w = lowBits(words.y, 23);
//#else
//  // Take the sign from bit 21 (i.e. use lower 22 bits).
  w = lowBits(w, 22);
//#endif
  return w;
}
#endif

// map abs(carry) to floats, with 2^32 corresponding to 1.0
// So that the maximum CARRY32 abs(carry), 2^31, is mapped to 0.5 (the same as the maximum ROE)
float OVERLOAD boundCarry(i32 c) { return ldexp(fabs((float) c), -32); }

// A 64-bit carry.  The FFT types that can also run with 32-bit carries (FFT64, FFT3231) measure it against CARRY32's 2^31, for
// -carryTune (valid while abs(carry) < 2^39).  The others only have 64-bit carries: there 2^63, their overflow, maps to 0.5.
float OVERLOAD boundCarry(i64 c) {
  if (FFT_TYPE == FFT64 || FFT_TYPE == FFT3231) { return ldexp(fabs((float) (i32) (c >> 8)), -24); }
  return boundCarry((i32) (c >> 32));
}

#if STATS || ROE
void updateStats(local u32 *lds, u32 num_threads, u32 num_blocks, global uint *bufROE, u32 posROE, float roundMax) {
  assert(roundMax >= 0);
  u32 me = get_local_id(0);
  u32 u32RoundMax = as_uint(roundMax);

  // Reduce to a handful of roundMax values
  // We could use shfl_down_sync (and AMD's equivalent) instead of LDS memory once num_threads < WAVEFRONT
  // (see https://github.com/mahmoudmaftah/MaxReduction-Cuda/blob/main/code/reduction_benchmarks.cu)
  while (num_threads > 8) {
    // Write roundMax for high half of threads to local memory.  Ignore threads not participating in the reduction.
    // bar(num_threads) rather than a hand-rolled "only if it is wider than a wavefront": that test assumes a
    // wavefront advances in lock-step, which holds on AMD but not on nVidia Volta and later, and nowhere else
    // at all.  bar() decides that by what the hardware guarantees, and with G_W == 64 and a 32-lane wavefront
    // two of the three reduction steps here were running with no barrier and no fence.  num_threads is a
    // compile-time workgroup size, so every thread makes the same number of passes and reaches both calls.
    bar(num_threads);
    if (me >= num_threads / 2 && me < num_threads) lds[me - num_threads / 2] = u32RoundMax;
    bar(num_threads);
    // Low half of threads do a max
    if (me < num_threads / 2) {
      u32 highHalfMax = lds[me];
      if (u32RoundMax < highHalfMax) u32RoundMax = highHalfMax;
    }
    // Cut num threads in half, loop
    num_threads /= 2;
  }

  // The bufROE entry to update is stored in the first bufROE entry.  This value used to be passed into carryFused as an argument.
  // CUDA graphs don't allow arguments to change.  Thus, calculating posROE and storing it in bufROE works better.
  if (me < num_threads) {
    posROE = bufROE[0];
    // The buffer holds STATS_SIZE samples.  The host resets the position only when it reads the samples, and the LL and
    // CERT loops never read the carry statistics, so once the buffer is full stop recording rather than write past it.
    if (posROE < STATS_SIZE) {
      atomic_max(bufROE + posROE + 2, u32RoundMax);

      // The second bufRoe entry is a count of the number atomic_maxes performed.  When the last atomic_max is done, increment posROE and clear the counter.
      if (me == 0) {
        u32 old_value = atomic_add(bufROE + 1, 1);
        if (old_value == num_blocks - 1) {
          bufROE[0] = posROE + 1;
          bufROE[1] = 0;
        }
      }
    }
  }
}
#endif

#if 0
// Check for round off errors above a threshold (default is 0.43)
void ROUNDOFF_CHECK(double x) {
#if DEBUG
#ifndef ROUNDOFF_LIMIT
#define ROUNDOFF_LIMIT 0.43
#endif
  float error = fabs(x - rint(x));
  if (error > ROUNDOFF_LIMIT) printf("Roundoff: %g %30.2f\n", error, x);
#endif
}
#endif


/************************************************************************/
/*   Split a value + carryIn into a big-or-little word and a carryOut   */
/************************************************************************/

Word OVERLOAD carryStep(i128 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  i64 w = lowBits(i128_lo64(x), nBits);
  *outCarry = i128_shrlo64(x, nBits) + (w < 0);
  return w;
}

Word OVERLOAD carryStep(i96 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  u32 nBitsLess32 = bitlen(isBigWord) - 32;

// This code can be tricky because we must not shift i32 or u32 variables by 32.
#if EXP / NWORDS >= 33
  i32 whi = lowBits(i96_mid32(x), nBitsLess32);
  *outCarry = ((i64)i96_hi64(x) - (i64)whi) >> nBitsLess32;
  return as_ulong((uint2)(i96_lo32(x), (u32)whi));
#elif EXP / NWORDS == 32
  i32 whi = xtract32(i96_lo64(x), nBitsLess32) >> 31;
  *outCarry = ((i64)i96_hi64(x) - (i64)whi) >> nBitsLess32;
  return as_ulong((uint2)(i96_lo32(x), (u32)whi));
#elif EXP / NWORDS == 31
  i32 w = lowBitsSafe32(i96_lo32(x), nBits);
  *outCarry = as_long((int2)(xtractSafe32(i96_lo64(x), nBits), xtractSafe32(i96_hi64(x), nBits))) + (w < 0);
  return w;
//  i64 w = lowBits(i96_lo64(x), nBits);
//  *outCarry = ((i96_hi64(x) << (32 - nBits)) | ((i96_lo32(x) >> 16) >> (nBits - 16))) + (w < 0);
//  return w;
#else
  i32 w = lowBits(i96_lo32(x), nBits);
  *outCarry = as_long((int2)(xtract32(i96_lo64(x), nBits), xtract32(i96_hi64(x), nBits))) + (w < 0);
  return w;
#endif
}

Word OVERLOAD carryStep(i64 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
#if EXP / NWORDS >= 33
  i32 xhi = hi32(x);
  i32 whi = lowBits(xhi, nBits - 32);
  *outCarry = (xhi - whi) >> (nBits - 32);
  return (Word) as_long((int2)(lo32(x), whi));
#elif EXP / NWORDS == 32
  i32 xhi = hi32(x);
  i64 w = lowBits(x, nBits);
  xhi -= (i32)hi32(w);
  *outCarry = xhi >> (nBits - 32);
  return w;
#elif EXP / NWORDS == 31
  i32 w = lowBitsSafe32(lo32(x), nBits);
  *outCarry = (x - w) >> nBits;
  return w;
#else
  Word w = lowBits(lo32(x), nBits);
  *outCarry = (x - w) >> nBits;
  return w;
#endif
}

Word OVERLOAD carryStep(i64 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
#if EXP / NWORDS >= 33
  i32 xhi = hi32(x);
  i32 w = lowBits(xhi, nBits - 32);
  *outCarry = (xhi >> (nBits - 32)) + (w < 0);
  return as_long((int2)(lo32(x), w));
#elif EXP / NWORDS == 32
  i32 xhi = hi32(x);
  i64 w = lowBits(x, nBits);
  *outCarry = (xhi >> (nBits - 32)) + (w < 0);
  return w;
#elif EXP / NWORDS == 31
  i32 w = lowBitsSafe32(lo32(x), nBits);
  *outCarry = xtractSafe32(x, nBits) + (w < 0);
  return w;
#else
  i32 w = lowBits(x, nBits);
  *outCarry = xtract32(x, nBits) + (w < 0);
  return w;
#endif
}

Word OVERLOAD carryStep(i32 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  *outCarry = (x - w) >> nBits;
  return w;
}

/*****************************************************************/
/*  Same as CarryStep but returns a faster unsigned result.      */
/*  Used on first word of pair in carryFused.                    */
/* CarryFinal will later turn this into a balanced signed value. */
/*****************************************************************/

Word OVERLOAD carryStepUnsignedSloppy(i128 x, i64 *outCarry, bool isBigWord) {
  const u32 bigwordBits = EXP / NWORDS + 1;
  u32 nBits = bitlen(isBigWord);

// Return a Word using the big word size.  Big word size is a constant which allows for more optimization.
  u64 w = ulowFixedBits(i128_lo64(x), bigwordBits);
  x = i128_masklo64(x, ~((u64)1 << (bigwordBits - 1)));
  *outCarry = i128_shrlo64(x, nBits);
  return w;
}

Word OVERLOAD carryStepUnsignedSloppy(i96 x, i64 *outCarry, bool isBigWord) {
  const u32 bigwordBits = EXP / NWORDS + 1;
  u32 nBits = bitlen(isBigWord);

// Return a Word using the big word size.  Big word size is a constant which allows for more optimization.
#if EXP / NWORDS >= 32                                  // nBits is 32 or more
  i64 xhi = as_ulong((uint2)(i96_mid32(x) & ~((1 << (bigwordBits - 32)) - 1), i96_hi32(x)));
  *outCarry = xhi >> (nBits - 32);
  return as_ulong((uint2)(i96_lo32(x), ulowFixedBits(i96_mid32(x), bigwordBits - 32)));
#elif EXP / NWORDS == 31 || EXP / NWORDS >= 22          // nBits = 31 or 32, fastest version. Should also work on smaller nBits.
  *outCarry = i96_hi64(x) << (32 - nBits);
  return i96_lo32(x);                                   // ulowBits(x, bigwordBits = 32);
#else                                                   // nBits less than 32
  u32 w = ulowFixedBits(i96_lo32(x), bigwordBits);
  *outCarry = as_long((int2)(xtract32(as_long((int2)(i96_lo32(x) - w, i96_mid32(x))), nBits), xtract32(i96_hi64(x), nBits)));
  return w;
#endif
}

Word OVERLOAD carryStepUnsignedSloppy(i64 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  *outCarry = x >> nBits;
  return ulowBits(x, nBits);
}

Word OVERLOAD carryStepUnsignedSloppy(i64 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  *outCarry = xtract32(x, nBits);
  return ulowBits(x, nBits);
}

Word OVERLOAD carryStepUnsignedSloppy(i32 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  *outCarry = x >> nBits;
  return ulowBits(x, nBits);
}

/**********************************************************************/
/*  Same as CarryStep but may return a faster big word signed result. */
/*  Used on second word of pair in carryFused when not near max BPW.  */
/*  Also used on first word in carryFinal when not near max BPW.      */
/**********************************************************************/

// We only allow sloppy results when not near the maximum bits-per-word.  For now, this is defined as 1.1 bits below maxbpw.
// No studies have been done on reducing this 1,1 value since this is a rather minor optimization.  Since the preprocessor can't
// handle floats, the MAXBPW value passed in is 100 * maxbpw.
#define SLOPPY_MAXBPW   (MAXBPW - 110)
#define ACTUAL_BPW      (EXP / (NWORDS / 100))

Word OVERLOAD carryStepSignedSloppy(i128 x, i64 *outCarry, bool isBigWord) {
#if ACTUAL_BPW > SLOPPY_MAXBPW
  return carryStep(x, outCarry, isBigWord);
#else

//GW:  Need to compare to simple carryStep
  
// Return a Word using the big word size.  Big word size is a constant which allows for more optimization.
  const u32 bigwordBits = EXP / NWORDS + 1;
  u32 nBits = bitlen(isBigWord);
  u64 xlo = i128_lo64(x);
  u64 xlo_topbit = xlo & ((u64)1 << (bigwordBits - 1));
  i64 w = ulowFixedBits(xlo, bigwordBits - 1) - xlo_topbit;
  *outCarry = i128_shrlo64(add(x, xlo_topbit), nBits);
  return w;
#endif
}

Word OVERLOAD carryStepSignedSloppy(i96 x, i64 *outCarry, bool isBigWord) {
#if ACTUAL_BPW > SLOPPY_MAXBPW
  return carryStep(x, outCarry, isBigWord);
#else

// Return a Word using the big word size.  Big word size is a constant which allows for more optimization.
  const u32 bigwordBits = EXP / NWORDS + 1;
  u32 nBits = bitlen(isBigWord);
#if EXP / NWORDS >= 32                                  // nBits is 32 or more
  return carryStep(x, outCarry, isBigWord);             // Should be just as fast as code below
//  u32 xmid_topbit = i96_mid32(x) & (1 << (bigwordBits - 32 - 1));
//  i32 whi = ulowFixedBits(i96_mid32(x), bigwordBits - 32 - 1) - xmid_topbit;
//  i64 xhi = i96_hi64(x) + xmid_topbit;
//  *outCarry = xhi >> (nBits - 32);
//  return as_long((int2)(i96_lo32(x), whi));
#elif EXP / NWORDS == 31 || (SLOPPY_MAXBPW >= 3200 && EXP / NWORDS >= 22) // nBits = 31 or 32, bigwordBits = 32 (or allowed to create 32-bit word for better performance)
  i32 w = i96_lo32(x);                                  // lowBits(x, bigwordBits = 32);
  *outCarry = (i96_hi64(x) + (w < 0)) << (32 - nBits);
  return w;
#else                                                   // nBits less than 32
  return carryStep(x, outCarry, isBigWord);             // Should be faster than code below
//  i32 w = lowFixedBits(i96_lo32(x), bigwordBits);
//  *outCarry = (as_long((int2)(xtract32(i96_lo64(x), bigwordBits), xtract32(i96_hi64(x), bigwordBits))) + (w < 0)) << (bigwordBits - nBits);
//  return w;
#endif
#endif
}

Word OVERLOAD carryStepSignedSloppy(i64 x, i64 *outCarry, bool isBigWord) {
#if ACTUAL_BPW > SLOPPY_MAXBPW
  return carryStep(x, outCarry, isBigWord);
#else

  // We're unlikely to find code that is better than carryStep
  return carryStep(x, outCarry, isBigWord);
#endif
}

Word OVERLOAD carryStepSignedSloppy(i64 x, i32 *outCarry, bool isBigWord) {
#if ACTUAL_BPW > SLOPPY_MAXBPW
  return carryStep(x, outCarry, isBigWord);
#else

//GW: I need to look at PTX code generated by the code below vs. carryStep

// Return a Word using the big word size.  Big word size is a constant which allows for more optimization.
  const u32 bigwordBits = EXP / NWORDS + 1;
  u32 nBits = bitlen(isBigWord);
#if EXP / NWORDS >= 32                                  // nBits is 32 or more
  u64 x_topbit = x & ((u64)1 << (bigwordBits - 1));
  i64 w = ulowFixedBits(x, bigwordBits - 1) - x_topbit;
  i32 xhi = (i32)hi32(x) + (i32)hi32(x_topbit);
  *outCarry = xhi >> (nBits - 32);
  return w;
// nBits = 31 or 32, bigwordBits = 32 (or allowed to create 32-bit word for better performance)
#elif EXP / NWORDS == 31 || (EXP / NWORDS >= 23 && SLOPPY_MAXBPW >= 3200)        
  i32 w = x;                                            // lowBits(x, bigwordBits = 32);
  *outCarry = ((i32)hi32(x) + (w < 0)) << (32 - nBits);
  return w;
#else                                                   // nBits less than 32         //GWBUG - is there a faster version?  Is this faster than plain old carryStep? No
//  u32 x_topbit = (u32) x & (1 << (bigwordBits - 1));
//  i32 w = ulowFixedBits((u32) x, bigwordBits - 1) - x_topbit;
//  *outCarry = (i64)(x + x_topbit) >> nBits;
//  return w;
  return carryStep(x, outCarry, isBigWord);
#endif
#endif
}

Word OVERLOAD carryStepSignedSloppy(i32 x, i32 *outCarry, bool isBigWord) {
  return carryStep(x, outCarry, isBigWord);
}



// Carry propagation from word and carry.  Used by carryB.cl.
Word2 carryWord(Word2 a, CarryABM* carry, bool b1, bool b2) {
  a.x = carryStep(a.x + *carry, carry, b1);
  a.y = carryStep(a.y + *carry, carry, b2);
  return a;
}

/**************************************************************************/
/*     Do this last, it depends on weightAndCarryOne defined above        */
/**************************************************************************/

/* Support both 32-bit and 64-bit carries */

/* iCARRY is set to all possible data types that CFCarry in carryfused could be.  Carryinc.cl is then #included for each data type. */
/* In essence, iCARRY means "a carry from carryShuttle or a carry from the first word of a pair". */

#if WordSize <= 4
// A 32-bit carry means FFT64's weightAndCarryOne returns RNDVAL + value un-stripped (bit 51 flipped at 19 bpw), and
// carryStep(i64, i32*) reads the carry from bits [nBits, nBits+32).  That window must stay within bits [0,52), so
// nBits <= 20; a big word has nBits = EXP / NWORDS + 1.  The host is supposed to select CARRY64 before this
// point (FFTShape::needsLargeCarry); fail loudly rather than compute wrong carries if it ever does not.
//
// The bound is the host's own, and the host does not make it depend on the FFT type: needsLargeCarry()
// returns true for every type at EXP / NWORDS >= 20, so CARRY_AUTO can never reach here.  Only an explicit
// 32-bit carry in the FFT spec ("-fft 3:256:2:256:212:0") can, and the NTT types need the check as much as
// FP64 does -- a GF61 carry at 25 bpw does not fit in i32 either, and nothing else would report it.
#if !(CARRY64 || (MUL3 && MUL3_CARRY64)) && EXP / NWORDS >= 20
#error "CARRY32 requires EXP / NWORDS < 20; this exponent needs CARRY64 (-carry long)"
#endif
#define iCARRY i32
#include "carryinc.cl"
#undef iCARRY
#endif

#if FFT_TYPE != FFT32 && FFT_TYPE != FFT31
#define iCARRY i64
#include "carryinc.cl"
#undef iCARRY
#endif
