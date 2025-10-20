// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"

#if CARRY64
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
#else
u32 OVERLOAD ulowBits(i32 u, u32 bits) { return (((u32) u << (32 - bits)) >> (32 - bits)); }
#endif
u32 OVERLOAD ulowBits(u32 u, u32 bits) { return ulowBits((i32) u, bits); }
// Return unsigned low bits (number of bits must be between 1 and 63)
u64 OVERLOAD ulowBits(i64 u, u32 bits) { return (((u64) u << (64 - bits)) >> (64 - bits)); }
u64 OVERLOAD ulowBits(u64 u, u32 bits) { return ulowBits((i64) u, bits); }

// Return unsigned low bits where number of bits is known at compile time (number of bits can be 0 to 32)
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_ubfe)
u32 OVERLOAD ulowFixedBits(i32 u, const u32 bits) { if (bits == 32) return u; return __builtin_amdgcn_ubfe(u, 0, bits); }
#else
u32 OVERLOAD ulowFixedBits(i32 u, const u32 bits) { if (bits == 32) return u; return u & ((1 << bits) - 1); }
#endif
u32 OVERLOAD ulowFixedBits(u32 u, const u32 bits) { return ulowFixedBits((i32) u, bits); }
// Return unsigned low bits where number of bits is known at compile time (number of bits can be 0 to 64)
u64 OVERLOAD ulowFixedBits(i64 u, const u32 bits) { return u & ((1LL << bits) - 1); }
u64 OVERLOAD ulowFixedBits(u64 u, const u32 bits) { return ulowFixedBits((i64) u, bits); }

// Return signed low bits (number of bits must be between 1 and 31)
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_sbfe)
i32 OVERLOAD lowBits(i32 u, u32 bits) { return __builtin_amdgcn_sbfe(u, 0, bits); }
#else
i32 OVERLOAD lowBits(i32 u, u32 bits) { return ((u << (32 - bits)) >> (32 - bits)); }
#endif
i32 OVERLOAD lowBits(u32 u, u32 bits) { return lowBits((i32) u, bits); }
// Return signed low bits (number of bits must be between 1 and 63)
i64 OVERLOAD lowBits(i64 u, u32 bits) { return ((u << (64 - bits)) >> (64 - bits)); }
i64 OVERLOAD lowBits(u64 u, u32 bits) { return lowBits((i64) u, bits); }

// Return signed low bits where number of bits is known at compile time (number of bits can be 0 to 32)
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_sbfe)
i32 OVERLOAD lowFixedBits(i32 u, const u32 bits) { if (bits == 32) return u; return __builtin_amdgcn_sbfe(u, 0, bits); }
#else
// This version should generate 2 shifts
i32 OVERLOAD lowFixedBits(i32 u, const u32 bits) { if (bits == 32) return u; return (u << (32 - bits)) >> (32 - bits); }
// This version should generate 2 ANDs and one subtract
//i32 OVERLOAD lowFixedBits(i32 u, const u32 bits) { if (bits == 32) return u; if (bits == 1) return -(u & 1); return ulowFixedBits(u, bits - 1) - (u & (1 << bits)); }
i32 OVERLOAD lowFixedBits(u32 u, const u32 bits) { return lowFixedBits((i32) u, bits); }
#endif
// Return signed low bits where number of bits is known at compile time (number of bits can be 1 to 63).  The two versions are the same speed on TitanV.
i64 OVERLOAD lowFixedBits(i64 u, const u32 bits) { if (bits <= 32) return lowFixedBits((i32) u, bits); return ((u << (64 - bits)) >> (64 - bits)); }
//i64 OVERLOAD lowFixedBits(i64 u, const u32 bits) { if (bits <= 32) return lowFixedBits((i32) u, bits); return (i64) ulowFixedBits(u, bits - 1) - (u & (1LL << (bits - 1))); }
i64 OVERLOAD lowFixedBits(u64 u, const u32 bits) { return lowFixedBits((i64) u, bits); }

// Extract 32 bits from a 64-bit value
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_alignbit)
i32 xtract32(i64 x, u32 bits) { return __builtin_amdgcn_alignbit(as_int2(x).y, as_int2(x).x, bits); }
#else
i32 xtract32(i64 x, u32 bits) { return x >> bits; }
#endif

u32 bitlen(bool b) { return EXP / NWORDS + b; }
bool test(u32 bits, u32 pos) { return (bits >> pos) & 1; }

#if FFT_FP64
// Rounding constant: 3 * 2^51, See https://stackoverflow.com/questions/17035464
#define RNDVAL (3.0 * (1l << 51))

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
float OVERLOAD boundCarry(i64 c) { return ldexp(fabs((float) (i32) (c >> 8)), -24); }

#if STATS || ROE
void updateStats(global uint *bufROE, u32 posROE, float roundMax) {
  assert(roundMax >= 0);
  // work_group_reduce_max() allocates an additional 256Bytes LDS for a 64lane workgroup, so avoid it.
  // u32 groupRound = work_group_reduce_max(as_uint(roundMax));
  // if (get_local_id(0) == 0) { atomic_max(bufROE + posROE, groupRound); }

  // Do the reduction directly over global mem.
  atomic_max(bufROE + posROE, as_uint(roundMax));
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


/***************************************************************************/
/*  From the FFT data, construct a value to normalize and carry propagate  */
/***************************************************************************/

#if FFT_TYPE == FFT64

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i64 weightAndCarryOne(T u, T invWeight, i64 inCarry, float* maxROE, int sloppy_result_is_acceptable) {

#if !MUL3

  // Convert carry into RNDVAL + carry.
  int2 tmp = as_int2(inCarry); tmp.y += as_int2(RNDVAL).y;
  double RNDVALCarry = as_double(tmp);

  // Apply inverse weight and RNDVAL+carry
  double d = fma(u, invWeight, RNDVALCarry);

  // Optionally calculate roundoff error
  float roundoff = fabs((float) fma(u, invWeight, RNDVALCarry - d));
  *maxROE = max(*maxROE, roundoff);

  // Convert to long (for CARRY32 case we don't need to strip off the RNDVAL bits)
  if (sloppy_result_is_acceptable) return as_long(d);
  else return RNDVALdoubleToLong(d);

#else  // We cannot add in the carry until after the mul by 3

  // Apply inverse weight and RNDVAL
  double d = fma(u, invWeight, RNDVAL);

  // Optionally calculate roundoff error
  float roundoff = fabs((float) fma(u, -invWeight, d - RNDVAL));
  *maxROE = max(*maxROE, roundoff);

  // Convert to long, mul by 3, and add carry
  return RNDVALdoubleToLong(d) * 3 + inCarry;

#endif
}

/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#elif FFT_TYPE == FFT32

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer.  Handle MUL3.
i32 weightAndCarryOne(F u, F invWeight, i32 inCarry, float* maxROE, int sloppy_result_is_acceptable) {

#if !MUL3

  // Convert carry into RNDVAL + carry.
  float RNDVALCarry = as_float(as_int(RNDVAL) + inCarry);                       // GWBUG - just the float arithmetic?  s.b. fast

  // Apply inverse weight and RNDVAL+carry
  float d = fma(u, invWeight, RNDVALCarry);

  // Optionally calculate roundoff error
  float roundoff = fabs(fma(u, invWeight, RNDVALCarry - d));
  *maxROE = max(*maxROE, roundoff);

  // Convert to int
  return RNDVALfloatToInt(d);

#else  // We cannot add in the carry until after the mul by 3

  // Apply inverse weight and RNDVAL
  float d = fma(u, invWeight, RNDVAL);

  // Optionally calculate roundoff error
  float roundoff = fabs(fma(u, -invWeight, d - RNDVAL));
  *maxROE = max(*maxROE, roundoff);

  // Convert to int, mul by 3, and add carry
  return RNDVALfloatToInt(d) * 3 + inCarry;

#endif
}

/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT31

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i64 weightAndCarryOne(Z61 u, u32 invWeight, i32 inCarry, u32* maxROE) {

  // Apply inverse weight
  u = shr(u, invWeight);

  // Convert input to balanced representation
  i32 value = get_balanced_Z31(u);

  // Optionally calculate roundoff error as proximity to M31/2.
  u32 roundoff = (u32) abs(value);
  *maxROE = max(*maxROE, roundoff);

  // Mul by 3 and add carry
#if MUL3
  return (i64)value * 3 + inCarry;
#endif
  return value + inCarry;
}

/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT61

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i64 weightAndCarryOne(Z61 u, u32 invWeight, i64 inCarry, u32* maxROE) {

  // Apply inverse weight
  u = shr(u, invWeight);

  // Convert input to balanced representation
  i64 value = get_balanced_Z61(u);

  // Optionally calculate roundoff error as proximity to M61/2.  28 bits of accuracy should be sufficient.
  u32 roundoff = (u32) abs((i32) (value >> 32));
  *maxROE = max(*maxROE, roundoff);

  // Mul by 3 and add carry
#if MUL3
  value *= 3;
#endif
  return value + inCarry;
}

/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP64 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT6431

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i96 weightAndCarryOne(T u, Z31 u31, T invWeight, u32 m31_invWeight, i64 inCarry, float* maxROE) {

  // Apply inverse weight and get the Z31 data
  u31 = shr(u31, m31_invWeight);
  u32 n31 = get_Z31(u31);

  // The final result must be n31 mod M31.  Use FP64 data to calculate this value.
  u = fma(u, invWeight, - (double) n31);                               // This should be close to a multiple of M31
  double uInt = fma(u, 4.656612875245796924105750827168e-10, RNDVAL);  // Divide by M31 and round to int
  i64 n64 = RNDVALdoubleToLong(uInt);

  // Optionally calculate roundoff error
  float roundoff = (float) fabs(fma(u, 4.656612875245796924105750827168e-10, RNDVAL - uInt));
  *maxROE = max(*maxROE, roundoff);

  // Compute the value using i96 math
  i64 vhi = n64 >> 33;
  u64 vlo = ((u64)n64 << 31) | n31;
  i96 value = make_i96(vhi, vlo);                   // (n64 << 31) + n31
  value = sub(value, make_i96(n64));                // n64 * M31 + n31

  // Mul by 3 and add carry
#if MUL3
  value = add(value, add(value, value));
#endif
  return add(value, make_i96(inCarry));
}

/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3231

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i64 weightAndCarryOne(float uF2, Z31 u31, float F2_invWeight, u32 m31_invWeight, i32 inCarry, float* maxROE) {

  // Apply inverse weight and get the Z31 data
  u31 = shr(u31, m31_invWeight);
  u32 n31 = get_Z31(u31);

  // The final result must be n31 mod M31.  Use FP32 data to calculate this value.
  uF2 = fma(uF2, F2_invWeight, - (float) n31);                           // This should be close to a multiple of M31
  float uF2int = fma(uF2, 4.656612875245796924105750827168e-10f, RNDVAL);   // Divide by M31 and round to int
  i32 nF2 = RNDVALfloatToInt(uF2int);

  i64 v = (((i64) nF2 << 31) | n31) - nF2;         // nF2 * M31 + n31

  // Optionally calculate roundoff error
  float roundoff = fabs(fma(uF2, 4.656612875245796924105750827168e-10f, RNDVAL - uF2int));
  *maxROE = max(*maxROE, roundoff);

  // Mul by 3 and add carry
#if MUL3
  v = v * 3;
#endif
  return v + inCarry;
}

/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M61^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3261

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i96 weightAndCarryOne(float uF2, Z61 u61, float F2_invWeight, u32 m61_invWeight, i64 inCarry, float* maxROE) {

  // Apply inverse weight and get the Z61 data
  u61 = shr(u61, m61_invWeight);
  u64 n61 = get_Z61(u61);

  // The final result must be n61 mod M61.  Use FP32 data to calculate this value.
  float n61f = (float)((u32)(n61 >> 32)) * -4294967296.0f;                    // Conversion from u64 to float might be slow, this might be faster
  uF2 = fma(uF2, F2_invWeight, n61f);                                         // This should be close to a multiple of M61
  float uF2int = fma(uF2, 4.3368086899420177360298112034798e-19f, RNDVAL);    // Divide by M61 and round to int
  i32 nF2 = RNDVALfloatToInt(uF2int);

  // Optionally calculate roundoff error
  float roundoff = fabs(fma(uF2, 4.3368086899420177360298112034798e-19f, RNDVAL - uF2int));
  *maxROE = max(*maxROE, roundoff);

  // Compute the value using i96 math
  i32 vhi = nF2 >> 3;
  u64 vlo = ((u64)nF2 << 61) | n61;
  i96 value = make_i96(vhi, vlo);                // (nF2 << 61) + n61
  value = sub(value, make_i96(nF2));             // nF2 * M61 + n61

  // Mul by 3 and add carry
#if MUL3
  value = add(value, add(value, value));
#endif
  return add(value, make_i96(inCarry));
}

/**************************************************************************/
/*    Similar to above, but for an NTT based on GF(M31^2)*GF(M61^2)       */
/**************************************************************************/

#elif FFT_TYPE == FFT3161

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i96 weightAndCarryOne(Z31 u31, Z61 u61, u32 m31_invWeight, u32 m61_invWeight, i64 inCarry, u32* maxROE) {

  // Apply inverse weights
  u31 = shr(u31, m31_invWeight);
  u61 = shr(u61, m61_invWeight);

  // Use chinese remainder theorem to create a 92-bit result.  Loosely copied from Yves Gallot's mersenne2 program.
  u32 n31 = get_Z31(u31);
  u61 = subq(u61, make_Z61(n31), 2);             // u61 - u31
  u61 = add(u61, shl(u61, 31));                  // u61 + (u61 << 31)

  // The resulting value will be get_Z61(u61) * M31 + n31 and if larger than ~M31*M61/2 return a negative value by subtracting M31 * M61.
  // We can save a little work by determining if the result will be large using just u61 and returning (get_Z61(u61) - M61) * M31 + n31.
  // This simplifies to get_balanced_Z61(u61) * M31 + n31.
  i64 n61 = get_balanced_Z61(u61);

  // Optionally calculate roundoff error as proximity to M61/2.  28 bits of accuracy should be sufficient.
  u32 roundoff = (u32) abs((i32)(n61 >> 32));
  *maxROE = max(*maxROE, roundoff);

  // Compute the value using i96 math
  i64 vhi = n61 >> 33;
  u64 vlo = ((u64)n61 << 31) | n31;
  i96 value = make_i96(vhi, vlo);                // (n61 << 31) + n31
  value = sub(value, make_i96(n61));             // n61 * M31 + n31

  // Mul by 3 and add carry
#if MUL3
  value = add(value, add(value, value));
#endif
  return add(value, make_i96(inCarry));
}

/******************************************************************************/
/*  Similar to above, but for a hybrid FFT based on FP32*GF(M31^2)*GF(M61^2)  */
/******************************************************************************/

#elif FFT_TYPE == FFT323161

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i128 weightAndCarryOne(float uF2, Z31 u31, Z61 u61, float F2_invWeight, u32 m31_invWeight, u32 m61_invWeight, i64 inCarry, float* maxROE) {

  // Apply inverse weights
  u31 = shr(u31, m31_invWeight);
  u61 = shr(u61, m61_invWeight);

  // Use chinese remainder theorem to create a 92-bit result.  Loosely copied from Yves Gallot's mersenne2 program.
  u32 n31 = get_Z31(u31);
  u61 = subq(u61, make_Z61(n31), 2);                 // u61 - u31
  u61 = add(u61, shl(u61, 31));                      // u61 + (u61 << 31)
  u64 n61 = get_Z61(u61);
  i128 n3161 = (((i128) n61 << 31) | n31) - n61;     // n61 * M31 + n31

  // The final result must be n3161 mod M31*M61.  Use FP32 data to calculate this value.
  float n3161f = (float)((u32)(n61 >> 32)) * -9223372036854775808.0f;        // Conversion from i128 to float might be slow, this might be faster
  uF2 = fma(uF2, F2_invWeight, n3161f);                                      // This should be close to a multiple of M31*M61
  float uF2int = fma(uF2, 2.0194839183061857038255724444152e-28f, RNDVAL);   // Divide by M31*M61 and round to int
  i32 nF2 = RNDVALfloatToInt(uF2int);

  i64 nF2m31 = ((i64) nF2 << 31) - nF2;                  // nF2 * M31
  i128 v = ((i128) nF2m31 << 61) - nF2m31 + n3161;       // nF2m31 * M61 + n3161

  // Optionally calculate roundoff error
  float roundoff = fabs(fma(uF2, 2.0194839183061857038255724444152e-28f, RNDVAL - uF2int));
  *maxROE = max(*maxROE, roundoff);

  // Mul by 3 and add carry
#if MUL3
  v = v * 3;
#endif
  v = v + inCarry;
  return v;
}

#else
error - missing weightAndCarryOne implementation
#endif


/************************************************************************/
/*   Split a value + carryIn into a big-or-little word and a carryOut   */
/************************************************************************/

Word OVERLOAD carryStep(i128 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  i64 w = lowBits((i64)x, nBits);
  *outCarry = (u64)(x >> nBits) + (w < 0);
  return w;
}

Word OVERLOAD carryStep(i96 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);

// This code can be tricky because we must not shift i32 or u32 variables by 32.
#if EXP / NWORDS >= 33
  i64 xhi = i96_hi64(x);
  i64 w = lowBits(xhi, nBits - 32);
  *outCarry = (xhi - w) >> (nBits - 32);
  return (w << 32) | i96_lo32(x);
#elif EXP / NWORDS == 32
  i64 xhi = i96_hi64(x);
  i64 w = lowBits(i96_lo64(x), nBits);
  *outCarry = (xhi - (w >> 32)) >> (nBits - 32);
  return w;
#elif EXP / NWORDS == 31
  i64 w = lowBits(i96_lo64(x), nBits);
  *outCarry = ((i96_hi64(x) << (32 - nBits)) | ((i96_lo32(x) >> 16) >> (nBits - 16))) + (w < 0);
  return w;
#else
  i32 w = lowBits(i96_lo32(x), nBits);
  *outCarry = ((i96_hi64(x) << (32 - nBits)) | (i96_lo32(x) >> nBits)) + (w < 0);
  return w;
#endif
}

Word OVERLOAD carryStep(i64 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
#if EXP / NWORDS >= 33
  i32 xhi = (x >> 32);
  i32 whi = lowBits(xhi, nBits - 32);
  *outCarry = (xhi - whi) >> (nBits - 32);
  return (Word) (((u64) whi << 32) | (u32)(x));
#elif EXP / NWORDS == 32
  i32 xhi = (x >> 32);
  i64 w = lowBits(x, nBits);
  xhi -= w >> 32;
  *outCarry = xhi >> (nBits - 32);
  return w;
#elif EXP / NWORDS == 31
  i64 w = lowBits(x, nBits);
  *outCarry = (x - w) >> nBits;
  return w;
#else
  Word w = lowBits((i32) x, nBits);
  *outCarry = (x - w) >> nBits;
  return w;
#endif
}

Word OVERLOAD carryStep(i64 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
#if EXP / NWORDS >= 33
  i32 xhi = (x >> 32);
  i32 w = lowBits(xhi, nBits - 32);
  *outCarry = (xhi >> (nBits - 32)) + (w < 0);
  return (Word) (((u64) w << 32) | (u32)(x));
#elif EXP / NWORDS == 32
  i32 xhi = (x >> 32);
  i64 w = lowBits(x, nBits);
  *outCarry = (i32) (xhi >> (nBits - 32)) + (w < 0);
  return w;
#elif EXP / NWORDS == 31
  i32 w = lowBits(x, nBits);
  *outCarry = (i32) (x >> nBits) + (w < 0);
  return w;
#else
  Word w = lowBits(x, nBits);
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
  u64 w = ulowFixedBits((u64)x, bigwordBits);
  const i128 topbitmask = ~((i128)1 << (bigwordBits - 1));
//GW Can we use unsigned shift (knowing the sign won't be lost due to truncating the result) -- this is really a 64-bit extract (or two 32-bit extrats) -- use elsewhere?)
  *outCarry = (x & topbitmask) >> nBits;
//GW use this style else where, check for more fixed low bits
  return w;
}

Word OVERLOAD carryStepUnsignedSloppy(i96 x, i64 *outCarry, bool isBigWord) {
  const u32 bigwordBits = EXP / NWORDS + 1;
  u32 nBits = bitlen(isBigWord);

// Return a Word using the big word size.  Big word size is a constant which allows for more optimization.
#if EXP / NWORDS >= 32                                  // nBits is 32 or more
  i64 xhi = i96_hi64(x) & ~((1ULL << (bigwordBits - 32)) - 1);
  *outCarry = xhi >> (nBits - 32);
  return ulowFixedBits(i96_lo64(x), bigwordBits);
#elif EXP / NWORDS == 31                                // nBits = 31 or 32
  *outCarry = i96_hi64(x) << (32 - nBits);
  return i96_lo32(x);                                   // ulowBits(x, bigwordBits = 32);
#else                                                   // nBits less than 32
  u32 w = ulowFixedBits(i96_lo32(x), bigwordBits);
  *outCarry = (i96_hi64(x) << (32 - nBits)) | ((i96_lo32(x) - w) >> nBits);
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
#define SLOPPY_MAXBPW   (MAXBPW - 1100)
#define ACTUAL_BPW      (EXP / (NWORDS / 100))

Word OVERLOAD carryStepSignedSloppy(i128 x, i64 *outCarry, bool isBigWord) {
#if ACTUAL_BPW > SLOPPY_MAXBPW
  return carryStep(x, outCarry, isBigWord);
#else

// Return a Word using the big word size.  Big word size is a constant which allows for more optimization.
  const u32 bigwordBits = EXP / NWORDS + 1;
  u32 nBits = bitlen(isBigWord);
  u64 xlo = (u64)x;
  u64 xlo_topbit = xlo & (1ULL << (bigwordBits - 1));
  i64 w = ulowFixedBits(xlo, bigwordBits - 1) - xlo_topbit;
  *outCarry = (x + xlo_topbit) >> nBits;
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
  u64 xlo = i96_lo64(x);
  u64 xlo_topbit = xlo & (1ULL << (bigwordBits - 1));
  i64 w = ulowFixedBits(xlo, bigwordBits - 1) - xlo_topbit;
  i64 xhi = i96_hi64(x) + (xlo_topbit >> 32);
  *outCarry = xhi >> (nBits - 32);
  return w;
#elif EXP / NWORDS == 31 || SLOPPY_MAXBPW >= 3200       // nBits = 31 or 32, bigwordBits = 32 (or allowed to create 32-bit word for better performance)
  i32 w = i96_lo32(x);                                  // lowBits(x, bigwordBits = 32);
  *outCarry = (i96_hi64(x) + (w < 0)) << (32 - nBits);
  return w;
#else                                                   // nBits less than 32       //GWBUG - is there a faster version?  Is this faster than plain old carryStep?
  i32 w = lowFixedBits(i96_lo32(x), bigwordBits);
  *outCarry = (((i96_hi64(x) << (32 - bigwordBits)) | (i96_lo32(x) >> bigwordBits)) + (w < 0)) << (bigwordBits - nBits);
  return w;
#endif
#endif
}

Word OVERLOAD carryStepSignedSloppy(i64 x, i64 *outCarry, bool isBigWord) {
#if ACTUAL_BPW > SLOPPY_MAXBPW
  return carryStep(x, outCarry, isBigWord);
#else

// GWBUG - not timed to see if it is faster.  Highly likely to be slower.
//  const u32 bigwordBits = EXP / NWORDS + 1;
//  u32 nBits = bitlen(isBigWord);
//  u32 w = lowBits(x, bigwordBits);
//  *outCarry = (((x << (32 - bigwordBits)) | ((u32) x >> bigwordBits)) + (w < 0)) << (bigwordBits - nBits);
//  return w;
  return carryStep(x, outCarry, isBigWord);
#endif
}

Word OVERLOAD carryStepSignedSloppy(i64 x, i32 *outCarry, bool isBigWord) {
#if ACTUAL_BPW > SLOPPY_MAXBPW
  return carryStep(x, outCarry, isBigWord);
#else

// Return a Word using the big word size.  Big word size is a constant which allows for more optimization.
  const u32 bigwordBits = EXP / NWORDS + 1;
  u32 nBits = bitlen(isBigWord);
#if EXP / NWORDS >= 32                                  // nBits is 32 or more
  u64 x_topbit = x & (1ULL << (bigwordBits - 1));
  i64 w = ulowFixedBits(x, bigwordBits - 1) - x_topbit;
  i32 xhi = (i32)(x >> 32) + (i32)(x_topbit >> 32);
  *outCarry = xhi >> (nBits - 32);
  return w;
// nBits = 31 or 32, bigwordBits = 32 (or allowed to create 32-bit word for better performance).  For reasons I don't fully understand the sloppy
// case fails if BPW is too low.  Probably something to do with a small BPW with sloppy 32-bit values would require CARRY_LONG to work properly.
// Not a major concern as end users should avoid small BPW as there is probably a more efficient NTT that could be used.
#elif EXP / NWORDS == 31 || (EXP / NWORDS >= 23 && SLOPPY_MAXBPW >= 3200)        
  i32 w = x;                                            // lowBits(x, bigwordBits = 32);
  *outCarry = ((i32)(x >> 32) + (w < 0)) << (32 - nBits);
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

#if WordSize <= 4
#define iCARRY i32
#include "carryinc.cl"
#undef iCARRY
#endif

#if FFT_TYPE != FFT32 && FFT_TYPE != FFT31
#define iCARRY i64
#include "carryinc.cl"
#undef iCARRY
#endif
