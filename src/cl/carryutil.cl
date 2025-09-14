// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"

#if CARRY64
typedef i64 CFcarry;
#else
typedef i32 CFcarry;
#endif

// The carry for the non-fused CarryA, CarryB, CarryM kernels.
// Simply use large carry always as the split kernels are slow anyway (and seldomly used normally).
typedef i64 CarryABM;

#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_sbfe)
i32 lowBits(i32 u, u32 bits) { return __builtin_amdgcn_sbfe(u, 0, bits); }
#else
i32 lowBits(i32 u, u32 bits) { return ((u << (32 - bits)) >> (32 - bits)); }
#endif

#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_ubfe)
u32 ulowBits(i32 u, u32 bits) { return __builtin_amdgcn_ubfe(u, 0, bits); }
#else
u32 ulowBits(i32 u, u32 bits) { return (((u32) u << (32 - bits)) >> (32 - bits)); }
#endif

#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_alignbit)
i32 xtract32(i64 x, u32 bits) { return __builtin_amdgcn_alignbit(as_int2(x).y, as_int2(x).x, bits); }
#else
i32 xtract32(i64 x, u32 bits) { return x >> bits; }
#endif

u32 bitlen(bool b) { return EXP / NWORDS + b; }
bool test(u32 bits, u32 pos) { return (bits >> pos) & 1; }

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

Word OVERLOAD carryStep(i96 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);

//GWBUG - is this ever faster?
//i128 x128 = ((i128) (i64) i96_hi64(x) << 32) | i96_lo32(x);
//i64 w = ((i64) x128 << (64 - nBits)) >> (64 - nBits);
//x128 -= w;
//*outCarry = x128 >> nBits;
//return w;

// This code is tricky because me must not shift i32 or u32 variables by 32.
#if EXP / NWORDS >= 33                          //GWBUG Would the EXP / NWORDS == 32 code be just as fast?
  i64 xhi = i96_hi64(x);
  i64 w = lowBits(xhi, nBits - 32);
  xhi -= w;
  *outCarry = xhi >> (nBits - 32);
  return (w << 32) | i96_lo32(x);
#elif EXP / NWORDS == 32
  i64 xhi = i96_hi64(x);
  i64 w = ((i64) i96_lo64(x) << (64 - nBits)) >> (64 - nBits);
//  xhi -= w >> 32;
//  *outCarry = xhi >> (nBits - 32);            //GWBUG -  Would adding (w < 0) be faster than subtracting w>>32 from xhi?
  *outCarry = (xhi >> (nBits - 32)) + (w < 0);
  return w;
#elif EXP / NWORDS == 31
  i64 w = ((i64) i96_lo64(x) << (64 - nBits)) >> (64 - nBits);
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
  i32 w = lowBits(xhi, nBits - 32);
  xhi -= w;
  *outCarry = xhi >> (nBits - 32);
  return (Word) (((u64) w << 32) | (u32)(x));
#elif EXP / NWORDS == 32
  i32 xhi = (x >> 32);
  i64 w = (x << (64 - nBits)) >> (64 - nBits);           // lowBits(x, nBits);
  xhi -= w >> 32;
  *outCarry = xhi >> (nBits - 32);
  return w;
#elif EXP / NWORDS == 31
  i64 w = (x << (64 - nBits)) >> (64 - nBits);           // lowBits(x, nBits);
  x -= w;
  *outCarry = x >> nBits;
  return w;
#else
  Word w = lowBits((i32) x, nBits);
  x -= w;
  *outCarry = x >> nBits;
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
  i64 w = (x << (64 - nBits)) >> (64 - nBits);           // lowBits(x, nBits);
  *outCarry = (i32) (xhi >> (nBits - 32)) + (w < 0);
  return w;
#elif EXP / NWORDS == 31
  i32 w = (x << (64 - nBits)) >> (64 - nBits);           // lowBits(x, nBits);
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

Word OVERLOAD carryStepSloppy(i96 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);

// GWBUG  Is this faster (or same speed) ????  This code doesn't work on TitanV???
//i128 x128 = ((i128) xhi << 32) | i96_lo32(x);
//*outCarry = x128 >> nBits;
//return ((u64) x128 << (64 - nBits)) >> (64 - nBits);

// This code is tricky because me must not shift i32 or u32 variables by 32.
#if EXP / NWORDS >= 33                                  // nBits is 33 or more
  i64 xhi = i96_hi64(x);
  *outCarry = xhi >> (nBits - 32);
  return (Word) (((u64) ulowBits((i32) xhi, nBits - 32) << 32) | i96_lo32(x));
#elif EXP / NWORDS == 32                                // nBits = 32 or 33
  i64 xhi = i96_hi64(x);
  *outCarry = xhi >> (nBits - 32);
  u64 xlo = i96_lo64(x);
  return (xlo << (64 - nBits)) >> (64 - nBits);         // ulowBits(xlo, nBits);
#elif EXP / NWORDS == 31                                // nBits = 31 or 32
  *outCarry = (i96_hi64(x) << (32 - nBits)) | ((i96_lo32(x) >> 16) >> (nBits - 16));
  return ((u64) i96_lo64(x) << (64 - nBits)) >> (64 - nBits);         // ulowBits(xlo, nBits);
#else                                                   // nBits less than 32
  *outCarry = (i96_hi64(x) << (32 - nBits)) | (i96_lo32(x) >> nBits);
  return ulowBits(i96_lo32(x), nBits);
#endif
}

Word OVERLOAD carryStepSloppy(i64 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  *outCarry = x >> nBits;
  return ulowBits(x, nBits);
}

Word OVERLOAD carryStepSloppy(i64 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  *outCarry = xtract32(x, nBits);
  return ulowBits(x, nBits);
}

Word OVERLOAD carryStepSloppy(i32 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  *outCarry = x >> nBits;
  return ulowBits(x, nBits);
}

// Carry propagation from word and carry.
Word2 carryWord(Word2 a, CarryABM* carry, bool b1, bool b2) {
  a.x = carryStep(a.x + *carry, carry, b1);
  a.y = carryStep(a.y + *carry, carry, b2);
  return a;
}

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

#endif

#if FFT_FP64 & !COMBO_FFT

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i64 weightAndCarryOne(T u, T invWeight, i64 inCarry, float* maxROE, int sloppy_result_is_acceptable) {

#if !MUL3

  // Convert carry into RNDVAL + carry.
  int2 tmp = as_int2(inCarry); tmp.y += as_int2(RNDVAL).y;
  double RNDVALCarry = as_double(tmp);

  // Apply inverse weight and RNDVAL+carry
  double d = fma(u, invWeight, RNDVALCarry);

  // Optionally calculate roundoff error
  float roundoff = fabs((float) fma(u, -invWeight, d - RNDVALCarry));
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

#elif FFT_FP32 & !COMBO_FFT

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

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer.  Handle MUL3.
i32 weightAndCarryOne(F u, F invWeight, i32 inCarry, float* maxROE, int sloppy_result_is_acceptable) {

#if !MUL3

  // Convert carry into RNDVAL + carry.
  float RNDVALCarry = as_float(as_int(RNDVAL) + inCarry);                       // GWBUG - just the float arithmetic?  s.b. fast

  // Apply inverse weight and RNDVAL+carry
  float d = fma(u, invWeight, RNDVALCarry);

  // Optionally calculate roundoff error
  float roundoff = fabs(fma(u, -invWeight, d - RNDVALCarry));
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

#elif NTT_GF31 & !COMBO_FFT

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i64 weightAndCarryOne(Z61 u, u32 invWeight, i64 inCarry, u32* maxROE) {

  // Apply inverse weight
  u = shr(u, invWeight);

  // Convert input to balanced representation
  i32 value = get_balanced_Z31(u);

  // Optionally calculate roundoff error as proximity to M31/2.
  u32 roundoff = (u32) abs(value);
  *maxROE = max(*maxROE, roundoff);

  // Mul by 3 and add carry
#if MUL3
  value *= 3;
#endif
  return value + inCarry;
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#elif NTT_GF61 & !COMBO_FFT

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

#elif FFT_FP64 & NTT_GF31

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i96 weightAndCarryOne(T u, Z31 u31, T invWeight, u32 m31_invWeight, i64 inCarry, float* maxROE) {

  // Apply inverse weight and get the Z31 data
  u31 = shr(u31, m31_invWeight);
  u32 n31 = get_Z31(u31);

  // The final result must be n31 mod M31.  Use FP64 data to calculate this value.
  u = u * invWeight - (double) n31;                    // This should be close to a multiple of M31
  u *= 4.656612875245796924105750827168e-10;            // Divide by M31.  Could divide by 2^31 (0.0000000004656612873077392578125) be accurate enough?  //GWBUG - check the generated code!  Use 1/M31???

  i64 n64 = RNDVALdoubleToLong(u + RNDVAL);

  i128 v = ((i128) n64 << 31) - n64;                      // n64 * M31
  v += n31;

  // Optionally calculate roundoff error
  float roundoff = (float) fabs(u - (double) n64);
  *maxROE = max(*maxROE, roundoff);

  // Mul by 3 and add carry
#if MUL3
  v = v * 3;
#endif
  v += inCarry;
  i96 value = make_i96((u64) (v >> 32), (u32) v);
  return value;
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_FP32 & NTT_GF31

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i64 weightAndCarryOne(float uF2, Z31 u31, float F2_invWeight, u32 m31_invWeight, i32 inCarry, float* maxROE) {

  // Apply inverse weight and get the Z31 data
  u31 = shr(u31, m31_invWeight);
  u32 n31 = get_Z31(u31);

  // The final result must be n31 mod M31.  Use FP32 data to calculate this value.
  uF2 = uF2 * F2_invWeight - (float) n31;                    // This should be close to a multiple of M31
  uF2 *= 0.0000000004656612873077392578125f;            // Divide by 2^31               //GWBUG - check the generated code!

//  i32 nF2 = rint(uF2);                                        // GWBUG - does this round cheaply?  Best way to round?
// Rounding constant: 3 * 2^22
#define RNDVAL (3.0f * (1 << 22))
  i32 nF2 = lowBits(as_int(uF2 + RNDVAL), 22);

  i64 v = ((i64) nF2 << 31) - nF2;                      // nF2 * M31
  v += n31;

  // Optionally calculate roundoff error
  float roundoff = fabs(uF2 - nF2);
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

#elif FFT_FP32 & NTT_GF61

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i96 weightAndCarryOne(float uF2, Z61 u61, float F2_invWeight, u32 m61_invWeight, i64 inCarry, float* maxROE) {

  // Apply inverse weight and get the Z61 data
  u61 = shr(u61, m61_invWeight);
  u64 n61 = get_Z61(u61);

#if 0
BUG - need more than 64 bit integers

  // The final result must be n61 mod M61.  Use FP32 data to calculate this value.
  uF2 = uF2 * F2_invWeight - (float) n61;                    // This should be close to a multiple of M61
  uF2 *= 4.3368086899420177360298112034798e-19f;             // Divide by 2^61               //GWBUG - check the generated code!

//  i32 nF2 = rint(uF2);                                        // GWBUG - does this round cheaply?  Best way to round?
// Rounding constant: 3 * 2^22
#define RNDVAL (3.0f * (1 << 22))
  i32 nF2 = lowBits(as_int(uF2 + RNDVAL), 22);

  i64 v = ((i64) nF2 << 61) - nF2;                              // nF2 * M61
  v += n61;

  // Optionally calculate roundoff error
  float roundoff = fabs(uF2 - (float) nF2);
  *maxROE = max(*maxROE, roundoff);
#else

  // The final result must be n61 mod M61.  Use FP32 data to calculate this value.
#define RNDVAL (3.0 * (1l << 51))
 double uuF2 = (double) uF2 * (double) F2_invWeight - (double) n61;                    // This should be close to a multiple of M61
 uuF2 = uuF2 * 4.3368086899420177360298112034798e-19;             // Divide by 2^61               //GWBUG - check the generated code!
volatile double xxF2 = uuF2 + RNDVAL;             // Divide by 2^61               //GWBUG - check the generated code!
  xxF2 -= RNDVAL;
  i32 nF2 = (int) xxF2;

  i128 v = ((i128) nF2 << 61) - nF2;                              // nF2 * M61
  v += n61;

  // Optionally calculate roundoff error
  float roundoff = (float) fabs(uuF2 - (double) nF2);
  *maxROE = max(*maxROE, roundoff);
#endif

  // Mul by 3 and add carry
#if MUL3
  v = v * 3;
#endif
  v += inCarry;
  i96 value = make_i96((u64) (v >> 32), (u32) v);
  return value;
}


/**************************************************************************/
/*    Similar to above, but for an NTT based on GF(M31^2)*GF(M61^2)       */
/**************************************************************************/

#elif NTT_GF31 & NTT_GF61

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i96 weightAndCarryOne(Z31 u31, Z61 u61, u32 m31_invWeight, u32 m61_invWeight, i64 inCarry, u32* maxROE) {

  // Apply inverse weights
  u31 = shr(u31, m31_invWeight);
  u61 = shr(u61, m61_invWeight);

  // Use chinese remainder theorem to create a 92-bit result.  Loosely copied from Yves Gallot's mersenne2 program.
  u32 n31 = get_Z31(u31);
  u61 = sub(u61, make_Z61(n31));                 // u61 - u31
  u61 = add(u61, shl(u61, 31));                  // u61 + (u61 << 31)
  u64 n61 = get_Z61(u61);

#if INT128_MATH
i128 v = ((i128) n61 << 31) + n31 - n61;                                                        //GWBUG - is this better/as good as int96 code?
//
//  i96 value = make_i96(n61 >> 1, ((u32) n61 << 31) | n31);     // (n61<<31) + n31
//  i96_sub(&value, n61);

  // Convert to balanced representation by subtracting M61*M31
if ((v >> 64) & 0xF8000000) v = v - (i128) M31 * (i128) M61;
//  if (i96_hi32(value) & 0xF8000000) i96_sub(&value, make_i96(0x0FFFFFFF, 0xDFFFFFFF, 0x80000001));

  // Optionally calculate roundoff error as proximity to M61*M31/2.  27 bits of accuracy should be sufficient.
//  u32 roundoff = (u32) abs((i32) i96_hi32(value));
u32 roundoff = (u32) abs((i32)(v >> 64));
  *maxROE = max(*maxROE, roundoff);

  // Mul by 3 and add carry
#if MUL3
v = v * 3;
//  i96_mul(&value, 3);
#endif
//  i96_add(&value, make_i96((u32)(inCarry >> 63), (u64) inCarry));
v = v + inCarry;
i96 value = make_i96((u64) (v >> 32), (u32) v);

#else

  i96 value = make_i96(n61 >> 1, ((u32) n61 << 31) | n31);     // (n61<<31) + n31
  i96_sub(&value, n61);

  // Convert to balanced representation by subtracting M61*M31
  if (i96_hi32(value) & 0xF8000000) i96_sub(&value, make_i96(0x0FFFFFFF, 0xDFFFFFFF, 0x80000001));

  // Optionally calculate roundoff error as proximity to M61*M31/2.  27 bits of accuracy should be sufficient.
  u32 roundoff = (u32) abs((i32) i96_hi32(value));
  *maxROE = max(*maxROE, roundoff);

  // Mul by 3 and add carry
#if MUL3
  i96_mul(&value, 3);
#endif
  i96_add(&value, make_i96((u32)(inCarry >> 63), (u64) inCarry));

#endif

  return value;
}


#else
error - missing carryUtil implementation
#endif



/**************************************************************************/
/*     Do this last, it depends on weightAndCarryOne defined above        */
/**************************************************************************/

/* Support both 32-bit and 64-bit carries */

#if WordSize <= 4
#define iCARRY i32
#include "carryinc.cl"
#undef iCARRY
#endif

#define iCARRY i64
#include "carryinc.cl"
#undef iCARRY

