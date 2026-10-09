// Copyright (C) Mihai Preda

// This file is included with different definitions for iCARRY.  iCARRY is all possible datatypes for CFCarry in carryFused.

/***************************************************************************/
/*  From the FFT data, construct a value to normalize and carry propagate  */
/***************************************************************************/

#if FFT_TYPE == FFT64

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i64 OVERLOAD weightAndCarryOne(T u, T invWeight, iCARRY inCarry, float* maxROE, int sloppy_result_is_acceptable) {

#if !MUL3

  // Convert carry into RNDVAL + carry.
  int2 tmp = as_int2((i64) inCarry); tmp.y += as_int2(RNDVAL).y;
  double RNDVALCarry = as_double(tmp);

  // Apply inverse weight and RNDVAL+carry
  double d = fma(u, invWeight, RNDVALCarry);

  // Optionally calculate roundoff error
  float roundoff = fabs((float) fma(u, invWeight, RNDVALCarry - d));
  *maxROE = max(*maxROE, roundoff);

  // Convert to long (for CARRY32 case we don't need to strip off the RNDVAL bits).
  // The CARRY32 carryStep reads the carry from bits [nBits, nBits+32), and RNDVAL sets bit 51 iff the value is >= 0.  Below 19 bpw
  // that window stays under bit 51.  At 19 bpw a big word's window is [20,52): flipping bit 51 turns it into the sign bit of the
  // value as a 52-bit two's complement integer (the bits above it are never read).
  if (sloppy_result_is_acceptable) return (EXP / NWORDS >= 19) ? as_long(d) ^ ((i64) 1 << 51) : as_long(d);
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
i32 OVERLOAD weightAndCarryOne(F u, F invWeight, iCARRY inCarry, float* maxROE, int sloppy_result_is_acceptable) {

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
i64 OVERLOAD weightAndCarryOne(Z31 u, u32 invWeight, iCARRY inCarry, u32* maxROE) {

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
i64 OVERLOAD weightAndCarryOne(Z61 u, u32 invWeight, iCARRY inCarry, u32* maxROE) {

  // Apply inverse weight
  u = shr(u, invWeight);

  // Convert input to balanced representation
  i64 value = get_balanced_Z61(u);

  // Optionally calculate roundoff error as proximity to M61/2.  28 bits of accuracy should be sufficient.
  u32 roundoff = (u32) abs((i32) hi32(value));
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
i96 OVERLOAD weightAndCarryOne(T u, Z31 u31, T invWeight, u32 m31_invWeight, bool hasInCarry, iCARRY inCarry, float* maxROE) {

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
  value = sub(value, n64);                          // n64 * M31 + n31

  // Mul by 3 and add carry
#if MUL3
  value = add(value, add(value, value));
#endif
  if (hasInCarry) value = add(value, inCarry);
  return value;
}

/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3231

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i64 OVERLOAD weightAndCarryOne(F uF, Z31 u31, F F2_invWeight, u32 m31_invWeight, iCARRY inCarry, float* maxROE) {

  // Apply inverse weight and get the Z31 data
  u31 = shr(u31, m31_invWeight);
  u32 n31 = get_Z31(u31);

  // The final result must be n31 mod M31.  Use FP32 data to calculate this value.
  uF = fma(uF, F2_invWeight, - (float) n31);                              // This should be close to a multiple of M31
  float uFint = fma(uF, 4.656612875245796924105750827168e-10f, RNDVAL);   // Divide by M31 and round to int
  i32 nF = RNDVALfloatToInt(uFint);

  i64 v = (((i64) nF << 31) | n31) - nF;         // nF * M31 + n31

  // Optionally calculate roundoff error
  float roundoff = fabs(fma(uF, 4.656612875245796924105750827168e-10f, RNDVAL - uFint));
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
i96 OVERLOAD weightAndCarryOne(F uF, Z61 u61, F F2_invWeight, u32 m61_invWeight, bool hasInCarry, iCARRY inCarry, float* maxROE) {

  // Apply inverse weight and get the Z61 data
  u61 = shr(u61, m61_invWeight);
  u64 n61 = get_Z61(u61);

  // The final result mod M61 must be n61.  Use FP32 data to calculate how many multiples of M61 need to be added to n61.
  float n61f = (float)hi32(n61) * -4294967296.0f;                           // Estimate -n61 as a float.
  uF = fma(uF, F2_invWeight, n61f);                                         // This should be close to an integer multiple of M61
  float uFint = fma(uF, 4.3368086899420177360298112034798e-19f, RNDVAL);    // Divide by M61 and round to int
  i32 nF = RNDVALfloatToInt(uFint);

  // Optionally calculate roundoff error
  float roundoff = fabs(fma(uF, 4.3368086899420177360298112034798e-19f, RNDVAL - uFint));
  *maxROE = max(*maxROE, roundoff);

  // Compute the value using i96 math
  i32 vhi = nF >> 3;
  u64 vlo = ((u64)nF << 61) | n61;
  i96 value = make_i96(vhi, vlo);               // (nF << 61) + n61
  value = sub(value, nF);                       // nF * M61 + n61

  // Mul by 3 and add carry
#if MUL3
  value = add(value, add(value, value));
#endif
  if (hasInCarry) value = add(value, inCarry);
  return value;
}

/**************************************************************************/
/*    Similar to above, but for an NTT based on GF(M31^2)*GF(M61^2)       */
/**************************************************************************/

#elif FFT_TYPE == FFT3161

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i96 OVERLOAD weightAndCarryOne(Z31 u31, Z61 u61, u32 m31_invWeight, u32 m61_invWeight, bool hasInCarry, iCARRY inCarry, u32* maxROE) {

  // Apply inverse weights
  u31 = shr(u31, m31_invWeight);
  u61 = shr(u61, m61_invWeight);

  // Use chinese remainder theorem to create a 92-bit result.  Loosely copied from Yves Gallot's mersenne2 program.
  u32 n31 = get_Z31(u31);
  u61 += make_u64(hi32(M61), lo32(M61) - n31);   // u61 - u31
  u61 += shl(u61, 31);                           // u61 + (u61 << 31)

  // The resulting value will be get_Z61(u61) * M31 + n31 and if larger than ~M31*M61/2 return a negative value by subtracting M31 * M61.
  // We can save a little work by determining if the result will be large using just u61 and returning (get_Z61(u61) - M61) * M31 + n31.
  // This simplifies to get_balanced_Z61(u61) * M31 + n31.
  i64 n61 = get_balanced_Z61(modM61(u61));

  // Optionally calculate roundoff error as proximity to M61/2.  28 bits of accuracy should be sufficient.
  u32 roundoff = (u32) abs((i32)hi32(n61));
  *maxROE = max(*maxROE, roundoff);

  // Compute the value using i96 math
  i64 vhi = n61 >> 1;
  u32 vlo = ((u32)n61 << 31) | n31;
  i96 value = make_i96(vhi, vlo);                // (n61 << 31) + n31
  value = sub(value, n61);                       // n61 * M31 + n31

  // Mul by 3 and add carry
#if MUL3
  value = add(value, add(value, value));
#endif
  if (hasInCarry) value = add(value, inCarry);
  return value;
}

/******************************************************************************/
/*  Similar to above, but for a hybrid FFT based on FP32*GF(M31^2)*GF(M61^2)  */
/******************************************************************************/

#elif FFT_TYPE == FFT323161

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i128 OVERLOAD weightAndCarryOne(F uF, Z31 u31, Z61 u61, F F2_invWeight, u32 m31_invWeight, u32 m61_invWeight, bool hasInCarry, iCARRY inCarry, float* maxROE) {

  // Apply inverse weights
  u31 = shr(u31, m31_invWeight);
  u61 = shr(u61, m61_invWeight);
  // Use chinese remainder theorem to create a 92-bit result.  Loosely copied from Yves Gallot's mersenne2 program.
  u32 n31 = get_Z31(u31);
  u61 += make_u64(hi32(M61), lo32(M61) - n31);       // u61 - u31
  u61 += shl(u61, 31);                               // u61 + (u61 << 31)
  u64 n61 = get_Z61(modM61(u61));
  // Let's call the 92-bit CRT result n3161.  At this point, n3161 = n61 * M31 + n31.

  // The final result mod M31*M61 must be n3161.  Use FP32 data to calculate how many multiples of M31*M61 need to be added to n3161.
  float n3161f = (float)hi32(n61) * -9223372036854775808.0f;               // Estimate -n3161 as a float.  -n61 << 31 should be close enough.
  uF = fma(uF, F2_invWeight, n3161f);                                      // This should be close to an integer multiple of M31*M61
  float uFint = fma(uF, 2.0194839183061857038255724444152e-28f, RNDVAL);   // Divide by M31*M61 and round to int
  i32 nF = RNDVALfloatToInt(uFint);

  // The final result will be nF * M31*M61 + n3161.  Rearranging to use as few 128-bit and 64-bit ops as possible:
  // = nF * M61 * M31 + n61 * M31 + n31
  // = (nF * M61 + n61) * M31 + n31
  // = ((nF << 61) - nF + n61) * M31 + n31
  // = (((nF << 61) - nF + n61) << 31) - ((nF << 61) - nF + n61) + n31
  // = (nF << 92) + ((n61 - nF) << 31) - (nF << 61) - (n61 - nF) + n31
  // = (nF << 92) - (nF << 61) + ((n61 - nF) << 31) - (n61 - nF) + n31
  // = (((nF << 31) - nF) << 61) + ((n61 - nF) << 31) - (n61 - nF) + n31
  // = (((nF << 32) - nF*2) << 60) + ((n61 - nF) << 31) - (n61 - nF) + n31

  // Since -(nF << 61) = -((nF << 30) << 31), the middle terms can be combined before shifting:
  // = (nF << 92) + ((n61 - nF - (nF << 30)) << 31) - (n61 - nF) + n31

  // Compute x = (n61 - nF) and x2 = x - (nF << 30), which still fits in an i64
  i64 x = (i64)n61 - nF;
  i64 x2 = x - ((i64)nF << 30);
#if !MUL3
  // Put the parts together.  The low 31 bits of x2 << 31 are zero, so n31 is or'ed in, and nF << 92 only touches the high 64 bits.
  i128 v = sub(make_i128((x2 >> 33) + ((i64)nF << 28), (x2 << 31) | n31), x);
#else
  // Mul by 3.  Tripling the parts is cheaper than tripling the i128, and each still fits: 3 * x2 and 3 * (n31 - x) are below 2^63.
  i64 x2_3 = x2 * 3;
  i128 v = add(make_i128((x2_3 >> 33) + ((i64)(nF * 3) << 28), x2_3 << 31), ((i64)n31 - x) * 3);
#endif

  // Optionally calculate roundoff error
  float roundoff = fabs(fma(uF, 2.0194839183061857038255724444152e-28f, RNDVAL - uFint));
  *maxROE = max(*maxROE, roundoff);

  // Add carry (MUL3 was applied above)
  if (hasInCarry) v = add(v, inCarry);
  return v;
}

#else
error - missing weightAndCarryOne implementation
#endif


#if !ICARRY_I96
Word2 OVERLOAD carryFinal(Word2 u, iCARRY inCarry, bool b1) {
  iCARRY tmpCarry;
  u.x = carryStepSignedSloppy(u.x + inCarry, &tmpCarry, b1);
  u.y += tmpCarry;
  return u;
}
#else
// A 96-bit carry is a struct, so it is added with add(i96, i64).  The carry out of the first word is small enough for an i64.
// Like the i64 version (whose carryStepSignedSloppy is carryStep) the word is balanced exactly.
Word2 OVERLOAD carryFinal(Word2 u, iCARRY inCarry, bool b1) {
  i64 tmpCarry;
  u.x = carryStep(add(inCarry, (i64) u.x), &tmpCarry, b1);
  u.y += tmpCarry;
  return u;
}
#endif

/*******************************************************************************************/
/*  Original FP64 version to start the carry propagation process for a pair of FFT values  */
/*******************************************************************************************/

#if FFT_TYPE == FFT64

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(T2 u, T2 invWeight, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight.x, inCarry, maxROE, sizeof(midCarry) == 4);
  Word a = carryStep(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight.y, midCarry, maxROE, sizeof(midCarry) == 4);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(T2 u, T2 invWeight, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight.x, inCarry, maxROE, sizeof(midCarry) == 4);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight.y, midCarry, maxROE, sizeof(midCarry) == 4);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#elif FFT_TYPE == FFT32

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(F2 u, F2 invWeight, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i32 tmp1 = weightAndCarryOne(u.x, invWeight.x, inCarry, maxROE, sizeof(midCarry) == 4);
  Word a = carryStep(tmp1, &midCarry, b1);
  i32 tmp2 = weightAndCarryOne(u.y, invWeight.y, midCarry, maxROE, sizeof(midCarry) == 4);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(F2 u, F2 invWeight, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i32 tmp1 = weightAndCarryOne(u.x, invWeight.x, inCarry, maxROE, sizeof(midCarry) == 4);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i32 tmp2 = weightAndCarryOne(u.y, invWeight.y, midCarry, maxROE, sizeof(midCarry) == 4);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT31

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(GF31 u, u32 invWeight1, u32 invWeight2, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight1, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight2, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(GF31 u, u32 invWeight1, u32 invWeight2, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight1, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight2, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT61

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(GF61 u, u32 invWeight1, u32 invWeight2, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight1, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight2, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(GF61 u, u32 invWeight1, u32 invWeight2, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight1, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight2, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP64 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT6431

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(T2 u, GF31 u31, T invWeight1, T invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                  bool hasInCarry, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i96 tmp1 = weightAndCarryOne(u.x, u31.x, invWeight1, m31_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(u.y, u31.y, invWeight2, m31_invWeight2, true, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(T2 u, GF31 u31, T invWeight1, T invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                        bool hasInCarry, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i96 tmp1 = weightAndCarryOne(u.x, u31.x, invWeight1, m31_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(u.y, u31.y, invWeight2, m31_invWeight2, true, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3231

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(F2 uF, GF31 u31, F invWeight1, F invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                  iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(uF.x, u31.x, invWeight1, m31_invWeight1, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(uF.y, u31.y, invWeight2, m31_invWeight2, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(F2 uF, GF31 u31, F invWeight1, F invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                        iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(uF.x, u31.x, invWeight1, m31_invWeight1, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(uF.y, u31.y, invWeight2, m31_invWeight2, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M61^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3261

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(F2 uF, GF61 u61, F invWeight1, F invWeight2, u32 m61_invWeight1, u32 m61_invWeight2,
                                  bool hasInCarry, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i96 tmp1 = weightAndCarryOne(uF.x, u61.x, invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(uF.y, u61.y, invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(F2 uF, GF61 u61, F invWeight1, F invWeight2, u32 m61_invWeight1, u32 m61_invWeight2,
                                        bool hasInCarry, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i96 tmp1 = weightAndCarryOne(uF.x, u61.x, invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(uF.y, u61.y, invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*    Similar to above, but for an NTT based on GF(M31^2)*GF(M61^2)       */
/**************************************************************************/

#elif FFT_TYPE == FFT3161

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(GF31 u31, GF61 u61, u32 m31_invWeight1, u32 m31_invWeight2, u32 m61_invWeight1, u32 m61_invWeight2,
                                  bool hasInCarry, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i96 tmp1 = weightAndCarryOne(u31.x, u61.x, m31_invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(u31.y, u61.y, m31_invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(GF31 u31, GF61 u61, u32 m31_invWeight1, u32 m31_invWeight2, u32 m61_invWeight1, u32 m61_invWeight2,
                                        bool hasInCarry, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i96 tmp1 = weightAndCarryOne(u31.x, u61.x, m31_invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(u31.y, u61.y, m31_invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

/******************************************************************************/
/*  Similar to above, but for a hybrid FFT based on FP32*GF(M31^2)*GF(M61^2)  */
/******************************************************************************/

#elif FFT_TYPE == FFT323161

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(F2 uF, GF31 u31, GF61 u61, F invWeight1, F invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                  u32 m61_invWeight1, u32 m61_invWeight2, bool hasInCarry, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i128 tmp1 = weightAndCarryOne(uF.x, u31.x, u61.x, invWeight1, m31_invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i128 tmp2 = weightAndCarryOne(uF.y, u31.y, u61.y, invWeight2, m31_invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(F2 uF, GF31 u31, GF61 u61, F invWeight1, F invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                        u32 m61_invWeight1, u32 m61_invWeight2, bool hasInCarry, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i128 tmp1 = weightAndCarryOne(uF.x, u31.x, u61.x, invWeight1, m31_invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i128 tmp2 = weightAndCarryOne(uF.y, u31.y, u61.y, invWeight2, m31_invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

#else
error - missing weightAndCarryPair implementation
#endif
