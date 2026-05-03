// Copyright (C) Mihai Preda and George Woltman

#define STEP (NWORDS - (EXP % NWORDS))
// bool isBigWord(u32 extra) { return extra < NWORDS - STEP; }

// Determine the fractional-bits-per-word for a given FFT word.  The fracbits value is (word * STEP % NWORDS) / NWORDS.
// The fracbits value is multiplied by 2^32 and truncated to make an integer.  Fracbits can be used to determine weights and big-vs-little-word flags.
// Weight is 2^(1 - fracbits), but if fracbits is zero it is 2^0.
// Big word is true if fracbits + FRAC_BPW_HI does not overflow, but also true if fracbits is zero.
// To eliminate special logic for the FFT word 0, we subtract one from fracbits.
u32 fracBits(u32 i) {
#if NWORDS_IS_POWER_OF_TWO
  return i * (FRAC_BPW_HI + 1) - 1;               // We know FRAC_BPW_LO is -1
#else
  return i * FRAC_BPW_HI + mul_hi(i, FRAC_BPW_LO) - 1;
#endif
}

// Somewhat similar to the above.  Also returns the number of big words that occurred in getting to a given word.
u64 comboFracBits(u32 i) {
#if NWORDS_IS_POWER_OF_TWO
  return (u64)i * (u64)(FRAC_BPW_HI + 1) - 1;     // We know FRAC_BPW_LO is -1
#else
  return (u64)i * (u64)FRAC_BPW_HI + mul_hi(i, FRAC_BPW_LO) - 1;
#endif
}

// Routines to acces the 8 precomputed step weights
u32 weightStepIndex(u32 i) { return i * STEP % NW * (8 / NW); }
u32 weightStepFracBits(u32 i) { return 0xFFFFFFFF - (weightStepIndex(i) << 29); }


#if FFT_FP64

T fweightStep(u32 i) {
  const T TWO_TO_NTH[8] = {
    // 2^(k/8) -1 for k in [0..8)
    0,
    0.090507732665257662,
    0.18920711500272105,
    0.29683955465100964,
    0.41421356237309503,
    0.54221082540794086,
    0.68179283050742912,
    0.83400808640934243,
  };
  return TWO_TO_NTH[weightStepIndex(i)];
}

T iweightStep(u32 i) {
  const T TWO_TO_MINUS_NTH[8] = {
    // 2^-(k/8) - 1 for k in [0..8)
    0,
    -0.082995956795328771,
    -0.15910358474628547,
    -0.2288945872960296,
    -0.29289321881345248,
    -0.35158022267449518,
    -0.40539644249863949,
    -0.45474613366737116,
  };
  return TWO_TO_MINUS_NTH[weightStepIndex(i)];
}

// This routine is not used.  It forces "-use NO_ASM" in Windows.  bfi should be replaced by a builtin if ever needed.
//u32 bfi(u32 u, u32 mask, u32 bits) {
//#if HAS_ASM
//  u32 out;
//  __asm("v_bfi_b32 %0, %1, %2, %3" : "=v"(out) : "v"(mask), "v"(u), "v"(bits));
//  return out;
//#else
//  // return (u & mask) | (bits & ~mask);
//  return (u & mask) | bits;
//#endif
//}

T optionalDouble(T iw) {
  // In a straightforward implementation, inverse weights are between 0.5 and 1.0.  We use inverse weights between 1.0 and 2.0
  // because it allows us to implement this routine with a single OR instruction on the exponent.   The original implementation
  // where this routine took as input values from 0.25 to 1.0 required both an AND and an OR instruction on the exponent.
  // return iw <= 1.0 ? iw * 2 : iw;
  assert(iw > 0.5 && iw < 2);
  uint2 u = as_uint2(iw);

  u.y |= 0x00100000;
  // u.y = bfi(u.y, 0xffefffff, 0x00100000);

  return as_double(u);
}

T optionalHalve(T w) {    // return w >= 4 ? w / 2 : w;
  // In a straightforward implementation, weights are between 1.0 and 2.0.  We use weights between 2.0 and 4.0 because
  // it allows us to implement this routine with a single AND instruction on the exponent.   The original implementation
  // where this routine took as input values from 1.0 to 4.0 required both an AND and an OR instruction on the exponent.
  assert(w >= 2 && w < 8);
  uint2 u = as_uint2(w);
  u.y &= 0xFFEFFFFF;
  //u.y = bfi(u.y, 0xffefffff, 0);
  return as_double(u);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

F fweightStep(u32 i) {
  const F TWO_TO_NTH[8] = {
    // 2^(k/8) -1 for k in [0..8)
    0,
    0.090507732665257662,
    0.18920711500272105,
    0.29683955465100964,
    0.41421356237309503,
    0.54221082540794086,
    0.68179283050742912,
    0.83400808640934243,
  };
  return TWO_TO_NTH[weightStepIndex(i)];
}

F iweightStep(u32 i) {
  const F TWO_TO_MINUS_NTH[8] = {
    // 2^-(k/8) - 1 for k in [0..8)
    0,
    -0.082995956795328771,
    -0.15910358474628547,
    -0.2288945872960296,
    -0.29289321881345248,
    -0.35158022267449518,
    -0.40539644249863949,
    -0.45474613366737116,
  };
  return TWO_TO_MINUS_NTH[weightStepIndex(i)];
}

F optionalDouble(F iw, int flag) {
  // The 23 bits of precision in a float is not enough to handle doubling and halving the same way FP64 does.
  // A straightforward implementation.  Inverse weights are between > 0.5 and <= 1.0.
  F doubled_iw = iw + iw;
  return flag ? doubled_iw : iw;
}

F optionalHalve(F w, int flag) {
  // A straightforward implementation.  Weights are between >= 1.0 and < 2.0.
  F halved_w = w * 0.5f;
  return flag ? halved_w : w;
}

#endif


/**************************************************************************/
/*            Helper routines for NTT weight calculations                 */
/**************************************************************************/

#if NTT_GF31

// if (weight_shift > 31) weight_shift -= 31;
// This version uses PTX instructions which may be faster on nVidia GPUs
u32 adjust_m31_weight_shift (u32 weight_shift) {
  return optional_mod(weight_shift, 31);
}

#endif


#if NTT_GF61

// if (weight_shift > 61) weight_shift -= 61;
// This version uses PTX instructions which may be faster on nVidia GPUs
u32 adjust_m61_weight_shift (u32 weight_shift) {
  return optional_mod(weight_shift, 61);
}

#endif

