// Copyright (C) Mihai Preda

#include "base.cl"
#include "fftwidth.cl"
#include "carryutil.cl"
#include "weight.cl"
#include "middle.cl"

void spin() {
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_s_sleep)
  __builtin_amdgcn_s_sleep(0);
#elif HAS_ASM
  __asm("s_sleep 0");
#else
  // nothing: just spin
  // on Nvidia: see if there's some brief sleep function
#endif
}

// Increasing WMUL to 2 reduces carryShuttle activity.  This led to a 1% speedup on Titan V.  Testing on other GPUs is needed.
#ifndef WMUL
#define WMUL 2
#endif

#if AMDGPU
#define CarryShuttleAccess(me,i)        ((me) * NW + (i))                       // Generates denser global_load_dwordx4 instructions
//#define CarryShuttleAccess(me,i)      ((me) * 4 + (i)%4 + (i)/4 * 4*G_W)      // Also generates global_load_dwordx4 instructions and unit stride when NW=8
#else
#define CarryShuttleAccess(me,i)        ((me) + (i) * G_W)                      // nVidia likes this unit stride better
#endif

// The last WMUL workgroup's carries have been written to global memory.  Now we shuffle WMUL-1 workgroups carries up using local memory.
void OVERLOAD shufl_carries_up(local void *lds2, i64 *carry, u32 me, u32 lowMe) {
  // If WMUL is one, there is no shuffling of carries
  if (WMUL == 1) return;

  const u32 lds_i64s = LDS_BYTES / sizeof(i64);                 // Number of i64s in LDS used by shufl for each WMUL line
  local i64 *lds = (local i64 *) lds2;

  // Handle nasty case where we are writing 8-byte quantities but SHUFL_BYTES_W is only 4 bytes
  if (SHUFL_BYTES_W == 4) {
    if (WMUL == 2) {
      // Full barrier needed as we are using the entire LDS buffer.
      bar();
      // Write the carries.  This will use the entire LDS buffer.
      if (me < G_W) for (i32 i = 0; i < NW; ++i) lds[i * G_W + lowMe] = carry[i];
      // Read carries from previous WMUL workgroup
      bar();
      if (me >= G_W) for (i32 i = 0; i < NW; ++i) carry[i] = lds[i * G_W + lowMe];
      // Full barrier needed as one workgroup just read data from two workgroups LDS buffer.  Not compatible with shufl().
      bar();
    }

    // The really nasty case where all the carries will not fit in LDS memory
    else {
      lds += (me / G_W) * lds_i64s + lowMe;                         // This WMUL workgroup's LDS area
      // Write half the carries to next WMUL's workgroup LDS area
      bar();
      if (me < (WMUL-1) * G_W) for (i32 i = 0; i < NW/2; ++i) lds[lds_i64s + i * G_W] = carry[i];
      // Read carries from our WMUL workgroup LDS area
      bar();
      if (me >= G_W) for (i32 i = 0; i < NW/2; ++i) carry[i] = lds[i * G_W];
      // Write the other half of the carries
      bar();
      if (me < (WMUL-1) * G_W) for (i32 i = 0; i < NW/2; ++i) lds[lds_i64s + i * G_W] = carry[i + NW/2];
      // Read carries from our WMUL workgroup LDS area.  Compatible with shufl, no trailing bar() needed.
      bar();
      if (me >= G_W) for (i32 i = 0; i < NW/2; ++i) carry[i + NW/2] = lds[i * G_W];
    }
  }

  // Easy case.  Write carries to local memory (except last WMUL workgroup which was written to global memory).
  else {
    lds += (me / G_W) * lds_i64s + lowMe;                         // This WMUL workgroup's LDS area
    // Full barrier needed as we are moving data to next WMUL workgroup's LDS area
    bar();
    if (me < (WMUL-1) * G_W) for (i32 i = 0; i < NW; ++i) lds[lds_i64s + i * G_W] = carry[i];
    // Full barrier needed as we just moved data from one WMUL workgroup LDS area to the another WMUL workgroup's LDS area
    bar();
    // Read carries from our WMUL workgroup's LDS area.  This is compatible with shufl and no trailing bar() is required.
    if (me >= G_W) for (i32 i = 0; i < NW; ++i) carry[i] = lds[i * G_W];
  }
}

// The last WMUL workgroup's carries have been written to global memory.  Now we shuffle WMUL-1 workgroup carries up using local memory.
void OVERLOAD shufl_carries_up(local void *lds2, i32 *carry, u32 me, u32 lowMe) {
  // If WMUL is one, there is no shuffling of carries
  if (WMUL == 1) return;

  const u32 lds_i32s = LDS_BYTES / sizeof(i32);                 // Number of i32s in LDS used by shufl for each WMUL line
  local i32 *lds = (local i32 *) lds2;
  lds += (me / G_W) * lds_i32s + lowMe;                         // This WMUL workgroup's LDS area

  // Write carries to local memory (except last WMUL workgroup which was written to global memory)
  // Full barrier needed as we are moving data to next WMUL workgroup's LDS area
  bar();
  if (me < (WMUL-1) * G_W) for (i32 i = 0; i < NW; ++i) lds[lds_i32s + i * G_W] = carry[i];
  // Full barrier needed as we just moved data from one WMUL workgroup LDS area to the another WMUL workgroup's LDS area
  bar();
  // Read carries from our WMUL workgroup's LDS area.  This is compatible with shufl and no trailing bar() is required.
  if (me >= G_W) for (i32 i = 0; i < NW; ++i) carry[i] = lds[i * G_W];
}


#if FFT_TYPE == FFT64

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W * WMUL) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                              ConstBigTab CONST_THREAD_WEIGHTS, BigTab THREAD_WEIGHTS, P(uint) bufROE) {
  local T2 lds[WMUL * LDS_BYTES / sizeof(T2)];

  T2 u[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
#if WMUL == 1
  u32 lowMe = me;
  u32 line = gr;
#else
  u32 lowMe = me % G_W;           // lane-id in one of the WMUL sub-workgroups.
  u32 line = gr * WMUL + me / G_W;
#endif
  if (line >= H) line -= H;

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  dependentLaunchWait();   // Previous kernel was fftMiddleOutFP64

  readCarryFusedLine(in, u, line, lowMe);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
  u32 zerohack = ZEROHACK_W * (u32) get_group_id(0) / 131072;
  fft_WIDTH1(lds + zerohack, u, smallTrig + zerohack, WMUL, lowMe);

  Word2 wu[NW];
#if !NVIDIAGPU || CUDA_BACKEND
  T2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), TSLOAD(&THREAD_WEIGHTS[G_W + line]));
#else
  T2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), CONST_THREAD_WEIGHTS[line % 64]);
  weights.x = optionalDouble(weights.x);
  weights.y = optionalHalve(weights.y);
  weights = fancyMul(weights, CONST_THREAD_WEIGHTS[64 + line / 64]);
#endif

#if MUL3
  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];
#else
  P(CFcarry) carryShuttlePtr = (P(CFcarry)) carryShuttle;
  CFcarry carry[NW+1];
#endif

  float roundMax = 0;
  float carryMax = 0;

  // Calculate the most significant 32-bits of FRAC_BPW * the word index.  Also add FRAC_BPW_HI to test first biglit flag.
  u32 word_index = (lowMe * H + line) * 2;
  u32 frac_bits = fracBits(word_index) + FRAC_BPW_HI;
  const u32 frac_bits_bigstep = fracBits(G_W * H * 2);
  u32 starting_frac_bits = frac_bits;

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  T invBase = optionalDouble(weights.x);
  for (u32 i = 0; i < NW; ++i) {
    T invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    T invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u[i]), U2(invWeight1, invWeight2),
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate frac_bits for next pair
    frac_bits += frac_bits_bigstep;
  }
  frac_bits = starting_frac_bits;     // Restore starting frac_bits for applying weights after carry propagation

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries for the last line in this group. Only groups 0 to H/WMUL-1 need to write carries out.
  // Group H/WMUL is a duplicate of group 0 (producing the same results) so we don't care about that group writing out,
  // but it's fine either way.
  if (gr < H / WMUL && me >= (WMUL-1) * G_W) {
    for (i32 i = 0; i < NW; ++i) { CSSTORE(&carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(lowMe, i)], carry[i]); }

    // Tell next group that its carries are ready
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar(G_W);
    if (lowMe == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lowMe % WAVEFRONT == 0) {
      u32 pos = gr * (G_W / WAVEFRONT) + lowMe / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Group zero will be redone when gr == H / WMUL
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Calculate inverse weights
  T base = optionalHalve(weights.y);
  for (u32 i = 0; i < NW; ++i) {
    T weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    u[i] = U2(weight1, weight2);
  }

  // Shuffle carries up
  shufl_carries_up(lds, carry, me, lowMe);

  // Wait until our carries are ready
  if (me < G_W) {
#if OLD_FENCE
    if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me == 0) ready[gr - 1] = 0;
#else
    u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
    if (me % WAVEFRONT == 0) {
      do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
    __asm("s_setprio 1");
#endif

    // Read from the carryShuttle carries produced by the previous WIDTH group.  Rotate carries from the last WIDTH line.
    // The new carry layout lets the AMD compiler generate global_load_dwordx4 instructions.
    if (gr < H / WMUL) {
      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess(me, i)]);
      }
    } else {

#if !OLD_FENCE
      // For gr==H/WMUL we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
      bar();
#endif

      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess((me + G_W - 1) % G_W, i) /* ((me!=0) + NW - 1 + i) % NW*/]);
      }

      if (me == 0) {
        carry[NW] = carry[NW-1];
        for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
        carry[0] = carry[NW];
      }
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words
  for (i32 i = 0; i < NW; ++i) {
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u[i] = U2(u[i].x * wu[i].x, u[i].y * wu[i].y);

    // Generate frac_bits for next pair
    frac_bits += frac_bits_bigstep;
  }

  dependentLaunch();   // Next kernel will be fftMiddleInFP64

  fft_WIDTH2(lds, u, smallTrig, WMUL, lowMe);
  writeCarryFusedLine(u, out, line, lowMe);
}


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#elif FFT_TYPE == FFT32

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W * WMUL) carryFused(P(F2) out, CP(F2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, TrigFP32 smallTrig,
                              ConstBigTabFP32 CONST_THREAD_WEIGHTS, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  local F2 lds[WMUL * LDS_BYTES / sizeof(F2)];

  F2 u[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
#if WMUL == 1
  u32 lowMe = me;
  u32 line = gr;
#else
  u32 lowMe = me % G_W;           // lane-id in one of the WMUL sub-workgroups.
  u32 line = gr * WMUL + me / G_W;
#endif
  if (line >= H) line -= H;

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  dependentLaunchWait();   // Previous kernel was fftMiddleOutFP32

  readCarryFusedLine(in, u, line, lowMe);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
  u32 zerohack = ZEROHACK_W * (u32) get_group_id(0) / 131072;
  fft_WIDTH1(lds + zerohack, u, smallTrig + zerohack, WMUL, lowMe);

  Word2 wu[NW];
  u32 me_frac_bits = fracBits(lowMe * H * 2);
#if !NVIDIAGPU || CUDA_BACKEND
  F2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), TSLOAD(&THREAD_WEIGHTS[G_W + line]));
  u32 line_frac_bits = fracBits(line * 2);
  u32 base_frac_bits = me_frac_bits + line_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > line_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > line_frac_bits);
#else
  F2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), CONST_THREAD_WEIGHTS[line % 64]);
  u32 partialLine_frac_bits = fracBits((line % 64) * 2);
  u32 base_frac_bits = me_frac_bits + partialLine_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > partialLine_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > partialLine_frac_bits);
  weights = fancyMul(weights, CONST_THREAD_WEIGHTS[64 + line / 64]);
  partialLine_frac_bits = fracBits(((line / 64) * 64) * 2);
  base_frac_bits = base_frac_bits + partialLine_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > partialLine_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > partialLine_frac_bits);
#endif

  P(CFcarry) carryShuttlePtr = (P(CFcarry)) carryShuttle;
  CFcarry carry[NW+1];

  float roundMax = 0;
  float carryMax = 0;

  // Calculate the most significant 32-bits of FRAC_BPW * the word index (it's the same as base_frac_bits).
  u32 word_index = (lowMe * H + line) * 2;
  const u32 frac_bits_bigstep = fracBits(G_W * H * 2 - 1);

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  F invBase = weights.x;
  u32 frac_bits = base_frac_bits;
  for (u32 i = 0; i < NW; ++i) {
    F invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)), frac_bits > base_frac_bits);
    F invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    frac_bits += FRAC_BPW_HI;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u[i]), U2(invWeight1, invWeight2),
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate frac_bits for next pair
    frac_bits += frac_bits_bigstep;
  }

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries for the last line in this group. Only groups 0 to H/WMUL-1 need to write carries out.
  // Group H/WMUL is a duplicate of group 0 (producing the same results) so we don't care about that group writing out,
  // but it's fine either way.
  if (gr < H / WMUL && me >= (WMUL-1) * G_W) {
    for (i32 i = 0; i < NW; ++i) { CSSTORE(&carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(lowMe, i)], carry[i]); }

    // Tell next group that its carries are ready
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar(G_W);
    if (lowMe == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lowMe % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + lowMe / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Group zero will be redone when gr == H / WMUL
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Shuffle carries up
  shufl_carries_up(lds, carry, me, lowMe);

  // Wait until our carries are ready
  if (me < G_W) {
#if OLD_FENCE
    if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me == 0) ready[gr - 1] = 0;
#else
    u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
    if (me % WAVEFRONT == 0) {
      do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
    __asm("s_setprio 1");
#endif

    // Read from the carryShuttle carries produced by the previous WIDTH group.  Rotate carries from the last WIDTH line.
    // The new carry layout lets the AMD compiler generate global_load_dwordx4 instructions.
    if (gr < H / WMUL) {
      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess(me, i)]);
      }
    } else {

#if !OLD_FENCE
      // For gr==H we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
      bar();
#endif

      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess((me + G_W - 1) % G_W, i) /* ((me!=0) + NW - 1 + i) % NW*/]);
      }

      if (me == 0) {
        carry[NW] = carry[NW-1];
        for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
        carry[0] = carry[NW];
      }
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words
  F base = weights.y;
  frac_bits = base_frac_bits;
  for (i32 i = 0; i < NW; ++i) {
    // Calculate inverse weights
    F weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)), frac_bits > base_frac_bits);
    F weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    frac_bits += FRAC_BPW_HI;
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u[i] = U2(weight1 * wu[i].x, weight2 * wu[i].y);

    // Generate frac_bits for next pair
    frac_bits += frac_bits_bigstep;
  }

  dependentLaunch();   // Next kernel will be fftMiddleInFP32

  fft_WIDTH2(lds, u, smallTrig, WMUL, lowMe);
  writeCarryFusedLine(u, out, line, lowMe);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT31

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W * WMUL) carryFused(P(GF31) out, CP(GF31) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, TrigGF31 smallTrig, P(uint) bufROE) {
  local GF31 lds[WMUL * LDS_BYTES / sizeof(GF31)];

  GF31 u[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
#if WMUL == 1
  u32 lowMe = me;
  u32 line = gr;
#else
  u32 lowMe = me % G_W;           // lane-id in one of the WMUL sub-workgroups.
  u32 line = gr * WMUL + me / G_W;
#endif
  if (line >= H) line -= H;

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  dependentLaunchWait();   // Previous kernel was fftMiddleOutGF31

  readCarryFusedLine(in, u, line, lowMe);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
  u32 zerohack = ZEROHACK_W * (u32) get_group_id(0) / 131072;
  fft_WIDTH1(lds + zerohack, u, smallTrig + zerohack, WMUL, lowMe);

  Word2 wu[NW];

  P(CFcarry) carryShuttlePtr = (P(CFcarry)) carryShuttle;
  CFcarry carry[NW+1];

  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (lowMe * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 60th root GF31.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep = (comboFracBits(G_W * H * 2 - 1) + make_u64((G_W * H * 2 - 1) * bigword_weight_shift_minus1, 0)) % (31ULL << 32);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  weight_shift = weight_shift % 31;
  u64 starting_combo_counter = combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = weight_shift + log2_NWORDS + 1;
  if (weight_shift > 31) weight_shift -= 31;

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  for (u32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u[i]), weight_shift0, weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }
  combo_counter = starting_combo_counter;     // Restore starting counter for applying weights after carry propagation

#if ROE
  float fltRoundMax = (float) roundMax / (float) M31;      // For speed, roundoff was computed as 32-bit integer.  Convert to float.
  updateStats(bufROE, posROE, fltRoundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries for the last line in this group. Only groups 0 to H/WMUL-1 need to write carries out.
  // Group H/WMUL is a duplicate of group 0 (producing the same results) so we don't care about that group writing out,
  // but it's fine either way.
  if (gr < H / WMUL && me >= (WMUL-1) * G_W) {
    for (i32 i = 0; i < NW; ++i) { CSSTORE(&carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(lowMe, i)], carry[i]); }

    // Tell next group that its carries are ready
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar(G_W);
    if (lowMe == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lowMe % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + lowMe / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Group zero will be redone when gr == H / WMUL
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Shuffle carries up
  shufl_carries_up(lds, carry, me, lowMe);

  // Wait until our carries are ready
  if (me < G_W) {
#if OLD_FENCE
    if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me == 0) ready[gr - 1] = 0;
#else
    u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
    if (me % WAVEFRONT == 0) {
      do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
    __asm("s_setprio 1");
#endif

    // Read from the carryShuttle carries produced by the previous WIDTH group.  Rotate carries from the last WIDTH line.
    if (gr < H / WMUL) {
      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess(me, i)]);
      }
    } else {

#if !OLD_FENCE
      // For gr==H we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
      bar();
#endif

      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess((me + G_W - 1) % G_W, i) /* ((me!=0) + NW - 1 + i) % NW*/]);
      }

      if (me == 0) {
        carry[NW] = carry[NW-1];
        for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
        carry[0] = carry[NW];
      }
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  for (i32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u[i] = U2(shl(make_Z31(wu[i].x), weight_shift0), shl(make_Z31(wu[i].y), weight_shift1));
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  dependentLaunch();   // Next kernel will be fftMiddleInGF31

  fft_WIDTH2(lds, u, smallTrig, WMUL, lowMe);
  writeCarryFusedLine(u, out, line, lowMe);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT61

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W * WMUL) carryFused(P(GF61) out, CP(GF61) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, TrigGF61 smallTrig, P(uint) bufROE) {
  local GF61 lds[WMUL * LDS_BYTES / sizeof(GF61)];

  GF61 u[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
#if WMUL == 1
  u32 lowMe = me;
  u32 line = gr;
#else
  u32 lowMe = me % G_W;           // lane-id in one of the WMUL sub-workgroups.
  u32 line = gr * WMUL + me / G_W;
#endif
  if (line >= H) line -= H;

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  dependentLaunchWait();   // Previous kernel was fftMiddleOutGF61

  readCarryFusedLine(in, u, line, lowMe);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
  u32 zerohack = ZEROHACK_W * (u32) get_group_id(0) / 131072;
  fft_WIDTH1(lds + zerohack, u, smallTrig + zerohack, WMUL, lowMe);

  Word2 wu[NW];

#if MUL3
  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];
#else
  P(CFcarry) carryShuttlePtr = (P(CFcarry)) carryShuttle;
  CFcarry carry[NW+1];
#endif

  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (lowMe * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 60th root GF61.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
  const u32 log2_root_two = (u32) (((1ULL << 60) / NWORDS) % 61);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep = (comboFracBits(G_W * H * 2 - 1) + make_u64((G_W * H * 2 - 1) * bigword_weight_shift_minus1, 0)) % (61ULL << 32);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  weight_shift = weight_shift % 61;
  u64 starting_combo_counter = combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = weight_shift + log2_NWORDS + 1;
  if (weight_shift > 61) weight_shift -= 61;

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  for (u32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u[i]), weight_shift0, weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }
  combo_counter = starting_combo_counter;     // Restore starting counter for applying weights after carry propagation

#if ROE
  float fltRoundMax = (float) roundMax / (float) (M61 >> 32);      // For speed, roundoff was computed as 32-bit integer.  Convert to float.
  updateStats(bufROE, posROE, fltRoundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries for the last line in this group. Only groups 0 to H/WMUL-1 need to write carries out.
  // Group H/WMUL is a duplicate of group 0 (producing the same results) so we don't care about that group writing out,
  // but it's fine either way.
  if (gr < H / WMUL && me >= (WMUL-1) * G_W) {
    for (i32 i = 0; i < NW; ++i) { CSSTORE(&carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(lowMe, i)], carry[i]); }

    // Tell next group that its carries are ready
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar(G_W);
    if (lowMe == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lowMe % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + lowMe / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Group zero will be redone when gr == H / WMUL
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Shuffle carries up
  shufl_carries_up(lds, carry, me, lowMe);

  // Wait until our carries are ready
  if (me < G_W) {
#if OLD_FENCE
    if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me == 0) ready[gr - 1] = 0;
#else
    u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
    if (me % WAVEFRONT == 0) {
      do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
    __asm("s_setprio 1");
#endif

    // Read from the carryShuttle carries produced by the previous WIDTH group.  Rotate carries from the last WIDTH line.
    // The new carry layout lets the AMD compiler generate global_load_dwordx4 instructions.
    if (gr < H / WMUL) {
      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess(me, i)]);
      }
    } else {

#if !OLD_FENCE
      // For gr==H we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
      bar();
#endif

      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess((me + G_W - 1) % G_W, i) /* ((me!=0) + NW - 1 + i) % NW*/]);
      }

      if (me == 0) {
        carry[NW] = carry[NW-1];
        for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
        carry[0] = carry[NW];
      }
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  for (i32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u[i] = U2(shl(make_Z61(wu[i].x), weight_shift0), shl(make_Z61(wu[i].y), weight_shift1));
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }

  dependentLaunch();   // Next kernel will be fftMiddleInGF61

  fft_WIDTH2(lds, u, smallTrig, WMUL, lowMe);
  writeCarryFusedLine(u, out, line, lowMe);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP64 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT6431

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W * WMUL) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                              ConstBigTab CONST_THREAD_WEIGHTS, BigTab THREAD_WEIGHTS, P(uint) bufROE) {
  local T2 lds[WMUL * LDS_BYTES / sizeof(T2)];
  local GF31 *lds31 = (local GF31 *) lds;

  T2 u[NW];
  GF31 u31[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
#if WMUL == 1
  u32 lowMe = me;
  u32 line = gr;
#else
  u32 lowMe = me % G_W;           // lane-id in one of the WMUL sub-workgroups.
  u32 line = gr * WMUL + me / G_W;
#endif
  if (line >= H) line -= H;

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);

#if HAS_ASM
  __asm("s_setprio 3");
#endif

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
  u32 zerohack = ZEROHACK_W * (u32) get_group_id(0) / 131072;

  readCarryFusedLine(in, u, line, lowMe);
  fft_WIDTH1(lds + zerohack, u, smallTrig + zerohack, WMUL, lowMe);

  dependentLaunchWait();   // Previous kernel was fftMiddleOutGF31

  readCarryFusedLine(in31, u31, line, lowMe);
  fft_WIDTH1(lds31 + zerohack, u31, smallTrig31 + zerohack, WMUL, lowMe);

  Word2 wu[NW];
#if !NVIDIAGPU || CUDA_BACKEND
  T2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), TSLOAD(&THREAD_WEIGHTS[G_W + line]));
#else
  T2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), CONST_THREAD_WEIGHTS[line % 64]);
  weights.x = optionalDouble(weights.x);
  weights.y = optionalHalve(weights.y);
  weights = fancyMul(weights, CONST_THREAD_WEIGHTS[64 + line / 64]);
#endif

  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];

  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (lowMe * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep = (comboFracBits(G_W * H * 2 - 1) + make_u64((G_W * H * 2 - 1) * bigword_weight_shift_minus1, 0)) % (31ULL << 32);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  weight_shift = weight_shift % 31;
  u64 starting_combo_counter = combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = weight_shift + log2_NWORDS + 1;
  if (weight_shift > 31) weight_shift -= 31;

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  T invBase = optionalDouble(weights.x);
  for (u32 i = 0; i < NW; ++i) {
    // Generate the FP64 weights and second GF31 weight shift
    T invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    T invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u[i]), SWAP_XY(u31[i]), invWeight1, invWeight2, weight_shift0, weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      LL != 0, (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }
  combo_counter = starting_combo_counter;     // Restore starting counter for applying weights after carry propagation

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries for the last line in this group. Only groups 0 to H/WMUL-1 need to write carries out.
  // Group H/WMUL is a duplicate of group 0 (producing the same results) so we don't care about that group writing out,
  // but it's fine either way.
  if (gr < H / WMUL && me >= (WMUL-1) * G_W) {
    for (i32 i = 0; i < NW; ++i) { CSSTORE(&carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(lowMe, i)], carry[i]); }

  // Tell next group that its carries are ready
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar(G_W);
    if (lowMe == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lowMe % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + lowMe / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Group zero will be redone when gr == H / WMUL
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Calculate inverse weights
  T base = optionalHalve(weights.y);
  for (u32 i = 0; i < NW; ++i) {
    T weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    u[i] = U2(weight1, weight2);
  }

  // Shuffle carries up
  shufl_carries_up(lds, carry, me, lowMe);

  // Wait until our carries are ready
  if (me < G_W) {
#if OLD_FENCE
    if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me == 0) ready[gr - 1] = 0;
#else
    u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
    if (me % WAVEFRONT == 0) {
      do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
    __asm("s_setprio 1");
#endif

    // Read from the carryShuttle carries produced by the previous WIDTH group.  Rotate carries from the last WIDTH line.
    // The new carry layout lets the AMD compiler generate global_load_dwordx4 instructions.
    if (gr < H / WMUL) {
      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess(me, i)]);
      }
    } else {

#if !OLD_FENCE
      // For gr==H/WMUL we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
      bar();
#endif

      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess((me + G_W - 1) % G_W, i) /* ((me!=0) + NW - 1 + i) % NW*/]);
      }

      if (me == 0) {
        carry[NW] = carry[NW-1];
        for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
        carry[0] = carry[NW];
      }
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  for (i32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u[i] = U2(u[i].x * wu[i].x, u[i].y * wu[i].y);
    u31[i] = U2(shl(make_Z31(wu[i].x), weight_shift0), shl(make_Z31(wu[i].y), weight_shift1));

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  fft_WIDTH2(lds, u, smallTrig, WMUL, lowMe);
  writeCarryFusedLine(u, out, line, lowMe);

  dependentLaunch();   // Next kernel will be fftMiddleInFP32

  fft_WIDTH2(lds31, u31, smallTrig31, WMUL, lowMe);
  writeCarryFusedLine(u31, out31, line, lowMe);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3231

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W * WMUL) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                              ConstBigTabFP32 CONST_THREAD_WEIGHTS, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  local F2 ldsF2[WMUL * LDS_BYTES / sizeof(F2)];
  local GF31 *lds31 = (local GF31 *) ldsF2;

  F2 uF2[NW];
  GF31 u31[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
#if WMUL == 1
  u32 lowMe = me;
  u32 line = gr;
#else
  u32 lowMe = me % G_W;           // lane-id in one of the WMUL sub-workgroups.
  u32 line = gr * WMUL + me / G_W;
#endif
  if (line >= H) line -= H;

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);

#if HAS_ASM
  __asm("s_setprio 3");
#endif

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
  u32 zerohack = ZEROHACK_W * (u32) get_group_id(0) / 131072;

  readCarryFusedLine(inF2, uF2, line, lowMe);
  fft_WIDTH1(ldsF2 + zerohack, uF2, smallTrigF2 + zerohack, WMUL, lowMe);

  dependentLaunchWait();   // Previous kernel was fftMiddleOutGF31

  readCarryFusedLine(in31, u31, line, lowMe);
  fft_WIDTH1(lds31 + zerohack, u31, smallTrig31 + zerohack, WMUL, lowMe);

  Word2 wu[NW];
  u32 me_frac_bits = fracBits(lowMe * H * 2);
#if !NVIDIAGPU || CUDA_BACKEND
  F2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), TSLOAD(&THREAD_WEIGHTS[G_W + line]));
  u32 line_frac_bits = fracBits(line * 2);
  u32 base_frac_bits = me_frac_bits + line_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > line_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > line_frac_bits);
#else
  F2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), CONST_THREAD_WEIGHTS[line % 64]);
  u32 partialLine_frac_bits = fracBits((line % 64) * 2);
  u32 base_frac_bits = me_frac_bits + partialLine_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > partialLine_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > partialLine_frac_bits);
  weights = fancyMul(weights, CONST_THREAD_WEIGHTS[64 + line / 64]);
  partialLine_frac_bits = fracBits(((line / 64) * 64) * 2);
  base_frac_bits = base_frac_bits + partialLine_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > partialLine_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > partialLine_frac_bits);
#endif

  P(i32) carryShuttlePtr = (P(i32)) carryShuttle;
  i32 carry[NW+1];

  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (lowMe * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep = (comboFracBits(G_W * H * 2 - 1) + make_u64((G_W * H * 2 - 1) * bigword_weight_shift_minus1, 0)) % (61ULL << 32);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  weight_shift = weight_shift % 31;
  u64 starting_combo_counter = combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = weight_shift + log2_NWORDS + 1;
  if (weight_shift > 31) weight_shift -= 31;

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  F invBase = weights.x;
  for (u32 i = 0; i < NW; ++i) {
    // Generate the FP32 weights and second GF31 weight shift
    F invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)), frac_bits > base_frac_bits);
    F invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(uF2[i]), SWAP_XY(u31[i]), invWeight1, invWeight2, weight_shift0, weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }
  combo_counter = starting_combo_counter;     // Restore starting counter for applying weights after carry propagation

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries for the last line in this group. Only groups 0 to H/WMUL-1 need to write carries out.
  // Group H/WMUL is a duplicate of group 0 (producing the same results) so we don't care about that group writing out,
  // but it's fine either way.
  if (gr < H / WMUL && me >= (WMUL-1) * G_W) {
    for (i32 i = 0; i < NW; ++i) { CSSTORE(&carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(lowMe, i)], carry[i]); }

    // Tell next group that its carries are ready
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar(G_W);
    if (lowMe == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lowMe % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + lowMe / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Group zero will be redone when gr == H / WMUL
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Shuffle carries up
  shufl_carries_up(ldsF2, carry, me, lowMe);

  // Wait until our carries are ready
  if (me < G_W) {
#if OLD_FENCE
    if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me == 0) ready[gr - 1] = 0;
#else
    u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
    if (me % WAVEFRONT == 0) {
      do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
    __asm("s_setprio 1");
#endif

    // Read from the carryShuttle carries produced by the previous WIDTH group.  Rotate carries from the last WIDTH line.
    // The new carry layout lets the AMD compiler generate global_load_dwordx4 instructions.
    if (gr < H / WMUL) {
      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess(me, i)]);
      }
    } else {

#if !OLD_FENCE
      // For gr==H/WMUL we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
      bar();
#endif

      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess((me + G_W - 1) % G_W, i) /* ((me!=0) + NW - 1 + i) % NW*/]);
      }

      if (me == 0) {
        carry[NW] = carry[NW-1];
        for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
        carry[0] = carry[NW];
      }
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  F base = weights.y;
  for (i32 i = 0; i < NW; ++i) {
    F weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)), frac_bits > base_frac_bits);
    F weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    uF2[i] = U2(weight1 * wu[i].x, weight2 * wu[i].y);
    u31[i] = U2(shl(make_Z31(wu[i].x), weight_shift0), shl(make_Z31(wu[i].y), weight_shift1));

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  fft_WIDTH2(ldsF2, uF2, smallTrigF2, WMUL, lowMe);
  writeCarryFusedLine(uF2, outF2, line, lowMe);

  dependentLaunch();   // Next kernel will be fftMiddleInFP32

  fft_WIDTH2(lds31, u31, smallTrig31, WMUL, lowMe);
  writeCarryFusedLine(u31, out31, line, lowMe);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M61^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3261

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W * WMUL) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                              ConstBigTabFP32 CONST_THREAD_WEIGHTS, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  local GF61 lds61[WMUL * LDS_BYTES / sizeof(GF61)];
  local F2 *ldsF2 = (local F2 *) lds61;

  F2 uF2[NW];
  GF61 u61[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
#if WMUL == 1
  u32 lowMe = me;
  u32 line = gr;
#else
  u32 lowMe = me % G_W;           // lane-id in one of the WMUL sub-workgroups.
  u32 line = gr * WMUL + me / G_W;
#endif
  if (line >= H) line -= H;

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

#if HAS_ASM
  __asm("s_setprio 3");
#endif

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
  u32 zerohack = ZEROHACK_W * (u32) get_group_id(0) / 131072;

  readCarryFusedLine(inF2, uF2, line, lowMe);
  fft_WIDTH1(ldsF2 + zerohack, uF2, smallTrigF2 + zerohack, WMUL, lowMe);

  dependentLaunchWait();   // Previous kernel was fftMiddleOutGF61

  readCarryFusedLine(in61, u61, line, lowMe);
  fft_WIDTH1(lds61 + zerohack, u61, smallTrig61 + zerohack, WMUL, lowMe);

  Word2 wu[NW];
  u32 me_frac_bits = fracBits(lowMe * H * 2);
#if !NVIDIAGPU || CUDA_BACKEND
  F2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), TSLOAD(&THREAD_WEIGHTS[G_W + line]));
  u32 line_frac_bits = fracBits(line * 2);
  u32 base_frac_bits = me_frac_bits + line_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > line_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > line_frac_bits);
#else
  F2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), CONST_THREAD_WEIGHTS[line % 64]);
  u32 partialLine_frac_bits = fracBits((line % 64) * 2);
  u32 base_frac_bits = me_frac_bits + partialLine_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > partialLine_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > partialLine_frac_bits);
  weights = fancyMul(weights, CONST_THREAD_WEIGHTS[64 + line / 64]);
  partialLine_frac_bits = fracBits(((line / 64) * 64) * 2);
  base_frac_bits = base_frac_bits + partialLine_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > partialLine_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > partialLine_frac_bits);
#endif

  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];

  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (lowMe * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
  const u32 log2_root_two = (u32) (((1ULL << 60) / NWORDS) % 61);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep = (comboFracBits(G_W * H * 2 - 1) + make_u64((G_W * H * 2 - 1) * bigword_weight_shift_minus1, 0)) % (61ULL << 32);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  weight_shift = weight_shift % 61;
  u64 starting_combo_counter = combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = weight_shift + log2_NWORDS + 1;
  if (weight_shift > 61) weight_shift -= 61;

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  F invBase = weights.x;
  for (u32 i = 0; i < NW; ++i) {
    // Generate the FP32 weights and second GF61 weight shift
    F invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)), frac_bits > base_frac_bits);
    F invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);

    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(uF2[i]), SWAP_XY(u61[i]), invWeight1, invWeight2, weight_shift0, weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      LL != 0, (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }
  combo_counter = starting_combo_counter;     // Restore starting counter for applying weights after carry propagation

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries for the last line in this group. Only groups 0 to H/WMUL-1 need to write carries out.
  // Group H/WMUL is a duplicate of group 0 (producing the same results) so we don't care about that group writing out,
  // but it's fine either way.
  if (gr < H / WMUL && me >= (WMUL-1) * G_W) {
    for (i32 i = 0; i < NW; ++i) { CSSTORE(&carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(lowMe, i)], carry[i]); }

    // Tell next group that its carries are ready
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar(G_W);
    if (lowMe == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lowMe % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + lowMe / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Group zero will be redone when gr == H / WMUL
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Shuffle carries up
  shufl_carries_up(lds61, carry, me, lowMe);

  // Wait until our carries are ready
  if (me < G_W) {
#if OLD_FENCE
    if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me == 0) ready[gr - 1] = 0;
#else
    u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
    if (me % WAVEFRONT == 0) {
      do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
    __asm("s_setprio 1");
#endif

    // Read from the carryShuttle carries produced by the previous WIDTH group.  Rotate carries from the last WIDTH line.
    // The new carry layout lets the AMD compiler generate global_load_dwordx4 instructions.
    if (gr < H / WMUL) {
      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess(me, i)]);
      }
    } else {

#if !OLD_FENCE
      // For gr==H/WMUL we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
      bar();
#endif

      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess((me + G_W - 1) % G_W, i) /* ((me!=0) + NW - 1 + i) % NW*/]);
      }

      if (me == 0) {
        carry[NW] = carry[NW-1];
        for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
        carry[0] = carry[NW];
      }
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  F base = weights.y;
  for (i32 i = 0; i < NW; ++i) {
    // Calculate inverse weights
    F weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)), frac_bits > base_frac_bits);
    F weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    uF2[i] = U2(weight1 * wu[i].x, weight2 * wu[i].y);
    u61[i] = U2(shl(make_Z61(wu[i].x), weight_shift0), shl(make_Z61(wu[i].y), weight_shift1));

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }

  fft_WIDTH2(ldsF2, uF2, smallTrigF2, WMUL, lowMe);
  writeCarryFusedLine(uF2, outF2, line, lowMe);

  dependentLaunch();   // Next kernel will be fftMiddleInFP32

  fft_WIDTH2(lds61, u61, smallTrig61, WMUL, lowMe);
  writeCarryFusedLine(u61, out61, line, lowMe);
}


/**************************************************************************/
/*    Similar to above, but for an NTT based on GF(M31^2)*GF(M61^2)       */
/**************************************************************************/

#elif FFT_TYPE == FFT3161

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W * WMUL) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig, P(uint) bufROE) {
  local GF61 lds61[WMUL * LDS_BYTES / sizeof(GF61)];
  local GF31 *lds31 = (local GF31 *) lds61;

  GF31 u31[NW];
  GF61 u61[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
#if WMUL == 1
  u32 lowMe = me;
  u32 line = gr;
#else
  u32 lowMe = me % G_W;           // lane-id in one of the WMUL sub-workgroups.
  u32 line = gr * WMUL + me / G_W;
#endif
  if (line >= H) line -= H;

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

#if HAS_ASM
  __asm("s_setprio 3");
#endif

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
  u32 zerohack = ZEROHACK_W * (u32) get_group_id(0) / 131072;

  readCarryFusedLine(in31, u31, line, lowMe);
  fft_WIDTH1(lds31 + zerohack, u31, smallTrig31 + zerohack, WMUL, lowMe);

  dependentLaunchWait();   // Previous kernel was fftMiddleOutGF61

  readCarryFusedLine(in61, u61, line, lowMe);
  fft_WIDTH1(lds61 + zerohack, u61, smallTrig61 + zerohack, WMUL, lowMe);

  Word2 wu[NW];

  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];

  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (lowMe * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 m31_log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 m31_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m31_log2_root_two % 31;
  const u32 m31_bigword_weight_shift_minus1 = (m31_bigword_weight_shift + 30) % 31;
  const u32 m61_log2_root_two = (u32) (((1ULL << 60) / NWORDS) % 61);
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m61_log2_root_two % 61;
  const u32 m61_bigword_weight_shift_minus1 = (m61_bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } m31_combo, m61_combo;
#define frac_bits           m31_combo.a[0]
#define m31_weight_shift    m31_combo.a[1]
#define m31_combo_counter   m31_combo.b
#define m61_weight_shift    m61_combo.a[1]
#define m61_combo_counter   m61_combo.b

  const u64 m31_combo_step = make_u64(m31_bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 m31_combo_bigstep = (comboFracBits(G_W * H * 2 - 1) + make_u64((G_W * H * 2 - 1) * m31_bigword_weight_shift_minus1, 0)) % (31ULL << 32);
  m31_combo_counter = comboFracBits(word_index) + make_u64(word_index * m31_bigword_weight_shift_minus1, 0xFFFFFFFF);
  m31_weight_shift = m31_weight_shift % 31;
  u64 m31_starting_combo_counter = m31_combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation
  const u64 m61_combo_step = make_u64(m61_bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 m61_combo_bigstep = (comboFracBits(G_W * H * 2 - 1) + make_u64((G_W * H * 2 - 1) * m61_bigword_weight_shift_minus1, 0)) % (61ULL << 32);
  m61_combo_counter = comboFracBits(word_index) + make_u64(word_index * m61_bigword_weight_shift_minus1, 0xFFFFFFFF);
  m61_weight_shift = m61_weight_shift % 61;
  u64 m61_starting_combo_counter = m61_combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift + log2_NWORDS + 1);
  m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift + log2_NWORDS + 1);

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  for (u32 i = 0; i < NW; ++i) {
    // Generate the second weight shifts
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u31[i]), SWAP_XY(u61[i]), m31_weight_shift0, m31_weight_shift1, m61_weight_shift0, m61_weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      LL != 0, (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }
  m31_combo_counter = m31_starting_combo_counter;     // Restore starting counter for applying weights after carry propagation
  m61_combo_counter = m61_starting_combo_counter;

#if ROE
  float fltRoundMax = (float) roundMax / (float) 0x1FFFFFFF;      // For speed, roundoff was computed as 32-bit integer.  Convert to float - divide by M61.
  updateStats(bufROE, posROE, fltRoundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries for the last line in this group. Only groups 0 to H/WMUL-1 need to write carries out.
  // Group H/WMUL is a duplicate of group 0 (producing the same results) so we don't care about that group writing out,
  // but it's fine either way.
  if (gr < H / WMUL && me >= (WMUL-1) * G_W) {
    for (i32 i = 0; i < NW; ++i) { CSSTORE(&carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(lowMe, i)], carry[i]); }

    // Tell next group that its carries are ready
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar(G_W);
    if (lowMe == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lowMe % WAVEFRONT == 0) {
      u32 pos = gr * (G_W / WAVEFRONT) + lowMe / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Group zero will be redone when gr == H / WMUL
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Shuffle carries up
  shufl_carries_up(lds61, carry, me, lowMe);

  // Wait until our carries are ready
  if (me < G_W) {
#if OLD_FENCE
    if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me == 0) ready[gr - 1] = 0;
#else
    u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
    if (me % WAVEFRONT == 0) {
      do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
    __asm("s_setprio 1");
#endif

    // Read from the carryShuttle carries produced by the previous WIDTH group.  Rotate carries from the last WIDTH line.
    // The new carry layout lets the AMD compiler generate global_load_dwordx4 instructions.
    if (gr < H / WMUL) {
      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess(me, i)]);
      }
    } else {

#if !OLD_FENCE
      // For gr==H/WMUL we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
      bar();
#endif

      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess((me + G_W - 1) % G_W, i) /* ((me!=0) + NW - 1 + i) % NW*/]);
      }

      if (me == 0) {
        carry[NW] = carry[NW-1];
        for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
        carry[0] = carry[NW];
      }
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  for (i32 i = 0; i < NW; ++i) {
    // Generate the second weight shifts
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u31[i] = U2(shl(make_Z31(wu[i].x), m31_weight_shift0), shl(make_Z31(wu[i].y), m31_weight_shift1));
    u61[i] = U2(shl(make_Z61(wu[i].x), m61_weight_shift0), shl(make_Z61(wu[i].y), m61_weight_shift1));

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }

  fft_WIDTH2(lds31, u31, smallTrig31, WMUL, lowMe);
  writeCarryFusedLine(u31, out31, line, lowMe);

  dependentLaunch();   // Next kernel will be fftMiddleInGF31

  fft_WIDTH2(lds61, u61, smallTrig61, WMUL, lowMe);
  writeCarryFusedLine(u61, out61, line, lowMe);
}


/******************************************************************************/
/*  Similar to above, but for a hybrid FFT based on FP32*GF(M31^2)*GF(M61^2)  */
/******************************************************************************/

#elif FFT_TYPE == FFT323161

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W * WMUL) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                              ConstBigTabFP32 CONST_THREAD_WEIGHTS, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  local GF61 lds61[WMUL * LDS_BYTES / sizeof(GF61)];
  local F2 *ldsF2 = (local F2 *) lds61;
  local GF31 *lds31 = (local GF31 *) lds61;

  F2 uF2[NW];
  GF31 u31[NW];
  GF61 u61[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
#if WMUL == 1
  u32 lowMe = me;
  u32 line = gr;
#else
  u32 lowMe = me % G_W;           // lane-id in one of the WMUL sub-workgroups.
  u32 line = gr * WMUL + me / G_W;
#endif
  if (line >= H) line -= H;

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

#if HAS_ASM
  __asm("s_setprio 3");
#endif

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
  u32 zerohack = ZEROHACK_W * (u32) get_group_id(0) / 131072;

  readCarryFusedLine(inF2, uF2, line, lowMe);
  fft_WIDTH1(ldsF2 + zerohack, uF2, smallTrigF2 + zerohack, WMUL, lowMe);

  readCarryFusedLine(in31, u31, line, lowMe);
  fft_WIDTH1(lds31 + zerohack, u31, smallTrig31 + zerohack, WMUL, lowMe);

  dependentLaunchWait();   // Previous kernel was fftMiddleOutGF61

  readCarryFusedLine(in61, u61, line, lowMe);
  fft_WIDTH1(lds61 + zerohack, u61, smallTrig61 + zerohack, WMUL, lowMe);

  Word2 wu[NW];
  u32 me_frac_bits = fracBits(lowMe * H * 2);
#if !NVIDIAGPU || CUDA_BACKEND
  F2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), TSLOAD(&THREAD_WEIGHTS[G_W + line]));
  u32 line_frac_bits = fracBits(line * 2);
  u32 base_frac_bits = me_frac_bits + line_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > line_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > line_frac_bits);
#else
  F2 weights = fancyMul(TFLOAD(&THREAD_WEIGHTS[lowMe]), CONST_THREAD_WEIGHTS[line % 64]);
  u32 partialLine_frac_bits = fracBits((line % 64) * 2);
  u32 base_frac_bits = me_frac_bits + partialLine_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > partialLine_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > partialLine_frac_bits);
  weights = fancyMul(weights, CONST_THREAD_WEIGHTS[64 + line / 64]);
  partialLine_frac_bits = fracBits(((line / 64) * 64) * 2);
  base_frac_bits = base_frac_bits + partialLine_frac_bits;
  weights.x = optionalDouble(weights.x, base_frac_bits > partialLine_frac_bits);
  weights.y = optionalHalve(weights.y, base_frac_bits > partialLine_frac_bits);
#endif

  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];

  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (lowMe * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 m31_log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 m31_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m31_log2_root_two % 31;
  const u32 m31_bigword_weight_shift_minus1 = (m31_bigword_weight_shift + 30) % 31;
  const u32 m61_log2_root_two = (u32) (((1ULL << 60) / NWORDS) % 61);
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m61_log2_root_two % 61;
  const u32 m61_bigword_weight_shift_minus1 = (m61_bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } m31_combo, m61_combo;
#define frac_bits           m31_combo.a[0]
#define m31_weight_shift    m31_combo.a[1]
#define m31_combo_counter   m31_combo.b
#define m61_weight_shift    m61_combo.a[1]
#define m61_combo_counter   m61_combo.b

  const u64 m31_combo_step = make_u64(m31_bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 m31_combo_bigstep = (comboFracBits(G_W * H * 2 - 1) + make_u64((G_W * H * 2 - 1) * m31_bigword_weight_shift_minus1, 0)) % (31ULL << 32);
  m31_combo_counter = comboFracBits(word_index) + make_u64(word_index * m31_bigword_weight_shift_minus1, 0xFFFFFFFF);
  m31_weight_shift = m31_weight_shift % 31;
  u64 m31_starting_combo_counter = m31_combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation
  const u64 m61_combo_step = make_u64(m61_bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 m61_combo_bigstep = (comboFracBits(G_W * H * 2 - 1) + make_u64((G_W * H * 2 - 1) * m61_bigword_weight_shift_minus1, 0)) % (61ULL << 32);
  m61_combo_counter = comboFracBits(word_index) + make_u64(word_index * m61_bigword_weight_shift_minus1, 0xFFFFFFFF);
  m61_weight_shift = m61_weight_shift % 61;
  u64 m61_starting_combo_counter = m61_combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift + log2_NWORDS + 1);
  m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift + log2_NWORDS + 1);

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  F invBase = weights.x;
  for (u32 i = 0; i < NW; ++i) {
    // Generate the FP32 weights and second GF31 and GF61 weight shift
    F invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)), frac_bits > base_frac_bits);
    F invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(uF2[i]), SWAP_XY(u31[i]), SWAP_XY(u61[i]), invWeight1, invWeight2, m31_weight_shift0, m31_weight_shift1, m61_weight_shift0, m61_weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      LL != 0, (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }
  m31_combo_counter = m31_starting_combo_counter;     // Restore starting counter for applying weights after carry propagation
  m61_combo_counter = m61_starting_combo_counter;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries for the last line in this group. Only groups 0 to H/WMUL-1 need to write carries out.
  // Group H/WMUL is a duplicate of group 0 (producing the same results) so we don't care about that group writing out,
  // but it's fine either way.
  if (gr < H / WMUL && me >= (WMUL-1) * G_W) {
    for (i32 i = 0; i < NW; ++i) { CSSTORE(&carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(lowMe, i)], carry[i]); }

    // Tell next group that its carries are ready
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar(G_W);
    if (lowMe == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lowMe % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + lowMe / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Group zero will be redone when gr == H / WMUL
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Shuffle carries up
  shufl_carries_up(lds61, carry, me, lowMe);

  // Wait until our carries are ready
  if (me < G_W) {
#if OLD_FENCE
    if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me == 0) ready[gr - 1] = 0;
#else
    u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
    if (me % WAVEFRONT == 0) {
      do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    // Clear carry ready flag for next iteration
    if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
    __asm("s_setprio 1");
#endif

    // Read from the carryShuttle carries produced by the previous WIDTH group.  Rotate carries from the last WIDTH line.
    // The new carry layout lets the AMD compiler generate global_load_dwordx4 instructions.
    if (gr < H / WMUL) {
      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess(me, i)]);
      }
    } else {

#if !OLD_FENCE
      // For gr==H/WMUL we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
      bar();
#endif

      for (i32 i = 0; i < NW; ++i) {
        carry[i] = CSLOAD(&carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess((me + G_W - 1) % G_W, i) /* ((me!=0) + NW - 1 + i) % NW*/]);
      }

      if (me == 0) {
        carry[NW] = carry[NW-1];
        for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
        carry[0] = carry[NW];
      }
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  F base = weights.y;
  for (i32 i = 0; i < NW; ++i) {
    // Calculate inverse weights
    F weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)), frac_bits > base_frac_bits);
    F weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    // Generate the second weight shifts
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    uF2[i] = U2(weight1 * wu[i].x, weight2 * wu[i].y);
    u31[i] = U2(shl(make_Z31(wu[i].x), m31_weight_shift0), shl(make_Z31(wu[i].y), m31_weight_shift1));
    u61[i] = U2(shl(make_Z61(wu[i].x), m61_weight_shift0), shl(make_Z61(wu[i].y), m61_weight_shift1));

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }

  fft_WIDTH2(ldsF2, uF2, smallTrigF2, WMUL, lowMe);
  writeCarryFusedLine(uF2, outF2, line, lowMe);

  dependentLaunch();   // Next kernel will be fftMiddleInFP32

  fft_WIDTH2(lds31, u31, smallTrig31, WMUL, lowMe);
  writeCarryFusedLine(u31, out31, line, lowMe);

  fft_WIDTH2(lds61, u61, smallTrig61, WMUL, lowMe);
  writeCarryFusedLine(u61, out61, line, lowMe);
}


#else
error - missing CarryFused kernel implementation
#endif
