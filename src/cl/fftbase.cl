// Copyright (C) Mihai Preda

#include "fft4.cl"
#include "fft8.cl"

// NOTE:  tailSquare, with its ability to optionally define tailSquareZero, does not allow us to know numWG at #include time.
// Thus, we must define macros that take numWG as in input argument.  This could be rectified by making tailSquareZero obey the TAIL_KERNELS setting.

// This section is not necessary.  On TitanV, CUDA 12.9, I see a 0.5% slowdown when not LDS sharing but compiled with the LDS sharing code.

#if LDSMUL == 1      // Not sharing LDS memory, use simplified code.

#define SHARING_LDS(numWG)        0
#define SBMUL(numWG)              1
#define LDSPAD_COUNT(numWG)       (!LDSPAD ? 0 : RADIX == 4 ? 12 : SHUFL_BYTES >= 16 ? 7 : 56)
#define LDS_SHUFL_BYTES(numWG)    ((WG * RADIX + LDSPAD_COUNT(numWG)) * SHUFL_BYTES)
#define LDS_BYTES(numWG)          (numWG * LDS_SHUFL_BYTES(numWG))

void OVERLOAD LDSinit(void local *lds, const u32 numWG) {
}

local void * OVERLOAD LDSptr(local void *lds, const u32 numWG) {
  return (local char *)lds + ((u32)get_local_id(0) / WG) * LDS_SHUFL_BYTES(numWG);
}

local void * OVERLOAD LDSsharing_ptr(local void *lds, const u32 numWG) {
  return LDSptr(lds, numWG);
}

void OVERLOAD LDSbar(const u32 numWG) {
  bar(WG);
}

void OVERLOAD LDStx_start(local void *lds, const u32 numWG) {
  LDSbar(numWG);
}

void OVERLOAD LDStx_end(local void *lds, const u32 numWG) {
}


// This section handles both cases of sharing and not sharing LDS memory

#else

// LDS access is shared if the kernel processes multiple independent workgroups, the user settable LDSMUL is more than one, and the GPU allows barriers on a subset of threads
#define SHARING_LDS(numWG)        (numWG > 1 && LDSMUL > 1 && (NVIDIAGPU || WG <= WAVEFRONT))
// If sharing LDS access, LDSMUL sets a limit on how many workgroups share the same LDS memory.  Sharing LDS allow shufl to use a multiple of SHUFL_BYTES.
#define SBMUL(numWG)              (!SHARING_LDS(numWG) ? 1 : numWG >= LDSMUL ? LDSMUL : numWG)
// Calculate the LDS padding used by shufl
#define LDSPAD_COUNT(numWG)       (!LDSPAD ? 0 : RADIX == 4 ? 12 : SBMUL(numWG) * SHUFL_BYTES >= 16 ? 7 : 56)
// LDS_SHUFL_BYTES is the number of LDS bytes *allocated* for each workgroup (SBMUL > 1 means the workgroup can *access* some multiple of LDS_SHUFL_BYTES)
#define LDS_SHUFL_BYTES(numWG)    ((WG * RADIX + LDSPAD_COUNT(numWG)) * SHUFL_BYTES)
// The workgroups are partitioned into groups of SBMUL that share one LDS region and one semaphore.
// SBMUL need not divide numWG, so round up: the last group is short but still spans SBMUL regions and
// owns a semaphore of its own.  Allocating only numWG regions, and only ever four semaphores, is not
// enough then -- LDSsharing_ptr hands out a region past the end of the array and LDSinit leaves the
// last semaphore uninitialised.  With SBMUL == 1, which is every configuration that does not opt into
// sharing, all of this is numWG regions and no semaphores, exactly as before.
#define LDS_GROUPS(numWG)         ((numWG + SBMUL(numWG) - 1) / SBMUL(numWG))
#define LDS_REGIONS(numWG)        (LDS_GROUPS(numWG) * SBMUL(numWG))
#define LDS_SEM_OFFSET(numWG)     (LDS_REGIONS(numWG) * LDS_SHUFL_BYTES(numWG))
#define LDS_BYTES(numWG)          (LDS_SEM_OFFSET(numWG) + (SHARING_LDS(numWG) ? LDS_GROUPS(numWG) * 4 : 0))

// Variant 2 keeps its own pointer into the shared region (partitioned_lds) and both reads and writes it
// in partial_tabMul4/8 outside the LDStx lock, with only a bar(WG) that does not cover the other
// workgroups sharing that memory.  That is what the "partitioned_LDS is a nightmare" note was about, so
// refuse the combination rather than corrupt quietly.  This also keeps LDSptr's truncating divide out of
// reach: it is only inexact when sharing rounds LDS_SHUFL_BYTES off a 16-byte boundary, and variant 2 is
// its only caller.
#if VARIANT == 2
#error LDSMUL > 1 is not supported with FFT variant 2 (partial_tabMul touches the shared LDS region outside the lock)
#endif

// shufl dispatches on SBMUL * SHUFL_BYTES with branches for >= 16, == 8 and == 4 and no fallback, so a
// product of 12 would return with the data unexchanged and no diagnostic.  Only SBMUL == 3 can produce
// it.  Rejecting on LDSMUL is slightly stronger than necessary -- a call with numWG < 3 would have come
// out at SBMUL < 3 -- but numWG is a runtime argument, and a build error beats silently wrong results.
#if LDSMUL >= 3 && SHUFL_BYTES == 4
#error LDSMUL >= 3 with SHUFL_BYTES == 4 gives SBMUL * SHUFL_BYTES == 12, which no shufl branch handles
#endif

// Initialize access to LDS memory.  It may be advantageous to have independent workgroups share access to LDS memory via a lock controlling a critical section.
// This may let a kernel use less LDS memory, or have each workgroup use more LDS memory to perform fewer passes of writing and reading LDS memory.
void OVERLOAD LDSinit(void local *lds, const u32 numWG) {
  // Init semaphores to unlocked state
  if (SHARING_LDS(numWG)) {
    if (get_local_id(0) == 0) {
      volatile local int *semaphores = (volatile local int *)(((local char *) lds) + LDS_SEM_OFFSET(numWG));
      // One per sharing group, and every group that exists: the highest index in use is
      // (numWG - 1) / SBMUL, which the old numWG / SBMUL bound missed whenever SBMUL did not divide numWG.
      for (u32 i = 0; i < LDS_GROUPS(numWG); i++) semaphores[i] = 0;
    }
    bar();
  }
}

// Return a pointer to the LDS memory allocated for this workgroup.  If SBMUL is greater than 1, workgroup may use additional memory by sharing
// with other workgroups and using locks to control access.
local void * OVERLOAD LDSptr(local void *lds, const u32 numWG) {
  return (local char *)lds + ((u32)get_local_id(0) / WG) * LDS_SHUFL_BYTES(numWG);
}

// Return a pointer to the LDS memory this workgroup is allowed to access when sharing with other workgroups.
local void * OVERLOAD LDSsharing_ptr(local void *lds, const u32 numWG) {
  if (!SHARING_LDS(numWG)) return LDSptr(lds, numWG);
  return (local char *)lds + ((u32)get_local_id(0) / WG / SBMUL(numWG)) * SBMUL(numWG) * LDS_SHUFL_BYTES(numWG);
}

// Wait for all of a workgroup's threads to arrive.
// NOTE: A "workgroup" is an independent group of threads doing FFT work (see WMUL in carryFused or TAIL_KERNELS=2).
void OVERLOAD LDSbar(const u32 numWG) {

  // No early return for WG <= WAVEFRONT here: bar(WG) and barsync() below both decide that for themselves,
  // by what the hardware guarantees rather than by size alone, and returning early would skip the warp
  // reconvergence and LDS fence they do on hardware that is not in lock-step.

  // If were not using semaphores to share LDS access, perform a standard bar.  The standard bar is free to implement a full bar across
  // all threads if that is more efficient than a bar across a subset of threads.
  if (!SHARING_LDS(numWG)) {
    bar(WG);
    return;
  }

  // Barrier on a subset of threads.
  barsync(numWG, WG);
}

// Start a new LDS access transaction.  This is required for sharing LDS memory with other workgroups.
// Historically, each workgroup had its own LDS area, and shufl routines performed a bar(WG) at the start of accessing LDS but not at the end.
// After calling shufl, a bar(WG) was required before next LDS memory usage.  All routines that use LDS memory OBEYED THIS PROTOCOL
// of bar(WG) before LDS use (full bar() if writing outside the workgroup's LDS area) and no bar(WG) after last use (full bar() if reading from
// outside the workgroup's LDS area).  If we're not sharing LDS access among multiple workgroups, maintain this historical implementation.
// If sharing LDS we have NEW REQUIREMENTS.  LDStx_end performs an LDSbar because workgroups write to more than just their own LDS area.  The LDSbar
// ensures the reads have completed before any future writes.  When accessing LDS memory without LDStx calls (see shufl_carries_up in carryFused and
// reverseLines in tailutil) they too must perform a bar() after the last read from LDS.
// NOTE: Pass in the original LDS pointer, not the pointer returned by LDSptr or LDSsharing_ptr.
void OVERLOAD LDStx_start(local void *lds, const u32 numWG) {
  // If each workgroup has its own LDS area, then no locks are needed to access shared memory.  Use the historical model of requiring a barrier before LDS access.
  if (!SHARING_LDS(numWG)) {
    LDSbar(numWG);
    return;
  }
  // Have first thread in a workgroup lock the semaphore controlling access to LDS memory
  if (get_local_id(0) % WG == 0) {
    volatile local int *semaphores = (volatile local int *)(((local char *) lds) + LDS_SEM_OFFSET(numWG));

    // Lock semaphore (set to one) to gain access to critical section
    // Spin until the semaphore was observed unlocked: cmpxchg returns the old value, so anything other
    // than 0 means the lock was not taken.  Testing for 1 alone would let any other value through here
    // without the lock, and LDStx_end would then clear a semaphore this workgroup never owned.
    while (atomic_cmpxchg(&semaphores[get_local_id(0) / WG / SBMUL(numWG)], 0, 1) != 0);
  }
  LDSbar(numWG);
}

// End an LDS access transaction
// NOTE: Pass in the original LDS pointer, not the pointer returned by LDSptr or LDSsharing_ptr.
void OVERLOAD LDStx_end(local void *lds, const u32 numWG) {
  // Historically, no trailing LDSbar is required when not sharing LDS memory.
  if (!SHARING_LDS(numWG)) return;

  // Since we are sharing LDS areas among multiple workgroups, we must wait for all of a workgroup's threads to finish their LDS access.
  LDSbar(numWG);
  // Unlock the semaphore
  if (get_local_id(0) % WG == 0) {
    volatile local int *semaphores = (volatile local int *)(((local char *) lds) + LDS_SEM_OFFSET(numWG));
    semaphores[get_local_id(0) / WG / SBMUL(numWG)] = 0;
  }
}

#endif


#define INCLUDE_FILE "shufl.cl"
#include "expand.cl"

#if FFT_FP64

void OVERLOAD chainMul4(T2 *u, T2 w) {
  u[1] = cmul(u[1], w);

  T2 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  base = ccubeTrig(base, w);
  u[3] = cmul(u[3], base);
}

#if 1
// This version of chainMul8 tries to minimize roundoff error even if more F64 ops are used.
// Trial and error looking at Z values on a WIDTH=512 FFT was used to determine when to switch from fancy to non-fancy powers of w.
void OVERLOAD chainMul8(T2 *u, T2 w) {
  u[1] = cmulFancy(u[1], w);

  T2 w2 = csqTrigFancy(w);
  u[2] = cmulFancy(u[2], w2);

  T2 w3;
  // Rocm optimizer behaves weirdly yet again. Using mul2 instead of 2.0* makes double-wide single-kernel tailSquare slower.
  // Yes, mul2 saves an FP64 op, but the compiler no longer saves the computed powers of w from the first fft_HEIGHT call
  // for use in the second fft_HEIGHT call for a large net increase in FP64 ops.
  if (DOING_WIDTH || VARIANT != 0) {
    w3 = ccubeTrigFancy(w2, w);
  } else {
    double a = 2*w2.y;
    w3 = U2(fma(a, -w.y, w.x), fma(a, w.x, a - w.y));
  }
  u[3] = cmulFancy(u[3], w3);

  w3.x += 1;
  T2 base = cmulFancy(w3, w);
  for (int i = 4; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmulFancy(base, w);
  }
}

#else
// This version of chainMul8 minimizes F64 ops even if that increases roundoff error.
// This version is faster on a Radeon 7 with worse roundoff.  However, FFT_width is even faster with better roundoff.
// This version is the same speed on a TitanV probably due to its great F64 throughput.
// This version is slower on R7Pro due to a rocm optimizer issue in double-wide single-kernel tailSquare using BCAST.  I could not find a work-around.
// Other GPUs???  This version might be useful.  If we decide to make this available, it will need a new width and height fft spec number.
// Consequently, an increase in the BPW table and increase work for -ztune and -tune.
void OVERLOAD chainMul8(T2 *u, T2 w) {
  u[1] = cmulFancy(u[1], w);

  T2 w2 = csqTrigFancy(w);
  u[2] = cmulFancy(u[2], w2);

  T2 w3 = ccubeTrigDefancy(w2, w);
  u[3] = cmul(u[3], w3);

  T2 w4 = csqTrigDefancy(w2);
  u[4] = cmul(u[4], w4);

  T2 w6 = csqTrig(w3);
  T2 w5, w7; cmul_a_by_fancyb_and_conjfancyb(&w7, &w5, w6, w);
  u[5] = cmul(u[5], w5);
  u[6] = cmul(u[6], w6);
  u[7] = cmul(u[7], w7);
}
#endif

void OVERLOAD chainMul(T2 *u, T2 w) {
  // Do a length 4 chain mul, w must not be in Fancy format
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul, w must be in Fancy format
  if (RADIX == 8) chainMul8(u, w);
}


#if AMDGPU && VARIANT == 0

int bcast4(int x)  { return __builtin_amdgcn_mov_dpp(x, 0, 0xf, 0xf, false); }
int bcast8(int x)  { return __builtin_amdgcn_ds_swizzle(x, 0x0018); }
int bcast16(int x) { return __builtin_amdgcn_ds_swizzle(x, 0x0010); }
int bcast64(int x) { return __builtin_amdgcn_readfirstlane(x); }

int bcastAux(int x, u32 span) {
  return span == 4 ? bcast4(x) : span == 8 ? bcast8(x) : span == 16 ? bcast16(x) : span == 64 ? bcast64(x) : x;
}

T2 bcast(T2 src, u32 span) {
  int4 s = as_int4(src);
  for (int i = 0; i < 4; ++i) { s[i] = bcastAux(s[i], span); }
  return as_double2(s);
}

#elif NVIDIAGPU && CUDA_BACKEND && VARIANT == 0

// CUDA warps are 32 lanes, so shfl.sync.idx (__shfl_sync) directly covers spans 4/8/16/32 -- the same
// per-segment broadcast as AMD's mov_dpp/ds_swizzle -- but there is no single-instruction equivalent of
// AMD's 64-wide readfirstlane (a CUDA warp cannot broadcast past its own 32 lanes).  Only reachable for
// WG > 32; not needed for the WIDTH/SMALL_HEIGHT=512, NW/NH=8 (WG=64, spans 1 and 8 only) config this
// was tested with -- trap loudly instead of silently returning the wrong (unbroadcast) value.
int bcast64(int x) { __trap(); return x; }

int bcastAux(int x, u32 span) {
  return span == 4 ? __shfl_sync(0xffffffffu, x, 0, 4)
       : span == 8 ? __shfl_sync(0xffffffffu, x, 0, 8)
       : span == 16 ? __shfl_sync(0xffffffffu, x, 0, 16)
       : span == 32 ? __shfl_sync(0xffffffffu, x, 0, 32)
       : span == 64 ? bcast64(x)
       : x;
}

T2 bcast(T2 src, u32 span) {
  int4 s = as_int4(src);
  // Unlike OpenCL's int4, CUDA's native int4 struct has no operator[]; index through
  // a plain int* instead of s.x/.y/.z/.w so bcastAux can still be a loop over 0..3.
  int* p = (int*)&s;
  for (int i = 0; i < 4; ++i) { p[i] = bcastAux(p[i], span); }
  return as_double2(s);
}

#else

// Variants 1 and 2 never broadcast: fft_common selects chainMul(u, w = bcast(w, s)) with a runtime
// `if (VARIANT == 0)`, so the call has to compile even though it is dead.  Return the input unchanged.
T2 bcast(T2 src, u32 span) { return src; }

#endif

void OVERLOAD fft_RADIX(T2 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 5
  fft5(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

// For FUSE_WEIGHT_BUTTERFLY.  fft_RADIX, but for the very first radix transform of a WIDTH-transform invocation whose caller (carryFused) has
// already performed that transform's first butterfly's adds/subs, fusing a forward weight multiply via FMA (see carryfused.cl).
void OVERLOAD fft_RADIX_skip1(T2 *u) {
#if RADIX == 4
  fft4_skip1(u);
#elif RADIX == 8
  fft8_skip1(u);
#else
#error FUSE_WEIGHT_BUTTERFLY not implemented for this RADIX
#endif
}

void OVERLOAD tabMul(Trig trig, T2 *u, u32 f, u32 me) {
#if 0
  u32 p = me / f * f;
#else
  u32 p = me & ~(f - 1);
#endif

// Compute trigs from scratch every time.  This can't possibly be a good idea on any GPUs.
#if 0
  T2 w = slowTrig_N(ND / RADIX / WG * p, ND / RADIX);
  T2 base = w;
  for (int i = 1; i < RADIX; ++i) {
    u[i] = cmul(u[i], w);
    w = cmul(w, base);
  }
  return;
#endif

// This code uses chained complex multiplies which could be faster on GPUs with great DP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Apparently, chained Fancy muls at these short n=4 and n=8 lengths are very accurate.

  if (TABMUL_CHAIN) {
    T2 w = TFLOAD(&trig[p]);
    chainMul(u, w);
    return;
  }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

  if (!TABMUL_CHAIN) {
    T2 w = TFLOAD(&trig[p]);

    if (RADIX >= 8) {
      u[1] = cmulFancy(u[1], w);
    } else {
      u[1] = cmul(u[1], w);
    }

    for (u32 i = 2; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

// Tabmul after doing an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4a(Trig trig, T2 *u, u32 f, u32 me) {

  if (f == 1) {                      // fft8_4 is performed first
    u32 p = me;

// This code uses chained complex multiplies which could be faster on GPUs with great DP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Perform two length=4 chain muls.

    if (TABMUL_CHAIN) {
      T2 w  = TFLOAD(&trig[p]);
      T2 w2 = TFLOAD(&trig[WG + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      T2 base  = csqTrig(w);
      T2 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

    if (!TABMUL_CHAIN) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG + p]));
      }
    }
  }

  else {                      // fft8_4 is performed after an initial fft8

// This code uses chained complex multiplies which could be faster on GPUs with great DP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Perform two length=4 chain muls.

    u32 p = me / 8;                 // Generate index into condensed trig table that does not have duplicated trig values
    trig += 7 * WG;                 // Skip over the trig values used in the first tabmul
    if (TABMUL_CHAIN) {
      T2 w  = TFLOAD(&trig[p]);
      T2 w2 = TFLOAD(&trig[WG/8 + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      T2 base  = csqTrig(w);
      T2 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

    if (!TABMUL_CHAIN) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG/8 + p]));
      }
    }
  }
}

// Later tabmuls after starting with an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4b(Trig trig, T2 *u, u32 f, u32 me) {

// This code uses chained complex multiplies which could be faster on GPUs with great DP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Apparently, chained Fancy muls at n=8 lengths are very accurate.

  if (TABMUL_CHAIN) {
    u32 p = me & ~(f - 1);
    T2 w = TFLOAD(&trig[p]);

//    u[1] = cmulFancy(u[1], w);                                // GW: - this should use Fancy, but tabmul8_4a does not and it could for half of the data
//    T2 w2 = csqTrigFancy(w);
//    u[2] = cmulFancy(u[2], w2);
//    T2 w3 = ccubeTrigFancy(w2, w);
//    u[3] = cmulFancy(u[3], w3);
//    w3.x += 1;
//    T2 base = cmulFancy(w3, w);
//    for (int i = 4; i < 8; ++i) {
//      u[i] = cmul(u[i], base);
//      base = cmulFancy(base, w);
//    }

    u[1] = cmul(u[1], w);                               // GW: - this should use Fancy, but tabmul8_4a does not and it could for half of the data
    T2 w2 = csqTrig(w);
    u[2] = cmul(u[2], w2);
    T2 w3 = ccubeTrig(w2, w);
    u[3] = cmul(u[3], w3);
    T2 base = cmul(w3, w);
    for (int i = 4; i < 8; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

  if (!TABMUL_CHAIN) {
    u32 p = (me/4) & ~(f/4 - 1);     // Generate index into condensed trig table that does not have duplicated trig values
    trig += 6 * WG;                  // Skip over the trig values used in tabmul8_4a

//GW:  Can any of these be Fancy? Yes, u[1] and u[2]
    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*(WG/4) + p]));
    }
  }
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-4 FFTs with more FMA instructions
//************************************************************************************

// Copy of macro from fft4 and fft8 with FMAs added
#define X2_via_FMA(a, c, b) { T2 t = a; a = fma(c, b, t); b = fma(-c, b, t); }

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul4_trig(Trig trig, T *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingle trig1 = (TrigSingle) trig;

  // Read 3 lines of sine/cosine values for the first fft4.  Read two of the lines as a pair as AMD likes T2 global memory reads
  Trig trig2 = (Trig) trig1;
  T2 sine_over_cosines = TFLOAD(&trig2[me]);
  preloads[0] = sine_over_cosines.x;
  preloads[1] = sine_over_cosines.y;
  // Read 3rd line
  preloads[2] = TFLOAD(&trig1[2*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul4(local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me) {
  local T *lds1 = (local T *) lds;
  TrigSingle trig1 = (TrigSingle) trig;
  trig1 += 4*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[4];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[5];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 4; ++i) {
    T sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/4) + (me/f)*(f/4)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 4; i += 2) {
      Trig trig2 = (Trig) (trig1 + i*WG);
      T2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine1, cosine2, cosine3/cosine1
    if (f < WG/4) preloads[0] = lds1[WG + ((me/f) & 3) * WG/4 + (0 * WG + me)/(4*f) * f/4];
    preloads[2] = lds1[WG + ((me/f) & 3) * WG/4 + (2 * WG + me)/(4*f) * f/4];
    preloads[3] = lds1[WG + ((me/f) & 3) * WG/4 + (3 * WG + me)/(4*f) * f/4];
    preloads[1] = lds1[WG + ((me/f) & 3) * WG/4 + (1 * WG + me)/(4*f) * f/4];
  }
}

// Finish off a partial tabMul while doing next fft4 making more use of FMA.
void finish_tabMul4_fft4(Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingle trig1 = (TrigSingle) trig;

  //
  // Mimic a traditional fft4 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/4) u[0] = u[0] * preloads[0];

  // Apply cosine2, cosine3/cosine1 to u[2] and u[3] using FMA
  X2_via_FMA(u[0], preloads[2], u[2]);
  X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);

  // Preload one line of sine/cosines and one line of cosines for later tabMuls.  We'll later broadcast these values as needed using LDS.
  if (f == 1) {
    preloads[4] = TFLOAD(&trig1[3*WG + me]);             // Sine/cosines for later tabMuls
    preloads[5] = TFLOAD(&trig1[4*WG + 4*WG + me]);      // Cosines for later tabMuls
  }

  // Do the last level of fft4 applying cosine1
  X2_via_FMA(u[0], preloads[1], u[1]);
  X2_via_FMA(u[2], preloads[1], u[3]);

  // revbin [0, 2, 1, 3] undo
  SWAP(u[1], u[2]);
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-8 FFTs with more FMA instructions
//************************************************************************************

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul8_trig(Trig trig, T *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingle trig1 = (TrigSingle) trig;

  // Read 7 lines of sine/cosine values for the first fft8.  Read six of the lines as pairs as AMD likes T2 global memory reads
  for (u32 i = 1; i < 7; i += 2) {
    Trig trig2 = (Trig) (trig1 + (i-1)*WG);
    T2 sine_over_cosines = TFLOAD(&trig2[me]);
    preloads[i-1] = sine_over_cosines.x;
    preloads[i] = sine_over_cosines.y;
  }
  // Read 7th line
  preloads[6] = TFLOAD(&trig1[6*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul8(local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me) {
  local T *lds1 = (local T *) lds;
  TrigSingle trig1 = (TrigSingle) trig;
  trig1 += 8*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[8];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[9];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 8; ++i) {
    T sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/8) + (me/f)*(f/8)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 8; i += 2) {
      Trig trig2 = (Trig) (trig1 + i*WG);
      T2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3, cosine2, cosine3/cosine1, cosine1
    // Load them in the order they will be used, though it probably won't matter.
    if (f < WG/8) preloads[0] = lds1[WG + ((me/f) & 7) * WG/8 + (0 * WG + me)/(8*f) * f/8];
    preloads[1] = lds1[WG + ((me/f) & 7) * WG/8 + (1 * WG + me)/(8*f) * f/8];
    preloads[4] = lds1[WG + ((me/f) & 7) * WG/8 + (4 * WG + me)/(8*f) * f/8];
    preloads[5] = lds1[WG + ((me/f) & 7) * WG/8 + (5 * WG + me)/(8*f) * f/8];
    preloads[6] = lds1[WG + ((me/f) & 7) * WG/8 + (6 * WG + me)/(8*f) * f/8];
    preloads[7] = lds1[WG + ((me/f) & 7) * WG/8 + (7 * WG + me)/(8*f) * f/8];
    preloads[2] = lds1[WG + ((me/f) & 7) * WG/8 + (2 * WG + me)/(8*f) * f/8];
    preloads[3] = lds1[WG + ((me/f) & 7) * WG/8 + (3 * WG + me)/(8*f) * f/8];
  }
}

// Finish off a partial tabMul while doing next fft8 making more use of FMA.
void finish_tabMul8_fft8(Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingle trig1 = (TrigSingle) trig;

  //
  // Mimic a traditional fft8 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/8) u[0] = u[0] * preloads[0];

  if (save_one_more_mul) {   // This should always be the best option.  ROCm optimizer is doing something weird in fft_WIDTH case.

    // Apply cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3/cosine1
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8 applying cosine1
//TODO: Save this MUL by SQRT(1/2) by pre-computing cosine1*SQRTHALF
    T cosine1_SQRT1_2 = preloads[1] * M_SQRT1_2;
    X2_via_FMA(u[0], preloads[1], u[1]);
    X2_via_FMA(u[2], preloads[1], u[3]);
    X2_via_FMA(u[4], cosine1_SQRT1_2, u[5]);
    X2_via_FMA(u[6], cosine1_SQRT1_2, u[7]);

  } else {

    // Apply cosine to u[1]
    u[1] = u[1] * preloads[1];

    // Apply cosine4, cosine5, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8
    X2(u[0], u[1]);
    X2(u[2], u[3]);
    X2ad(u[4], u[5], M_SQRT1_2);
    X2ad(u[6], u[7], M_SQRT1_2);
  }

  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}


void OVERLOAD fft_common(local T2 *lds, T2 *u, Trig trig, T2 w, u32 numWG, u32 lowMe, int callnum) {

  // This line mimics shufl -- partition lds for variant 2
  local T2* partitioned_lds = LDSptr(lds, numWG);

// Variant 0 uses broadcast instructions.  Only available on AMD and NVIDIA GPUs.

#if VARIANT == 0

#if WG * RADIX > 1024
#error VARIANT == 0 only supported for FFT size <= 1024
#elif NVIDIAGPU && WG * RADIX == 1024 && RADIX == 4
#error VARIANT == 0 not supported for radix-4, FFT size 1024 on nVidia GPUs
#elif !AMDGPU && !NVIDIAGPU
#error VARIANT == 0 only supported by AMD or NVIDIA GPUs
#endif

// There is a slight difference between fft_WIDTH and fft_HEIGHT.  Tail square computes the trig values
// to be broadcast, while fft_WIDTH does not.  Compute the trig values now for fft_WIDTH,
#if DOING_WIDTH
#if RADIX == 8
  w = fancyTrig_N(ND / (WG * RADIX) * lowMe);
#else
  w = slowTrig_N(ND / (WG * RADIX) * lowMe, ND / RADIX);
#endif
#endif

#endif

// Variant 2 uses more FMA instructions than the original FFT code.
// The tabMul after fft8 only does a partial complex multiply, saving a mul-by-cosine for the next fft8 using FMA instructions.
// To maximize FMA opportunities we precompute trig values as cosine and sine/cosine rather than cosine and sine.
// The downside is sine/cosine cannot be computed with chained multiplies.

// Variant 2 code for SIZE=256, RADIX=4
#if WG == 64 && RADIX == 4 && VARIANT == 2

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  if (FUSE_WEIGHT_BUTTERFLY && DOING_WIDTH && callnum == 2) fft4_skip1(u); else fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish third tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);

// Variant 2 code for SIZE=512, RADIX=8
#elif WG == 64 && RADIX == 8 && VARIANT == 2

  T preloads[10];                       // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8 + SAVE_ONE_MUL*2*WG*8;   // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  if (FUSE_WEIGHT_BUTTERFLY && DOING_WIDTH && callnum == 2) fft8_skip1(u); else fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, SAVE_ONE_MUL);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish second tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, SAVE_ONE_MUL);  // We'd rather set save_one_more_mul to 1

// Variant 2 code for SIZE=1024, RADIX=4
#elif WG == 256 && RADIX == 4 && VARIANT == 2

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  if (FUSE_WEIGHT_BUTTERFLY && DOING_WIDTH && callnum == 2) fft4_skip1(u); else fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish the third tabMul and perform fourth fft4.  Do fourth partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish fourth tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 64, numWG, lowMe, 1);

// Variant 2 code for SIZE=1024, RADIX=8, threads=128.  Same 8*8*16 structure as the plain version below,
// but the first radix-8 stage (and its tabMul) is done with the FMA-based partial/finish tabMul8 machinery.
// The second radix-8 stage keeps the plain tabMul since shufl_and_fft2 (a combined shuffle + radix-2 fft)
// isn't twiddle-aware -- it works the same regardless of how the preceding tabMul applied its twiddles --
// and the final fft8_16a/fft8_16b already has its own separate FMA-style twiddle handling.
#elif WG == 128 && RADIX == 8 && VARIANT == 2

  T preloads[10];               // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  Trig trig2 = trig + WG*8;     // Skip past old FFT_width trig values to the !save_one_more_mul trig values.  Keep trig unmodified for the second stage's plain tabMul.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig2, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  if (FUSE_WEIGHT_BUTTERFLY && DOING_WIDTH && callnum == 2) fft8_skip1(u); else fft8(u);
  partial_tabMul8(partitioned_lds, trig2, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.
  finish_tabMul8_fft8(trig2, preloads, u, 1, numWG, lowMe, 0);

  // Second radix-8 stage stays plain -- shufl_and_fft2 doesn't care how the twiddle was applied.
  tabMul(trig, u, 8, lowMe);
  shufl_and_fft2(lds, u, 8, numWG, lowMe);

  if (lowMe < WG / 2) fft8_16a(u); else fft8_16b(u);

// Variant 2 code for SIZE=4K, RADIX=8
#elif WG == 512 && RADIX == 8 && VARIANT == 2

  T preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values to the !save_one_more_mul trig values

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  if (FUSE_WEIGHT_BUTTERFLY && DOING_WIDTH && callnum == 2) fft8_skip1(u); else fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish the second tabMul and perform third fft8.  Do third partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish third tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 64, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1


// Custom code for SIZE=256, RADIX=8, threads=32.  Performed as 4 * 8 * 8.  Radix-8 allows fewer
// shufls and tabmuls than radix-4.  Fewer instructions, but more registers.
// Uses only 32 threads which is fine on nVidia, lousy on radeon VII (use WMUL=2, TAIL_KERNELS=2).
//
// Details for memory layout, trig data, and shufls:
// Mem:   0     1...  31
//        32
//        ...
//        196
//        224   ...  255
// Only do a radix-4 fft.  Non-standard TABMUL:    (64 3/4 cmuls in blocks of 1 duplicated trig values)
// trig powers are: 0*0 0*1 .. 0*31
//                  0*32    .. 0*63
//                  1*0 1*1 .. 1*31
//                  1*32    .. 1*63
//                  2*0 2*1 .. 2*31
//                  2*32    .. 2*63
//                  3*0 3*1 .. 3*31
//                  3*32    .. 3*63           total trig data (6*32*16=3KB)
// non-standard shufl out:
//        0  64 .. 192   1... 7...
//        8
//        16
//        ...
//        48
//        56
// standard TABMUL:   (8 7/8 cmuls in blocks of 4 duplicated trig values)
// trig powers are: 0000 0000 .. 0000*7
//                  0000 4444 .. 4444*7
//                  0000 8888 .. 8888*7
//                  0000 12 ...
//                  0000 16 ...
//                  0000 20 ...
//                  0000 24 ...
//                  0000 28 ...           total trig data (7*8*16=896B)
// standard shufl out:
//        0  64 .. 192   8...  56...
//        1
//        2
//        ...
//        6
//        7
//
// FP64 and FP32 could benefit by starting the next fft8 after the radix-4 tabmul (easy FMA opportunities).

// Code for SIZE=256, RADIX=8
#elif WG == 32 && NW == 8

#if FUSE_WEIGHT_BUTTERFLY && DOING_WIDTH
#error FUSE_WEIGHT_BUTTERFLY not implemented for SIZE=256, RADIX=8
#endif
#if VARIANT == 0
#error Variant 0 not implemented for SIZE=256, RADIX=8
#endif

  fft8_4(u);
  tabMul8_4a(trig, u, 1, lowMe);
  shufl(lds, u, 1, 4, numWG, lowMe);

  fft8(u);
  tabMul8_4b(trig, u, 4, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  fft8(u);

// Custom code for SIZE=1024, RADIX=8, threads=128.  Performed as 8 * 8 * 2 * 8.  Radix-8 allows fewer
// shufls and tabmuls than radix-4.  Fewer instructions, but more registers.  If we process 4 (or 8)
// independent width lines then we should be able to avoid the mul by w^0 in the next to last radix-8 step.
//
// Details for memory layout, trig data, and shufls:
// Mem:   0     1 ...  127
//        128
//        256
//        384
//        512
//        640
//        768
//        896   ...  1023
// standard TABMUL:    (128 7/8 cmuls in blocks of 1 duplicated trig values)
// trig powers are: 0 0 0  .. 0*127
//                  0 1 2  .. 1*127
//                  0 2 4  .. 2*127
//                  0 3 6  .. 3*127
//                  0 4 8  .. 4*127
//                  0 5 10 .. 5*127
//                  0 6 12 .. 6*127
//                  0 7 14 .. 7*127           total trig data (7*128*16=14KB)
// standard shufl out:
//        0   128 .. 896   1... 15...
//        16
//        32
//        48
//        64
//        80
//        96
//        112
// standard TABMUL:   (16 7/8 cmuls in blocks of 8 duplicated trig values)
// trig powers are: 00000000 00000000 .. 00000000*15
//                  00000000 11111111 .. 11111111*15
//                  00000000 22222222 .. 22222222*15
//                  00000000 33333333 .. 33333333*15
//                  00000000 44444444 .. 44444444*15
//                  00000000 55555555 .. 55555555*15
//                  00000000 66666666 .. 66666666*15
//                  00000000 77777777 .. 77777777*15           total trig data (7*16*16=1.75KB)
// shufl out with an fft2:
//        0  128 .. 896  16 .. 112...  8...
//        1
//        2
//        3
//        4
//        5
//        6
//        7

// Code for SIZE=1024, RADIX=8
#elif WG == 128 && RADIX == 8

  if (FUSE_WEIGHT_BUTTERFLY && DOING_WIDTH && callnum == 2) fft8_skip1(u); else fft8(u);
  if (VARIANT == 0) chainMul(u, w);
  else tabMul(trig, u, 1, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  fft8(u);
  if (VARIANT == 0) chainMul(u, w = bcast(w, 8));
  else tabMul(trig, u, 8, lowMe);
  shufl_and_fft2(lds, u, 8, numWG, lowMe);

  if (lowMe < WG / 2) fft8_16a(u); else fft8_16b(u);

#else

  // Old / original version

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    if (FUSE_WEIGHT_BUTTERFLY && DOING_WIDTH && callnum == 2 && s == 1) fft_RADIX_skip1(u); else fft_RADIX(u);
    if (VARIANT == 0) chainMul(u, w = bcast(w, s));
    else tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

#endif
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

void OVERLOAD fft_RADIX(F2 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

void OVERLOAD chainMul4(F2 *u, F2 w) {
  u[1] = cmul(u[1], w);

  F2 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  base = ccubeTrig(base, w);
  u[3] = cmul(u[3], base);
}

void OVERLOAD chainMul8(F2 *u, F2 w) {
  u[1] = cmulFancy(u[1], w);
                                                  //GWBUG - see FP64 version for many possible optimizations
  F2 w2 = csqTrigFancy(w);
  u[2] = cmulFancy(u[2], w2);

  F2 w3 = ccubeTrigFancy(w2, w);
  u[3] = cmulFancy(u[3], w3);

  w3.x += 1;
  F2 base = cmulFancy(w3, w);
  for (int i = 4; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmulFancy(base, w);
  }
}

void OVERLOAD chainMul(F2 *u, F2 w) {
  // Do a length 4 chain mul
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul
  if (RADIX == 8) chainMul8(u, w);
}

void OVERLOAD tabMul(TrigFP32 trig, F2 *u, u32 f, u32 me) {
  u32 p = me & ~(f - 1);

// This code uses chained complex multiplies which could be faster on GPUs with great mul throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN32) {
    chainMul(u, TFLOAD(&trig[p]));
    return;
  }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN32) {
    if (RADIX >= 8) {
      u[1] = cmulFancy(u[1], TFLOAD(&trig[p]));
    } else {
      u[1] = cmul(u[1], TFLOAD(&trig[p]));
    }
    for (u32 i = 2; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

// Tabmul after doing an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4a(TrigFP32 trig, F2 *u, u32 f, u32 me) {

  if (f == 1) {                      // fft8_4 is performed first
    u32 p = me;

// This code uses chained complex multiplies which could be faster on GPUs with great SP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Perform two length=4 chain muls.

    if (TABMUL_CHAIN32) {
      F2 w  = TFLOAD(&trig[p]);
      F2 w2 = TFLOAD(&trig[WG + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      F2 base  = csqTrig(w);
      F2 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN32) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG + p]));
      }
    }
  }

  else {                      // fft8_4 is performed after an initial fft8

// This code uses chained complex multiplies which could be faster on GPUs with great SP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Perform two length=4 chain muls.

    u32 p = me / 8;                 // Generate index into condensed trig table that does not have duplicated trig values
    trig += 7 * WG;                 // Skip over the trig values used in the first tabmul
    if (TABMUL_CHAIN32) {
      F2 w  = TFLOAD(&trig[p]);
      F2 w2 = TFLOAD(&trig[WG/8 + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      F2 base  = csqTrig(w);
      F2 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN32) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG/8 + p]));
      }
    }
  }
}

// Later tabmuls after starting with an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4b(TrigFP32 trig, F2 *u, u32 f, u32 me) {

// This code uses chained complex multiplies which could be faster on GPUs with great SP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Apparently, chained Fancy muls at n=8 lengths are very accurate.

  if (TABMUL_CHAIN32) {
    u32 p = me & ~(f - 1);
    F2 w = TFLOAD(&trig[p]);

//    u[1] = cmulFancy(u[1], w);                                // GW: - this should use Fancy, but tabmul8_4a does not and it could for half of the data
//    T2 w2 = csqTrigFancy(w);
//    u[2] = cmulFancy(u[2], w2);
//    T2 w3 = ccubeTrigFancy(w2, w);
//    u[3] = cmulFancy(u[3], w3);
//    w3.x += 1;
//    T2 base = cmulFancy(w3, w);
//    for (int i = 4; i < 8; ++i) {
//      u[i] = cmul(u[i], base);
//      base = cmulFancy(base, w);
//    }

    u[1] = cmul(u[1], w);                               // GW: - this should use Fancy, but tabmul8_4a does not and it could for half of the data
    F2 w2 = csqTrig(w);
    u[2] = cmul(u[2], w2);
    F2 w3 = ccubeTrig(w2, w);
    u[3] = cmul(u[3], w3);
    F2 base = cmul(w3, w);
    for (int i = 4; i < 8; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN32) {
    u32 p = (me/4) & ~(f/4 - 1);     // Generate index into condensed trig table that does not have duplicated trig values
    trig += 6 * WG;                  // Skip over the trig values used in tabmul8_4a

//GW:  Can any of these be Fancy? Yes, u[1] and u[2]
    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*(WG/4) + p]));
    }
  }
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-4 FFTs with more FMA instructions
//************************************************************************************

// Some OpenCL compilers are having trouble with fma on floats.  Specifically, line "X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);".
// Since we're not enabling FP32 variant 2 by default, don't include these "more FMA" routines.

#if ENABLE_FP32_VARIANT_2

// Copy of macro from fft4 and fft8 with FMAs added
#define X2_via_FMA(a, c, b) { F2 t = a; a = fma(c, b, t); b = fma(-c, b, t); }

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul4_trig(TrigFP32 trig, F *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  // Read 3 lines of sine/cosine values for the first fft4.  Read two of the lines as a pair as AMD likes T2 global memory reads
  TrigFP32 trig2 = (TrigFP32) trig1;
  F2 sine_over_cosines = TFLOAD(&trig2[me]);
  preloads[0] = sine_over_cosines.x;
  preloads[1] = sine_over_cosines.y;
  // Read 3rd line
  preloads[2] = TFLOAD(&trig1[2*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul4(local F2 *lds, TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me) {
  local F *lds1 = (local F *) lds;
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;
  trig1 += 4*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[4];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[5];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 4; ++i) {
    F sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/4) + (me/f)*(f/4)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 4; i += 2) {
      TrigFP32 trig2 = (TrigFP32) (trig1 + i*WG);
      F2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine1, cosine2, cosine3/cosine1
    if (f < WG/4) preloads[0] = lds1[WG + ((me/f) & 3) * WG/4 + (0 * WG + me)/(4*f) * f/4];
    preloads[2] = lds1[WG + ((me/f) & 3) * WG/4 + (2 * WG + me)/(4*f) * f/4];
    preloads[3] = lds1[WG + ((me/f) & 3) * WG/4 + (3 * WG + me)/(4*f) * f/4];
    preloads[1] = lds1[WG + ((me/f) & 3) * WG/4 + (1 * WG + me)/(4*f) * f/4];
  }
}

// Finish off a partial tabMul while doing next fft4 making more use of FMA.
void finish_tabMul4_fft4(TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  //
  // Mimic a traditional fft4 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/4) u[0] = u[0] * preloads[0];

  // Apply cosine2, cosine3/cosine1 to u[2] and u[3] using FMA
  X2_via_FMA(u[0], preloads[2], u[2]);
  X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);

  // Preload one line of sine/cosines and one line of cosines for later tabMuls.  We'll later broadcast these values as needed using LDS.
  if (f == 1) {
    preloads[4] = TFLOAD(&trig1[3*WG + me]);             // Sine/cosines for later tabMuls
    preloads[5] = TFLOAD(&trig1[4*WG + 4*WG + me]);      // Cosines for later tabMuls
  }

  // Do the last level of fft4 applying cosine1
  X2_via_FMA(u[0], preloads[1], u[1]);
  X2_via_FMA(u[2], preloads[1], u[3]);

  // revbin [0, 2, 1, 3] undo
  SWAP(u[1], u[2]);
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-8 FFTs with more FMA instructions
//************************************************************************************

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul8_trig(TrigFP32 trig, F *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  // Read 7 lines of sine/cosine values for the first fft8.  Read six of the lines as pairs as AMD likes T2 global memory reads
  for (u32 i = 1; i < 7; i += 2) {
    TrigFP32 trig2 = (TrigFP32) (trig1 + (i-1)*WG);
    F2 sine_over_cosines = TFLOAD(&trig2[me]);
    preloads[i-1] = sine_over_cosines.x;
    preloads[i] = sine_over_cosines.y;
  }
  // Read 7th line
  preloads[6] = TFLOAD(&trig1[6*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul8(local F2 *lds, TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me) {
  local F *lds1 = (local F *) lds;
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;
  trig1 += 8*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[8];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[9];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 8; ++i) {
    F sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/8) + (me/f)*(f/8)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 8; i += 2) {
      TrigFP32 trig2 = (TrigFP32) (trig1 + i*WG);
      F2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3, cosine2, cosine3/cosine1, cosine1
    // Load them in the order they will be used, though it probably won't matter.
    if (f < WG/8) preloads[0] = lds1[WG + ((me/f) & 7) * WG/8 + (0 * WG + me)/(8*f) * f/8];
    preloads[1] = lds1[WG + ((me/f) & 7) * WG/8 + (1 * WG + me)/(8*f) * f/8];
    preloads[4] = lds1[WG + ((me/f) & 7) * WG/8 + (4 * WG + me)/(8*f) * f/8];
    preloads[5] = lds1[WG + ((me/f) & 7) * WG/8 + (5 * WG + me)/(8*f) * f/8];
    preloads[6] = lds1[WG + ((me/f) & 7) * WG/8 + (6 * WG + me)/(8*f) * f/8];
    preloads[7] = lds1[WG + ((me/f) & 7) * WG/8 + (7 * WG + me)/(8*f) * f/8];
    preloads[2] = lds1[WG + ((me/f) & 7) * WG/8 + (2 * WG + me)/(8*f) * f/8];
    preloads[3] = lds1[WG + ((me/f) & 7) * WG/8 + (3 * WG + me)/(8*f) * f/8];
  }
}

// Finish off a partial tabMul while doing next fft8 making more use of FMA.
void finish_tabMul8_fft8(TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  //
  // Mimic a traditional fft8 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/8) u[0] = u[0] * preloads[0];

  if (save_one_more_mul) {   // This should always be the best option.  ROCm optimizer is doing something weird in fft_WIDTH case.

    // Apply cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3/cosine1
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8 applying cosine1
//TODO: Save this MUL by SQRT(1/2) by pre-computing cosine1*SQRTHALF
    F cosine1_SQRT1_2 = preloads[1] * (float) M_SQRT1_2;
    X2_via_FMA(u[0], preloads[1], u[1]);
    X2_via_FMA(u[2], preloads[1], u[3]);
    X2_via_FMA(u[4], cosine1_SQRT1_2, u[5]);
    X2_via_FMA(u[6], cosine1_SQRT1_2, u[7]);

  } else {

    // Apply cosine to u[1]
    u[1] = u[1] * preloads[1];

    // Apply cosine4, cosine5, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8
    X2(u[0], u[1]);
    X2(u[2], u[3]);
    X2ad(u[4], u[5], M_SQRT1_2);
    X2ad(u[6], u[7], M_SQRT1_2);
  }

  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif

// Variant 2 code uses more FMA instructions than the original fft version.
// The tabMul after fft8 only does a partial complex multiply, saving a mul-by-cosine for the next fft8 using FMA instructions.
// To maximize FMA opportunities we precompute trig values as cosine and sine/cosine rather than cosine and sine.
// The downside is sine/cosine cannot be computed with chained multiplies.

void OVERLOAD fft_common(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, u32 lowMe, int callnum) {

  // This line mimics shufl -- partition lds
  local F2* partitioned_lds = LDSptr(lds, numWG);

// Variant 2 code for SIZE=256, RADIX=4
#if ENABLE_FP32_VARIANT_2 && WG == 64 && RADIX == 4 && VARIANT == 2

  F preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish third tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);

// Variant 2 code for SIZE=512, RADIX=8
#elif ENABLE_FP32_VARIANT_2 && WG == 64 && RADIX == 8 && VARIANT == 2

  F preloads[10];                        // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8 + SAVE_ONE_MUL*2*WG*8;    // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, SAVE_ONE_MUL);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish second tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, SAVE_ONE_MUL);

// Variant 2 code for SIZE=1024, RADIX=4
#elif ENABLE_FP32_VARIANT_2 && WG == 256 && RADIX == 4 && VARIANT == 2

  F preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish the third tabMul and perform fourth fft4.  Do fourth partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish fourth tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 64, numWG, lowMe, 1);

// Variant 2 code for SIZE=4K, RADIX=8
#elif ENABLE_FP32_VARIANT_2 && WG == 512 && RADIX == 8 && VARIANT == 2

  F preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values to the !save_one_more_mul trig values

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish the second tabMul and perform third fft8.  Do third partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish third tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 64, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1

// Code for SIZE=256, RADIX=8
#elif WG == 32 && NW == 8

  fft8_4(u);
  tabMul8_4a(trig, u, 1, lowMe);
  shufl(lds, u, 1, 4, numWG, lowMe);

  fft8(u);
  tabMul8_4b(trig, u, 4, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  fft8(u);

// Code for SIZE=1024, RADIX=8
#elif WG == 128 && RADIX == 8

  fft8(u);
  tabMul(trig, u, 1, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  fft8(u);
  tabMul(trig, u, 8, lowMe);
  shufl_and_fft2(lds, u, 8, numWG, lowMe);

  if (lowMe < WG / 2) fft8_16a(u); else fft8_16b(u);

#else

  // Old / original version

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

#endif
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD fft_RADIX(GF31 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

void OVERLOAD chainMul4(GF31 *u, GF31 w) {
  u[1] = cmul(u[1], w);

  GF31 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  base = ccubeTrig(base, w);
  u[3] = cmul(u[3], base);
}

void OVERLOAD chainMul8(GF31 *u, GF31 w) {
  u[1] = cmul(u[1], w);

  GF31 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  base = ccubeTrig(base, w);
  for (int i = 3; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmul(base, w);
  }
}

void OVERLOAD chainMul(GF31 *u, GF31 w) {
  // Do a length 4 chain mul
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul
  if (RADIX == 8) chainMul8(u, w);
}

void OVERLOAD tabMul(TrigGF31 trig, GF31 *u, u32 f, u32 me) {
  u32 p = me & ~(f - 1);

// This code uses chained complex multiplies which could be faster on GPUs with great mul throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN31) {
    chainMul(u, TFLOAD(&trig[p]));
    return;
  }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN31) {
    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

// Tabmul after doing an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4a(TrigGF31 trig, GF31 *u, u32 f, u32 me) {

  if (f == 1) {                      // fft8_4 is performed first
    u32 p = me;

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.
// Perform two length=4 chain muls.

    if (TABMUL_CHAIN31) {
      GF31 w  = TFLOAD(&trig[p]);
      GF31 w2 = TFLOAD(&trig[WG + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      GF31 base  = csqTrig(w);
      GF31 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN31) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG + p]));
      }
    }
  }

  else {                      // fft8_4 is performed after an initial fft8

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.
// Perform two length=4 chain muls.

    u32 p = me / 8;                 // Generate index into condensed trig table that does not have duplicated trig values
    trig += 7 * WG;                 // Skip over the trig values used in the first tabmul
    if (TABMUL_CHAIN31) {
      GF31 w  = TFLOAD(&trig[p]);
      GF31 w2 = TFLOAD(&trig[WG/8 + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      GF31 base  = csqTrig(w);
      GF31 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN31) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG/8 + p]));
      }
    }
  }
}

// Later tabmuls after starting with an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4b(TrigGF31 trig, GF31 *u, u32 f, u32 me) {

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN31) {
    u32 p = me & ~(f - 1);
    GF31 w = TFLOAD(&trig[p]);

    u[1] = cmul(u[1], w);
    GF31 w2 = csqTrig(w);
    u[2] = cmul(u[2], w2);
    GF31 w3 = ccubeTrig(w2, w);
    u[3] = cmul(u[3], w3);
    GF31 base = cmul(w3, w);
    for (int i = 4; i < 8; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN31) {
    u32 p = (me/4) & ~(f/4 - 1);     // Generate index into condensed trig table that does not have duplicated trig values
    trig += 6 * WG;                  // Skip over the trig values used in tabmul8_4a

    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*(WG/4) + p]));
    }
  }
}

void OVERLOAD fft_common(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, u32 lowMe) {

// Code for SIZE=256, RADIX=8
#if WG == 32 && NW == 8

  fft8_4(u);
  tabMul8_4a(trig, u, 1, lowMe);
  shufl(lds, u, 1, 4, numWG, lowMe);

  fft8(u);
  tabMul8_4b(trig, u, 4, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  fft8(u);

// Code for SIZE=1024, RADIX=8
#elif WG == 128 && RADIX == 8

  fft8(u);
  tabMul(trig, u, 1, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  fft8(u);
  tabMul(trig, u, 8, lowMe);
  shufl_and_fft2(lds, u, 8, numWG, lowMe);

  if (lowMe < WG / 2) fft8_16a(u); else fft8_16b(u);

#else

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

#endif

}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD fft_RADIX(GF61 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

void OVERLOAD chainMul4(GF61 *u, GF61 w) {
  u[1] = cmul(u[1], w);

  GF61 base = csq(w);                   //GWBUG - see FP64 version for possible optimization
  u[2] = cmul(u[2], base);

  base = cmul(base, w);                 //GWBUG - see FP64 version for possible optimization
  u[3] = cmul(u[3], base);
}

void OVERLOAD chainMul8(GF61 *u, GF61 w) {
  u[1] = cmul(u[1], w);

  GF61 w2 = csq(w);
  u[2] = cmul(u[2], w2);

  GF61 base = cmul(w2, w);              //GWBUG - see FP64 version for many possible optimizations
  for (int i = 3; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmul(base, w);
  }
}

void OVERLOAD chainMul(GF61 *u, GF61 w) {
  // Do a length 4 chain mul
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul
  if (RADIX == 8) chainMul8(u, w);
}

void OVERLOAD tabMul(TrigGF61 trig, GF61 *u, u32 f, u32 me) {
  u32 p = me & ~(f - 1);

// This code uses chained complex multiplies which could be faster on GPUs with great mul throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN61) {
    chainMul(u, TFLOAD(&trig[p]));
    return;
  }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN61) {
    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

// Tabmul after doing an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4a(TrigGF61 trig, GF61 *u, u32 f, u32 me) {

  if (f == 1) {                      // fft8_4 is performed first
    u32 p = me;

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.
// Perform two length=4 chain muls.

    if (TABMUL_CHAIN61) {
      GF61 w  = TFLOAD(&trig[p]);
      GF61 w2 = TFLOAD(&trig[WG + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      GF61 base  = csqTrig(w);
      GF61 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN61) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG + p]));
      }
    }
  }

  else {                      // fft8_4 is performed after an initial fft8

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.
// Perform two length=4 chain muls.

    u32 p = me / 8;                 // Generate index into condensed trig table that does not have duplicated trig values
    trig += 7 * WG;                 // Skip over the trig values used in the first tabmul
    if (TABMUL_CHAIN61) {
      GF61 w  = TFLOAD(&trig[p]);
      GF61 w2 = TFLOAD(&trig[WG/8 + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      GF61 base  = csqTrig(w);
      GF61 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN61) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG/8 + p]));
      }
    }
  }
}

// Later tabmuls after starting with an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4b(TrigGF61 trig, GF61 *u, u32 f, u32 me) {

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN61) {
    u32 p = me & ~(f - 1);
    GF61 w = TFLOAD(&trig[p]);

    u[1] = cmul(u[1], w);
    GF61 w2 = csqTrig(w);
    u[2] = cmul(u[2], w2);
    GF61 w3 = ccubeTrig(w2, w);
    u[3] = cmul(u[3], w3);
    GF61 base = cmul(w3, w);
    for (int i = 4; i < 8; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

  if (!TABMUL_CHAIN61) {
    u32 p = (me/4) & ~(f/4 - 1);     // Generate index into condensed trig table that does not have duplicated trig values
    trig += 6 * WG;                  // Skip over the trig values used in tabmul8_4a

    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*(WG/4) + p]));
    }
  }
}

void OVERLOAD fft_common(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, u32 lowMe) {

// Code for SIZE=256, RADIX=8
#if WG == 32 && NW == 8

  fft8_4(u);
  tabMul8_4a(trig, u, 1, lowMe);
  shufl(lds, u, 1, 4, numWG, lowMe);

  fft8(u);
  tabMul8_4b(trig, u, 4, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  fft8(u);

// Code for SIZE=1024, RADIX=8
#elif WG == 128 && RADIX == 8

  fft8(u);
  tabMul(trig, u, 1, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  fft8(u);
  tabMul(trig, u, 8, lowMe);
  shufl_and_fft2(lds, u, 8, numWG, lowMe);

  if (lowMe < WG / 2) fft8_16a(u); else fft8_16b(u);

#else

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

#endif

}

#endif
