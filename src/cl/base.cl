// Copyright (C) Mihai Preda and George Woltman

#pragma once

/* Tunable paramaters for -ctune :

IN_WG, OUT_WG: 64, 128, 256. Default: 128.
IN_SIZEX, OUT_SIZEX: 4, 8, 16, 32. Default: 16.
UNROLL_W: 0, 1. Default: 0 on AMD, 1 on Nvidia.
UNROLL_H: 0, 1. Default: 1.
*/

/* List of code-specific macros. These are set by the C++ host code or derived
EXP        the exponent
WIDTH
SMALL_HEIGHT
MIDDLE
CARRY_LEN
NW
NH
AMDGPU  : if this is an AMD GPU
NVIDIAGPU : if this is an nVidia GPU
HAS_ASM : set if we believe __asm() can be used for AMD GCN
HAS_PTX : set if we believe __asm() can be used for nVidia PTX

-- Derived from above:
BIG_HEIGHT == SMALL_HEIGHT * MIDDLE
ND         number of dwords == WIDTH * MIDDLE * SMALL_HEIGHT
NWORDS     number of words  == ND * 2
G_W        "group width"  == WIDTH / NW
G_H        "group height" == SMALL_HEIGHT / NH
 */

#define STR(x) XSTR(x)
#define XSTR(x) #x

#define OVERLOAD __attribute__((overloadable))

#pragma OPENCL FP_CONTRACT ON

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

// 64-bit atomics are not used ATM
// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#if DEBUG
#define assert(condition) if (!(condition)) { printf("assert(%s) failed at line %d\n", STR(condition), __LINE__ - 1); }
// __builtin_trap();
#else
#define assert(condition)
//__builtin_assume(condition)
#endif // DEBUG

#ifndef AMDGPU
#define AMDGPU 0
#endif
#ifndef NVIDIAGPU
#define NVIDIAGPU 0
#endif

#if NO_ASM
#define HAS_ASM 0
#define HAS_PTX 0
#elif AMDGPU
#define HAS_ASM 1
#define HAS_PTX 0
#elif NVIDIAGPU
#define HAS_ASM 0
#define HAS_PTX 1200        // Assume CUDA 12.00 support until we can figure out how to automatically determine this at runtime
#else
#define HAS_ASM 0
#define HAS_PTX 0
#endif

// Default is not adding -2 to results for LL
#if !defined(LL)
#define LL 0
#endif

// On Nvidia we need the old sync between groups in carryFused
#if !defined(OLD_FENCE) && !AMDGPU
#define OLD_FENCE 1
#endif

// Nontemporal reads and writes might be a little bit faster on many GPUs by keeping more reusable data in the caches.
// However, on those GPUs with large caches there should be a significant speed gain from keeping FFT data in the caches.
// Default to the big win when caching is beneficial rather than the tiny gain when non-temporal is better.
#if !defined(NONTEMPORAL)
#define NONTEMPORAL 0
#endif

// FFT variant is in 3 parts.  One digit for WIDTH, one digit for MIDDLE, one digit for HEIGHT.
// For WIDTH and HEIGHT there are 3 variants:
// 0   compute one trig, bcast, chainmul                                        previously was :even/:odd BCAST=1
// 1   if TABMUL_CHAIN, read one trig then chainmul                             previously was :0/:1
//     if !TABMUL_CHAIN, read all trigs, no chainmul                            previously was :2/:3
// 2   read all trigs in sin/cos format for more FMA                            previously was :2/:3 UNROLL_W=3
// Note: smaller numbers above do more F64 and are less accurate, larger numbers have more memory accesses and are more accurate
// For MIDDLE there are two variants:
// 0   full length chainmul
// 1   lots of computing trigs, very short chainmul for maximum accuracy        previously was :1/:3
#define FFT_VARIANT_W    (FFT_VARIANT / 100)
#define FFT_VARIANT_M    (FFT_VARIANT % 100 / 10)
#define FFT_VARIANT_H    (FFT_VARIANT % 10)
#if FFT_VARIANT_W > 2
#error FFT_VARIANT_W must be between 0 and 2
#endif
#if FFT_VARIANT_M > 1
#error FFT_VARIANT_M must be between 0 and 1
#endif
#if FFT_VARIANT_H > 2
#error FFT_VARIANT_H must be between 0 and 2
#endif
// C code ensures that only AMD GPUs use FFT_VARIANT_W=0 and FFT_VARIANT_H=0.  However, this does not guarantee that the OpenCL compiler supports
// the necessary amdgcn builtins.  If those builtins are not present convert to variant one.
#if AMDGPU
#if !defined(__has_builtin) || !__has_builtin(__builtin_amdgcn_mov_dpp) || !__has_builtin(__builtin_amdgcn_ds_swizzle) || !__has_builtin(__builtin_amdgcn_readfirstlane)
#if FFT_VARIANT_W == 0
#warning Missing builtins for FFT_VARIANT_W=0, switching to FFT_VARIANT_W=1
#undef FFT_VARIANT_W
#define FFT_VARIANT_W 1
#endif
#if FFT_VARIANT_H == 0
#warning Missing builtins for FFT_VARIANT_H=0, switching to FFT_VARIANT_H=1
#undef FFT_VARIANT_H
#define FFT_VARIANT_H 1
#endif
#endif
#endif

// Shufl width in bytes (can be 4, 8, or 16).  See fftbase.cl.  Allow different shufl widths for fft_width and fft_height.
// Default is 8 bytes (one double).  Historically best for Radeon VII and TitanV.  This setting will affect how much LDS
// memory is needed which in turn may affect occupancy and thus performance.
#if !defined(SHUFL_BYTES_W)
#define SHUFL_BYTES_W 8
#endif
#if !defined(SHUFL_BYTES_H)
#define SHUFL_BYTES_H 8
#endif

#if !defined(TABMUL_CHAIN)
#define TABMUL_CHAIN 0
#endif
#if !defined(TABMUL_CHAIN31)
#define TABMUL_CHAIN31 0
#endif
#if !defined(TABMUL_CHAIN32)
#define TABMUL_CHAIN32 0
#endif
#if !defined(TABMUL_CHAIN61)
#define TABMUL_CHAIN61 0
#endif
#if !defined(MODM31)
#define MODM31 0
#endif

#if !defined(MIDDLE_CHAIN)
#define MIDDLE_CHAIN 0
#endif

#if !defined(UNROLL_W)
#if AMDGPU
#define UNROLL_W 0
#else
#define UNROLL_W 1
#endif
#endif

#if !defined(UNROLL_H)
#if AMDGPU && (SMALL_HEIGHT >= 1024)
#define UNROLL_H 0
#else
#define UNROLL_H 1
#endif
#endif

#if !defined(ZEROHACK_W)
#define ZEROHACK_W 1
#endif

#if !defined(ZEROHACK_H)
#define ZEROHACK_H 1
#endif

// Expected defines: EXP the exponent.
// WIDTH, SMALL_HEIGHT, MIDDLE.

#define BIG_HEIGHT (SMALL_HEIGHT * MIDDLE)
#define ND (WIDTH * BIG_HEIGHT)
#define NWORDS (ND * 2u)

#if (NW != 4 && NW != 8) || (NH != 4 && NH != 8)
#error NW and NH must be passed in, expected value 4 or 8.
#endif

#define G_W (WIDTH / NW)
#define G_H (SMALL_HEIGHT / NH)

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;

// Data types for data stored in FFTs and NTTs during the transform
typedef double T;           // For historical reasons, classic FFTs using doubles call their data T and T2.
typedef double2 T2;         // A complex value using doubles in a classic FFT.
typedef float F;            // A classic FFT using floats.  Use typedefs F and F2.
typedef float2 F2;
typedef uint Z31;           // A value calculated mod M31.  For a GF(M31^2) NTT.
typedef uint2 GF31;         // A complex value using two Z31s.  For a GF(M31^2) NTT.
typedef ulong Z61;          // A value calculated mod M61.  For a GF(M61^2) NTT.
typedef ulong2 GF61;        // A complex value using two Z61s.  For a GF(M61^2) NTT.
//typedef ulong NCW;          // A value calculated mod 2^64 - 2^32 + 1.
//typedef ulong2 NCW2;        // A complex value using NCWs.  For a Nick Craig-Wood's insipred NTT using prime 2^64 - 2^32 + 1.

// Defines for the various supported FFTs/NTTs.  These match the enumeration in FFTConfig.h.  Sanity check for supported FFT/NTT.
#define FFT64           0
#define FFT3161         1
#define FFT3261         2
#define FFT61           3
#define FFT323161       4
#define FFT3231         50
#define FFT6431         51
#define FFT31           52
#define FFT32           53
#if FFT_TYPE < 0 || (FFT_TYPE > 4 && FFT_TYPE < 50) || FFT_TYPE > 53
#error - unsupported FFT/NTT
#endif
// Word and Word2 define the data type for FFT integers passed between the CPU and GPU.
#if WordSize == 8
typedef i64 Word;
typedef long2 Word2;
#elif WordSize == 4
typedef i32 Word;
typedef int2 Word2;
#else
error - unsupported integer WordSize
#endif

// Routine to create a pair
double2 OVERLOAD U2(double a, double b) { return (double2) (a, b); }
float2 OVERLOAD U2(float a, float b) { return (float2) (a, b); }
int2 OVERLOAD U2(int a, int b) { return (int2) (a, b); }
long2 OVERLOAD U2(long a, long b) { return (long2) (a, b); }
uint2 OVERLOAD U2(uint a, uint b) { return (uint2) (a, b); }
ulong2 OVERLOAD U2(ulong a, ulong b) { return (ulong2) (a, b); }

// Other handy macros
#define RE(a) (a.x)
#define IM(a) (a.y)

#define P(x) global x * restrict
#define CP(x) const P(x)

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void


// For reasons unknown, loading trig values into nVidia's constant cache has terrible performance
#if AMDGPU
typedef constant const T2* Trig;
typedef constant const T* TrigSingle;
typedef constant const F2* TrigFP32;
typedef constant const GF31* TrigGF31;
typedef constant const GF61* TrigGF61;
#else
typedef global const T2* Trig;
typedef global const T* TrigSingle;
typedef global const F2* TrigFP32;
typedef global const GF31* TrigGF31;
typedef global const GF61* TrigGF61;
#endif
// However, caching weights in nVidia's constant cache improves performance.
// Even better is to not pollute the constant cache with weights that are used only once.
// This requires two typedefs depending on how we want to use the BigTab pointer.
// For AMD we can declare BigTab as constant or global - it doesn't really matter.
typedef constant const double2* ConstBigTab;
typedef constant const float2* ConstBigTabFP32;
#if AMDGPU
typedef constant const double2* BigTab;
typedef constant const float2* BigTabFP32;
#else
typedef global const double2* BigTab;
typedef global const float2* BigTabFP32;
#endif

//
// nVidia GPUs have lots of different caching options for loads and stores.
// AMD GPUs have have far fewer options for loads and stores.
// These routines let us try the different options.
//

// Basic load and store.  Presumably stored in all caches using a standard LRU algorithm.

#define LOAD(mem)        *(mem)
#define STORE(mem,val)   *(mem) = val

// Non-temporal load and store.

#if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
#define NTLOAD(mem)        __builtin_nontemporal_load(mem)
#else
#define NTLOAD           LOAD
#endif

#if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_store)
#define NTSTORE(mem,val)   __builtin_nontemporal_store(val, mem)
#else
#define NTSTORE          STORE
#endif

// Routines for loading data from memory into the L2 cache but not the L1 cache.

#if HAS_PTX >= 200         // Cache hints requires sm_20 support or higher
T2 OVERLOAD L2LOAD(CP(T2) mem) {
  T2 retval;
  __asm("ld.global.cg.v2.f64  {%0, %1}, [%2];" : "=d"(retval.x), "=d"(retval.y) : "l"(mem));
  return retval;
}
T OVERLOAD L2LOAD(TrigSingle mem) {
  T retval;
  __asm("ld.global.cg.f64  %0, [%1];" : "=d"(retval) : "l"(mem));
  return retval;
}
F2 OVERLOAD L2LOAD(CP(F2) mem) {
  F2 retval;
  __asm("ld.global.cg.v2.f32  {%0, %1}, [%2];" : "=f"(retval.x), "=f"(retval.y) : "l"(mem));
  return retval;
}
i64 OVERLOAD L2LOAD(i64 *mem) {
  i64 retval;
  __asm("ld.global.cg.b64  %0, [%1];" : "=l"(retval) : "l"(mem));
  return retval;
}
GF61 OVERLOAD L2LOAD(TrigGF61 mem) {
  GF61 retval;
  __asm("ld.global.cg.v2.b64  {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(mem));
  return retval;
}
i32 OVERLOAD L2LOAD(i32 *mem) {
  i32 retval;
  __asm("ld.global.cg.b32  %0, [%1];" : "=r"(retval) : "l"(mem));
  return retval;
}
GF31 OVERLOAD L2LOAD(TrigGF31 mem) {
  GF31 retval;
  __asm("ld.global.cg.v2.b32  {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(mem));
  return retval;
}
#else
#define L2LOAD     LOAD
#endif

// Routines for storing to L2 cache bypassing L1 cache.

#if HAS_PTX >= 200        // Cache hints requires sm_20 support or higher
void OVERLOAD L2STORE(P(T2) mem, T2 val) {
  __asm("st.global.cg.v2.f64  [%0], {%1, %2};" : : "l"(mem), "d"(val.x), "d"(val.y));
}
void OVERLOAD L2STORE(P(F2) mem, F2 val) {
  __asm("st.global.cg.v2.f32  [%0], {%1, %2};" : : "l"(mem), "f"(val.x), "f"(val.y));
}
void OVERLOAD L2STORE(i64 *mem, i64 val) {
  __asm("st.global.cg.b64  [%0], %1;" : : "l"(mem), "l"(val));
}
void OVERLOAD L2STORE(i32 *mem, i32 val) {
  __asm("st.global.cg.b32  [%0], %1;" : : "l"(mem), "r"(val));
}
#else
#define L2STORE    STORE
#endif

// Routines for loading data from memory into the L1 and L2 caches, but cache line is marked evict first.

#if HAS_PTX >= 200         // Cache hints requires sm_20 support or higher
T2 OVERLOAD EFLOAD(CP(T2) mem) {
  T2 retval;
  __asm("ld.global.cs.v2.f64  {%0, %1}, [%2];" : "=d"(retval.x), "=d"(retval.y) : "l"(mem));
  return retval;
}
T OVERLOAD EFLOAD(TrigSingle mem) {
  T retval;
  __asm("ld.global.cs.f64  %0, [%1];" : "=d"(retval) : "l"(mem));
  return retval;
}
F2 OVERLOAD EFLOAD(CP(F2) mem) {
  F2 retval;
  __asm("ld.global.cs.v2.f32  {%0, %1}, [%2];" : "=f"(retval.x), "=f"(retval.y) : "l"(mem));
  return retval;
}
i64 OVERLOAD EFLOAD(i64 *mem) {
  i64 retval;
  __asm("ld.global.cs.b64  %0, [%1];" : "=l"(retval) : "l"(mem));
  return retval;
}
GF61 OVERLOAD EFLOAD(TrigGF61 mem) {
  GF61 retval;
  __asm("ld.global.cs.v2.b64  {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(mem));
  return retval;
}
i32 OVERLOAD EFLOAD(i32 *mem) {
  i32 retval;
  __asm("ld.global.cs.b32  %0, [%1];" : "=r"(retval) : "l"(mem));
  return retval;
}
GF31 OVERLOAD EFLOAD(TrigGF31 mem) {
  GF31 retval;
  __asm("ld.global.cs.v2.b32  {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(mem));
  return retval;
}
#else
#define EFLOAD    LOAD
#endif

// Routines for storing to L1 and L2 caches with cache line marked evict first.

#if HAS_PTX >= 200        // Cache hints requires sm_20 support or higher
void OVERLOAD EFSTORE(P(T2) mem, T2 val) {
  __asm("st.global.cs.v2.f64  [%0], {%1, %2};" : : "l"(mem), "d"(val.x), "d"(val.y));
}
void OVERLOAD EFSTORE(P(F2) mem, F2 val) {
  __asm("st.global.cs.v2.f32  [%0], {%1, %2};" : : "l"(mem), "f"(val.x), "f"(val.y));
}
void OVERLOAD EFSTORE(i64 *mem, i64 val) {
  __asm("st.global.cs.b64  [%0], %1;" : : "l"(mem), "l"(val));
}
void OVERLOAD EFSTORE(i32 *mem, i32 val) {
  __asm("st.global.cs.b32  [%0], %1;" : : "l"(mem), "r"(val));
}
#else
#define EFSTORE   STORE
#endif

// Routines for loading a value and marking it for "last use".

#if HAS_PTX >= 200         // Cache hints requires sm_20 support or higher
T2 OVERLOAD LULOAD(Trig mem) {
  T2 retval;
  __asm("ld.global.lu.v2.f64  {%0, %1}, [%2];" : "=d"(retval.x), "=d"(retval.y) : "l"(mem));
  return retval;
}
T OVERLOAD LULOAD(TrigSingle mem) {
  T retval;
  __asm("ld.global.lu.f64  %0, [%1];" : "=d"(retval) : "l"(mem));
  return retval;
}
F2 OVERLOAD LULOAD(TrigFP32 mem) {
  F2 retval;
  __asm("ld.global.lu.v2.f32  {%0, %1}, [%2];" : "=f"(retval.x), "=f"(retval.y) : "l"(mem));
  return retval;
}
i64 OVERLOAD LULOAD(i64 *mem) {
  i64 retval;
  __asm("ld.global.lu.b64  %0, [%1];" : "=l"(retval) : "l"(mem));
  return retval;
}
GF61 OVERLOAD LULOAD(TrigGF61 mem) {
  GF61 retval;
  __asm("ld.global.lu.v2.b64  {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(mem));
  return retval;
}
i32 OVERLOAD LULOAD(i32 *mem) {
  i32 retval;
  __asm("ld.global.lu.b32  %0, [%1];" : "=r"(retval) : "l"(mem));
  return retval;
}
GF31 OVERLOAD LULOAD(TrigGF31 mem) {
  GF31 retval;
  __asm("ld.global.lu.v2.b32  {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(mem));
  return retval;
}
#else
#define LULOAD    LOAD
#endif

// Routines for loading a read-only and placing it in the non-coherent texture cache.

#if HAS_PTX >= 500         // Texture cache requires sm_50 support or higher
T2 OVERLOAD NCLOAD(Trig mem) {
  T2 retval;
  __asm("ld.global.nc.v2.f64  {%0, %1}, [%2];" : "=d"(retval.x), "=d"(retval.y) : "l"(mem));
  return retval;
}
T OVERLOAD NCLOAD(TrigSingle mem) {
  T retval;
  __asm("ld.global.nc.f64  %0, [%1];" : "=d"(retval) : "l"(mem));
  return retval;
}
F2 OVERLOAD NCLOAD(TrigFP32 mem) {
  F2 retval;
  __asm("ld.global.nc.v2.f32  {%0, %1}, [%2];" : "=f"(retval.x), "=f"(retval.y) : "l"(mem));
  return retval;
}
i64 OVERLOAD NCLOAD(i64 *mem) {
  i64 retval;
  __asm("ld.global.nc.b64  %0, [%1];" : "=l"(retval) : "l"(mem));
  return retval;
}
GF61 OVERLOAD NCLOAD(TrigGF61 mem) {
  GF61 retval;
  __asm("ld.global.nc.v2.b64  {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(mem));
  return retval;
}
i32 OVERLOAD NCLOAD(i32 *mem) {
  i32 retval;
  __asm("ld.global.nc.b32  %0, [%1];" : "=r"(retval) : "l"(mem));
  return retval;
}
GF31 OVERLOAD NCLOAD(TrigGF31 mem) {
  GF31 retval;
  __asm("ld.global.lu.v2.b32  {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(mem));
  return retval;
}
#else
#define NCLOAD    LOAD
#endif

//
//  These macros map various types of data accesses to one of the load/store routines above
//

// Routines for loading/storing FFT data.  Lots of data, kernels read it once, write it once.  If possible, data should not be written to L1 cache.
// If L2 cache is "small", we should look for ways to prioritize keeping data that is re-used in the L2 cache.

#define FFTLOAD_TYPE     LOADS % 10
#define CSLOAD_TYPE      (LOADS / 10) % 10
#define TFLOAD_TYPE      (LOADS / 100) % 10
#define TSLOAD_TYPE      (LOADS / 1000) % 10
#define TOLOAD_TYPE      (LOADS / 10000) % 10

#define FFTSTORE_TYPE     STORES % 10
#define CSSTORE_TYPE      (STORES / 10) % 10

#if FFTLOAD_TYPE == 1
#define FFTLOAD    NTLOAD
#elif FFTLOAD_TYPE == 2
#define FFTLOAD    L2LOAD
#elif FFTLOAD_TYPE == 3
#define FFTLOAD    EFLOAD
#elif FFTLOAD_TYPE == 4
#define FFTLOAD    LULOAD
#else
#define FFTLOAD    LOAD
#endif

#if FFTSTORE_TYPE == 1
#define FFTSTORE   NTSTORE
#elif FFTSTORE_TYPE == 2
#define FFTSTORE   L2STORE
#elif FFTSTORE_TYPE == 3
#define FFTSTORE   EFSTORE
#else
#define FFTSTORE   STORE
#endif

// Routines for loading/storing carryShuttle data.  CarryFused writes it once, and reads it once.  The data is never used again.
// If possible, data should not be written to L1 cache and not written to memory after it is read.

#if CSLOAD_TYPE == 1
#define CSLOAD    NTLOAD
#elif CSLOAD_TYPE == 2
#define CSLOAD    L2LOAD
#elif CSLOAD_TYPE == 3
#define CSLOAD    EFLOAD
#elif CSLOAD_TYPE == 4
#define CSLOAD    LULOAD
#else
#define CSLOAD    LOAD
#endif

#if CSSTORE_TYPE == 1
#define CSSTORE   NTSTORE
#elif CSSTORE_TYPE == 2
#define CSSTORE   L2STORE
#elif CSSTORE_TYPE == 3
#define CSSTORE   EFSTORE
#else
#define CSSTORE   STORE
#endif

// Routines for loading trig data that is frequently reused.  If possible, data should saved in L1 and L2 caches and perhaps marked evict last.
// TF stands for "Trig Frequently reused".  It is highly unlikely that any option other than the default LOAD makes sense.

#if TFLOAD_TYPE == 1
#define TFLOAD    NTLOAD
#elif TFLOAD_TYPE == 2
#define TFLOAD    L2LOAD
#elif TFLOAD_TYPE == 3
#define TFLOAD    EFLOAD
#elif TFLOAD_TYPE == 4
#define TFLOAD    LULOAD
#elif TFLOAD_TYPE == 5
#define TFLOAD    NCLOAD
#else
#define TFLOAD    LOAD
#endif

// Routines for loading trig data that is used once but is smaller than a cache line.  The rest of the cache line will be needed soon.
// If possible, data should saved in L1(?) and L2 caches and perhaps marked evict first.
// TS stands for "Trig Several reuses".

#if TSLOAD_TYPE == 1
#define TSLOAD    NTLOAD
#elif TSLOAD_TYPE == 2
#define TSLOAD    L2LOAD
#elif TSLOAD_TYPE == 3
#define TSLOAD    EFLOAD
#elif TSLOAD_TYPE == 4
#define TSLOAD    LULOAD
#elif TSLOAD_TYPE == 5
#define TSLOAD    NCLOAD
#else
#define TSLOAD    LOAD
#endif

// Routines for loading trig data that is used once and is a cache line or larger.
// If possible, data should saved in L2 caches if the L2 cache is very large.
// TO stands for "Trig used Once".

#if TOLOAD_TYPE == 1
#define TOLOAD    NTLOAD
#elif TOLOAD_TYPE == 2
#define TOLOAD    L2LOAD
#elif TOLOAD_TYPE == 3
#define TOLOAD    EFLOAD
#elif TOLOAD_TYPE == 4
#define TOLOAD    LULOAD
#elif TOLOAD_TYPE == 5
#define TOLOAD    NCLOAD
#else
#define TOLOAD    LOAD
#endif

// Prefetch macros.  Unused at present, I tried using them in fftMiddleInGF61 on a 5080 with no benefit.
void PREFETCHL1(const __global void *addr) {
#if HAS_PTX >= 200         // Prefetch instruction requires sm_20 support or higher
  __asm("prefetch.global.L1  [%0];" : : "l"(addr));
#endif
}
void PREFETCHL2(const __global void *addr) {
#if HAS_PTX >= 200         // Prefetch instruction requires sm_20 support or higher
  __asm("prefetch.global.L2  [%0];" : : "l"(addr));
#endif
}

#if FFT_FP64
void OVERLOAD read(u32 WG, u32 N, T2 *u, const global T2 *in, u32 base) {
  in += base + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { u[i] = in[i * WG]; }
}

void OVERLOAD write(u32 WG, u32 N, T2 *u, global T2 *out, u32 base) {
  out += base + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { out[i * WG] = u[i]; }
}
#endif

#if FFT_FP32
void OVERLOAD read(u32 WG, u32 N, F2 *u, const global F2 *in, u32 base) {
  in += base + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { u[i] = in[i * WG]; }
}

void OVERLOAD write(u32 WG, u32 N, F2 *u, global F2 *out, u32 base) {
  out += base + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { out[i * WG] = u[i]; }
}
#endif

#if NTT_GF31
void OVERLOAD read(u32 WG, u32 N, GF31 *u, const global GF31 *in, u32 base) {
  in += base + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { u[i] = in[i * WG]; }
}

void OVERLOAD write(u32 WG, u32 N, GF31 *u, global GF31 *out, u32 base) {
  out += base + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { out[i * WG] = u[i]; }
}
#endif

#if NTT_GF61
void OVERLOAD read(u32 WG, u32 N, GF61 *u, const global GF61 *in, u32 base) {
  in += base + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { u[i] = in[i * WG]; }
}

void OVERLOAD write(u32 WG, u32 N, GF61 *u, global GF61 *out, u32 base) {
  out += base + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { out[i * WG] = u[i]; }
}
#endif

// On "classic" AMD GCN GPUs such as Radeon VII, the wavefront size was always 64. On RDNA GPUs the wavefront can
// be configured to be either 64 or 32. We use the FAST_BARRIER define as an indicator for GCN GPUs.
// On Nvidia GPUs the wavefront size is 32.
#if !WAVEFRONT
#if FAST_BARRIER && AMDGPU
#define WAVEFRONT 64
#else
#define WAVEFRONT 32
#endif
#endif

void OVERLOAD bar(void) {
  // barrier(CLK_LOCAL_MEM_FENCE) is correct, but it turns out that on some GPUs
  // (in particular on Radeon VII and Radeon PRO VII) barrier(0) works as well and is faster.
  // So allow selecting the faster path when it works with -use FAST_BARRIER
#if FAST_BARRIER
  barrier(0);
#else
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
}

void OVERLOAD bar(const u32 WG) {
  if (WG > WAVEFRONT) {
#if ENABLE_BARSYNC && HAS_PTX >= 200         // bar.sync with thread count requires sm_20 support or higher.  Slower on TitanV, need to try on later nVidia GPUs.
    __asm("bar.sync %0, %1;" : : "r"(get_local_id(0) / WG + 1), "n"(WG));
#else
    bar();
#endif
  }
}

// A half-barrier is only needed when half-a-workgroup needs a barrier.
// This is used e.g. by the double-wide tailSquare, where LDS is split between the halves.
void halfBar() { if (get_enqueued_local_size(0) / 2 > WAVEFRONT) { bar(); } }

