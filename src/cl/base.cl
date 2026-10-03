// Copyright (C) Mihai Preda and George Woltman

#pragma once

/* Tunable paramaters for -ctune :

IN_WG, OUT_WG: 64, 128, 256. Default: 128.
IN_SIZEX, OUT_SIZEX: 4, 8, 16, 32. Default: 16.
UNROLL_W: 0, 1.  1 = fully unroll fft_WIDTH's radix loop (variants 0 and 1, FP32, NTTs), 0 = never unroll it.  Default: 0 on AMD, 1 on Nvidia.
UNROLL_H: 0, 1.  Same for fft_HEIGHT.  Default: 1 (0 on AMD for SMALL_HEIGHT >= 1024).
HOIST_W, HOIST_H: 0..3.  Limits how early the compiler may compute fft_WIDTH's / fft_HEIGHT's per-radix-step LDS addresses, twiddle loads
  and (variant 0) twiddle power chains.  Hoisting these to the start of the kernel can cost many VGPRs.  0 = compiler decides.
  1 = not before the start of their radix step (variant 2: their stage, i.e. after the previous shufl).
  2 = additionally, twiddle loads and power chains not before that step's butterflies (applied in tabMul and chainMul), and
      variant 2's next-stage preloads not before the point partial_tabMul / finish_tabMul issue them.
  3 = variant 2 only: additionally, load the cosines after the shufl instead of keeping them live across it.  Default: 0.
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
HAS_ASM : set if we believe __asm() can be used for AMD GCN -- pretty much deprecated, we use amdgcn_builtins instead
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

#pragma clang diagnostic ignored "-Wconstant-logical-operand"

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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < CC
#define HAS_PTX __CUDA_ARCH__     // The CUDA shim fell back to an older compute_XY than the GPU: the PTX must be valid for it
#else
#define HAS_PTX CC     // C code computed the nVidia GPU's compute capability
#endif
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

// The default is the in-place FFT data layout for nVidia GPUs, not in-place otherwise.
// This must match the in_place default in clDefines() in Gpu.cpp.
#if !defined(INPLACE)
#if NVIDIAGPU
#define INPLACE 1
#else
#define INPLACE 0
#endif
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
// C code ensures that only AMD and nVidia GPUs use FFT_VARIANT_W=0 and FFT_VARIANT_H=0.  However, this does not
// guarantee that the OpenCL compiler supports the necessary amdgcn builtins (AMD) or that the device is new enough
// for shfl.sync (nVidia, needs sm_30, i.e. HAS_PTX>=300).  If not, convert to variant one.
#if AMDGPU || NVIDIAGPU
#if (AMDGPU && (!defined(__has_builtin) || !__has_builtin(__builtin_amdgcn_mov_dpp) || !__has_builtin(__builtin_amdgcn_ds_swizzle) || !__has_builtin(__builtin_amdgcn_readfirstlane))) \
 || (NVIDIAGPU && HAS_PTX < 300)
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

// Shufl can pad (or swizzle) to avoid LDS bank conflicts.  See fftbase.cl.  You would think this option would be good (or bad)
// for both fft_width and fft_height, but the rocm optimizer is super-finicky.  Default to using LDS padding.
#if !defined(LDSPAD_W)
#define LDSPAD_W 1
#endif
#if !defined(LDSPAD_H)
#define LDSPAD_H 1
#endif

// By default, LDS access is not shared among workgroups.
#if !defined(LDSMUL_W)
#define LDSMUL_W 1
#endif
#if !defined(LDSMUL_H)
#define LDSMUL_H 1
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

#if !defined(HOIST_W)
#define HOIST_W 0
#endif
#if !defined(HOIST_H)
#define HOIST_H 0
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

#if !defined(MULTI_Q)
#define MULTI_Q 0
#endif

#if !defined(L2_STRIPING)
#define L2_STRIPING 0
#endif

// Expected defines: EXP the exponent.
// WIDTH, SMALL_HEIGHT, MIDDLE.

#define BIG_HEIGHT (SMALL_HEIGHT * MIDDLE)
#define ND (WIDTH * BIG_HEIGHT)
#define NWORDS (ND * 2u)
#define NWORDS_IS_POWER_OF_TWO  !(NWORDS & (NWORDS - 1))

#if (NW != 4 && NW != 8) || (NH != 4 && NH != 8)
#error NW and NH must be passed in, expected value 4 or 8.
#endif

#define G_W (WIDTH / NW)
#define G_H (SMALL_HEIGHT / NH)

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;

// PFA: MIDDLE = 3 * PFA_M2 (PFA_M2 = 1, 2 or 4) on a pure-NTT type is done as a Good-Thomas prime-factor transform.  The ND pairs
// of words are split into 3 rows of PFA_L pairs by the Chinese remainder theorem: pair p is in row p % 3 at binary index p % PFA_L.
// Each row is an ordinary power-of-two transform of WIDTH * PFA_M2 * SMALL_HEIGHT pairs (width, middle radix PFA_M2, height).  The
// radix-3 between the rows (fftMiddleIn/Out) uses only a cube root of unity, which lies in Z/pZ, and no twiddles.  The rows are
// stored in the stock MIDDLE layout: line g holds row g % 3 at binary line g % PFA_BH, i.e. at the binary indices x * PFA_BH + g % PFA_BH.
// Then the carry out of the pair at (x, g) goes to (x, g + 1), except that lines PFA_BH and 2*PFA_BH (like line 0) take their
// carries from column x - 1.  The tail pairs each row with itself: line kx + PFA_TW*k3 (row frequency k3, kx < PFA_TW) pairs with
// (PFA_TW - kx) + PFA_TW*k3.
#if !defined(PFA)
#define PFA 0
#endif
#if PFA
#if (MIDDLE != 3 && MIDDLE != 6 && MIDDLE != 12) || FFT_FP64 || FFT_FP32
#error PFA needs MIDDLE=3, 6 or 12 and a pure NTT type
#endif
#define PFA_M2 (MIDDLE / 3)                     // The power-of-two part of MIDDLE
#define PFA_BH (SMALL_HEIGHT * PFA_M2)          // The lines (BIG_HEIGHT) of a row
#define PFA_TW (WIDTH * PFA_M2)                 // The tail lines of a row
#define PFA_L (WIDTH * PFA_BH)                  // The pairs of a row
// The logical pair stored at column x of line g
u32 pfaPair(u32 x, u32 g) {
  u32 q = x * PFA_BH + g % PFA_BH;
  return q + PFA_L * ((g % 3 + 3 - q % 3) * (PFA_L % 3) % 3);     // PFA_L % 3 is its own inverse mod 3
}
// Does line g take its carries from column x - 1 of the previous line?
bool pfaRotatedLine(u32 g) { return g % PFA_BH == 0; }
// The u line of the g-th tail pair, numbering the pairs kx = 0..PFA_TW/2-1 (or 1..PFA_TW/2-1 without the self-paired lines) of each k3.
u32 pfaTailLine(u32 g, bool withSelfPairs) {
  u32 n = withSelfPairs ? PFA_TW / 2 : PFA_TW / 2 - 1;
  return g / n * PFA_TW + g % n + (withSelfPairs ? 0 : 1);
}
u32 pfaTailPartner(u32 line) { u32 kx = line % PFA_TW; return line - kx + (kx ? PFA_TW - kx : PFA_TW / 2); }
// Where a tail line's trig values are, numbering the lines kx = 0..PFA_TW/2 of each k3 (see genSmallTrigComboGF61)
u32 pfaTailTrigIndex(u32 line) { return line / PFA_TW * (PFA_TW / 2 + 1) + line % PFA_TW; }
#define TAIL_PARTNER(line)      pfaTailPartner(line)
#define TAIL_TRIG_LINE(line)    pfaTailTrigIndex(line)
#define TAIL_SELF_PAIRED(line)  ((line) % PFA_TW == 0)
#else
// Stock tail: line pairs with H - line, lines 0 and H/2 pair with themselves (H = WIDTH * MIDDLE)
#define TAIL_PARTNER(line)      ((line) ? WIDTH * MIDDLE - (line) : WIDTH * MIDDLE / 2)
#define TAIL_TRIG_LINE(line)    (line)
#define TAIL_SELF_PAIRED(line)  ((line) == 0)
#endif

// 2 is a primitive q-th root of unity mod Mq, so the NWORDS-th root of two used by the IBDWT weights is 2^(NWORDS^-1 mod q).
// For a power-of-two NWORDS that is 2^(2^(q-1) / NWORDS) as 2^(q-1) == 1 mod q.  With PFA, NWORDS is three times a power of two
// and 3^-1 mod q is 41 for q = 61 and 21 for q = 31.
#if PFA
#define LOG2_ROOT_TWO61 ((u32) ((((1ULL << 60) / (NWORDS / 3)) % 61) * 41 % 61))
#define LOG2_ROOT_TWO31 ((u32) ((((1ULL << 30) / (NWORDS / 3)) % 31) * 21 % 31))
#else
#define LOG2_ROOT_TWO61 ((u32) (((1ULL << 60) / NWORDS) % 61))
#define LOG2_ROOT_TWO31 ((u32) (((1ULL << 30) / NWORDS) % 31))
#endif
// log2 of MIDDLE's share of the NTT's output scale.  With PFA the factor 3 is removed in fftMiddleOut, see PFA_INV3_*, leaving PFA_M2.
#define LOG2_MIDDLE_NTT (PFA ? (MIDDLE == 12 ? 2 : MIDDLE == 6 ? 1 : 0) : MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4)

// The host sets NO_FP64 for devices without cl_khr_fp64 (e.g. Mesa rusticl on AMD).  Such devices can still run the FFT types
// that have no FP64 data (M31+M61, M61, and the FP32 hybrids).  All uses of double must then be compiled out.
#if !defined(NO_FP64)
#define NO_FP64 0
#endif
#if NO_FP64 && FFT_FP64
#error FFT types with FP64 data need a device with cl_khr_fp64
#endif

// Data types for data stored in FFTs and NTTs during the transform
#if !NO_FP64
typedef double T;           // For historical reasons, classic FFTs using doubles call their data T and T2.
typedef double2 T2;         // A complex value using doubles in a classic FFT.
#else
typedef ulong T;            // Kernels also use T2 pointers as untyped buffers (and trig tables) for NTT data.  Without FP64,
typedef ulong2 T2;          // a same-size integer type stands in.
#endif
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

// The FP64 FFT can save a few FP64 ops by applying some of the weights using FMA.  nVidia compilers are clever enough to do this automatically.
// AMD's rocm compiler needs us to do this explicitly (see carryfused.cl's precompute and fft_common's use of it below).  Only wired up
// for FFT_TYPE==FFT64 with fft_WIDTH's RADIX 4 or 8 (fft4_skip1 / fft8_skip1 in fft4.cl / fft8.cl); not the 32-thread
// WIDTH=256 special case (WIDTH==256, NW==8, fft8_4-based, see fft_common).
#if !defined(FUSE_WEIGHT_BUTTERFLY)
#if AMDGPU && FFT_TYPE == FFT64 && !(WIDTH == 256 && NW == 8)
#define FUSE_WEIGHT_BUTTERFLY 1
#else
#define FUSE_WEIGHT_BUTTERFLY 0
#endif
#endif

// The fused precompute above only exists in the FFT64 carryFused (see carryfused.cl).  An explicit
// -use FUSE_WEIGHT_BUTTERFLY=1 override on any other FFT type would skip the first WIDTH butterfly
// without ever applying its weight, silently corrupting the result.
#if FUSE_WEIGHT_BUTTERFLY && FFT_TYPE != FFT64
#error FUSE_WEIGHT_BUTTERFLY is only implemented for FFT_TYPE == FFT64
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
#if !NO_FP64
double2 OVERLOAD U2(double a, double b) { return (double2) (a, b); }
#endif
float2 OVERLOAD U2(float a, float b) { return (float2) (a, b); }
int2 OVERLOAD U2(int a, int b) { return (int2) (a, b); }
long2 OVERLOAD U2(i64 a, i64 b) { return (long2) (a, b); }
uint2 OVERLOAD U2(uint a, uint b) { return (uint2) (a, b); }
ulong2 OVERLOAD U2(unsigned long a, unsigned long b) { return (ulong2) ((ulong)a, (ulong)b); }              // Two versions dealing with longs to handle TAILTGF61 constant
#if !NO_INT128                // Without 128-bit integers (see KernelCompiler.cpp) there are no long long literals to handle
ulong2 OVERLOAD U2(unsigned long long a, unsigned long long b) { return (ulong2) ((ulong)a, (ulong)b); }
#endif

// Other handy macros
#define RE(a) (a.x)
#define IM(a) (a.y)

#define P(x) global x * restrict
#define CP(x) const P(x)

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

// AMD only: Gpu.cpp can pass -DAMD_WAVES_PER_EU=n (ask for at least n waves per SIMD) or -DAMD_NUM_VGPR=n (explicit VGPR count), either of which caps
// the kernel's VGPR usage.  Used to avoid the one-wave-per-SIMD occupancy cliff (more than 128 VGPRs on gfx9).  See Gpu::amdRegisterOption.
#if AMDGPU && defined(AMD_NUM_VGPR)
#define KERNEL_CAP(x) kernel __attribute__((reqd_work_group_size(x, 1, 1), amdgpu_num_vgpr(AMD_NUM_VGPR))) void
#elif AMDGPU && defined(AMD_WAVES_PER_EU)
#define KERNEL_CAP(x) kernel __attribute__((reqd_work_group_size(x, 1, 1), amdgpu_waves_per_eu(AMD_WAVES_PER_EU))) void
#else
#define KERNEL_CAP(x) KERNEL(x)
#endif

// ENABLE_RESTRICT=1 marks the trig and weight table pointers below as restrict.  That lets the compiler hoist their loads (on nVidia they become ld.global.nc),
// which is sometimes faster but can cost many more registers.  Off by default.
#ifndef ENABLE_RESTRICT
#define ENABLE_RESTRICT 0
#endif
#if ENABLE_RESTRICT
#define TABLE_RESTRICT restrict
#else
#define TABLE_RESTRICT
#endif

// For reasons unknown, loading trig values into nVidia's constant cache has terrible performance
#if AMDGPU
typedef constant const T2* TABLE_RESTRICT Trig;
typedef constant const T* TABLE_RESTRICT TrigSingle;
typedef constant const F2* TABLE_RESTRICT TrigFP32;
typedef constant const F* TABLE_RESTRICT TrigSingleFP32;
typedef constant const GF31* TABLE_RESTRICT TrigGF31;
typedef constant const GF61* TABLE_RESTRICT TrigGF61;
#else
typedef global const T2* TABLE_RESTRICT Trig;
typedef global const T* TABLE_RESTRICT TrigSingle;
typedef global const F2* TABLE_RESTRICT TrigFP32;
typedef global const F* TABLE_RESTRICT TrigSingleFP32;
typedef global const GF31* TABLE_RESTRICT TrigGF31;
typedef global const GF61* TABLE_RESTRICT TrigGF61;
#endif
// However, caching weights in nVidia's constant cache improves performance.
// Even better is to not pollute the constant cache with weights that are used only once.
// This requires two typedefs depending on how we want to use the BigTab pointer.
// For AMD we can declare BigTab as constant or global - it doesn't really matter.
#if !NO_FP64
typedef constant const double2* TABLE_RESTRICT ConstBigTab;
#endif
typedef constant const float2* TABLE_RESTRICT ConstBigTabFP32;
#if AMDGPU
#if !NO_FP64
typedef constant const double2* TABLE_RESTRICT BigTab;
#endif
typedef constant const float2* TABLE_RESTRICT BigTabFP32;
#else
#if !NO_FP64
typedef global const double2* TABLE_RESTRICT BigTab;
#endif
typedef global const float2* TABLE_RESTRICT BigTabFP32;
#endif

//
// nVidia GPUs have lots of different caching options for loads and stores.
// AMD GPUs have have far fewer options for loads and stores.
// These routines and macros let us try the different options.
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
F OVERLOAD L2LOAD(TrigSingleFP32 mem) {
  F retval;
  __asm("ld.global.cg.f32  %0, [%1];" : "=f"(retval) : "l"(mem));
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
void OVERLOAD L2STORE(P(GF61) mem, GF61 val) {
  __asm("st.global.cg.v2.b64  [%0], {%1, %2};" : : "l"(mem), "l"(val.x), "l"(val.y));
}
void OVERLOAD L2STORE(P(GF31) mem, GF31 val) {
  __asm("st.global.cg.v2.b32  [%0], {%1, %2};" : : "l"(mem), "r"(val.x), "r"(val.y));
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

// Routines for loading data from memory into the L1 and L2 caches, but cache line is marked evict first to limit cache pollution.

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
F OVERLOAD EFLOAD(TrigSingleFP32 mem) {
  F retval;
  __asm("ld.global.cs.f32  %0, [%1];" : "=f"(retval) : "l"(mem));
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
void OVERLOAD EFSTORE(P(GF61) mem, GF61 val) {
  __asm("st.global.cs.v2.b64  [%0], {%1, %2};" : : "l"(mem), "l"(val.x), "l"(val.y));
}
void OVERLOAD EFSTORE(P(GF31) mem, GF31 val) {
  __asm("st.global.cs.v2.b32  [%0], {%1, %2};" : : "l"(mem), "r"(val.x), "r"(val.y));
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
F OVERLOAD LULOAD(TrigSingleFP32 mem) {
  F retval;
  __asm("ld.global.lu.f32  %0, [%1];" : "=f"(retval) : "l"(mem));
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

// Routines for loading a read-only value and placing it in the non-coherent texture cache.

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
F OVERLOAD NCLOAD(TrigSingleFP32 mem) {
  F retval;
  __asm("ld.global.nc.f32  %0, [%1];" : "=f"(retval) : "l"(mem));
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
  __asm("ld.global.nc.v2.b32  {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(mem));
  return retval;
}
#else
#define NCLOAD    LOAD
#endif

// Routines for loading data from memory into the L1 and L2 caches.  This should be same as the default LOAD macro.

#if HAS_PTX >= 200         // Cache hints requires sm_20 support or higher
T2 OVERLOAD CALOAD(CP(T2) mem) {
  T2 retval;
  __asm("ld.global.ca.v2.f64  {%0, %1}, [%2];" : "=d"(retval.x), "=d"(retval.y) : "l"(mem));
  return retval;
}
T OVERLOAD CALOAD(TrigSingle mem) {
  T retval;
  __asm("ld.global.ca.f64  %0, [%1];" : "=d"(retval) : "l"(mem));
  return retval;
}
F2 OVERLOAD CALOAD(CP(F2) mem) {
  F2 retval;
  __asm("ld.global.ca.v2.f32  {%0, %1}, [%2];" : "=f"(retval.x), "=f"(retval.y) : "l"(mem));
  return retval;
}
F OVERLOAD CALOAD(TrigSingleFP32 mem) {
  F retval;
  __asm("ld.global.ca.f32  %0, [%1];" : "=f"(retval) : "l"(mem));
  return retval;
}
i64 OVERLOAD CALOAD(i64 *mem) {
  i64 retval;
  __asm("ld.global.ca.b64  %0, [%1];" : "=l"(retval) : "l"(mem));
  return retval;
}
GF61 OVERLOAD CALOAD(TrigGF61 mem) {
  GF61 retval;
  __asm("ld.global.ca.v2.b64  {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(mem));
  return retval;
}
i32 OVERLOAD CALOAD(i32 *mem) {
  i32 retval;
  __asm("ld.global.ca.b32  %0, [%1];" : "=r"(retval) : "l"(mem));
  return retval;
}
GF31 OVERLOAD CALOAD(TrigGF31 mem) {
  GF31 retval;
  __asm("ld.global.ca.v2.b32  {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(mem));
  return retval;
}
#else
#define CALOAD    LOAD
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
#elif FFTLOAD_TYPE == 5
#define FFTLOAD    NCLOAD
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
#elif CSLOAD_TYPE == 5
#define CSLOAD    NCLOAD
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
// If possible, data should be saved in L1(?) and L2 caches and perhaps marked evict first.
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

// On "classic" AMD GCN GPUs such as Radeon VII, the wavefront size is always 64. On RDNA GPUs the wavefront can
// be configured to be either 64 or 32 (ROCm OpenCL uses 32). On AMD this comes from the host's query of
// CL_DEVICE_WAVEFRONT_WIDTH_AMD (Gpu.cpp) rather than being guessed here -- see the FAST_BARRIER comment below
// for why a guess based on compiler-predefined macros was tried and abandoned. On Nvidia GPUs the wavefront
// size is 32. This is a fallback for whenever the host did not provide a value.
#if !WAVEFRONT
#if AMDGPU
#define WAVEFRONT 64
#else
#define WAVEFRONT 32
#endif
#endif

#ifndef AMD_BARRIER_NO_WAIT
#define AMD_BARRIER_NO_WAIT 0
#endif

// FAST_BARRIER replaces barrier(CLK_LOCAL_MEM_FENCE) with barrier(0), a bare s_barrier on AMD.  That is only safe where the
// compiler puts an "s_waitcnt lgkmcnt(0)" (wait for outstanding LDS accesses) in front of every s_barrier by itself: GCN up
// to gfx908/gfx90c, running in their native 64-wide wavefront.  RDNA parts run wave32 under ROCm's OpenCL compiler (even
// though the hardware supports wave64 too) -- caught here by WAVEFRONT != 64.  gfx90a and gfx94x/gfx95x (CDNA2/CDNA3) are
// *also* natively wave64 but have the same "back-off" barrier that does not wait as RDNA, so WAVEFRONT alone cannot tell
// them apart from gfx906 -- the host passes AMD_BARRIER_NO_WAIT=1 there instead, from a device-name check (Gpu.cpp).
// This used to be a defined(__gfx906__)-style compiler-macro check instead of a host-queried WAVEFRONT/name check:
// verified empirically (an #error probe compiled through gpuowl's real OpenCL runtime, not a standalone clang invocation)
// that those macros are not defined at all on at least one ROCm version's actual compile path (comgr's OpenCL JIT), which
// silently forced FAST_BARRIER off on every AMD GPU including the ones, like gfx906, it was supposed to stay on for.
// Do not go back to compiler-macro detection here.
#ifndef FAST_BARRIER
#define FAST_BARRIER 0      // Default to the safe case, FAST_BARRIER is risky!
#endif
#if FAST_BARRIER && AMDGPU && (WAVEFRONT != 64 || AMD_BARRIER_NO_WAIT)
#undef FAST_BARRIER
#define FAST_BARRIER 0
#endif

// Default settings for USE_REGISTER_BARSYNC.  OpenCL on nVidia has compiler issues when USE_REGISTER_BARSYNC=0.  Annoying, as register bar.sync is slower in many cases.
#ifndef USE_REGISTER_BARSYNC
#if CUDA_BACKEND
#define USE_REGISTER_BARSYNC 0
#else
#define USE_REGISTER_BARSYNC 1
#endif
#endif

// Force divergent threads in a warp to converge.  AMD GCN does not require this, all threads in a WAVEFRONT operate in lockstep.  Early CUDA versions did also.
// The sync is needed in cases where one thread is setting a flag or state on behalf of all the threads in a WAVEFRONT.  For example, carryFused has thread 0 set
// the carries-are-ready flag on behalf of all 32 threads in a warp.
void sync(void) {
#if HAS_PTX >= 600         // bar.warp.sync requires sm_60 support or higher
  __asm("bar.warp.sync 0xffffffff;" : : );
#endif
}

// Create a barrier across all threads.  Primarily used to coordinate access to local memory.
// A local memory fence is optional.  This is a DANGEROUS practice!!
// On Radeon VII and Radeon PRO VII barrier with no local memory fence works (in most cases -- see barFence routine) and is much faster.
// On nVidia hardware, a barrier instruction automatically creates a local memory fence.
// Hardware from other vendors and other AMD GPUs has not been thoroughly researched.  The FAST_BARRIER option allows selecting the faster path when it works.
void OVERLOAD bar(void) {
  barrier(FAST_BARRIER ? 0 : CLK_LOCAL_MEM_FENCE);
}

// Create a barrier across a subset of threads OR across all threads if that is faster.
// Again, the local memory fence is optional controlled by the FAST_BARRIER setting.
// At this point in time, only nVidia GPUs support creating a barrier across a subset of threads.
void OVERLOAD bar(const u32 WG) {
  // A group no larger than a wavefront can skip the barrier only where the hardware really does run a whole
  // wavefront in lock-step.  AMD GCN does, and so did nVidia before Volta.  Volta and later do not:
  // Independent Thread Scheduling lets the threads of a warp drift apart, and every caller of bar(WG)
  // exchanges data through LDS right afterwards, so the warp has to be reconverged and its LDS traffic
  // ordered -- which is what sync() plus the fence do, for a fraction of the cost of a barrier.  Anything
  // else (Intel, a CPU device under POCL) offers no lock-step guarantee at all: use a real barrier there.
  if (WG <= WAVEFRONT) {
#if HAS_PTX >= 600         // bar.warp.sync requires sm_60 or higher; ITS arrived in sm_70
    sync();
    mem_fence(CLK_LOCAL_MEM_FENCE);
    return;
#elif AMDGPU || HAS_PTX >= 200
    if (!FAST_BARRIER) mem_fence(CLK_LOCAL_MEM_FENCE);
    return;
#endif
  }
#if ENABLE_BARSYNC && HAS_PTX >= 200         // bar.sync with thread count requires sm_20 support or higher.  Slower on TitanV, need to try on later nVidia GPUs.
  __asm("bar.sync %0, %1;" : : "r"(get_local_id(0) / WG + 1), "n"(WG));
// The above is GROSSLY slow on an RTX 5070Ti.  The code below is much faster (may need to be expanded to handle more than four named barriers).
// WARNING, WARNING, WARNING: On TitanV using CUDA 12.9 tools and driver 580, similar code in LDSbar does not work in openCL (but works in CUDA build).
//    if (get_local_id(0) / WG + 1 == 1) __asm("bar.sync 1, %0;" : : "n"(WG));
//    else if (get_local_id(0) / WG + 1 == 2) __asm("bar.sync 2, %0;" : : "n"(WG));
//    else if (get_local_id(0) / WG + 1 == 2) __asm("bar.sync 3, %0;" : : "n"(WG));
//    else __asm("bar.sync 4, %0;" : : "n"(WG));
#else
  bar();
#endif
}

// Create a barrier across all threads.  Like bar(), primarily used to coordinate access to local memory.
// However a local memory fence is NOT optional, guaranteeing safe behavior.
// On Radeon VII and Radeon PRO VII barrier it was discoverred that 4 byte shufls in shufl.cl required a local memory barrier.
// My theory is something like 8-byte writes store LSW, then MSW.  An 8-byte read accesses LSW first, with the write of MSW giving just
// enough cushion for the read of LSW to be safe.  That cushion disappears with 4-byte writes and reads.  Alas, that theory does not
// hold up in real practice.
void OVERLOAD barFence(void) {
  barrier(CLK_LOCAL_MEM_FENCE);
}

// Like bar(WG), except a local memory fence is NOT optional, guaranteeing safe behavior.
void OVERLOAD barFence(const u32 WG) {
  // Catch the one case where regular bar(WG) can skip the local memory fence
  if (WG <= WAVEFRONT) {
#if AMDGPU || (HAS_PTX >= 200 && HAS_PTX < 600)
    mem_fence(CLK_LOCAL_MEM_FENCE);
    return;
#endif
  }
  bar(WG);
}

// Create a barrier across a subset of threads.  Substituting a barrier on all threads is not permitted, so this is only
// defined where the hardware can do it (PTX bar.sync with a thread count, sm_20 or higher).  On any other GPU a call to
// barsync() fails to compile at the call site instead of the whole of base.cl failing whether or not it is used.
#if HAS_PTX >= 200
void barsync(const u32 numWG, const u32 WG) {
  // As in bar(WG) above, except that substituting a barrier over all threads is not allowed here, so on
  // Volta and later the warp-wide sync is the only option.  (This routine is nVidia-only to begin with.)
  if (WG <= WAVEFRONT) {
#if HAS_PTX >= 600
    sync();
    mem_fence(CLK_LOCAL_MEM_FENCE);
#endif
    return;
  }
#if USE_REGISTER_BARSYNC   // bar.sync with a register is horribly slow on an RTX 5070Ti.
  __asm("bar.sync %0, %1;" : : "r"(get_local_id(0) / WG + 1), "n"(WG));
#else                      // WARNING, WARNING, WARNING: On TitanV using CUDA 12.9 tools and driver 580, this branch does not work in openCL (but works in CUDA build).
  for (u32 i = 1; i <= numWG; i++) {
    if (i == get_local_id(0) / WG + 1) {
      __asm("bar.sync %0, %1;" : : "n"(i), "n"(WG));
      break;
    }
  }
#endif
}
#endif

// OPAQUE(x) hides x's value from the optimizer, forcing expressions that use x afterwards to be recomputed rather than
// reused from registers.  The asm constraint letter is backend-specific, and the other backend's is a compile error.
// OPAQUE takes a 32-bit integer, OPAQUE_F64 a double (PTX needs a different constraint letter for each).
#if HAS_ASM
#define OPAQUE(x) __asm volatile("" : "+v"(x))
#define OPAQUE_F64(x) __asm volatile("" : "+v"(x))
#elif HAS_PTX
#define OPAQUE(x) __asm volatile("" : "+r"(x))
#define OPAQUE_F64(x) __asm volatile("" : "+d"(x))
#else
#define OPAQUE(x)
#define OPAQUE_F64(x)
#endif

// nVidia GPUs (Hopper architecture sm 9.0 and later) support Programatic Dependent Launch where the tail end execution of one kernel can overlap
// with the beginning of the next kernel.  This requires a special launch kernel command that is only available in CUDA 12.0 and later.
// These routines let us take advantage of this CUDA feature.  These routines do nothing in OpenCL.

// Switched on per run with -use PDL=1. The CUDA shim launches a kernel with
// programmatic stream serialization exactly when its compiled code contains
// the wait below (it reads the PTX), so a kernel that never waits is never
// allowed to start early. Off, both routines compile to nothing and every
// launch is ordinary.
#ifndef PDL
#define PDL 0
#endif

void dependentLaunch() {
#if CUDA_BACKEND && HAS_PTX >= 900 && PDL
  __asm volatile("griddepcontrol.launch_dependents;");    // same as cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void dependentLaunchWait() {
#if CUDA_BACKEND && HAS_PTX >= 900 && PDL
  __asm volatile("griddepcontrol.wait;");                 // same as cudaGridDependencySynchronize();
#endif
}

