// OpenCL → CUDA compatibility header for PRPLL
// Allows .cl kernel files to compile under NVRTC with minimal changes.
// Used together with NvrtcProgram::preprocessOpenCL() which strips:
//   - __global/global pointer qualifiers (can't #define without breaking __global__)
//   - #pragma OPENCL ... directives
//   - __attribute__((overloadable)) and __attribute__((reqd_work_group_size(...)))

#pragma once

// Set flag that OpenCL can access.  Let's us add CUDA-only code to the OpenCL sources.
#define CUDA_BACKEND 1

// ---- Qualifiers ----
// __kernel / kernel → extern "C" __global__ (CUDA kernel launch qualifier)
// extern "C" is needed so cuModuleGetFunction() can find kernels by unmangled name
#define __kernel extern "C" __global__
#define kernel extern "C" __global__

// __local / local — OpenCL address space qualifier for shared memory.
// In CUDA, __shared__ can only be used on variable declarations, NOT on function parameters.
// The preprocessor handles this: it strips "local" from function parameter lists
// and adds __shared__ for variable declarations matching "local TYPE NAME[".
#define __local
#define local

// __constant / constant → const (CUDA __constant__ is file-scope only, can't be used
// for kernel params). The compiler auto-uses __ldg() for const pointers on sm_35+.
#define __constant const
#define constant const

// restrict → __restrict__ (different keyword in CUDA)
#define restrict __restrict__

// ---- Work-item functions ----
#define get_local_id(d)    ((unsigned int)threadIdx.x)
#define get_group_id(d)    ((unsigned int)blockIdx.x)
#define get_local_size(d)  ((unsigned int)blockDim.x)
#define get_global_id(d)   ((unsigned int)(blockIdx.x * blockDim.x + threadIdx.x))
#define get_num_groups(d)  ((unsigned int)gridDim.x)
#define get_global_size(d) ((unsigned int)(gridDim.x * blockDim.x))
#define get_enqueued_local_size(d) get_local_size(d)

// ---- Barriers ----
#define CLK_LOCAL_MEM_FENCE 0
#define CLK_GLOBAL_MEM_FENCE 0
#define barrier(flags) __syncthreads()

// ---- Memory fences ----
// OpenCL write_mem_fence / read_mem_fence → CUDA __threadfence()
#define write_mem_fence(flags) __threadfence()
#define read_mem_fence(flags) __threadfence()
#define mem_fence(flags) __threadfence()

// ---- Overloadable ----
// CUDA C++ supports function overloading natively
#define OVERLOAD

// ---- OpenCL extension macros ----
#define cl_khr_fp64 1
#define cl_khr_subgroups 1

// ---- Kernel macro ----
// PRPLL uses KERNEL(WG_SIZE) void kernelName(...)
// base.cl defines: #define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void
// Our preprocessor replaces base.cl's KERNEL macro with a CUDA version.
// This fallback is only used if base.cl hasn't been included yet.
// KERNEL macro fallback — usually overridden by base.cl KERNEL macro replacement in cudawrap.cpp
#define KERNEL(x) extern "C" __global__ void __launch_bounds__(x)

// ---- Pointer macros ----
#define P(x)  x* __restrict__
#define CP(x) const x* __restrict__

// ---- OpenCL type aliases ----

// Standard PRPLL type aliases
typedef unsigned int uint;

// These match OpenCL's types exactly. The preprocessor strips base.cl's re-definitions of i32/u32 to avoid redeclaration errors.
typedef int           i32;
typedef unsigned int  u32;

// OpenCL defines long, ulong, long2, ulong2, etc. as 64-bits.  CUDA defines them as 32-bits (MSVC) or 64-bits (Linux).
// CUDA defines longlong, ulonglong, longlong2, ulonglong2, etc. as 64-bits.  Map OpenCL types to CUDA types.

//#define long       long long        // Obviously, we can't uncomment this #define.  Instead, we must make sure the opencl code never uses this data type.  Use i64 instead.
#define long2	     longlong2
#define ulong	     unsigned long long
#define ulong2	     ulonglong2
#define make_long2   make_longlong2
#define make_ulong2  make_ulonglong2

// These must match the 64-bit data types defined above.  The preprocessor strips base.cl's re-definitions of i64/u64 to avoid redeclaration errors.
// Must use 'long long' / 'unsigned long long' to match CUDA vector type members so that Z61 (typedef'd from ulong) matches ulong2 member types.
// Otherwise overloaded functions like add(Z31,Z31) vs add(Z61,Z61) become ambiguous when called with ulong2 member values (which are 'unsigned long').
typedef long long          i64;
typedef unsigned long long u64;

// ---- Math constants ----
#ifndef M_PI
#define M_PI      3.14159265358979323846
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

// ---- Vector arithmetic operators ----
// OpenCL supports +, -, *, / on vector types natively. CUDA does not.
// double2
__device__ __forceinline__ double2 operator+(double2 a, double2 b) { return make_double2(a.x+b.x, a.y+b.y); }
__device__ __forceinline__ double2 operator-(double2 a, double2 b) { return make_double2(a.x-b.x, a.y-b.y); }
__device__ __forceinline__ double2 operator*(double2 a, double2 b) { return make_double2(a.x*b.x, a.y*b.y); }
__device__ __forceinline__ double2 operator-(double2 a) { return make_double2(-a.x, -a.y); }
__device__ __forceinline__ double2 operator*(double s, double2 a) { return make_double2(s*a.x, s*a.y); }
__device__ __forceinline__ double2 operator*(double2 a, double s) { return make_double2(a.x*s, a.y*s); }
__device__ __forceinline__ double2& operator+=(double2& a, double2 b) { a.x+=b.x; a.y+=b.y; return a; }
__device__ __forceinline__ double2& operator-=(double2& a, double2 b) { a.x-=b.x; a.y-=b.y; return a; }

// float2
__device__ __forceinline__ float2 operator+(float2 a, float2 b) { return make_float2(a.x+b.x, a.y+b.y); }
__device__ __forceinline__ float2 operator-(float2 a, float2 b) { return make_float2(a.x-b.x, a.y-b.y); }
__device__ __forceinline__ float2 operator*(float2 a, float2 b) { return make_float2(a.x*b.x, a.y*b.y); }
__device__ __forceinline__ float2 operator-(float2 a) { return make_float2(-a.x, -a.y); }
__device__ __forceinline__ float2 operator*(float s, float2 a) { return make_float2(s*a.x, s*a.y); }
__device__ __forceinline__ float2 operator*(float2 a, float s) { return make_float2(a.x*s, a.y*s); }
__device__ __forceinline__ float2& operator+=(float2& a, float2 b) { a.x+=b.x; a.y+=b.y; return a; }
__device__ __forceinline__ float2& operator-=(float2& a, float2 b) { a.x-=b.x; a.y-=b.y; return a; }

// int2
__device__ __forceinline__ int2 operator+(int2 a, int2 b) { return make_int2(a.x+b.x, a.y+b.y); }
__device__ __forceinline__ int2 operator-(int2 a, int2 b) { return make_int2(a.x-b.x, a.y-b.y); }
__device__ __forceinline__ int2 operator*(int2 a, int2 b) { return make_int2(a.x*b.x, a.y*b.y); }
__device__ __forceinline__ int2 operator-(int2 a) { return make_int2(-a.x, -a.y); }
__device__ __forceinline__ int2& operator+=(int2& a, int2 b) { a.x+=b.x; a.y+=b.y; return a; }
__device__ __forceinline__ int2& operator-=(int2& a, int2 b) { a.x-=b.x; a.y-=b.y; return a; }

// uint2
__device__ __forceinline__ uint2 operator+(uint2 a, uint2 b) { return make_uint2(a.x+b.x, a.y+b.y); }
__device__ __forceinline__ uint2 operator-(uint2 a, uint2 b) { return make_uint2(a.x-b.x, a.y-b.y); }
__device__ __forceinline__ uint2 operator*(uint2 a, uint2 b) { return make_uint2(a.x*b.x, a.y*b.y); }
__device__ __forceinline__ uint2& operator+=(uint2& a, uint2 b) { a.x+=b.x; a.y+=b.y; return a; }
__device__ __forceinline__ uint2& operator-=(uint2& a, uint2 b) { a.x-=b.x; a.y-=b.y; return a; }

// long2
__device__ __forceinline__ long2 operator+(long2 a, long2 b) { return {a.x+b.x, a.y+b.y}; }
__device__ __forceinline__ long2 operator-(long2 a, long2 b) { return {a.x-b.x, a.y-b.y}; }
__device__ __forceinline__ long2 operator*(long2 a, long2 b) { return {a.x*b.x, a.y*b.y}; }
__device__ __forceinline__ long2 operator-(long2 a) { return {-a.x, -a.y}; }
__device__ __forceinline__ long2& operator+=(long2& a, long2 b) { a.x+=b.x; a.y+=b.y; return a; }
__device__ __forceinline__ long2& operator-=(long2& a, long2 b) { a.x-=b.x; a.y-=b.y; return a; }

// ulong2
__device__ __forceinline__ ulong2 operator+(ulong2 a, ulong2 b) { return {a.x+b.x, a.y+b.y}; }
__device__ __forceinline__ ulong2 operator-(ulong2 a, ulong2 b) { return {a.x-b.x, a.y-b.y}; }
__device__ __forceinline__ ulong2 operator*(ulong2 a, ulong2 b) { return {a.x*b.x, a.y*b.y}; }
__device__ __forceinline__ ulong2& operator+=(ulong2& a, ulong2 b) { a.x+=b.x; a.y+=b.y; return a; }
__device__ __forceinline__ ulong2& operator-=(ulong2& a, ulong2 b) { a.x-=b.x; a.y-=b.y; return a; }

// Scalar * vector operators for types not built-in to NVRTC
// (NVRTC already provides double2*double, int2*int, uint2*uint, etc.)
// These cover cross-type scalar*vector that OpenCL supports natively.
__device__ __forceinline__ long2 operator*(i64 s, long2 v) { return {s*v.x, s*v.y}; }
__device__ __forceinline__ long2 operator*(long2 v, i64 s) { return {v.x*s, v.y*s}; }
__device__ __forceinline__ ulong2 operator*(ulong s, ulong2 v) { return {s*v.x, s*v.y}; }
__device__ __forceinline__ ulong2 operator*(ulong2 v, ulong s) { return {v.x*s, v.y*s}; }
// int * ulong2 (common in NTT code: int literal * GF61)
__device__ __forceinline__ ulong2 operator*(int s, ulong2 v) { return {(ulong)s*v.x, (ulong)s*v.y}; }
__device__ __forceinline__ ulong2 operator*(ulong2 v, int s) { return {v.x*(ulong)s, v.y*(ulong)s}; }

// ---- Vector constructors (U2) ----
// OpenCL (type2)(a,b) cast syntax is converted to make_type2(a,b) by the preprocessor.
// base.cl defines U2() as overloaded functions — they'll work after preprocessing.
// NVRTC provides make_double2, make_float2, make_int2, make_uint2, make_long2, make_ulong2 built-in.

// ---- Type reinterpretation (as_*) ----
// Scalar ↔ vector bitwise reinterpretations (OpenCL as_type functions)

// as_uint2: split 64-bit value into two 32-bit halves
__device__ __forceinline__ uint2 as_uint2(double v) { union { uint2 ui2; double d; } u; u.d = v; return u.ui2; }
__device__ __forceinline__ uint2 as_uint2(ulong v) { union { uint2 ui2; ulong ul; } u; u.ul = v; return u.ui2; }
__device__ __forceinline__ uint2 as_uint2(i64 v) { union { uint2 ui2; i64 l; } u; u.l = v; return u.ui2; }

// as_int2: split 64-bit value into two signed 32-bit halves
__device__ __forceinline__ int2 as_int2(double v) { union { int2 i2; double d; } u; u.d = v; return u.i2; }
__device__ __forceinline__ int2 as_int2(i64 v) { union { int2 i2; i64 l; } u; u.l = v; return u.i2; }

// as_double: reinterpret bits as double
__device__ __forceinline__ double as_double(int2 v) { union { int2 i2; double d; } u; u.i2 = v; return u.d; }
__device__ __forceinline__ double as_double(uint2 v) { union { uint2 ui2; double d; } u; u.ui2 = v; return u.d; }
__device__ __forceinline__ double as_double(ulong v) { return __longlong_as_double(v); }
__device__ __forceinline__ double as_double(i64 v) { return __longlong_as_double(v); }

// as_ulong: reinterpret as unsigned 64-bit
__device__ __forceinline__ ulong as_ulong(uint2 v) { union { uint2 ui2; ulong ul; } u; u.ui2 = v; return u.ul; }
__device__ __forceinline__ ulong as_ulong(int2 v) { union { int2 i2; ulong ul; } u; u.i2 = v; return u.ul; }
__device__ __forceinline__ ulong as_ulong(double v) { return (ulong)__double_as_longlong(v); }

// as_long: reinterpret as signed 64-bit
__device__ __forceinline__ i64 as_long(int2 v) { union { int2 i2; i64 l; } u; u.i2 = v; return u.l; }
__device__ __forceinline__ i64 as_long(uint2 v) { union { uint2 ui2; i64 l; } u; u.ui2 = v; return u.l; }
__device__ __forceinline__ i64 as_long(double v) { return (i64)__double_as_longlong(v); }

// as_float / as_int / as_uint: 32-bit reinterprets
__device__ __forceinline__ float as_float(int v) { return __int_as_float(v); }
__device__ __forceinline__ float as_float(uint v) { return __int_as_float((int)v); }
__device__ __forceinline__ int as_int(float v) { return __float_as_int(v); }
__device__ __forceinline__ uint as_uint(float v) { return (uint)__float_as_int(v); }

// 16-byte reinterprets: int4 ↔ double2 ↔ ulong2
__device__ __forceinline__ int4 as_int4(double2 v) {
  union { double2 d; int4 i; } u;
  u.d = v;
  return u.i;
}
__device__ __forceinline__ int4 as_int4(ulong2 v) {
  union { ulong2 ul; int4 i; } u;
  u.ul = v;
  return u.i;
}
__device__ __forceinline__ double2 as_double2(int4 v) {
  union { int4 i; double2 d; } u;
  u.i = v;
  return u.d;
}
__device__ __forceinline__ ulong2 as_ulong2(int4 v) {
  union { int4 i; ulong2 ul; } u;
  u.i = v;
  return u.ul;
}
__device__ __forceinline__ double2 as_double2(ulong2 v) {
  union { ulong2 ul; double2 d; } u;
  u.ul = v;
  return u.d;
}
__device__ __forceinline__ ulong2 as_ulong2(double2 v) {
  union { double2 d; ulong2 ul; } u;
  u.d = v;
  return u.ul;
}

// ---- Math builtins ----
// fma for vector types (OpenCL supports element-wise fma on vector types)
__device__ __forceinline__ double2 fma(double2 a, double2 b, double2 c) {
  return make_double2(fma(a.x, b.x, c.x), fma(a.y, b.y, c.y));
}
__device__ __forceinline__ float2 fma(float2 a, float2 b, float2 c) {
  return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
}
// Mixed scalar-vector fma: fma(scalar, vec2, vec2) — broadcasts scalar
__device__ __forceinline__ double2 fma(double a, double2 b, double2 c) {
  return make_double2(fma(a, b.x, c.x), fma(a, b.y, c.y));
}
__device__ __forceinline__ float2 fma(float a, float2 b, float2 c) {
  return make_float2(fmaf(a, b.x, c.x), fmaf(a, b.y, c.y));
}

// mul_hi: upper half of multiplication
__device__ __forceinline__ uint mul_hi(uint a, uint b) {
  return __umulhi(a, b);
}
__device__ __forceinline__ ulong mul_hi(ulong a, ulong b) {
  return __umul64hi(a, b);
}
__device__ __forceinline__ uint mad_hi(uint a, uint b, uint c) {
  return __umulhi(a, b) + c;
}

// ---- Atomic operations ----
#define atomic_max(p, v) atomicMax((unsigned int*)(p), (unsigned int)(v))
#define atomic_add(p, v) atomicAdd(p, v)

// OpenCL 2.0 C11-style atomics — optimized for CUDA carry stairway pattern.
// The carryFused kernel uses: producer writes data, threadfence, bar, atomic_store(flag, 1)
// then consumer does: atomic_load(flag) in spin loop, bar, threadfence, read data.
// We minimize redundant fences while maintaining correctness.
__device__ __forceinline__ void atomic_store_uint(volatile unsigned int* p, unsigned int v) {
  // Volatile store only — no fence needed here. The caller always does
  // write_mem_fence(CLK_GLOBAL_MEM_FENCE) [= __threadfence()] before calling
  // atomic_store(), which already orders all prior writes before this store.
  // Adding a second __threadfence() here was redundant but costly (~100-400 cycles).
  *p = v;
}
__device__ __forceinline__ unsigned int atomic_load_uint(volatile unsigned int* p) {
  // Acquire load: volatile ensures we re-read from memory, not from register.
  // No fence needed here — the caller does read_mem_fence AFTER confirming the flag.
  return *p;
}
#define atomic_store(p, v) atomic_store_uint((volatile unsigned int*)(p), (unsigned int)(v))
#define atomic_load_explicit(p, order, scope) atomic_load_uint((volatile unsigned int*)(p))
#define memory_order_relaxed 0
#define memory_order_acquire 0
#define memory_order_release 0
#define memory_scope_device 0
typedef volatile unsigned int atomic_uint;

// ---- Inline assembly ----
// OpenCL uses __asm(); NVRTC uses asm()
#define __asm asm

// ---- sub_group / warp functions ----
#define sub_group_broadcast(v, lane) __shfl_sync(0xFFFFFFFF, (v), (lane))

// ---- Mark as CUDA compilation ----
#define CUDA_BACKEND 1

// ---- Word2 constructor (typedef for long2 or int2) ----
// base.cl defines Word2 as long2 (WordSize==8) or int2 (WordSize==4).
// The preprocessor converts (Word2)(a, b) → make_Word2(a, b).
// Must key on WordSize, not CARRY64, because FFT3261 has WordSize=8 without CARRY64.
#if WordSize == 8
__device__ __forceinline__ long2 make_Word2(i64 a, i64 b) { return make_long2(a, b); }
#else
__device__ __forceinline__ int2 make_Word2(int a, int b) { return make_int2(a, b); }
#endif

// ---- Force NVIDIAGPU and HAS_PTX ----
#ifndef NVIDIAGPU
#define NVIDIAGPU 1
#endif
#ifndef HAS_PTX
#define HAS_PTX 1200
#endif
#ifndef AMDGPU
#define AMDGPU 0
#endif
