// OpenCL → CUDA compatibility header for PRPLL
// Allows .cl kernel files to compile under NVRTC with minimal changes.
// Used together with NvrtcProgram::preprocessOpenCL() which strips:
//   - __global/global pointer qualifiers (can't #define without breaking __global__)
//   - #pragma OPENCL ... directives
//   - __attribute__((overloadable)) and __attribute__((reqd_work_group_size(...)))

#pragma once

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
// CRITICAL: On Linux x86_64, CUDA's built-in vector types (long2, ulong2) use
// 'long' / 'unsigned long' for their members. These are DISTINCT C++ types from
// 'long long' / 'unsigned long long' even though both are 64-bit.
// We MUST use 'unsigned long' for ulong so that Z61 (typedef'd from ulong) matches
// ulong2 member types. Otherwise overloaded functions like add(Z31,Z31) vs add(Z61,Z61)
// become ambiguous when called with ulong2 member values (which are 'unsigned long').
typedef unsigned long ulong;
typedef long          slong;  // OpenCL's signed 'long' (64-bit)

// Standard PRPLL type aliases
typedef unsigned int uint;

// These match OpenCL's types exactly. The preprocessor strips base.cl's
// re-definitions of i32/u32/i64/u64 to avoid redeclaration errors.
// Must use 'long' / 'unsigned long' to match CUDA vector type members.
typedef unsigned int  u32;
typedef int           i32;
typedef unsigned long u64;
typedef long          i64;

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
__device__ __forceinline__ long2 operator*(long long s, long2 v) { return {s*v.x, s*v.y}; }
__device__ __forceinline__ long2 operator*(long2 v, long long s) { return {v.x*s, v.y*s}; }
__device__ __forceinline__ ulong2 operator*(unsigned long long s, ulong2 v) { return {s*v.x, s*v.y}; }
__device__ __forceinline__ ulong2 operator*(ulong2 v, unsigned long long s) { return {v.x*s, v.y*s}; }
// int * ulong2 (common in NTT code: int literal * GF61)
__device__ __forceinline__ ulong2 operator*(int s, ulong2 v) { return {(unsigned long long)s*v.x, (unsigned long long)s*v.y}; }
__device__ __forceinline__ ulong2 operator*(ulong2 v, int s) { return {v.x*(unsigned long long)s, v.y*(unsigned long long)s}; }

// ---- Vector constructors (U2) ----
// OpenCL (type2)(a,b) cast syntax is converted to make_type2(a,b) by the preprocessor.
// base.cl defines U2() as overloaded functions — they'll work after preprocessing.
// NVRTC provides make_double2, make_float2, make_int2, make_uint2, make_long2, make_ulong2 built-in.

// ---- Type reinterpretation (as_*) ----
// Scalar ↔ vector bitwise reinterpretations (OpenCL as_type functions)

// as_uint2: split 64-bit value into two 32-bit halves
__device__ __forceinline__ uint2 as_uint2(double v) {
  unsigned long long bits = __double_as_longlong(v);
  return make_uint2((unsigned int)(bits), (unsigned int)(bits >> 32));
}
__device__ __forceinline__ uint2 as_uint2(unsigned long long v) {
  return make_uint2((unsigned int)(v), (unsigned int)(v >> 32));
}
__device__ __forceinline__ uint2 as_uint2(unsigned long v) {
  return make_uint2((unsigned int)(v), (unsigned int)((unsigned long long)v >> 32));
}
__device__ __forceinline__ uint2 as_uint2(long long v) {
  return make_uint2((unsigned int)((unsigned long long)v), (unsigned int)((unsigned long long)v >> 32));
}
__device__ __forceinline__ uint2 as_uint2(long v) {
  return make_uint2((unsigned int)((unsigned long)v), (unsigned int)((unsigned long long)v >> 32));
}

// as_int2: split 64-bit value into two signed 32-bit halves
__device__ __forceinline__ int2 as_int2(double v) {
  unsigned long long bits = __double_as_longlong(v);
  return make_int2((int)(unsigned int)(bits), (int)(unsigned int)(bits >> 32));
}
__device__ __forceinline__ int2 as_int2(long long v) {
  return make_int2((int)(unsigned int)((unsigned long long)v), (int)(unsigned int)((unsigned long long)v >> 32));
}
__device__ __forceinline__ int2 as_int2(long v) {
  return make_int2((int)(unsigned int)((unsigned long)v), (int)(unsigned int)((unsigned long long)v >> 32));
}

// as_double: reinterpret bits as double
__device__ __forceinline__ double as_double(int2 v) {
  unsigned long long bits = ((unsigned long long)(unsigned int)v.y << 32) | (unsigned int)v.x;
  return __longlong_as_double(bits);
}
__device__ __forceinline__ double as_double(uint2 v) {
  unsigned long long bits = ((unsigned long long)v.y << 32) | v.x;
  return __longlong_as_double(bits);
}
__device__ __forceinline__ double as_double(unsigned long long v) {
  return __longlong_as_double(v);
}
__device__ __forceinline__ double as_double(unsigned long v) {
  return __longlong_as_double((unsigned long long)v);
}
__device__ __forceinline__ double as_double(long long v) { return __longlong_as_double(v); }
__device__ __forceinline__ double as_double(long v) { return __longlong_as_double((long long)v); }

// as_ulong: reinterpret as unsigned 64-bit (returns 'unsigned long' to match ulong typedef)
__device__ __forceinline__ unsigned long as_ulong(uint2 v) {
  return (unsigned long)(((unsigned long long)v.y << 32) | v.x);
}
__device__ __forceinline__ unsigned long as_ulong(int2 v) {
  return (unsigned long)(((unsigned long long)(unsigned int)v.y << 32) | (unsigned int)v.x);
}
__device__ __forceinline__ unsigned long as_ulong(double v) {
  return (unsigned long)__double_as_longlong(v);
}

// as_long: reinterpret as signed 64-bit (returns 'long' to match slong/i64)
__device__ __forceinline__ long as_long(int2 v) {
  return (long)(((unsigned long long)(unsigned int)v.y << 32) | (unsigned int)v.x);
}
__device__ __forceinline__ long as_long(uint2 v) {
  return (long)(((unsigned long long)v.y << 32) | v.x);
}
__device__ __forceinline__ long as_long(double v) { return (long)__double_as_longlong(v); }

// as_float / as_int / as_uint: 32-bit reinterprets
__device__ __forceinline__ float as_float(int v) { return __int_as_float(v); }
__device__ __forceinline__ float as_float(unsigned int v) { return __int_as_float((int)v); }
__device__ __forceinline__ int as_int(float v) { return __float_as_int(v); }
__device__ __forceinline__ unsigned int as_uint(float v) { return (unsigned int)__float_as_int(v); }

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
__device__ __forceinline__ unsigned int mul_hi(unsigned int a, unsigned int b) {
  return __umulhi(a, b);
}
// Overloads for both 'unsigned long long' and 'unsigned long' (distinct types on Linux x86_64)
__device__ __forceinline__ unsigned long long mul_hi(unsigned long long a, unsigned long long b) {
  return __umul64hi(a, b);
}
__device__ __forceinline__ unsigned long mul_hi(unsigned long a, unsigned long b) {
  return (unsigned long)__umul64hi((unsigned long long)a, (unsigned long long)b);
}
__device__ __forceinline__ unsigned int mad_hi(unsigned int a, unsigned int b, unsigned int c) {
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
__device__ __forceinline__ long2 make_Word2(long a, long b) { return make_long2(a, b); }
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
