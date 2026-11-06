// Copyright (C) Mihai Preda and George Woltman

// P-1 stage 2:  tailMulLow with the difference of two Low operands, acc * (a - b), a = A_k and b = B_r both fully forward
// transformed (fftHin's output).  The same tail kernels as tailmul.cl compiled with MUL_LOW:  the kernels take b after a, and after
// every read of the Low operand tailmul.cl's MULTIPLY_DELTA lines read b and subtract it (in floating point, or with math.cl's modular
// sub() for GF31/GF61).  One extra read in the tail instead of a separate pass forming the difference (three buffer passes).
// The MULTIPLY_DELTA blocks in tailmul.cl are left out of PRP's kernel cache keys (see KernelCompiler), so PRP's kernels and their
// cache files do not change.

#if !MUL_LOW
#error pm1tailmul.cl needs MUL_LOW
#endif
#define MULTIPLY_DELTA 1

#include "base.cl"
#include "math.cl"

// p -= b's part of the type of p, read as read() reads (middle.cl).  Without a lane argument the lane is get_local_id(0).
#if FFT_FP64
void OVERLOAD readSubLow(u32 WG_SZ, u32 N, T2 *u, CP(T2) b, u32 base, u32 lane) {
  b += base + lane;
  for (u32 i = 0; i < N; ++i) { u[i] = u[i] - FFTLOAD(&b[i * WG_SZ]); }
}
void OVERLOAD readSubLow(u32 WG_SZ, u32 N, T2 *u, CP(T2) b, u32 base) { readSubLow(WG_SZ, N, u, b, base, get_local_id(0)); }
#endif

#if FFT_FP32
void OVERLOAD readSubLow(u32 WG_SZ, u32 N, F2 *u, CP(T2) b, u32 base, u32 lane) {
  CP(F2) b32 = (CP(F2)) b + base + lane;
  for (u32 i = 0; i < N; ++i) { u[i] = u[i] - FFTLOAD(&b32[i * WG_SZ]); }
}
void OVERLOAD readSubLow(u32 WG_SZ, u32 N, F2 *u, CP(T2) b, u32 base) { readSubLow(WG_SZ, N, u, b, base, get_local_id(0)); }
#endif

#if NTT_GF31
void OVERLOAD readSubLow(u32 WG_SZ, u32 N, GF31 *u, CP(T2) b, u32 base, u32 lane) {
  CP(GF31) b31 = (CP(GF31)) (b + DISTGF31) + base + lane;
  for (u32 i = 0; i < N; ++i) { u[i] = sub(u[i], FFTLOAD(&b31[i * WG_SZ])); }
}
void OVERLOAD readSubLow(u32 WG_SZ, u32 N, GF31 *u, CP(T2) b, u32 base) { readSubLow(WG_SZ, N, u, b, base, get_local_id(0)); }
#endif

#if NTT_GF61
void OVERLOAD readSubLow(u32 WG_SZ, u32 N, GF61 *u, CP(T2) b, u32 base, u32 lane) {
  CP(GF61) b61 = (CP(GF61)) (b + DISTGF61) + base + lane;
  for (u32 i = 0; i < N; ++i) { u[i] = sub(u[i], FFTLOAD(&b61[i * WG_SZ])); }
}
void OVERLOAD readSubLow(u32 WG_SZ, u32 N, GF61 *u, CP(T2) b, u32 base) { readSubLow(WG_SZ, N, u, b, base, get_local_id(0)); }
#endif

// The extra argument b, after a
#define tailMulZero(out, in, a, ...) tailMulZero(out, in, a, CP(T2) b, __VA_ARGS__)
#define tailMul(out, in, a, ...) tailMul(out, in, a, CP(T2) b, __VA_ARGS__)
#define tailMulZeroGF31(out, in, a, ...) tailMulZeroGF31(out, in, a, CP(T2) b, __VA_ARGS__)
#define tailMulGF31(out, in, a, ...) tailMulGF31(out, in, a, CP(T2) b, __VA_ARGS__)
#define tailMulZeroGF61(out, in, a, ...) tailMulZeroGF61(out, in, a, CP(T2) b, __VA_ARGS__)
#define tailMulGF61(out, in, a, ...) tailMulGF61(out, in, a, CP(T2) b, __VA_ARGS__)

#include "tailmul.cl"
