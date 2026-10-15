// Copyright (C) Mihai Preda and George Woltman

#pragma once

#define A(k) u[(base + k * step) % M]

#if FFT_FP64

// 18 FMA + 12 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 5-k, then each output pair j, 5-j is
// (u0 + sum_k cos(j*k*tau/5) sum_k) +- sum_k sin(j*k*tau/5) dif_k, with w = e^(2*pi*i/5).  cbase = u0 + cos(1*tau/5) * sum zeroes one
// term per cos row (the cosine nearest the middle of the range: the smallest coefficients), and the sin rows are scaled by sin(2*tau/5),
// the most accurately represented sine.
void OVERLOAD fft5by(T2 *u, u32 base, u32 step, u32 M) {
  const double COS1 = 0.3090169943749474241023;    // cos(1*tau/5): subtracted from every cos coefficient, it zeroes one term per cos row
  const double C2_C1 = -1.1180339887498948482046;  // cos(2*tau/5) - cos(1*tau/5)
  const double SIN2 = 0.5877852522924731291687;    // sin(2*tau/5): factored out of the sin rows, it makes one term per row a plain add
  const double S1_S2 = 1.6180339887498948482046;   // sin(1*tau/5) / sin(2*tau/5)

  X2_mul_t4(A(1), A(4));
  X2_mul_t4(A(2), A(3));

  T2 sum = A(1) + A(2);
  T2 cbase = fmaT2(COS1, sum, A(0));                    // A(0) + COS1 * sum: each cos row below then has one term fewer
  A(0) = A(0) + sum;

  T2 tmp34a = fmaT2(C2_C1, A(1), cbase);
  T2 tmp25a = fmaT2(C2_C1, A(2), cbase);

  T2 tmp25b = fmaT2(S1_S2, A(4), A(3));
  T2 tmp34b = fmaT2(S1_S2, -A(3), A(4));

  fma_addsub(A(1), A(4), SIN2, tmp25a, tmp25b);
  fma_addsub(A(2), A(3), SIN2, tmp34a, tmp34b);
}

void OVERLOAD fft5(T2 *u) { return fft5by(u, 0, 1, 5); }

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// 18 FMA + 12 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 5-k, then each output pair j, 5-j is
// (u0 + sum_k cos(j*k*tau/5) sum_k) +- sum_k sin(j*k*tau/5) dif_k, with w = e^(2*pi*i/5).  cbase = u0 + cos(1*tau/5) * sum zeroes one
// term per cos row (the cosine nearest the middle of the range: the smallest coefficients), and the sin rows are scaled by sin(2*tau/5),
// the most accurately represented sine.
void OVERLOAD fft5by(F2 *u, u32 base, u32 step, u32 M) {
  const float COS1 = 0.3090169943749474241023f;    // cos(1*tau/5): subtracted from every cos coefficient, it zeroes one term per cos row
  const float C2_C1 = -1.1180339887498948482046f;  // cos(2*tau/5) - cos(1*tau/5)
  const float SIN2 = 0.5877852522924731291687f;    // sin(2*tau/5): factored out of the sin rows, it makes one term per row a plain add
  const float S1_S2 = 1.6180339887498948482046f;   // sin(1*tau/5) / sin(2*tau/5)

  X2_mul_t4(A(1), A(4));
  X2_mul_t4(A(2), A(3));

  F2 sum = A(1) + A(2);
  F2 cbase = fmaT2(COS1, sum, A(0));                    // A(0) + COS1 * sum: each cos row below then has one term fewer
  A(0) = A(0) + sum;

  F2 tmp34a = fmaT2(C2_C1, A(1), cbase);
  F2 tmp25a = fmaT2(C2_C1, A(2), cbase);

  F2 tmp25b = fmaT2(S1_S2, A(4), A(3));
  F2 tmp34b = fmaT2(S1_S2, -A(3), A(4));

  fma_addsub(A(1), A(4), SIN2, tmp25a, tmp25b);
  fma_addsub(A(2), A(3), SIN2, tmp34a, tmp34b);
}

void OVERLOAD fft5(F2 *u) { return fft5by(u, 0, 1, 5); }

#endif

/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

#if PFA == 5 && PFA_W61 != 725554454131936870ull
#error PFA_W61 does not match the root hardwired here (pfaRootOfUnity in TrigBufCache.cpp)
#endif

// Radix-5 DFT, y_j = sum_k u_k w^(jk) with w = 37^((p-1)/5) (pfaRootOfUnity), for the radix-PFA of a PFA middle.  The method of
// the FP version above: pair inputs k and 5-k (X2 makes the sums and differences), then output pair j, 5-j is
// (u0 + sum_k c_jk sum_k) +- sum_k s_jk dif_k, with c_m = (w^m + w^-m)/2 and s_m = (w^m - w^-m)/2 in Z/pZ.  base = u0 + c1 * (the sum of
// the sums) zeroes one term per cos row.  Each row is a 128-bit dot product reduced once.
void OVERLOAD fft5by(GF61 *u, u32 base, u32 step, u32 M) {
  const Z61 C1 = 1317869282320094880ull;
  const Z61 C2_C1 = 823025949180351166ull;    // c2 - c1
  const Z61 S1 = 1713528181025535941ull;
  const Z61 S2 = 359591414418026295ull;

  // Quick (unreduced) arithmetic, ranges in M61s as in math.cl.  Inputs 0..1+.
  GF61 sum1 = A(1) + A(4);                    // 0..2+
  GF61 dif1 = A(1) + neg(A(4), 2);             // 1-..3+, positive for the multiplies
  GF61 sum2 = A(2) + A(3);                    // 0..2+
  GF61 dif2 = A(2) + neg(A(3), 2);             // 1-..3+, positive for the multiplies
  GF61 sall = modM61q(sum1 + sum2, 0);                  // 0..4+ before the mod
  // base = a0 + c1 * sall: each cos row then uses c_m - c1 and skips the c1 term
  u128 basex = mad64(sall.x, C1, A(0).x), basey = mad64(sall.y, C1, A(0).y);
  A(0) = modM61q(A(0) + sall, 0);                          // 0..2+ before the mod

  // One row at a time, each k's cos and sin terms together (the row's 128-bit accumulators are dead before the next row starts);
  // the outputs are kept until the last row
  u128 pc1x = basex, pc1y = basey;
  u128 qs1x = mul64(dif1.x, S1), qs1y = mul64(dif1.y, S1);
  pc1x = mad64(sum2.x, C2_C1, pc1x); pc1y = mad64(sum2.y, C2_C1, pc1y);
  qs1x = mad64(dif2.x, S2, qs1x); qs1y = mad64(dif2.y, S2, qs1y);
  GF61 pc1 = U2(weakModM61(pc1x, 128), weakModM61(pc1y, 128));        // 0..2+
  GF61 qs1 = U2(weakModM61(qs1x, 128), weakModM61(qs1y, 128));        // 0..2+
  GF61 y1 = modM61q(pc1 + qs1, 0), y4 = modM61q(pc1 - qs1, 3);  // 0..4+ and -2-..2+ before the mods

  u128 pc2x = basex, pc2y = basey;
  pc2x = mad64(sum1.x, C2_C1, pc2x); pc2y = mad64(sum1.y, C2_C1, pc2y);
  u128 qs2x = mul64(dif1.x, S2), qs2y = mul64(dif1.y, S2);
  qs2x = mad64(dif2.x, (M61 - S1), qs2x); qs2y = mad64(dif2.y, (M61 - S1), qs2y);
  GF61 pc2 = U2(weakModM61(pc2x, 128), weakModM61(pc2y, 128));        // 0..2+
  GF61 qs2 = U2(weakModM61(qs2x, 128), weakModM61(qs2y, 128));        // 0..2+
  GF61 y2 = modM61q(pc2 + qs2, 0), y3 = modM61q(pc2 - qs2, 3);  // 0..4+ and -2-..2+ before the mods

  A(1) = y1; A(4) = y4;
  A(2) = y2; A(3) = y3;
}

void OVERLOAD fft5(GF61 *u) { fft5by(u, 0, 1, 5); }

#endif


#undef A
