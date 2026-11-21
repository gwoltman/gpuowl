// Copyright Mihai Preda

#pragma once

#include "Buffer.h"
#include "FFTConfig.h"

#include <mutex>
#include <utility>

using TrigBuf = Buffer<double2>;
using TrigPtr = shared_ptr<TrigBuf>;

class StrongCache {
  vector<TrigPtr> ptrs;
  u32 pos{};

public:
  explicit StrongCache(u32 size) : ptrs(size) {}

  void add(TrigPtr ptr) {
    ptrs.at(pos) = std::move(ptr);
    if (++pos >= ptrs.size()) { pos = 0; }
  }
};

class TrigBufCache {  
  const Context* context;
  std::mutex mut;

  std::map<tuple<u32, u32, u32, u32, u32>, TrigPtr::weak_type> small;
  std::map<tuple<u32, u32, u32, u32>, TrigPtr::weak_type> middle;

  // The shared-pointers below keep the most recent set of buffers alive even without any Gpu instance
  // referencing them. This allows a single worker to delete & re-create the Gpu instance and still reuse the buffers.
  StrongCache smallCache{4};
  StrongCache middleCache{4};

public:
  TrigBufCache(const Context* context) :
    context{context}
  {}

  ~TrigBufCache();

  TrigPtr smallTrigCombo(Args *args, FFTConfig fft, u32 width, u32 middle, u32 height, u32 nH, bool tail_single_wide);
  TrigPtr middleTrig(Args *args, FFTConfig fft, u32 SMALL_H, u32 MIDDLE, u32 W, u32 msplit);
  TrigPtr smallTrig(Args *args, FFTConfig fft, u32 width, u32 nW, u32 middle, u32 height, u32 nH, bool tail_single_wide);
};


double2 root1Fancy(u32 N, u32 k);               // For small angles, return "fancy" cos - 1 for increased precision
double2 root1(u32 N, u32 k);

float2 root1FancyFP32(u32 N, u32 k);            // For small angles, return "fancy" cos - 1 for increased precision
float2 root1FP32(u32 N, u32 k);

uint2 root1GF31(u32 N, u32 k);
ulong2 root1GF61(u32 N, u32 k);

// The defaults of the -use keys CMUL61 and CMUL61WHERE, as in base.cl
inline u32 cmul61Default(bool amd) { return amd ? 20 : 11; }
inline u32 cmul61WhereDefault(bool amd) { return amd ? 1 : 0; }

// PFA (see base.cl): arithmetic mod p = 2^q - 1 (q = 31 or 61), and the primitive R-th root of unity in Z/pZ that the radix-R
// between the rows uses, g^((p - 1) / R) with g a primitive root mod p (37 for M61, 7 for M31).  Kernels and tables must agree on it.
u64 pfaMulMod(u32 q, u64 a, u64 b);
u64 pfaPowMod(u32 q, u64 a, u64 e);
u64 pfaRootOfUnity(u32 q, u32 R);

// Compute the size of the largest possible trig buffer given width, middle, height (in number of double2 values)
// The width table is the combo table only when TrigBufCache::smallTrig shares it: W == H and nW == nH (not so with 256RADIX8 on one side)
#define SMALLTRIG_FP64_SIZE(W,M,H,nW,nH)        ((W) != (H) || (nW) != (nH) || (H) == 0 ? (W) * 5 : SMALLTRIGCOMBO_FP64_SIZE(W,M,H,nH)) // See genSmallTrigFP64
#define SMALLTRIGCOMBO_FP64_SIZE(W,M,H,nH)      ((H) * 5 + ((W) * (M) / 2 + 1 + (M)) * 2 * (H) / (nH))      // + M: PFA region B                          // See genSmallTrigComboFP64
#define MIDDLETRIG_FP64_SIZE(W,M,H)             ((M) & ((M) - 1) ? (H) * ((M) - 1) + (W) + (H) : (H) + (W) + (H))  // Larger for a PFA middle                                                     // See genMiddleTrigFP64

// Compute the size of the largest possible trig buffer given width, middle, height (in number of float2 values)
#define SMALLTRIG_FP32_SIZE(W,M,H,nW,nH)        ((W) != (H) || (nW) != (nH) || (H) == 0 ? (W) * 5 : SMALLTRIGCOMBO_FP32_SIZE(W,M,H,nH)) // See genSmallTrigFP32
#define SMALLTRIGCOMBO_FP32_SIZE(W,M,H,nH)      ((H) * 5 + ((W) * (M) / 2 + 1 + (M)) * 2 * (H) / (nH))      // + M: PFA region B                          // See genSmallTrigComboFP32
#define MIDDLETRIG_FP32_SIZE(W,M,H)             ((M) & ((M) - 1) ? (H) * ((M) - 1) + (W) + (H) : (H) + (W) + (H))  // Larger for a PFA hybrid                                                     // See genMiddleTrigFP32

// Compute the size of the largest possible trig buffer given width, middle, height (in number of uint2 values)
#define SMALLTRIG_GF31_SIZE(W,M,H,nW,nH)        ((W) != (H) || (nW) != (nH) || (H) == 0 ? (W) : SMALLTRIGCOMBO_GF31_SIZE(W,M,H,nH))     // See genSmallTrigGF31
#define SMALLTRIGCOMBO_GF31_SIZE(W,M,H,nH)      ((H) + ((W) * (M) / 2 + ((M) / (((M) & (~(M) + 1)) + ((M) == 0)))) * 2 * (H) / (nH))                            // See genSmallTrigComboGF31
#define MIDDLETRIG_GF31_SIZE(W,M,H)             ((H) * ((M) - 1) + (W) + (H))                                           // See genMiddleTrigGF31

// Compute the size of the largest possible trig buffer given width, middle, height (in number of ulong2 values)
#define SMALLTRIG_GF61_SIZE(W,M,H,nW,nH)        ((W) != (H) || (nW) != (nH) || (H) == 0 ? (W) * 2 : SMALLTRIGCOMBO_GF61_SIZE(W,M,H,nH))     // See genSmallTrigGF61
#define SMALLTRIGCOMBO_GF61_SIZE(W,M,H,nH)      ((H) * 2 + ((W) * (M) / 2 + ((M) / (((M) & (~(M) + 1)) + ((M) == 0)))) * 2 * (H) / (nH))                            // See genSmallTrigComboGF61
#define MIDDLETRIG_GF61_SIZE(W,M,H)             ((H) * ((M) - 1) + (W) + (H))                                           // See genMiddleTrigGF61

// Convert above sizes to distances (in units of double2)
#define SMALLTRIG_FP64_DIST(W,M,H,nW,nH)        SMALLTRIG_FP64_SIZE(W,M,H,nW,nH)
#define SMALLTRIGCOMBO_FP64_DIST(W,M,H,nH)      SMALLTRIGCOMBO_FP64_SIZE(W,M,H,nH)
#define MIDDLETRIG_FP64_DIST(W,M,H)             MIDDLETRIG_FP64_SIZE(W,M,H)

#define SMALLTRIG_FP32_DIST(W,M,H,nW,nH)        SMALLTRIG_FP32_SIZE(W,M,H,nW,nH) * sizeof(float) / sizeof(double)
#define SMALLTRIGCOMBO_FP32_DIST(W,M,H,nH)      SMALLTRIGCOMBO_FP32_SIZE(W,M,H,nH) * sizeof(float) / sizeof(double)
#define MIDDLETRIG_FP32_DIST(W,M,H)             MIDDLETRIG_FP32_SIZE(W,M,H) * sizeof(float) / sizeof(double)

#define SMALLTRIG_GF31_DIST(W,M,H,nW,nH)        SMALLTRIG_GF31_SIZE(W,M,H,nW,nH) * sizeof(uint) / sizeof(double)
#define SMALLTRIGCOMBO_GF31_DIST(W,M,H,nH)      SMALLTRIGCOMBO_GF31_SIZE(W,M,H,nH) * sizeof(uint) / sizeof(double)
#define MIDDLETRIG_GF31_DIST(W,M,H)             MIDDLETRIG_GF31_SIZE(W,M,H) * sizeof(uint) / sizeof(double)

#define SMALLTRIG_GF61_DIST(W,M,H,nW,nH)        SMALLTRIG_GF61_SIZE(W,M,H,nW,nH) * sizeof(ulong) / sizeof(double)
#define SMALLTRIGCOMBO_GF61_DIST(W,M,H,nH)      SMALLTRIGCOMBO_GF61_SIZE(W,M,H,nH) * sizeof(ulong) / sizeof(double)
#define MIDDLETRIG_GF61_DIST(W,M,H)             MIDDLETRIG_GF61_SIZE(W,M,H) * sizeof(ulong) / sizeof(double)

// MSPLIT (see msplit.cl):  the middleMul2 table of the split 64-bit type, after the regular middle tables of the buffer (the GF61 ones,
// the last, taken at their largest:  with the Karatsuba terms of their twiddles).  FP64:  W "fancy" roots root(MS * W, k) then H "fancy"
// roots root(MS * H, k), the steps w^MS of the middleMul2 and middleMul chains of a thread.  GF61:  root(MIDDLE * W, k), k < MS * W,
// stored as the trig1 table (in the CMUL61_MID2 format with its Karatsuba terms at MS * W + k).  Offsets in double2 from the start of
// the middle trig buffer, 0 for a type that isn't split; end is the buffer's size.
struct MsplitTrig { u32 fp64{}, gf61{}, end{}; };
inline MsplitTrig msplitTrig(u32 W, u32 M, u32 H, bool fp64, bool fp32, bool gf31, bool gf61, bool msplit) {
  MsplitTrig o{};
  u32 n = (fp64 ? MIDDLETRIG_FP64_DIST(W, M, H) : 0) + (fp32 ? MIDDLETRIG_FP32_DIST(W, M, H) : 0)
          + (gf31 ? MIDDLETRIG_GF31_DIST(W, M, H) : 0) + (gf61 ? 2 * (H * M + W) : 0);
  u32 const ms = M == 9 ? 3 : 4;
  if (msplit && fp64) { o.fp64 = n; n += W + H; }
  if (msplit && gf61) { o.gf61 = n; n += 2 * ms * W; }
  o.end = n;
  return o;
}
