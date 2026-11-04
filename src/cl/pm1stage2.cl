// Copyright (C) Mihai Preda and George Woltman

// P-1 stage 2:  out = a - b on fully forward-transformed data (fftHin's output, the form tailMulLow reads as its second operand).
// The transform is linear, so this is the transform of the difference.  Every FFT/NTT part of the buffer is subtracted:  the FP64 or
// FP32 part at the start, the GF31 part at DISTGF31 and the GF61 part at DISTGF61 (in units of T2).  SUB_N2 is the number of complex
// values in each part, padding included (the padding holds nothing that is read).

#include "base.cl"
#include "math.cl"

KERNEL(256) subLow(P(T2) out, CP(T2) a, CP(T2) b) {
  u32 const step = get_global_size(0);
  u32 const me = get_global_id(0);

#if FFT_FP64
  for (u32 i = me; i < SUB_N2; i += step) { out[i] = a[i] - b[i]; }
#endif

#if FFT_FP32
  {
    P(F2) out32 = (P(F2)) out;
    CP(F2) a32 = (CP(F2)) a;
    CP(F2) b32 = (CP(F2)) b;
    for (u32 i = me; i < SUB_N2; i += step) { out32[i] = a32[i] - b32[i]; }
  }
#endif

#if NTT_GF31
  {
    P(GF31) out31 = (P(GF31)) (out + DISTGF31);
    CP(GF31) a31 = (CP(GF31)) (a + DISTGF31);
    CP(GF31) b31 = (CP(GF31)) (b + DISTGF31);
    for (u32 i = me; i < SUB_N2; i += step) { out31[i] = sub(a31[i], b31[i]); }
  }
#endif

#if NTT_GF61
  {
    P(GF61) out61 = (P(GF61)) (out + DISTGF61);
    CP(GF61) a61 = (CP(GF61)) (a + DISTGF61);
    CP(GF61) b61 = (CP(GF61)) (b + DISTGF61);
    for (u32 i = me; i < SUB_N2; i += step) { out61[i] = sub(a61[i], b61[i]); }
  }
#endif
}
