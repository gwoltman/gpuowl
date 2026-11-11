// Copyright (C) Mihai Preda

#include "base.cl"

// The position in sequential order of the pair at column x of line g.  With PFA (see base.cl) that is logical pair pfaPair(x, g),
// so the words come out of transposeOut, and go into transposeIn, in logical order.
u32 seqPos(u32 x, u32 g) {
#if PFA
  return pfaPair(x, g);
#else
  return x * BIG_HEIGHT + g;
#endif
}

// toSeq: from transposed (BIG_HEIGHT lines of WIDTH) to sequential (WIDTH rows of BIG_HEIGHT), else the reverse.
// The workgroup moves the 64x64 tile at rows r0.. and columns c0.. of its input.
#define TRANSPOSE_TILE(toSeq) \
  u32 W = toSeq ? WIDTH : BIG_HEIGHT, H = toSeq ? BIG_HEIGHT : WIDTH; \
  u32 GPW = W / 64, GPH = H / 64; \
  u32 g = get_group_id(0); \
  u32 gy = g % GPH; \
  u32 gx = g / GPH; \
  gx = (gy + gx) % GPW; \
  u32 r0 = 64 * gy, c0 = 64 * gx; \
  u32 me = get_local_id(0);

// transposeOut adds the words it reads to *sumOut, which Gpu::readChecked compares with the sum of the words the host received.
// The words are summed as they are loaded, so the check also covers the transpose.
void addSum(ulong sum, local ulong *lds, global ulong *sumOut) {
  u32 me = get_local_id(0);
  bar();
  lds[me] = sum;
  bar();
  for (u32 s = 32; s; s /= 2) {
    if (me < s) { lds[me] += lds[me + s]; }
    bar();
  }
  if (me == 0) {
    sum = lds[0];
    u32 prev = atomic_add((global u32 *) sumOut, (u32) sum);
    atomic_add((global u32 *) sumOut + 1, (u32) ((sum + prev) >> 32));    // the high half and the low half's carry
  }
}

// The input and output index of tile element i, me (output rows are the input's columns)
#define IN_POS(toSeq, i)  (toSeq ? (r0 + i) * W + c0 + me : seqPos(r0 + i, c0 + me))
#define OUT_POS(toSeq, i) (toSeq ? seqPos(c0 + i, r0 + me) : (c0 + i) * H + r0 + me)

#if WordSize <= 4

void transposeWords(bool toSeq, local Word2 *lds, global const Word2 *restrict in, global Word2 *restrict out, global ulong *sumOut) {
  TRANSPOSE_TILE(toSeq);
  ulong sum = 0;
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    Word2 w = in[IN_POS(toSeq, i)];
    lds[i * 64 + me] = w;
    sum += (i64) w.x + (i64) w.y;
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    out[OUT_POS(toSeq, i)] = lds[me * 64 + i];
  }
  if (toSeq) { addSum(sum, (local ulong *) lds, sumOut); }
}

// from transposed to sequential.
KERNEL(64) transposeOut(P(Word2) out, CP(Word2) in, global ulong *sumOut) {
  local Word2 lds[4096];
  transposeWords(true, lds, in, out, sumOut);
}

// from sequential to transposed.
KERNEL(64) transposeIn(P(Word2) out, CP(Word2) in) {
  local Word2 lds[4096];
  transposeWords(false, lds, in, out, 0);
}

#else

void transposeWords(bool toSeq, local Word *lds, global const Word2 *restrict in, global Word2 *restrict out, global ulong *sumOut) {
  TRANSPOSE_TILE(toSeq);
  ulong sum = 0;
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    Word w = in[IN_POS(toSeq, i)].x;
    lds[i * 64 + me] = w;
    sum += (i64) w;
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    out[OUT_POS(toSeq, i)].x = lds[me * 64 + i];
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    Word w = in[IN_POS(toSeq, i)].y;
    lds[i * 64 + me] = w;
    sum += (i64) w;
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    out[OUT_POS(toSeq, i)].y = lds[me * 64 + i];
  }
  if (toSeq) { addSum(sum, (local ulong *) lds, sumOut); }
}

// from transposed to sequential.
KERNEL(64) transposeOut(P(Word2) out, CP(Word2) in, global ulong *sumOut) {
  local Word lds[4096];
  transposeWords(true, lds, in, out, sumOut);
}

// from sequential to transposed.
KERNEL(64) transposeIn(P(Word2) out, CP(Word2) in) {
  local Word lds[4096];
  transposeWords(false, lds, in, out, 0);
}

#endif
