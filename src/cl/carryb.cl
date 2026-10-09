// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"
#include "carryutil.cl"
#include "weight.cl"

KERNEL(G_W) carryB(P(Word2) io, CP(CarryABM) carryIn) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;

  // Derive the big vs. little flags from the fractional number of bits in each FFT word rather read the flags from memory.
  // Calculate the most significant 32-bits of FRAC_BPW * the index of the FFT word.  Also add FRAC_BPW_HI to test first biglit flag.
  u32 line = gy * CARRY_LEN;
#if PFA
  u32 word_index = pfaPair(G_W * gx + me, line) * 2;
#else
  u32 word_index = (gx * G_W * H + me * H + line) * 2;
#endif
  u32 frac_bits = fracBits(word_index) + FRAC_BPW_HI;

  io += G_W * gx + WIDTH * CARRY_LEN * gy;

  u32 HB = BIG_HEIGHT / CARRY_LEN;

  u32 prev = (gy + HB * G_W * gx + HB * me + (HB * WIDTH - 1)) % (HB * WIDTH);
  u32 prevLine = prev % HB;
  u32 prevCol  = prev / HB;
#if PFA
  // Chunks starting at a line that is a multiple of PFA_BH take their carry from column x - 1 (as the chunk at line 0 does).
  if (gy && pfaRotatedLine(line)) { prevCol = (G_W * gx + me + WIDTH - 1) % WIDTH; }
#endif

  CarryABM carry = carryIn[WIDTH * prevLine + prevCol];

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = i * WIDTH + me;
    bool biglit0 = frac_bits + (2*i) * FRAC_BPW_HI <= FRAC_BPW_HI;
    bool biglit1 = frac_bits + (2*i) * FRAC_BPW_HI >= -FRAC_BPW_HI;   // Same as frac_bits + (2*i) * FRAC_BPW_HI + FRAC_BPW_HI <= FRAC_BPW_HI;
    // carryB has no carry-out: a carry leaving the last word of this group would be dropped, silently
    // losing 1 ulp at the first word of the next group.  On the last word pair, add the carry into the
    // high word without normalizing it -- as carryFinal does at the end of the fused carry chain -- so
    // that nothing can escape the group.  An un-normalized word holds the same value and is normalized
    // by the next iteration.
    if (i == CARRY_LEN - 1) {
      io[p] = carryWordLast(io[p], carry, biglit0);
      return;
    }
    io[p] = carryWord(io[p], &carry, biglit0, biglit1);
    if (carryIsZero(carry)) { return; }
  }
}
