// Copyright (C) Mihai Preda and George Woltman

#pragma once

// P-1 stage 1 inside a PRP test (-prpPm1 <B1>[,<B2>]), from the PRP's own squarings.
//
// The PRP residue after k squarings is u_k = 3^(2^k), so the stage 1 residue 3^E (E = pm1::stage1Exponent(p, B1)) is the product of
// the u_k over the set bits k of E.  It is collected by the k-ary method:  the bits of E are cut into windows of L bits at the grid
// s, s + L, s + 2L, ... from the start s of a segment, and at grid point w the PRP residue u_w is multiplied into the bucket B[v],
// v = bits [w, w + L) of E (v = 0 included).  At the end of a segment, prod_v B[v]^v is the product of the u_k over the set bits of
// the segment;  with the residue of the earlier segments, X, put in B[1] when the segment starts, it is the new X.  One multiplication
// every L squarings instead of one squaring per bit:  about 1/L of a stand-alone stage 1, for 2^L buffers of GPU memory.
//
// The PRP's Gerbicz check covers the u_k.  The accumulator has a check of its own, made when a segment is folded, at every verified
// PRP check (so its savefile is written beside each verified PRP savefile):
//   T = prod_v B[v] = X * W, W = u_s * u_{s+L} * ... * u_{s+(n-1)L}, and W^(2^L) * u_s = W * u_{s+nL} (Gerbicz):
//     T^(2^L) * u_s * X = T * u_{s+nL} * X^(2^L)
//   checks every multiplication into a bucket.  The fold computes prod_v B[v]^v by the running product from the top, and also the
//   product prod_v B[v]^(2^L - 1 - v) by the running product from the bottom, by other multiplications:  the two running products end in
//   T, and  prod_v B[v]^v * prod_v B[v]^(2^L - 1 - v) * T = T^(2^L)  checks the fold.
// A failed check goes back to the last verified PRP savefile and its accumulator savefile.
//
// When all the bits of E are in X (the PRP is at about iteration log2(E) = 1.44 B1), the next PRP check is the last fold, and then:
// the GCD on the GPU, the P-1 result line ("integrated":true), stage 2 if B2 > B1 (the PRP waits for it), and a factor ends the PRP.
// When -prpPm1 is not given none of this exists:  no buffers, no savefiles, no kernels; the PRP loop is unchanged.

#include "AllocTrac.h"
#include "Buffer.h"
#include "common.h"
#include "Saver.h"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

class Gpu;

// The P-1 result of a PRP test:  writes the result line (B2 = 0 for a result of stage 1 alone).
using Pm1Report = std::function<void(u64 B1, u64 B2, const vector<string>& factors, u32 nErrors)>;

class Pm1Prp {
public:
  // Null when the P-1 can't be done in this PRP test (B1 too large for the exponent:  it is logged)
  static std::unique_ptr<Pm1Prp> make(Gpu& gpu, u64 B1, u64 B2, Pm1Report report);
  ~Pm1Prp();

  // After the PRP loaded `state`:  restores the accumulator of the same iteration, or brings the PRP back to the last iteration that
  // has one (and returns that state).  Without one (a test started without -prpPm1, the savefiles lost) the P-1 is given up, with a log
  // line.  A finished stage 1 waiting for stage 2 (a stop in stage 2) runs it now.
  PRPState sync(PRPState state);

  // The next PRP iteration that has an accumulator step, u64(-1) when there is none
  [[nodiscard]] u64 nextK() const { return active ? next : u64(-1); }
  // The PRP iteration at which the last bits of E are verified (a PRP check is wanted there), u64(-1) if not yet known or not needed
  [[nodiscard]] u64 checkAt() const { return active ? finalCheck : u64(-1); }

  // Whether the step at nextK needs u_k in words (bufData):  to copy it into a bucket not touched yet, or as u_s (first of a segment).
  // Otherwise it takes u_k from the fused carry's output in buf1, as the next squaring does.
  [[nodiscard]] bool wantsWords() const;
  // At iteration nextK, with u_k in bufData (haveWords) or in buf1 at LEAD_WIDTH.  True when it leaves the transform of u_k in buf1
  // at LEAD_MIDDLE (a multiplication), false after a copy.
  bool step(u64 k, bool haveWords);

  enum Outcome {CONTINUE, RELOAD, FACTOR};
  // At a PRP check that passed at iteration k (PRP residue prp64), before its savefile is written:  folds the segment, checks it and
  // writes the accumulator savefile; when stage 1 is complete, its GCD and result.  RELOAD:  the accumulator check failed.
  // FACTOR:  the result line is written and the PRP must stop.
  Outcome verified(u64 k, u64 prp64);
  // After the PRP savefile of that check is queued:  stage 2 when it is due (it may throw "stop requested").
  Outcome afterSave();

  [[nodiscard]] bool foundFactor() const { return factor; }

private:
  Pm1Prp(Gpu& gpu, u64 B1, u64 B2, Pm1Report report, vector<u32> exp, u64 hiBit);

  struct Val;

  Gpu& gpu;
  u64 const E, B1, B2;
  Pm1Report report;
  vector<u32> const exp;      // the bits of E
  u64 const hiBit;            // its highest set bit
  Saver<P1PrpState> saver;

  bool active = false;        // accumulating
  bool done = false;          // finished, given up, or not possible in this test:  nothing more to do in this process
  bool factor = false;
  bool stage2Due = false;
  u32 L = 0;                  // window length
  u64 segStart = 0;           // the first grid point of the segment
  u64 next = 0;               // the next grid point
  u64 lastStep = 0;           // the last grid point stepped
  bool stepped = false;       // in this segment
  u64 finalCheck = u64(-1);
  u32 nErrors = 0;            // accumulator check failures
  int nSeqErrors = 0;
  u64 nFused = 0, nWordSteps = 0;   // steps of this process, by where u_k came from
  Words X;                    // the verified product of the earlier segments, empty for 1
  u64 doneK = 0, done64 = 0;  // the PRP check (iteration, res64) at which stage 1 was done

  // GPU buffers, only while accumulating
  std::optional<AllocTrac::Allowance> allowance;
  vector<Buffer<Word>> bucket;
  vector<bool> touched;       // a bucket not touched is 1
  std::unique_ptr<Buffer<Word>> u0;
  vector<Buffer<Word>> temp;

  u32 chooseL(u64 bitsLeft, u64 itersLeft, u64 maxBufs) const;
  bool alloc(u64 fromK);
  void release();
  void startSegment(u64 at);
  u32 windowAt(u64 w) const;
  void mulInto(Val& a, const Val& b);
  bool same(Buffer<Word>& a, Buffer<Word>& b);
  bool fold(u64 k, Words& out);
  Outcome finish(u64 k, u64 prp64);
  void giveUp(const string& why, u64 k = u64(-1), u64 prp64 = 0);
  void save(u64 k, u64 prp64, u32 stage);
};
