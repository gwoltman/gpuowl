# P-1 factoring in PRPLL

PRPLL runs P-1 on Mersenne numbers 2^p - 1 entirely on the GPU: stage 1, the paired stage 2, and the GCD for each stage.
It never uses the CPU for bignum work and does not link GMP.
P-1 looks for a factor q = 2kp + 1 whose k is B1-smooth (stage 1), or B1-smooth apart from one prime in (B1, B2] (stage 2).

## Worktodo lines

Put the lines in `worktodo-<N>.txt`, the same file that holds PRP work. The assignment ID (AID) is optional, as in Prime95.

    Pminus1=[AID,]1,2,<p>,-1,<B1>,<B2>[,...]
    Pfactor=[AID,]1,2,<p>,-1,<tfBits>,<testsSaved>[,...]

- **`Pminus1=`** runs P-1 with the bounds given. With `B2 <= B1`, including `B2 = 0`, only stage 1 runs. B1 must be at least 2.
  PRPLL reads but does not use the fields Prime95 allows after B2 (the trial-factoring depth, `B2_start`, and the list of known factors).
- **`Pfactor=`** runs P-1 with bounds that PRPLL chooses (see [Bounds](#bounds-of-pfactor-lines)).
  - `tfBits` is the depth of trial factoring already done: no factor is below 2^tfBits. It must be in [0, 100).
  - `testsSaved` is the number of primality tests a factor would save. It means the same as in Prime95's Pfactor lines (for
    example 1 or 2). It must be at least 0, and values above 100 are capped at 100, as Prime95 caps them.
- Only k=1, b=2, c=-1 (Mersenne numbers) is accepted. The exponent must be prime and must fit an FFT. Any other line is ignored.

When the work is done, PRPLL deletes the line and appends a JSON result to `results-<N>.txt`:

    {"status":"NF", "exponent":41000083, "worktype":"P-1", "b1":126000, "b2":2790000, "fft-length":1835008, "errors":{"gerbicz":0}, ...}
    {"status":"F", "exponent":1000099, "worktype":"P-1", "b1":4370, "b2":540000, ..., "factors":["155058493988826487335266033969"], ...}

- `b1` and `b2` are the bounds that were actually covered. `b2` is present only when stage 2 ran.
- Factors are given in decimal. A GCD that is a product of several factors is reported whole, as Prime95 does: it is not split.
- `errors` holds the number of failed Gerbicz checks in stage 1. With `-use PM1_JACOBI=1` it also holds the number of failed Jacobi checks.

## Stage 1

Stage 1 computes x = 3^E mod 2^p - 1, where E = 2p times the product, over the primes q <= B1, of the largest power of q that is <= B1.
This E is the same as mprime's, so the residues compare equal to mprime's.
- It uses one squaring per bit of E, about 1.44 B1 bits. Each squaring also multiplies by 3 when its bit is set.
- On an RX 7900 XTX it runs about 0.6% slower per iteration than PRP on the same FFT.
- PRPLL keeps E itself in the `p1-<p>/` folder.

**Gerbicz check.** Every iteration is covered by a Gerbicz-type check. With u_j the residue after j sub-blocks of a block and c_j the
exponent chunk that sub-block consumes, checksum 1 is the running product u_0 * ... * u_m and checksum 2 is u_0 * P^(2^L) * 3^C
(P = u_0 * ... * u_{m-1}, C = sum c_j); unlike PRP's check it starts over with every block.
- A block of L sub-blocks of L iterations is verified at its end. L is about sqrt(bits / 20), so a run has about 20 checks.
  The check costs about 2/L of stage 1.
- PRPLL saves only verified residues, in `p1-<p>/`.
- A failed check goes back to the last savefile with a shorter block. The number of failures goes in the result.

**`-use PM1_JACOBI=1`** adds a Jacobi-symbol check on the GPU.
- It checks every savefile that is loaded: (x | 2^p-1) must be (-1)^(the top k bits of E).
- It checks the final residue: the symbol must be +1.
- Each check costs about as much as one GCD, so it is off by default.

**GCD.** After stage 1, PRPLL computes gcd(x - 1, 2^p - 1) on the GPU. This is a quadratic GCD; on the XTX it takes 2 s at 10M
bits, 17.6 s at 41M and 85 s at 100M. A factor ends the assignment. There is no CPU GCD: if the GPU GCD can't run, the residue goes
to mprime when `-mprimeDir` is given, and otherwise the run stops with an error.

## Stage 2

Stage 2 is the paired "n^2" form of the old gpuowl:
- It multiplies an accumulator by x^((kD)^2) - x^(r^2) for each needed (k, r). Because (kD)^2 - r^2 = (kD - r)(kD + r), one
  multiplication covers both kD - r and kD + r when both are primes in (B1, B2].
- About 25-40% of the primes are paired this way. The rest cost one multiplication each.
- The table holds x^(r^2) for r < D/2 coprime to D. D runs from 210 (24 table buffers) to 4620 (480).
- The difference A_k - B_r is taken inside the tail multiplication kernel.
- One multiplication costs about 1.1-1.2 PRP iterations (XTX).
- The run ends with gcd(acc, 2^p - 1) on the GPU.

**Checks.** No Gerbicz-style check exists for a product like this, so stage 2 uses these checks instead:
- The table is built twice and its checksums are verified again at every checkpoint.
- The last table entry and A_k are checked against a direct exponentiation.
- acc must be nonzero.
- The round-off error (ROE) of every carry is checked.
- One block in 32, chosen at random, is computed twice and the results compared.

A failed check goes back to the last checkpoint. Checkpoints are written to `p2-<p>/`, at least every 10 minutes.

**Memory and `-maxAlloc`.** Stage 2 takes as many big buffers as its memory budget allows: D's table plus 5 more.
- The budget is the free GPU memory when stage 2 starts, less 1 GB.
- Where the free memory can't be queried (OpenCL without `CL_DEVICE_GLOBAL_FREE_MEMORY_AMD`, e.g. Nvidia's OpenCL or PoCL), it is
  90% of the GPU's RAM, less what PRPLL holds and 256 MB.
- `-maxAlloc <size>` caps the budget at `<size>`, less what PRPLL holds and 256 MB. The cap is counted per device.
- Without `-maxAlloc` there is no cap, and PRP's default allocation limit is raised by the budget while stage 2 runs.
- Only one stage 2 runs per device at a time. A lock file `memlock-<device uid or index>` in the work directory enforces this, so
  two PRPLL processes on one GPU must share a directory for the lock to work.
- Below 24 + 5 buffers, stage 2 declines.

**When stage 2 declines** (too little memory, or no FFT that can run it):
- With `-mprimeDir`, the stage 1 residue goes to mprime (next section).
- Without it, the result is written for B1 alone (no `b2`) and the `p1-<p>/` savefile is kept.

**Stopping.** `-iters <N>` stops stage 1 after N iterations, or stage 2 after about N multiplications. A signal also stops the run.
In both cases the worktodo line and the savefiles are kept. A restart loads the finished stage 1 and resumes stage 2 from its
savefile.

## Hand-off to mprime: `-mprimeDir <dir>`

With `-mprimeDir`, PRPLL writes the stage 1 residue as a Prime95 savefile in `<dir>` when stage 2 can't run on the GPU:
- The file is version 8, state DONE, with B_done = C_done = B1, under Prime95's own file name (e.g. `m0400381`, or `mP000003` for 25000003).
- PRPLL appends a `Pminus1=` line with the bounds to `<dir>/worktodo.add`.
- mprime (30.10 or later) resumes such a file into stage 2 when the worktodo line's B2 is above C_done: it runs the stage 1 GCD and
  stage 2 and reports the result. PRPLL deletes its own line and writes no result.
- A hand-off for the GCD alone asks for B2 = B1 + 1, because mprime runs the stage 1 GCD only on its way into stage 2.
- mprime rejects lines with B1 < 50000 unless its `prime.txt` has `AllowLowB1=1`.
- PRPLL's own tests of the hand-off (test/pm1/mprime.sh) used an mprime that can dump its stage 1 residue, to compare it with
  PRPLL's; with an mprime that can't, the script skips that comparison.

`-mprimeDir` is optional: without it, PRPLL does all the work it can itself.

## FFT choice

- **No `-fft`:** stage 2 uses the fastest fitting line of `tune-pm1.txt` (written by `-tune pm1`), then of `tune.txt`, then the
  untuned candidates. The candidates are the smallest fitting shape of each type with an NTT and no FP64 (61, 3231, 3261, 3161,
  323161), then the FP64 types (64, 6431).
  - For a stage 2 above about B2 = 8M, the untuned candidates are timed for a few seconds and the fastest is used.
  - Stage 1 takes the same order but is not timed.
- **Stage 2 margin:** stage 2's FFT needs 0.5 bits per word of margin over the PRP limit. The difference of two residues doubles
  the input, and stage 2 has no Gerbicz check.
- **Kernel build failure:** if an FFT's kernels can't be built, the next FFT is used.
- **`-fft`:** an `-fft` spec is never replaced. If its FFT can't run a stage, PRPLL logs why and stops.
- **Unsupported types:** stage 2 does not support M31 or FP32 alone (FFT31, FFT32).

**`-tune pm1`** times stage 2 multiplications, not PRP squarings, for every FFT type stage 2 supports.
- It writes the fastest FFT per exponent range to `tune-pm1.txt`, with its stage 2 settings (INPLACE, TAIL_KERNELS, MULTI_Q,
  L2_STRIPING).
- It does not change `tune.txt` or `config.txt`.
- See `prpll -h` for the options it takes.

## Bounds of Pfactor lines

The bounds maximise the expected saving, in PRP iterations of p:

    saving = P(factor) * testsSaved * p - (stage 1 + GCD + (1 - P(stage 1 factor)) * (stage 2 + GCD))

**Probability** (`src/Pm1Bounds.cpp`). The probability model is the Dickman rho model of the old gpuowl bounds calculator
(`pm1/pm1.cpp`); its references are Brent, and Kruppa, chapter 5.3.3.
- It uses factors of the form 2kp + 1, with no factor below 2^tfBits.
- It agrees with mprime's `pm1prob` (a copy of the same model) to within 0.2% wherever the probability is above 0.1%.

**Costs** (measured on the RX 7900 XTX):
- Stage 1: 1.006 PRP iterations per bit of E, plus the Gerbicz check.
- Stage 2: 1.15 PRP iterations per multiplication. The multiplications counted are:
  - one per needed (k, r), from an estimate of the paired fraction (D/phi(D)) / ln q, within 1% of the planner's exact counts;
  - two per k;
  - the table (built twice);
  - the duplicated blocks.
- Each GCD: 10000 + 0.0023 p iterations.

**Memory.** D is the one stage 2 would choose for the memory it would have. This is estimated at the start of the assignment from
the free memory with the stage 1 FFT's buffers allocated, and the buffer size of stage 2's first FFT.
- If the memory can't hold the smallest table, B2 = B1 (stage 1 only), and the log says so. Stage 1 then runs even when it does not
  pay for itself alone, as long as P-1 with a stage 2 would.
- If no bounds save anything even with stage 2's largest table, the line is removed without a result (as Prime95 does). This happens,
  for example, with `testsSaved = 0`, or at small exponents, where the GPU GCD is a large share of the cost.

**Logging and resumes.** The chosen bounds are logged with their chance of a factor, their cost and the saving. They are stored
in the P-1 savefile (`p1-<p>/`, header version 2), so a resume keeps them. A version 1 savefile has only B1; for it, PRPLL chooses B2.

**Compared with mprime 31.07.** mprime's bounds come from its own `Pfactor=` run.
- *Small memory:* about 35 temporaries (Memory = 450, 650, 900, 1700, 1700 and 2000 MB, from the top row down). With fewer than 40
  temporaries, mprime costs only its pairing stage 2. PRPLL's bounds are given for room for 30 table buffers (D = 210).
- *16 GB:* mprime costs its polynomial stage 2 too. PRPLL's bounds are given for room for 1300 table buffers, as on a 24 GB GPU at
  41M.

| p | tfBits | testsSaved | mprime, small memory | PRPLL, 30 buffers | mprime, 16 GB | PRPLL, 1300 buffers |
|---|---|---|---|---|---|---|
| 30000001 | 72 | 1 | B1=95000 B2=1840000, 2.12% | B1=97100 B2=1950000, 2.16% | B1=178000 B2=43830000, 4.45% | B1=101000 B2=2350000, 2.28% |
| 41000017 | 74 | 1 | B1=123000 B2=2400000, 1.97% | B1=123000 B2=2480000, 1.99% | B1=210000 B2=36635000, 3.79% | B1=126000 B2=2790000, 2.06% |
| 60000011 | 75 | 1 | B1=185000 B2=3930000, 2.27% | B1=186000 B2=4060000, 2.29% | B1=288000 B2=40275000, 3.87% | B1=194000 B2=4940000, 2.41% |
| 110000017 | 77 | 1 | B1=336000 B2=7955000, 2.58% | B1=343000 B2=8250000, 2.61% | B1=472000 B2=42865000, 3.76% | B1=357000 B2=10100000, 2.74% |
| 110000017 | 77 | 2 | B1=720000 B2=19825000, 3.48% | B1=745000 B2=21100000, 3.54% | B1=960000 B2=98345000, 4.78% | B1=774000 B2=25800000, 3.69% |
| 332000027 | 81 | 1 | B1=947000 B2=21700000, 2.86% | B1=1010000 B2=28000000, 3.04% | B1=1116000 B2=47045000, 3.40% | B1=1040000 B2=34300000, 3.17% |

- **Against mprime's pairing stage 2**, which costs about what PRPLL's does, the bounds are close: B1 within 1.00-1.07x and B2
  within 1.03-1.29x of mprime's.
- **Against mprime's polynomial stage 2**, PRPLL's B2 is 1.4-19 times lower: 19x at 30M, 1.4x at 332M. mprime's polynomial stage 2
  covers a prime far more cheaply than a paired stage 2 can, and PRPLL has no polynomial stage 2.
- **At p = 5000011** (tf 66), mprime runs P-1 (B1 = 28000-39000) but PRPLL's model says it does not pay. The two GPU GCDs cost
  about 0.9% of a PRP test there.
- **Probability:** at mprime's own bounds, PRPLL's model gives mprime's printed chance of a factor to within 1% (relative) in
  every case above.

## P-1 inside a PRP test (`-prpPm1`)

`-prpPm1 <B1>[,<B2>]` runs P-1 stage 1 on the PRP test's own squarings. It is off by default: without it a PRP test has no extra
buffer, kernel or savefile, and the PRP loop does the same work per iteration as before.

- **How:** the PRP residue u_k = 3^(2^k), and 3^E is the product of the u_k over the set bits k of E. Every L iterations the
  residue is multiplied into one of 2^L buckets chosen by the L bits of E at that point (one multiplication per L squarings). The
  buckets are combined when the PRP passes the last set bit of E, at iteration about 1.44 * B1.
- **Cost** (7900 XTX, measured): about 13.5% of a stand-alone stage 1 (35 s over a 250 s PRP at 40M with B1 = 1M; 86 s over 515 s at
  100M), plus the GCD (17 s at 40M, 87 s at 100M). As a share of a full 100M PRP test (about 9.9 h) that is 0.5%, 1.4% and 2.6% for
  B1 = 1M, 5M and 10M (the 5M and 10M figures are extrapolated), against running stage 1 on its own first.
- **Memory:** 2^L residue buffers, L chosen (about 8 to 11) from the memory free on the GPU less 1 GB; `-maxAlloc` caps it.
- **Checks:** the PRP's Gerbicz check covers the squarings. The accumulator has its own exact check at every fold, a Gerbicz-style
  identity over the bucket grid that covers every multiplication into a bucket and the fold itself. A failed check rolls back to the
  last accumulator savefile; after 3 failures in a row P-1 is given up and the PRP carries on.
- **Savefiles:** `p1prp-<exponent>/`, next to each verified PRP savefile. The PRP savefile format is unchanged. After a PRP
  rollback to an iteration with no accumulator save, the accumulator goes back to its last save (at most 2 check intervals,
  otherwise P-1 is given up with a log line).
- **Only for a PRP test started with the flag** (from iteration 0). Turned on mid-run it logs that P-1 is not possible and the PRP
  continues.
- **Result:** at completion the GPU GCD runs and a P-1 result line is written with `"integrated":true`. A factor ends the PRP test
  and deletes the worktodo line. With B2 > B1 and no factor, stage 2 runs while the PRP test waits; a failure in stage 2 gives a B1-only
  result and the PRP goes on.
- The bounds of `Pfactor=` are not wired to `-prpPm1`; give B1 (and B2) explicitly.

## Not implemented

- **Polynomial (FFT) stage 2** as in mprime (Montgomery-Silverman, Montgomery-Kruppa), and prime pairing beyond the natural kD +- r
  pairs. B2 is limited to 2^32 - 8192.
- **CPU fallbacks:** no CPU GCD or bignum fallback, by design.
- **Unused line fields:** the known factors and `B2_start` of the worktodo lines.
- **Factor splitting:** a composite GCD is not split into its factors.
- **Certificates:** no P-1 certificate. None exists in mprime or Mlucas either.
- **CUDA:** CUDA graphs do not cover the multiplication by 3, so stage 1 runs below PRP speed under CUDA.
- **Cost model:** it is the XTX's. On other GPUs, in particular for the GCD, the costs differ, and the model has no setting for that.
