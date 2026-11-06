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

**GCD.** After stage 1, PRPLL computes gcd(x - 1, 2^p - 1) on the GPU. This is a quadratic GCD; on the XTX it takes 1.3 s at 10M
bits, 10 s at 41M and 51 s at 100M. A factor ends the assignment. There is no CPU GCD: if the GPU GCD can't run, the residue goes
to mprime when `-mprimeDir` is given, and otherwise the run stops with an error.

## Stage 2

Stage 2 is the paired "n^2" form of the old gpuowl, with a larger table and primes paired across D-sections:
- The table holds B_n = x^(n^2) for every n < M coprime to D (D even, so every such n is odd). The run steps A_k = x^((kD)^2).
- Each multiplication of the accumulator by A_k - B_n is an "item". Because (kD)^2 - n^2 = (kD - n)(kD + n), one item covers both
  kD - n and kD + n.
- Two primes q1 > q2 share an item exactly when q1 + q2 = 0 (mod 2D) and q1 - q2 < 2M; then k = (q1 + q2) / 2D and n = (q1 - q2) / 2.
  With n > D/2 the two primes are in different D-sections. A prime without a partner is a single, with k = round(q/D) and n < D/2.
- With M = D/2 (L = 2M/D = 1) this is the old plan, where each prime has exactly one possible partner. Each larger L gives each prime
  about one more candidate partner.
- **Pairing.** The primes of a class pair {c, -c mod 2D} form a bipartite graph in which each prime's possible partners are an interval
  of the other class. Matching each prime with the oldest pending prime of the other class that is still within 2M is therefore a
  maximum matching, so the plan has the fewest items for its table. The tests check it against Hopcroft-Karp.
- The pairing runs on windows of at least 2^22 integers (and at least 512 * 2M). Each window is planned on its own, so a resume or a
  rollback rebuilds the items of one window, and the host keeps only one window's items (at most a few MB) at a time. The cost is under
  0.1% of the pairs.
- The plan needs no floating point and depends only on B1, B2, D and M. It takes about 9 s at B2 = 1e9 (48M primes) and 32 s at 4e9.
- Every prime factor of D must be at most B1: those primes are in the stage 1 exponent, and a prime of (B1, B2] that divided D could
  not be covered. D is a multiple of 30, 210 or 2310 (or 330, 462), from 30 to 9240, so stage 2 needs B1 >= 5.
- The difference A_k - B_n is taken inside the tail multiplication kernel.
- One multiplication costs about 1.1-1.2 PRP iterations (XTX).
- The run ends with gcd(acc, 2^p - 1) on the GPU.

**Costs**, in multiplications by A_k - B_n (XTX, 41M, measured):
- one item: 1;
- one k step: 2.0. A_{k+1} = C_k A_k goes straight to its transform, and C_{k+1} = C_k G back to the half-transformed state that both
  products read, so C_k has no words. The accumulator's chain of multiplications stays in transform form across the k step, unless the
  next block is duplicated or a checkpoint is due. (2.6 when C_k was kept in words and every block ended in words.)
- the table: about 2.5 per odd n < M at D = 630 (2.0 for the two multiplications, which are a k step's, and 1.1 per entry for its
  check transform and sums).

**Choice of D and M.** Stage 2 takes the D and M with the least estimated cost for its memory and B2. The pairing estimate comes from a
table made from exact plans; it is within 0.7% of the exact item counts (check_bounds). The log shows the chosen D and M, the pairs
(and how many are across sections), the multiplications, the estimated cost, and what the old plan would have cost. For example, at
B1 = 250, B2 = 30M (1.86M primes):

| table entries | D | M (L) | primes paired | of them across sections | items | k steps | est. cost | vs. old plan |
|---|---|---|---|---|---|---|---|---|
| 4 | 30 | 15 (1) | 24.1% | 0 | 1634115 | 999993 | 3.76M | (no room) |
| 8 | 60 | 30 (1) | 23.5% | 0 | 1639126 | 499997 | 2.73M | (no room) |
| 24 | 210 | 105 (1) | 27.1% | 0 | 1606305 | 142857 | 1.95M | (no room) |
| 100 | 210 | 438 (4.2) | 61.8% | 435866 | 1283960 | 142857 | 1.62M | -9% |
| 240 | 420 | 1050 (5) | 66.4% | 492120 | 1241392 | 71429 | 1.43M | -17% |
| 485 | 420 | 2120 (10.1) | 81.1% | 676288 | 1104310 | 71429 | 1.29M | -22% |
| 1000 | 630 | 4374 (13.9) | 86.1% | 738242 | 1058458 | 47620 | 1.20M | -28% |
| 1545 | 630 | 6758 (21.5) | 91.0% | 803071 | 1012239 | 47620 | 1.15M | -31% |
| 1450 (measured run) | 630 | 6342 (20.1) | 90.4% | 794978 | 1017968 | 47620 | 1.16M | -30% |

The old plan (b3807834) took at most 480 table entries (D = 4620) and needed 24 + 5 buffers at least; "vs. old plan" compares the
estimated costs (the old plan's with its own k step and table costs, 3.4, 2.9 and 0.8) with the same number of buffers (with 100 the
old plan takes D = 840, with 240 D = 1470, from 485 on D = 4620: 1.66M, 1592318 multiplications, 30.0% paired).
Measured on the XTX at p = 41000017 (`-fft 1:1K:2:256`, 15 MB buffers, 2 runs each): stage 2 took 305 s with the old plan
(D = 4620) and 247 s with this one in the same memory (D = 420, M = 2154, 493 table buffers, 1244767 multiplications, 81.4% of the
primes paired): 19% less. Both found the same factor. With all of the GPU's memory (22.7 GB, buffers
counted at 16 MB) it took 214-219 s (D = 630, M = 6342, 1450 table buffers, 1113208 multiplications, 90.4% of the primes paired): 28%
less than the old plan.

**Compared with mprime's pairing.** The quoted pairing figures of mprime's relative-prime sets (L sets) are about 33% (2 sets), 61%
(3), 70% (4), 77% (5), 82.5% (6), 87% (7), 91% (8), 96% (10), 98.2% (12) and 99.6% (20). PRPLL's maximum matching pairs, at
B2 around 10M (D = 210), 43% (L = 2), 53% (3), 61% (4), 66% (5), 71% (6), 77% (8), 81% (10), 84% (12), 88% (16), 90% (20), 94% (32).
Its figures are higher at L = 2 and lower from L = 3 on. The difference was not resolved: mprime's figures are for its own benchmark
bounds, and mprime also relocates primes (see [Not implemented](#not-implemented)), which gives each prime more partners. The table
of mprime's one-sided sets (n = r + sD, r < D/2) pairs the same as PRPLL's all-n < M table at equal size (within 0.1%), but needs
twice the range of n to build.

**Checks.** No Gerbicz-style check exists for a product like this, so stage 2 uses these checks instead:
- Each table entry is transformed twice and the two checksums compared. The table checksums are verified again at every checkpoint.
- The last table entry is checked against a direct exponentiation. Every value of the build feeds into it, so this also covers the
  multiplications of the build.
- A_k is checked against a direct exponentiation at every checkpoint.
- acc must be nonzero.
- The round-off error (ROE) of the carries is checked.
- One block in 32, chosen at random, is computed twice and the results compared.

A failed check goes back to the last checkpoint. Checkpoints are written to `p2-<p>/`, at least every 10 minutes. A checkpoint holds
B1, B2, the residue's res64, D, M and the pairing's version. A version 1 checkpoint (no M) is the plan with M = D/2, so it is resumed
as such.

**Memory and `-maxAlloc`.** Stage 2 needs, beyond the stage 2 GPU's own buffers, its table, three more big buffers (A_k and G, in
transformed form, and C_k half transformed) and two residue buffers. The table size is the one with the least estimated cost that fits the budget.
- The budget is the free GPU memory when stage 2 starts, less 1 GB.
- Where the free memory can't be queried (OpenCL without `CL_DEVICE_GLOBAL_FREE_MEMORY_AMD`, e.g. Nvidia's OpenCL or PoCL), it is
  90% of the GPU's RAM, less what PRPLL holds and 256 MB.
- `-maxAlloc <size>` caps the budget at `<size>`, less what PRPLL holds and 256 MB. The cap is counted per device.
- Without `-maxAlloc` there is no cap, and PRP's default allocation limit is raised by the budget while stage 2 runs.
- Only one stage 2 runs per device at a time. A lock file `memlock-<device uid or index>` in the work directory enforces this, so
  two PRPLL processes on one GPU must share a directory for the lock to work.
- Each buffer is counted at its size rounded up to 2 MB (GPU memory pages; see the memory test below for why).
- **Memory test.** Before the table is built, a pattern is copied into every table buffer and read back. A buffer that does not hold
  it, or a table entry that is wrong in two builds in a row, makes stage 2 plan again with a smaller table. On the XTX, a table that
  overfilled the GPU (1547 buffers of 15 MB counted at 15 MB, 24.2 GB counted at 2 MB pages, with 23.7 GB reported free) gave one wrong
  entry, the same one in every build, when the buffers were first written by the table build. Touched by the memory test first, the same
  table held its values but ran 15% slower. Tables of 41 MB buffers held up to 22.2 GB. 2 MB pages fit these results, but so does a fixed
  shortfall of about 1 GB beyond the headroom; which it is was not settled (a full-memory run on an FFT whose buffers are a multiple of
  2 MB would tell). The memory test protects the results either way, not the speed.
- The smallest table has 4 entries (D = 30), so stage 2 runs with 7 big buffers and 2 residue buffers beyond the GPU's own. With a
  small table the k steps dominate, so it is slower:

| exponent (stage 2 FFT) | big buffer | residue buffer | stage 2 GPU's own | stage 2 at least (4 entries, D = 30) |
|---|---|---|---|---|
| 41000017 (1:1K:2:256) | 15 MB | 8 MB | 0.08 GB | 121 MB |
| 110000017 (1:256:11:512) | 41 MB | 22 MB | 0.20 GB | 0.33 GB |
| 3321928097 (1:2K:11:2K, estimated) | 1.29 GB | 0.69 GB | about 6.4 GB | 10.4 GB |

- At 110000017 (XTX, B1 = 250, B2 = 30M) the old plan declines below a cap of about 1.6 GB. This stage 2 ran at caps of 0.8 GB
  (D = 30, 5 entries: 3.6M multiplications, about 39 minutes at 648 us each), 1.5 GB (D = 150, 22 entries: 2.03M, about 21 minutes),
  2 GB (D = 210, 35 entries: 1.82M against the old plan's 1.89M) and 4 GB (D = 420, 85 entries: 1.63M against 1.70M).
- At 3321928097 (estimated from the buffer sizes, not run): stage 2 needs about 6.4 + 10.4 GB, i.e. `-maxAlloc` of at least 17 GB and a
  GPU with about 18 GB free. A 24 GB GPU has room for about 8 table entries there (D = 30 or 60), a 16 GB GPU none. The GPU's own
  6.4 GB is scaled from 110000017's 0.20 GB (3 big and 3 residue buffers and a little more).

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
  - Stage 1 (no `-fft`, no `tune.txt` line) takes the FFT a PRP test of the exponent would take: the smallest that fits, FP64 first
    at equal size. Stage 1 is Gerbicz-checked as PRP is. On the XTX at p = 41000017 this is 896K 4:256:7:256 at about 170 us per
    iteration, where stage 2's order would give FFT61 1.75M at about 193.
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
- Stage 2: 1.15 PRP iterations per multiplication by A_k - B_n, for the plan stage 2 would choose (D and M) and its estimated cost in
  those multiplications (see [Stage 2](#stage-2)): the items, 2.0 per k step, the table, and the duplicated blocks. The paired fraction
  comes from a table of exact plans (`check_bounds pairtable`, real primes) by the density of the primes in the classes coprime to D,
  (D / phi(D)) / ln q, and L; it is within 0.7% of the exact item counts (24 cases from L = 1 to 40, B2 from 1e5 to 4e9).
- Each GCD: 5000 + 0.00125 p iterations.

**Memory.** The plan is the one stage 2 would choose for the memory it would have. This is estimated at the start of the assignment from
the free memory with the stage 1 FFT's buffers allocated, and the buffer size of stage 2's first FFT.
- If the memory can't hold the smallest table (4 entries, D = 30), B2 = B1 (stage 1 only), and the log says so. Stage 1 then runs even when it does not
  pay for itself alone, as long as P-1 with a stage 2 would.
- If no bounds save anything even with stage 2's largest table, the line is removed without a result (as Prime95 does). This happens,
  for example, with `testsSaved = 0`, or at small exponents, where the GPU GCD is a large share of the cost.

**Logging and resumes.** The chosen bounds are logged with their chance of a factor, their cost and the saving. They are stored
in the P-1 savefile (`p1-<p>/`, header version 2), so a resume keeps them. A version 1 savefile has only B1; for it, PRPLL chooses B2.

**Compared with mprime 31.07.** mprime's bounds come from its own `Pfactor=` run.
- *Small memory:* about 35 temporaries (Memory = 450, 650, 900, 1700, 1700 and 2000 MB, from the top row down). With fewer than 40
  temporaries, mprime costs only its pairing stage 2. PRPLL's bounds are given for room for 30 table buffers (the planner takes D = 210,
  M = 132).
- *16 GB:* mprime costs its polynomial stage 2 too. PRPLL's bounds are given for room for 1300 table buffers, as on a 24 GB GPU at
  41M (D = 630, M = 5688, L = 18: about 86% of the primes paired).

| p | tfBits | testsSaved | mprime, small memory | PRPLL, 30 buffers | mprime, 16 GB | PRPLL, 1300 buffers |
|---|---|---|---|---|---|---|
| 30000001 | 72 | 1 | B1=95000 B2=1840000, 2.12% | B1=97400 B2=1990000, 2.18% | B1=178000 B2=43830000, 4.45% | B1=111000 B2=3610000, 2.56% |
| 41000017 | 74 | 1 | B1=123000 B2=2400000, 1.97% | B1=124000 B2=2540000, 2.01% | B1=210000 B2=36635000, 3.79% | B1=141000 B2=4610000, 2.36% |
| 60000011 | 75 | 1 | B1=185000 B2=3930000, 2.27% | B1=186000 B2=4140000, 2.30% | B1=288000 B2=40275000, 3.87% | B1=211000 B2=7460000, 2.67% |
| 110000017 | 77 | 1 | B1=336000 B2=7955000, 2.58% | B1=344000 B2=8410000, 2.62% | B1=472000 B2=42865000, 3.76% | B1=386000 B2=15100000, 3.01% |
| 110000017 | 77 | 2 | B1=720000 B2=19825000, 3.48% | B1=747000 B2=21500000, 3.55% | B1=960000 B2=98345000, 4.78% | B1=828000 B2=38000000, 4.00% |
| 332000027 | 81 | 1 | B1=947000 B2=21700000, 2.86% | B1=1010000 B2=28500000, 3.05% | B1=1116000 B2=47045000, 3.40% | B1=1120000 B2=50700000, 3.45% |

- **Against mprime's pairing stage 2**, which costs about what PRPLL's does, the bounds are close: B1 within 1.01-1.07x and B2
  within 1.05-1.31x of mprime's.
- **Against mprime's polynomial stage 2**, PRPLL's B2 is still lower at the smaller exponents: 12x at 30M, 8x at 41M, 2.8x at 110M.
  At 332M PRPLL's B2 (50.7M) and chance of a factor (3.45%) are at mprime's (47.0M, 3.40%). mprime's polynomial stage 2 covers a
  prime far more cheaply than a paired stage 2 can, and PRPLL has no polynomial stage 2. The larger table of this stage 2 raised
  PRPLL's chance of a factor with 1300 buffers from 2.06% to 2.36% at 41M and from 2.74% to 3.01% at 110M (tests saved 1).
- **At p = 5000011** (tf 66), mprime runs P-1 (B1 = 28000-39000) but PRPLL's model says it hardly pays (B1 = 15900, B2 = 323000 with
  1300 buffers: a saving of 0.002% of a PRP test). The two GPU GCDs cost about 0.45% of a PRP test there.
- **Probability:** at mprime's own bounds, PRPLL's model gives mprime's printed chance of a factor to within 1% (relative) in
  every case above.

## P-1 inside a PRP test (`-prpPm1`)

`-prpPm1 <B1>[,<B2>]` runs P-1 stage 1 on the PRP test's own squarings. It is off by default: without it a PRP test has no extra
buffer, kernel or savefile, and the PRP loop does the same work per iteration as before.

- **How:** the PRP residue u_k = 3^(2^k), and 3^E is the product of the u_k over the set bits k of E. Every L iterations the
  residue is multiplied into one of 2^L buckets chosen by the L bits of E at that point (one multiplication per L squarings). The
  buckets are combined when the PRP passes the last set bit of E, at iteration about 1.44 * B1.
- **Cost** (7900 XTX, measured): about 13.5% of a stand-alone stage 1 (35 s over a 250 s PRP at 40M with B1 = 1M; 86 s over 515 s at
  100M), plus the GCD (10 s at 40M, 51 s at 100M). As a share of a full 100M PRP test (about 9.9 h) that is 0.4%, 1.35% and 2.6% for
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

- **Polynomial (FFT) stage 2** as in mprime (Montgomery-Silverman, Montgomery-Kruppa). B2 is limited to 2^32 - 8192.
- **Relocation** of the primes below B2/m (m a prime not dividing D) to their multiples q*m, which mprime uses to shorten the range of
  D-sections and give the pairing more partners.
- **The table chosen for its pairs:** n with small prime factors outside D pair more often (Hardy-Littlewood); the table is all n < M.
- **CPU fallbacks:** no CPU GCD or bignum fallback, by design.
- **Unused line fields:** the known factors and `B2_start` of the worktodo lines.
- **Factor splitting:** a composite GCD is not split into its factors.
- **Certificates:** no P-1 certificate. None exists in mprime or Mlucas either.
- **CUDA:** CUDA graphs do not cover the multiplication by 3, so stage 1 runs below PRP speed under CUDA.
- **Cost model:** it is the XTX's. On other GPUs, in particular for the GCD, the costs differ, and the model has no setting for that.
