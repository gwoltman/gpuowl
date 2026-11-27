# PRP tests of cofactors of 2^E - 1 (-cofactor)

`-cofactor` lets PRPLL PRP-test N = (2^E - 1) / F, a cofactor of 2^E - 1 whose divisor F is known, for example
N = (2^(pq) - 1) / ((2^p - 1)(2^q - 1)) with p and q prime. E may be composite. PRPLL does the work on the GPU, working
mod 2^E - 1 as in a normal PRP test, and writes a residue file. `tools/cofactor-check.c`, a small GMP program, then
computes N and gives the verdict: base-3 PRP or composite.

## Quick start

Build the checker once:

    cc -O2 -o cofactor-check tools/cofactor-check.c -lgmp

For E = 866409 = 3 * 288803, F = (2^3 - 1)(2^288803 - 1):

    ./cofactor-check --exp 866409 2^3-1 2^288803-1       # writes 866409.n
    prpll -cofactor -prp 866409                          # writes 866409.pow3n
    ./cofactor-check 866409.pow3n 866409 2^3-1 2^288803-1

The last command prints `N = (2^866409-1)/F, 173876 digits:  base-3 PRP`, or `composite`. Its exit code is 0 for a PRP,
1 for composite and 2 for an error.

To queue several tests, run `cofactor-check --exp` for each E, put one line per E in `worktodo-0.txt`, and run
`prpll -cofactor` (on the command line, not in config.txt):

    PRP=1,2,866409,-1
    PRP=1,2,<next E>,-1

The factors of F are given as `2^k-1` or in decimal, any number of them, as long as their product divides 2^E - 1.

## The two ways PRPLL can run

**With `<E>.n`** (written by `cofactor-check --exp`): PRPLL computes x = 3^N mod 2^E - 1 by square-and-multiply over the
bits of N, about log2(N) iterations, and writes x to `<E>.pow3n`. N divides 2^E - 1, so x mod N = 3^N mod N, and N is a
base-3 Fermat PRP if that is 3. This is the faster way: for F = (2^p - 1)(2^q - 1) it skips p + q of the E iterations,
which is a third of them for p = 3 and a fifth for p = 5. The loop is P-1 stage 1's, with its Gerbicz-type check every
L^2 iterations, its savefiles and its resume.

**Without `<E>.n`**: PRPLL runs the normal PRP squarings, E iterations, under the usual Gerbicz check, and writes
R = 3^(2^E) mod 2^E - 1 to `<E>.cofactor`. Since 2^E = N F + 1, a base-3 PRP N has R = 3^(F + 1) mod N (the PRP-CF test
GIMPS uses for Mersenne cofactors). `cofactor-check <E>.cofactor E f1 f2 ...` tests that.
For E = p q with primes p < q it is quick: PRPLL also saves the residues at iterations p, q and p + q, as
`<E>.cofactor.<k>`, and since F + 1 = 2^(p+q) - 2^p - 2^q + 2 for F = (2^p - 1)(2^q - 1), the check becomes
R * 3^(2^p) * 3^(2^q) = 9 * 3^(2^(p+q)) mod N, a few multiplications. For any other F, or without those files,
`cofactor-check` computes 3^(F + 1) mod N, which takes about log2(F) squarings mod N and is slow when F is large.

For a Wagstaff number, E = 2q and F = 3(2^q - 1), so N = (2^q + 1) / 3; both ways work. There F + 1 = 9(N - 1) + 4, so R is
81 mod N when N is a PRP, but that shortcut holds only for p = 2.

## What -cofactor changes

- PRP lines and `-prp` may have a composite E (without -cofactor such a line is ignored, and `-prp` moves to the previous prime).
- No PRP proof and no P-1 inside the PRP test: both assume a prime E. `-proof <N>` with N > 0, `-ll`, `-verify` and
  `-use PRPPM1` / `PRPPM1_B1` / `PRPPM1_B2` together with `-cofactor` are an error.
- No line is written to `results-<N>.txt`: the PP / CC line in the log is about 2^E - 1, not N. The verdict on N comes
  only from `cofactor-check`.

## Files

All residue files hold E bits as little-endian 32-bit words, ceil(E / 32) of them. GMP reads one with
`mpz_import(z, nWords, -1, 4, 0, 0, data)`.

| File | Written by | Contents |
|---|---|---|
| `<E>.n` | `cofactor-check --exp` | N |
| `<E>.pow3n` | `prpll -cofactor`, with `<E>.n` | 3^N mod 2^E - 1 |
| `<E>.cofactor` | `prpll -cofactor`, without `<E>.n` | 3^(2^E) mod 2^E - 1 |
| `<E>.cofactor.<k>` | `prpll -cofactor`, without `<E>.n`, for E = p q | 3^(2^k) mod 2^E - 1, for k = p, q, p + q |

`cofactor-check --ref E` recomputes `<E>.cofactor` with GMP (E squarings, slow) and compares, to test a PRPLL build.

## Limits and hints

- The smallest FFT handles E from about 640000; below that PRPLL reports "No FFT".
- An E that shares a factor with the FFT size gives much larger roundoff errors, and PRPLL warns about it. Choose an FFT
  of another size with `-fft`.
