[![Actions Status](https://github.com/preda/gpuowl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/preda/gpuowl/actions/workflows/ci.yml)

## Must read papers

### Multiplication by FFT

- [Discrete Weighted Transforms and Large Integer Arithmetic](https://www.ams.org/journals/mcom/1994-62-205/S0025-5718-1994-1185244-1/S0025-5718-1994-1185244-1.pdf), Richard Crandall and Barry Fagin, 1994
- [Rapid Multiplication Modulo the Sum And Difference of Highly Composite Numbers](https://www.daemonology.net/papers/fft.pdf), Colin Percival, 2002

### P-1

- [An FFT Extension to the P-1 Factoring Algorithm](https://www.ams.org/journals/mcom/1990-54-190/S0025-5718-1990-1011444-3/S0025-5718-1990-1011444-3.pdf), Montgomerry & Silverman, 1990
- [Improved Stage 2 to P+/-1 Factoring Algorithms](https://inria.hal.science/inria-00188192v3/document), Montgomerry & Kruppa, 2008


# PRPLL

## PRobable Prime and Lucas-Lehmer mersenne categorizer
(pronounced *purrple categorizer*)

PRPLL implements two primality tests for Mersenne numbers: PRP ("PRobable Prime") and LL ("Lucas-Lehmer") as the name suggests.

PRPLL is an OpenCL (GPU) and CUDA program for primality testing Mersenne numbers.


## Build

Invoke `make` in the source directory.

### Windows (MSVC)

Build with `msbuild PRPLL.sln /p:Configuration=OpenCL-Release` (or
`CUDA-Release`, `OpenCL-Debug`, `CUDA-Debug`), or open `PRPLL.sln` in Visual
Studio and build the matching configuration. Like `make`, the project
generates `src/bundle.cpp` (with `genbundle.sh`) and `src/version.inc` (with
`git describe`) itself, so it needs Git for Windows on `PATH`: its `bash.exe`
runs `genbundle.sh`. Pass `/p:Bash=path\to\bash.exe` to use another bash, and
`/p:PrpllVersion=...` to set the version string of a build without `.git`.


## Use
See `prpll -h` for the command line options.

P-1 factoring (`Pminus1=` and `Pfactor=` worktodo lines, stage 1 and stage 2 on the GPU, the hand-off to mprime) is described in
[docs/pm1.md](docs/pm1.md).


## License

This project is licensed under the **GNU General Public License v3.0** - see [LICENSE](LICENSE) for details.


## Credits

[PRPLL](https://github.com/preda/gpuowl) was originally authored (as gpuowl) by **Mihai Preda**.  **George Woltman** authored optimizations and NTT code contributions.  The CUDA backend was authored by **"Sherpa"** in honor of John Allen Frey.


## Why LL

For Mersenne primes search, the PRP test is by far preferred over LL, such that LL is not used anymore for search.
But LL is still used to verify a prime found by PRP (which is a very rare occurence).

### Lucas-Lehmer (LL)
This is a test that proves whether a Mersenne number is prime or not, but without providing a factor in the case where it is not prime.
The Lucas-Lehmer test is very simple to describe: iterate the function f(x)=(x^2 - 2) modulo M(p) starting with the number 4. If
after p-2 iterations the result is 0, then M(p) is certainly prime, otherwise M(p) is certainly not prime.

PRPLL checks an LL test with a Jacobi symbol: (s - 2 | M(p)) must be -1 at every iteration after the first. The check catches
about half of all errors: an error is caught at the first check after it, or never, so checking more often does not catch more;
it only shortens the rollback. The final residue is always checked, and on a failure the test goes back to the last residue
that passed. `-use LL_JACOBI=0` turns the check off. Unlike PRP's Gerbicz check, this is no protection against every error.

Lucas-Lehmer, while a very efficient primality test, still takes a rather long time for large Mersenne numbers
(on the order of weeks of intense compute), thus it is only applied to the Mersenne candidates that survived the cheaper preliminary
filters TF and P-1.

### PRP
The probable prime test can prove that a candidate is composite (without providing a factor), but does not prove that a candidate
is prime (only stating that it _probably_ is prime) -- although in practice the difference between probable prime and proved
prime is extremely small for large Mersenne candidates.

The PRP test is very similar computationally to LL: PRP iterates f(x) = x^2 modulo M(p) starting from 3. If after p iterations the result is 9 modulo M(p), then M(p) is probably prime, otherwise M(p) is certainly not prime. The cost
of PRP is exactly the same as LL.

### Shift (`-shift`)
With `-shift <N>` (or `-shift random`) a PRP test or a CERT computes on x * 2^s modulo M(p) instead of x, as Prime95 and Mlucas
do with their "shift count". Multiplying by 2^s is a rotation of the p-bit residue, so it costs nothing in the FFT: a squaring
doubles the shift (s becomes 2s mod p) and the PRP multiply by 3 keeps it, so PRPLL only tracks s on the host. Two runs of the
same test with different shifts then put different data through the GPU, while every residue that leaves the GPU is rotated
back: the logged res64, the residues in the savefiles, proof residues, the final res64 and res2048 are exactly those of an
unshifted run (and the proof is the same). The results line always has "shift-count": the shift of the final residue, as
Prime95 reports it (0 without a shift).

`<N>` is the shift at iteration 0; `random` draws one when the test starts. As in Prime95 and Mlucas, the whole test then
keeps that one shift: the savefiles (PRP savefile version 14, CERT version 3) store the shift of their iteration, and a
restart, or the rollback after a failed Gerbicz check, goes on with the shift of the savefile it loads; `-shift` is not
applied to it. Only a savefile of an older version, which has no shift, takes the `-shift` of the command line. Older PRPLL
builds can't read the new savefiles. A shifted PRP rebuilds its data from the error check on every start
(blockSize squarings, as a resume does). The error check holds with the shift: the check is shifted by a = s / (2^blockSize - 1)
mod p, and for the rare exponents where p divides 2^blockSize - 1 PRPLL keeps the check unshifted and multiplies by 3 * 2^s
instead of 3. LL tests ignore `-shift`, and a shifted PRP test gets no P-1 inside it (that needs the unshifted residues).
Without `-shift`, a PRP test from the worktodo file that has no P-1 inside it (tests-saved 0, or P-1 declined) runs on a
random shift, logged as "random: no P-1 for this test"; a PRP test with P-1, a `-prp` test (often a timing run) and a CERT
run unshifted. Any `-shift`, also `-shift 0` (no shift), means no P-1 inside the PRP test.
