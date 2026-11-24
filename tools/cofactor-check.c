// The GMP half of a PRP-CF test of N = (2^E - 1) / F, from the residue file that "prpll -cofactor" writes.
//
// prpll -cofactor -prp E  writes  E.cofactor:  R = 3^(2^E) mod (2^E - 1), as little-endian u32 words.  Since 2^E = N F + 1,
// R = 3^(N F + 1);  if N is a base-3 Fermat PRP then R = 3^(F + 1) mod N.  This program reads R, reduces it mod N and compares.
//
// For E = p q and F = (2^p - 1)(2^q - 1), F + 1 = 2^(p+q) - 2^p - 2^q + 2, so the test is R 3^(2^p) 3^(2^q) = 9 3^(2^(p+q)) mod N.
// prpll saves those three residues as E.cofactor.<k>;  when they are next to E.cofactor, the check is a few multiplications.
// Otherwise it computes 3^(F + 1) mod N, about log2(F) modular squarings of N:  slow for large E.
//
//   cofactor-check E.cofactor E f1 [f2 ...]   F = f1 * f2 * ..., each fi given as 2^k-1 (e.g. 2^101-1) or in decimal
//   cofactor-check --ref E                    compute R with GMP (E squarings) and compare it with E.cofactor, to test prpll
//
// Build:  cc -O2 -o cofactor-check cofactor-check.c -lgmp

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void readResidue(mpz_t r, const char* path, unsigned long E) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(2); }
  size_t const nWords = (E + 31) / 32;
  unsigned* w = calloc(nWords + 1, 4);
  size_t const n = fread(w, 4, nWords + 1, f);
  fclose(f);
  if (n != nWords) { fprintf(stderr, "%s: %zu words, expected %zu for E=%lu\n", path, n, nWords, E); exit(2); }
  mpz_import(r, nWords, -1, 4, 0, 0, w);
  free(w);
}

// x mod 2^E - 1, for 0 <= x < (2^E - 1)^2
static void modMersenne(mpz_t x, unsigned long E, mpz_t tmp, const mpz_t M) {
  mpz_tdiv_q_2exp(tmp, x, E);
  mpz_tdiv_r_2exp(x, x, E);
  mpz_add(x, x, tmp);
  if (mpz_cmp(x, M) >= 0) { mpz_sub(x, x, M); }
}

// The k of a factor given as 2^k-1, else 0
static unsigned long mersenneK(const char* s) {
  unsigned long k;
  char rest;
  return sscanf(s, "2^%lu-1%c", &k, &rest) == 1 ? k : 0;
}

static int fileExists(const char* path) {
  FILE* f = fopen(path, "rb");
  if (f) { fclose(f); }
  return f != NULL;
}

static void parseFactor(mpz_t f, const char* s) {
  unsigned long k;
  char rest;
  if (sscanf(s, "2^%lu-1%c", &k, &rest) == 1) {
    mpz_set_ui(f, 0);
    mpz_setbit(f, k);
    mpz_sub_ui(f, f, 1);
  } else if (mpz_set_str(f, s, 10) != 0) {
    fprintf(stderr, "factor '%s': expected 2^k-1 or a decimal number\n", s);
    exit(2);
  }
}

int main(int argc, char** argv) {
  mpz_t M, R, t;
  mpz_inits(M, R, t, NULL);

  if (argc == 3 && !strcmp(argv[1], "--ref")) {
    unsigned long const E = strtoul(argv[2], NULL, 10);
    mpz_setbit(M, E);
    mpz_sub_ui(M, M, 1);
    mpz_t x;
    mpz_init_set_ui(x, 3);
    for (unsigned long i = 0; i < E; ++i) {
      mpz_mul(x, x, x);
      modMersenne(x, E, t, M);
    }
    char path[64];
    snprintf(path, sizeof(path), "%lu.cofactor", E);
    readResidue(R, path, E);
    int const same = mpz_cmp(x, R) == 0;
    printf("%s: %s GMP's 3^(2^%lu) mod 2^%lu-1 (low 64 bits %016lx)\n", path, same ? "equals" : "DIFFERS FROM", E, E,
           (unsigned long) mpz_getlimbn(x, 0));
    return same ? 0 : 1;
  }

  if (argc < 4) {
    fprintf(stderr, "usage:  %s <E>.cofactor <E> <f1> [<f2> ...]   (fi as 2^k-1 or decimal)\n"
                    "        %s --ref <E>\n", argv[0], argv[0]);
    return 2;
  }
  unsigned long const E = strtoul(argv[2], NULL, 10);
  mpz_setbit(M, E);
  mpz_sub_ui(M, M, 1);
  readResidue(R, argv[1], E);

  mpz_t F, f, N, rem, want, Fp1, three;
  mpz_inits(F, f, N, rem, want, Fp1, NULL);
  mpz_init_set_ui(three, 3);
  mpz_set_ui(F, 1);
  for (int i = 3; i < argc; ++i) {
    parseFactor(f, argv[i]);
    mpz_mul(F, F, f);
  }
  mpz_tdiv_qr(N, rem, M, F);
  if (mpz_sgn(rem) != 0) { fprintf(stderr, "F does not divide 2^%lu-1\n", E); return 2; }

  mpz_mod(R, R, N);
  unsigned long const p = argc == 5 ? mersenneK(argv[3]) : 0, q = argc == 5 ? mersenneK(argv[4]) : 0;
  char pathP[1100], pathQ[1100], pathPQ[1100];
  snprintf(pathP, sizeof(pathP), "%s.%lu", argv[1], p);
  snprintf(pathQ, sizeof(pathQ), "%s.%lu", argv[1], q);
  snprintf(pathPQ, sizeof(pathPQ), "%s.%lu", argv[1], p + q);
  int prp;
  if (p && q && p * q == E && fileExists(pathP) && fileExists(pathQ) && fileExists(pathPQ)) {
    // R 3^(2^p) 3^(2^q) = 9 3^(2^(p+q)) mod N
    mpz_t a;
    mpz_init(a);
    readResidue(a, pathP, E);
    mpz_mul(R, R, a);
    mpz_mod(R, R, N);
    readResidue(a, pathQ, E);
    mpz_mul(R, R, a);
    mpz_mod(R, R, N);
    readResidue(want, pathPQ, E);
    mpz_mul_ui(want, want, 9);
    mpz_mod(want, want, N);
    prp = mpz_cmp(R, want) == 0;
    mpz_clear(a);
  } else {
    fprintf(stderr, "(no %s.<k> residues for F = (2^p-1)(2^q-1) with p q = E:  computing 3^(F+1) mod N, which is slow for large E)\n",
            argv[1]);
    mpz_add_ui(Fp1, F, 1);
    mpz_powm(want, three, Fp1, N);
    prp = mpz_cmp(R, want) == 0;
  }
  printf("N = (2^%lu-1)/F, %zu digits:  %s\n", E, mpz_sizeinbase(N, 10), prp ? "base-3 PRP" : "composite");
  return prp ? 0 : 1;
}
