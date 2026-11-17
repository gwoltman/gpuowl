#!/usr/bin/env python3
# Reads the Mlucas P-1 files that prpll's -mlucasDir writes and that Mlucas writes itself, for test/pm1/mlucas.sh.
#   save <p<p> file> <p>          the savefile: checks its layout and its Res64/Res35m1/Res36m1 checksums
#   prod <p<p>.s1_prod file>      the stage 1 product file: checks its limb-sum checksum, prints B1 and the bit count
#   ours <s1_prod> <p> <B1>       the product file holds prpll's E = 2p * (the largest power <= B1 of every prime <= B1)
#   relate <handed p<p>> <handed s1_prod> <Mlucas's own p<p>.s1> <Mlucas's own s1_prod> <p>
#                                 Mlucas's own stage 1 exponent E' is a multiple of prpll's E, and Mlucas's own residue 3^E' is
#                                 prpll's residue x = 3^E raised to E' / E
# Layouts as Mlucas.c read_ppm1_savefiles / read_ppm1_residue and pm1.c read_pm1_s1_prod read them.
import sys


def le(b):
    return int.from_bytes(b, 'little')


def save(path, p):
    d = open(path, 'rb').read()
    nbytes = (p + 7) // 8
    assert d[0] == 3 and d[1] == 1, "test type P-1 (3), modulus type Mersenne (1)"
    nsq = le(d[2:10])
    x = le(d[10:10 + nbytes])
    o = 10 + nbytes
    res64, r35, r36 = le(d[o:o + 8]), le(d[o + 8:o + 13]), le(d[o + 13:o + 18])
    o += 18
    assert res64 == x % 2**64, "Res64"
    assert r35 == x % (2**35 - 1), "Res35m1"
    assert r36 == x % (2**36 - 1), "Res36m1"
    kblocks, shift = le(d[o:o + 3]), le(d[o + 3:o + 11])
    o += 11
    assert shift == 0, "residue shift"
    rest = d[o:]
    assert len(rest) in (0, 8), "the two 4-byte error counts (or none, a v19 file)"
    return nsq, x, kblocks


def prod(path):
    d = open(path, 'rb').read()
    assert d[0] == 3 and d[1] == 1, "test type P-1 (3), modulus type Mersenne (1)"
    b1, nbits = le(d[2:6]), le(d[6:10])
    nbytes = (nbits + 7) // 8
    body = d[10:10 + nbytes]
    padded = body + bytes(-len(body) % 8)
    s = sum(le(padded[i:i + 8]) for i in range(0, len(padded), 8)) % 2**64
    assert s == le(d[10 + nbytes:18 + nbytes]), "limb-sum checksum"
    assert len(d) == 18 + nbytes, "file length"
    rev = le(body)
    e = 1 << nbits                       # the leading bit, implicit in the file
    for j in range(nbits):
        if rev >> j & 1:
            e |= 1 << (nbits - 1 - j)
    return b1, nbits, e


def primes(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\0\0'
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n + 1) if sieve[i]]


def main():
    cmd = sys.argv[1]
    if cmd == 'save':
        nsq, x, kb = save(sys.argv[2], int(sys.argv[3]))
        print(f"savefile OK: nsquares {nsq}, res64 {x % 2**64:016x}, FFT {kb}K")
    elif cmd == 'prod':
        b1, nbits, _ = prod(sys.argv[2])
        print(f"product file OK: B1 {b1}, {nbits} bits after the leading one")
    elif cmd == 'ours':
        b1, nbits, e = prod(sys.argv[2])
        p, B1 = int(sys.argv[3]), int(sys.argv[4])
        want = 2 * p
        for q in primes(B1):
            qk = q
            while qk * q <= B1:
                qk *= q
            want *= qk
        ok = b1 == B1 and e == want
        print(f"product file {'EQUAL' if ok else 'DIFFERENT'}: 2p * prime powers <= {B1} ({want.bit_length()} bits)")
        sys.exit(0 if ok else 1)
    elif cmd == 'relate':
        p = int(sys.argv[6])
        m = 2**p - 1
        nsq, x, _ = save(sys.argv[2], p)
        _, nb, e = prod(sys.argv[3])
        assert nsq == nb, "the savefile's nsquares is the product file's bit count"
        _, theirs, _ = save(sys.argv[4], p)
        _, _, e2 = prod(sys.argv[5])
        if e2 % e:
            print("Mlucas's own exponent is not a multiple of prpll's")
            sys.exit(1)
        r = e2 // e
        ok = pow(x, r, m) == theirs
        print(f"residue relation {'OK' if ok else 'FAILS'}: Mlucas's own residue {theirs % 2**64:016x} = (prpll's "
              f"{x % 2**64:016x})^{r} (Mlucas's exponent / prpll's, {r.bit_length()} bits)")
        sys.exit(0 if ok else 1)
    else:
        sys.exit("unknown command " + cmd)


main()
