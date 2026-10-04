# Usage: python3 tools/lds_conflicts.py   (see middleShuffleIdx in src/cl/fft-middle.cl)
# LDS bank-conflict simulation of PRPLL's middleShuffle variants.
# Model: 32 banks x 4 bytes.  A W-byte access per lane is split into phases of 128/W consecutive lanes (128 bytes per phase).
# Conflict degree of a phase = max over banks of the number of DISTINCT 4-byte words hitting that bank (same word = broadcast).
def degree(addrs_bytes, W):
    lanes = len(addrs_bytes); per = 128 // W; worst = 1
    for p0 in range(0, lanes, per):
        banks = {}
        for a in addrs_bytes[p0:p0 + per]:
            for w in range(a // 4, (a + W) // 4):
                banks.setdefault(w % 32, set()).add(w)
        worst = max(worst, max(len(s) for s in banks.values()))
    return worst

def diag(me, wg, bs):
    ROWS = wg // bs; row, col = me // bs, me % bs; row_w, col_w = me % ROWS, me // ROWS
    if bs == 2 * ROWS:   return row*bs + (col + 2*row) % bs, row_w*bs + (col_w + 2*row_w) % bs
    if ROWS == 2 * bs:   return col*ROWS + (row + 2*col) % ROWS, col_w*ROWS + (row_w + 2*col_w) % ROWS
    if bs == ROWS:       return row*bs + (col + row) % bs, row_w*bs + (col_w + row_w) % bs
    return col*ROWS + row, me
def plain(me, wg, bs): return (me % bs) * (wg // bs) + me // bs, me

def check(name, fn, wg, bs, W):
    p1 = [fn(m, wg, bs)[0] * W for m in range(wg)]; p2 = [fn(m, wg, bs)[1] * W for m in range(wg)]
    # sanity: p1/p2 permutations and round trip (value written by thread t at p1[t] read by thread r with p2[r]==p1[t])
    assert sorted(p1) == sorted(p2) == [i * W for i in range(wg)]
    return degree(p1, W), degree(p2, W)

print("Not-in-place middleShuffle (MIDDLE_IN/OUT_LDS_TRANSPOSE), per access: (write, read) worst conflict degree")
print("  variant                         elem  " + "  ".join(f"WG{wg}/X{bs}" for wg, bs in [(128,16),(64,8),(128,8),(256,16),(64,16),(128,32),(256,32),(64,4)]))
cfgs = [(128,16),(64,8),(128,8),(256,16),(64,16),(128,32),(256,32),(64,4)]
for name, fn, W in [("FP64 / GF61, MIDDLE<=8: plain", plain, 8), ("FP64 / GF61, MIDDLE>8: diagonal", diag, 4),
                    ("FP32 / GF31, any MIDDLE: diagonal", diag, 4)]:
    print(f"  {name:34s} {W}B   " + "  ".join(f"{str(check(name, fn, wg, bs, W)):>9s}" for wg, bs in cfgs))

print("In-place 16x16 middleShuffle (256 threads), per access: worst conflict degree")
for W, what in [(16, "FP64 / GF61 (16-byte)"), (8, "FP32 / GF31 (8-byte)")]:
    A = [((m % 16) * 16 + ((m // 16) ^ (m % 16))) * W for m in range(256)]   # x*16 + (y^x)
    B = [((m // 16) * 16 + ((m % 16) ^ (m // 16))) * W for m in range(256)]  # y*16 + (x^y)
    print(f"  {what:24s}  write A {degree(A, W)}  read B {degree(B, W)}  write B {degree(B, W)}  read A {degree(A, W)}")

print("Diagonal addressing with whole 8-byte elements (Z61 / double), per access: (write, read)")
print("  " + "  ".join(f"WG{wg}/X{bs}:{check('d8', diag, wg, bs, 8)}" for wg, bs in cfgs))
