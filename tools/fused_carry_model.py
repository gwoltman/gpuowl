# Model of carryFused's carry scheme, to show why it needs long carry at low bits-per-word (see Gpu.cpp, useLongCarry).
# Squares a residue mod 2^E - 1 held in N balanced IBDWT words of about b = E/N bits.  "fused" does what carryFused does:
# each pair's carry-out is computed from the pair alone, then the carry-in from the previous pair is added to the pair's
# low word and that word's excess goes, unnormalized, into the high word (carryFinal).  "long" propagates carries fully.
# The values stay exact either way; what differs is the size of the words and so of the next convolution's outputs.
# Prints every iters/10 iterations the largest |convolution output| in units of sqrt(N)*2^(2b) and the largest |word| in
# units of 2^b, or stops when the outputs blow up.  Needs numpy.
# Usage: fused_carry_model.py N b iters [fused|long]     e.g. 4096 5.5 300 fused
import sys, numpy as np
N, b, iters = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "fused"
E = int(b * N)
M = (1 << E) - 1
starts = [-(-E * j // N) for j in range(N + 1)]          # word j holds bits [ceil(E*j/N), ceil(E*(j+1)/N))
nb = [starts[j + 1] - starts[j] for j in range(N)]
wt = np.array([2.0 ** (starts[j] - E * j / N) for j in range(N)])

def carry_step(x, j):                                     # balanced word and carry, as carryStep
  c = (x + (1 << (nb[j] - 1))) >> nb[j]
  return x - (c << nb[j]), c

def to_int(w): return sum(int(w[j]) << starts[j] for j in range(N)) % M

def from_int(v):
  w = []
  for j in range(N): w.append(v & ((1 << nb[j]) - 1)); v >>= nb[j]
  c = 0
  for j in range(N): w[j], c = carry_step(w[j] + c, j)
  w[0] += c
  return w

def square(w):                                            # the weighted cyclic convolution, rounded to integers
  a = np.array(w, dtype=np.float64) * wt
  f = np.fft.fft(a)
  z = np.real(np.fft.ifft(f * f)) / wt
  return [int(x) for x in np.rint(z)], float(np.max(np.abs(z)))

def carry_long(z):
  w, c = list(z), 0
  for _ in range(2):
    for j in range(N): w[j], c = carry_step(w[j] + c, j)
  w[0] += c
  return w

def carry_fused(z):
  w, cout = [0] * N, [0] * (N // 2)
  for k in range(N // 2):                                 # pair alone (the carry-out goes to the carry shuttle)
    j = 2 * k
    w[j], c = carry_step(z[j], j)
    w[j + 1], cout[k] = carry_step(z[j + 1] + c, j + 1)
  for k in range(N // 2):                                 # carryFinal: carry-in into the low word, excess into the high word
    j = 2 * k
    w[j], t = carry_step(w[j] + cout[k - 1], j)           # k = 0 takes the carry wrapped around from the top (2^E == 1)
    w[j + 1] += t
  return w

w = from_int(0x123456789ABCDEF * 12345 + 3)
v = to_int(w)
scale = np.sqrt(N) * 2.0 ** (2 * b)
blk, mx, maxw = max(1, iters // 10), 0, 0
for it in range(1, iters + 1):
  z, zmax = square(w)
  if zmax > 2 ** 44:                                      # beyond what the float convolution computes exactly
    print(f"it {it}: convolution output {zmax:.3g} = {zmax / scale:.3g} * sqrt(N)*2^(2b) -- blown up")
    sys.exit()
  w = carry_fused(z) if mode == "fused" else carry_long(z)
  v = v * v % M
  mx, maxw = max(mx, zmax / scale), max(maxw, max(abs(x) for x in w) / 2 ** b)
  if it % blk == 0:
    print(f"it {it:5d}: max conv / (sqrt(N)*2^(2b)) = {mx:8.3f}   max |word| / 2^b = {maxw:8.3f}   exact: {to_int(w) == v}")
    mx, maxw = 0, 0
