#!/usr/bin/env python3
"""Compile PRPLL's kernels with NVRTC the way the CUDA build does, on a machine with no GPU.

The CUDA build compiles its kernels (src/cl/*.cl through opencl_compat.cuh) only at run time, on the user's GPU, so a kernel
that OpenCL accepts and NVRTC rejects is invisible to every CI build.  This runs the real CUDA build against a fake CUDA driver
(fakecuda.cpp: one nVidia GPU of the chosen compute capability, no hardware behind it): prpll picks its kernels and their -D
options as it would on that GPU and the real NVRTC compiles them.  Kernels are not run.

usage: nvrtc-check.py <CUDA build of prpll> [--cc 89,min] [-j N] [--list] [--only REGEX]
  --cc    compute capabilities to compile for: numbers (89 = sm_89), "min" and "max" for the oldest and newest this NVRTC knows
  --list  print the runs that would be made and exit
Exit status 1 if any kernel fails to compile (the NVRTC diagnostics are printed, grouped), 2 if there is nothing to check.

The runs are a small set of FFTs, taken from the list prpll prints (-h), that together cover every FFT type, width, middle and
height, every variant digit, INPLACE=0/1/2, both carry widths and the TAIL_KERNELS settings, for each number type (FP64, FP32,
GF31, GF61) a type contains.  Those are PRP runs; for every FFT type there is also a small LL test and a proof verification, which
load the kernels the PRP loop does not (carry and carryFused for LL, tailMul with MUL_LOW, the *Zero and Hin kernels).

prpll must be able to load a libcuda.so.1 (the CUDA toolkit's driver stub is enough): the real NVRTC loads one itself and has to
find the driver's, not the fake, so the fake is LD_PRELOADed over prpll's own calls instead of replacing the library.
"""
import argparse, concurrent.futures as cf, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.join(HERE, "..", "..", "test", "786433-10.proof")   # the proof the POCL CI job verifies; exponent 786433
# The number types of each FFT type (FFTConfig.h FFT_TYPES): D = FP64, S = FP32, t = GF31, g = GF61
PARTS = {0: "D", 1: "tg", 2: "Sg", 3: "g", 4: "Stg", 50: "St", 51: "Dt", 52: "t", 53: "S"}
# Types that -h does not list: reachable only through an explicit -fft prefix.  Take a few shapes each
EXTRA_TYPES = (50, 52, 53)
EXTRA_SHAPES = ("256:2:256", "512:8:512", "256:16:1K")
# Largest FFT (words) to compile for: the host builds its tables on the CPU, and the kernel code does not depend on the size
MAX_N = 1 << 24
INPLACE = (0, 1, 2)
EFRACS = (0.97, 0.6)            # of the largest exponent for the FFT: the wide and the narrow carry
TAIL_KERNELS = (None, 0, 1, 3)  # None: the default

def size(s):
    return int(s.rstrip("K")) * (1024 if s.endswith("K") else 1)

def run(prpll, args, env=None, **kw):
    p = subprocess.run([prpll] + args, capture_output=True, text=True, env=env, **kw)
    return p.stdout + p.stderr, p.returncode

def plan(prpll, env):
    """[(mode, spec, INPLACE, TAIL_KERNELS, exponent)]: the LL and verify runs, and a greedy cover of the PRP features"""
    out, _ = run(prpll, ["-h"], env)
    specs = sorted({s for line in out.splitlines() if re.match(r"^\s*[\d.]+[MK]?\s+[\d.]+M\s+[\d.]+\s", line) for s in line.split()[3].split(",")})
    specs += [f"{t}:{shape}" for t in EXTRA_TYPES for shape in EXTRA_SHAPES]
    # -info gives each spec's variants and the exponent each takes (in M, one decimal)
    info, _ = run(prpll, ["-info", ",".join(specs)], env)
    shapes = []     # (type, W, M, H, variant, words, max exponent)
    for m in re.finditer(r"^\S+ \S+\s+(\S+) \| ([\d.]+) \|\s+([\d.]+)$", info, re.M):
        f = m.group(1).split(":")
        w, mid, h, var = f[-4:]
        shapes.append((int(f[0]) if len(f) == 5 else 0, w, mid, h, var, 2 * size(w) * int(mid) * size(h), float(m.group(3)) * 1e6))
    chosen = []
    for typ in PARTS:
        # A small LL test and a proof verification of every type.  Verification takes the FFT as given only if it can handle the
        # proof's exponent, so only a tiny FFT will do.  Verifications at every TAIL_KERNELS: the PRP runs below build tailMul and
        # tailSquare for each, but only a verification builds tailMul with MUL_LOW
        small = [s for s in shapes if s[:4] == (typ, "256", "2", "256")]
        if small:
            chosen.append(("ll", f"{typ}:256:2:256:{small[-1][4]}", 1, None, int(small[-1][6] * 0.9)))
            chosen += [("verify", f"{typ}:256:2:256", 1, tk, 786433) for tk in (0, 1, 2, 3)]
    items = []
    for typ, w, mid, h, var, n, maxexp in shapes:
        if n > MAX_N: continue
        spec = f"{typ}:{w}:{mid}:{h}:{var}" if typ else f"{w}:{mid}:{h}:{var}"
        for ip in INPLACE:
            for ef in EFRACS:
                for tk in TAIL_KERNELS:
                    feats = {(typ, "carry", ef)}
                    for p in PARTS[typ]:
                        feats |= {(p, "W", w, var[0]), (p, "M", mid), (p, "H", h, var[2]), (p, "m", var[1]), (p, "IP", ip, int(mid) % 8 or 8), (p, "TK", tk)}
                    items.append((n, ("prp", spec, ip, tk, int(maxexp * ef)), feats))
    want = set().union(*(i[2] for i in items)) if items else set()
    while want:
        # most new features per word of FFT (the host's CPU work); ties go to the smaller FFT
        best = max(items, key=lambda i: (len(i[2] & want) / (1 + i[0] / MAX_N), -i[0]))
        chosen.append(best[1])
        want -= best[2]
    return chosen

def command(mode, spec, ip, tk, exp):
    use = f"INPLACE={ip}" + (f",TAIL_KERNELS={tk}" if tk is not None else "")
    what = {"prp": ["-prp", str(exp), "-proof", "0"], "ll": ["-ll", str(exp)], "verify": ["-verify", PROOF]}[mode]
    # The kernels are all built by the first few iterations.  The fake GPU computes nothing, so what the run reports means nothing
    return ["-v", "1", "-device", "0", "-fft", spec, "-iters", "1000", "-log", "1000", "-use", use] + what

def check(prpll, env, tmp, item):
    d = tempfile.mkdtemp(dir=tmp)
    try:
        log, rc = run(prpll, ["-dir", d] + command(*item), dict(env, FAKECUDA_RES="4" if item[0] == "ll" else "3"), timeout=900)
    except subprocess.TimeoutExpired:
        log, rc = "timeout", 1
    shutil.rmtree(d, ignore_errors=True)
    # "Loaded" (-v 1) says kernels were built; rc < 0 is a crash.  A run that ends in an error of its own is fine: the fake GPU computes nothing
    return rc >= 0 and "NVRTC COMPILE FAILED" not in log and "Can't compile" not in log and "Loaded" in log, log

def errors(log):
    """The NVRTC diagnostics of a failed run: [(error line, the source line and caret under it)], else its last lines"""
    lines = log.splitlines()
    errs = [(l, "\n".join(lines[i + 1:i + 5])) for i, l in enumerate(lines) if re.search(r"\(\d+\): error", l)]
    return errs or [("\n".join(lines[-6:]), "")]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prpll"); ap.add_argument("--cc", default="89,min"); ap.add_argument("-j", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--list", action="store_true"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    prpll = os.path.abspath(a.prpll)
    with tempfile.TemporaryDirectory(prefix="nvrtc-check-") as tmp:
        fake = os.path.join(tmp, "fakecuda.so")
        subprocess.run([os.environ.get("CXX", "c++"), "-O1", "-shared", "-fPIC", "-o", fake, os.path.join(HERE, "fakecuda.cpp"), "-ldl"], check=True)
        env = dict(os.environ, LD_PRELOAD=fake, FAKECUDA_CC="89")
        items = plan(prpll, env)
        # A listing format that changed must not turn into a check of nothing
        fft_types = {int(i[1].split(":")[0]) if i[1].count(":") == 4 else 0 for i in items}
        if fft_types != set(PARTS) or {i[0] for i in items} != {"prp", "ll", "verify"}:
            print(f"::error::the plan has FFT types {sorted(fft_types)}, expected {sorted(PARTS)}, and runs of {sorted({i[0] for i in items})}: did the output of 'prpll -h' or '-info' change?")
            return 2
        items = [i for i in items if re.search(a.only, i[1])]
        if a.list:
            for i in items: print(*i)
            return 0
        print(f"{len(items)} runs: {sum(i[0] == 'prp' for i in items)} PRP, {sum(i[0] == 'll' for i in items)} LL, {sum(i[0] == 'verify' for i in items)} proof verifications", flush=True)
        bad = 0
        for cc in a.cc.split(","):
            env["FAKECUDA_CC"] = cc
            groups = {}     # NVRTC error -> (context, [runs that hit it])
            arch = "sm_" + cc
            nbad = 0
            with cf.ThreadPoolExecutor(a.j) as ex:
                for item, (ok, log) in zip(items, ex.map(lambda i: check(prpll, env, tmp, i), items)):
                    if m := re.search(r"Fake NVIDIA GPU \((sm_\d+)\)", log): arch = m.group(1)    # "min" and "max" are numbers only to the fake
                    if ok: continue
                    nbad += 1
                    for err, ctx in errors(log):
                        groups.setdefault(err, (ctx, []))[1].append(" ".join(command(*item)[4:]))
            for err, (ctx, runs) in groups.items():
                print(f"::error::NVRTC for {arch} rejects a kernel of {len(runs)} runs")
                print(f"{err}\n{ctx}\nprpll options of the runs:\n  " + "\n  ".join(sorted(set(runs))[:20]) + ("\n  ..." if len(set(runs)) > 20 else ""), flush=True)
            print(f"{arch}: {len(items)} runs, {nbad} failed", flush=True)
            bad += nbad
        return 1 if bad else 0

sys.exit(main())
