// Copyright 2017 Mihai Preda.

#include "Gpu.h"

#include "checkpoint.h"
#include "Stats.h"
#include "state.h"
#include "timeutil.h"
#include "args.h"
#include "Primes.h"
#include "Result.h"
#include "Signal.h"
#include "FFTConfig.h"

#include <cmath>
#include <cassert>
#include <cstring>
#include <algorithm>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#define TAU (2 * M_PIl)

using double2 = pair<double, double>;

static_assert(sizeof(double2) == 16, "size double2");

// Returns the primitive root of unity of order N, to the power k.
static double2 root1(u32 N, u32 k) {
  long double angle = - TAU / N * k;
  return double2{double(cosl(angle)), double(sinl(angle))};
}

static double2 *smallTrigBlock(int W, int H, double2 *p) {
  for (int line = 1; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      *p++ = root1(W * H, line * col);
    }
  }
  return p;
}

static cl_mem genSmallTrig(cl_context context, int size, int radix) {
  auto *tab = new double2[size]();
  auto *p = tab + radix;
  int w = 0;
  for (w = radix; w < size; w *= radix) { p = smallTrigBlock(w, std::min(radix, size / w), p); }
  assert(p - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double2) * size, tab);
  delete[] tab;
  return buf;
}

static void setupWeights(cl_context context, Buffer &bufA, Buffer &bufI, int W, int H, int E) {
  int N = 2 * W * H;
  auto weights = genWeights(E, W, H);  
  bufA.reset(makeBuf(context, BUF_CONST, sizeof(double) * N, weights.first.data()));
  bufI.reset(makeBuf(context, BUF_CONST, sizeof(double) * N, weights.second.data()));
}

Gpu::Gpu(u32 E, u32 W, u32 BIG_H, u32 SMALL_H, int nW, int nH,
         cl_program program, const vector<cl_device_id> &devices, cl_context context,
         bool timeKernels, bool useLongCarry) :
  E(E),
  N(W * BIG_H * 2),
  hN(N / 2),
  nW(nW),
  nH(nH),
  bufSize(N * sizeof(double)),
  useLongCarry(useLongCarry),
  useMiddle(BIG_H != SMALL_H),
  queue(makeQueue(devices.front(), context)),  

#define LOAD(name, workGroups) name(program, queue.get(), devices.front(), workGroups, #name, timeKernels)
  LOAD(carryFused, BIG_H + 1),
  LOAD(fftP, BIG_H),
  LOAD(fftW, BIG_H),
  LOAD(fftH, (hN / SMALL_H)),
  LOAD(fftMiddleIn,  hN / (256 * (BIG_H / SMALL_H))),
  LOAD(fftMiddleOut, hN / (256 * (BIG_H / SMALL_H))),
  LOAD(carryA,   nW * (BIG_H/16)),
  LOAD(carryM,   nW * (BIG_H/16)),
  LOAD(carryB,   nW * (BIG_H/16)),
  LOAD(transposeW,   (W/64) * (BIG_H/64)),
  LOAD(transposeH,   (W/64) * (BIG_H/64)),
  LOAD(transposeIn,  (W/64) * (BIG_H/64)),
  LOAD(transposeOut, (W/64) * (BIG_H/64)),
  LOAD(multiply, hN / SMALL_H),
  LOAD(multiplySub, hN / SMALL_H),
  LOAD(tailFused, (hN / SMALL_H) / 2),
  LOAD(readResidue, 1),
  LOAD(isNotZero, 256),
  LOAD(isEqual, 256),
#undef LOAD

  bufData( makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
  bufCheck(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
  bufAux(  makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
  
  bufTrigW(genSmallTrig(context, W, nW)),
  bufTrigH(genSmallTrig(context, SMALL_H, nH)),
  buf1{makeBuf(    context, BUF_RW, bufSize)},
  buf2{makeBuf(    context, BUF_RW, bufSize)},
  buf3{makeBuf(    context, BUF_RW, bufSize)},
  bufCarry{makeBuf(context, BUF_RW, bufSize / 2)},
  bufReady{makeBuf(context, BUF_RW, BIG_H * sizeof(int))},
  bufSmallOut(makeBuf(context, CL_MEM_READ_WRITE, 256 * sizeof(int)))
{    
  setupWeights(context, bufA, bufI, W, BIG_H, E);

  carryFused.setFixedArgs(3, bufA, bufI, bufTrigW);
  fftP.setFixedArgs(2, bufA, bufTrigW);
  fftW.setFixedArgs(1, bufTrigW);
  fftH.setFixedArgs(1, bufTrigH);
    
  carryA.setFixedArgs(3, bufI);
  carryM.setFixedArgs(3, bufI);
  tailFused.setFixedArgs(1, bufTrigH);
    
  queue.zero(bufReady, BIG_H * sizeof(int));
}

void logTimeKernels(std::initializer_list<Kernel *> kerns) {
  double total = 0;
  vector<pair<Stats, string>> infos;
  for (Kernel *k : kerns) {
    Stats s = k->resetStats();
    if (s.time > 0 && s.nSq > 0) {
      infos.push_back(make_pair(s, k->getName()));
      total += s.time;
    }
  }

  std::sort(infos.begin(), infos.end(), [](const auto& a, const auto& b){ return a.first.time > b.first.time; });

  for (auto& [stats, name]: infos) {
    float percent = 100 / total * stats.time;
    if (true || percent >= .1f) {
      log("%4.1f%% %-14s : %6.0f us/call x %5d calls\n",
          percent, name.c_str(), stats.time / stats.nSq, stats.nSq);
    }
  }
  log("\n");
}

static FFTConfig getFFTConfig(const vector<FFTConfig> &configs, u32 E, int argsFftSize) {
  int i = 0;
  int n = int(configs.size());
  if (argsFftSize < 10) { // fft delta or not specified.
    while (i < n - 1 && configs[i].maxExp < E) { ++i; }      
    i = max(0, min(i + argsFftSize, n - 1));
  } else { // user-specified fft size.
    while (i < n - 1 && u32(argsFftSize) > configs[i].fftSize) { ++i; }      
  }
  return configs[i];
}

vector<int> Gpu::readSmall(Buffer &buf, u32 start) {
  readResidue(buf, bufSmallOut, start);
  return queue.read<int>(bufSmallOut, 128);                    
}

unique_ptr<Gpu> Gpu::make(u32 E, const Args &args) {
  vector<FFTConfig> configs = FFTConfig::genConfigs();
        
  FFTConfig config = getFFTConfig(configs, E, args.fftSize);
  int WIDTH        = config.width;
  int SMALL_HEIGHT = config.height;
  int MIDDLE       = config.middle;
  int N = WIDTH * SMALL_HEIGHT * MIDDLE * 2;

  int nW = (WIDTH == 1024 || WIDTH == 256) ? 4 : 8;
  int nH = (SMALL_HEIGHT == 1024 || SMALL_HEIGHT == 256) ? 4 : 8;

  float bitsPerWord = E / float(N);
  string strMiddle = (MIDDLE == 1) ? "" : (string(", Middle ") + std::to_string(MIDDLE));
  log("%u FFT %dK: Width %dx%d, Height %dx%d%s; %.2f bits/word\n",
      E, N / 1024, WIDTH / nW, nW, SMALL_HEIGHT / nH, nH, strMiddle.c_str(), bitsPerWord);

  if (bitsPerWord > 20) {
    log("FFT size too small for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too small";
  }

  if (bitsPerWord < 1.5) {
    log("FFT size too large for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too large";
  }
    
  bool useLongCarry = (bitsPerWord < 14.5f)
    || (args.carry == Args::CARRY_LONG)
    || (args.carry == Args::CARRY_AUTO && WIDTH >= 2048);
  
  log("using %s carry kernels\n", useLongCarry ? "long" : "short");

  string clArgs = args.clArgs;
  if (!args.dump.empty()) { clArgs += " -save-temps=" + args.dump + "/" + numberK(N); }

  bool timeKernels = args.timeKernels;
    
  if (args.devices.empty()) { throw "No OpenCL device"; }

  Context context(createContext(args.devices));
  auto devices = toDeviceIds(args.devices);
  Holder<cl_program> program(compile(devices, context.get(), "gpuowl", clArgs,
                                     {{"EXP", E}, {"WIDTH", WIDTH}, {"SMALL_HEIGHT", SMALL_HEIGHT}, {"MIDDLE", MIDDLE}},
                                     args.usePrecompiled));
  if (!program) { throw "OpenCL compilation"; }

  return make_unique<Gpu>(E, WIDTH, SMALL_HEIGHT * MIDDLE, SMALL_HEIGHT, nW, nH,
                          program.get(), devices, context.get(), timeKernels, useLongCarry);
}

vector<u32> Gpu::readData()  { return compactBits(readOut(bufData),  E); }
vector<u32> Gpu::readCheck() { return compactBits(readOut(bufCheck), E); }

vector<u32> Gpu::writeData(const vector<u32> &v) {
  writeIn(v, bufData);
  return v;
}

vector<u32> Gpu::writeCheck(const vector<u32> &v) {
  writeIn(v, bufCheck);
  return v;
}

// The modular multiplication io *= in.
void Gpu::modMul(Buffer &in, Buffer &io, bool mul3) {
  fftP(in, buf1);
  tW(buf1, buf3);
    
  fftP(io, buf1);
  tW(buf1, buf2);
    
  fftH(buf2);
  fftH(buf3);
  multiply(buf2, buf3);
  fftH(buf2);

  tH(buf2, buf1);    

  fftW(buf1);
  mul3 ? carryM(buf1, io, bufCarry) : carryA(buf1, io, bufCarry);
  carryB(io, bufCarry);
};

void Gpu::writeState(const vector<u32> &check, u32 blockSize) {
  assert(blockSize > 0);
    
  writeCheck(check);
  queue.copy<int>(bufCheck, bufData, N);
  queue.copy<int>(bufCheck, bufAux, N);

  u32 n = 0;
  for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
    dataLoop(n);
    modMul(bufAux, bufData);
    queue.copy<int>(bufData, bufAux, N);
  }

  assert((n & (n - 1)) == 0);
  assert(blockSize % n == 0);
    
  blockSize /= n;
  assert(blockSize >= 2);
  
  for (u32 i = 0; i < blockSize - 2; ++i) {
    dataLoop(n);
    modMul(bufAux, bufData);
  }
  
  dataLoop(n);
  modMul(bufAux, bufData, true);
}

void Gpu::updateCheck() { modMul(bufData, bufCheck); }
  
bool Gpu::doCheck(int blockSize) {
  queue.copy<int>(bufCheck, bufAux, N);
  modSqLoop(bufAux, blockSize, true);
  updateCheck();
  return equalNotZero(bufCheck, bufAux);
}

void Gpu::logTimeKernels() {
  ::logTimeKernels({&carryFused, &fftP, &fftW, &fftH, &fftMiddleIn, &fftMiddleOut,
        &carryA, &carryM, &carryB,
        &transposeW, &transposeH, &transposeIn, &transposeOut,
        &multiply, &multiplySub, &tailFused, &readResidue, &isNotZero, &isEqual});
}

void Gpu::tW(Buffer &in, Buffer &out) {
  transposeW(in, out);
  if (useMiddle) { fftMiddleIn(out); }
}

void Gpu::tH(Buffer &in, Buffer &out) {
  if (useMiddle) { fftMiddleOut(in); }
  transposeH(in, out);
}
  
vector<int> Gpu::readOut(Buffer &buf) {
  transposeOut(buf, bufAux);
  return queue.read<int>(bufAux, N);
}

void Gpu::writeIn(const vector<u32> &words, Buffer &buf) { writeIn(expandBits(words, N, E), buf); }

void Gpu::writeIn(const vector<int> &words, Buffer &buf) {
  queue.write(bufAux, words);
  transposeIn(bufAux, buf);
}

void Gpu::modSqLoop(Buffer &io, u32 reps, bool mul3) {
  assert(reps > 0);
  bool dataIsOut = true;
        
  for (decltype(reps) i = 0; i < reps; ++i) {
    if (dataIsOut) { fftP(io, buf1); }
    tW(buf1, buf2);
    tailFused(buf2);
    tH(buf2, buf1);

    dataIsOut = useLongCarry || (i == reps - 1);
    if (dataIsOut) {
      fftW(buf1);
      mul3 && i == reps-1 ? carryM(buf1, io, bufCarry) : carryA(buf1, io, bufCarry);
      carryB(io, bufCarry);
    } else {
      carryFused(buf1, bufCarry, bufReady);
    }
  }
}

bool Gpu::equalNotZero(Buffer &buf1, Buffer &buf2) {
  queue.zero(bufSmallOut, sizeof(int));
  u32 sizeBytes = N * sizeof(int);
  isNotZero(sizeBytes, buf1, bufSmallOut);
  isEqual(sizeBytes, buf1, buf2, bufSmallOut);
  return queue.read<int>(bufSmallOut, 1)[0];
}
  
u64 Gpu::bufResidue(Buffer &buf) {
  u32 earlyStart = N/2 - 32;
  vector<int> readBuf = readSmall(buf, earlyStart);
  return residueFromRaw(E, N, readBuf);
}

static string makeLogStr(u32 E, string status, int k, u64 res, Stats info, u32 nIters) {
  float msPerSq = info.time / info.nSq;
  int etaMins = (nIters - k) * msPerSq * (1 / 60000.f) + .5f;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;

  char buf[256];
  string ghzStr;
  
  snprintf(buf, sizeof(buf), "%u %2s %8d %5.2f%%; %.2f ms/sq;%s ETA %dd %02d:%02d; %016llx",
           E, status.c_str(), k, k / float(nIters) * 100,
           msPerSq,
           ghzStr.c_str(), days, hours, mins, res);
  return buf;
}

static void doLog(int E, int k, u32 timeCheck, u64 res, bool checkOK, Stats &stats, u32 nIters) {
  log("%s (check %.2fs)\n",      
      makeLogStr(E, checkOK ? "OK" : "EE", k, res, stats, nIters).c_str(),
      timeCheck * .001f);
  stats.reset();
}

static void doSmallLog(int E, int k, u64 res, Stats &stats, u32 nIters) {
  log("%s\n", makeLogStr(E, "", k, res, stats, nIters).c_str());
  stats.reset();
}

static bool equalMinus3(const vector<u32> &a) {
  if (a[0] != ~3u) { return false; }
  for (auto it = next(a.begin()); it != a.end(); ++it) { if (*it) { return false; }}
  return true;
}

PRPState Gpu::loadPRP(u32 E, u32 iniBlockSize) {
  auto loaded = PRPState::load(E, iniBlockSize);

  writeState(loaded.check, loaded.blockSize);

  u64 res64 = dataResidue();
  bool ok = (res64 == loaded.res64);
  updateCheck();
  if (!ok) {
    log("%u EE loaded: %d, blockSize %d, %016llx (expected %016llx)\n",
        E, loaded.k, loaded.blockSize, res64, loaded.res64);
    throw "error on load";
  }

  return loaded;
}

PRPResult Gpu::isPrimePRP(u32 E, const Args &args) {
  PRPState loaded = loadPRP(E, args.blockSize);

  u32 k = loaded.k;
  u32 blockSize = loaded.blockSize;
  assert(blockSize > 0 && 10000 % blockSize == 0);
  
  const u32 kEnd = E - 1; // Type-4 per http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  assert(k < kEnd);

  const u32 checkStep = blockSize * blockSize;
  
  u32 startK = k;
  
  Signal signal;
  Stats stats;

  // Number of sequential errors (with no success in between). If this ever gets high enough, stop.
  int nSeqErrors = 0;

  bool isPrime = false;
  Timer timer;

  u64 finalRes64 = 0;
  u32 nTotalIters = ((kEnd - 1) / blockSize + 1) * blockSize;
  while (true) {
    assert(k % blockSize == 0);
    if (k < kEnd && k + blockSize >= kEnd) {
      dataLoop(kEnd - k);
      auto words = this->roundtripData();
      finalRes64 = residue(words);
      isPrime = equalMinus3(words);

      log("%s %8d / %d, %016llx\n", isPrime ? "PP" : "CC", kEnd, E, finalRes64);
      
      int itersLeft = blockSize - (kEnd - k);
      if (itersLeft > 0) { dataLoop(itersLeft); }
    } else {
      dataLoop(blockSize);
    }
    k += blockSize;

    queue.finish();
        
    stats.add(timer.deltaMillis(), blockSize);
    bool doStop = signal.stopRequested();
    if (doStop) {
      log("Stopping, please wait..\n");
      signal.release();
    }

    bool doCheck = (k % checkStep == 0) || (k >= kEnd && k < kEnd + blockSize) || doStop || (k - startK == 2 * blockSize);
    
    if (!doCheck) {
      this->updateCheck();
      if (k % 10000 == 0) {
        doSmallLog(E, k, dataResidue(), stats, nTotalIters);
        if (args.timeKernels) { this->logTimeKernels(); }
      }
      continue;
    }

    vector<u32> check = this->roundtripCheck();
    bool ok = this->doCheck(blockSize);

    u64 res64 = dataResidue();

    // the check time (above) is accounted separately, not added to iteration time.
    doLog(E, k, timer.deltaMillis(), res64, ok, stats, nTotalIters);
    
    if (ok) {
      if (k < kEnd) { PRPState{k, blockSize, res64, check}.save(E); }
      if (isPrime || k >= kEnd) { return PRPResult{"", isPrime, finalRes64, 3}; }
      nSeqErrors = 0;      
    } else {
      if (++nSeqErrors > 2) {
        log("%d sequential errors, will stop.\n", nSeqErrors);
        throw "too many errors";
      }
      
      auto loaded = loadPRP(E, blockSize);
      k = loaded.k;
      assert(blockSize == loaded.blockSize);
    }
    if (args.timeKernels) { this->logTimeKernels(); }
    if (doStop) { throw "stop requested"; }
  }
}
