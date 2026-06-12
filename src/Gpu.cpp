// Copyright (C) Mihai Preda and George Woltman.

#include "Gpu.h"
#include "Proof.h"
#include "TimeInfo.h"
#include "Trig.h"
#include "state.h"
#include "Args.h"
#include "Signal.h"
#include "FFTConfig.h"
#include "Queue.h"
#include "Task.h"
#include "KernelCompiler.h"
#include "Saver.h"
#include "timeutil.h"
#include "TrigBufCache.h"
#include "fs.h"
#include "Sha3Hash.h"

#include <algorithm>
#include <limits>
#include <iomanip>
#include <array>
#include <cinttypes>
#include <cstring>

#define _USE_MATH_DEFINES
#include <cmath>
#include <ranges>
#include <utility>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

#ifndef M_LN2l
#define M_LN2l 0.69314718055994530941723212145818L
#endif

#ifndef M_LN2
#define M_LN2 0.69314718055994530941723212145818
#endif

enum {
CARRY_LEN = 8
};

namespace {

u32 kAt(u32 H, u32 line, u32 col) { return (line + col * H) * 2; }

double weight(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2l((long double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

double invWeight(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2l(-(long double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

// MSVC does not truly support long double.  Use expm1 rather than exp2 and subtracting one.
double weightM1(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return expm1l(M_LN2l * (long double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

double invWeightM1(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return expm1l(M_LN2l * - (long double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

double boundUnderOne(double x) { return std::min(x, nexttoward(1, 0)); }

float weight32(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2((double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

float invWeight32(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2(-(double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

float weightM132(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return expm1(M_LN2 * (double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

float invWeightM132(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return expm1(M_LN2 * - (double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

Weights genWeights(FFTConfig fft, u64 E, u32 W, u32 H, u32 nW, bool nvidiaGpu) {
  u32 const N = 2u * W * H;
  u32 const groupWidth = W / nW;

  vector<double> weightsConstIF;
  vector<double> weightsIF;

  if (fft.FFT_FP64) {
    // Inverse + Forward
    for (u32 thread = 0; thread < groupWidth; ++thread) {
      auto iw = invWeight(N, E, H, 0, thread, 0);
      auto w = weight(N, E, H, 0, thread, 0);
      weightsIF.push_back(2 * boundUnderOne(iw));
      weightsIF.push_back(2 * w);
    }

    // the group order matches CarryA/M (not fftP/CarryFused).
    for (u32 gy = 0; gy < H; ++gy) {
      weightsIF.push_back(invWeightM1(N, E, H, gy, 0, 0));
      weightsIF.push_back(weightM1(N, E, H, gy, 0, 0));
    }

    // nVidia GPUs have a fast constant cache that only works on buffer sizes less than 64KB.  Create two smaller buffers
    // that can be used to create the large group order buffer with a single multiply.
    if (nvidiaGpu) {
      for (u32 gy = 0; gy < 64; ++gy) {
        weightsConstIF.push_back(invWeightM1(N, E, H, gy, 0, 0));
        weightsConstIF.push_back(weightM1(N, E, H, gy, 0, 0));
      }
      for (u32 gy = 0; gy < H; gy += 64) {
        weightsConstIF.push_back(invWeightM1(N, E, H, gy, 0, 0));
        weightsConstIF.push_back(weightM1(N, E, H, gy, 0, 0));
      }
    }
  }

  else if (fft.FFT_FP32) {
    vector<float> weightsConstIF32;
    vector<float> weightsIF32;
    // Inverse + Forward
    for (u32 thread = 0; thread < groupWidth; ++thread) {
      auto iw = invWeight32(N, E, H, 0, thread, 0) ;
      auto w = weight32(N, E, H, 0, thread, 0) ;
      // Weights are scaled by 2^-24 and 2^48 so that multiplicaton by 1/epsilon does not generate infinty results (width and height variant 2).
      iw = iw * 281474976710656.0f;
      w = w * 0.000000059604644775390625f;
      weightsIF32.push_back(iw);
      weightsIF32.push_back(w);
    }

    // the group order matches CarryA/M (not fftP/CarryFused).
    for (u32 gy = 0; gy < H; ++gy) {
      weightsIF32.push_back(invWeightM132(N, E, H, gy, 0, 0));
      weightsIF32.push_back(weightM132(N, E, H, gy, 0, 0));
    }

    // nVidia GPUs have a fast constant cache that only works on buffer sizes less than 64KB.  Create two smaller buffers
    // that can be used to create the large group order buffer with a single multiply.
    if (nvidiaGpu) {
      for (u32 gy = 0; gy < 64; ++gy) {
        weightsConstIF32.push_back(invWeightM132(N, E, H, gy, 0, 0));
        weightsConstIF32.push_back(weightM132(N, E, H, gy, 0, 0));
      }
      for (u32 gy = 0; gy < H; gy += 64) {
        weightsConstIF32.push_back(invWeightM132(N, E, H, gy, 0, 0));
        weightsConstIF32.push_back(weightM132(N, E, H, gy, 0, 0));
      }
    }

    // Copy the float vectors to the double vectors
    weightsIF.resize(weightsIF32.size() / 2);
    memcpy((double *) weightsIF.data(), weightsIF32.data(), weightsIF32.size() * sizeof(float));
    weightsConstIF.resize(weightsConstIF32.size() / 2);
    memcpy((double *) weightsConstIF.data(), weightsConstIF32.data(), weightsConstIF32.size() * sizeof(float));
  }

  return Weights{.weightsConstIF=weightsConstIF, .weightsIF=weightsIF};
}

string toLiteral(i32 value) { return to_string(value); }
string toLiteral(u32 value) { return to_string(value) + 'u'; }
[[maybe_unused]] string toLiteral(long value) { return to_string(value) + "ll"; }
[[maybe_unused]] string toLiteral(unsigned long value) { return to_string(value) + "ull"; }
[[maybe_unused]] string toLiteral(long long value) { return to_string(value) + "ll"; }
[[maybe_unused]] string toLiteral(unsigned long long value) { return to_string(value) + "ull"; }

template<typename F>
string toLiteral(F value) {
  std::ostringstream ss;
  ss << std::setprecision(numeric_limits<F>::max_digits10) << value;
  if (sizeof(F) == 4) ss << "f";
  string s = std::move(ss).str();

  // verify exact roundtrip
  [[maybe_unused]] F back = 0;
  sscanf(s.c_str(), (sizeof(F) == 4) ? "%f" : "%lf", &back);
  assert(back == value);
  
  return s;
}

template<typename T>
string toLiteral(const vector<T>& v) {
  assert(!v.empty());
  string s = "{";
  for (auto x : v) {
    s += toLiteral(x) + ",";
  }
  s += "}";
  return s;
}

template<typename T, size_t N>
string toLiteral(const std::array<T, N>& v) {
  string s = "{";
  for (T x : v) {
    s += toLiteral(x) + ",";
  }
  s += "}";
  return s;
}

string toLiteral(const string& s) { return s; }

[[maybe_unused]] string toLiteral(float2 cs) { return "U2("s + toLiteral(cs.first) + ',' + toLiteral(cs.second) + ')'; }
[[maybe_unused]] string toLiteral(double2 cs) { return "U2("s + toLiteral(cs.first) + ',' + toLiteral(cs.second) + ')'; }
[[maybe_unused]] string toLiteral(int2 cs) { return "U2("s + toLiteral(cs.first) + ',' + toLiteral(cs.second) + ')'; }
[[maybe_unused]] string toLiteral(uint2 cs) { return "U2("s + toLiteral(cs.first) + ',' + toLiteral(cs.second) + ')'; }
[[maybe_unused]] string toLiteral(ulong2 cs) { return "U2("s + toLiteral(cs.first) + ',' + toLiteral(cs.second) + ')'; }

template<typename T>
string toDefine(const string& k, const T& v) { return " -D"s + k + '=' + toLiteral(v); }

template<typename T>
string toDefine(const T& vect) {
  string s;
  for (const auto& [k, v] : vect) { s += toDefine(k, v); }
  return s;
}

constexpr bool isInList(const string& s, initializer_list<string> list) {
  for (const string& e : list) { if (e == s) { return true; }}
  return false;
}

string clDefines(const Args& args, cl_device_id id, FFTConfig fft, const vector<KeyVal>& extraConf, u64 E, bool doLog,
                 bool &tail_single_wide, bool &tail_single_kernel, u32 &in_place, u32 &pad_size, u32 &wmul) {
  map<string, string> config;

  // Highest priority is the requested "extra" conf
  config.insert(extraConf.begin(), extraConf.end());

  // Next, args config
  config.insert(args.flags.begin(), args.flags.end());

  // Lowest priority: the per-FFT config if any
  if (auto it = args.perFftConfig.find(fft.shape.spec()); it != args.perFftConfig.end()) {
    // log("Found %s\n", fft.shape.spec().c_str());
    config.insert(it->second.begin(), it->second.end());
  }

  // Default value for -use options that must also be parsed in C++ code
  tail_single_wide = false, tail_single_kernel = true;         // Default tailSquare is double-wide in one kernel
  in_place = 0;                                         // Default is not in-place
  wmul = 2;                                             // Default is carryFused processes two lines at a time
  pad_size = isAmdGpu(id) ? 256 : 0;                    // Default is 256 bytes for AMD, 0 for others

  // Validate -use options
  for (const auto& [k, v] : config) {
    bool const isValid = isInList(k, {
                              "FAST_BARRIER",
                              "STATS",
                              "IN_SIZEX",
                              "IN_WG",
                              "OUT_SIZEX",
                              "OUT_WG",
                              "UNROLL_H",
                              "UNROLL_W",
                              "ZEROHACK_H",
                              "ZEROHACK_W",
                              "NO_ASM",
                              "DEBUG",
                              "CARRY64",
                              "BIGLIT",                 // Deprecated
                              "NONTEMPORAL",            // Deprecated
                              "INPLACE",
                              "PAD",
                              "MIDDLE_IN_LDS_TRANSPOSE",
                              "MIDDLE_OUT_LDS_TRANSPOSE",
                              "TAIL_KERNELS",
                              "TAIL_TRIGS",
                              "TAIL_TRIGS31",
                              "TAIL_TRIGS32",
                              "TAIL_TRIGS61",
                              "TABMUL_CHAIN",
                              "TABMUL_CHAIN31",
                              "TABMUL_CHAIN32",
                              "TABMUL_CHAIN61",
                              "MODM31",
                              "LOADS","STORES",
                              "NOREG",                  // CUDA - experimental
                              "WMUL"
                            });
    if (!isValid) {
      log("Warning: unrecognized -use key '%s'\n", k.c_str());
    }

    // Some -use options are needed in both OpenCL code and C++ initialization code
    if (k == "TAIL_KERNELS") {
      if (atoi(v.c_str()) == 0) tail_single_wide = true, tail_single_kernel = true;
      if (atoi(v.c_str()) == 1) tail_single_wide = true, tail_single_kernel = false;
      if (atoi(v.c_str()) == 2) tail_single_wide = false, tail_single_kernel = true;
      if (atoi(v.c_str()) == 3) tail_single_wide = false, tail_single_kernel = false;
    }
    if (k == "INPLACE") in_place = atoi(v.c_str());
    if (k == "WMUL") wmul = atoi(v.c_str());
    if (k == "PAD") pad_size = atoi(v.c_str());
  }

  // Maximum WMUL is 32KB / (WIDTH * SHUFL_BYTES_W).  If using the 32KB maximum, LDS padding must be disabled.
  // Furthermore, I've seen the CUDA compiler refuse to create a kernel with 1024 threads.  Thus, we limit WMUL to 2 for a 1K width and to 1 for a 4K width.
  {
    u32 const shufl_bytes_w = args.value("SHUFL_BYTES_W", 8);
    u32 max_wmul = 32768 / (fft.shape.width * shufl_bytes_w);
    if (max_wmul > 2 && fft.shape.width >= 1024) max_wmul = 2;
    if (max_wmul > 1 && fft.shape.width >= 4096) max_wmul = 1;
    if (wmul > max_wmul) {
      wmul = max_wmul;
      config["WMUL"] = to_string(wmul);
      log("WMUL setting too large for this FFT width.  Changing to WMUL=%d\n", wmul);
    }
    if (fft.shape.width * shufl_bytes_w * wmul >= 32768) {
      log("Local shared memory limit of 32KB exceeded.  Changing to LDSPAD_W=0\n");
      config["LDSPAD_W"] = to_string(0);
    }
  }

  string defines = toDefine(config);
  if (doLog) { log("config: %s\n", defines.c_str()); }

  defines += toDefine(initializer_list<pair<string, u32>>{
                    {"EXP", E},
                    {"WIDTH", fft.shape.width},
                    {"SMALL_HEIGHT", fft.shape.height},
                    {"MIDDLE", fft.shape.middle},
                    {"CARRY_LEN", CARRY_LEN},
                    {"NW", fft.shape.nW()},
                    {"NH", fft.shape.nH()}
                  });

  if (isAmdGpu(id)) { defines += toDefine("AMDGPU", 1); }
  if (isNvidiaGpu(id)) { defines += toDefine("NVIDIAGPU", 1); }
  if (isNvidiaGpu(id)) { defines += toDefine("CC", getNvidiaComputeCapability(id)); }

  if ((fft.carry == CARRY_AUTO && fft.shape.needsLargeCarry(E)) || (fft.carry == CARRY_64)) {
    if (doLog) { log("Using CARRY64\n"); }
    defines += toDefine("CARRY64", 1);
  }

  u32 const N = fft.shape.size();
  defines += toDefine("FFT_VARIANT", fft.variant);
  defines += toDefine("MAXBPW", (u32)(fft.maxBpw() * 100.0f));

  if (fft.FFT_FP64 || fft.FFT_FP32) {
    defines += toDefine("WEIGHT_STEP", weightM1(N, E, fft.shape.height * fft.shape.middle, 0, 0, 1));
    defines += toDefine("IWEIGHT_STEP", invWeightM1(N, E, fft.shape.height * fft.shape.middle, 0, 0, 1));
    if (fft.FFT_FP64) defines += toDefine("TAILT", root1Fancy(fft.shape.height * 2, 1));
    else defines += toDefine("TAILT", root1FancyFP32(fft.shape.height * 2, 1));

    TrigCoefs const coefs = trigCoefs(fft.shape.size() / 4);
    defines += toDefine("TRIG_SCALE", int(coefs.scale));
    defines += toDefine("TRIG_SIN",  coefs.sinCoefs);
    defines += toDefine("TRIG_COS",  coefs.cosCoefs);
  }
  if (fft.NTT_GF31) {
    defines += toDefine("TAILTGF31", root1GF31(fft.shape.height * 2, 1));
  }
  if (fft.NTT_GF61) {
    defines += toDefine("TAILTGF61", root1GF61(fft.shape.height * 2, 1));
  }

  // Send the FFT/NTT type and booleans that enable/disable code for each possible FP and NTT
  defines += toDefine("FFT_TYPE", (int) fft.shape.fft_type);
  defines += toDefine("WordSize", fft.WordSize);

  // When using multiple NTT primes or hybrid FFT/NTT, each FFT/NTT prime's data buffer and trig values are combined into one buffer.
  // The openCL code needs to know the offset to the data and trig values.  Distances are in "number of double2 values".
  if (fft.FFT_FP64 && fft.NTT_GF31) {
    // GF31 data is located after the FP64 data.  Compute size of the FP64 data and trigs.
    defines += toDefine("DISTGF31",      FP64_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF31", SMALLTRIG_FP64_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF31", MIDDLETRIG_FP64_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF31", SMALLTRIGCOMBO_FP64_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
  }
  else if (fft.FFT_FP32 && fft.NTT_GF31 && fft.NTT_GF61) {
    // GF31 and GF61 data is located after the FP32 data.  Compute size of the FP32 data and trigs.
    u32 sz1, sz2, sz3, sz4;
    defines += toDefine("DISTGF31",      sz1 = FP32_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF31", sz2 = SMALLTRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF31", sz3 = MIDDLETRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF31", sz4 = SMALLTRIGCOMBO_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTGF61",      sz1 + GF31_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF61", sz2 + SMALLTRIG_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF61", sz3 + MIDDLETRIG_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF61", sz4 + SMALLTRIGCOMBO_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
  }
  else if (fft.FFT_FP32 && fft.NTT_GF31) {
    // GF31 data is located after the FP32 data.  Compute size of the FP32 data and trigs.
    defines += toDefine("DISTGF31",      FP32_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF31", SMALLTRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF31", MIDDLETRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF31", SMALLTRIGCOMBO_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
  }
  else if (fft.FFT_FP32 && fft.NTT_GF61) {
    // GF61 data is located after the FP32 data.  Compute size of the FP32 data and trigs.
    defines += toDefine("DISTGF61",      FP32_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF61", SMALLTRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF61", MIDDLETRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF61", SMALLTRIGCOMBO_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
  }
  else if (fft.NTT_GF31 && fft.NTT_GF61) {
    defines += toDefine("DISTGF31",      0);
    defines += toDefine("DISTWTRIGGF31", 0);
    defines += toDefine("DISTMTRIGGF31", 0);
    defines += toDefine("DISTHTRIGGF31", 0);
    // GF61 data is located after the GF31 data.  Compute size of the GF31 data and trigs.
    defines += toDefine("DISTGF61",      GF31_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF61", SMALLTRIG_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF61", MIDDLETRIG_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF61", SMALLTRIGCOMBO_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
  }
  else if (fft.NTT_GF31) {
    defines += toDefine("DISTGF31",      0);
    defines += toDefine("DISTWTRIGGF31", 0);
    defines += toDefine("DISTMTRIGGF31", 0);
    defines += toDefine("DISTHTRIGGF31", 0);
  }
  else if (fft.NTT_GF61) {
    defines += toDefine("DISTGF61",      0);
    defines += toDefine("DISTWTRIGGF61", 0);
    defines += toDefine("DISTMTRIGGF61", 0);
    defines += toDefine("DISTHTRIGGF61", 0);
  }

  // Calculate fractional bits-per-word = (E % N) / N * 2^64
  u32 const bpw_hi = (u64(E % N) << 32) / N;
  u32 const bpw_lo = (((u64(E % N) << 32) % N) << 32) / N;
  u64 bpw = (u64(bpw_hi) << 32) + bpw_lo;
  bpw--; // bpw must not be an exact value -- it must be less than exact value to get last biglit value right
  defines += toDefine("FRAC_BPW_HI", (u32) (bpw >> 32));
  defines += toDefine("FRAC_BPW_LO", (u32) bpw);

  return defines;
}

template<typename T>
pair<vector<T>, vector<T>> split(const vector<T>& v, const vector<u32>& select) {
  vector<T> a;
  vector<T> b;
  auto selIt = select.begin();
  u32 selNext = selIt == select.end() ? u32(-1) : *selIt;
  for (u32 i = 0; i < v.size(); ++i) {
    if (i == selNext) {
      b.push_back(v[i]);
      ++selIt;
      selNext = selIt == select.end() ? u32(-1) : *selIt;
    } else {
      a.push_back(v[i]);
    }
  }
  return {a, b};
}

RoeInfo roeStat(const vector<float>& roe) {
  double sumRoe = 0;
  double sum2Roe = 0;
  double maxRoe = 0;

  for (auto xf : roe) {
    double const x = xf;
    assert(x >= 0);
    maxRoe = max(x, maxRoe);
    sumRoe  += x;
    sum2Roe += x * x;
  }
  u32 const n = roe.size();

  double const sdRoe = sqrt(n * sum2Roe - sumRoe * sumRoe) / n;
  double const meanRoe = sumRoe / n;

  return {n, maxRoe, meanRoe, sdRoe};
}

class IterationTimer {
  Timer timer;
  u64 kStart;

public:
  explicit IterationTimer(u64 kStart) : kStart(kStart) { }

  float reset(u64 k) {
    float const secs = timer.reset();

    u64 const its = max(u64(1), k - kStart);
    kStart = k;
    return secs / its;
  }
};

u32 baseCheckStep(u32 blockSize) {
  switch (blockSize) {
    case 200:  return 40'000;
    case 400:  return 160'000;
    case 500:  return 200'000;
    case 1000: return 1'000'000;
    default:
      assert(false);
      return 0;
  }
}

u32 checkStepForErrors(u32 blockSize, u32 nErrors) {
  u32 const step = baseCheckStep(blockSize);
  return nErrors ? step / 2 : step;
}

string toHex(u32 x) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%08x", x);
  return buf;
}

string toHex(const vector<u32>& v) {
  string s;
  for (auto it = v.rbegin(), end = v.rend(); it != end; ++it) {
    s += toHex(*it);
  }
  return s;
}

string formatSecsPerIter(float secsPerIter) {
  char buf[64];
  float const usecsPerIter = secsPerIter * 1.0e6f;    // Convert to micro-seconds
  if (usecsPerIter > 1000.0f) {
    snprintf(buf, sizeof(buf), "%4.0f", usecsPerIter);
  } else {
    snprintf(buf, sizeof(buf), "%6.1f", usecsPerIter);
  }
  return string(buf);  
}

} // namespace

// --------

unique_ptr<Gpu> Gpu::make(u64 E, GpuCommon shared, FFTConfig fftConfig, const vector<KeyVal>& extraConf, bool logFftSize) {
  return make_unique<Gpu>(shared, fftConfig, E, extraConf, logFftSize);
}

Gpu::~Gpu() {
  // Background tasks may have captured *this*, so wait until those are complete before destruction
  background->waitEmpty();
}

// Part of GPU initialization is to compute the default number of registers each kernel should target during compilation.
// Kernel register usage is critical for maximizing GPU occupancy.  The default values can be overrriden with command line arguments.
// This feature currently only works for the CUDA compiler.
string Gpu::numCudaRegisters(enum WHICH_KERNEL which_kernel) {
#if CUDA_BACKEND
  int regs = 0;
  const char *use_override = "";
  // Allow command line to prefer the CUDA compiler's default number of registers
  if (args.value("NOREG", 0)) return string("");
  // Determine a kernel specific default maximum number of GPU registers (values set to -1 have not been tuned for best default value)
  switch (which_kernel) {
  case CARRYFUSED:         // Register usage depends on NW, the FFT/NTT type, and perhaps the long carry setting
    switch (fft.shape.fft_type) {
    case FFT64:
      regs = nW == 8 ? 80 : 64;
      use_override = "REGCF64";
      break;
    case FFT3161:
      regs = nW == 8 ? 96 : 64;
      use_override = "REGCF3161";
      break;
    case FFT3261:
      regs = nW == 8 ? 96 : 64;
      use_override = "REGCF3261";
      break;
    case FFT61:
      regs = nW == 8 ? 80 : 64;
      use_override = "REGCF61";
      break;
    case FFT323161:
      regs = nW == 8 ? 128 : 80;
      use_override = "REGCF323161";
      break;
    case FFT3231:
      regs = -1;
      use_override = "REGCF3231";
      break;
    case FFT6431:
      regs = -1;
      use_override = "REGCF6431";
      break;
    case FFT31:
      regs = -1;
      use_override = "REGCF31";
      break;
    case FFT32:
      regs = -1;
      use_override = "REGCF32";
      break;
    }
    break;
  case MIDIN:              // Register usage depends on MIDDLE and the FP32/FP64
    if (fft.FFT_FP64) {
      if (fft.shape.middle >= 16) regs = 96;
      else if (fft.shape.middle >= 14) regs = 88;
      else if (fft.shape.middle >= 13) regs = 80;
      else if (fft.shape.middle >= 10) regs = 72;
      else if (fft.shape.middle >= 7) regs = 64;
      else if (fft.shape.middle >= 5) regs = 56;
      else if (fft.shape.middle >= 4) regs = 48;
      else regs = -1;
      use_override = "REGMI64";
    } else {
      if (fft.shape.middle == 16) regs = 56;
      else if (fft.shape.middle == 8) regs = 40;
      else if (fft.shape.middle == 4) regs = 32;
      else regs = -1;
      use_override = "REGMI32";
    }
    break;
  case MIDIN31:            // Register usage depends on MIDDLE
    if (fft.shape.middle == 16) regs = 56;
    else if (fft.shape.middle == 8) regs = 44;
    else if (fft.shape.middle == 4) regs = 32;
    else regs = -1;  
    use_override = "REGMI31";
    break;
  case MIDIN61:            // Register usage depends on MIDDLE
    if (fft.shape.middle == 16) regs = 96;
    else if (fft.shape.middle == 8) regs = 72;
    else if (fft.shape.middle == 4) regs = 64;
    else regs = -1;  
    use_override = "REGMI61";
    break;
  case TAIL:               // Register usage depends on NH and the FP32/FP64 (assumes double-wide kernel)
    if (fft.FFT_FP64) {
      regs = nH == 8 ? 88 : 64;
      use_override = "REGTS64";
    } else {
      regs = nH == 8 ? 64 : 48;
      use_override = "REGTS32";
    }
    break;
  case TAIL31:             // Register usage depends on NH (assumes double-wide kernel)
    regs = nH == 8 ? 64 : 48;
    use_override = "REGTS31";
    break;
  case TAIL61:             // Register usage depends on NH (assumes double-wide kernel)
    regs = nH == 8 ? 96 : 64;
    use_override = "REGTS61";
    break;
  case MIDOUT:             // Register usage depends on MIDDLE and the FFT/NTT type
    if (fft.FFT_FP64) {
      if (fft.shape.middle >= 15) regs = 96;
      else if (fft.shape.middle >= 14) regs = 88;
      else if (fft.shape.middle >= 11) regs = 80;
      else if (fft.shape.middle >= 10) regs = 72;
      else if (fft.shape.middle >= 7) regs = 64;
      else if (fft.shape.middle >= 5) regs = 56;
      else if (fft.shape.middle >= 4) regs = 48;
      else regs = -1;
      use_override = "REGMO64";
    } else {
      if (fft.shape.middle == 16) regs = 56;
      else if (fft.shape.middle == 8) regs = 40;
      else if (fft.shape.middle == 4) regs = 32;
      else regs = -1;
      use_override = "REGMO32";
    }
    break;
  case MIDOUT31:           // Register usage depends on MIDDLE
    if (fft.shape.middle == 16) regs = 48;
    else if (fft.shape.middle == 8) regs = 40;
    else if (fft.shape.middle == 4) regs = 32;
    else regs = -1;
    use_override = "REGMO31";
    break;
  case MIDOUT61:           // Register usage depends on MIDDLE
    if (fft.shape.middle == 16) regs = 96;
    else if (fft.shape.middle == 8) regs = 64;
    else if (fft.shape.middle == 4) regs = 64;
    else regs = -1;
    use_override = "REGMO61";
    break;
  }
  // Get the optional override register count
  int const override_regs = args.value(use_override, 0);
  // If a specified override is small, use the count as a CUDA launch_bounds rather than a maximum register count
  if (override_regs && (override_regs > 0 && override_regs <= 16)) return string("-DCUDA_MIN_BLOCKS=") + to_string(override_regs) + " ";
  // If specified, override the default maximum register count
  if (override_regs) regs = override_regs;
  // Sometimes the results using CUDA compiler's default launch_bounds without setting an explicit launch bounds or maxrrregcount can't be beat
  if (regs == -1) return string("");
  // Format an explicit register count setting
  return string("--maxrregcount=") + to_string(regs) + " ";
#else
  return string("");
#endif
}

// Kernels are compiled one at a time, but OpenCL source files contain multiple kernels.  This routine set the #defines necessary so that only one kernel is compiled.
// While not strictly necessary, startup speed will be a bit faster if we do less compilations.
string Gpu::kernelDefines(enum WHICH_KERNEL_TYPE which_kernel) {
  string defines;
  // Determine the kernel specific #defines
  switch (which_kernel) {
  case KFP:         // FP64 or FP32 kernel
    defines += toDefine("FFT_FP64", (int) fft.FFT_FP64);
    defines += toDefine("FFT_FP32", (int) fft.FFT_FP32);
    defines += toDefine("NTT_GF31", 0);
    defines += toDefine("NTT_GF61", 0);
    break;
  case K31:         // GF1 kernel
    defines += toDefine("FFT_FP64", 0);
    defines += toDefine("FFT_FP32", 0);
    defines += toDefine("NTT_GF31", (int) fft.NTT_GF31);
    defines += toDefine("NTT_GF61", 0);
    break;
  case K61:         // GF61 kernel
    defines += toDefine("FFT_FP64", 0);
    defines += toDefine("FFT_FP32", 0);
    defines += toDefine("NTT_GF31", 0);
    defines += toDefine("NTT_GF61", (int) fft.NTT_GF61);
    break;
  case KALL:        // Kernels, like carryFused, that need all defines set properly
    defines += toDefine("FFT_FP64", (int) fft.FFT_FP64);
    defines += toDefine("FFT_FP32", (int) fft.FFT_FP32);
    defines += toDefine("NTT_GF31", (int) fft.NTT_GF31);
    defines += toDefine("NTT_GF61", (int) fft.NTT_GF61);
    break;
  }
  return defines + " ";
}

enum {
ROE_SIZE = 100000,
CARRY_SIZE = 100000
};

Gpu::Gpu(GpuCommon s, FFTConfig fft, u64 E, const vector<KeyVal>& extraConf, bool logFftSize) :
  shared(s),
  background{shared.background},
  args{*shared.args},
  E(E),
  N(fft.shape.size()),
  fft(fft),
  WIDTH(fft.shape.width),
  SMALL_H(fft.shape.height),
  BIG_H(SMALL_H * fft.shape.middle),
  hN(N / 2),
  nW(fft.shape.nW()),
  nH(fft.shape.nH()),
  useLongCarry{args.carry == CARRY_64},
  queue{*shared.context, args.profile},
      
  compiler{args, shared.context, clDefines(args, shared.context->deviceId(), fft, extraConf, E, logFftSize, tail_single_wide, tail_single_kernel, in_place, pad_size, wmul)},

#define K(name, ...) name(#name, &compiler, profile.make(#name), &queue, __VA_ARGS__)

  K(kfftMidIn,             "fftmiddlein.cl",  "fftMiddleIn",  hN / (BIG_H / SMALL_H), kernelDefines(KFP) + numCudaRegisters(MIDIN)),
  K(kfftHin,               "ffthin.cl",  "fftHin",  hN / nH, kernelDefines(KFP)),
  K(ktailSquareZero,       "tailsquare.cl", "tailSquareZero", SMALL_H / nH * 2, kernelDefines(KFP)),
  K(ktailSquare,           "tailsquare.cl", "tailSquare",
                                               !tail_single_wide && !tail_single_kernel ? hN / nH - SMALL_H / nH * 2 : // Double-wide tailSquare with two kernels
                                               !tail_single_wide ? hN / nH :                                           // Double-wide tailSquare with one kernel
                                               !tail_single_kernel ? hN / nH / 2 - SMALL_H / nH :                      // Single-wide tailSquare with two kernels
                                               hN / nH / 2, kernelDefines(KFP) + numCudaRegisters(TAIL)),    // Single-wide tailSquare with one kernel
  K(ktailMul,              "tailmul.cl", "tailMul", hN / nH / 2, kernelDefines(KFP)),
  K(ktailMulLow,           "tailmul.cl", "tailMul", hN / nH / 2, kernelDefines(KFP) + "-DMUL_LOW=1"),
  K(kfftMidOut,            "fftmiddleout.cl", "fftMiddleOut", hN / (BIG_H / SMALL_H), kernelDefines(KFP) + numCudaRegisters(MIDOUT)),
  K(kfftW,                 "fftw.cl", "fftW", hN / nW, kernelDefines(KFP)),

  K(kfftMidInGF31,         "fftmiddlein.cl",  "fftMiddleInGF31",  hN / (BIG_H / SMALL_H), kernelDefines(K31) + numCudaRegisters(MIDIN31)),
  K(kfftHinGF31,           "ffthin.cl",  "fftHinGF31",  hN / nH, kernelDefines(K31)),
  K(ktailSquareZeroGF31,   "tailsquare.cl", "tailSquareZeroGF31", SMALL_H / nH * 2, kernelDefines(K31)),
  K(ktailSquareGF31,       "tailsquare.cl", "tailSquareGF31",
                                               !tail_single_wide && !tail_single_kernel ? hN / nH - SMALL_H / nH * 2 : // Double-wide tailSquare with two kernels
                                               !tail_single_wide ? hN / nH :                                           // Double-wide tailSquare with one kernel
                                               !tail_single_kernel ? hN / nH / 2 - SMALL_H / nH :                      // Single-wide tailSquare with two kernels
                                               hN / nH / 2, kernelDefines(K31) + numCudaRegisters(TAIL31)),  // Single-wide tailSquare with one kernel
  K(ktailMulGF31,          "tailmul.cl", "tailMulGF31", hN / nH / 2, kernelDefines(K31)),
  K(ktailMulLowGF31,       "tailmul.cl", "tailMulGF31", hN / nH / 2, kernelDefines(K31) + "-DMUL_LOW=1"),
  K(kfftMidOutGF31,        "fftmiddleout.cl", "fftMiddleOutGF31", hN / (BIG_H / SMALL_H), kernelDefines(K31) + numCudaRegisters(MIDOUT31)),
  K(kfftWGF31,             "fftw.cl", "fftWGF31", hN / nW, kernelDefines(K31)),

  K(kfftMidInGF61,         "fftmiddlein.cl",  "fftMiddleInGF61",  hN / (BIG_H / SMALL_H), kernelDefines(K61) + numCudaRegisters(MIDIN61)),
  K(kfftHinGF61,           "ffthin.cl",  "fftHinGF61",  hN / nH, kernelDefines(K61)),
  K(ktailSquareZeroGF61,   "tailsquare.cl", "tailSquareZeroGF61", SMALL_H / nH * 2, kernelDefines(K61)),
  K(ktailSquareGF61,       "tailsquare.cl", "tailSquareGF61",
                                               !tail_single_wide && !tail_single_kernel ? hN / nH - SMALL_H / nH * 2 : // Double-wide tailSquare with two kernels
                                               !tail_single_wide ? hN / nH :                                           // Double-wide tailSquare with one kernel
                                               !tail_single_kernel ? hN / nH / 2 - SMALL_H / nH :                      // Single-wide tailSquare with two kernels
                                               hN / nH / 2, kernelDefines(K61) + numCudaRegisters(TAIL61)),  // Single-wide tailSquare with one kernel
  K(ktailMulGF61,          "tailmul.cl", "tailMulGF61", hN / nH / 2, kernelDefines(K61)),
  K(ktailMulLowGF61,       "tailmul.cl", "tailMulGF61", hN / nH / 2, kernelDefines(K61) + "-DMUL_LOW=1"),
  K(kfftMidOutGF61,        "fftmiddleout.cl", "fftMiddleOutGF61", hN / (BIG_H / SMALL_H), kernelDefines(K61) + numCudaRegisters(MIDOUT61)),
  K(kfftWGF61,             "fftw.cl", "fftWGF61", hN / nW, kernelDefines(K61)),

  K(kfftP,                 "fftp.cl", "fftP", hN / nW, kernelDefines(KALL)),
  K(kCarryA,               "carry.cl", "carry", hN / CARRY_LEN, kernelDefines(KALL)),
  K(kCarryAROE,            "carry.cl", "carry", hN / CARRY_LEN, kernelDefines(KALL) + "-DROE=1"),
  K(kCarryM,               "carry.cl", "carry", hN / CARRY_LEN, kernelDefines(KALL) + "-DMUL3=1"),
  K(kCarryMROE,            "carry.cl", "carry", hN / CARRY_LEN, kernelDefines(KALL) + "-DMUL3=1 -DROE=1"),
  K(kCarryLL,              "carry.cl", "carry", hN / CARRY_LEN, kernelDefines(KALL) + "-DLL=1"),
  K(kCarryFused,           "carryfused.cl", "carryFused", WIDTH * (BIG_H + wmul) / nW, kernelDefines(KALL) + numCudaRegisters(CARRYFUSED)),
  K(kCarryFusedROE,        "carryfused.cl", "carryFused", WIDTH * (BIG_H + wmul) / nW, kernelDefines(KALL) + numCudaRegisters(CARRYFUSED) + "-DROE=1"),
  K(kCarryFusedMul,        "carryfused.cl", "carryFused", WIDTH * (BIG_H + wmul) / nW, kernelDefines(KALL) + numCudaRegisters(CARRYFUSED) + "-DMUL3=1"),
  K(kCarryFusedMulROE,     "carryfused.cl", "carryFused", WIDTH * (BIG_H + wmul) / nW, kernelDefines(KALL) + numCudaRegisters(CARRYFUSED) + "-DMUL3=1 -DROE=1"),
  K(kCarryFusedLL,         "carryfused.cl", "carryFused", WIDTH * (BIG_H + wmul) / nW, kernelDefines(KALL) + numCudaRegisters(CARRYFUSED) + "-DLL=1"),

  K(carryB,                "carryb.cl", "carryB",   hN / CARRY_LEN, kernelDefines(KALL)),

  // 64
  K(transpIn,  "transpose.cl", "transposeIn",  hN / 64),
  K(transpOut, "transpose.cl", "transposeOut", hN / 64),

  K(readResidue, "etc.cl", "readResidue", 32, "-DREADRESIDUE=1"),

  // 256
  K(kernIsEqual, "etc.cl", "isEqual", 256 * 256, "-DISEQUAL=1"),
  K(sum64,       "etc.cl", "sum64",   256 * 256, "-DSUM64=1"),

  K(testTrig, "selftest.cl", "testTrig", 256 * 256),
  K(testFFT4, "selftest.cl", "testFFT4", 256),
  K(testFFT14, "selftest.cl", "testFFT14", 256),
  K(testFFT15, "selftest.cl", "testFFT15", 256),
  K(testFFT, "selftest.cl", "testFFT", 256),
  K(testTime, "selftest.cl", "testTime", 4096 * 64),

#undef K

  bufTrigH{shared.bufCache->smallTrigCombo(shared.args, fft, WIDTH, fft.shape.middle, SMALL_H, nH, tail_single_wide)},
  bufTrigM{shared.bufCache->middleTrig(shared.args, fft, SMALL_H, BIG_H / SMALL_H, WIDTH)},
  bufTrigW{shared.bufCache->smallTrig(shared.args, fft, WIDTH, nW, fft.shape.middle, SMALL_H, nH, tail_single_wide)},

  weights{genWeights(fft, E, WIDTH, BIG_H, nW, isNvidiaGpu(shared.context->deviceId()))},
  bufConstWeights{shared.context, std::move(weights.weightsConstIF)},
  bufWeights{shared.context,      std::move(weights.weightsIF)},

#define BUF(name, ...) name{profile.make(#name), &queue, __VA_ARGS__}

  // GPU Buffers containing integer data.  Since this buffer is type i64, if fft.WordSize < 8 then we need less memory allocated.
  BUF(bufData, N * fft.WordSize / sizeof(Word)),
  BUF(bufAux, N * fft.WordSize / sizeof(Word)),
  BUF(bufCheck, N * fft.WordSize / sizeof(Word)),
  // Every double-word (i.e. N/2) produces one carry. In addition we may have one extra group thus WIDTH more carries.
  BUF(bufCarry, N / 2 + WIDTH),
  BUF(bufReady, (N / 2 + WIDTH) / 32), // Every wavefront (32 or 64 lanes) needs to signal "carry is ready"

  BUF(bufSmallOut, 256),
  BUF(bufSumOut,     1),
  BUF(bufTrue,       1),
  BUF(bufROE, ROE_SIZE),
  BUF(bufStatsCarry, CARRY_SIZE),

  BUF(buf1, TOTAL_DATA_SIZE(fft, WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size)),
  BUF(buf2, TOTAL_DATA_SIZE(fft, WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size)),
  BUF(buf3, TOTAL_DATA_SIZE(fft, WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size)),
#undef BUF

  statsBits{u32(args.value("STATS", 0))},
  timeBufVect{profile.make("proofBufVect")}
  
{    

  float const bitsPerWord = E / float(N);
  if (logFftSize) {
    log("FFT: %s %s (%.2f bpw)\n", numberK(N).c_str(), fft.spec().c_str(), bitsPerWord);

    // Sometimes we do want to run a FFT beyond a reasonable BPW (e.g. during -ztune), and these situations
    // coincide with logFftSize == false
    if (fft.maxExp() < E) {
      log("Warning: %s (max %" PRIu64 ") may be too small for %" PRIu64 "\n", fft.spec().c_str(), fft.maxExp(), E);
    }
  }

  if (bitsPerWord < fft.minBpw()) {
    log("FFT size too large for exponent (%.2f bits/word < %.2f bits/word).\n", bitsPerWord, fft.minBpw());
    throw "FFT size too large";
  }

  useLongCarry = useLongCarry || (bitsPerWord < 10.0);

  if (useLongCarry) { log("Using long carry!\n"); }

  if (fft.FFT_FP64 || fft.FFT_FP32) {
    kfftMidIn.setFixedArgs(2, bufTrigM);
    kfftHin.setFixedArgs(2, bufTrigH);
    ktailSquareZero.setFixedArgs(2, bufTrigH);
    ktailSquare.setFixedArgs(2, bufTrigH);
    ktailMulLow.setFixedArgs(3, bufTrigH);
    ktailMul.setFixedArgs(3, bufTrigH);
    kfftMidOut.setFixedArgs(2, bufTrigM);
    kfftW.setFixedArgs(2, bufTrigW);
  }

  if (fft.NTT_GF31) {
    kfftMidInGF31.setFixedArgs(2, bufTrigM);
    kfftHinGF31.setFixedArgs(2, bufTrigH);
    ktailSquareZeroGF31.setFixedArgs(2, bufTrigH);
    ktailSquareGF31.setFixedArgs(2, bufTrigH);
    ktailMulLowGF31.setFixedArgs(3, bufTrigH);
    ktailMulGF31.setFixedArgs(3, bufTrigH);
    kfftMidOutGF31.setFixedArgs(2, bufTrigM);
    kfftWGF31.setFixedArgs(2, bufTrigW);
  }

  if (fft.NTT_GF61) {
    kfftMidInGF61.setFixedArgs(2, bufTrigM);
    kfftHinGF61.setFixedArgs(2, bufTrigH);
    ktailSquareZeroGF61.setFixedArgs(2, bufTrigH);
    ktailSquareGF61.setFixedArgs(2, bufTrigH);
    ktailMulLowGF61.setFixedArgs(3, bufTrigH);
    ktailMulGF61.setFixedArgs(3, bufTrigH);
    kfftMidOutGF61.setFixedArgs(2, bufTrigM);
    kfftWGF61.setFixedArgs(2, bufTrigW);
  }

  if (fft.FFT_FP64 || fft.FFT_FP32) {                         // The FP versions take bufWeight arguments
    kfftP.setFixedArgs(2, bufTrigW, bufWeights);
    for (Kernel* k : {&kCarryA, &kCarryAROE, &kCarryM, &kCarryMROE, &kCarryLL}) { k->setFixedArgs(3, bufCarry, bufWeights); }
    for (Kernel* k : {&kCarryA, &kCarryM, &kCarryLL}) { k->setFixedArgs(5, bufStatsCarry); }
    for (Kernel* k : {&kCarryAROE, &kCarryMROE})      { k->setFixedArgs(5, bufROE); }
    for (Kernel* k : {&kCarryFused, &kCarryFusedROE, &kCarryFusedMul, &kCarryFusedMulROE, &kCarryFusedLL}) {
      k->setFixedArgs(3, bufCarry, bufReady, bufTrigW, bufConstWeights, bufWeights);
    }
    for (Kernel* k : {&kCarryFusedROE, &kCarryFusedMulROE})           { k->setFixedArgs(8, bufROE); }
    for (Kernel* k : {&kCarryFused, &kCarryFusedMul, &kCarryFusedLL}) { k->setFixedArgs(8, bufStatsCarry); }
  } else {
    kfftP.setFixedArgs(2, bufTrigW);
    for (Kernel* k : {&kCarryA, &kCarryAROE, &kCarryM, &kCarryMROE, &kCarryLL}) { k->setFixedArgs(3, bufCarry); }
    for (Kernel* k : {&kCarryA, &kCarryM, &kCarryLL}) { k->setFixedArgs(4, bufStatsCarry); }
    for (Kernel* k : {&kCarryAROE, &kCarryMROE})      { k->setFixedArgs(4, bufROE); }
    for (Kernel* k : {&kCarryFused, &kCarryFusedROE, &kCarryFusedMul, &kCarryFusedMulROE, &kCarryFusedLL}) {
      k->setFixedArgs(3, bufCarry, bufReady, bufTrigW);
    }
    for (Kernel* k : {&kCarryFusedROE, &kCarryFusedMulROE}) { k->setFixedArgs(6, bufROE); }
    for (Kernel* k : {&kCarryFused, &kCarryFusedMul, &kCarryFusedLL}) { k->setFixedArgs(6, bufStatsCarry); }
  }

  carryB.setFixedArgs(1, bufCarry);

  kernIsEqual.setFixedArgs(2, bufTrue);

  bufReady.zero();
  bufROE.zero();
  bufStatsCarry.zero();
  bufTrue.write({1});

  if (args.verbose) {
    selftestTrig();
  }

  // If MULTI_Q option is set there are fewer kernels executed in the main queue, but there are some additional syncEvents and waits.
  // That's a total of 4 kernels (carryFused, MidIn/Out, TailSquare) plus 1 syncEvent plus 1 or more syncWaits.
  // If MULTI_Q option is not set there is carryFused + MidIn/Out and TailSquare for each NTT modulus.
  // NOTE: We dont take into account the optional tailSquareZero kernel.  We theoretically should.
  if (args.value("MULTI_Q", 0))
    queue.setSquareKernels(5 + ((fft.FFT_FP64 + fft.FFT_FP32 + fft.NTT_GF31 + fft.NTT_GF61) - 1));
  else
    queue.setSquareKernels(1 + 3 * (fft.FFT_FP64 + fft.FFT_FP32 + fft.NTT_GF31 + fft.NTT_GF61));
  queue.finish();
}

// Optionallly split some of the MiddleIn/Tail/MiddleOut kernels off od executing on the main queue to run on an auxiliary queue.
// This will increase GPU occupancy but will negatively impact L2 cache coherency.
// If the L2 cache is large enough so that all FFT data fits in the cache, this ought to be a win.
// If the L2 cache is small enough such that L2 cache hits are very low anyway, this might be a win.

void Gpu::splitQueue() {

  // If MULTI_Q -use not set, return
  if (!args.value("MULTI_Q", 0)) return;

  // Create aux queues.  For now, we only have one auxiliary queue.  We could do more.
  if (auxQueues.empty()) {
    auxQueues.emplace_back(*shared.context, args.profile, true);
  }

  // Queue a sync event in the main queue.  Have all auxiliary queues wait on the event.
  EventHolder event = queue.createSyncEvent();
  for (auto & auxQueue : auxQueues) {
    auxQueue.waitForSyncEvent(&event);
  }

  // Assign kernels to running on the main queue or an auxiliary queue

  int which_queue = -1;

  // For no particularly good reason, put a kernel that operates on 64-bit vaules in the main queue.
  if (fft.NTT_GF61) {
    if (which_queue != -1) {
      kfftMidInGF61.setQueue(&auxQueues[which_queue]);
      kfftHinGF61.setQueue(&auxQueues[which_queue]);
      ktailSquareZeroGF61.setQueue(&auxQueues[which_queue]);
      ktailSquareGF61.setQueue(&auxQueues[which_queue]);
      ktailMulGF61.setQueue(&auxQueues[which_queue]);
      ktailMulLowGF61.setQueue(&auxQueues[which_queue]);
      kfftMidOutGF61.setQueue(&auxQueues[which_queue]);
      kfftWGF61.setQueue(&auxQueues[which_queue]);
    }
    which_queue++;
  }

  if (fft.FFT_FP64 || fft.FFT_FP32) {
    if (which_queue != -1) {
      kfftMidIn.setQueue(&auxQueues[which_queue]);
      kfftHin.setQueue(&auxQueues[which_queue]);
      ktailSquareZero.setQueue(&auxQueues[which_queue]);
      ktailSquare.setQueue(&auxQueues[which_queue]);
      ktailMul.setQueue(&auxQueues[which_queue]);
      ktailMulLow.setQueue(&auxQueues[which_queue]);
      kfftMidOut.setQueue(&auxQueues[which_queue]);
      kfftW.setQueue(&auxQueues[which_queue]);
    }
    // For no particularly good reason, put kernels that operate on 32-bit value in the same queue unless there are no kernels operating on 64-bit values
    if (fft.FFT_FP64 || (fft.FFT_FP32 && which_queue == -1)) {
      which_queue++;
    }
  }

  if (fft.NTT_GF31) {
    if (which_queue != -1) {
      kfftMidInGF31.setQueue(&auxQueues[which_queue]);
      kfftHinGF31.setQueue(&auxQueues[which_queue]);
      ktailSquareZeroGF31.setQueue(&auxQueues[which_queue]);
      ktailSquareGF31.setQueue(&auxQueues[which_queue]);
      ktailMulGF31.setQueue(&auxQueues[which_queue]);
      ktailMulLowGF31.setQueue(&auxQueues[which_queue]);
      kfftMidOutGF31.setQueue(&auxQueues[which_queue]);
      kfftWGF31.setQueue(&auxQueues[which_queue]);
    }
    //which_queue++;
  }
}

void Gpu::mergeQueue() {

  // If MULTI_Q -use not set, return
  if (!args.value("MULTI_Q", 0)) return;

  // Queue a sync event in each auxiliary queue(s).  Wait on the event(s) in the main queue.
  for (auto & auxQueue : auxQueues) {
    EventHolder event = auxQueue.createSyncEvent();
    queue.waitForSyncEvent(&event);
  }

  // Return kernels to running on the main queue
  // NOTE: I believe there is no need to switch queues back and forth between the main and auxiliary queues.  No one currently uses the cache_group == 0 option.
  if (fft.NTT_GF61) {
    kfftMidInGF61.setQueue(&queue);
    kfftHinGF61.setQueue(&queue);
    ktailSquareZeroGF61.setQueue(&queue);
    ktailSquareGF61.setQueue(&queue);
    ktailMulGF61.setQueue(&queue);
    ktailMulLowGF61.setQueue(&queue);
    kfftMidOutGF61.setQueue(&queue);
    kfftWGF61.setQueue(&queue);
  }
  if (fft.FFT_FP64 || fft.FFT_FP32) {
    kfftMidIn.setQueue(&queue);
    kfftHin.setQueue(&queue);
    ktailSquareZero.setQueue(&queue);
    ktailSquare.setQueue(&queue);
    ktailMul.setQueue(&queue);
    ktailMulLow.setQueue(&queue);
    kfftMidOut.setQueue(&queue);
    kfftW.setQueue(&queue);
  }
  if (fft.NTT_GF31) {
    kfftMidInGF31.setQueue(&queue);
    kfftHinGF31.setQueue(&queue);
    ktailSquareZeroGF31.setQueue(&queue);
    ktailSquareGF31.setQueue(&queue);
    ktailMulGF31.setQueue(&queue);
    ktailMulLowGF31.setQueue(&queue);
    kfftMidOutGF31.setQueue(&queue);
    kfftWGF31.setQueue(&queue);
  }
}

// Replay the recorded bottom half kernels in a cache friendly order.  We support several
// options here using multiple openCl command queues.
void Gpu::replay() {

  // If using multiple command queues, handle that now.
  splitQueue();

  // For better L2 cache locality, operate on all the FP data, then operate on all the GF31 data, then GF61.
  for (int cache_group = 1; cache_group <= NUM_CACHE_GROUPS; ++cache_group) {

    // Check for irrelevant cache gouup
    if (cache_group == 1 && !(fft.FFT_FP64 || fft.FFT_FP32)) continue;
    if (cache_group == 2 && !fft.NTT_GF31) continue;
    if (cache_group == 3 && !fft.NTT_GF61) continue;

    // Iterate over the recorded kernels
    int arg = 0;
    for (auto kern : recorded_kernels) {

      // Call the appropriate kernel
      if (kern == KMIDIN) {
        Buffer<double>  const*buf = recorded_kernel_args[arg++];
        // If not in place, the input is from the scratch buffer
        Buffer<double>  const*in = in_place ? buf : &buf3;
        Buffer<double>  const*out = buf;
        if (cache_group == 1) kfftMidIn(*out, *in);
        if (cache_group == 2) kfftMidInGF31(*out, *in);
        if (cache_group == 3) kfftMidInGF61(*out, *in);
      }

      if (kern == KFFTHIN) {
        Buffer<double>  const*out = recorded_kernel_args[arg++];
        Buffer<double>  const*in = recorded_kernel_args[arg++];
        if (cache_group == 1) kfftHin(*out, *in);
        if (cache_group == 2) kfftHinGF31(*out, *in);
        if (cache_group == 3) kfftHinGF61(*out, *in);
      }

      if (kern == KTAILSQUARE) {
        Buffer<double>  const*buf = recorded_kernel_args[arg++];
        // If not in place, the output is to the scratch buffer
        Buffer<double>  const*in = buf;
        Buffer<double>  const*out = in_place ? buf : &buf3;
        if (!tail_single_kernel) {
          if (cache_group == 1) ktailSquareZero(*out, *in);
          if (cache_group == 2) ktailSquareZeroGF31(*out, *in);
          if (cache_group == 3) ktailSquareZeroGF61(*out, *in);
        }
        if (cache_group == 1) ktailSquare(*out, *in);
        if (cache_group == 2) ktailSquareGF31(*out, *in);
        if (cache_group == 3) ktailSquareGF61(*out, *in);
      }

      if (kern == KTAILMUL) {
        Buffer<double>  const*buf = recorded_kernel_args[arg++];
        Buffer<double>  const*in2 = recorded_kernel_args[arg++];
        // If not in place, the output is to the scratch buffer
        Buffer<double>  const*in1 = buf;
        Buffer<double>  const*out = in_place ? buf : &buf3;
        if (cache_group == 1) ktailMul(*out, *in1, *in2);
        if (cache_group == 2) ktailMulGF31(*out, *in1, *in2);
        if (cache_group == 3) ktailMulGF61(*out, *in1, *in2);
      }

      if (kern == KTAILMULLOW) {
        Buffer<double>  const*buf = recorded_kernel_args[arg++];
        Buffer<double>  const*in2 = recorded_kernel_args[arg++];
        // If not in place, the output is to the scratch buffer
        Buffer<double>  const*in1 = buf;
        Buffer<double>  const*out = in_place ? buf : &buf3;
        if (cache_group == 1) ktailMulLow(*out, *in1, *in2);
        if (cache_group == 2) ktailMulLowGF31(*out, *in1, *in2);
        if (cache_group == 3) ktailMulLowGF61(*out, *in1, *in2);
      }

      if (kern == KMIDOUT) {
        Buffer<double>  const*buf = recorded_kernel_args[arg++];
        // If not in place, the input is from the scratch buffer
        Buffer<double>  const*in = in_place ? buf : &buf3;
        Buffer<double>  const*out = buf;
        if (cache_group == 1) kfftMidOut(*out, *in);
        if (cache_group == 2) kfftMidOutGF31(*out, *in);
        if (cache_group == 3) kfftMidOutGF61(*out, *in);
      }

      if (kern == KFFTW) {
        Buffer<double>  const*out = recorded_kernel_args[arg++];
        Buffer<double>  const*in = recorded_kernel_args[arg++];
        if (cache_group == 1) kfftW(*out, *in);
        if (cache_group == 2) kfftWGF31(*out, *in);
        if (cache_group == 3) kfftWGF61(*out, *in);
      }
    }
  }

  // Empty the recorded kernels queue
  recorded_kernels.clear();
  recorded_kernel_args.clear();

  // If using multiple command queues, go back to a single command queue
  mergeQueue();
}

// Call the appropriate kernels to support hybrid FFTs and NTTs

void Gpu::fftP(Buffer<double>& buf, Buffer<Word>& in) {
  // If not in place, instead write the output to the scratch buffer
  Buffer<double>  const*out = in_place ? &buf : &buf3;
  kfftP(*out, in);
}

void Gpu::fftMidIn(Buffer<double>& buf) {
  // Record this call for later playback
  recorded_kernels.push_back(KMIDIN);
  recorded_kernel_args.push_back(&buf);
}

void Gpu::fftHin(Buffer<double>& out, Buffer<double>& in) {
  // Record this call for later playback
  recorded_kernels.push_back(KFFTHIN);
  recorded_kernel_args.push_back(&out);
  recorded_kernel_args.push_back(&in);
}

void Gpu::tailSquare(Buffer<double>& buf) {
  // Record this call for later playback
  recorded_kernels.push_back(KTAILSQUARE);
  recorded_kernel_args.push_back(&buf);
}

void Gpu::tailMul(Buffer<double>& buf, Buffer<double>& in2) {
  // Record this call for later playback
  recorded_kernels.push_back(KTAILMUL);
  recorded_kernel_args.push_back(&buf);
  recorded_kernel_args.push_back(&in2);
}

void Gpu::tailMulLow(Buffer<double>& buf, Buffer<double>& in2) {
  // Record this call for later playback
  recorded_kernels.push_back(KTAILMULLOW);
  recorded_kernel_args.push_back(&buf);
  recorded_kernel_args.push_back(&in2);
}

void Gpu::fftMidOut(Buffer<double>& buf) {
  // Record this call for later playback
  recorded_kernels.push_back(KMIDOUT);
  recorded_kernel_args.push_back(&buf);
}

void Gpu::fftW(Buffer<double>& out, Buffer<double>& in) {
  // Record this call for later playback
  recorded_kernels.push_back(KFFTW);
  recorded_kernel_args.push_back(&out);
  recorded_kernel_args.push_back(&in);
  // This kernel always ends the "bottom half".  Replay the recorded kernel calls.
  replay();
}

void Gpu::carryA(Buffer<Word>& out, Buffer<double>& in) {
  assert(roePos <= ROE_SIZE);
  roePos < wantROE ? kCarryAROE(out, in, roePos++)
                   : kCarryA(out, in, updateCarryPos(1 << 2));
}

void Gpu::carryM(Buffer<Word>& out, Buffer<double>& in) {
  assert(roePos <= ROE_SIZE);
  roePos < wantROE ? kCarryMROE(out, in, roePos++)
                   : kCarryM(out, in, updateCarryPos(1 << 3));
}

void Gpu::carryLL(Buffer<Word>& out, Buffer<double>& in) {
  kCarryLL(out, in, updateCarryPos(1 << 2));
}

void Gpu::carryFused(Buffer<double>& buf) {
  // This kernel always ends the "bottom half".  Replay the recorded kernel calls.
  replay();
  assert(roePos <= ROE_SIZE);
  // Like fftP, if not in place write the output to the scratch buffer
  Buffer<double>  const*in = &buf;
  Buffer<double>  const*out = in_place ? &buf : &buf3;
  roePos < wantROE ? kCarryFusedROE(*out, *in, roePos++)
                   : kCarryFused(*out, *in, updateCarryPos(1 << 0));
}

void Gpu::carryFusedMul(Buffer<double>& buf) {
  // This kernel always ends the "bottom half".  Replay the recorded kernel calls.
  replay();
  assert(roePos <= ROE_SIZE);
  // Like fftP, if not in place write the output to the scratch buffer
  Buffer<double>  const*in = &buf;
  Buffer<double>  const*out = in_place ? &buf : &buf3;
  roePos < wantROE ? kCarryFusedMulROE(*out, *in, roePos++)
                   : kCarryFusedMul(*out, *in, updateCarryPos(1 << 1));
}

void Gpu::carryFusedLL(Buffer<double>& buf) {
  // This kernel always ends the "bottom half".  Replay the recorded kernel calls.
  replay();
  // Like fftP, if not in place write the output to the scratch buffer
  Buffer<double>  const*in = &buf;
  Buffer<double>  const*out = in_place ? &buf : &buf3;
  kCarryFusedLL(*out, *in, updateCarryPos(1 << 0));
}


#if 0
void Gpu::measureTransferSpeed() {
  u32 SIZE_MB = 16;
  vector<double> data(SIZE_MB * 1024 * 1024, 1);
  Buffer<double> buf{profile.make("DMA"), &queue, SIZE};

  Timer t;
  for (int i = 0; i < 4; ++i) {
    buf.write(data);
    log("buffer Write : %f GB/s\n", double(SIZE / 1024 / 1024) * sizeof(double) / (1024 * t.reset()));
  }

  for (int i = 0; i < 4; ++i) {
    buf.read(data);
    // queue.finish();
    log("buffer READ : %f GB/s\n", double(SIZE / 1024 / 1024) * sizeof(double) / (1024 * t.reset()));
  }

  queue.finish();
}
#endif

u32 Gpu::updateCarryPos(u32 bit) {
  return (statsBits & bit) && (carryPos < CARRY_SIZE) ? carryPos++ : carryPos;
}

vector<Buffer<Word>> Gpu::makeBufVector(u32 size) {
  vector<Buffer<Word>> r;
  r.reserve(size);
for (u32 i = 0; i < size; ++i) { r.emplace_back(timeBufVect, &queue, N); }
  return r;
}

pair<RoeInfo, RoeInfo> Gpu::readROE() {
  assert(roePos <= ROE_SIZE);
  if (roePos) {
    vector<float> const roe = bufROE.read(roePos);
    assert(roe.size() == roePos);
    bufROE.zero(roePos);
    roePos = 0;
    auto [squareRoe, mulRoe] = split(roe, mulRoePos);
    mulRoePos.clear();
    return {roeStat(squareRoe), roeStat(mulRoe)};
  } 
    return {};
 
}

RoeInfo Gpu::readCarryStats() {
  assert(carryPos <= CARRY_SIZE);
  if (carryPos == 0) { return {}; }
  vector<float> const carry = bufStatsCarry.read(carryPos);
  assert(carry.size() == carryPos);
  bufStatsCarry.zero(carryPos);
  carryPos = 0;

  RoeInfo ret = roeStat(carry);

#if 0
  log("%s\n", ret.toString().c_str());

  std::sort(carry.begin(), carry.end());
  File fo = File::openAppend("carry.txt");
  auto it = carry.begin();
  u32 n = carry.size();
  u32 c = 0;
  for (int i=0; i < 500; ++i) {
    double y = 0.23 + (0.48 - 0.23) / 500 * i;
    while (it < carry.end() && *it < y) {
      ++c;
      ++it;
    }
    fo.printf("%f %f\n", y, c / double(n));
  }

  // for (auto x : carry) { fo.printf("%f\n", x); }
  fo.printf("\n\n");
#endif

  return ret;
}

template<typename T>
static bool isAllZero(vector<T> v) { return std::all_of(v.begin(), v.end(), [](T x) { return x == 0;}); }

// Read from GPU, verifying the transfer with a sum, and retry on failure.
vector<Word> Gpu::readChecked(Buffer<Word>& buf) {
  for (int nRetry = 0; nRetry < 3; ++nRetry) {
    bufSumOut.zero();
    sum64(bufSumOut, N, buf);
    vector<u64> expectedVect(1);
    bufSumOut.readAsync(expectedVect);

    vector<Word> data = readOut(buf);
    u64 hostSum = 0;
    for (auto it = data.begin(), end = data.end(); it < end; ++it) hostSum += u64(*it);

    u64 const gpuSum = expectedVect[0];
    if (hostSum == gpuSum) {
      // A buffer containing all-zero is exceptional, so mark that through the empty vector.
      if (gpuSum == 0 && isAllZero(data)) {
        log("Read ZERO\n");
        return {};
      }
      return data;
    }

    log("GPU read failed: %016" PRIx64 " (gpu) != %016" PRIx64 " (host)\n", gpuSum, hostSum);
  }
  throw "GPU persistent read errors";
}

Words Gpu::readAndCompress(Buffer<Word>& buf)  { return compactBits(readChecked(buf), E); }
vector<u32> Gpu::readCheck() { return readAndCompress(bufCheck); }
vector<u32> Gpu::readData() { return readAndCompress(bufData); }

// ioA := ioA * inB; inB must be the output of fftMidIn; inB is preserved
void Gpu::mul(Buffer<Word>& ioA, Buffer<double>& inB, Buffer<double>& tmp1, bool mul3) {
  fftP(tmp1, ioA);
  fftMidIn(tmp1);
  tailMul(tmp1, inB);
  fftMidOut(tmp1);
  fftW(buf3, tmp1);

  // Register the current ROE pos as multiplication (vs. a squaring)
  if (mulRoePos.empty() || mulRoePos.back() < roePos) { mulRoePos.push_back(roePos); }

  if (mul3) { carryM(ioA, buf3); } else { carryA(ioA, buf3); }
  carryB(ioA);
}

// ioA := ioA * inB; inB will end up in buf1 in the LEAD_MIDDLE state
void Gpu::modMul(Buffer<Word>& ioA, Buffer<Word>& inB, bool mul3) {
  modMul(ioA, inB, LEAD_NONE, mul3);
};

// ioA := ioA * inB; inB will end up in buf1 in the LEAD_MIDDLE state
void Gpu::modMul(Buffer<Word>& ioA, Buffer<Word>& inB, enum LEAD_TYPE leadInB, bool mul3) {
  if (leadInB == LEAD_NONE) fftP(buf1, inB);
  if (leadInB != LEAD_MIDDLE) fftMidIn(buf1);
  replay();  // Work around an odd bug.  The above executed fftP writing to buf3 if !in_place and queued fftMidIn.  If we don't replay now, mul will call fftP again overwriting buf3.
  mul(ioA, buf1, buf2, mul3);
};

void Gpu::writeState(u64 k, const vector<u32>& check, u32 blockSize) {
  assert(blockSize > 0);
  writeIn(bufCheck, check);

  bufData << bufCheck;
  bufAux  << bufCheck;

  if (k) {  // Only verify bufData that was read in from a save file
    u32 n;
    for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
      squareLoop(bufData, 0, n);
      modMul(bufData, bufAux);
      bufAux << bufData;
    }

    assert((n & (n - 1)) == 0);
    assert(blockSize % n == 0);

    blockSize /= n;
    assert(blockSize >= 2);

    for (u32 i = 0; i < blockSize - 2; ++i) {
      squareLoop(bufData, 0, n);
      modMul(bufData, bufAux);
    }

    squareLoop(bufData, 0, n);
  }
  modMul(bufData, bufAux, true);
}

bool Gpu::doCheck(u32 blockSize) {
  squareLoop(bufAux, bufCheck, 0, blockSize, true);
  modMul(bufCheck, bufData);
  return isEqual(bufCheck, bufAux);
}

void Gpu::logTimeKernels() {
  auto prof = profile.get();
  u64 total = 0;
  for (const TimeInfo* p : prof) { total += p->times[2]; }
  if (!total) { return; } // no profile
  
  char buf[256];
  // snprintf(buf, sizeof(buf), "Profile:\n ");

  string s = "Profile:\n";
  for (const TimeInfo* p : prof) {
    u32 const n = p->n;
    assert(n);
    double const f = 1e-3 / n;
    double const percent = 100.0 / total * p->times[2];
    if (!args.verbose && percent < 0.2) { break; }
    snprintf(buf, sizeof(buf),
             args.verbose ? "%s %5.2f%% %-11s : %6.0f us/call x %5d calls  (%.3f %.0f)\n"
                          : "%s %5.2f%% %-11s %4.0f x%6d  %.3f %.0f\n",
             logContext().c_str(),
             percent, p->name.c_str(), p->times[2] * f, n, p->times[0] * (f * 1e-3), p->times[1] * (f * 1e-3));
    s += buf;
  }
  log("%s", s.c_str());
  // log("Total time %.3fs\n", total * 1e-9);
  profile.reset();
}

vector<Word> Gpu::readWords(Buffer<Word> &buf) {
  // GPU is returning either 4-byte or 8-byte integers.  C++ code is expecting 8-byte integers.  Handle the "no conversion" case.
  if (fft.WordSize == 8) return buf.read();
  // Convert 32-bit GPU Words into 64-bit C++ Words
  vector<Word> GPUdata = buf.read();
  vector<Word> CPUdata;
  CPUdata.resize(GPUdata.size() * 2);
  for (u32 i = 0; i < GPUdata.size(); ++i) {
    CPUdata[2*i] = (i32) GPUdata[i];
    CPUdata[2*i+1] = (GPUdata[i] >> 32);
  }
  return CPUdata;
}

void Gpu::writeWords(Buffer<Word>& buf, vector<Word> &words) {
  // GPU is expecting either 4-byte or 8-byte integers.  C++ code is using 8-byte integers.  Handle the "no conversion" case.
  if (fft.WordSize == 8) buf.write(words);
  // Convert 64-bit C++ Words into 32-bit GPU Words
  else {
    vector<Word> GPUdata;
    GPUdata.resize(words.size() / 2);
    assert((words.size() & 1) == 0);
    for (u32 i = 0; i < words.size(); i += 2) {
      GPUdata[i/2] = ((i64) words[i+1] << 32) | (u32) words[i];
    }
    buf.write(GPUdata);
  }
}

vector<Word> Gpu::readOut(Buffer<Word> &buf) {
  transpOut(bufAux, buf);
  return readWords(bufAux);
}

void Gpu::writeIn(Buffer<Word>& buf, const vector<u32>& words) { writeIn(buf, expandBits(words, N, E)); }

void Gpu::writeIn(Buffer<Word>& buf, vector<Word>&& words) {
  writeWords(bufAux, words);
  transpIn(buf, bufAux);
}

Words Gpu::expExp2(const Words& A, u32 n) {
  u32 const logStep   = 10000;
  u32 const blockSize = 100;
  
  writeIn(bufData, A);
  IterationTimer timer{0};
  u32 k = 0;
  while (k < n) {
    u32 const its = std::min(blockSize, n - k);
    squareLoop(bufData, 0, its);
    k += its;
    queue.finish();
    if (k % logStep == 0) {
      float const secsPerIt = timer.reset(k);
      log("%u / %u, %s us/it\n", k, n, formatSecsPerIter(secsPerIt).c_str());
    }
  }
  return readData();
}

// A:= A^h * B
void Gpu::expMul(Buffer<Word>& A, u64 h, Buffer<Word>& B) {
  exponentiate(A, h);
  modMul(A, B);
}

// return A^x * B
Words Gpu::expMul(const Words& A, u64 h, const Words& B, bool doSquareB) {
  writeIn(bufCheck, B);
  if (doSquareB) { square(bufCheck); }

  writeIn(bufData, A);
  expMul(bufData, h, bufCheck);
  return readData();
}

static bool testBit(u64 x, int bit) { return x & (u64(1) << bit); }

// See "left-to-right binary exponentiation" on wikipedia
void Gpu::exponentiate(Buffer<Word>& bufInOut, u64 exp) {
  if (exp == 0) {
    bufInOut.set(1);
  } else if (exp > 1) {
    fftP(buf1, bufInOut);
    fftMidIn(buf1);
    fftHin(buf2, buf1); // save fully FFTed "base" to buf2
    bool midInAlreadyDone = true;

    int p = 63;
    while (!testBit(exp, p)) { --p; }

    for (--p; ; --p) {
      if (!midInAlreadyDone) fftMidIn(buf1);
      tailSquare(buf1);
      fftMidOut(buf1);
      midInAlreadyDone = false;

      if (testBit(exp, p)) {
        doCarry(buf1, bufInOut);
        fftMidIn(buf1);
        tailMulLow(buf1, buf2);
        fftMidOut(buf1);
      }

      if (!p) { break; }

      doCarry(buf1, bufInOut);
    }

    fftW(buf3, buf1);
    carryA(bufInOut, buf3);
    carryB(bufInOut);
  }
}

// does either carryFused() or the expanded version depending on useLongCarry
void Gpu::doCarry(Buffer<double>& in, Buffer<Word>& wordBuf) {
  if (useLongCarry) {
    fftW(buf3, in);
    carryA(wordBuf, buf3);
    carryB(wordBuf);
    fftP(in, wordBuf);
  } else {
    carryFused(in);
  }
}

// Use buf1 (and buf23 if not in place) to do a single squaring.
void Gpu::square(Buffer<Word>& out, Buffer<Word>& in, enum LEAD_TYPE leadIn, enum LEAD_TYPE leadOut, bool doMul3, bool doLL) {
  // leadOut = LEAD_MIDDLE is not supported (slower than LEAD_WIDTH)
  assert(leadOut != LEAD_MIDDLE);
  // LL does not do Mul3
  assert(!(doMul3 && doLL));

  // In place FFTs use buf1.  Not in place FFTs also use buf3.
  // If leadIn is LEAD_NONE, in contains the input data, squaring starts at fftP
  // If leadIn is LEAD_WIDTH, buf1 (or buf3 if not in place) contains the input data, squaring starts at fftMidIn
  // If leadIn is LEAD_MIDDLE, buf1 contains the input data, squaring starts at tailSquare
  // If leadOut is LEAD_WIDTH, then buf1 (or buf3 if not in place) will contain the output of carryFused -- to be used as input to the next squaring.
  if (leadIn == LEAD_NONE) fftP(buf1, in);
  if (leadIn != LEAD_MIDDLE) fftMidIn(buf1);
  tailSquare(buf1);
  fftMidOut(buf1);

  // If leadOut is not allowed then we cannot use the faster carryFused kernel
  if (leadOut == LEAD_NONE) {
    fftW(buf3, buf1);
    if (!doLL && !doMul3) {
      carryA(out, buf3);
    } else if (doLL) {
      carryLL(out, buf3);
    } else {
      carryM(out, buf3);
    }
    carryB(out);
  }

  // Use CarryFused
  else {
    assert(!useLongCarry);
    assert(!doMul3);
    if (doLL) {
      carryFusedLL(buf1);
    } else {
      carryFused(buf1);
    }
  }
}

u32 Gpu::squareLoop(Buffer<Word>& out, Buffer<Word>& in, u64 from, u64 to, bool doTailMul3) {
  assert(from < to);
  enum LEAD_TYPE leadIn = LEAD_NONE;
  for (u64 k = from; k < to; ++k) {
    enum LEAD_TYPE const leadOut = useLongCarry || (k == to - 1) ? LEAD_NONE : LEAD_WIDTH;
    square(out, (k==from) ? in : out, leadIn, leadOut, doTailMul3 && (k == to - 1));
    leadIn = leadOut;
  }
  return to;
}

bool Gpu::isEqual(Buffer<Word>& in1, Buffer<Word>& in2) {
  kernIsEqual(in1, in2);
  int isEq = 0;
  bufTrue.read(&isEq, 1);
  if (!isEq) { bufTrue.write({1}); }
  return isEq;
}

u64 Gpu::bufResidue(Buffer<Word> &buf) {
  readResidue(bufSmallOut, buf);
  vector<Word> words = readWords(bufSmallOut);

  int carry = 0;
  for (int i = 0; i < 32; ++i) {
    u32 const len = bitlen(N, E, N - 32 + i);
    i64 const w = (i64) words[i] + carry;
    carry = (int) (w >> len);
  }

  u64 res = 0;
  int hasBits = 0;
  for (int k = 0; k < 32 && hasBits < 64; ++k) {
    u32 const len = bitlen(N, E, k);
    i64 const tmp = (i64) words[32 + k] + carry;
    carry = (int) (tmp >> len);
    u64 const w = tmp - ((i64) carry << len);
    assert(w < (1ULL << len));
    res += w << hasBits;
    hasBits += len;
  }
  return res;
}

static string formatETA(u32 secs) {
  u32 const etaMins = (secs + 30) / 60;
  int const days  = etaMins / (24 * 60);
  int const hours = etaMins / 60 % 24;
  int const mins  = etaMins % 60;
  char buf[64];
  if (days) {
    snprintf(buf, sizeof(buf), "%dd %02d:%02d", days, hours, mins);
  } else {
    snprintf(buf, sizeof(buf), "%02d:%02d", hours, mins);
  }
  return string(buf);  
}

static string getETA(u32 step, u32 total, float secsPerStep) {
  u32 const etaSecs = max(0u, u32((total - step) * secsPerStep));
  return formatETA(etaSecs);
}

string RoeInfo::toString() const {
  if (!N) { return {}; }

  char buf[256];
  snprintf(buf, sizeof(buf), "Z(%u)=%.1f Max %f mean %f sd %f (%f, %f)",
           N, z(.5f), max, mean, sd, gumbelMiu, gumbelBeta);
  return buf;
}

static string makeLogStr(const string& status, u64 k, u64 res, float secsPerIt, u64 nIters) {
  char buf[256];
  
  snprintf(buf, sizeof(buf), "%2s %9" PRIu64 " %016" PRIx64 " %s ETA %s; ",
           status.c_str(), k, res, /* k / float(nIters) * 100, */
           formatSecsPerIter(secsPerIt).c_str(), getETA(k, nIters, secsPerIt).c_str());
  return buf;
}

void Gpu::doBigLog(u64 k, u64 res, bool checkOK, float secsPerIt, u64 nIters, u32 nErrors) {
  auto [roeSq, roeMul] = readROE();
  double const z = roeSq.z();
  zAvg.update(z, roeSq.N);
  if (roeSq.max > 0.005)
    log("%sZ=%.0f (avg %.1f), ROEmax=%.3f, ROEavg=%.3f. %s\n", makeLogStr(checkOK ? "OK" : "EE", k, res, secsPerIt, nIters).c_str(),
        z, zAvg.avg(), roeSq.max, roeSq.mean, (nErrors ? " "s + to_string(nErrors) + " errors"s : ""s).c_str());
  else
    log("%sZ=%.0f (avg %.1f) %s\n", makeLogStr(checkOK ? "OK" : "EE", k, res, secsPerIt, nIters).c_str(),
        z, zAvg.avg(), (nErrors ? " "s + to_string(nErrors) + " errors"s : ""s).c_str());

  if (roeSq.N > 2 && (z < 6 || (fft.shape.fft_type == FFT64 && z < 20))) {
    log("Danger ROE! Z=%.1f is too small, increase precision or FFT size!\n", z);
  }

  // Unless ROE log is not explicitly requested, measure only a few iterations to minimize overhead
  wantROE = args.logROE ? ROE_SIZE : 400;

  RoeInfo const carryStats = readCarryStats();
  if (carryStats.N > 2) {
    u32 const m = ldexp(carryStats.max, 32);
    double const z = carryStats.z();
    log("Carry: %x Z(%u)=%.1f\n", m, carryStats.N, z);
  }
}

bool Gpu::equals9(const Words& a) {
  if (a[0] != 9) { return false; }
  for (auto it = next(a.begin()); it != a.end(); ++it) { if (*it) { return false; }}
  return true;
}

static int ulps(double a, double b) {
  if (a == 0 && b == 0) { return 0; }

  u64 const aa = as<u64>(a);
  u64 const bb = as<u64>(b);
  bool const sameSign = (aa >> 63) == (bb >> 63);
  int const delta = sameSign ? bb - aa : bb + aa;
  return delta;
}

[[maybe_unused]] static double trigNorm(double c, double s) {
  double const c2 = c * c;
  double const err = fma(c, c, -c2);
  double const norm = c2 + fma(s, s, err);
  return norm;
}

void Gpu::selftestTrig() {

#if FFT_FP64
  const u32 n = hN / 8;
  testTrig(buf1);
  vector<double> trig = buf1.read(n * 2);
  int sup = 0, sdown = 0;
  int cup = 0, cdown = 0;
  int oneUp = 0, oneDown = 0;
  for (u32 k = 0; k < n; ++k) {
    double c = trig[2*k];
    double s = trig[2*k + 1];

#if 0
    auto [refCos, refSin] = root1(hN, k);
#else
    long double angle = M_PIl * k / (hN/2);
    double refSin = sinl(angle);
    double refCos = cosl(angle);
#endif

    if (s > refSin) { ++sup; }
    if (s < refSin) { ++sdown; }
    if (c > refCos) { ++cup; }
    if (c < refCos) { ++cdown; }
    
    double norm = trigNorm(c, s);

    if (norm < 1.0) { ++oneDown; }
    if (norm > 1.0) { ++oneUp; }
  }

  log("TRIG sin(): imperfect %d / %d (%.2f%%), balance %d\n",
      sup + sdown, n, (sup + sdown) * 100.0 / n, sup - sdown);
  log("TRIG cos(): imperfect %d / %d (%.2f%%), balance %d\n",
      cup + cdown, n, (cup + cdown) * 100.0 / n, cup - cdown);
  log("TRIG norm: up %d, down %d\n", oneUp, oneDown);
#endif

  if (isAmdGpu(shared.context->deviceId())) {
    vector<string> WHATS {"V_NOP", "V_ADD_I32", "V_FMA_F32", "V_ADD_F64", "V_FMA_F64", "V_MUL_F64", "V_MAD_U64_U32"};
    for (int w = 0; std::cmp_less(w, WHATS.size()); ++w) {
      const int what = w;
      testTime(what, bufCarry);
      vector<i64> const times = bufCarry.read(4096 * 2);
      [[maybe_unused]] i64 const prev = 0;
      u64 min = -1;
      u64 sum = 0;
      for (long const x : times) {
        
#if 0
        if (x != prev) {
          log("%4d : %ld\n", i, x);
          prev = x;
        }
#endif
        if (x > 0 && std::cmp_less(x, min)) { min = x; }
        if (x > 0) { sum += x; }
      }
      log("%-15s : %.2f cycles latency; time min: %d; avg %.0f\n",
          WHATS[w].c_str(), double(min - 40) / 48, int(min), double(sum) / times.size());
    }
  }
}

static u32 mod3(const std::vector<u32> &words) {
  u32 r = 0;
  // uses the fact that 2**32 % 3 == 1.
  for (u32 const w : words) { r += w % 3; }
  return r % 3;
}

static void doDiv3(u64 E, Words& words) {
  u32 r = (3 - mod3(words)) % 3;
  assert(r < 3);
  int const topBits = E % 32;
  assert(topBits > 0 && topBits < 32);
  {
    u64 const w = (u64(r) << topBits) + words.back();
    words.back() = w / 3;
    r = w % 3;
  }
  for (auto it = words.rbegin() + 1, end = words.rend(); it != end; ++it) {
    u64 const w = (u64(r) << 32) + *it;
    *it = w / 3;
    r = w % 3;
  }
}

void Gpu::doDiv9(u64 E, Words& words) {
  doDiv3(E, words);
  doDiv3(E, words);
}

fs::path Gpu::saveProof(const Args& args, ProofSet& proofSet) {
  bool problem_proof = false;
  for ( ; ; ) {
    for (int retry = 0; retry == 0 || (retry == 1 && !problem_proof); ++retry) {
      auto [proof, hashes] = proofSet.computeProof(this);
      fs::path const tmpFile = proof.file(args.proofToVerifyDir);
      proof.save(tmpFile);
            
      fs::path proofFile = proof.file(args.proofResultDir);

      bool const ok = Proof::load(tmpFile).verify(this, hashes);
      log("Proof '%s' verification %s\n", tmpFile.string().c_str(), ok ? "OK" : "FAILED");
      if (ok) {
        fancyRename(tmpFile, proofFile);
        log("Proof '%s' generated\n", proofFile.string().c_str());
        return proofFile;
      }
    }
    problem_proof = true;
    proofSet.reducePower();
    if (proofSet.power < 4) break;
  }
  throw "bad proof generation";
}

PRPState Gpu::loadPRP(Saver<PRPState>& saver) {
  for (int nTries = 0; nTries < 2; ++nTries) {
    if (nTries) {
      saver.dropMostRecent();    // Try an earlier savefile
    }

    PRPState state = saver.load();
    writeState(state.k, state.check, state.blockSize);
    u64 const res = dataResidue();

    if (res == state.res64) {
      log("OK %9" PRIu64 " on-load: blockSize %d, %016" PRIx64 "\n", state.k, state.blockSize, res);
      return state;
      // return {loaded.k, loaded.blockSize, loaded.nErrors};
    }

    log("EE %9" PRIu64 " on-load: %016" PRIx64 " vs. %016" PRIx64 "\n", state.k, res, state.res64);

    if (!state.k) { break; }  // We failed on PRP start
  }

  throw "Error on load";
}

u32 Gpu::getProofPower(u64 k) {
  u32 const power = ProofSet::effectivePower(E, args.getProofPow(E), k);

  if (power != args.getProofPow(E)) {
    log("Proof using power %u (vs %u)\n", power, args.getProofPow(E));
  }

  if (!power) {
    log("Proof generation disabled!\n");
  } else {
    log("Proof of power %u requires about %.1fGB of disk space\n", power, ProofSet::diskUsageGB(E, power));
  }
  return power;
}

tuple<bool, RoeInfo> Gpu::measureCarry() {
  u32 blockSize{}, iters{}, warmup{};

  blockSize = 200;
  iters = 2000;
  warmup = 50;

  assert(iters % blockSize == 0);

  u32 k = 0;
  PRPState const state{.exponent=E, .k=0, .blockSize=blockSize, .res64=3, .check=makeWords(E, 1), .nErrors=0};
  writeState(state.k, state.check, state.blockSize);
  {
    u64 const res = dataResidue();
    if (res != state.res64) {
      log("residue expected %016" PRIx64 " found %016" PRIx64 "\n", state.res64, res);
    }
    assert(res == state.res64);
  }

  enum LEAD_TYPE leadIn = LEAD_NONE;
  modMul(bufCheck, bufData, leadIn);
  leadIn = LEAD_MIDDLE;

  enum LEAD_TYPE const leadOut = useLongCarry ? LEAD_NONE : LEAD_WIDTH;
  square(bufData, bufData, leadIn, leadOut);
  leadIn = leadOut;
  ++k;

  while (k < warmup) {
    square(bufData, bufData, leadIn, leadOut);
    leadIn = leadOut;
    ++k;
  }

  readCarryStats(); // ignore the warm-up iterations

  if (Signal::stopRequested()) { throw "stop requested"; }

  while (true) {
    while (k % blockSize < blockSize-1) {
      square(bufData, bufData, leadIn, leadOut);
      leadIn = leadOut;
      ++k;
    }
    square(bufData, bufData, leadIn, LEAD_NONE);
    leadIn = LEAD_NONE;
    ++k;

    if (k >= iters) { break; }

    modMul(bufCheck, bufData, leadIn);
    leadIn = LEAD_MIDDLE;
    if (Signal::stopRequested()) { throw "stop requested"; }
  }

  [[maybe_unused]] u64 const res = dataResidue();
  if (Signal::stopRequested()) { throw "stop requested"; }

  bool const ok = doCheck(blockSize);
  auto stats = readCarryStats();

  // log("%s %016" PRIx64 " %s\n", ok ? "OK" : "EE", res, roe.toString(statsBits).c_str());
  return {ok, stats};
}

tuple<bool, u64, RoeInfo, RoeInfo> Gpu::measureROE(bool  /*quick*/) {
  u32 blockSize{}, iters{}, warmup{};

  {
    blockSize = 200;
    iters = 2000;
    warmup = 50;
  }

  assert(iters % blockSize == 0);

  wantROE = ROE_SIZE; // should be large enough to capture fully this measureROE()

  u32 k = 0;
  PRPState const state{.exponent=E, .k=0, .blockSize=blockSize, .res64=3, .check=makeWords(E, 1), .nErrors=0};
  writeState(state.k, state.check, state.blockSize);
  {
    u64 const res = dataResidue();
    if (res != state.res64) {
      log("residue expected %016" PRIx64 " found %016" PRIx64 "\n", state.res64, res);
    }
    assert(res == state.res64);
  }

  enum LEAD_TYPE leadIn = LEAD_NONE;
  modMul(bufCheck, bufData, leadIn);
  leadIn = LEAD_MIDDLE;

  enum LEAD_TYPE const leadOut = useLongCarry ? LEAD_NONE : LEAD_WIDTH;
  square(bufData, bufData, leadIn, leadOut);
  leadIn = leadOut;
  ++k;

  while (k < warmup) {
    square(bufData, bufData, leadIn, leadOut);
    leadIn = leadOut;
    ++k;
  }

  readROE(); // ignore the warm-up iterations

  if (Signal::stopRequested()) { throw "stop requested"; }

  while (true) {
    while (k % blockSize < blockSize-1) {
      square(bufData, bufData, leadIn, leadOut);
      leadIn = leadOut;
      ++k;
    }
    square(bufData, bufData, leadIn, LEAD_NONE);
    leadIn = LEAD_NONE;
    ++k;

    if (k >= iters) { break; }

    modMul(bufCheck, bufData, leadIn);
    leadIn = LEAD_MIDDLE;
    if (Signal::stopRequested()) { throw "stop requested"; }
  }

  [[maybe_unused]] u64 const res = dataResidue();
  if (Signal::stopRequested()) { throw "stop requested"; }

  bool const ok = doCheck(blockSize);
  auto roes = readROE();

  wantROE = 0;
  // log("%s %016" PRIx64 " %s\n", ok ? "OK" : "EE", res, roe.toString(statsBits).c_str());
  return {ok, res, roes.first, roes.second};
}

double Gpu::timePRP(int quick) {        // Quick varies from 1 (slowest, longest) to 10 (quickest, shortest)
  u32 blockSize{}, iters{}, warmup{};

  if (quick == 10)     iters =   400, blockSize = 200;
  else if (quick == 9) iters =   600, blockSize = 300;
  else if (quick == 8) iters =   900, blockSize = 300;
  else if (quick == 7) iters =  1200, blockSize = 400;
  else if (quick == 6) iters =  1800, blockSize = 600;
  else if (quick == 5) iters =  3000, blockSize = 1000;
  else if (quick == 4) iters =  5000, blockSize = 1000;
  else if (quick == 3) iters =  8000, blockSize = 1000;
  else if (quick == 2) iters = 12000, blockSize = 1000;
  else if (quick == 1) iters = 20000, blockSize = 1000;
  warmup = 20;

  assert(iters % blockSize == 0);

  u32 k = 0;
  PRPState const state{.exponent=E, .k=0, .blockSize=blockSize, .res64=3, .check=makeWords(E, 1), .nErrors=0};
  writeState(state.k, state.check, state.blockSize);
  assert(dataResidue() == state.res64);

  enum LEAD_TYPE leadIn = LEAD_NONE;
  modMul(bufCheck, bufData, leadIn);
  leadIn = LEAD_MIDDLE;

  enum LEAD_TYPE const leadOut = useLongCarry ? LEAD_NONE : LEAD_WIDTH;
  square(bufData, bufData, leadIn, leadOut);
  leadIn = leadOut;
  ++k;

  while (k < warmup) {
    square(bufData, bufData, leadIn, leadOut);
    leadIn = leadOut;
    ++k;
  }
  queue.finish();
  if (Signal::stopRequested()) { throw "stop requested"; }

  Timer t;
  queue.setSquareTime(0);     // Busy wait on nVidia to get the most accurate timings while tuning
  while (true) {
    while (k % blockSize < blockSize-1) {
      square(bufData, bufData, leadIn, leadOut);
      leadIn = leadOut;
      ++k;
    }
    square(bufData, bufData, leadIn, LEAD_NONE);
    leadIn = LEAD_NONE;
    ++k;

    if (k >= iters) { break; }

    modMul(bufCheck, bufData, leadIn);
    leadIn = LEAD_MIDDLE;
    if (Signal::stopRequested()) { throw "stop requested"; }
  }
  queue.finish();
  double secsPerIt = t.reset() / (iters - warmup);

  if (Signal::stopRequested()) { throw "stop requested"; }

  u64 const res = dataResidue();
  bool const ok = doCheck(blockSize);
  if (!ok) {
    log("Error %016" PRIx64 "\n", res);
    secsPerIt = 0.1; // a large value to mark the error
  }
  return secsPerIt * 1e6;
}

PRPResult Gpu::isPrimePRP(const Task& task) {
  assert(E == task.exponent);

  // This timer is used to measure total elapsed time to be written to the savefile.
  Timer elapsedTimer;

  u32 nErrors = 0;
  int nSeqErrors = 0;
  u64 lastFailedRes64 = 0;
  u32 const logStep = args.logStep;

 reload:
  elapsedTimer.reset();
  u32 blockSize{};
  u64 k{};
  double elapsedBefore = 0;

  {
    PRPState const state = loadPRP(*getSaver());
    nErrors = std::max(nErrors, state.nErrors);
    blockSize = state.blockSize;
    k = state.k;
    elapsedBefore = state.elapsed;
  }

  assert(blockSize > 0 && logStep % blockSize == 0);

  u32 const checkStep = checkStepForErrors(blockSize, nErrors);
  assert(checkStep % logStep == 0);

  u32 const power = getProofPower(k);
  
  ProofSet proofSet{E, power};

  bool isPrime = false;

  u64 finalRes64 = 0;
  vector<u32> res2048;

  // We extract the res64 at kEnd.
  // For M=2^E-1, residue "type-3" == 3^(M+1), and residue "type-1" == type-3 / 9,
  // See http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  // For both type-1 and type-3 we need to do E squarings (as M+1==2^E).
  const u64 kEnd = E;
  assert(k < kEnd);

  // We continue beyound kEnd: to the next multiple of blockSize, to do a check there
  u64 const kEndEnd = roundUp(kEnd, blockSize);

  bool skipNextCheckUpdate = false;

  u64 persistK = proofSet.next(k);
  enum LEAD_TYPE leadIn = LEAD_NONE;

  assert(k % blockSize == 0);
  assert(checkStep % blockSize == 0);

  const u64 startK = k;
  IterationTimer iterationTimer{k};

  wantROE = 0; // skip the initial iterations

  while (true) {
    assert(k < kEndEnd);

    if (!wantROE && k - startK > 30) { wantROE = args.logROE ? ROE_SIZE : 2'000; }

    if (skipNextCheckUpdate) {
      skipNextCheckUpdate = false;
    } else if (k % blockSize == 0) {
      modMul(bufCheck, bufData, leadIn);
      leadIn = LEAD_MIDDLE;
    }

    ++k; // !! early inc

    bool const doStop = (k % blockSize == 0) && (Signal::stopRequested() || (args.iters && k - startK >= args.iters));
    bool const doCheck = doStop || (k % checkStep == 0) || (k >= kEndEnd) || (k - startK == 2 * blockSize);
    bool const doLog = k % logStep == 0;
    enum LEAD_TYPE const leadOut = doCheck || doLog || k == persistK || k == kEnd || useLongCarry ? LEAD_NONE : LEAD_WIDTH;

    if (doStop) { log("Stopping, please wait..\n"); }

    square(bufData, bufData, leadIn, leadOut, false);
    leadIn = leadOut;

    if (k == persistK) {
      vector<Word> const rawData = readChecked(bufData);
      if (rawData.empty()) {
        log("Data error ZERO\n");
        ++nErrors;
        goto reload;
      }
      (*background)([=, E=this->E] { ProofSet::save(E, power, k, compactBits(rawData, E)); });
      persistK = proofSet.next(k);
    }

    if (k == kEnd) {
      Words words = readData();
      isPrime = equals9(words);
      doDiv9(E, words);
      finalRes64 = residue(words);
      res2048.clear();
      assert(words.size() >= 64);
      res2048.insert(res2048.end(), words.begin(), std::next(words.begin(), 64));
      log("%s %8" PRIu64 " / %" PRIu64 ", %s\n", isPrime ? "PP" : "CC", kEnd, E, hex(finalRes64).c_str());
    }

    if (!doCheck && !doLog) continue;

    u64 const res = dataResidue();
    float const secsPerIt = iterationTimer.reset(k);
    queue.setSquareTime((int) (secsPerIt * 1'000'000));

    vector<Word> rawCheck = readChecked(bufCheck);
    if (rawCheck.empty()) {
      ++nErrors;
      log("%9" PRIu64 " %016" PRIx64 " read NULL check\n", k, res);
      if (++nSeqErrors > 2) { throw "sequential errors"; }
      goto reload;
    }

    if (!doCheck) {
      (*background)([=, this] {
        getSaver()->saveUnverified({.exponent=E, .k=k, .blockSize=blockSize, .res64=res, .check=compactBits(rawCheck, E), .nErrors=nErrors,
                                    .elapsed=elapsedBefore + elapsedTimer.at()});
      });

      log("   %9" PRIu64 " %016" PRIx64 " %s\n", k, res, formatSecsPerIter(secsPerIt).c_str());
      RoeInfo const carryStats = readCarryStats();
      if (carryStats.N) {
        u32 const m = ldexp(carryStats.max, 32);
        double const z = carryStats.z();
        log("Carry: %x Z(%u)=%.1f\n", m, carryStats.N, z);
      }
    } else {
      bool const ok = this->doCheck(blockSize);
      [[maybe_unused]] float const secsCheck = iterationTimer.reset(k);

      if (ok) {
        nSeqErrors = 0;
        // lastFailedRes64 = 0;
        skipNextCheckUpdate = true;

        if (k < kEnd) {
          (*background)([=, this, rawCheck = std::move(rawCheck)] {
            getSaver()->save({.exponent=E, .k=k, .blockSize=blockSize, .res64=res, .check=compactBits(rawCheck, E), .nErrors=nErrors, .elapsed=elapsedBefore + elapsedTimer.at()});
          });
        }

        doBigLog(k, res, ok, secsPerIt, kEndEnd, nErrors);

        if (k >= kEndEnd) {
          fs::path const proofFile = saveProof(args, proofSet);
          return {.isPrime=isPrime, .res64=finalRes64, .nErrors=nErrors, .proofPath=proofFile.string(), .res2048=toHex(res2048)};
        }        
      } else {
        ++nErrors;
        doBigLog(k, res, ok, secsPerIt, kEndEnd, nErrors);
        if (++nSeqErrors > 2) {
          log("%d sequential errors, will stop.\n", nSeqErrors);
          throw "too many errors";
        }
        if (res == lastFailedRes64) {
          log("Consistent error %016" PRIx64 ", will stop.\n", res);
          throw "consistent error";
        }
        lastFailedRes64 = res;
        if (!doStop) { goto reload; }
      }

      logTimeKernels();

      if (doStop) {
        queue.finish();
        throw "stop requested";
      }

      iterationTimer.reset(k);
    }
  }
}

LLResult Gpu::isPrimeLL(const Task& task) {
  assert(E == task.exponent);
  wantROE = 0;

  Timer elapsedTimer;

  Saver<LLState> saver{E, 1000, args.nSavefiles};

  reload:
  elapsedTimer.reset();

  u64 startK = 0;
  double elapsedBefore = 0;
  {
    LLState state = saver.load();

    elapsedBefore = state.elapsed;
    startK = state.k;
    u64 const expectedRes = (u64(state.data[1]) << 32) | state.data[0];
    writeIn(bufData, state.data);
    u64 const res = dataResidue();
    if (res != expectedRes) { throw "Invalid savefile (res64)"; }
    assert(res == expectedRes);
    log("LL loaded @ %" PRIu64 " : %016" PRIx64 "\n", startK, res);
  }

  IterationTimer iterationTimer{startK};

  u64 k = startK;
  u64 const kEnd = E - 2;
  enum LEAD_TYPE leadIn = LEAD_NONE;

  while (true) {
    ++k;
    bool doStop = (k >= kEnd) || (args.iters && k - startK >= args.iters);

    if (Signal::stopRequested()) {
      doStop = true;
      log("Stopping, please wait..\n");
    }

    bool const doLog = (k % args.logStep == 0) || doStop;
    enum LEAD_TYPE const leadOut = doLog || useLongCarry ? LEAD_NONE : LEAD_WIDTH;

    squareLL(bufData, leadIn, leadOut);
    leadIn = leadOut;

    if (!doLog) continue;

    u64 res64 = 0;
    auto data = readData();
    bool const isAllZero = data.empty();

    if (isAllZero) {
      if (k < kEnd) {
        log("Error: early ZERO @ %" PRIu64 "\n", k);
        if (doStop) {
          throw "stop requested";
        } 
          goto reload;
       
      }
      res64 = 0;
    } else {
      assert(data.size() >= 2);
      res64 = (u64(data[1]) << 32) | data[0];
      saver.save({.exponent=E, .k=k, .data=std::move(data), .elapsed=elapsedBefore + elapsedTimer.at()});
    }

    float const secsPerIt = iterationTimer.reset(k);
    queue.setSquareTime((int) (secsPerIt * 1'000'000));
    log("%9" PRIu64 " %016" PRIx64 " %s ETA %s\n", k, res64, formatSecsPerIter(secsPerIt).c_str(), getETA(k, kEnd, secsPerIt).c_str());

    if (k >= kEnd) { return {.isPrime=isAllZero, .res64=res64}; }

    if (doStop) { throw "stop requested"; }
  }
}

array<u64, 4> Gpu::isCERT(const Task& task) {
  assert(E == task.exponent);
  wantROE = 0;

  // Get CERT start value
  char fname[32];
  sprintf(fname, "M%" PRIu64 ".cert", E);

// AutoPrimenet.py does not add the cert entry to worktodo.txt until it has successfully downloaded the .cert file.

  { // Enclosing this code in braces ensures the file will be closed by the File destructor.  The later file deletion requires the file be closed in Windows.
    File fi = File::openReadThrow(fname);
    u32 const nBytes = u32((E - 1) / 8 + 1);
    Words const B = fi.readBytesLE(nBytes);
    writeIn(bufData, B);
  }

  Timer elapsedTimer;

  elapsedTimer.reset();

  u32 const startK = 0;

  IterationTimer iterationTimer{startK};

  u32 k = 0;
  u32 const kEnd = task.squarings;
  enum LEAD_TYPE leadIn = LEAD_NONE;

  while (true) {
    ++k;
    bool doStop = (k >= kEnd);

    if (Signal::stopRequested()) {
      doStop = true;
      log("Stopping, please wait..\n");
    }

    bool const doLog = (k % 100'000 == 0) || doStop;
    enum LEAD_TYPE const leadOut = doLog || useLongCarry ? LEAD_NONE : LEAD_WIDTH;

    squareCERT(bufData, leadIn, leadOut);
    leadIn = leadOut;

    if (!doLog) continue;

    Words data = readData();
    assert(data.size() >= 2);
    u64 const res64 = (u64(data[1]) << 32) | data[0];

    float const secsPerIt = iterationTimer.reset(k);
    queue.setSquareTime((int) (secsPerIt * 1'000'000));
    log("%7u / %7u %016" PRIx64 " %s ETA %s\n", k, kEnd, res64, formatSecsPerIter(secsPerIt).c_str(), getETA(k, kEnd, secsPerIt).c_str());

    if (k >= kEnd) {
      fs::remove (fname);
      return std::move(SHA3{}.update(data.data(), (E-1)/8+1)).finish();
    }

    if (doStop) { throw "stop requested"; }
  }
}


void Gpu::clear(bool isPRP) {
  if (isPRP) {
    Saver<PRPState>::clear(E);
  } else {
    Saver<LLState>::clear(E);
  }
}

Saver<PRPState> *Gpu::getSaver() {
  if (!saver) { saver = make_unique<Saver<PRPState>>(E, args.blockSize, args.nSavefiles); }
  return saver.get();
}
