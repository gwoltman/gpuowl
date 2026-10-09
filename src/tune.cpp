// Copyright (C) Mihai Preda

#include "tune.h"
#include "Args.h"
#include "FFTConfig.h"
#include "Gpu.h"
#include "GpuCommon.h"
#include "Primes.h"
#include "log.h"
#include "File.h"
#include "TuneEntry.h"

#include <limits>
#include <map>
#include <set>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cmath>


using namespace std;

vector<string> split(const string& s, char delim) {
  vector<string> ret;
  size_t start = 0;
  while (true) {
    size_t const p = s.find(delim, start);
    if (p == string::npos) {
      ret.push_back(s.substr(start));
      break;
    } 
      ret.push_back(s.substr(start, p - start));
   
    start = p + 1;
  }
  return ret;
}

namespace {

vector<TuneConfig> permute(const vector<pair<string, vector<string>>>& params) {
  vector<TuneConfig> configs;

  int const n = int(params.size());
  vector<int> vpos(n);
  while (true) {
    TuneConfig config;
    for (int i = 0; i < n; ++i) {
      config.emplace_back(params[i].first, params[i].second[vpos[i]]);
    }
    configs.push_back(config);

    int i;
    for (i = n-1; i >= 0; --i) {
      if (vpos[i] < int(params[i].second.size()) - 1) {
        ++vpos[i];
        break;
      } 
        vpos[i] = 0;
     
    }

    if (i < 0) { return configs; }
  }
}

vector<TuneConfig> getTuneConfigs(const string& tune) {
  vector<pair<string, vector<string>>> params;
  for (auto& part : split(tune, ';')) {
    auto keyVal = split(part, '=');
    assert(keyVal.size() == 2);
    string const& key = keyVal.front();
    const string& val = keyVal.back();
    params.emplace_back(key, split(val, ','));
  }
  return permute(params);
}

string toString(const TuneConfig& config) {
  string s{};
  for (const auto& [k, v] : config) { s += k + '=' + v + ','; }
  s.pop_back();
  return s;
}

struct Entry {
  FFTShape shape;
  TuneConfig config;
  double cost;
};

string formatEntry(const Entry& e) {
  char buf[256];
  snprintf(buf, sizeof(buf), "! %s %s # %.0f\n",
           e.shape.spec().c_str(), toString(e.config).c_str(), e.cost);
  return buf;
}

string formatConfigResults(const vector<Entry>& results) {
  string s;
  for (const Entry& e : results) { if (e.shape.width) { s += formatEntry(e); } }
  return s;
}

// Time one tune candidate.  A candidate the GPU can't run (a kernel that fails to compile or link, an
// out-of-memory or out-of-resources error) is logged and costs infinity, so it never wins and the tune
// moves on to the next candidate instead of aborting.  Deliberate stops ("stop requested") still propagate.
// usedWmul, if given, receives the WMUL the candidate actually ran with (clDefines lowers one the FFT or device cannot use).
// fftUses are the FFT's tune.txt settings (see Gpu::make).
double timeConfig(u64 exponent, GpuCommon shared, FFTConfig fft, const vector<KeyVal>& config, int quick = 7, u32* usedWmul = nullptr,
                  const vector<KeyVal>& fftUses = {}) {
  try {
    auto gpu = Gpu::make(exponent, shared, fft, config, false, fftUses);
    if (usedWmul) { *usedWmul = gpu->effectiveWmul(); }
    return gpu->timePRP(quick);
  } catch (const std::exception& e) {
    log("%s failed: %s\n", fft.spec().c_str(), e.what());
  } catch (const string& mes) {
    log("%s failed: %s\n", fft.spec().c_str(), mes.c_str());
  }
  return numeric_limits<double>::infinity();
}

} // namespace

float Tune::maxBpw(FFTConfig fft) {

//  float bpw = oldBpw;

  const float TARGET = 28;
  const u32 sample_size = 5;

  // Estimate how much bpw needs to change to increase/decrease Z by 1.
  // This doesn't need to be a very accurate estimate.
  // This estimate comes from analyzing a 4M FFT and a 7.5M FFT.
  // The 4M FFT needed a .015 step, the 7.5M FFT needed a .012 step.
  float bpw_step = float(.015 + (log2(fft.size()) - log2(4.0*1024*1024)) / (log2(7.5*1024*1024) - log2(4.0*1024*1024)) * (.012 - .015));

  // Pick a bpw that might be close to Z=34, it is best to err on the high side of Z=34
  float bpw1 = fft.maxBpw() - 9 * bpw_step;                                      // Old bpw gave Z=28, we want Z=34 (or more)

// The code below was used when building the maxBpw table from scratch
//  u32   non_best_width = N_VARIANT_W - 1 - variant_W(fft.variant);              // Number of notches below best-Z width variant
//  u32   non_best_middle = N_VARIANT_M - 1 - variant_M(fft.variant);             // Number of notches below best-Z middle variant
//  float bpw1 = 18.3 - 0.275 * (log2(fft.size()) - log2(256 * 13 * 1024 * 2)) - // Default max bpw from an old gpuowl version
//              9 * bpw_step -                                                    // Default above should give Z=28, we want Z=34 (or more)
//                (.08/.012 * bpw_step) * non_best_width -                        // 7.5M FFT has ~.08 bpw difference for each width variant below best variant
//                (.06 + .04 * (fft.shape.middle - 4) / 11) * non_best_middle;    // Assume .1 bpw difference MIDDLE=15 and .06 for MIDDLE=4
//Above fails for FFTs below 512K.  Perhaps we should ditch the above and read from the existing fftbpw.h data to get our starting guess.
//if (fft.size() < 512000) bpw1 = 19, bpw_step = .02;

  // Fine tune our estimate for Z=34
  float z1 = zForBpw(bpw1, fft, 1);
printf ("Guess bpw for %s is %.2f first Z34 is %.2f\n", fft.spec().c_str(), bpw1, z1);
  while (z1 < 31.0f || z1 > 37.0f) {
    float const prev_bpw1 = bpw1;
    float const prev_z1 = z1;
    bpw1 = bpw1 + (z1 - 34.0f) * bpw_step;
    z1 = zForBpw(bpw1, fft, 1);
printf ("Reguess bpw for %s is %.2f first Z34 is %.2f\n", fft.spec().c_str(), bpw1, z1);
    bpw_step = - (bpw1 - prev_bpw1) / (z1 - prev_z1);
    bpw_step = std::max(bpw_step, 0.005f);
    bpw_step = std::min(bpw_step, 0.025f);
  }

  // Get more samples for this bpw -- average in the sample we already have
  z1 = (z1 + (sample_size - 1) * zForBpw(bpw1, fft, sample_size - 1)) / sample_size;

  // Pick a bpw somewhere near Z=22 then fine tune the guess
  float bpw2 = bpw1 + (z1 - 22.0f) * bpw_step;
  float z2 = zForBpw(bpw2, fft, 1);
printf ("Guess bpw for %s is %.2f first Z22 is %.2f\n", fft.spec().c_str(), bpw2, z2);
  while (z2 < 20.0f || z2 > 25.0f) {
    float const prev_bpw2 = bpw2;
    float const prev_z2 = z2;
//    bool error_recovery = (z2 <= 0.0);
//    if (error_recovery) bpw2 -= bpw_step; else
    bpw2 = bpw2 + (z2 - 21.0f) * bpw_step;
    z2 = zForBpw(bpw2, fft, 1);
printf ("Reguess bpw for %s is %.2f first Z22 is %.2f\n", fft.spec().c_str(), bpw2, z2);
//  if (error_recovery) { if (z2 >= 20.0) break; else continue; }
    bpw_step = - (bpw2 - prev_bpw2) / (z2 - prev_z2);
    bpw_step = std::max(bpw_step, 0.005f);
    bpw_step = std::min(bpw_step, 0.025f);
  }

  // Get more samples for this bpw -- average in the sample we already have
  z2 = (z2 + (sample_size - 1) * zForBpw(bpw2, fft, sample_size - 1)) / sample_size;

  // Interpolate for the TARGET Z value
  return bpw2 + (bpw1 - bpw2) * (TARGET - z2) / (z1 - z2);
}

float Tune::zForBpw(float bpw, FFTConfig fft, u32 count) {
  u64 exponent = (count == 1) ? primes.prevPrime(u64(fft.size() * bpw)) : primes.nextPrime(u64(fft.size() * bpw));
  float total_z = 0.0f;
  for (u32 i = 0; i < count; i++, exponent = primes.nextPrime (exponent + 1)) {
    auto [ok, res, roeSq, roeMul] = Gpu::make(exponent, shared, fft, {}, false)->measureROE(true);
    float const z = float(roeSq.z());
    total_z += z;
log("Zforbpw %.2f (z %.2f) : %s\n", bpw, z, fft.spec().c_str());
    if (!ok) { log("Error at bpw %.2f (z %.2f) : %s\n", bpw, z, fft.spec().c_str()); continue; }
  }
//printf ("Out zForBpw %s %.2f avg %.2f\n", fft.spec().c_str(), bpw, total_z / count);
  return total_z / count;
}

void Tune::ztune() {
  File const ztune = File::openAppend("ztune.txt");
  ztune.printf("\n// %s\n\n", shortTimeStr().c_str());

  // Study a specific shape and variant
  if (false) {
    FFTShape const shape = FFTShape(FFT64, 512, 15, 512);
    u32 const variant = 202;
    u32 const sample_size = 5;
    FFTConfig const fft{shape, variant, CARRY_AUTO};
    for (float bpw = 18.18f; bpw < 18.305f; bpw += 0.02f) {
      float const z = zForBpw(bpw, fft, sample_size);
      log ("Avg zForBpw %s %.2f %.2f\n", fft.spec().c_str(), bpw, z);
    }
  }

  // Generate a decent-sized sample that correlates bpw and Z in a range that is close to the target Z value of 28.
  // For no particularly good reason, I strive to find the bpw for Z values near 35 and 21.
  // Over this narrow Z range, linear curve fit should work well.  The Z data is noisy, so more samples is better.

  auto configs = FFTShape::multiSpec(shared.args->fftSpec);
  for (FFTShape const shape : configs) {

    // 4K widths store data on variants 100, 101, 202, 110, 111, 212
    u32 bpw_variants[NUM_BPW_ENTRIES] = {000, 101, 202, 10, 111, 212};
    if (shape.width > 1024) bpw_variants[0] = 100, bpw_variants[3] = 110;

    // Copy the existing bpw array (in case we're replacing only some of the entries)
    array<float, NUM_BPW_ENTRIES> bpw;
    bpw = shape.bpw;

    // Not all shapes have their maximum bpw per-computed.  But one can work on a non-favored shape by specifying it on the command line.
    if (configs.size() > 1) {
      if (!shape.isFavoredShape()) { log ("Skipping %s\n", shape.spec().c_str()); continue; }
    }

    // Test specific variants needed for the maximum bpw table in fftbpw.h
    for (u32 j = 0; j < NUM_BPW_ENTRIES; ++j) {
      FFTConfig const fft{shape, bpw_variants[j], CARRY_AUTO};
      bpw[j] = maxBpw(fft);
    }
    string const s = "\""s + shape.spec() + "\"";
//    ztune.printf("{%12s, {%.3f, %.3f, %.3f, %.3f, %.3f, %.3f}},\n", s.c_str(), bpw[0], bpw[1], bpw[2], bpw[3], bpw[4], bpw[5]);
    ztune.printf("{%12s, {", s.c_str());
    for (u32 j = 0; j < NUM_BPW_ENTRIES; ++j) ztune.printf("%s%.3f", j ? ", " : "", bpw[j]);
    ztune.printf("}},\n");
  }
}

void Tune::carryTune() {
  File const fo = File::openAppend("carrytune.txt");
  fo.printf("\n// %s\n\n", shortTimeStr().c_str());
  u32 prevSize = 0;
  for (FFTShape const shape : FFTShape::multiSpec(shared.args->fftSpec)) {
    FFTConfig const fft{shape, LAST_VARIANT, CARRY_AUTO};
    if (prevSize == fft.size()) { continue; }
    prevSize = fft.size();

    // Measure the plain carryFused (STATS bit 0) and its MUL3 version (STATS bit 1), each at its own 32-bit carry limit
    for (bool const mul3 : {false, true}) {
      shared.args->flags["STATS"] = mul3 ? "2" : "1";
      vector<float> zv;
      double m = 0;
      const float mid = fft.shape.narrowCarryBPW(mul3);
      for (float const bpw : {mid - 0.05f, mid + 0.05f}) {
        u64 const exponent = primes.nearestPrime(u64(fft.size() * bpw));
        auto [ok, carry] = Gpu::make(exponent, shared, fft, {}, false)->measureCarry(mul3);
        m = carry.max;
        if (!ok) { log("Error %s at %f\n", fft.spec().c_str(), bpw); }
        zv.push_back(float(carry.z()));
      }

      float const avg = (zv[0] + zv[1]) / 2;
      u64 const exponent = u64(mid * fft.size());
      double const pErr100 = -expm1(-exp(-avg) * exponent * 100);
      log("%14s%s %.3f : %.3f (%.3f %.3f) %f %.0f%%\n", fft.spec().c_str(), mul3 ? " MUL3" : "", mid, avg, zv[0], zv[1], m, pErr100 * 100);
      fo.printf("%f %f%s\n", log2(fft.size()), avg, mul3 ? " MUL3" : "");
    }
  }
}

template<typename T>
static void add(vector<T>& a, const vector<T>& b) {
  a.insert(a.end(), b.begin(), b.end());
}

void Tune::ctune() {
  Args  const*args = shared.args;

  vector<string> ctune = args->ctune;
  if (ctune.empty()) { ctune.emplace_back("IN_WG=256,128,64;IN_SIZEX=32,16,8;OUT_WG=256,128,64;OUT_SIZEX=32,16,8"); }

  vector<vector<TuneConfig>> configsVect;
  configsVect.reserve(ctune.size());
for (const string& s : ctune) {
    configsVect.push_back(getTuneConfigs(s));
  }

  vector<Entry> results;

  auto shapes = FFTShape::multiSpec(args->fftSpec);
  {
    string str;
    for (const auto& s : shapes) { str += s.spec() + ','; }
    if (!str.empty()) { str.pop_back(); }
    log("FFTs: %s\n", str.c_str());
  }

  for (FFTShape const shape : shapes) {
    FFTConfig const fft{shape, 101, CARRY_32};
    u64 const exponent = primes.prevPrime(fft.maxExp());
    // log("tuning %10s with exponent %" PRIu64 "\n", fft.shape.spec().c_str(), exponent);

    vector<int> bestPos(configsVect.size());
    Entry best{.shape={}, .config={}, .cost=1e9};

    for (u32 i = 0; i < configsVect.size(); ++i) {
      for (u32 pos = i ? 1 : 0; pos < configsVect[i].size(); ++pos) {
        vector<KeyVal> c;

        for (u32 k = 0; k < i; ++k) {
          add(c, configsVect[k][bestPos[k]]);
        }
        add(c, configsVect[i][pos]);
        for (u32 k = i + 1; k < configsVect.size(); ++k) {
          add(c, configsVect[k][bestPos[k]]);
        }
        auto cost = timeConfig(exponent, shared, fft, c);

        bool const isBest = (cost < best.cost);
        if (isBest) {
          bestPos[i] = pos;
          best = {.shape=shape, .config=c, .cost=cost};
        }
        log("%c %6.0f : %s %s\n",
            isBest ? '*' : ' ', cost, shape.spec().c_str(), toString(c).c_str());
      }
    }
    results.push_back(best);
    log("%s", formatEntry(best).c_str());
  }
  log("\nBest configs (lines can be copied to config.txt):\n%s", formatConfigResults(results).c_str());
}

// Add better -use settings to list of changes to be made to config.txt
static void configsUpdate(double current_cost, double best_cost, double threshold, const char *key, u32 value, vector<pair<string,int>> &newConfigKeyVals, vector<pair<string,int>> &suggestedConfigKeyVals) {
  if (best_cost == current_cost) return;
  if (!std::isfinite(best_cost)) return;     // every setting failed its check (Gpu::timePRP returned infinity)
  // If best cost is better than current cost by a substantial margin (the threshold) then add the key value pair to suggestedConfigKeyVals
  if (best_cost < (1.0 - threshold) * current_cost)
    newConfigKeyVals.emplace_back(key, value);
  // Otherwise, add the key value pair to newConfigKeyVals
  else
    suggestedConfigKeyVals.emplace_back(key, value);
}

void Tune::tune() {
  Args *args = shared.args;
  vector<FFTShape> shapes = FFTShape::multiSpec(args->fftSpec);

  // There are some options and variants that are different based on GPU manufacturer
  bool const AMDGPU = isAmdGpu(shared.context->deviceId());
  bool const NVIDIAGPU = isNvidiaGpu(shared.context->deviceId());
  int const NO_ASM = args->value("NO_ASM", 0);
  // Variant zero (BCAST) needs either an AMD GPU whose OpenCL compiler has the amdgcn builtins, or an nVidia GPU
  // new enough for shfl.sync (sm_30+; Gpu::make otherwise runs it as variant one).  Have NO_ASM bypass variant zero.
  bool const VARIANT0 = !NO_ASM
    && ((AMDGPU && hasAmdBcastBuiltins(shared.context->get(), shared.context->deviceId()))
        || (NVIDIAGPU && getNvidiaComputeCapability(shared.context->deviceId()) >= 300));

  bool tune_config = true;
  bool regs_only = false;
  bool variants_only = false;
  double fine_pct = 0;                          // -tune fine: FFTs within this percent of earning a tune.txt entry get their register limits tuned
  bool time_FFTs = false;
  bool time_NTTs = false;
  u32 time_FP32 = 1;                            // FFTs with an FP32 part, an optional group (fp32=0 is the old nofp32)
  u32 time_FFT6431 = 0;                         // FP64+M31 FFTs, an optional group like the ones below
  // Optional groups of FFTs: 0 = don't time them, 1 = time them, 2 = time only the groups set to 2.  A bare name means 1.
  u32 time_1K_256 = 0;                          // 1K:256 and 256:1K shapes (512:512 is almost always better)
  u32 time_M61 = 0;                             // Type 3, M61-only NTTs
  u32 time_PFA = 1;                             // Hybrid FFT/NTTs (an FP32 or FP64 part) with a non-power-of-two middle.  Timed by default.
  bool time_inplace_only = NVIDIAGPU;           // Default is nVidia is better off with INPLACE=1, AMD GPUs need to time extra options used when INPLACE=0
  int quick = 7;                                // Run config from slowest (quick=1) to fastest (quick=10)
  u64 min_exponent = 75000000;
  u64 max_exponent = 350000000;
  if (!args->fftSpec.empty()) { min_exponent = 0; max_exponent = 1000000000000ull; }
  bool range_given = false;                     // minexp= or maxexp= was given

  // Parse input args
  for (const string& s : split(args->tune, ',')) {
    if (s.empty()) continue;
    if (s == "noconfig") tune_config = false;
    if (s == "regs") regs_only = true;
    if (s == "variants") variants_only = true;
    if (s == "fine") fine_pct = 2;
    if (s == "fp64") time_FFTs = true;
    if (s == "ntt") time_NTTs = true;
    if (s == "fp6431") time_FFT6431 = 1;         // It is rare to have a GPU good at both FP64 and integer ops.  TitanV is one.  Allow tuning FFT6431.
    if (s == "nofp32") time_FP32 = 0;            // Workaround bug in some openCL compilers that cannot compile our FP32 openCL code.  Same as fp32=0.
    if (s == "fp32") time_FP32 = 1;
    if (s == "inplace") time_inplace_only = true;
    if (s == "1k256") time_1K_256 = 1;
    if (s == "m61") time_M61 = 1;
    if (s == "pfa") time_PFA = 1;
    auto keyVal = split(s, '=');
    if (keyVal.size() == 2) {
      if (keyVal.front() == "quick") quick = stoi(keyVal.back());
      if (keyVal.front() == "minexp") { min_exponent = stoull(keyVal.back()); range_given = true; }
      if (keyVal.front() == "maxexp") { max_exponent = stoull(keyVal.back()); range_given = true; }
      if (keyVal.front() == "1k256") time_1K_256 = stoi(keyVal.back());
      if (keyVal.front() == "m61") time_M61 = stoi(keyVal.back());
      if (keyVal.front() == "pfa") time_PFA = stoi(keyVal.back());
      if (keyVal.front() == "fp6431") time_FFT6431 = stoi(keyVal.back());
      if (keyVal.front() == "fp32") time_FP32 = stoi(keyVal.back());
      if (keyVal.front() == "fine") fine_pct = stod(keyVal.back());
    }
  }
  quick = std::max(quick, 1);
  quick = std::min(quick, 10);

  // Only (re)tune the register limits of the existing tune.txt entries -- those of the -fft shapes, if given
  if (regs_only) {
    regTune(quick, args->fftSpec.empty() ? vector<FFTShape>{} : shapes);
    return;
  }


  // -tune fine looks for FFTs that earn a tune.txt entry once their register limits are tuned.  The config.txt settings are left alone.
  if (fine_pct > 0) {
    tune_config = false;
#ifndef CUDA_BACKEND
    log("-tune fine: register limits are only tuned in the CUDA build.  Timing FFTs as -tune does.\n");
    fine_pct = 0;
#endif
  }

  // Giving only one of minexp=/maxexp= leaves the other at its default (75M/350M), so e.g. "-tune maxexp=50000000"
  // alone leaves min_exponent at 75M above it.  The FFT-selection loop below (fft.maxExp() < min_exponent /
  // fft.maxExp() > 1.25*max_exponent) would then silently time nothing useful instead of the small-exponent FFTs the
  // user asked for.  Fail loudly instead of leaving the user staring at an empty tune.txt.
  if (min_exponent > max_exponent) {
    log("-tune: minexp=%" PRIu64 " is greater than maxexp=%" PRIu64 "; give both minexp= and maxexp= to tune a "
        "narrow range, e.g. -tune minexp=10000000,maxexp=20000000 for a small exponent such as PRP-CF at 18M\n",
        min_exponent, max_exponent);
    throw "-tune minexp/maxexp range";
  }

  // Only add the more accurate variants of the existing tune.txt entries -- those of the -fft shapes, if given, and for the exponents
  // between minexp and maxexp if either is given (otherwise all of tune.txt).
  if (variants_only) {
    variantTune(quick, VARIANT0, args->fftSpec.empty() ? vector<FFTShape>{} : shapes,
                range_given ? min_exponent : 0, range_given ? max_exponent : UINT64_MAX / 2);
    return;
  }

  // Devices without FP64 (e.g. Mesa rusticl on AMD) can only run the FFT types that have no FP64 data
  if (!hasFP64(shared.context->deviceId())) {
    log("This device does not support FP64.  Only FFT types without FP64 data will be tuned.\n");
    std::erase_if(shapes, [](const FFTShape& sh) { return FFTConfig{sh, 202, CARRY_AUTO}.FFT_FP64; });
    if (shapes.empty()) { log("No FFT without FP64 in '%s'\n", args->fftSpec.c_str()); throw "No FFT"; }
    time_FFTs = false;
    time_FFT6431 = 0;
    time_NTTs = true;
  }

  // A group set to 2 asks to time only those FFTs, e.g. fp6431=2 to add FFT6431 to a tune.txt made with -tune fp64.  The config.txt
  // settings are left as they are.
  bool const onlyGroups = time_1K_256 == 2 || time_M61 == 2 || time_PFA == 2 || time_FFT6431 == 2 || time_FP32 == 2;
  // A group set to 2 must not be left out by the FFT types timed: the FFTs with an FP32 part are NTT hybrids, and the PFA hybrids
  // are FP32 (NTT) or FP64+M31 FFTs
  if (time_FP32 == 2 || time_PFA == 2) { time_NTTs = true; }
  if (time_PFA == 2 && !time_FFT6431 && hasFP64(shared.context->deviceId())) { time_FFT6431 = 1; }
  if (onlyGroups && tune_config) {
    log("Only timing the FFTs of the groups set to 2.  The config.txt settings are not tuned.\n");
    tune_config = false;
  }

  // Look for best settings of various options.  Append best settings to config.txt.
  if (tune_config) {
    vector<pair<string,int>> newConfigKeyVals;
    vector<pair<string,int>> suggestedConfigKeyVals;

    // Select/init the default FFTshape(s) and FFTConfig(s) for optimal -use options testing
    FFTShape defaultFFTShape, defaultNTTShape, *defaultShape;

    // If user gave us an fft-spec, use that to time options
    if (!args->fftSpec.empty()) {
      defaultShape = shapes.data();
      if (shapes[0].fft_type == FFT64) {
        defaultFFTShape = shapes[0];
        time_FFTs = true;
      } else {
        defaultNTTShape = shapes[0];
        time_NTTs = true;
      }
    }
    // If user specified FP64-timings, time a wavefront exponent using an 7.5M FFT
    // If user specified NTT-timings, time a wavefront exponent using an 4M M31+M61 NTT
    else if (time_FFTs || time_NTTs) {
      if (time_FFTs) {
        defaultFFTShape = FFTShape(FFT64, 512, 15, 512);
        defaultShape = &defaultFFTShape;
      }
      if (time_NTTs) {
        defaultNTTShape = FFTShape(FFT3161, 512, 8, 512);
        defaultShape = &defaultNTTShape;
      }
    }
    // No user specifications.  Time an FP64 FFT and a GF31*GF61 NTT to see if the GPU is more suited for FP64 work or NTT work.
    else {
      log("Checking whether this GPU is better suited for double-precision FFTs or integer NTTs.\n");
      defaultFFTShape = FFTShape(FFT64, 512, 16, 512);
      FFTConfig const fft{defaultFFTShape, 101, CARRY_32};
      double const fp64_time = timeConfig(141000001, shared, fft, {}, quick);
      log("Time for FP64 FFT %12s is %6.1f\n", fft.spec().c_str(), fp64_time);
      defaultNTTShape = FFTShape(FFT3161, 512, 8, 512);
      FFTConfig const ntt{defaultNTTShape, 202, CARRY_AUTO};
      double const ntt_time = timeConfig(141000001, shared, ntt, {}, quick);
      log("Time for M31*M61 NTT %12s is %6.1f\n", ntt.spec().c_str(), ntt_time);
      if (fp64_time < ntt_time) {
        defaultShape = &defaultFFTShape;
        time_FFTs = true;
        if (fp64_time < 0.80 * ntt_time) {
          log("FP64 FFTs are significantly faster than integer NTTs.  No NTT tuning will be performed.\n");
        } else {
          log("FP64 FFTs are not significantly faster than integer NTTs.  NTT tuning will be performed.\n");
          time_NTTs = true;
        }
      } else {
        defaultShape = &defaultNTTShape;
        time_NTTs = true;
        if (fp64_time > 1.20 * ntt_time) {
          log("FP64 FFTs are significantly slower than integer NTTs.  No FP64 tuning will be performed.\n");
        } else {
          log("FP64 FFTs are not significantly slower than integer NTTs.  FP64 tuning will be performed.\n");
          time_FFTs = true;
        }
      }
    }

    log("\n");
    log("Beginning timing of various options.  These settings will be appended to config.txt.\n");
    log("Please read config.txt after -tune completes.\n");
    log("\n");

    u32 const variant = (defaultShape == &defaultFFTShape) ? 101 : 202;
//GW: if fft spec on the command line specifies a variant then we should use that variant (I get some interesting results with 000 vs 101 vs 201 vs 202 likely due to rocm optimizer)

    // IN_WG/SIZEX, OUT_WG/SIZEX, PAD, MIDDLE_IN/OUT_LDS_TRANSPOSE apply only if INPLACE=0
    u32 const current_inplace = args->value("INPLACE", 0);
    args->flags["INPLACE"] = to_string(0);

    // Find best IN_WG,IN_SIZEX,OUT_WG,OUT_SIZEX settings
    if (!time_inplace_only) {
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_in_wg = 0;
      u32 best_in_sizex = 0;
      u32 const current_in_wg = args->value("IN_WG", 128);
      u32 const current_in_sizex = args->value("IN_SIZEX", 16);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const in_wg : {64, 128, 256}) {
        for (u32 const in_sizex : {8, 16, 32}) {
          args->flags["IN_WG"] = to_string(in_wg);
          args->flags["IN_SIZEX"] = to_string(in_sizex);
          double const cost = timeConfig(exponent, shared, fft, {}, quick);
          log("Time for %12s using IN_WG=%u, IN_SIZEX=%u is %6.1f\n", fft.spec().c_str(), in_wg, in_sizex, cost);
          if (in_wg == current_in_wg && in_sizex == current_in_sizex) current_cost = cost;
          if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_in_wg = in_wg; best_in_sizex = in_sizex; }
        }
      }
      log("Best IN_WG, IN_SIZEX is %u, %u.  Default is 128, 16.\n", best_in_wg, best_in_sizex);
      configsUpdate(current_cost, best_cost, 0.003, "IN_WG", best_in_wg, newConfigKeyVals, suggestedConfigKeyVals);
      configsUpdate(current_cost, best_cost, 0.003, "IN_SIZEX", best_in_sizex, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["IN_WG"] = to_string(best_in_wg);
      args->flags["IN_SIZEX"] = to_string(best_in_sizex);

      u32 best_out_wg = 0;
      u32 best_out_sizex = 0;
      u32 const current_out_wg = args->value("OUT_WG", 128);
      u32 const current_out_sizex = args->value("OUT_SIZEX", 16);
      best_cost = -1.0;
      current_cost = -1.0;
      for (u32 const out_wg : {64, 128, 256}) {
        for (u32 const out_sizex : {8, 16, 32}) {
          args->flags["OUT_WG"] = to_string(out_wg);
          args->flags["OUT_SIZEX"] = to_string(out_sizex);
          double const cost = timeConfig(exponent, shared, fft, {}, quick);
          log("Time for %12s using OUT_WG=%u, OUT_SIZEX=%u is %6.1f\n", fft.spec().c_str(), out_wg, out_sizex, cost);
          if (out_wg == current_out_wg && out_sizex == current_out_sizex) current_cost = cost;
          if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_out_wg = out_wg; best_out_sizex = out_sizex; }
        }
      }
      log("Best OUT_WG, OUT_SIZEX is %u, %u.  Default is 128, 16.\n", best_out_wg, best_out_sizex);
      configsUpdate(current_cost, best_cost, 0.003, "OUT_WG", best_out_wg, newConfigKeyVals, suggestedConfigKeyVals);
      configsUpdate(current_cost, best_cost, 0.003, "OUT_SIZEX", best_out_sizex, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["OUT_WG"] = to_string(best_out_wg);
      args->flags["OUT_SIZEX"] = to_string(best_out_sizex);
    }

    // Find best PAD setting.  Default is 256 bytes for AMD, 0 for all others.
    if (!time_inplace_only) {
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_pad = 0;
      u32 const current_pad = args->value("PAD", AMDGPU ? 256 : 0);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const pad : {0, 64, 128, 256, 512}) {
        args->flags["PAD"] = to_string(pad);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using PAD=%u is %6.1f\n", fft.spec().c_str(), pad, cost);
        if (pad == current_pad) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_pad = pad; }
      }
      log("Best PAD is %u bytes.  Default PAD is %u bytes.\n", best_pad, AMDGPU ? 256 : 0);
      configsUpdate(current_cost, best_cost, 0.000, "PAD", best_pad, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["PAD"] = to_string(best_pad);
    }

    // Find best MIDDLE_IN_LDS_TRANSPOSE setting
    if (!time_inplace_only) {
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_middle_in_lds_transpose = 0;
      u32 const current_middle_in_lds_transpose = args->value("MIDDLE_IN_LDS_TRANSPOSE", 1);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const middle_in_lds_transpose : {0, 1}) {
        args->flags["MIDDLE_IN_LDS_TRANSPOSE"] = to_string(middle_in_lds_transpose);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using MIDDLE_IN_LDS_TRANSPOSE=%u is %6.1f\n", fft.spec().c_str(), middle_in_lds_transpose, cost);
        if (middle_in_lds_transpose == current_middle_in_lds_transpose) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_middle_in_lds_transpose = middle_in_lds_transpose; }
      }
      log("Best MIDDLE_IN_LDS_TRANSPOSE is %u.  Default MIDDLE_IN_LDS_TRANSPOSE is 1.\n", best_middle_in_lds_transpose);
      configsUpdate(current_cost, best_cost, 0.000, "MIDDLE_IN_LDS_TRANSPOSE", best_middle_in_lds_transpose, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["MIDDLE_IN_LDS_TRANSPOSE"] = to_string(best_middle_in_lds_transpose);
    }

    // Find best MIDDLE_OUT_LDS_TRANSPOSE setting
    if (!time_inplace_only) {
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_middle_out_lds_transpose = 0;
      u32 const current_middle_out_lds_transpose = args->value("MIDDLE_OUT_LDS_TRANSPOSE", 1);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const middle_out_lds_transpose : {0, 1}) {
        args->flags["MIDDLE_OUT_LDS_TRANSPOSE"] = to_string(middle_out_lds_transpose);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using MIDDLE_OUT_LDS_TRANSPOSE=%u is %6.1f\n", fft.spec().c_str(), middle_out_lds_transpose, cost);
        if (middle_out_lds_transpose == current_middle_out_lds_transpose) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_middle_out_lds_transpose = middle_out_lds_transpose; }
      }
      log("Best MIDDLE_OUT_LDS_TRANSPOSE is %u.  Default MIDDLE_OUT_LDS_TRANSPOSE is 1.\n", best_middle_out_lds_transpose);
      configsUpdate(current_cost, best_cost, 0.000, "MIDDLE_OUT_LDS_TRANSPOSE", best_middle_out_lds_transpose, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["MIDDLE_OUT_LDS_TRANSPOSE"] = to_string(best_middle_out_lds_transpose);
    }

    // If only timing INPLACE=1 options, then set INPLACE
    if (time_inplace_only) {
      args->flags["INPLACE"] = to_string(1);
      newConfigKeyVals.emplace_back("INPLACE", 1);
    }
    // Find best INPLACE setting
    else {
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_inplace = 0;
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const inplace : {0, 1}) {
        args->flags["INPLACE"] = to_string(inplace);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using INPLACE=%u is %6.1f\n", fft.spec().c_str(), inplace, cost);
        if (inplace == current_inplace) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_inplace = inplace; }
      }
      log("Best INPLACE is %u.  Default INPLACE is 0.  Best INPLACE setting may be different for other FFT lengths.\n", best_inplace);
      configsUpdate(current_cost, best_cost, 0.002, "INPLACE", best_inplace, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["INPLACE"] = to_string(best_inplace);
    }

    // Find best LOADS/STORES settings
    if (true) {
      u32 loads = args->value("LOADS", 0);
      u32 stores = args->value("STORES", 0);

      // Find best FFT data LOADS setting
      if (true) {
        FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
        u64 const exponent = primes.prevPrime(fft.maxExp());
        u32 best_fft_load = 0;
        double best_cost = -1.0;
        for (u32 const fft_load : {0, 1, 2, 3, 4}) {
          if (fft_load >= 2 && (!NVIDIAGPU || NO_ASM)) continue;
          args->flags["LOADS"] = to_string(loads / 10 * 10 + fft_load);
          double const cost = timeConfig(exponent, shared, fft, {}, quick);
          log("Time for %12s using FFT load=%u is %6.1f\n", fft.spec().c_str(), fft_load, cost);
          if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_fft_load = fft_load; }
        }
        log("Best FFT load is %u.  Default is 0.\n", best_fft_load);
        loads = loads / 10 * 10 + best_fft_load;
        args->flags["LOADS"] = to_string(loads);
      }

      // Find best FFT data STORES setting
      if (true) {
        FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
        u64 const exponent = primes.prevPrime(fft.maxExp());
        u32 best_fft_store = 0;
        double best_cost = -1.0;
        for (u32 const fft_store : {0, 1, 2, 3}) {
          if (fft_store >= 2 && (!NVIDIAGPU || NO_ASM)) continue;
          args->flags["STORES"] = to_string(stores / 10 * 10 + fft_store);
          double const cost = timeConfig(exponent, shared, fft, {}, quick);
          log("Time for %12s using FFT store=%u is %6.1f\n", fft.spec().c_str(), fft_store, cost);
          if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_fft_store = fft_store; }
        }
        log("Best FFT store is %u.  Default is 0.\n", best_fft_store);
        stores = stores / 10 * 10 + best_fft_store;
        args->flags["STORES"] = to_string(stores);
      }

      // Find best carryShuttle LOADS/STORES settings
      if (true) {
        FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
        u64 const exponent = primes.prevPrime(fft.maxExp());
        u32 best_cs_load = 0, best_cs_store = 0;
        double best_cost = -1.0;
        for (u32 const cs : {0, 1, 2}) {   // Test three combinations:  Default load/store, non-temporal, last-use load with L2 store
          if (cs >= 2 && (!NVIDIAGPU || NO_ASM)) continue;
          u32 const cs_load = cs == 0 ? 0 : cs == 1 ? 1 : 4;
          u32 const cs_store = cs == 0 ? 0 : cs == 1 ? 1 : 2;
          args->flags["LOADS"] = to_string(loads / 100 * 100 + cs_load * 10 + loads % 10);
          args->flags["STORES"] = to_string(stores / 100 * 100 + cs_store * 10 + stores % 10);
          double const cost = timeConfig(exponent, shared, fft, {}, quick);
          log("Time for %12s using carry shuttle load=%u, store=%u is %6.1f\n", fft.spec().c_str(), cs_load, cs_store, cost);
          if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_cs_load = cs_load; best_cs_store = cs_store; }
        }
        log("Best carry shuttle load/store is %u/%u.  Default is 0/0.\n", best_cs_load, best_cs_store);
        loads = loads / 100 * 100 + best_cs_load * 10 + loads % 10;
        stores = stores / 100 * 100 + best_cs_store * 10 + stores % 10;
        args->flags["LOADS" ] = to_string(loads);
        args->flags["STORES"] = to_string(stores);
      }

      // Find best TRIG frequently used data LOADS setting
      if (true) {
        FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
        u64 const exponent = primes.prevPrime(fft.maxExp());
        u32 best_trig_load = 0;
        double best_cost = -1.0;
        for (u32 const trig_load : {0, 5}) {
          if (trig_load >= 2 && (!NVIDIAGPU || NO_ASM)) continue;
          args->flags["LOADS"] = to_string(loads / 1000 * 1000 + trig_load * 100 + loads % 100);
          double const cost = timeConfig(exponent, shared, fft, {}, quick);
          log("Time for %12s using Trig frequently used load=%u is %6.1f\n", fft.spec().c_str(), trig_load, cost);
          if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_trig_load = trig_load; }
        }
        log("Best Trig frequently used load is %u.  Default is 0.\n", best_trig_load);
        loads = loads / 1000 * 1000 + best_trig_load * 100 + loads % 100;
        args->flags["LOADS" ] = to_string(loads);
      }

      // Find best TRIG several uses data LOADS setting
      if (true) {
        FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
        u64 const exponent = primes.prevPrime(fft.maxExp());
        u32 best_trig_load = 0;
        double best_cost = -1.0;
        for (u32 const trig_load : {0, 1, 2, 3, 4, 5}) {
          if (trig_load >= 2 && (!NVIDIAGPU || NO_ASM)) continue;
          args->flags["LOADS"] = to_string(loads / 10000 * 10000 + trig_load * 1000 + loads % 1000);
          double const cost = timeConfig(exponent, shared, fft, {}, quick);
          log("Time for %12s using Trig several uses load=%u is %6.1f\n", fft.spec().c_str(), trig_load, cost);
          if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_trig_load = trig_load; }
        }
        log("Best Trig several uses load is %u.  Default is 0.\n", best_trig_load);
        loads = loads / 10000 * 10000 + best_trig_load * 1000 + loads % 1000;
        args->flags["LOADS" ] = to_string(loads);
      }

      // Find best TRIG used once data LOADS setting
      if (true) {
        FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
        u64 const exponent = primes.prevPrime(fft.maxExp());
        u32 best_trig_load = 0;
        double best_cost = -1.0;
        for (u32 const trig_load : {0, 1, 2, 3, 4, 5}) {
          if (trig_load >= 2 && (!NVIDIAGPU || NO_ASM)) continue;
          args->flags["LOADS"] = to_string(trig_load * 10000 + loads % 10000);
          double const cost = timeConfig(exponent, shared, fft, {}, quick);
          log("Time for %12s using Trig used once load=%u is %6.1f\n", fft.spec().c_str(), trig_load, cost);
          if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_trig_load = trig_load; }
        }
        log("Best Trig used once load is %u.  Default is 0.\n", best_trig_load);
        loads = best_trig_load * 10000 + loads % 10000;
        args->flags["LOADS" ] = to_string(loads);
      }

      // Write accumulated LOADS/STORES settings
      configsUpdate(1.000, 0.000, 0.000, "LOADS", loads, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["LOADS"] = to_string(loads);
      configsUpdate(1.000, 0.000, 0.000, "STORES", stores, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["STORES"] = to_string(stores);
    }

#ifndef CUDA_BACKEND
    // Find best FAST_BARRIER setting.  Skip it where it provably can't do anything: base.cl forces it off
    // whenever the compiled wavefront isn't 64 (RDNA under ROCm's OpenCL compiler always picks 32) or the
    // device is CDNA2/CDNA3 (gfx90a, gfx94x/gfx95x -- wave64 too, but with the same non-waiting barrier as
    // RDNA), so testing FAST_BARRIER=1 there would just repeat the FAST_BARRIER=0 timing.
    if (!amdFastBarrierUnsafe(shared.context->deviceId())) {                 // FAST_BARRIER now works for nVidia GPUs too (from what I've seen)
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_fast_barrier = 0;
      u32 const current_fast_barrier = args->value("FAST_BARRIER", 0);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const fast_barrier : {0, 1}) {
        args->flags["FAST_BARRIER"] = to_string(fast_barrier);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using FAST_BARRIER=%u is %6.1f\n", fft.spec().c_str(), fast_barrier, cost);
        if (fast_barrier == current_fast_barrier) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_fast_barrier = fast_barrier; }
      }
      log("Best FAST_BARRIER is %u.  Default FAST_BARRIER is 0.\n", best_fast_barrier);
      configsUpdate(current_cost, best_cost, 0.000, "FAST_BARRIER", best_fast_barrier, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["FAST_BARRIER"] = to_string(best_fast_barrier);
    }
#endif

    // Find best TAIL_KERNELS setting
    if (true) {
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_tail_kernels = 0;
      u32 const current_tail_kernels = args->value("TAIL_KERNELS", 2);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const tail_kernels : {0, 1, 2, 3}) {
        args->flags["TAIL_KERNELS"] = to_string(tail_kernels);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using TAIL_KERNELS=%u is %6.1f\n", fft.spec().c_str(), tail_kernels, cost);
        if (tail_kernels == current_tail_kernels) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tail_kernels = tail_kernels; }
      }
      if (best_tail_kernels & 1)
        log("Best TAIL_KERNELS is %u.  Default TAIL_KERNELS is 2.\n", best_tail_kernels);
      else
        log("Best TAIL_KERNELS is %u (but best may be %u when running two workers on one GPU).  Default TAIL_KERNELS is 2.\n", best_tail_kernels, best_tail_kernels | 1);
      configsUpdate(current_cost, best_cost, 0.000, "TAIL_KERNELS", best_tail_kernels, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["TAIL_KERNELS"] = to_string(best_tail_kernels);
    }

    // Find best TAIL_TRIGS setting
    if (time_FFTs) {
      FFTConfig const fft{defaultFFTShape, 101, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_tail_trigs = 0;
      u32 const current_tail_trigs = args->value("TAIL_TRIGS", 2);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const tail_trigs : {0, 1, 2}) {
        args->flags["TAIL_TRIGS"] = to_string(tail_trigs);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using TAIL_TRIGS=%u is %6.1f\n", fft.spec().c_str(), tail_trigs, cost);
        if (tail_trigs == current_tail_trigs) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tail_trigs = tail_trigs; }
      }
      log("Best TAIL_TRIGS is %u.  Default TAIL_TRIGS is 2.\n", best_tail_trigs);
      configsUpdate(current_cost, best_cost, 0.003, "TAIL_TRIGS", best_tail_trigs, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["TAIL_TRIGS"] = to_string(best_tail_trigs);
    }

    // Find best TAIL_TRIGS31 setting
    if (time_NTTs) {
      FFTConfig fft{defaultNTTShape, 202, CARRY_AUTO};
      if (!fft.NTT_GF31) fft = FFTConfig(FFTShape(FFT3161, 512, 8, 512), 202, CARRY_AUTO);
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_tail_trigs = 0;
      u32 const current_tail_trigs = args->value("TAIL_TRIGS31", 0);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const tail_trigs : {0, 1}) {
        args->flags["TAIL_TRIGS31"] = to_string(tail_trigs);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using TAIL_TRIGS31=%u is %6.1f\n", fft.spec().c_str(), tail_trigs, cost);
        if (tail_trigs == current_tail_trigs) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tail_trigs = tail_trigs; }
      }
      log("Best TAIL_TRIGS31 is %u.  Default TAIL_TRIGS31 is 0.\n", best_tail_trigs);
      configsUpdate(current_cost, best_cost, 0.003, "TAIL_TRIGS31", best_tail_trigs, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["TAIL_TRIGS31"] = to_string(best_tail_trigs);
    }

    // Find best TAIL_TRIGS32 setting
    if (time_NTTs && time_FP32) {
      FFTConfig fft{defaultNTTShape, 202, CARRY_AUTO};
      if (!fft.FFT_FP32) fft = FFTConfig(FFTShape(FFT3261, 512, 8, 512), 202, CARRY_AUTO);
      u64 const exponent = primes.prevPrime(u64(fft.maxBpw() * 0.95 * fft.shape.size()));   // Back off the maxExp as different settings will have different maxBpw
      u32 best_tail_trigs = 0;
      u32 const current_tail_trigs = args->value("TAIL_TRIGS32", 2);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const tail_trigs : {0, 1, 2}) {
        args->flags["TAIL_TRIGS32"] = to_string(tail_trigs);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using TAIL_TRIGS32=%u is %6.1f\n", fft.spec().c_str(), tail_trigs, cost);
        if (tail_trigs == current_tail_trigs) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tail_trigs = tail_trigs; }
      }
      log("Best TAIL_TRIGS32 is %u.  Default TAIL_TRIGS32 is 2.\n", best_tail_trigs);
      configsUpdate(current_cost, best_cost, 0.003, "TAIL_TRIGS32", best_tail_trigs, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["TAIL_TRIGS32"] = to_string(best_tail_trigs);
    }

    // Find best TAIL_TRIGS61 setting
    if (time_NTTs) {
      FFTConfig fft{defaultNTTShape, 202, CARRY_AUTO};
      if (!fft.NTT_GF61) fft = FFTConfig(FFTShape(FFT3161, 512, 8, 512), 202, CARRY_AUTO);
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_tail_trigs = 0;
      u32 const current_tail_trigs = args->value("TAIL_TRIGS61", 0);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const tail_trigs : {0, 1}) {
        args->flags["TAIL_TRIGS61"] = to_string(tail_trigs);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using TAIL_TRIGS61=%u is %6.1f\n", fft.spec().c_str(), tail_trigs, cost);
        if (tail_trigs == current_tail_trigs) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tail_trigs = tail_trigs; }
      }
      log("Best TAIL_TRIGS61 is %u.  Default TAIL_TRIGS61 is 0.\n", best_tail_trigs);
      configsUpdate(current_cost, best_cost, 0.003, "TAIL_TRIGS61", best_tail_trigs, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["TAIL_TRIGS61"] = to_string(best_tail_trigs);
    }

    // Find best TABMUL_CHAIN setting
    if (time_FFTs) {
      FFTConfig const fft{defaultFFTShape, 101, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_tabmul_chain = 0;
      u32 const current_tabmul_chain = args->value("TABMUL_CHAIN", 0);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const tabmul_chain : {0, 1}) {
        args->flags["TABMUL_CHAIN"] = to_string(tabmul_chain);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using TABMUL_CHAIN=%u is %6.1f\n", fft.spec().c_str(), tabmul_chain, cost);
        if (tabmul_chain == current_tabmul_chain) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tabmul_chain = tabmul_chain; }
      }
      log("Best TABMUL_CHAIN is %u.  Default TABMUL_CHAIN is 0.\n", best_tabmul_chain);
      configsUpdate(current_cost, best_cost, 0.003, "TABMUL_CHAIN", best_tabmul_chain, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["TABMUL_CHAIN"] = to_string(best_tabmul_chain);
    }

    // Find best TABMUL_CHAIN31 setting
    if (time_NTTs) {
      FFTConfig fft{defaultNTTShape, 202, CARRY_AUTO};
      if (!fft.NTT_GF31) fft = FFTConfig(FFTShape(FFT3161, 512, 8, 512), 202, CARRY_AUTO);
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_tabmul_chain = 0;
      u32 const current_tabmul_chain = args->value("TABMUL_CHAIN31", 0);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const tabmul_chain : {0, 1}) {
        args->flags["TABMUL_CHAIN31"] = to_string(tabmul_chain);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using TABMUL_CHAIN31=%u is %6.1f\n", fft.spec().c_str(), tabmul_chain, cost);
        if (tabmul_chain == current_tabmul_chain) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tabmul_chain = tabmul_chain; }
      }
      log("Best TABMUL_CHAIN31 is %u.  Default TABMUL_CHAIN31 is 0.\n", best_tabmul_chain);
      configsUpdate(current_cost, best_cost, 0.003, "TABMUL_CHAIN31", best_tabmul_chain, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["TABMUL_CHAIN31"] = to_string(best_tabmul_chain);
    }

    // Find best TABMUL_CHAIN32 setting
    if (time_NTTs && time_FP32) {
      FFTConfig fft{defaultNTTShape, 202, CARRY_AUTO};
      if (!fft.FFT_FP32) fft = FFTConfig(FFTShape(FFT3261, 512, 8, 512), 202, CARRY_AUTO);
      u64 const exponent = primes.prevPrime(u64(fft.maxBpw() * 0.95 * fft.shape.size()));   // Back off the maxExp as different settings will have different maxBpw
      u32 best_tabmul_chain = 0;
      u32 const current_tabmul_chain = args->value("TABMUL_CHAIN32", 0);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const tabmul_chain : {0, 1}) {
        args->flags["TABMUL_CHAIN32"] = to_string(tabmul_chain);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using TABMUL_CHAIN32=%u is %6.1f\n", fft.spec().c_str(), tabmul_chain, cost);
        if (tabmul_chain == current_tabmul_chain) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tabmul_chain = tabmul_chain; }
      }
      log("Best TABMUL_CHAIN32 is %u.  Default TABMUL_CHAIN32 is 0.\n", best_tabmul_chain);
      configsUpdate(current_cost, best_cost, 0.003, "TABMUL_CHAIN32", best_tabmul_chain, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["TABMUL_CHAIN32"] = to_string(best_tabmul_chain);
    }

    // Find best TABMUL_CHAIN61 setting
    if (time_NTTs) {
      FFTConfig fft{defaultNTTShape, 202, CARRY_AUTO};
      if (!fft.NTT_GF61) fft = FFTConfig(FFTShape(FFT3161, 512, 8, 512), 202, CARRY_AUTO);
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_tabmul_chain = 0;
      u32 const current_tabmul_chain = args->value("TABMUL_CHAIN61", 0);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const tabmul_chain : {0, 1}) {
        args->flags["TABMUL_CHAIN61"] = to_string(tabmul_chain);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using TABMUL_CHAIN61=%u is %6.1f\n", fft.spec().c_str(), tabmul_chain, cost);
        if (tabmul_chain == current_tabmul_chain) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tabmul_chain = tabmul_chain; }
      }
      log("Best TABMUL_CHAIN61 is %u.  Default TABMUL_CHAIN61 is 0.\n", best_tabmul_chain);
      configsUpdate(current_cost, best_cost, 0.003, "TABMUL_CHAIN61", best_tabmul_chain, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["TABMUL_CHAIN61"] = to_string(best_tabmul_chain);
    }

    // Find best MODM31 setting
    if (time_NTTs) {
      FFTConfig fft{defaultNTTShape, 202, CARRY_AUTO};
      if (!fft.NTT_GF31) fft = FFTConfig(FFTShape(FFT3161, 512, 8, 512), 202, CARRY_AUTO);
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_modm31 = 0;
      u32 const current_modm31 = args->value("MODM31", 0);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const modm31 : {0, 1, 2}) {
        args->flags["MODM31"] = to_string(modm31);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using MODM31=%u is %6.1f\n", fft.spec().c_str(), modm31, cost);
        if (modm31 == current_modm31) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_modm31 = modm31; }
      }
      log("Best MODM31 is %u.  Default MODM31 is 0.\n", best_modm31);
      configsUpdate(current_cost, best_cost, 0.000, "MODM31", best_modm31, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["MODM31"] = to_string(best_modm31);
    }

    // Find best UNROLL_W setting
    if (true) {
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_unroll_w = 0;
      u32 const current_unroll_w = args->value("UNROLL_W", 1);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const unroll_w : {0, 1}) {
        args->flags["UNROLL_W"] = to_string(unroll_w);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using UNROLL_W=%u is %6.1f\n", fft.spec().c_str(), unroll_w, cost);
        if (unroll_w == current_unroll_w) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_unroll_w = unroll_w; }
      }
      log("Best UNROLL_W is %u.  Default UNROLL_W is 1.\n", best_unroll_w);
      configsUpdate(current_cost, best_cost, 0.003, "UNROLL_W", best_unroll_w, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["UNROLL_W"] = to_string(best_unroll_w);
    }

    // Find best UNROLL_H setting
    if (true) {
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_unroll_h = 0;
      u32 const current_unroll_h = args->value("UNROLL_H", 1);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const unroll_h : {0, 1}) {
        args->flags["UNROLL_H"] = to_string(unroll_h);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using UNROLL_H=%u is %6.1f\n", fft.spec().c_str(), unroll_h, cost);
        if (unroll_h == current_unroll_h) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_unroll_h = unroll_h; }
      }
      log("Best UNROLL_H is %u.  Default UNROLL_H is 1.\n", best_unroll_h);
      configsUpdate(current_cost, best_cost, 0.003, "UNROLL_H", best_unroll_h, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["UNROLL_H"] = to_string(best_unroll_h);
    }

    // Find best ZEROHACK_W setting
    if (true) {
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_zerohack_w = 0;
      u32 const current_zerohack_w = args->value("ZEROHACK_W", 1);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const zerohack_w : {0, 1}) {
        args->flags["ZEROHACK_W"] = to_string(zerohack_w);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using ZEROHACK_W=%u is %6.1f\n", fft.spec().c_str(), zerohack_w, cost);
        if (zerohack_w == current_zerohack_w) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_zerohack_w = zerohack_w; }
      }
      log("Best ZEROHACK_W is %u.  Default ZEROHACK_W is 1.\n", best_zerohack_w);
      configsUpdate(current_cost, best_cost, 0.003, "ZEROHACK_W", best_zerohack_w, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["ZEROHACK_W"] = to_string(best_zerohack_w);
    }

    // Find best ZEROHACK_H setting
    if (true) {
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_zerohack_h = 0;
      u32 const current_zerohack_h = args->value("ZEROHACK_H", 1);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const zerohack_h : {0, 1}) {
        args->flags["ZEROHACK_H"] = to_string(zerohack_h);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using ZEROHACK_H=%u is %6.1f\n", fft.spec().c_str(), zerohack_h, cost);
        if (zerohack_h == current_zerohack_h) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_zerohack_h = zerohack_h; }
      }
      log("Best ZEROHACK_H is %u.  Default ZEROHACK_H is 1.\n", best_zerohack_h);
      configsUpdate(current_cost, best_cost, 0.003, "ZEROHACK_H", best_zerohack_h, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["ZEROHACK_H"] = to_string(best_zerohack_h);
    }

    // Find best WMUL setting
    if (true && defaultShape->width != 4096) {
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_wmul = 0;
      u32 const current_wmul = args->value("WMUL", 2);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const wmul : {1, 2, 4}) {
        args->flags["WMUL"] = to_string(wmul);
        // A WMUL this FFT or device cannot use is lowered (e.g. 4 to 2 for a 1K width).  Score and save the value that ran.
        u32 used = wmul;
        double const cost = timeConfig(exponent, shared, fft, {}, quick, &used);
        log("Time for %12s using WMUL=%u is %6.1f\n", fft.spec().c_str(), used, cost);
        if (wmul == current_wmul) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_wmul = used; }
      }
      log("Best WMUL is %u.  Default WMUL is 2.\n", best_wmul);
      configsUpdate(current_cost, best_cost, 0.000, "WMUL", best_wmul, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["WMUL"] = to_string(best_wmul);
    }

    // Find best MULTI_Q setting (MULTI_Q can only be advantageous when using multiple data types such as FFT3161, FFT6431, etc).
    // Since FFT6431 is rarely used and FFT3161 is very common, we case off the time_NTTs boolean.
    if (time_NTTs) {
      FFTConfig fft{defaultNTTShape, 202, CARRY_AUTO};
      if (!fft.NTT_GF61) fft = FFTConfig(FFTShape(FFT3161, 512, 8, 512), 202, CARRY_AUTO);
      u64 exponent = primes.prevPrime(fft.maxExp());
      u32 best_multi_q = 0;
      u32 current_multi_q = args->value("MULTI_Q", 0);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 multi_q : {0, 1}) {
        args->flags["MULTI_Q"] = to_string(multi_q);
        double cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using MULTI_Q=%u is %6.1f\n", fft.spec().c_str(), multi_q, cost);
        if (multi_q == current_multi_q) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_multi_q = multi_q; }
      }
      log("Best MULTI_Q is %u.  Default MULTI_Q is 0.\n", best_multi_q);
      configsUpdate(current_cost, best_cost, 0.000, "MULTI_Q", best_multi_q, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["MULTI_Q"] = to_string(best_multi_q);
    }

    // Find best CUDA compiler options
#if CUDA_BACKEND
    // Find best L1CUDA setting.
    if (true) {
      FFTConfig fft{*defaultShape, variant, CARRY_AUTO};
      u64 exponent = primes.prevPrime(fft.maxExp());
      u32 best_l1cuda = 0;
      u32 current_l1cuda = args->value("L1CUDA", 0);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 l1cuda : {0, 1, 2, 3}) {
        args->flags["L1CUDA"] = to_string(l1cuda);
        double cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using L1CUDA=%u is %6.1f\n", fft.spec().c_str(), l1cuda, cost);
        if (l1cuda == current_l1cuda) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_l1cuda = l1cuda; }
      }
      log("Best L1CUDA is %u.  Default L1CUDA is 0.\n", best_l1cuda);
      configsUpdate(current_cost, best_cost, 0.000, "L1CUDA", best_l1cuda, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["L1CUDA"] = to_string(best_l1cuda);
    }

    // Find best GRAPHS setting.  Require a clear advantage to override the default GRAPHS setting.  GRAPHS=1 will use less CPU time.
    if (true) {
      FFTConfig fft{*defaultShape, variant, CARRY_AUTO};
      u64 exponent = primes.prevPrime(fft.maxExp());
      u32 best_graphs = 0;
      u32 current_graphs = args->value("GRAPHS", 1);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 graphs : {0, 1}) {
        args->flags["GRAPHS"] = to_string(graphs);
        double cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using GRAPHS=%u is %6.1f\n", fft.spec().c_str(), graphs, cost);
        if (graphs == current_graphs) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_graphs = graphs; }
      }
      log("Best GRAPHS is %u.  Default GRAPHS is 1.\n", best_graphs);
      configsUpdate(current_cost, best_cost, 0.003, "GRAPHS", best_graphs, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["GRAPHS"] = to_string(best_graphs);
    }

#endif

    // Find best BIGLIT setting
    if (false && time_FFTs) {        // Deprecated
      FFTConfig const fft{*defaultShape, variant, CARRY_AUTO};
      u64 const exponent = primes.prevPrime(fft.maxExp());
      u32 best_biglit = 0;
      u32 const current_biglit = args->value("BIGLIT", 1);
      double best_cost = -1.0;
      double current_cost = -1.0;
      for (u32 const biglit : {0, 1}) {
        args->flags["BIGLIT"] = to_string(biglit);
        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        log("Time for %12s using BIGLIT=%u is %6.1f\n", fft.spec().c_str(), biglit, cost);
        if (biglit == current_biglit) current_cost = cost;
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_biglit = biglit; }
      }
      log("Best BIGLIT is %u.  Default BIGLIT is 1.  The BIGLIT=0 option will probably be deprecated.\n", best_biglit);
      configsUpdate(current_cost, best_cost, 0.003, "BIGLIT", best_biglit, newConfigKeyVals, suggestedConfigKeyVals);
      args->flags["BIGLIT"] = to_string(best_biglit);
    }

    // Output new settings to config.txt
    File config = File::openAppend("config.txt");
    if (!newConfigKeyVals.empty()) {
      config.write("\n# New settings based on a -tune run.");
      for (u32 i = 0; i < newConfigKeyVals.size(); ++i) {
        config.write(i == 0 ? "\n   -use " : ",");
        config.printf("%s=%u", newConfigKeyVals[i].first.c_str(), newConfigKeyVals[i].second);
      }
      config.write("\n");
    }
    if (!suggestedConfigKeyVals.empty()) {
      config.write("\n# These settings were slightly faster in a -tune run.");
      config.write("\n# It is suggested that each setting be timed over a longer duration to see if the setting really is faster.");
      for (u32 i = 0; i < suggestedConfigKeyVals.size(); ++i) {
        config.write(i == 0 ? "\n#  -use " : ",");
        config.printf("%s=%u", suggestedConfigKeyVals[i].first.c_str(), suggestedConfigKeyVals[i].second);
      }
      config.write("\n");
    }
    if (args->logStep < 100000) {
      config.write("\n# Less frequent save file creation improves throughput.");
      config.write("\n  -log 1000000\n");
    }
    if (args->workers < 2) {
      config.write("\n# Running two workers sometimes gives better throughput.  AutoPrimeNet will need to create a second worktodo file (use --num-workers 2).");
      config.write("\n#  -workers 2\n");
// Recent change (October 2026) to tailSquare's reverseLine may make TAIL_KERNELS=3 obsolete
//      config.write("\n# Changing TAIL_KERNELS to 3 when running two workers may be better.");
//      config.write("\n#  -use TAIL_KERNELS=3\n");
    }
  }

  // Flags that prune the amount of shapes and variants to time.
  // These should be computed automatically and saved in the tune.txt or config.txt file.
  // Tune.txt file should have a version number.

  // A command line option to run more combinations (higher number skips more combos)
  int skip_some_WH_variants = 1;                // 0 = skip nothing, 1 = skip slower widths/heights unless they have better Z, 2 = only run fastest widths/heights

  // The width = height = 512 FFT shape is so good, we probably don't need to time the width = 1024, height = 256 shape.  Even more true for 2K and 256!
  bool skip_2K_256 = true;
  bool skip_2K_512 = true;

// make command line args for this? 
skip_some_WH_variants = 2;   // should default be 1??

  // For each width, time the 001, 101, and 201 FP64 variants to find the fastest width variant.
  // In an ideal world we'd use the -time feature and look at the kCarryFused timing.  Then we'd save this info in config.txt or tune.txt.
  map<int, u32> fastest_width_variants;

  // For each height, time the 100, 101, and 102 FP64 variants to find the fastest height variant.
  // In an ideal world we'd use the -time feature and look at the tailSquare timing.  Then we'd save this info in config.txt or tune.txt.
  map<int, u32> fastest_height_variants;

  vector<TuneEntry> results = TuneEntry::readTuneFile(*args);
  set<string> oldSpecs;
  for (const TuneEntry& e : results) { oldSpecs.insert(e.fft.spec()); }
  set<string> addedSpecs;                       // the entries this run added (the register limits are tuned for these)

#ifdef CUDA_BACKEND
  CudaSmLimits const sm = cudaSmLimits();
#endif

  // Time FFT shapes smallest-to-largest exponent handled
  std::ranges::stable_sort(shapes, [](const FFTShape& a, const FFTShape& b) { return a.maxExp() < b.maxExp(); });

  // Loop through all possible FFT shapes
  for (const FFTShape& shape : shapes) {

    // Skip some FFTs and NTTs
    if (shape.fft_type == FFT64 && !time_FFTs) continue;
    if (shape.fft_type == FFT6431 && !time_FFT6431) continue;
    if (shape.fft_type != FFT64 && shape.fft_type != FFT6431 && !time_NTTs) continue;
    if ((shape.fft_type == FFT3261 || shape.fft_type == FFT323161 || shape.fft_type == FFT3231 || shape.fft_type == FFT32) && !time_FP32) continue;

    // Skip the optional groups that are not wanted (a single -fft shape is always timed).  A shape in a group set to 0 is
    // skipped.  If any group is set to 2, only shapes in a group set to 2 are timed.
    if (shapes.size() > 1) {
      FFTConfig const anyVariant{shape, 202, CARRY_AUTO};
      bool skip = false, inOnlyGroup = false;
      for (auto [inGroup, setting] : {pair{(shape.width == 256 && shape.height == 1024) || (shape.width == 1024 && shape.height == 256), time_1K_256},
                                      pair{shape.fft_type == FFT61, time_M61},
                                      pair{shape.fft_type == FFT6431, time_FFT6431},
                                      pair{shape.fft_type == FFT3261 || shape.fft_type == FFT323161 || shape.fft_type == FFT3231 || shape.fft_type == FFT32, time_FP32},
                                      pair{shape.isPfa() && (anyVariant.FFT_FP64 || anyVariant.FFT_FP32), time_PFA}}) {
        if (!inGroup) continue;
        if (setting == 0) skip = true;
        if (setting == 2) inOnlyGroup = true;
      }
      if (skip || (onlyGroups && !inOnlyGroup)) continue;
    }

    // Time an exponent that's good for all variants and carry-config.
    u64 const exponent = primes.prevPrime(FFTConfig{shape, shape.width <= 1024 ? 0u : 100u, CARRY_32}.maxExp());
    u32 adjusted_quick = (exponent < 50000000) ? quick - 1 : (exponent < 170000000) ? quick : (exponent < 350000000) ? quick + 1 : quick + 2;
    adjusted_quick = std::max<u32>(adjusted_quick, 1);
    adjusted_quick = std::min<u32>(adjusted_quick, 10);

    // Loop through all possible variants
    for (u32 variant = 0; variant <= LAST_VARIANT; variant = next_variant (variant)) {

      // Only FP64 code supports variants.  Middle variant 1 (more accurate, slower) is timed after this sweep, only for the variants
      // that earned an entry.
      if (variant != 202 && !FFTConfig{shape, variant, CARRY_AUTO}.FFT_FP64) continue;
      if (variant_M(variant) == 1) continue;

      // Only AMD GPUs profitably support variant zero (BCAST) and only if width <= 1024.  CLANG doesn't support builtins.  Have NO_ASM bypass variant zero.
      // nVidia now supports variant zero, but is slower on TitanV
      if (variant_W(variant) == 0) {
        if (!VARIANT0) continue;
        if (shape.width > 1024) continue;
      }

      // Only AMD GPUs profitably support variant zero (BCAST) and only if height <= 1024.
      // nVidia now supports variant zero, but is slower on TitanV
      if (variant_H(variant) == 0) {
        if (!VARIANT0) continue;
        if (shape.height > 1024) continue;
      }

      // Reject shapes that won't be used to test exponents in the user's desired range.
      // We need to test significantly higher than max_exponent in search of a favored FFT shape that produces the best timing.
      {
        FFTConfig const fft{shape, variant, CARRY_AUTO};
        if (fft.maxExp() < min_exponent) continue;
        if (fft.maxExp() > 1.3*max_exponent) continue;
        if (shape.fft_type == FFT64 && fft.maxExp() > 1.2*max_exponent) continue;
      }

      // If only one shape was specified on the command line, time it.  This lets the user time any shape, including non-favored ones.
      if (shapes.size() > 1) {

        // Skip less-favored shapes
        if (!shape.isFavoredShape()) continue;

        // Skip some combinations that are unlikely to be fruitful
        if (shape.width == 256 && shape.height == 2048 && skip_2K_256) continue;
        if (shape.width == 2048 && shape.height == 256 && skip_2K_256) continue;
        if (shape.width == 512 && shape.height == 2048 && skip_2K_512) continue;
        if (shape.width == 2048 && shape.height == 512 && skip_2K_512) continue;

        // Skip variants where width or height are not using the fastest variant.
        // NOTE: We ought to offer a tune=option where we also test more accurate variants to extend the FFT's max exponent.
        if (skip_some_WH_variants && FFTConfig{shape, variant, CARRY_AUTO}.FFT_FP64) {
          u32 fastest_width = 1;
          if (auto it = fastest_width_variants.find(shape.width); it != fastest_width_variants.end()) {
            fastest_width = it->second;
          } else {
            FFTShape const test = FFTShape(FFT64, shape.width, 12, 256);
            double cost, min_cost = -1.0;
            for (u32 w = 0; w < N_VARIANT_W; w++) {
              if (w == 0 && !VARIANT0) continue;
              if (w == 0 && test.width > 1024) continue;
              FFTConfig const fft{test, variant_WMH (w, 0, 1), CARRY_32};
              cost = timeConfig(primes.prevPrime(fft.maxExp()), shared, fft, {}, adjusted_quick);
              log("Fast width search %6.1f %12s\n", cost, fft.spec().c_str());
              if (min_cost < 0.0 || cost < min_cost) { min_cost = cost; fastest_width = w; }
            }
            fastest_width_variants[shape.width] = fastest_width;
          }
          if (skip_some_WH_variants == 2 && variant_W(variant) != fastest_width) continue;
          if (skip_some_WH_variants == 1 &&
              FFTConfig{shape, variant, CARRY_32}.maxBpw() <
                  FFTConfig{shape, variant_WMH (fastest_width, variant_M(variant), variant_H(variant)), CARRY_32}.maxBpw()) continue;
        }
        if (skip_some_WH_variants && FFTConfig{shape, variant, CARRY_AUTO}.FFT_FP64) {
          u32 fastest_height = 1;
          if (auto it = fastest_height_variants.find(shape.height); it != fastest_height_variants.end()) {
            fastest_height = it->second;
          } else {
            FFTShape const test = FFTShape(FFT64, shape.height, 12, shape.height);
            double cost, min_cost = -1.0;
            for (u32 h = 0; h < N_VARIANT_H; h++) {
              if (h == 0 && !VARIANT0) continue;
              if (h == 0 && test.height > 1024) continue;
              FFTConfig const fft{test, variant_WMH (1, 0, h), CARRY_32};
              cost = timeConfig(primes.prevPrime(fft.maxExp()), shared, fft, {}, quick);
              log("Fast height search %6.1f %12s\n", cost, fft.spec().c_str());
              if (min_cost < 0.0 || cost < min_cost) { min_cost = cost; fastest_height = h; }
            }
            fastest_height_variants[shape.height] = fastest_height;
          }
          if (skip_some_WH_variants == 2 && variant_H(variant) != fastest_height) continue;
          if (skip_some_WH_variants == 1 &&
              FFTConfig{shape, variant, CARRY_32}.maxBpw() <
                  FFTConfig{shape, variant_WMH (variant_W(variant), variant_M(variant), fastest_height), CARRY_32}.maxBpw()) continue;
        }
      }

//GW: If variant is specified on command line, time it (and only it)??  Or an option to only time one variant number??

      vector carryToTest{CARRY_AUTO};
      if (shape.fft_type == FFT64) {
        carryToTest[0] = CARRY_32;
        // We need to test both carry-32 and carry-64 only when the carry transition is within the BPW range.
        if (FFTConfig{shape, variant, CARRY_64}.maxBpw() > FFTConfig{shape, variant, CARRY_32}.maxBpw()) {
          carryToTest.push_back(CARRY_64);
        }
      }

      for (auto carry : carryToTest) {
        FFTConfig const fft{shape, variant, carry};

        // Skip middle = 1, CARRY_32 if maximum exponent would be the same as middle = 0, CARRY_32
        if (variant_M(variant) > 0 && carry == CARRY_32 && fft.maxExp() <= FFTConfig{shape, variant - 10, CARRY_32}.maxExp()) continue;

        double const cost = timeConfig(exponent, shared, fft, {}, quick);
        TuneEntry entry{cost, fft, {}};
#ifdef CUDA_BACKEND
        // -tune fine: an FFT that earns a tune.txt entry, or comes within fine_pct of one, gets its best register limits.  Then it
        // has to earn the entry with them.
        if (fine_pct > 0 && std::isfinite(cost) && TuneEntry{cost * (1 - fine_pct / 100), fft, {}}.willUpdate(results)) {
          log("~ %6.1f %12s %9" PRIu64 " is within %g%% of a tune.txt entry: tuning its register limits\n", cost, fft.spec().c_str(), fft.maxExp(), fine_pct);
          entry = regTuneEntry(entry, quick, sm);
        }
#endif
        bool const isUseful = std::isfinite(entry.cost) && entry.update(results);
        log("%c %6.1f %12s %9" PRIu64 "\n", isUseful ? '*' : ' ', entry.cost, fft.spec().c_str(), fft.maxExp());
        if (isUseful) {
          TuneEntry::writeTuneFile(results);
          addedSpecs.insert(fft.spec());
        }
      }
    }
  }

  // FFTs up to 30% above max_exponent were timed to find the cheapest that handles max_exponent.  Of the new entries above
  // max_exponent only that one is kept.  (The entries tune.txt had before are kept, they may come from tuning a higher range.)
  auto pruneAbove = [&]() {
    if (!args->fftSpec.empty()) { return; }
    vector<TuneEntry> kept;
    bool haveAbove = false;
    for (const TuneEntry& e : results) {      // in increasing max exponent
      bool const above = e.fft.maxExp() >= max_exponent;
      if (!above || !haveAbove || oldSpecs.contains(e.fft.spec())) { kept.push_back(e); }
      haveAbove = haveAbove || above;
    }
    if (kept.size() < results.size()) {
      log("Removed %u new tune.txt entries above maxexp=%" PRIu64 ", one entry handles it\n", u32(results.size() - kept.size()), max_exponent);
      results = std::move(kept);
      TuneEntry::writeTuneFile(results);
    }
  };
  pruneAbove();

  // Middle variant 1 handles a slightly higher max exponent than middle variant 0, usually at a small cost (on a TITAN V about 1%
  // in the middle kernels).  Time it for the middle variant 0 FFTs below max_exponent that earned an entry above: it may earn the
  // next one.
  {
    vector<FFTConfig> m1;
    for (const TuneEntry& e : results) {
      FFTConfig const& fft = e.fft;
      if (!addedSpecs.contains(fft.spec()) || !fft.FFT_FP64 || variant_M(fft.variant) != 0 || fft.maxExp() >= max_exponent) { continue; }
      FFTConfig const next{fft.shape, fft.variant + 10, fft.carry};
      if (oldSpecs.contains(next.spec()) || addedSpecs.contains(next.spec()) || next.maxExp() <= fft.maxExp()) { continue; }
      m1.push_back(next);
    }
    if (!m1.empty()) { log("Timing middle variant 1 of the %u new tune.txt entries with middle variant 0\n", u32(m1.size())); }
    for (const FFTConfig& fft : m1) {
      u64 const exponent = primes.prevPrime(FFTConfig{fft.shape, fft.shape.width <= 1024 ? 0u : 100u, CARRY_32}.maxExp());
      u32 adjusted_quick = (exponent < 50000000) ? quick - 1 : (exponent < 170000000) ? quick : (exponent < 350000000) ? quick + 1 : quick + 2;
      adjusted_quick = std::clamp<u32>(adjusted_quick, 1, 10);
      double const cost = timeConfig(exponent, shared, fft, {}, adjusted_quick);
      bool const isUseful = std::isfinite(cost) && TuneEntry{cost, fft, {}}.update(results);
      log("%c %6.1f %12s %9" PRIu64 "\n", isUseful ? '*' : ' ', cost, fft.spec().c_str(), fft.maxExp());
      if (isUseful) {
        TuneEntry::writeTuneFile(results);
        addedSpecs.insert(fft.spec());
      }
    }
  }
  pruneAbove();      // a middle variant 1 may now be the cheapest above max_exponent

#ifdef CUDA_BACKEND
  // The FFTs were timed with the compiler's default registers.  Now find the best register limits for the entries this run
  // added.  The other entries keep theirs (-tune regs retunes them).  -tune fine has already done this for every FFT that earned an entry.
  if (fine_pct == 0) { regTune(quick, args->fftSpec.empty() ? vector<FFTShape>{} : shapes, &addedSpecs); }
#endif
}

void Tune::variantTune(int quick, bool variant0, const vector<FFTShape>& onlyShapes, u64 minExp, u64 maxExp) {
  vector<TuneEntry> results = TuneEntry::readTuneFile(*shared.args);
  if (results.empty()) {
    log("-tune variants: tune.txt has no entries\n");
    return;
  }
  set<string> specs;                  // the FFTs in tune.txt, which are not timed again
  vector<FFTShape> shapes;            // the shapes to time the variants of, once each
  for (const TuneEntry& e : results) {
    specs.insert(e.fft.spec());
    FFTShape const& shape = e.fft.shape;
    if (!e.fft.FFT_FP64) { continue; }      // Only FP64 code supports variants
    if (!onlyShapes.empty() && std::ranges::none_of(onlyShapes, [&](const FFTShape& s) { return s.spec() == shape.spec(); })) { continue; }
    if (std::ranges::none_of(shapes, [&](const FFTShape& s) { return s.spec() == shape.spec(); })) { shapes.push_back(shape); }
  }
  // As -tune times an FFT shape, but every variant: middle variant 1 too, and the width/height variants -tune skips as slower
  vector<vector<FFTConfig>> candidates(shapes.size());
  for (size_t i = 0; i < shapes.size(); ++i) {
    FFTShape const& shape = shapes[i];
    for (u32 variant = 0; variant <= LAST_VARIANT; variant = next_variant(variant)) {
      if (variant_W(variant) == 0 && (!variant0 || shape.width > 1024)) { continue; }
      if (variant_H(variant) == 0 && (!variant0 || shape.height > 1024)) { continue; }
      vector carries{CARRY_AUTO};
      if (shape.fft_type == FFT64) {
        carries = {CARRY_32};
        if (FFTConfig{shape, variant, CARRY_64}.maxBpw() > FFTConfig{shape, variant, CARRY_32}.maxBpw()) { carries.push_back(CARRY_64); }
      }
      for (auto carry : carries) {
        FFTConfig const fft{shape, variant, carry};
        if (specs.contains(fft.spec())) { continue; }
        // Unlike -tune, nothing above maxExp: tune.txt already has the entry that handles maxExp
        if (fft.maxExp() < minExp || fft.maxExp() > maxExp) { continue; }
        // Skip middle = 1, CARRY_32 if maximum exponent would be the same as middle = 0, CARRY_32
        if (variant_M(variant) > 0 && carry == CARRY_32 && fft.maxExp() <= FFTConfig{shape, variant - 10, CARRY_32}.maxExp()) { continue; }
        candidates[i].push_back(fft);
      }
    }
  }
  u32 const nShapes = std::ranges::count_if(candidates, [](const auto& c) { return !c.empty(); });
  log("Timing the other variants of %u FFT shapes in tune.txt, as -tune does.  Then the register usage of those that earn\n", nShapes);
  log("a tune.txt entry is tuned.\n");

  set<string> addedSpecs;
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (candidates[i].empty()) { continue; }
    FFTShape const& shape = shapes[i];
    u64 const exponent = primes.prevPrime(FFTConfig{shape, shape.width <= 1024 ? 0u : 100u, CARRY_32}.maxExp());
    u32 adjusted_quick = (exponent < 50000000) ? quick - 1 : (exponent < 170000000) ? quick : (exponent < 350000000) ? quick + 1 : quick + 2;
    adjusted_quick = std::clamp<u32>(adjusted_quick, 1, 10);
    for (const FFTConfig& fft : candidates[i]) {
      double const cost = timeConfig(exponent, shared, fft, {}, adjusted_quick);
      bool const isUseful = std::isfinite(cost) && TuneEntry{cost, fft, {}}.update(results);
      log("%c %6.1f %12s %9" PRIu64 "\n", isUseful ? '*' : ' ', cost, fft.spec().c_str(), fft.maxExp());
      if (isUseful) {
        TuneEntry::writeTuneFile(results);
        addedSpecs.insert(fft.spec());
      }
    }
  }

#ifdef CUDA_BACKEND
  // The variants were timed with the compiler's default registers, as -tune does.  Now tune the registers of those that earned an entry.
  regTune(quick, {}, &addedSpecs);
#endif
}

void Tune::regTune([[maybe_unused]] int quick, [[maybe_unused]] const vector<FFTShape>& onlyShapes, [[maybe_unused]] const set<string>* onlySpecs) {
#ifndef CUDA_BACKEND
  log("-tune regs: register limits are only tuned in the CUDA build\n");
#else
  vector<TuneEntry> const entries = TuneEntry::readTuneFile(*shared.args);
  if (entries.empty()) {
    log("-tune regs: tune.txt has no entries to tune\n");
    return;
  }
  CudaSmLimits const sm = cudaSmLimits();
  log("\n");
  log("Tuning register usage of %stune.txt entries.  Per SM: %d registers, %d threads, %d blocks, %d bytes shared memory.\n",
      onlySpecs ? "new " : "", sm.regsPerSM, sm.maxThreadsPerSM, sm.maxBlocksPerSM, sm.sharedPerSM);

  vector<TuneEntry> done;
  for (u32 i = 0; i < entries.size(); ++i) {
    const TuneEntry& e = entries[i];
    bool const selected = (onlyShapes.empty() || std::ranges::any_of(onlyShapes, [&](const FFTShape& s) { return s.spec() == e.fft.shape.spec(); }))
                          && (!onlySpecs || onlySpecs->contains(e.fft.spec()));
    done.push_back(selected ? regTuneEntry(e, quick, sm) : e);

    // Rewrite tune.txt after each entry: the entries done so far, and the rest as they were
    vector<TuneEntry> results;
    for (const TuneEntry& d : done) { d.update(results); }
    for (u32 j = i + 1; j < entries.size(); ++j) { entries[j].update(results); }
    TuneEntry::writeTuneFile(results);
  }
#endif
}

#ifdef CUDA_BACKEND
namespace {

// A kernel's occupancy as its registers set it.  The registers of a block are allocated per warp in units of 256, and occupancy
// changes where the blocks that fit in the SM's register file change.  k0 is the blocks per SM with the compiler's default
// registers, maxBlocks the most that the other limits (blocks, threads, shared memory per SM) allow.
struct Occupancy {
  int k0{};
  int maxBlocks{};
  int warps{};
  int maxRegs{};
  int regsPerSM{};

  // The most registers a thread can use with this many blocks per SM
  [[nodiscard]] int regsFor(int blocks) const { return std::min(maxRegs, regsPerSM / (blocks * warps) / 256 * 256 / 32); }

  // Whether launch bounds of this many blocks can be tried.  One block per SM below the default occupancy was always slower.
  // REGxxxx values up to 16 are launch bounds.
  [[nodiscard]] bool canTry(int blocks) const {
    return blocks >= 1 && (blocks >= 2 || blocks == k0) && blocks <= std::min(maxBlocks, 16) && regsFor(blocks) >= 24;
  }
};

Occupancy occupancy(const CudaKernelResources& r, const CudaSmLimits& sm) {
  Occupancy o{};
  if (r.threads <= 0 || r.regs <= 0) { return o; }
  o.warps = (r.threads + 31) / 32;
  o.maxRegs = std::min(255, sm.regsPerBlock / r.threads);
  o.regsPerSM = sm.regsPerSM;
  o.maxBlocks = std::min(sm.maxBlocksPerSM, sm.maxThreadsPerSM / r.threads);
  if (r.sharedBytes) { o.maxBlocks = std::min(o.maxBlocks, sm.sharedPerSM / (r.sharedBytes + sm.reservedSharedPerBlock)); }
  o.k0 = std::min(sm.regsPerSM / ((r.regs * 32 + 255) / 256 * 256 * o.warps), o.maxBlocks);
  return o;
}

// The search for one kernel's best register limit.  Launch bounds come first, as minimum blocks per SM: k0 (the default occupancy,
// which still compiles differently), k0 - 1 (more registers), k0 + 1 (less), and k0 + 2 if k0 + 1 was faster than k0.  Then,
// the maximum register count of the fastest of those occupancies (or of k0 if the default was fastest): a maximum
// register count compiles differently from launch bounds of the same occupancy, sometimes better.
struct KernelSearch {
  Gpu::RegTunable base;
  Occupancy occ;
  int step = 0;
  map<int, double> us;              // the time of each candidate tried, by REGxxxx value

  [[nodiscard]] double usOf(int v) const { auto it = us.find(v); return it == us.end() ? numeric_limits<double>::infinity() : it->second; }

  // The blocks per SM of the fastest launch bounds, or k0 if none beat the default
  [[nodiscard]] int bestBlocks() const {
    int best = occ.k0;
    double bestUs = base.usPerCall;
    for (auto [v, t] : us) { if (v <= 16 && t < bestUs) { bestUs = t; best = v; } }
    return best;
  }

  // The next candidate to time, 0 when the search is done
  int next() {
    int const k0 = occ.k0;
    while (step < 5) {
      switch (step++) {
      case 0: if (occ.canTry(k0)) { return k0; } break;
      case 1: if (occ.canTry(k0 - 1)) { return k0 - 1; } break;
      case 2: if (occ.canTry(k0 + 1)) { return k0 + 1; } break;
      case 3: if (occ.canTry(k0 + 2) && usOf(k0 + 1) < std::min(usOf(k0), base.usPerCall)) { return k0 + 2; } break;
      case 4: if (int const regs = occ.regsFor(bestBlocks()); regs > 16 && !us.contains(regs)) { return regs; } break;
      }
    }
    return 0;
  }

  // What will be tried, for the log
  [[nodiscard]] string plan() const {
    int const k0 = occ.k0;
    string s;
    for (int k : {k0 - 1, k0, k0 + 1}) { if (occ.canTry(k)) { s += (s.empty() ? "" : ", ") + to_string(k); } }
    if (occ.canTry(k0 + 2)) { s += (s.empty() ? "(" : " (") + to_string(k0 + 2) + ")"; }
    if (!s.empty()) { s += " blocks"; }
    s += s.empty() ? "a register limit" : ", then a register limit";
    return s;
  }
};

// A candidate as "R regs " or "K blocks", both 9 characters for lining up the log
string regValueName(int v) {
  char buf[32];
  snprintf(buf, sizeof(buf), v <= 16 ? "%2d blocks" : "%3d regs ", v);
  return buf;
}

} // namespace

TuneEntry Tune::regTuneEntry(const TuneEntry& e, int quick, const CudaSmLimits& sm) {
  Args* args = shared.args;
  u64 const exponent = primes.prevPrime(e.fft.maxExp());
  string const spec = e.fft.spec();

  // The line's settings other than the register limits and WMUL apply to every timing.  WMUL is decided here too (see below).
  vector<KeyVal> uses;
  for (const KeyVal& kv : e.uses) { if (!Args::isRegisterKey(kv.first) && kv.first != "WMUL") { uses.push_back(kv); } }

  // One timing with per-kernel profiling.  Empty if the GPU can't run it.  wmul, if given, receives the WMUL it ran with.
  auto timeKernels = [&](const vector<KeyVal>& regConf, u32* wmul = nullptr) -> vector<Gpu::RegTunable> {
    try {
      auto gpu = Gpu::make(exponent, shared, e.fft, regConf, false, uses);
      if (wmul) { *wmul = gpu->effectiveWmul(); }
      if (!std::isfinite(gpu->timePRP(quick))) { return {}; }
      return gpu->regTunables();
    } catch (const std::exception& ex) {
      log("%s failed: %s\n", spec.c_str(), ex.what());
    } catch (const string& mes) {
      log("%s failed: %s\n", spec.c_str(), mes.c_str());
    }
    return {};
  };

  log("\n");
  log("Tuning %s\n", spec.c_str());
  bool const oldProfile = args->profile;
  args->profile = true;
  vector<KernelSearch> searches;
  u32 wmul = 0;
  for (const Gpu::RegTunable& t : timeKernels({{"NOREG", "1"}}, &wmul)) {
    KernelSearch k{t, occupancy(t.res, sm), 0, {}};
    string const plan = k.occ.k0 ? k.plan() : "";
    log("%-18s%3d regs, %d threads, %d bytes shared:%6.1f us.  %s%s\n", t.kernelName.c_str(), t.res.regs, t.res.threads,
        t.res.sharedBytes, t.usPerCall, plan.empty() ? "Nothing to try" : "Will try ", plan.c_str());
    if (!plan.empty()) { searches.push_back(std::move(k)); }
  }

  // Each pass times every kernel's next candidate, those whose search is done at the compiler's default
  while (true) {
    vector<KeyVal> conf{{"NOREG", "0"}};
    vector<int> trying(searches.size());
    for (size_t i = 0; i < searches.size(); ++i) {
      trying[i] = searches[i].next();
      conf.emplace_back(searches[i].base.key, trying[i] ? to_string(trying[i]) : "-1");
    }
    if (std::ranges::all_of(trying, [](int v) { return v == 0; })) { break; }

    vector<Gpu::RegTunable> const times = timeKernels(conf);
    for (size_t i = 0; i < searches.size(); ++i) {
      if (!trying[i]) { continue; }
      for (const Gpu::RegTunable& t : times) {
        if (t.key != searches[i].base.key) { continue; }
        searches[i].us[trying[i]] = t.usPerCall;
        char got[32];
        snprintf(got, sizeof(got), "%3d regs%s:", t.res.regs, t.res.localBytes ? " (spills)" : "");
        log("%-18s Trying %s -> got %-19s %6.1f us\n", t.kernelName.c_str(), regValueName(trying[i]).c_str(), got, t.usPerCall);
      }
    }
  }
  // Each kernel's best: any improvement in its time wins.  0 is the compiler's default.
  vector<int> bestValue(searches.size(), 0);
  vector<double> bestUs(searches.size());
  for (size_t i = 0; i < searches.size(); ++i) {
    bestUs[i] = searches[i].base.usPerCall;
    for (auto [v, t] : searches[i].us) { if (t < bestUs[i]) { bestUs[i] = t; bestValue[i] = v; } }
  }

  // WMUL=1 halves carryFused's threads per block, so it can run an odd number of blocks per SM: an occupancy WMUL=2 can't have.
  // Try launch bounds of 2 * lbcf + 1 blocks with WMUL=1, lbcf being the best blocks per SM with WMUL=2.  A failure is remembered for
  // the rest of the tune, for the FFTs with the same carryFused (FFT type, width, width variant, carry) and the same lbcf: on the
  // TITAN V most fail, and this saves their pass.  A success is timed again for each FFT, the decision needs its per-kernel time.
  bool wmul1Wins = false;
  int wmul1Blocks = 0;
  double wmul1Us = 0;
  size_t const cf = std::ranges::find_if(searches, [](const KernelSearch& k) { return k.base.key.starts_with("REGCF"); }) - searches.begin();
  if (wmul == 2 && cf < searches.size()) {
    const KernelSearch& k = searches[cf];
    int const lbcf = k.bestBlocks();
    wmul1Blocks = 2 * lbcf + 1;
    CudaKernelResources half = k.base.res;
    half.threads /= 2;
    half.sharedBytes /= 2;
    Occupancy const occ1 = occupancy(half, sm);
    string const key = to_string(int(e.fft.shape.fft_type)) + ':' + to_string(e.fft.shape.width) + ':' + to_string(variant_W(e.fft.variant))
                       + ':' + to_string(int(e.fft.carry)) + ':' + to_string(lbcf);
    auto const known = wmul1Results.find(key);
    if (!occ1.k0 || wmul1Blocks > std::min(occ1.maxBlocks, 16) || occ1.regsFor(wmul1Blocks) < 24) {
      log("%-18s WMUL=1 can't run %d blocks per SM\n", k.base.kernelName.c_str(), wmul1Blocks);
    } else if (known != wmul1Results.end() && !known->second) {
      log("%-18s WMUL=1 with %d blocks was slower for an earlier FFT with this carryFused, not trying it\n", k.base.kernelName.c_str(), wmul1Blocks);
    } else {
      vector<KeyVal> conf{{"NOREG", "0"}, {"WMUL", "1"}};
      for (size_t i = 0; i < searches.size(); ++i) {
        conf.emplace_back(searches[i].base.key, to_string(i == cf ? wmul1Blocks : bestValue[i] ? bestValue[i] : -1));
      }
      for (const Gpu::RegTunable& t : timeKernels(conf)) {
        if (t.key != k.base.key) { continue; }
        wmul1Us = t.usPerCall;
        wmul1Wins = wmul1Us < bestUs[cf];
        char got[32];
        snprintf(got, sizeof(got), "%3d regs%s:", t.res.regs, t.res.localBytes ? " (spills)" : "");
        log("%-18s Trying %s -> got %-19s %6.1f us, WMUL=1\n", t.kernelName.c_str(), regValueName(wmul1Blocks).c_str(), got, t.usPerCall);
      }
      wmul1Results[key] = wmul1Wins;
    }
  }
  args->profile = oldProfile;

  // Each kernel's improved setting, with the saving per iteration its per-kernel time predicts
  struct Win {
    string kernelName;
    vector<KeyVal> settings;
    double saving;
  };
  vector<Win> wins;
  for (size_t i = 0; i < searches.size(); ++i) {
    const KernelSearch& k = searches[i];
    if (i == cf && wmul1Wins) {
      string const setting = k.base.key + '=' + to_string(wmul1Blocks) + ':';
      log("%-18s best is WMUL=1, %-12s%6.1f us vs %.1f us (%.1f%%)\n", k.base.kernelName.c_str(), setting.c_str(), wmul1Us,
          k.base.usPerCall, 100.0 * (wmul1Us / k.base.usPerCall - 1));
      wins.push_back({k.base.kernelName, {{"WMUL", "1"}, {k.base.key, to_string(wmul1Blocks)}}, k.base.usPerCall - wmul1Us});
    } else if (!bestValue[i]) {
      log("%-18s best is the compiler's default\n", k.base.kernelName.c_str());
    } else {
      string const setting = k.base.key + '=' + to_string(bestValue[i]) + ':';
      log("%-18s best is %-12s%6.1f us vs %.1f us (%.1f%%)\n", k.base.kernelName.c_str(), setting.c_str(), bestUs[i],
          k.base.usPerCall, 100.0 * (bestUs[i] / k.base.usPerCall - 1));
      wins.push_back({k.base.kernelName, {{k.base.key, to_string(bestValue[i])}}, k.base.usPerCall - bestUs[i]});
    }
  }

  // The settings for a timing: every kernel at the compiler's default, except for these wins
  auto confOf = [&](const vector<const Win*>& with) {
    map<string, string> m{{"NOREG", "0"}};
    for (const KernelSearch& k : searches) { m[k.base.key] = "-1"; }
    for (const Win* w : with) { for (const auto& [key, val] : w->settings) { m[key] = val; } }
    return vector<KeyVal>(m.begin(), m.end());
  };
  vector<const Win*> accepted;
  for (const Win& w : wins) { accepted.push_back(&w); }

  // tune.txt costs are whole iterations without profiling
  double baseCost = timeConfig(exponent, shared, e.fft, {{"NOREG", "1"}}, quick, nullptr, uses);
  double tunedCost = baseCost;

  // When the kernels of two queues overlap (MULTI_Q on an FFT with two or more number types), a kernel's time alone does not say
  // how it does next to the other queue's kernels: e.g. more registers can leave less room for them.  Add the kernels' settings
  // one at a time, largest predicted saving first, and keep each only if the whole iteration gets faster.
  int const multiQ = [&] { for (const KeyVal& kv : uses) { if (kv.first == "MULTI_Q") { return atoi(kv.second.c_str()); } } return args->value("MULTI_Q", 0); }();
  bool const overlap = multiQ && int(e.fft.FFT_FP64) + int(e.fft.FFT_FP32) + int(e.fft.NTT_GF31) + int(e.fft.NTT_GF61) >= 2;
  if (overlap && wins.size() >= 2 && std::isfinite(baseCost)) {
    log("%-18s The queues overlap (MULTI_Q): adding the kernels' settings one at a time\n", "");
    vector<const Win*> order = accepted;
    std::ranges::sort(order, [](const Win* a, const Win* b) { return a->saving > b->saving; });
    accepted.clear();
    for (const Win* w : order) {
      vector<const Win*> trial = accepted;
      trial.push_back(w);
      double const cost = timeConfig(exponent, shared, e.fft, confOf(trial), quick, nullptr, uses);
      bool const keep = cost < tunedCost;
      string settings;
      for (const auto& [key, val] : w->settings) { settings += (settings.empty() ? "" : ",") + key + '=' + val; }
      log("%-18s %-22s %.1f us/iter vs %.1f: %s\n", w->kernelName.c_str(), settings.c_str(), cost, tunedCost, keep ? "kept" : "dropped");
      if (keep) {
        accepted = std::move(trial);
        tunedCost = cost;
      }
    }
  } else if (!wins.empty()) {
    tunedCost = timeConfig(exponent, shared, e.fft, confOf(accepted), quick, nullptr, uses);
  }
  if (!std::isfinite(baseCost) && !std::isfinite(tunedCost)) {
    log("%s: timing failed, tune.txt entry left as it was\n", spec.c_str());
    return e;
  }

  vector<KeyVal> const tunedConf = confOf(accepted);
  vector<KeyVal> winners;
  double predicted = 0;               // the saving per iteration the per-kernel times predict
  for (const Win* w : accepted) {
    winners.insert(winners.end(), w->settings.begin(), w->settings.end());
    predicted += w->saving;
  }

  // A whole iteration timing varies about 1% from one Gpu to the next.  When the per-kernel times predict a clear saving that the
  // whole iteration doesn't show, time both again and decide on the averages.
  if (!winners.empty() && tunedCost >= baseCost && std::isfinite(tunedCost) && predicted > 0.005 * baseCost) {
    log("%-18s %.1f us/iter with the compiler's default registers, %.1f tuned, but the kernels predict %.1f%% less.  Timing again.\n",
        "", baseCost, tunedCost, 100.0 * predicted / baseCost);
    double const baseCost2 = timeConfig(exponent, shared, e.fft, {{"NOREG", "1"}}, quick, nullptr, uses);
    double const tunedCost2 = timeConfig(exponent, shared, e.fft, tunedConf, quick, nullptr, uses);
    if (std::isfinite(baseCost2) && std::isfinite(tunedCost2)) {
      baseCost = (baseCost + baseCost2) / 2;
      tunedCost = (tunedCost + tunedCost2) / 2;
    }
  }

  TuneEntry out{baseCost, e.fft, uses};
  if (tunedCost < baseCost) {
    out.cost = tunedCost;
    out.uses.insert(out.uses.end(), winners.begin(), winners.end());
  }
  string settings;
  for (const KeyVal& kv : out.uses) { if (Args::isRegisterKey(kv.first) || kv.first == "WMUL") { settings += ' ' + kv.first + '=' + kv.second; } }
  log("%-18s %.1f us/iter with the compiler's default registers, %.1f tuned.  tune.txt:%s\n", "Final result:", baseCost, tunedCost,
      settings.empty() ? " default registers" : settings.c_str());
  return out;
}
#endif
