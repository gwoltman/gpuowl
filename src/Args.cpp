// Copyright (C) Mihai Preda

#include "Args.h"
#include "File.h"
#include "clwrap.h"
#include "gpuid.h"
#include "Proof.h"
#include "version.h"

#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <cctype>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <charconv>

// This is a copy of the args.verbose level.  It allows the CUDA wrapper to access the value.
int prpll_verbose = 0;

int Args::value(const string& key, int valNotFound) const {
  auto it = flags.find(key);
  if (it == flags.end()) { return valNotFound; }
  return atoi(it->second.c_str());
}

string Args::mergeArgs(int argc, char **argv) {
  string ret;
  for (int i = 1; i < argc; ++i) {
    ret += argv[i];
    ret += " ";
  }
  return ret;
}

vector<KeyVal> Args::splitArgLine(const string& inputLine) {
  vector<KeyVal> ret;

  string prev;
  for (const string& s : split(inputLine, ' ')) {
    if (s.empty()) { continue; }

    if (prev.empty()) {
      if (s[0] != '-') {
        log("Args: expected '-' before '%s'\n", s.c_str());
        throw "Argument syntax";
      }

      prev = s;
    } else {
      // A token such as "-5" is a negative value for the preceding option (e.g. -od -5), not a new option.
      bool const isNegativeNumber = s[0] == '-' && s.size() > 1 && (isdigit((unsigned char) s[1]) || s[1] == '.');
      if (s[0] == '-' && !isNegativeNumber) {
        ret.push_back({prev, {}});
        prev = s;
      } else {
        ret.emplace_back(prev, s);
        prev.clear();
      }
    }
  }
  if (!prev.empty()) {
    assert(prev[0] == '-');
    ret.push_back({prev, {}});
  }
  return ret;
}

// Aliases for -use keys, accepted so a variant spelling doesn't silently do nothing. George has
// repeatedly told users on the forum to "Try -use NOASM" (no underscore); keep that working by
// mapping it to the real key NO_ASM before it reaches -use validation or the OpenCL -D defines.
static const std::map<string, string> useKeyAliases = {
  {"NOASM", "NO_ASM"},
};

// Splits a string of the form "Foo=bar,C,D=1" into key=value pairs, with value defaulting to "1".
vector<KeyVal> Args::splitUses(string ss) { // pass by value is intentional
  vector<KeyVal> ret;
  std::ranges::replace(ss, ',', ' ');
  std::istringstream iss{ss};
  vector<string> const uses{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
  for (const string &s : uses) {
    auto pos = s.find('=');
    string key = (pos == string::npos) ? s : s.substr(0, pos);
    string const val = (pos == string::npos) ? "1"s : s.substr(pos+1);
    if (auto it = useKeyAliases.find(key); it != useKeyAliases.end()) {
      log("-use %s taken as %s\n", key.c_str(), it->second.c_str());
      key = it->second;
    }
    ret.emplace_back(key, val);
  }
  return ret;
}

// Checks the comma separated -tune options up front: Tune::tune() silently skips anything it does not recognise,
// so a typo such as "maxexponent=" would otherwise tune the default exponent range for hours without a word.
static void checkTuneOptions(const string& options) {
  for (const string& s : split(options, ',')) {
    if (s.empty() || s == "noconfig" || s == "fp64" || s == "ntt" || s == "fp6431" || s == "nofp32" || s == "inplace") { continue; }
    auto pos = s.find('=');
    string const key = s.substr(0, pos);
    if (pos != string::npos && (key == "quick" || key == "minexp" || key == "maxexp")) {
      string const val = s.substr(pos + 1);
      u64 n = 0;
      auto [end, ec] = std::from_chars(val.data(), val.data() + val.size(), n);
      if (val.empty() || ec != std::errc{} || end != val.data() + val.size() || (key == "quick" && (n < 1 || n > 10))) {
        log("-tune %s expects %s (found '%s')\n", key.c_str(), key == "quick" ? "a value from 1 to 10" : "a whole number, e.g. 5000000000", val.c_str());
        throw "-tune option value";
      }
      continue;
    }
    log("-tune option '%s' not understood; valid options are noconfig, inplace, fp64, ntt, nofp32, fp6431, minexp=<val>, maxexp=<val>, quick=<val>\n", s.c_str());
    throw "-tune option";
  }
}

void Args::readConfig(const fs::path& path) {
  if (File file = File::openRead(path)) {
    file.allowUnterminatedLastLine();
    for (string line : file) {
      line = rstripNewline(line);
      parse(line);
    }
  }
}

u32 Args::getProofPow(u64 exponent) const {
  if (proofPow == -1) { return ProofSet::bestPower(exponent); }
  assert(proofPow >= 1);
  return proofPow;
}

string Args::tailDir() const { return fs::path{dir}.filename().string(); }

bool Args::hasFlag(const string& key) const { return flags.contains(key); }

void Args::printHelp() {
  printf(R"(
PRPLL is "PRobable Prime and Lucas-Lehmer Categorizer", AKA "Purple-cat"

PRPLL is an OpenCL/CUDA (GPU) program for primality testing Mersenne numbers (of the form 2^n - 1).

To check that OpenCL is installed correctly use the command "clinfo". If clinfo does not find any
devices or otherwise fails, this program will not run.

This program is tested on Linux/ROCm (AMD GPUs); it also runs on Windows and on Nvidia GPUs.

For information about Mersenne primes search see https://www.mersenne.org/

Run "prpll -h"; If this displays a list of OpenCL devices, it means that PRPLL is detecting the GPUs
and should be able to run.


Worktodo:
PRPLL keeps the active tasks in per-worker files worktodo-0.txt, worktodo-1.txt etc in the local directory.
These per-worker files are supplied from the global worktodo.txt file if -pool is used.
In turn the work files can be supplied through AutoPrimeNet, located at https://download.mersenne.ca/AutoPrimeNet

It is also possible to manually add exponents by adding lines of the form "PRP=118063003" to worktodo-<N>.txt


The configuration options listed below can be passed on the command line or can be put in a file
named "config.txt" in the prpll run directory.


-h                 : print general help, list of FFTs, list of devices
-info <fft>        : print detailed information about the given FFT; e.g. -info 1K:13:256
-dir <folder>      : specify local work directory (containing worktodo-<N>.txt, results-<N>.txt, config.txt,
                     gpuowl-<N>.log)
-pool <dir>        : specify a directory with the shared (pooled) worktodo.txt and config.txt
                     Multiple PRPLL instances, each in its own directory, can share a pool of assignments.
                     Results are still written locally, to results-<N>.txt in each instance's own directory.
-verbose           : print more log, useful for developers
-version           : print only the version and exit
-user <name>       : specify the mersenne.org user name (for result reporting)
-workers <N>       : specify the number of parallel PRP tests to run (default 1)

-fft <spec>        : specify FFT or FFTs to use:
                     - a specific configuration: 256:13:1K
                     - a FFT size: 6.5M
                     - a size range: 7M-8M
                     - a list: 256:13:1K,8M
                     See the list of FFTs at the end.

-od <value>        : Overdrive the FFT range (ROE, CARRY32 limits). This allows to use a lower FFT for a given
                     exponent (thus faster), but increases the risk of errors. The presence of errors is detected,
                     but the errors are nevertheless costly computationally and better avoided.
                     A <value> of 1 extends the range by 0.1%% (and this would be acceptable); a value of 10
                     extends the range by 1%% (and this would be quite too much WRT errors).

-block <value>     : PRP block size, one of: 1000, 500, 200. Default 1000.
-carry long|short  : force carry type. Short carry may be faster, but requires high bits/word.
-prp <exponent>    : run a single PRP test and exit, ignoring worktodo.txt
-ll <exponent>     : run a single LL test and exit, ignoring worktodo.txt
-verify <file>     : verify PRP-proof contained in <file>
-smallest          : work on smallest exponent in worktodo.txt rather than the first exponent in worktodo.txt    
-proof <power>     : generate proof of power <power> (default: optimal depending on exponent).
                     A lower power reduces disk space requirements but increases the verification cost.
                     A higher power increases disk usage a lot.
                     e.g. proof power 10 for a 120M exponent uses about %.0fGB of disk space.
-iters <N>         : run next PRP test for <N> iterations and exit.
-save <N>          : specify the number of savefiles to keep (default %u).
-noclean           : do not delete data after the test is complete.
-cache             : use binary kernel cache; useful with repeated use of -roeTune and -tune
-roe               : measure the Round-Off Error (Z) for more iterations (slow)
-time              : collect and print a per-kernel GPU timing profile
-log <N>           : log progress and checkpoint every <N> iterations (positive multiple of 1000; default 20000)

-use <define>      : comma separated list of defines for configuring openCL code, such as:
  -use FAST_BARRIER: on AMD Radeon VII and older AMD GPUs, use a faster barrier().  This option
                     may not work on Nvidia GPUs or on RDNA AMD GPUs where it produces errors
                     (which are nevertheless detected).
  -use NO_ASM      : do not use __asm() blocks (inline assembly); also accepted as NOASM
  -use TAIL_KERNELS=<val> : change how tailSquare and tailMul operate according to <val>:
                     0 = single wide, single kernel
                     1 = single wide, two kernels
                     2 = double wide, single kernel
                     3 = double wide, two kernels
  -use TAIL_TRIGS=<val> : change how tailSquare computes final trig values according to <val>:
                     2 = calculate from scratch, no memory read
                     1 = calculate using one complex multiply from cached memory and uncached memory
                     0 = read trig values from memory
  -use INPLACE=n   : Perform tranforms in-place.  Great if the reduced memory usage fits in the GPU's L2 cache.
                     0 = not in-place, 1 = nVidia friendly access pattern, 2 = AMD friendly access pattern.
  -use PAD=<val>   : insert pad bytes to possibly improve memory access patterns.  Val is number bytes to pad.
  -use MIDDLE_IN_LDS_TRANSPOSE=0|1  : Transpose values in local memory before writing to global memory
  -use MIDDLE_OUT_LDS_TRANSPOSE=0|1 : Transpose values in local memory before writing to global memory
  -use TABMUL_CHAIN=<val>: Controls how trig values are obtained in WIDTH and HEIGHT when FFT-spec is 1.
                     0 = Read one trig value and compute the next 3 or 7.
                     1 = All trig values are pre-computed and read from memmory.

  -use DEBUG       : enable asserts in OpenCL kernels (slow, developers)
  -use STATS=<val> : enable carry statistics collection & logging (developers), for the kernel according to <val>:
                     1 = CarryFused, 2 = CarryFusedMul, 4 = CarryA, 8 = CarryMul

-tune <options>    : Looks for best settings to include in config.txt.  Times many FFTs to find fastest one to test exponents -- written to tune.txt.
                     An -fft <spec> can be given on the command line to limit which FFTs are timed.
                     Options are not required.  If present, the options are a comma separated list from below.
                         noconfig     - Skip timings to find best config.txt settings.
                         inplace      - Skip timings for not-in-place FFTs and NTTs.  All nVidia GPUs seem to prefer in-place FFTs and NTTs.
                         fp64         - Tune for settings that affect FP64 FFTs.  Time FP64 FFTs for tune.txt.
                         ntt          - Tune for settings that affect integer NTTs.  Time integer NTTs for tune.txt.
                         nofp32       - Do not tune for settings that affect FP32 FFTs.  Some openCL compilers have trouble with FP32.
                         minexp=<val> - Time FFTs to find the best one for exponents greater than <val>.
                         maxexp=<val> - Time FFTs to find the best one for exponents less than <val>.
                         fp6431       - Time FP64+M31 FFTs for tune.txt.  Only GPUs with great FP64 performance will find this beneficial.
                         quick=<val>  - Use higher values for a quicker, potentially less accurate tune.  Val ranges from 1 to 10.
-device <N>        : select the GPU at position N in the list of devices
-uid    <UID>      : select the GPU with the given UID (on ROCm/AMDGPU, Linux)
-pci    <BDF>      : select the GPU with the given PCI BDF, e.g. "0c:00.0"

Device selection : use one of -uid <UID>, -pci <BDF>, -device <N>, see the list below

)", ProofSet::diskUsageGB(120000000, 10), nSavefiles);

  vector<cl_device_id> deviceIds = getAllDeviceIDs();
  if (!deviceIds.empty()) {
    printf(" N  : PCI BDF |   UID            |   Driver                 |    Device\n");
  }
  for (unsigned i = 0; i < deviceIds.size(); ++i) {
    cl_device_id id = deviceIds[i];
    string const bdf = getBdfFromDevice(id);
    printf("%2u  : %7s | %16s | %-24s | %s | %s\n",
           i,
           bdf.c_str(),
           getUidFromBdf(bdf).c_str(),
           getDriverVersion(id).c_str(),
           getDeviceName(id).c_str(),
           getBoardName(id).c_str()
           );

  }
  printf("\nFFT Configurations (specify with -fft <type>:<width>:<middle>:<height> from the set below):\n");

  vector<FFTShape> const configs = FFTShape::allShapes();
  for (auto [type, name] : {pair{FFT64, "FP64"}, {FFT3161, "M31+M61 NTT"}, {FFT3261, "FP32+M61"}, {FFT61, "M61 NTT"},
                            {FFT323161, "FP32+M31+M61"}, {FFT6431, "FP64+M31"}}) {
    printf("\nFFT type %d: %s\n"
           " Size   MaxExp   BPW    FFT\n", type, name);
    u32 activeSize = 0;
    float maxBpw = 0;
    string variants;
    auto flush = [&]() {
      if (variants.empty()) { return; }
      printf("%5s  %7.2fM  %.2f  %s\n",
             numberK(activeSize).c_str(),
             // activeSize * FFTShape::MIN_BPW / 1'000'000,
             activeSize * maxBpw / 1'000'000.0,
             maxBpw,
             variants.c_str());
      variants.clear();
    };
    for (const FFTShape& c : configs) {
      if (c.fft_type != type) continue;
      if (c.size() != activeSize) {
        flush();
        activeSize = c.size();
        maxBpw = 0;
      }
      maxBpw = max(maxBpw, c.maxBpw());
      if (!variants.empty()) { variants.push_back(','); }
      variants += c.spec();
    }
    flush();
  }
}

void Args::parse(const string& line) {
  if (line.empty() || line[0] == '#') { return; }

  if (line[0] == '!') {
    // conditional defines predicated on a FFT
    char fftBuf[32];
    char configBuf[256];
    if (sscanf(line.c_str(), "! %31s %255s", fftBuf, configBuf) != 2) {   // otherwise the buffers are uninitialised
      log("config line ignored (expected \"! <fft> <use-flags>\"): \"%s\"\n", line.c_str());
      return;
    }
    string const fft = fftBuf;
    string const config = configBuf;
    perFftConfig[fft] = splitUses(config);
    return;
  }

  if (!silent) { log("config: %s\n", line.c_str()); }

  auto args = splitArgLine(line);

  for (const auto& [key, s] : args) {
    // log("key '%s'\n", key.c_str());
    if (key == "-h" || key == "--help") {
      printHelp();
      throw "help";
    } if (key == "-version") {
      // Plain stdout, no log prefix: the flag exists for scripts and launchers
      // that record which build wrote a result (Task.cpp reports VERSION to
      // PrimeNet), so the one line must be the version and nothing else.
      printf("%s\n", (VERSION[0] == 'v') ? VERSION + 1 : VERSION);
      fflush(stdout);
      throw "version";
    } if (key == "-info") {
      if (s.empty()) {
        log("-info expects an FFT spec, e.g. -info 1K:13:256\n");
        throw "-info <fft>";
      }
      log(" FFT              | BPW   | Max exp (M)\n");
      for (const FFTShape& shape : FFTShape::multiSpec(s)) {
        for (u32 variant = 0; variant <= LAST_VARIANT; variant = next_variant (variant)) {
          if (variant != LAST_VARIANT && shape.fft_type != FFT64) continue;
          FFTConfig const fft{shape, variant, CARRY_AUTO};
          log("%12s | %.2f | %5.1f\n", fft.spec().c_str(), fft.maxBpw(), fft.maxExp() / 1'000'000.0);
        }
      }
      throw "info";
    } if (key == "-od") {
      double od = stod(s);
      fftOverdrive = 1 + od / 1000;
    } else if (key == "-roe") {
      assert(s.empty());
      logROE = true;
    } else if (key == "-tune") {
      doTune = true;
      if (!s.empty()) { checkTuneOptions(s); tune = s; }
//    } else if (key == "-ctune") {
//      doCtune = true;
//      if (!s.empty()) { ctune.push_back(s); }
    } else if (key == "-ztune") {
      doZtune = true;
    } else if (key == "-carryTune") {
      carryTune = true;
    } else if (key == "-verbose" || key == "-v") {
      if (s.empty()) verbose = 1;
      else verbose = stoi(s);
      prpll_verbose = verbose;
    } else if (key == "-time") {
      profile = true;
    } else if (key == "-workers") {
      if (s.empty()) {
        log("-workers expects <N>\n");
        throw "-workers <N>";
      }
      workers = stoi(s);
      if (workers < 1 || workers > 4) {
        throw "Number of workers must be between 1 and 4";
      }
    } else if (key == "-cache") {
      useCache = true;
    } else if (key == "-noclean") {
      clean = false;
    } else if (key == "-proof") {
      int power = 0;
      if (s.empty() || (power = stoi(s)) < 1 || power > 13) {
        log("-proof expects <power> 1-13 (found '%s')\n", s.c_str());
        throw "-proof <power>";
      }
      proofPow = power;
      assert(proofPow >= 1);
    } else if (key == "-keep") {
      if (s != "proof") {
        log("-keep requires 'proof'\n");
        throw "-keep without proof";
      }
      keepProof = true;
    } else if (key == "-verify") {
      if (s.empty()) {
        log("-verify needs <proof-file>\n");
        throw "-verify without proof-file";
      }
      verifyPath = s;
    }
    else if (key == "-pool") {
      masterDir = s;
      if (!masterDir.is_absolute()) {
        log("-pool <path> requires an absolute path\n");
        throw("-pool <path> requires an absolute path");
      }
    }
    else if (key == "-maxAlloc" || key == "-maxalloc") {                // DEPRECATED, was only used for P-1 buffers.  Parsing left in place so previous users do not get an error.
      if (s.empty()) {                                                  // s.back() below would be undefined
        log("-maxAlloc expects a value, e.g. -maxAlloc 4G\n");
        throw "-maxAlloc <size>";
      }
      u32 multiple = (s.back() == 'G') ? (1u << 30) : (1u << 20);
      maxAlloc = size_t(stod(s) * multiple + .5);
    }
    else if (key == "-iters") { iters = stoi(s); assert(iters > 0); }   // any positive count; release never enforced the old multiple-of-10000 rule
    else if (key == "-prp" || key == "-PRP") { prpExp = stoll(s); }
    else if (key == "-ll" || key == "-LL") { llExp = stoll(s); }
    else if (key == "-smallest") { smallest = true; }
    else if (key == "-fft") { fftSpec = s; }
    else if (key == "-user") { user = s; }
    else if (key == "-device" || key == "-d") { device = stoi(s); }
    else if (key == "-uid") { device = getPosFromUid(s); }
    else if (key == "-pci") { device = getPosFromBdf(s); }
    else if (key == "-dir") { dir = s; }
    else if (key == "-carry") {
      if (s == "short" || s == "long") {
        carry = s == "short" ? CARRY_32 : CARRY_64;
      } else {
        log("-carry expects short|long\n");
        throw "-carry expects short|long";
      }
    } else if (key == "-block") {
      blockSize = stoi(s);
      if (blockSize != 1000 && blockSize != 500 && blockSize != 200) {
        log("-block must be one of 1000, 500, 200\n");
        throw "invalid block size";
      }
    } else if (key == "-log") {
      logStep = stoi(s);
      if (logStep == 0 || logStep % 1000 != 0) {       // 0 would divide by zero in the PRP loop
        log("-log must be a positive multiple of 1000\n");
        throw "invalid log size";
      }
    } else if (key == "-use") {
      for (const auto& [key, val] : splitUses(s)) {
        auto it = flags.find(key);
        if (it != flags.end() && it->second != val) {
          log("warning: -use %s=%s overrides %s=%s\n", key.c_str(), val.c_str(), it->first.c_str(), it->second.c_str());
        }
        flags[key] = val;
      }
    } else if (key == "-unsafeMath") {                                  // DEPRECATED, not in -help.  The flag has not reached the compiler since 424a54e,
      safeMath = false;                                                 // and measured on gfx1100 -cl-unsafe-math-optimizations gives no speedup and a lower
                                                                        // roundoff margin (reassoc folds fancyMul's fma).  Parsing left in place so previous
                                                                        // users do not get an error; safeMath kept in case a developer wants to try it again.
    } else if (key == "-save") {
      int const n = stoi(s);
      if (n < 1) {                                     // 0 makes Saver::trimFiles index v[-1]
        log("-save must be at least 1\n");
        throw "invalid -save value";
      }
      nSavefiles = n;
    } else {
      log("Argument '%s' '%s' not understood\n", key.c_str(), s.c_str());
      throw "args";
    }
  }
}

void Args::setDefaults() {
  uid = getUidFromPos(device);
  cl_device_id dev = getDevice(device);
  log("device %d, OpenCL %s, %s, unique id '%s'\n", device, getDriverVersionByPos(device).c_str(),
      isAmdGpu(dev) ? getBoardName(dev).c_str() : getDeviceName(dev).c_str(), uid.c_str());
  
  if (!masterDir.empty()) {
    assert(masterDir.is_absolute());
    for (filesystem::path* p : {&proofResultDir, &proofToVerifyDir, &cacheDir}) {
      if (p->is_relative()) { *p = masterDir / *p; }
    }
  }

  for (auto& p : {proofResultDir, proofToVerifyDir, cacheDir}) { fs::create_directory(p); }
}
