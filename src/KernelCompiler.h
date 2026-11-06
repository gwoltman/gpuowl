// Copyright (C) Mihai Preda

#pragma once

#include <string_view>

#include "clwrap.h"

#include <vector>
#include <string>
#include <future>
#include <map>
#include <set>

class Args;
class Context;

// Whether an exception's message says that a kernel could not be built (compile error, missing kernel, workgroup too large):  the
// errors after which another FFT may still work
inline bool isBuildFailure(std::string_view mes) {
  return mes.starts_with("Can't compile") || mes.starts_with("OpenCL kernel ") || mes.starts_with("workgroup size too large");
}

class KernelCompiler {
  std::string cacheDir;
  cl_context context;
  std::string linkArgs;
  std::string baseArgs;
  const bool useCache;
  const bool useMemCache;   // -tune, -ztune, -carryTune: reuse this process's compiles (see memCache in KernelCompiler.cpp)
  const int verbose;
  const bool saveTemps; // The AMD OpenCL runtime writes -save-temps files into the current directory (AMD_OCL_BUILD_OPTIONS_APPEND or
                        // AMD_OCL_LINK_OPTIONS_APPEND has -save-temps, set by -v 10 or by the user): they are deleted after each compile.
  const bool asmDump;   // -v 10 or -v 11: pick up the compiler's assembly (AMD OpenCL). -v 10 reports only
                        // the "important" kernels (see isImportantKernel in KernelCompiler.cpp); -v 11 reports all of them.

  std::vector<Program> clSources;
  std::vector<const char*> clSourceNames;   // the names of clSources
  std::vector<std::pair<std::string, std::string>> files;

  u64 contextHash{};
  std::map<std::string, u64> ownHash;   // the hash of each pm1*.cl file, which is not in contextHash

  // -v 10 assembly dump: some kernelNames (e.g. "carryFused", "carry") are compiled several times
  // under this same name with different defines (plain/-DROE=1/-DMUL3=1/...). The args of every
  // kernel declared under each kernelName, to tell those apart by the defines that differ; and the
  // .s names used so far.
  std::map<std::string, std::vector<std::string>> declaredArgs;
  mutable std::set<std::string> asmDumpNames;
  mutable bool asmMissingNoted = false;
  mutable bool asmCacheNoted = false;
  int maxWaves = 0;   // waves per SIMD, AMD; 0 when not known
  // For the -v 10 LDS-occupancy check (AMD only; 0 when not known/available):
  int simdPerCU = 0;        // SIMDs per compute unit
  int wavefrontWidth = 0;   // threads per wavefront
  u64 ldsPerCU = 0;         // LDS bytes per compute unit (a workgroup's LDS is shared by the whole CU)

  [[nodiscard]] Program build(const string& fileName, const string& args) const;
  [[nodiscard]] Program compile(const string& fileName, const string& kernelName, const string& args) const;
  [[nodiscard]] KernelHolder loadAux(const string& fileName, const string& kernelName, const string& args) const;

public:
  const cl_device_id deviceId;

  // gcdBundle: compile the GCD kernels (src/cl/gcd/*.cl, see genbundle.sh) instead of the main ones.  The two bundles hash apart,
  // so the GCD kernels do not change the keys of the main kernels' cached binaries.
  KernelCompiler(const Args& args, const Context* context, const string& clArgs, bool gcdBundle = false);

  ~KernelCompiler();
  
  // Called by each Kernel on construction, before any is loaded (see declaredArgs).
  void declare(const string& kernelName, const string& args);

  [[nodiscard]] std::future<KernelHolder> load(const string& fileName, const string& kernelName, const string& args) const;
};
