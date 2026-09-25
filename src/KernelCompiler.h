// Copyright (C) Mihai Preda

#pragma once

#include "clwrap.h"

#include <vector>
#include <string>
#include <future>
#include <map>
#include <set>

class Args;
class Context;

class KernelCompiler {
  std::string cacheDir;
  cl_context context;
  std::string linkArgs;
  std::string baseArgs;
  const bool useCache;
  const int verbose;
  const bool asmDump;   // -v 10: pick up the compiler's assembly (AMD OpenCL)

  std::vector<Program> clSources;
  std::vector<std::pair<std::string, std::string>> files;

  u64 contextHash{};

  // -v 10 assembly dump: some kernelNames (e.g. "carryFused", "carry") are compiled several times
  // under this same name with different defines (plain/-DROE=1/-DMUL3=1/...). The args of every
  // kernel declared under each kernelName, to tell those apart by the defines that differ; and the
  // .s names used so far.
  std::map<std::string, std::vector<std::string>> declaredArgs;
  mutable std::set<std::string> asmDumpNames;
  mutable bool asmMissingNoted = false;
  mutable bool asmCacheNoted = false;
  int maxWaves = 0;   // waves per SIMD, AMD; 0 when not known

  [[nodiscard]] Program build(const string& fileName, const string& args) const;
  [[nodiscard]] Program compile(const string& fileName, const string& kernelName, const string& args) const;
  [[nodiscard]] KernelHolder loadAux(const string& fileName, const string& kernelName, const string& args) const;

public:
  const cl_device_id deviceId;

  KernelCompiler(const Args& args, const Context* context, const string& clArgs);
  
  // Called by each Kernel on construction, before any is loaded (see declaredArgs).
  void declare(const string& kernelName, const string& args);

  [[nodiscard]] std::future<KernelHolder> load(const string& fileName, const string& kernelName, const string& args) const;
};
