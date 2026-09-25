// Copyright (C) Mihai Preda

#pragma once

#include "clwrap.h"

#include <vector>
#include <string>
#include <future>
#include <map>

class Args;
class Context;

class KernelCompiler {
  std::string cacheDir;
  cl_context context;
  std::string linkArgs;
  std::string baseArgs;
  std::string dump;
  const bool useCache;
  const int verbose;
  const bool asmDump;   // -v 10: pick up the compiler's assembly (AMD OpenCL)

  std::vector<Program> clSources;
  std::vector<std::pair<std::string, std::string>> files;

  u64 contextHash{};

  // -v 10 assembly dump: some kernelNames (e.g. "carryFused", "carry") are compiled several times
  // under this same name with different defines (plain/-DROE=1/-DMUL3=1/...). Counts how many .s
  // files have been written per kernelName so far, so each compile gets its own file instead of
  // each later variant overwriting the previous one's dump.
  mutable std::map<std::string, int> asmDumpCounts;

  [[nodiscard]] Program build(const string& fileName, const string& args) const;
  [[nodiscard]] Program compile(const string& fileName, const string& kernelName, const string& args) const;
  [[nodiscard]] KernelHolder loadAux(const string& fileName, const string& kernelName, const string& args) const;

public:
  const cl_device_id deviceId;

  KernelCompiler(const Args& args, const Context* context, const string& clArgs);
  
  [[nodiscard]] std::future<KernelHolder> load(const string& fileName, const string& kernelName, const string& args) const;
};
