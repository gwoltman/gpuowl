#include "KernelCompiler.h"
#include "Context.h"
#include "Sha3Hash.h"
#include "log.h"
#include "timeutil.h"
#include "Args.h"

#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <future>
#include <semaphore>
#include <thread>
#include <algorithm>
#include <utility>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <regex>
#include <set>

using namespace std;
namespace fs = std::filesystem;

// Implemented in bundle.cpp
const std::vector<const char*>& getClFileNames();
const std::vector<const char*>& getClFiles();

static_assert(sizeof(Program) == sizeof(cl_program));

// -cl-fast-relaxed-math  -cl-unsafe-math-optimizations -cl-denorms-are-zero -cl-mad-enable
// Other options:
// * -cl-uniform-work-group-size
// * -fno-bin-llvmir
// * various: -fno-bin-source -fno-bin-amdil

#ifndef CUDA_BACKEND
// Does the device's compiler accept this -cl-std?  Compiles an empty kernel with just that option.
static bool acceptsClStd(cl_context context, cl_device_id deviceId, const string& clStd) {
  Program probe = loadSource(context, "kernel void probe() {}\n");
  if (!probe) { return false; }
  string const opts = "-cl-std=" + clStd;
  return clCompileProgram(probe.get(), 1, &deviceId, opts.c_str(), 0, nullptr, nullptr, nullptr, nullptr) == CL_SUCCESS;
}
#endif

#ifndef CUDA_BACKEND
// -v 10 (see main()'s re-exec) makes the AMD OpenCL runtime write its -save-temps output into the current directory,
// whatever path the option names, under names of its own:
// * "<digits>.s/.cl/.i/.so" (e.g. "3949457118.s") from AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps=x on older ROCm;
// * "x_<N>_<gfx>.cl/.i" from that same option on current ROCm (7.x), which there yields no assembly;
// * "_temp_<N>_<gfx>.s/.so" and "_temp_<N>_<gfx>_linked.bc" from AMD_OCL_LINK_OPTIONS_APPEND=-save-temps-all (current ROCm);
// * "<digits>(_linked<digits>)+.s/.so/.bc" (e.g. "830107278_linked2226068292_linked32993898123623157245.s") on a ROCm
//   (verified: 6.3.3, gfx906) whose build stage honors the first option (constant name -- a hash of the literal "x") and
//   whose link stage(s) each chain their own "_linked<hash>" onto it, rather than either of the two schemes above.
// A name like "2026.log" or "42" may just as well be the user's own file, so a name is only taken for a compiler temp
// file when it also appeared (or changed) during the compile at hand: see snapshotDir() and newCompilerTemps().
static bool isCompilerTempName(const string& name) {
  string const stem = name.substr(0, name.find('.'));
  auto digitsEnd = [&stem](size_t pos) {
    while (pos < stem.size() && isdigit((unsigned char) stem[pos])) { ++pos; }
    return pos;
  };
  size_t pos = digitsEnd(0);
  if (pos > 0) {
    while (stem.compare(pos, 7, "_linked") == 0) {
      size_t const next = digitsEnd(pos + 7);
      if (next == pos + 7) { break; }
      pos = next;
    }
    if (pos == stem.size()) { return true; }
  }
  for (string const prefix : {"x_", "_temp_"}) {
    if (stem.starts_with(prefix)) {
      size_t const end = digitsEnd(prefix.size());
      if (end > prefix.size() && stem.compare(end, 4, "_gfx") == 0) { return true; }
    }
  }
  return false;
}

// The most waves a SIMD can hold, as LLVM's AMDGPU backend has it, from the device name ("gfx1100", "gfx90a:xnack-").
// 0 when not known.
static int maxWavesPerSimd(const string& deviceName) {
  if (!deviceName.starts_with("gfx")) { return 0; }
  string const gfx = deviceName.substr(0, deviceName.find(':'));
  if (gfx == "gfx90a" || gfx.starts_with("gfx94") || gfx.starts_with("gfx95")) { return 8; }   // CDNA2, CDNA3
  if (gfx.size() <= 6) { return 10; }                                                          // GCN, Vega, CDNA1
  if (gfx.starts_with("gfx101")) { return 20; }                                                // RDNA1
  return 16;                                                                                   // RDNA2 and later
}

using DirSnapshot = map<string, fs::file_time_type>;

// The entries of the current directory, with their modification times.
static DirSnapshot snapshotDir() {
  DirSnapshot snapshot;
  error_code ec;
  for (auto const& entry : fs::directory_iterator(".", ec)) {
    snapshot[entry.path().filename().string()] = entry.last_write_time(ec);
  }
  return snapshot;
}

// The compiler temp files that are new, or were rewritten, since the snapshot "before". Everything else in the
// directory -- the user's files, and the kernelName.s files saved by earlier compiles -- is left alone.
static vector<fs::path> newCompilerTemps(const DirSnapshot& before) {
  vector<fs::path> found;
  error_code ec;
  for (auto const& entry : fs::directory_iterator(".", ec)) {
    string const name = entry.path().filename().string();
    if (!entry.is_regular_file(ec) || !isCompilerTempName(name)) { continue; }
    auto it = before.find(name);
    if (it == before.end() || it->second != entry.last_write_time(ec)) { found.push_back(entry.path()); }
  }
  sort(found.begin(), found.end());
  return found;
}

static void removeFiles(const vector<fs::path>& files) {
  error_code ec;
  for (const fs::path& p : files) { fs::remove(p, ec); }
}
#endif

// In OpenCL C long long is a 128-bit type, so every "ull" literal (in the kernels and in the host's -D values) makes the
// arithmetic around it 128-bit.  Mesa rusticl translates the kernels to SPIR-V without optimizing that away, cannot handle
// 128-bit integers, and aborts the whole process ("InvalidBitWidth: Invalid bit width in input: 128").  Every such literal
// fits in 64 bits, so there the long long literals are narrowed to long, and base.cl/math.cl are told (NO_INT128) to avoid
// their 128-bit types too.
static string narrowLongLongLiterals(const string& s) {
  static const regex longLongLiteral{R"(\b(0[xX][0-9a-fA-F]+|[0-9]+)([uU]?)[lL][lL]\b)"};
  return regex_replace(s, longLongLiteral, "$1$2L");
}

KernelCompiler::KernelCompiler(const Args& args, const Context* context, const string& clArgs) :
  cacheDir{args.cacheDir.string()},
  context{context->get()},
  linkArgs{},   // no compile-only options here: clLinkProgram accepts only linker options (POCL enforces it)
  baseArgs{},
  useCache{args.useCache},
  verbose{args.verbose},
  asmDump{args.verbose >= 10 && getenv("PRPLL_ASM_REEXEC")},   // main() re-execs with the compiler options set only for AMD
  deviceId{context->deviceId()}
{
  // Every GPU driver we run on accepts -cl-std=CL2.0.  Some OpenCL 3.0 implementations (POCL; likely Mesa rusticl)
  // offer no OpenCL C 2.0 at all and reject the option, but do offer OpenCL C 3.0, whose optional features cover what
  // the kernels use from 2.0 (generic address space, memory-order atomics).  Probe once and fall back.
  string clStd = "CL2.0";
#ifndef CUDA_BACKEND
  // With -v 10 the probe compiles leave -save-temps files too; drop them.
  DirSnapshot const before = asmDump ? snapshotDir() : DirSnapshot{};
  if (!acceptsClStd(context->get(), deviceId, "CL2.0") && acceptsClStd(context->get(), deviceId, "CL3.0")) {
    clStd = "CL3.0";
    log("OpenCL C 2.0 is not available on this device; compiling the kernels as OpenCL C 3.0\n");
  }
  if (asmDump) { removeFiles(newCompilerTemps(before)); }
  if (isAmdGpu(deviceId)) { maxWaves = maxWavesPerSimd(getDeviceName(deviceId)); }
#endif
  baseArgs = "-cl-finite-math-only -cl-std=" + clStd + ' ' + clArgs;
  // Rusticl is detected, other drivers with the same limitation can be handled with -use NO_INT128=1
  bool const askedNoInt128 = clArgs.find("-DNO_INT128=1") != string::npos;
  bool const noInt128 = askedNoInt128 || isRusticl(deviceId);
  if (noInt128) { baseArgs = narrowLongLongLiterals(baseArgs) + (askedNoInt128 ? "" : " -DNO_INT128=1"); }

  string const hw = getDriverVersion(deviceId) + ':' + getDeviceName(deviceId);
  if (args.verbose) { log("OpenCL: %s, args %s\n", hw.c_str(), baseArgs.c_str()); }

  SHA3 hasher;
  hasher.update(hw);
  hasher.update(baseArgs);

  auto& clNames = getClFileNames();
  auto& clFiles = getClFiles();
  assert(clNames.size() == clFiles.size());
  int const n = int(clNames.size());
  for (int i = 0; i < n; ++i) {
    string const src = noInt128 ? narrowLongLongLiterals(clFiles[i]) : string(clFiles[i]);
    files.emplace_back(clNames[i], src);
    clSources.push_back(loadSource(context->get(), src));

    hasher.update(clNames[i]);
    hasher.update(src);
  }
  contextHash = std::move(hasher).finish()[0];
  // log("OpenCL %d files, hash %016" PRIx64 "\n", n, contextHash);
}

#ifndef CUDA_BACKEND
static string readWholeFile(const fs::path& p) {
  ifstream f(p, ios::binary);
  ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Resource-usage fields for one kernel, read out of its .amdhsa_kernel ... .end_amdhsa_kernel
// block in the assembly (AMDGPU/ROCm only -- see reference_rocm_isa_dumping memory for details
// on this format), and out of the "; Name: value" comments that follow that block.
// A CUDA-like -v report: registers, LDS, and whether it's spilling to scratch.
struct KernelStats {
  bool found = false;
  long vgprs = -1, vgprsAllocated = -1, sgprs = -1, ldsBytes = -1, scratchBytes = -1, occupancy = -1;
};

static KernelStats parseKernelStats(const string& asmText, const string& kernelName) {
  KernelStats st;
  // The whole program is in the file, so e.g. "tailMulZero" may precede "tailMul": the name must end there.
  string const startMarker = ".amdhsa_kernel " + kernelName;
  size_t start = asmText.find(startMarker);
  while (start != string::npos && start + startMarker.size() < asmText.size()
         && !isspace((unsigned char) asmText[start + startMarker.size()])) {
    start = asmText.find(startMarker, start + 1);
  }
  if (start == string::npos) { return st; }
  size_t end = asmText.find(".end_amdhsa_kernel", start);
  if (end == string::npos) { end = asmText.size(); }
  string const block = asmText.substr(start, end - start);
  st.found = true;

  // The comments land just after .end_amdhsa_kernel, outside the block above: take them from there
  // up to the next kernel (or end of file).
  size_t const nextKernel = asmText.find(".amdhsa_kernel", end);
  string const tail = asmText.substr(end, (nextKernel == string::npos) ? string::npos : nextKernel - end);

  auto grab = [](const string& text, const char* key) -> long {
    size_t const p = text.find(key);
    return (p == string::npos) ? -1 : strtol(text.c_str() + p + strlen(key), nullptr, 10);
  };
  // .amdhsa_next_free_vgpr is the allocation, which the compiler pads up to what the occupancy allows
  // (e.g. an LDS-bound kernel that uses 7 VGPRs gets 241); "; NumVgprs" is what the code uses.
  st.vgprs = grab(tail, "; NumVgprs:");
  st.vgprsAllocated = grab(block, ".amdhsa_next_free_vgpr");
  // "; TotalNumSgprs" counts VCC etc. too, as the code object's .sgpr_count does; .amdhsa_next_free_sgpr does not.
  st.sgprs = grab(tail, "; TotalNumSgprs:");
  if (st.sgprs < 0) { st.sgprs = grab(block, ".amdhsa_next_free_sgpr"); }
  st.ldsBytes = grab(block, ".amdhsa_group_segment_fixed_size");
  st.scratchBytes = grab(block, ".amdhsa_private_segment_fixed_size");
  st.occupancy = grab(tail, "; Occupancy:");
  return st;
}

// The options of args that are not common to all of siblings (the args of every kernel declared under the same
// kernelName, see declare()), e.g. "-DROE=1" for that carryFused variant; empty for a kernelName declared once.
static string distinguishingArgs(const string& args, const vector<string>& siblings) {
  string ret;
  istringstream in(args);
  for (string word; in >> word;) {
    bool const common = std::ranges::all_of(siblings, [&word](const string& other) {
      istringstream otherIn(other);
      for (string w; otherIn >> w;) { if (w == word) { return true; } }
      return false;
    });
    if (!common) { ret += (ret.empty() ? "" : " ") + word; }
  }
  return ret;
}

// "-DMUL3=1 -DROE=1" -> "MUL3_ROE", for a file name.
static string fileSuffix(const string& args) {
  string ret;
  istringstream in(args);
  for (string word; in >> word;) {
    if (word.starts_with("-D")) { word = word.substr(2); }
    if (word.ends_with("=1")) { word.resize(word.size() - 2); }
    for (char& c : word) { if (!isalnum((unsigned char) c) && c != '_') { c = '_'; } }
    ret += (ret.empty() ? "" : "_") + word;
  }
  return ret;
}
#endif

Program KernelCompiler::build(const string& fileName, const string& extraArgs) const {
  Program p1 = loadSource(context, "#include \""s + fileName + "\"\n");
  assert(p1);

  string args = baseArgs + ' ' + extraArgs;

#ifdef CUDA_BACKEND
  int err = clCompileProgram(p1.get(), 1, &deviceId, args.c_str(),
                             u32(clSources.size()), (const cl_program*) (clSources.data()), getClFileNames().data(),
                             nullptr, nullptr);
#else
  // Skip first file (opencl_compat.cuh) if this is a standard openCL application rather than a CUDA translation
  int err = clCompileProgram(p1.get(), 1, &deviceId, args.c_str(),
                             u32(clSources.size())-1, (const cl_program*) (clSources.data()+1), getClFileNames().data()+1,
                             nullptr, nullptr);
#endif
  if (string const mes = getBuildLog(p1.get(), deviceId); !mes.empty()) { log("%s\n", mes.c_str()); }
  if (err != CL_SUCCESS) {
    log("Compiling '%s' error %s (args %s)\n", fileName.c_str(), errMes(err).c_str(), args.c_str());
    return {};
  }

  Program p2{clLinkProgram(context, 1, &deviceId, linkArgs.c_str(),
                           1, (cl_program *) &p1, nullptr, nullptr, &err)};
  // The linker's diagnostics live on the linked program.  Asking p1 again instead says nothing about the link
  // -- and repeats the compile log that was already printed above.  A failed clLinkProgram may hand back no
  // program at all, and then there is nothing to query.
  if (p2) { if (string const mes = getBuildLog(p2.get(), deviceId); !mes.empty()) { log("%s\n", mes.c_str()); } }
  if (err != CL_SUCCESS) {
    log("Linking '%s' error %s (args %s)\n", fileName.c_str(), errMes(err).c_str(), linkArgs.c_str());
#ifndef CUDA_BACKEND
    if (err == CL_INVALID_LINKER_OPTIONS && asmDump) {
      // Most likely an older runtime rejecting the -save-temps-all that main() appends to every link for -v 10.
      // The runtime reads AMD_OCL_LINK_OPTIONS_APPEND once, at startup, so it cannot be dropped from here on.
      const char* const linkAppend = getenv("AMD_OCL_LINK_OPTIONS_APPEND");
      log("-v 10: the OpenCL runtime rejected AMD_OCL_LINK_OPTIONS_APPEND=\"%s\"; run without -v 10, or set "
          "AMD_OCL_LINK_OPTIONS_APPEND to an empty value to keep -v 10 without it\n", linkAppend ? linkAppend : "");
    }
#endif
    // clLinkProgram may still hand back a program object on failure (e.g. to hold the build log).
    // Discard it: an unlinked/half-linked program has no executable, and returning it here would
    // make the caller fail later with a bare CL_INVALID_PROGRAM_EXECUTABLE from clCreateKernel
    // instead of the "Can't compile" path that a compile failure takes just above.
    return {};
  }

  return p2;
}

Program KernelCompiler::compile(const string& fileName, [[maybe_unused]] const string& kernelName, const string& extraArgs) const {
#ifndef CUDA_BACKEND
  if (asmDump) {
    // Compiles are serial on this backend (see load() below), so what the compiler leaves in the directory
    // during this call belongs to this kernel.
    DirSnapshot const before = snapshotDir();
    Program program = build(fileName, extraArgs);
    vector<fs::path> const temps = newCompilerTemps(before);
    bool saved = false;
    for (const fs::path& p : temps) {
      if (p.extension() != ".s") { continue; }
      string const asmText = readWholeFile(p);

      KernelStats const st = parseKernelStats(asmText, kernelName);
      if (!st.found) { continue; }

      // Some kernelNames are compiled more than once under the same exported name, with different defines (e.g.
      // "carryFused" plain/-DROE=1/-DMUL3=1/...): tell them apart by those defines, in the log and in the file name.
      auto it = declaredArgs.find(kernelName);
      string const variant = (it == declaredArgs.end()) ? "" : distinguishingArgs(extraArgs, it->second);
      string const base = variant.empty() ? kernelName : kernelName + '_' + fileSuffix(variant);
      string outName = base + ".s";
      for (int n = 2; !asmDumpNames.insert(outName).second; ++n) { outName = base + '_' + to_string(n) + ".s"; }

      string const vgprs = st.vgprs < 0 ? to_string(st.vgprsAllocated) + " vgprs"
                                        : to_string(st.vgprs) + " vgprs (" + to_string(st.vgprsAllocated) + " allocated)";
      string const waves = to_string(st.occupancy) + (maxWaves ? "/" + to_string(maxWaves) : "") + " waves/SIMD";
      log("%s%s%s: %s, %ld sgprs, %ld bytes lds, %ld bytes scratch, occupancy %s%s -> %s\n",
          kernelName.c_str(), variant.empty() ? "" : " ", variant.c_str(), vgprs.c_str(), st.sgprs, st.ldsBytes,
          st.scratchBytes, waves.c_str(), st.scratchBytes > 0 ? " (SPILLING)" : "", outName.c_str());
      { ofstream out(outName, ios::binary); out << asmText; }
      saved = true;
      break;
    }
    if (program && !saved && !asmMissingNoted) {
      asmMissingNoted = true;
      log("-v 10: the OpenCL compiler left no assembly for %s (nor, likely, for the other kernels)\n", kernelName.c_str());
    }
    // The .s is copied out; its siblings (.cl, .i, .so, .bc) are of no use.
    removeFiles(temps);
    return program;
  }
#endif
  return build(fileName, extraArgs);
}

static string to_hex(u64 d) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%016" PRIx64, d);
  return buf;
}

KernelHolder KernelCompiler::loadAux(const string& fileName, const string& kernelName, const string& args) const {
  Timer const timer;
  bool fromCache = true;

  Program program;
  string cacheFile;

  if (useCache) {
    string const f = kernelName + '-' + to_hex(SHA3::hash(contextHash, fileName, kernelName, args)[0]);
    cacheFile = cacheDir + '/' + f;
    program = loadBinary(context, deviceId, cacheFile);
  }

  if (!program) {
    fromCache = false;
    program = compile(fileName, kernelName, args);
  } else if (asmDump && !asmCacheNoted) {
    asmCacheNoted = true;
    log("-v 10: %s and possibly more kernels come from the cache '%s' and so have no assembly; "
        "run with an empty cache, or without -cache, to see them\n", kernelName.c_str(), cacheDir.c_str());
  }

  if (!program) {
    log("Can't compile %s\n", fileName.c_str());
    throw "Can't compile " + fileName;
  }

  KernelHolder ret{loadKernel(program.get(), kernelName.c_str())};
  if (!ret) {
    log("Can't find %s in %s\n", kernelName.c_str(), fileName.c_str());
    throw "Can't find "s + kernelName + " in " + fileName;
  }

  if (!fromCache) {
    if (useCache) {
      if (verbose) { log("saving binary to '%s'\n", cacheFile.c_str()); }
      saveBinary(program.get(), cacheFile);
    }
    if (verbose) { log("Loaded %s %s: %.0fms\n", kernelName.c_str(), args.c_str(), timer.at() * 1000); }
  }

  return ret;
}

void KernelCompiler::declare(const string& kernelName, const string& args) { declaredArgs[kernelName].push_back(args); }

std::future<KernelHolder> KernelCompiler::load(const string& fileName, const string& kernelName, const string& args) const {
#ifdef CUDA_BACKEND
  // NVRTC compiles independently per thread, so the ~36 kernels of a Gpu
  // compile in parallel — bounded to the core count and to eight: each
  // NVRTC instance holds a few hundred MB while it runs, so 36 at once, or
  // one per thread of a 32-thread machine, would take gigabytes of host
  // memory for a speedup that eight threads over 36 kernels already give
  // most of. The CUDA shim makes the context current per thread and guards
  // its shared module counts. The thread logs through this worker's log
  // file and context (LogLink): the log is thread-local, and a fresh thread
  // would otherwise print its "Loaded" lines to stdout alone, unprefixed.
  static std::counting_semaphore<8> slots{std::max(1u, std::min(8u, std::thread::hardware_concurrency()))};
  LogLink const link = logLink();
  return async(std::launch::async, [this, fileName, kernelName, args, link] {
    slots.acquire();
    struct Release { std::counting_semaphore<8>& s; ~Release() { s.release(); } } release{slots};
    LogLinkScope const logScope{link};
    return loadAux(fileName, kernelName, args);
  });
#else
  // Serial: the ROCm compiler serializes parallel builds anyway (no benefit
  // measured), and the OpenCL runtime's thread-safety for concurrent
  // clCompileProgram varies by vendor.
  std::promise<KernelHolder> promise;
  promise.set_value(loadAux(fileName, kernelName, args));
  return promise.get_future();
#endif
}
