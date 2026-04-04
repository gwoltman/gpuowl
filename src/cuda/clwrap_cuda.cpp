// CUDA Driver API implementation of OpenCL API functions.
// This replaces clwrap.cpp when building with the native CUDA backend,
// mapping all cl* calls to cu* equivalents via the CUDA Driver API.

#include "tinycuda.h"
#include "cudawrap.h"  // For NvrtcProgram::preprocessOpenCL and compile

#include <cstdio>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <unordered_set>
#ifdef __linux__
#include <unistd.h>
#endif

using namespace std;

// Track allocated cl_mem objects so clSetKernelArg can distinguish buffer args from scalars.
// In OpenCL, buffer args are passed as &memobj where memobj is cl_mem (a pointer to _cl_mem).
// We need to convert these to CUdeviceptr for CUDA kernel launch.
static unordered_set<cl_mem> g_allocatedBuffers;

// Global CUDA context — set once by clCreateContext, used to ensure current before CUDA calls
static CUcontext g_cudaContext = nullptr;

static void ensureContextCurrent() {
  if (g_cudaContext) {
    cuCtxSetCurrent(g_cudaContext);
  }
}

// Global state for CUDA initialization
static bool g_cudaInitialized = false;
static void ensureCudaInit() {
  if (!g_cudaInitialized) {
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr, "cuInit failed: %d\n", (int)err);
    }
    g_cudaInitialized = true;
  }
}

// Global device list (allocated once, never freed)
static vector<_cl_device_id> g_devices;
static bool g_devicesEnumerated = false;

static void enumerateDevices() {
  if (g_devicesEnumerated) return;
  ensureCudaInit();
  int count = 0;
  cuDeviceGetCount(&count);
  g_devices.resize(count);
  for (int i = 0; i < count; i++) {
    cuDeviceGet(&g_devices[i].dev, i);
  }
  g_devicesEnumerated = true;
}

// ---- OpenCL API implementations ----

extern "C" {

unsigned clGetPlatformIDs(unsigned num, cl_platform_id* platforms, unsigned* numRet) {
  // CUDA has no "platforms" concept — just return 1 dummy
  if (numRet) *numRet = 1;
  if (platforms && num >= 1) platforms[0] = nullptr;
  return CL_SUCCESS;
}

int clGetDeviceIDs(cl_platform_id, cl_device_type, unsigned num, cl_device_id* devices, unsigned* numRet) {
  enumerateDevices();
  unsigned n = g_devices.size();
  if (numRet) *numRet = n;
  if (devices) {
    for (unsigned i = 0; i < min(num, n); i++) {
      devices[i] = &g_devices[i];
    }
  }
  return n > 0 ? CL_SUCCESS : CL_DEVICE_NOT_FOUND;
}

cl_context clCreateContext(const intptr_t*, unsigned nDevices, const cl_device_id* devices,
                           void (*)(const char*, const void*, size_t, void*), void*, int* err) {
  if (!devices || nDevices == 0) { if (err) *err = CL_INVALID_DEVICE; return nullptr; }
  auto* ctx = new _cl_context;
  ctx->dev = devices[0]->dev;
#if CUDA_VERSION >= 13000
  CUctxCreateParams params{};
  CUresult r = cuCtxCreate_v4(&ctx->ctx, &params, 0, ctx->dev);
#else
  CUresult r = cuCtxCreate(&ctx->ctx, 0, ctx->dev);
#endif
  if (r != CUDA_SUCCESS) {
    delete ctx;
    if (err) *err = CL_OUT_OF_RESOURCES;
    return nullptr;
  }
  g_cudaContext = ctx->ctx;  // Track for ensureContextCurrent()

  // L2 persistence: no benefit measured for this workload.
  // cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 16 * 1024 * 1024);

  if (err) *err = CL_SUCCESS;
  return ctx;
}

int clReleaseContext(cl_context ctx) {
  if (ctx) {
    cuCtxDestroy(ctx->ctx);
    delete ctx;
  }
  return CL_SUCCESS;
}

int clReleaseProgram(cl_program p) {
  if (p) {
    // NOTE: Do NOT unload the module here. PRPLL's loadAux() gets a kernel from
    // the program, then releases the program. The kernel's CUfunction remains valid
    // only while the CUmodule is loaded. In OpenCL, clCreateKernel retains the
    // program. In our CUDA shim, we simply never unload modules — they persist for
    // the process lifetime. This is safe because PRPLL creates a fixed set of kernels
    // at startup and uses them until exit.
    // if (p->moduleLoaded) cuModuleUnload(p->module);
    delete p;
  }
  return CL_SUCCESS;
}

int clReleaseCommandQueue(cl_command_queue q) {
  if (q) {
    cuStreamDestroy(q->stream);
    delete q;
  }
  return CL_SUCCESS;
}

// ---- Program compilation (NVRTC) ----

cl_program clCreateProgramWithSource(cl_context ctx, unsigned count, const char** strings,
                                      const size_t* lengths, int* err) {
  auto* prog = new _cl_program;
  for (unsigned i = 0; i < count; i++) {
    if (lengths && lengths[i]) {
      prog->source.append(strings[i], lengths[i]);
    } else {
      prog->source.append(strings[i]);
    }
  }
  if (err) *err = CL_SUCCESS;
  return prog;
}

cl_program clCreateProgramWithBinary(cl_context ctx, unsigned nDevices, const cl_device_id*,
                                      const size_t* lengths, const unsigned char** binaries,
                                      int* binaryStatus, int* err) {
  // "Binary" in CUDA land = PTX string
  auto* prog = new _cl_program;
  if (lengths && binaries && lengths[0] > 0) {
    prog->ptx.assign((const char*)binaries[0], lengths[0]);
    prog->compiled = true;
    // Load the module (JIT-compile PTX to SASS)
    ensureContextCurrent();
    CUresult r = cuModuleLoadData(&prog->module, prog->ptx.c_str());
    if (r == CUDA_SUCCESS) {
      prog->moduleLoaded = true;
      if (binaryStatus) binaryStatus[0] = CL_SUCCESS;
    } else {
      fprintf(stderr, "cuModuleLoadData from cache failed: %d, PTX size=%zu\n", (int)r, lengths[0]);
      prog->compiled = false;
      if (binaryStatus) binaryStatus[0] = CL_INVALID_BINARY;
      if (err) { *err = CL_INVALID_BINARY; return prog; }
    }
  }
  if (err) *err = CL_SUCCESS;
  return prog;
}

// Build log storage (per-program)
static string g_lastBuildLog;

int clCompileProgram(cl_program prog, unsigned nDevices, const cl_device_id* devices, const char* options,
                     unsigned numHeaders, const cl_program* headers, const char* const* headerNames,
                     void (*)(cl_program, void*), void*) {
  if (!prog) return CL_INVALID_PROGRAM;

  // Get device arch for NVRTC
  CUdevice dev = devices ? devices[0]->dev : g_devices[0].dev;
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

  char archOpt[32];
  snprintf(archOpt, sizeof(archOpt), "--gpu-architecture=sm_%d%d", major, minor);

  // Parse OpenCL options string into NVRTC options
  // Convert OpenCL build options to NVRTC equivalents:
  //   -cl-std=CL2.0         → -std=c++17
  //   -cl-finite-math-only  → --fmad=true (enable FMA contraction, the safe subset)
  //   -Dfoo=bar             → -Dfoo=bar (pass through)
  vector<string> nvrtcOpts;
  nvrtcOpts.push_back(archOpt);
  nvrtcOpts.push_back("-default-device");
  nvrtcOpts.push_back("-std=c++17");
  nvrtcOpts.push_back("-w");  // Suppress NVRTC macro redefinition warnings

  // FMA contraction: OpenCL uses -cl-finite-math-only + #pragma OPENCL FP_CONTRACT ON
  // to allow the compiler to contract a*b+c into FMA instructions. NVRTC's --fmad=true
  // is the safe equivalent — it ONLY enables FMA contraction without the dangerous
  // parts of -use_fast_math (no flush-to-zero, no reduced-precision division/sqrt).
  // This is critical for FFT performance: every butterfly is multiply-add pairs.
  nvrtcOpts.push_back("--fmad=true");

  // NOTE: --restrict (all kernel pointers are __restrict__) was tested but causes GPU read
  // errors — some PRPLL kernels use in-place operations where in/out buffers alias.
  // Do NOT enable globally. The compiler still auto-uses __ldg() for const pointers on sm_35+.

  // Debug: dump full options string
  {
    static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
    static bool dumpedOpts = false;
    if (dumpPrefix && !dumpedOpts && options) {
      dumpedOpts = true;
      fprintf(stderr, "clCompileProgram options: [%s]\n", options);
      FILE* optLog = fopen("kernel_regs.log", "a");
      if (optLog) { fprintf(optLog, "clCompileProgram options: [%s]\n", options); fclose(optLog); }
    }
  }

  int maxregcount = 0;
  if (options) {
    istringstream iss(options);
    string tok;
    while (iss >> tok) {
      if (tok.substr(0, 2) == "-D") {
        // Fix AMD-only FFT variants for NVIDIA: variant_W=0 and variant_H=0 require
        // AMD builtins (__builtin_amdgcn_ds_bpermute etc). Replace with variant 2.
        // FFT_VARIANT is a 3-digit number WMH: e.g. 000, 101, 202
        if (tok.find("FFT_VARIANT=") != string::npos) {
          size_t eqPos = tok.find('=');
          string valStr = tok.substr(eqPos + 1);
          // Strip trailing 'u' suffix
          if (!valStr.empty() && valStr.back() == 'u') valStr.pop_back();
          int val = atoi(valStr.c_str());
          int vW = val / 100;
          int vM = (val % 100) / 10;
          int vH = val % 10;
          if (vW == 0) vW = 2;  // AMD BCAST → NVIDIA generic
          if (vH == 0) vH = 2;
          int newVal = vW * 100 + vM * 10 + vH;
          tok = "-DFFT_VARIANT=" + to_string(newVal) + "u";
        }
        nvrtcOpts.push_back(tok);
      } else if (tok == "-cl-finite-math-only" || tok == "-cl-fast-relaxed-math") {
        // FMA contraction already enabled above via --fmad=true.
        // Do NOT use -use_fast_math here — it enables flush-to-zero and
        // reduced-precision division/sqrt which breaks tailMul accuracy.
      } else if (tok.substr(0, 14) == "--maxrregcount") {
        nvrtcOpts.push_back(tok);
        maxregcount = atoi(tok.substr(15, 3).c_str());
      }
      // Skip other -cl-* options (not applicable to NVRTC)
    }
  }

  // Build NVRTC headers from the cl_program header array
  vector<pair<string, string>> nvrtcHeaders;

  // Add all OpenCL source headers
  for (unsigned i = 0; i < numHeaders; i++)  {
    // First header: opencl_compat.cuh (inject as virtual NVRTC header)
    if (i == 0) {
      nvrtcHeaders.push_back({"opencl_compat.cuh", headers[0]->source});
    }
    // Remaining headers need preprocessing
    else if (headers[i] && headerNames[i]) {
      // Preprocess OpenCL source for CUDA compatibility
      string processedSrc = NvrtcProgram::preprocessOpenCL(headers[i]->source);
      // Debug: verify KERNEL macro replacement
// I'm not sure what Sherpa was trying to print out here.  It prints out nothing useful.
//      {
//        static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
//        if (dumpPrefix && string(headerNames[i]) == "base.cl") {
//          auto pos = processedSrc.find("KERNEL");
//          if (pos != string::npos) {
//            string ctx = processedSrc.substr(pos > 20 ? pos-20 : 0, 120);
//            fprintf(stderr, "base.cl KERNEL context: [%s]\n", ctx.c_str());
//          }
//        }
//      }
      nvrtcHeaders.push_back({headerNames[i], processedSrc});
    }
  }

  // Preprocess the main source
  string processedSource = NvrtcProgram::preprocessOpenCL(prog->source);

  // Prepend opencl_compat.cuh include if not already there
  if (processedSource.find("opencl_compat.cuh") == string::npos) {
    processedSource = "#include \"opencl_compat.cuh\"\n" + processedSource;
  }

  // Debug: dump preprocessed source when PRPLL_DUMP_PTX is set
  {
    static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
    if (dumpPrefix) {
      static int srcCount = 0;
      char fname[512];
      snprintf(fname, sizeof(fname), "%s_src_%d.cu", dumpPrefix, srcCount++);
      FILE* f = fopen(fname, "w");
      if (f) {
        fwrite(processedSource.c_str(), 1, processedSource.size(), f);
        fclose(f);
        fprintf(stderr, "Source dumped to %s (%zu bytes)\n", fname, processedSource.size());
      }
    }
  }

  // Store preprocessed source for __launch_bounds__ parsing in clCreateKernel
  prog->preprocessedSource = processedSource;
  for (auto& [name, src] : nvrtcHeaders) {
    prog->preprocessedSource += "\n";
    prog->preprocessedSource += src;
  }

  // Debug: dump NVRTC options when dumping PTX
  {
    static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
    static bool dumpedOnce = false;
    if (dumpPrefix && !dumpedOnce) {
      dumpedOnce = true;
      fprintf(stderr, "NVRTC options (%zu):\n", nvrtcOpts.size());
      for (auto& o : nvrtcOpts) fprintf(stderr, "  %s\n", o.c_str());
    }
  }

  try {
    prog->ptx = NvrtcProgram::compile(processedSource, "prpll_kernel.cu", nvrtcOpts, nvrtcHeaders);
    prog->compiled = true;
    g_lastBuildLog.clear();
  } catch (const exception& e) {
    g_lastBuildLog = e.what();
    prog->compiled = false;
    fprintf(stderr, "NVRTC COMPILE FAILED: %s\n", e.what());
    // Dump the full preprocessed source for debugging
    {
      char fname[64];
      static int failCount = 0;
      snprintf(fname, sizeof(fname), "prpll_fail_%d.cu", failCount++);
      FILE* f = fopen(fname, "w");
      if (f) {
        fprintf(f, "// === FAILED Main source ===\n%s\n", processedSource.c_str());
        for (auto& [name, src] : nvrtcHeaders) {
          fprintf(f, "\n// === Header: %s (%zu bytes) ===\n%s\n", name.c_str(), src.size(), src.c_str());
        }
        fclose(f);
        fprintf(stderr, "Dumped failed source to %s\n", fname);
      }
    }
    return CL_COMPILE_PROGRAM_FAILURE;
  }

  // If --maxrregcount is set it seems nvrtc compile ignores the setting.  Instead modify the PTX and load the modified PTX.

  if (maxregcount) {
    string maxntidPattern = ".maxntid ";
    string maxnregPattern = ".maxnreg " + to_string(maxregcount) + "\n";
    for (size_t startpos = 0; ; ) {
      size_t pos = prog->ptx.find(maxntidPattern, startpos);
      if (pos == string::npos) break;
      prog->ptx.insert(pos, maxnregPattern);
      startpos = pos + 20;
    }
  }

  return CL_SUCCESS;
}

cl_program clLinkProgram(cl_context ctx, unsigned nDevices, const cl_device_id*,
                          const char* options, unsigned nProgs, const cl_program* progs,
                          void (*)(cl_program, void*), void*, int* err) {
  ensureContextCurrent();
  // In CUDA, compilation produces PTX directly — no separate link step needed.
  // Just load the PTX as a CUmodule.
  if (!progs || nProgs == 0 || !progs[0] || !progs[0]->compiled) {
    if (err) *err = CL_LINK_PROGRAM_FAILURE;
    return nullptr;
  }

  auto* linked = new _cl_program;
  linked->ptx = progs[0]->ptx;
  linked->compiled = true;
  // Carry preprocessed source through for KERNEL(N) parsing in clCreateKernel
  for (unsigned i = 0; i < nProgs; ++i) {
    if (progs[i] && !progs[i]->preprocessedSource.empty()) {
      linked->preprocessedSource += progs[i]->preprocessedSource;
      linked->preprocessedSource += "\n";
    }
  }

  // Use cuModuleLoadDataEx with error log to see JIT errors
  char jitErrorLog[8192] = {};
  char jitInfoLog[4096] = {};
  CUjit_option jitOpts[] = {
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER
  };
  void* jitOptVals[] = {
    (void*)(size_t)sizeof(jitErrorLog), (void*)jitErrorLog,
    (void*)(size_t)sizeof(jitInfoLog), (void*)jitInfoLog
  };
  CUresult r = cuModuleLoadDataEx(&linked->module, linked->ptx.c_str(), 4, jitOpts, jitOptVals);
  if (r != CUDA_SUCCESS) {
    const char* errName = nullptr;
    cuGetErrorName(r, &errName);
    fprintf(stderr, "cuModuleLoadData FAILED: %s (%d)\n", errName ? errName : "?", (int)r);
    if (jitErrorLog[0]) fprintf(stderr, "JIT error log: %s\n", jitErrorLog);
    if (jitInfoLog[0]) fprintf(stderr, "JIT info log: %s\n", jitInfoLog);
    // Dump first 2000 chars of PTX for debugging
    fprintf(stderr, "PTX size: %zu bytes\n", linked->ptx.size());
    // Dump full PTX to file
    {
      FILE* ptxFile = fopen("failed_ptx.ptx", "w");
      if (ptxFile) {
        fwrite(linked->ptx.c_str(), 1, linked->ptx.size(), ptxFile);
        fclose(ptxFile);
        fprintf(stderr, "Dumped failed PTX to failed_ptx.ptx\n");
      }
    }
    delete linked;
    if (err) *err = CL_LINK_PROGRAM_FAILURE;
    return nullptr;
  }
  linked->moduleLoaded = true;

  // Dump PTX to file when PRPLL_DUMP_PTX is set (e.g., PRPLL_DUMP_PTX=kernel)
  // Creates files like kernel_0.ptx, kernel_1.ptx, etc.
  {
    static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
    if (dumpPrefix) {
      static int ptxCount = 0;
      char fname[512];
      snprintf(fname, sizeof(fname), "%s_%d.ptx", dumpPrefix, ptxCount++);
      FILE* f = fopen(fname, "w");
      if (f) {
        fwrite(linked->ptx.c_str(), 1, linked->ptx.size(), f);
        fclose(f);
        fprintf(stderr, "PTX dumped to %s (%zu bytes)\n", fname, linked->ptx.size());
      }
    }
  }

  if (err) *err = CL_SUCCESS;
  return linked;
}

int clBuildProgram(cl_program prog, unsigned nDevices, const cl_device_id* devices,
                   const char* options, void (*)(cl_program, void*), void*) {
  // If the program was loaded from binary (cached PTX) and already has a module,
  // skip recompilation — the module is already JIT'd and ready.
  if (prog && prog->moduleLoaded) {
    return CL_SUCCESS;
  }

  // clBuildProgram = compile + link in one step
  int err = clCompileProgram(prog, nDevices, devices, options, 0, nullptr, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) return err;

  CUresult r = cuModuleLoadData(&prog->module, prog->ptx.c_str());
  if (r != CUDA_SUCCESS) return CL_BUILD_PROGRAM_FAILURE;
  prog->moduleLoaded = true;
  return CL_SUCCESS;
}

int clGetProgramBuildInfo(cl_program prog, cl_device_id, cl_program_build_info info,
                           size_t size, void* value, size_t* sizeRet) {
  if (info == CL_PROGRAM_BUILD_LOG) {
    size_t len = g_lastBuildLog.size() + 1;
    if (sizeRet) *sizeRet = len;
    if (value && size >= len) {
      memcpy(value, g_lastBuildLog.c_str(), len);
    }
  }
  return CL_SUCCESS;
}

int clGetProgramInfo(cl_program prog, cl_program_info info, size_t size, void* value, size_t* sizeRet) {
  if (!prog) return CL_INVALID_PROGRAM;
  if (info == CL_PROGRAM_BINARY_SIZES) {
    size_t ptxSize = prog->ptx.size();
    if (sizeRet) *sizeRet = sizeof(size_t);
    if (value && size >= sizeof(size_t)) memcpy(value, &ptxSize, sizeof(size_t));
  } else if (info == CL_PROGRAM_BINARIES) {
    if (sizeRet) *sizeRet = sizeof(unsigned char*);
    if (value && size >= sizeof(unsigned char*)) {
      unsigned char** ptrs = (unsigned char**)value;
      if (ptrs[0]) memcpy(ptrs[0], prog->ptx.data(), prog->ptx.size());
    }
  }
  return CL_SUCCESS;
}

// ---- Kernel ----

cl_kernel clCreateKernel(cl_program prog, const char* name, int* err) {
  ensureContextCurrent();
  if (!prog || !prog->moduleLoaded) {
    if (err) *err = CL_INVALID_PROGRAM;
    return nullptr;
  }
  auto* k = new _cl_kernel;
  k->name = name;
  k->parentModule = prog->module;
  CUresult r = cuModuleGetFunction(&k->func, prog->module, name);
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "cuModuleGetFunction('%s') failed: %d, moduleLoaded=%d, module=%p\n",
            name, (int)r, prog->moduleLoaded, (void*)prog->module);
    delete k;
    if (err) *err = CL_INVALID_KERNEL_NAME;
    return nullptr;
  }

  // Shared memory carveout: default adaptive carveout is optimal for mixed kernel workloads.

  // Log register and shared memory usage per kernel when PRPLL_DUMP_PTX is set
  {
    static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
    if (dumpPrefix) {
      int numRegs = 0, shmem = 0, localmem = 0, maxThreads = 0;
      cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, k->func);
      cuFuncGetAttribute(&shmem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, k->func);
      cuFuncGetAttribute(&localmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , k->func);
      cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, k->func);
      fprintf(stderr, "  %-25s: %3d regs, %5d shmem, %d localmem, maxThreads=%d\n", name, numRegs, shmem, localmem, maxThreads);
      // Also write to file since WSL2+CUDA swallows stderr
      FILE* regLog = fopen("kernel_regs.log", "a");
      if (regLog) { fprintf(regLog, "  %-25s: %3d regs, %5d shmem, %d localmem, maxThreads=%d\n", name, numRegs, shmem, localmem, maxThreads); fclose(regLog); }
    }
  }

  // Parse .maxntid from PTX to get __launch_bounds__ value.
  // PTX pattern: .visible .entry <name>(...)\n.maxntid N, 1, 1
  k->reqWorkGroupSize = 256; // fallback
  {
    const string& ptx = prog->ptx;
    string entryPattern = ".entry " + string(name) + "(";
    size_t pos = ptx.find(entryPattern);
    if (pos != string::npos) {
      // Found the kernel entry. Now find .maxntid before the next .entry or opening brace
      size_t searchEnd = ptx.find(".entry ", pos + 1);
      if (searchEnd == string::npos) searchEnd = ptx.size();
      string maxntidPattern = ".maxntid ";
      size_t mpos = ptx.find(maxntidPattern, pos);
      if (mpos != string::npos && mpos < searchEnd) {
        int val = atoi(ptx.c_str() + mpos + maxntidPattern.size());
        if (val > 0) {
          k->reqWorkGroupSize = val;
        }
      }
    }
  }

  if (err) *err = CL_SUCCESS;
  return k;
}

int clReleaseKernel(cl_kernel k) {
  delete k;
  return CL_SUCCESS;
}

int clSetKernelArg(cl_kernel k, unsigned pos, size_t size, const void* value) {
  if (!k) return CL_INVALID_KERNEL;

  // Detect cl_mem buffer arguments and convert to CUdeviceptr.
  // In OpenCL, buffer args are set with clSetKernelArg(k, i, sizeof(cl_mem), &memobj).
  // sizeof(cl_mem) == sizeof(void*) == 8 on 64-bit. The value at 'value' is a cl_mem pointer.
  // We need to store the CUdeviceptr (GPU address) instead of the cl_mem (host pointer).
  if (size == sizeof(cl_mem) && value) {
    cl_mem mem = *(cl_mem*)value;
    if (mem && g_allocatedBuffers.count(mem)) {
      CUdeviceptr devPtr = mem->ptr;
      k->setArg(pos, sizeof(CUdeviceptr), &devPtr);
      return CL_SUCCESS;
    }
    // NULL cl_mem → pass a null device pointer
    if (!mem) {
      CUdeviceptr devPtr = 0;
      k->setArg(pos, sizeof(CUdeviceptr), &devPtr);
      return CL_SUCCESS;
    }
  }

  k->setArg(pos, size, value);
  return CL_SUCCESS;
}

// ---- Buffer ----

cl_mem clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size, void* hostPtr, int* err) {
  auto* buf = new _cl_mem;
  buf->size = size;
  ensureContextCurrent();
  CUresult r = cuMemAlloc(&buf->ptr, size);
  if (r != CUDA_SUCCESS) {
    delete buf;
    if (err) *err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    return nullptr;
  }
  // Handle CL_MEM_COPY_HOST_PTR
  if ((flags & CL_MEM_COPY_HOST_PTR) && hostPtr) {
    cuMemcpyHtoD(buf->ptr, hostPtr, size);
  }
  g_allocatedBuffers.insert(buf);
  if (err) *err = CL_SUCCESS;
  return buf;
}

int clReleaseMemObject(cl_mem buf) {
  if (buf) {
    ensureContextCurrent();
    g_allocatedBuffers.erase(buf);
    cuMemFree(buf->ptr);
    delete buf;
  }
  return CL_SUCCESS;
}

// ---- Command Queue ----

cl_command_queue clCreateCommandQueueWithProperties(cl_context ctx, cl_device_id dev,
                                                     const cl_queue_properties* props, int* err) {
  auto* q = new _cl_command_queue;
  q->context = ctx;
  q->profiling = false;

  // Check for profiling flag
  if (props) {
    for (int i = 0; props[i]; i += 2) {
      if (props[i] == CL_QUEUE_PROPERTIES && (props[i+1] & CL_QUEUE_PROFILING_ENABLE)) {
        q->profiling = true;
      }
    }
  }

  // Make sure context is current
  cuCtxSetCurrent(ctx->ctx);
  CUresult r = cuStreamCreate(&q->stream, CU_STREAM_NON_BLOCKING);
  if (r != CUDA_SUCCESS) {
    delete q;
    if (err) *err = CL_OUT_OF_RESOURCES;
    return nullptr;
  }
  if (err) *err = CL_SUCCESS;
  return q;
}

// ---- Enqueue operations ----

int clEnqueueNDRangeKernel(cl_command_queue q, cl_kernel k, unsigned workDim,
                            const size_t* globalOffset, const size_t* globalSize,
                            const size_t* localSize, unsigned nWaits,
                            const cl_event* waits, cl_event* event) {
  if (!q || !k) return CL_INVALID_VALUE;
  ensureContextCurrent();

  size_t gs = globalSize[0];
  size_t ls = localSize ? localSize[0] : 256;
  size_t numBlocks = (gs + ls - 1) / ls;

  // Build args array
  void* argPtrs[_cl_kernel::MAX_ARGS];
  k->buildArgPointers(argPtrs);

  // Env-gated kernel profiling (PRPLL_PROFILE=1) — takes priority over event profiling
  static bool doProfile = (getenv("PRPLL_PROFILE") != nullptr);

  // Event handling (skipped when env profiler is active)
  if (!doProfile && event && q->profiling) {
    auto* ev = new _cl_event;
    cuEventCreate(&ev->start, CU_EVENT_DEFAULT);
    cuEventCreate(&ev->end, CU_EVENT_DEFAULT);
    ev->hasTimings = true;
    ev->commandType = CL_COMMAND_NDRANGE_KERNEL;
    cuEventRecord(ev->start, q->stream);
    CUresult r = cuLaunchKernel(k->func, numBlocks, 1, 1, ls, 1, 1, 0, q->stream, argPtrs, nullptr);
    cuEventRecord(ev->end, q->stream);
    *event = ev;
    return r == CUDA_SUCCESS ? CL_SUCCESS : CL_OUT_OF_RESOURCES;
  }
  if (doProfile) {
    static std::map<std::string, double> kTime;
    static std::map<std::string, int> kCount;
    static std::map<std::string, int> kRegs;
    static std::map<std::string, int> kShmem;
    static int totalLaunches = 0;
    static CUevent pStart = nullptr, pEnd = nullptr;
    if (!pStart) { cuEventCreate(&pStart, CU_EVENT_DEFAULT); cuEventCreate(&pEnd, CU_EVENT_DEFAULT); }

    cuEventRecord(pStart, q->stream);
    CUresult r = cuLaunchKernel(k->func, numBlocks, 1, 1, ls, 1, 1, 0, q->stream, argPtrs, nullptr);
    cuEventRecord(pEnd, q->stream);
    cuEventSynchronize(pEnd);
    float ms = 0;
    cuEventElapsedTime(&ms, pStart, pEnd);
    kTime[k->name] += ms;
    kCount[k->name]++;
    if (!kRegs.count(k->name)) {
      int regs = 0, shmem = 0;
      cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, k->func);
      cuFuncGetAttribute(&shmem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, k->func);
      kRegs[k->name] = regs;
      kShmem[k->name] = shmem;
    }
    totalLaunches++;

    if (totalLaunches % 10000 == 0) {
      fprintf(stderr, "\n=== NTT KERNEL PROFILE (%d launches) ===\n", totalLaunches);
      std::vector<std::pair<double, std::string>> sorted;
      double totalMs = 0;
      for (auto& [n, t] : kTime) { sorted.push_back({t, n}); totalMs += t; }
      std::sort(sorted.rbegin(), sorted.rend());
      for (auto& [t, n] : sorted) {
        fprintf(stderr, "  %6.1f ms (%5.1f%%) %5d calls  avg %.3f ms  regs=%d shmem=%d  %s\n",
                t, 100.0*t/totalMs, kCount[n], t/kCount[n], kRegs[n], kShmem[n], n.c_str());
      }
      fprintf(stderr, "  TOTAL: %.1f ms\n===\n\n", totalMs);
    }
    if (event) *event = nullptr;
    return r == CUDA_SUCCESS ? CL_SUCCESS : CL_OUT_OF_RESOURCES;
  }

  CUresult r = cuLaunchKernel(k->func, numBlocks, 1, 1, ls, 1, 1, 0, q->stream, argPtrs, nullptr);
  if (r != CUDA_SUCCESS) {
    const char* errName = nullptr;
    cuGetErrorName(r, &errName);
    fprintf(stderr, "cuLaunchKernel FAILED for '%s': %s (%d)\n", k->name.c_str(), errName ? errName : "?", (int)r);
  }

  if (event) *event = nullptr;
  return r == CUDA_SUCCESS ? CL_SUCCESS : CL_OUT_OF_RESOURCES;
}

int clEnqueueReadBuffer(cl_command_queue q, cl_mem buf, cl_bool blocking,
                         size_t offset, size_t size, void* ptr,
                         unsigned nWaits, const cl_event* waits, cl_event* event) {
  // Must use stream-ordered copy because the stream was created with CU_STREAM_NON_BLOCKING,
  // which means cuMemcpyDtoH (NULL stream) won't wait for pending kernels on this stream.
  CUresult r = cuMemcpyDtoHAsync(ptr, buf->ptr + offset, size, q->stream);
  if (r == CUDA_SUCCESS && blocking) {
    r = cuStreamSynchronize(q->stream);
  }
  if (event) *event = nullptr;
  return r == CUDA_SUCCESS ? CL_SUCCESS : CL_OUT_OF_RESOURCES;
}

int clEnqueueWriteBuffer(cl_command_queue q, cl_mem buf, cl_bool blocking,
                          size_t offset, size_t size, const void* ptr,
                          unsigned nWaits, const cl_event* waits, cl_event* event) {
  // Must use stream-ordered copy (same reason as clEnqueueReadBuffer above)
  CUresult r = cuMemcpyHtoDAsync(buf->ptr + offset, ptr, size, q->stream);
  if (r == CUDA_SUCCESS && blocking) {
    r = cuStreamSynchronize(q->stream);
  }
  if (event) *event = nullptr;
  return r == CUDA_SUCCESS ? CL_SUCCESS : CL_OUT_OF_RESOURCES;
}

int clEnqueueCopyBuffer(cl_command_queue q, cl_mem src, cl_mem dst,
                         size_t srcOffset, size_t dstOffset, size_t size,
                         unsigned nWaits, const cl_event* waits, cl_event* event) {
  CUresult r = cuMemcpyDtoDAsync(dst->ptr + dstOffset, src->ptr + srcOffset, size, q->stream);
  if (event) *event = nullptr;
  return r == CUDA_SUCCESS ? CL_SUCCESS : CL_OUT_OF_RESOURCES;
}

int clEnqueueFillBuffer(cl_command_queue q, cl_mem buf, const void* pattern,
                         size_t patternSize, size_t offset, size_t size,
                         unsigned nWaits, const cl_event* waits, cl_event* event) {
  CUresult r;
  if (patternSize == 1) {
    unsigned char val;
    memcpy(&val, pattern, 1);
    r = cuMemsetD8Async(buf->ptr + offset, val, size, q->stream);
  } else if (patternSize == 4) {
    unsigned int val;
    memcpy(&val, pattern, 4);
    r = cuMemsetD32Async(buf->ptr + offset, val, size / 4, q->stream);
  } else {
    // For other pattern sizes, fall back to memset 0 (common case is zero-fill)
    r = cuMemsetD8Async(buf->ptr + offset, 0, size, q->stream);
  }
  if (event) *event = nullptr;
  return r == CUDA_SUCCESS ? CL_SUCCESS : CL_OUT_OF_RESOURCES;
}

int clEnqueueMarkerWithWaitList(cl_command_queue q, unsigned nWaits,
                                 const cl_event* waits, cl_event* event) {
  if (event) {
    auto* ev = new _cl_event;
    cuEventCreate(&ev->end, CU_EVENT_DEFAULT);
    cuEventRecord(ev->end, q->stream);
    ev->commandType = CL_COMMAND_MARKER;
    *event = ev;
  }
  return CL_SUCCESS;
}

int clFlush(cl_command_queue q) {
  // CUDA streams auto-flush; no-op
  return CL_SUCCESS;
}

int clFinish(cl_command_queue q) {
  if (q) cuStreamSynchronize(q->stream);
  return CL_SUCCESS;
}

// ---- Events ----

int clReleaseEvent(cl_event ev) {
  delete ev;
  return CL_SUCCESS;
}

int clWaitForEvents(unsigned n, const cl_event* events) {
  for (unsigned i = 0; i < n; i++) {
    if (events[i] && events[i]->end) {
      cuEventSynchronize(events[i]->end);
    }
  }
  return CL_SUCCESS;
}

int clGetEventInfo(cl_event ev, cl_event_info info, size_t size, void* value, size_t* sizeRet) {
  if (!ev) return CL_INVALID_VALUE;
  if (info == CL_EVENT_COMMAND_EXECUTION_STATUS) {
    int status = CL_COMPLETE;
    if (ev->end) {
      CUresult r = cuEventQuery(ev->end);
      if (r == CUDA_ERROR_NOT_READY) status = CL_RUNNING;
    }
    if (sizeRet) *sizeRet = sizeof(int);
    if (value && size >= sizeof(int)) memcpy(value, &status, sizeof(int));
  } else if (info == CL_EVENT_COMMAND_TYPE) {
    u32 type = ev->commandType;
    if (sizeRet) *sizeRet = sizeof(u32);
    if (value && size >= sizeof(u32)) memcpy(value, &type, sizeof(u32));
  }
  return CL_SUCCESS;
}

int clGetEventProfilingInfo(cl_event ev, cl_profiling_info info, size_t size, void* value, size_t* sizeRet) {
  if (!ev || !ev->hasTimings) return CL_PROFILING_INFO_NOT_AVAILABLE;

  // CUDA events give elapsed time between two events, not absolute timestamps.
  // We fake absolute timestamps by using a base time.
  u64 timestamp = 0;
  if (info == CL_PROFILING_COMMAND_START || info == CL_PROFILING_COMMAND_SUBMIT ||
      info == CL_PROFILING_COMMAND_QUEUED) {
    timestamp = 0;  // Relative start
  } else if (info == CL_PROFILING_COMMAND_END || info == CL_PROFILING_COMMAND_COMPLETE) {
    float ms = 0;
    cuEventElapsedTime(&ms, ev->start, ev->end);
    timestamp = (u64)(ms * 1e6);  // Convert ms to ns
  }

  if (sizeRet) *sizeRet = sizeof(u64);
  if (value && size >= sizeof(u64)) memcpy(value, &timestamp, sizeof(u64));
  return CL_SUCCESS;
}

// ---- Device info ----

int clGetDeviceInfo(cl_device_id dev, cl_device_info info, size_t size, void* value, size_t* sizeRet) {
  if (!dev) return CL_INVALID_DEVICE;

  switch (info) {
  case CL_DEVICE_NAME: {
    char name[256];
    cuDeviceGetName(name, sizeof(name), dev->dev);
    size_t len = strlen(name) + 1;
    if (sizeRet) *sizeRet = len;
    if (value && size >= len) memcpy(value, name, len);
    break;
  }
  case CL_DEVICE_VENDOR_ID: {
    // Return NVIDIA vendor ID
    unsigned int vid = 0x10DE;
    if (sizeRet) *sizeRet = sizeof(vid);
    if (value && size >= sizeof(vid)) memcpy(value, &vid, sizeof(vid));
    break;
  }
  case CL_DEVICE_MAX_COMPUTE_UNITS: {
    int units = 0;
    cuDeviceGetAttribute(&units, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev->dev);
    unsigned int val = units;
    if (sizeRet) *sizeRet = sizeof(val);
    if (value && size >= sizeof(val)) memcpy(value, &val, sizeof(val));
    break;
  }
  case CL_DEVICE_MAX_CLOCK_FREQUENCY: {
    int mhz = 0;
    cuDeviceGetAttribute(&mhz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev->dev);
    unsigned int val = mhz / 1000;  // kHz to MHz
    if (sizeRet) *sizeRet = sizeof(val);
    if (value && size >= sizeof(val)) memcpy(value, &val, sizeof(val));
    break;
  }
  case CL_DEVICE_GLOBAL_MEM_SIZE: {
    size_t mem = 0;
    cuDeviceTotalMem(&mem, dev->dev);
    u64 val = mem;
    if (sizeRet) *sizeRet = sizeof(val);
    if (value && size >= sizeof(val)) memcpy(value, &val, sizeof(val));
    break;
  }
  case CL_DRIVER_VERSION:
  case CL_DEVICE_VERSION: {
    int ver = 0;
    cuDriverGetVersion(&ver);
    char verStr[64];
    snprintf(verStr, sizeof(verStr), "CUDA %d.%d", ver / 1000, (ver % 1000) / 10);
    size_t len = strlen(verStr) + 1;
    if (sizeRet) *sizeRet = len;
    if (value && size >= len) memcpy(value, verStr, len);
    break;
  }
  case CL_DEVICE_ERROR_CORRECTION_SUPPORT: {
    int ecc = 0;
    cuDeviceGetAttribute(&ecc, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, dev->dev);
    cl_bool val = ecc;
    if (sizeRet) *sizeRet = sizeof(val);
    if (value && size >= sizeof(val)) memcpy(value, &val, sizeof(val));
    break;
  }
  case CL_DEVICE_BUILT_IN_KERNELS: {
    const char* empty = "";
    if (sizeRet) *sizeRet = 1;
    if (value && size >= 1) memcpy(value, empty, 1);
    break;
  }
  case CL_DEVICE_BOARD_NAME_AMD:
  case CL_DEVICE_PCIE_ID_AMD:
  case CL_DEVICE_TOPOLOGY_AMD: {
    // AMD-specific queries — return failure
    return CL_INVALID_VALUE;
  }
  case CL_DEVICE_GLOBAL_FREE_MEMORY_AMD: {
    size_t freeMem = 0, totalMem = 0;
    cuMemGetInfo(&freeMem, &totalMem);
    // AMD returns in KB
    u64 freeKB = freeMem / 1024;
    if (sizeRet) *sizeRet = sizeof(freeKB);
    if (value && size >= sizeof(freeKB)) memcpy(value, &freeKB, sizeof(freeKB));
    break;
  }
  default:
    return CL_INVALID_VALUE;
  }
  return CL_SUCCESS;
}

int clGetPlatformInfo(cl_platform_id, cl_device_info info, size_t size, void* value, size_t* sizeRet) {
  if (info == CL_PLATFORM_VERSION) {
    const char* ver = "CUDA (via PRPLL CUDA backend)";
    size_t len = strlen(ver) + 1;
    if (sizeRet) *sizeRet = len;
    if (value && size >= len) memcpy(value, ver, len);
    return CL_SUCCESS;
  }
  return CL_INVALID_VALUE;
}

int clGetCommandQueueInfo(cl_command_queue q, cl_command_queue_info info,
                           size_t size, void* value, size_t* sizeRet) {
  if (info == CL_QUEUE_CONTEXT) {
    if (sizeRet) *sizeRet = sizeof(cl_context);
    if (value && size >= sizeof(cl_context)) memcpy(value, &q->context, sizeof(cl_context));
    return CL_SUCCESS;
  }
  return CL_INVALID_VALUE;
}

// ---- Kernel info ----

int clGetKernelInfo(cl_kernel k, cl_kernel_info info, size_t size, void* value, size_t* sizeRet) {
  if (!k) return CL_INVALID_KERNEL;
  if (info == CL_KERNEL_NUM_ARGS) {
    int n = k->numArgs;
    if (sizeRet) *sizeRet = sizeof(n);
    if (value && size >= sizeof(n)) memcpy(value, &n, sizeof(n));
  } else if (info == CL_KERNEL_ATTRIBUTES) {
    const char* empty = "";
    if (sizeRet) *sizeRet = 1;
    if (value && size >= 1) memcpy(value, empty, 1);
  }
  return CL_SUCCESS;
}

int clGetKernelArgInfo(cl_kernel k, unsigned pos, cl_kernel_arg_info info,
                        size_t size, void* value, size_t* sizeRet) {
  if (info == CL_KERNEL_ARG_NAME) {
    char name[32];
    snprintf(name, sizeof(name), "arg%u", pos);
    size_t len = strlen(name) + 1;
    if (sizeRet) *sizeRet = len;
    if (value && size >= len) memcpy(value, name, len);
  }
  return CL_SUCCESS;
}

int clGetKernelWorkGroupInfo(cl_kernel k, cl_device_id dev, cl_kernel_work_group_info info,
                              size_t size, void* value, size_t* sizeRet) {
  if (!k) return CL_INVALID_KERNEL;
  ensureContextCurrent();
  if (info == CL_KERNEL_COMPILE_WORK_GROUP_SIZE) {
    // Return the __launch_bounds__ value parsed from source during clCreateKernel.
    // This matches OpenCL's CL_KERNEL_COMPILE_WORK_GROUP_SIZE which returns reqd_work_group_size.
    // Previously we used CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK which returns the hardware max
    // based on register/shared memory usage — NOT the declared group size. This caused wrong
    // block sizes for every kernel (e.g., tailMul expected 64 threads but got 1024).
    int wgSize = k->reqWorkGroupSize > 0 ? k->reqWorkGroupSize : 256;
    size_t wgs[3] = { (size_t)wgSize, 1, 1 };
    if (sizeRet) *sizeRet = sizeof(wgs);
    if (value && size >= sizeof(wgs)) memcpy(value, wgs, sizeof(wgs));
  }
  return CL_SUCCESS;
}

// ---- SVM (not used but must exist) ----

void* clSVMAlloc(cl_context, cl_svm_mem_flags, size_t size, unsigned) {
  CUdeviceptr ptr;
  ensureContextCurrent();
  cuMemAlloc(&ptr, size);
  return (void*)(uintptr_t)ptr;
}

void clSVMFree(cl_context, void* ptr) {
  ensureContextCurrent();
  cuMemFree((CUdeviceptr)(uintptr_t)ptr);
}

int clSetKernelArgSVMPointer(cl_kernel k, unsigned pos, const void* ptr) {
  CUdeviceptr dp = (CUdeviceptr)(uintptr_t)ptr;
  k->setArg(pos, sizeof(dp), &dp);
  return CL_SUCCESS;
}

} // extern "C"

// C++ linkage — must be outside the extern "C" block above.

// Set L2 cache persistence for multiple read-only buffers on the given stream.
// Computes the minimum address span covering all buffers, then sets one access policy
// window with hitRatio sized so that only the actual buffer bytes get persisting treatment,
// not the gaps between non-contiguous allocations.
void cudaSetL2Persistent(cl_command_queue q, const std::vector<cl_mem>& buffers) {
  if (!q) return;

  // Find address span and total data size
  CUdeviceptr minAddr = ~(CUdeviceptr)0;
  CUdeviceptr maxAddr = 0;
  size_t totalDataBytes = 0;

  for (auto buf : buffers) {
    if (!buf || buf->size == 0) continue;
    CUdeviceptr lo = buf->ptr;
    CUdeviceptr hi = buf->ptr + buf->size;
    if (lo < minAddr) minAddr = lo;
    if (hi > maxAddr) maxAddr = hi;
    totalDataBytes += buf->size;
  }

  if (totalDataBytes == 0 || maxAddr <= minAddr) return;

  size_t spanBytes = (size_t)(maxAddr - minAddr);

  // Query the device's max access policy window size
  int maxWindowSize = 0;
  cuDeviceGetAttribute(&maxWindowSize, CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, 0);
  if (maxWindowSize > 0 && spanBytes > (size_t)maxWindowSize) {
    fprintf(stderr, "L2 persist: span %zuMB exceeds max window %dMB, clamping\n",
            spanBytes / (1024*1024), maxWindowSize / (1024*1024));
    spanBytes = maxWindowSize;
  }

  // hitRatio = actual data / window span. This way only the real buffer data gets
  // persisting treatment, and any gaps between allocations get streaming treatment.
  float hitRatio = (float)totalDataBytes / (float)spanBytes;
  if (hitRatio > 1.0f) hitRatio = 1.0f;

  CUstreamAttrValue attr;
  memset(&attr, 0, sizeof(attr));
  attr.accessPolicyWindow.base_ptr = (void*)(uintptr_t)minAddr;
  attr.accessPolicyWindow.num_bytes = spanBytes;
  attr.accessPolicyWindow.hitRatio = hitRatio;
  attr.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
  attr.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_STREAMING;

  CUresult r = cuStreamSetAttribute(q->stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr);
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "L2 persist: cuStreamSetAttribute failed (%d)\n", (int)r);
  } else {
    fprintf(stderr, "L2 persist: window %zuMB (%.1f%% hit ratio), %zuMB actual data, %zu buffers\n",
            spanBytes / (1024*1024), hitRatio * 100.0f, totalDataBytes / (1024*1024),
            buffers.size());
  }
}

