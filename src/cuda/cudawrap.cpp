// CUDA Driver API wrappers implementation

#include "cudawrap.h"

#include <stdexcept>
#include <sstream>
#include <cstdio>
#include <cstdarg>
#include <cstring>

// Local log function — avoids dependency on PRPLL's log.h/File.h chain
static void cuda_log(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
}

// ---- Error checking ----

void checkCuda(CUresult err, const char* file, int line, const char* func, const char* expr) {
  if (err != CUDA_SUCCESS) {
    const char* errName = nullptr;
    const char* errStr = nullptr;
    cuGetErrorName(err, &errName);
    cuGetErrorString(err, &errStr);
    char buf[512];
    snprintf(buf, sizeof(buf), "CUDA error %d (%s): %s at %s:%d in %s: %s",
             (int)err, errName ? errName : "?", errStr ? errStr : "?",
             file, line, func, expr);
    cuda_log("%s\n", buf);
    throw std::runtime_error(buf);
  }
}

void checkNvrtc(nvrtcResult err, const char* file, int line, const char* func, const char* expr) {
  if (err != NVRTC_SUCCESS) {
    char buf[512];
    snprintf(buf, sizeof(buf), "NVRTC error %d (%s) at %s:%d in %s: %s",
             (int)err, nvrtcGetErrorString(err), file, line, func, expr);
    cuda_log("%s\n", buf);
    throw std::runtime_error(buf);
  }
}

// ---- Device management ----

std::vector<CUdevice> getAllDevices() {
  CU_CHECK(cuInit(0));
  int count = 0;
  CU_CHECK(cuDeviceGetCount(&count));
  std::vector<CUdevice> devices(count);
  for (int i = 0; i < count; ++i) {
    CU_CHECK(cuDeviceGet(&devices[i], i));
  }
  return devices;
}

std::string getDeviceName(CUdevice dev) {
  char name[256];
  CU_CHECK(cuDeviceGetName(name, sizeof(name), dev));
  return name;
}

std::string getDriverVersion() {
  int ver = 0;
  CU_CHECK(cuDriverGetVersion(&ver));
  char buf[32];
  snprintf(buf, sizeof(buf), "%d.%d", ver / 1000, (ver % 1000) / 10);
  return buf;
}

float getGpuRamGB(CUdevice dev) {
  size_t bytes = 0;
  CU_CHECK(cuDeviceTotalMem(&bytes, dev));
  return bytes / (1024.0f * 1024.0f * 1024.0f);
}

u64 getFreeMem(CUdevice dev) {
  // Need a context to query free memory
  size_t free_bytes = 0, total = 0;
  CU_CHECK(cuMemGetInfo(&free_bytes, &total));
  return free_bytes;
}

std::string getShortInfo(CUdevice dev) {
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
  char buf[256];
  snprintf(buf, sizeof(buf), "%s (sm_%d%d, %.1f GB)", getDeviceName(dev).c_str(),
           major, minor, getGpuRamGB(dev));
  return buf;
}

// ---- Context ----

CudaContext::CudaContext(CUdevice dev) : device(dev) {
#if CUDA_VERSION >= 13000
  CUctxCreateParams params{};
  CU_CHECK(cuCtxCreate_v4(&ctx, &params, CU_CTX_SCHED_BLOCKING_SYNC, dev));
#else
  CU_CHECK(cuCtxCreate(&ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev));
#endif
}

CudaContext::~CudaContext() {
  if (ctx) cuCtxDestroy(ctx);
}

void CudaContext::makeCurrent() {
  CU_CHECK(cuCtxSetCurrent(ctx));
}

// ---- Module ----

CudaModule::CudaModule(const std::string& ptx, const std::string& name) {
  CUjit_option options[] = { CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER };
  char errorLog[4096] = {};
  void* optionValues[] = { (void*)(size_t)sizeof(errorLog), (void*)errorLog };

  CUresult err = cuModuleLoadDataEx(&module, ptx.c_str(), 2, options, optionValues);
  if (err != CUDA_SUCCESS) {
    cuda_log("Module load error for '%s': %s\n", name.c_str(), errorLog);
    checkCuda(err, __FILE__, __LINE__, __func__, "cuModuleLoadDataEx");
  }
}

CudaModule::~CudaModule() {
  if (module) cuModuleUnload(module);
}

CUfunction CudaModule::getFunction(const char* name) const {
  CUfunction func{};
  CU_CHECK(cuModuleGetFunction(&func, module, name));
  return func;
}

// ---- Stream ----

CudaStream::CudaStream() {
  CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
}

CudaStream::~CudaStream() {
  if (stream) cuStreamDestroy(stream);
}

void CudaStream::sync() {
  CU_CHECK(cuStreamSynchronize(stream));
}

// ---- Buffer ----

CudaBuffer::CudaBuffer(size_t bytes) : bytes(bytes) {
  if (bytes > 0) {
    CU_CHECK(cuMemAlloc(&ptr, bytes));
  }
}

CudaBuffer::~CudaBuffer() {
  if (ptr) cuMemFree(ptr);
}

void CudaBuffer::readSync(void* dst, size_t n) const {
  assert(n <= bytes);
  CU_CHECK(cuMemcpyDtoH(dst, ptr, n));
}

void CudaBuffer::writeSync(const void* src, size_t n) {
  assert(n <= bytes);
  CU_CHECK(cuMemcpyHtoD(ptr, src, n));
}

void CudaBuffer::zero() {
  if (bytes > 0) {
    CU_CHECK(cuMemsetD8(ptr, 0, bytes));
  }
}

void CudaBuffer::copyFrom(const CudaBuffer& src) {
  assert(bytes == src.bytes);
  CU_CHECK(cuMemcpyDtoD(ptr, src.ptr, bytes));
}

void CudaBuffer::fillPattern(const void* pattern, size_t patternSize) {
  if (patternSize == 4) {
    u32 val;
    memcpy(&val, pattern, 4);
    CU_CHECK(cuMemsetD32(ptr, val, bytes / 4));
  } else if (patternSize == 1) {
    u8 val;
    memcpy(&val, pattern, 1);
    CU_CHECK(cuMemsetD8(ptr, val, bytes));
  } else {
    // For other sizes, use cuMemsetD32 with multiple passes or copy pattern manually
    // This is rarely used
    assert(false && "fillPattern: unsupported pattern size");
  }
}

// ---- NVRTC Compilation ----

// Helper: skip balanced parentheses starting at '(' at position i
static size_t skipBalancedParens(const std::string& s, size_t i) {
  if (i >= s.size() || s[i] != '(') return i;
  int depth = 1;
  i++;
  while (i < s.size() && depth > 0) {
    if (s[i] == '(') depth++;
    else if (s[i] == ')') depth--;
    i++;
  }
  return i;
}

// Strip OpenCL-specific constructs that NVRTC can't handle:
// 1. __global/global pointer qualifiers (without breaking __global__)
// 2. #pragma OPENCL ... directives
// 3. __attribute__((overloadable)) and __attribute__((reqd_work_group_size(...)))
// 4. OpenCL vector cast syntax: (type2)(a, b) → make_type2(a, b) [deferred to compat header]
std::string NvrtcProgram::preprocessOpenCL(const std::string& source) {
  std::string result;
  result.reserve(source.size());
  size_t i = 0;
  while (i < source.size()) {
    // Strip #pragma OPENCL ... lines
    if (i + 14 <= source.size() && source.compare(i, 14, "#pragma OPENCL") == 0) {
      bool atLineStart = (i == 0 || source[i-1] == '\n');
      if (atLineStart) {
        while (i < source.size() && source[i] != '\n') i++;
        result += "// [stripped pragma]";
        continue;
      }
    }

    // Replace base.cl's KERNEL macro with CUDA-compatible version
    // OpenCL: #define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void
    // CUDA:   #define KERNEL(x) extern "C" __global__ void __launch_bounds__(x)
    if (i + 15 <= source.size() && source.compare(i, 15, "#define KERNEL(") == 0) {
      bool atLineStart = (i == 0 || source[i-1] == '\n');
      if (atLineStart) {
        while (i < source.size() && source[i] != '\n') i++;
        result += "#ifdef CUDA_MIN_BLOCKS\n";
        result += "#define KERNEL(x) extern \"C\" __global__ void __launch_bounds__(x, CUDA_MIN_BLOCKS)\n";
        result += "#else\n";
        result += "#define KERNEL(x) extern \"C\" __global__ void __launch_bounds__(x)\n";
        result += "#endif";
        continue;
      }
    }

    // Strip base.cl's OpenCL typedefs that conflict with opencl_compat.cuh
    // Only strip exact OpenCL patterns (using OpenCL types like 'long', 'ulong', 'uint')
    // NOT our compat header's versions (which use 'long long', 'unsigned long long', etc.)
    if (i + 7 <= source.size() && source.compare(i, 7, "typedef") == 0) {
      bool atLineStart = (i == 0 || source[i-1] == '\n');
      if (atLineStart) {
        size_t lineEnd = source.find('\n', i);
        if (lineEnd == std::string::npos) lineEnd = source.size();
        std::string line = source.substr(i, lineEnd - i);
        // Strip trailing whitespace/CR for comparison
        while (!line.empty() && (line.back() == ' ' || line.back() == '\r')) line.pop_back();
        // Only strip the EXACT OpenCL patterns from base.cl:
        if (line == "typedef int i32;" ||
            line == "typedef uint u32;" ||
            line == "typedef long i64;" ||
            line == "typedef ulong u64;") {
          result += "// [stripped OpenCL typedef]";
          i = lineEnd;
          continue;
        }
      }
    }

    // Convert vector array access: .a[0] → .a.x, .a[1] → .a.y
    // OpenCL allows vec2[i] indexing; CUDA uses .x/.y member access.
    // Used in fftp.cl: union { uint2 a; u64 b; } m31_combo; #define frac_bits m31_combo.a[0]
    if (i + 5 <= source.size() && source.compare(i, 5, ".a[0]") == 0) {
      result += ".a.x";
      i += 5;
      continue;
    }
    if (i + 5 <= source.size() && source.compare(i, 5, ".a[1]") == 0) {
      result += ".a.y";
      i += 5;
      continue;
    }

    // Handle __attribute__((...)) for overloadable and reqd_work_group_size
    if (i + 15 <= source.size() && source.compare(i, 15, "__attribute__((") == 0) {
      size_t nameStart = i + 15;
      if (nameStart + 12 <= source.size() && source.compare(nameStart, 12, "overloadable") == 0) {
        // Strip __attribute__((overloadable)) — C++ has native overloading
        i = skipBalancedParens(source, i + 13);
        while (i < source.size() && source[i] == ' ') i++;
        continue;
      }
      if (nameStart + 20 <= source.size() && source.compare(nameStart, 20, "reqd_work_group_size") == 0) {
        // Convert __attribute__((reqd_work_group_size(N, 1, 1))) → __launch_bounds__(N)
        // Extract the first number from the args
        size_t argsStart = nameStart + 20;
        while (argsStart < source.size() && source[argsStart] == '(') argsStart++;
        size_t numEnd = argsStart;
        while (numEnd < source.size() && source[numEnd] >= '0' && source[numEnd] <= '9') numEnd++;
        if (numEnd > argsStart) {
          std::string wgSize = source.substr(argsStart, numEnd - argsStart);
          result += "__launch_bounds__(" + wgSize + ") ";
        }
        // Skip past the entire __attribute__((...))
        i = skipBalancedParens(source, i + 13);
        while (i < source.size() && source[i] == ' ') i++;
        continue;
      }
    }

    // Convert OpenCL vector cast syntax: (type2)(a, b) → make_type2(a, b)
    // Matches: (double2), (float2), (int2), (uint2), (long2), (ulong2)
    if (source[i] == '(' && i + 1 < source.size()) {
      static const char* vecTypes[] = {
        "double2)", "float2)", "int2)", "uint2)", "long2)", "ulong2)", "Word2)", nullptr
      };
      static const char* makeNames[] = {
        "make_double2", "make_float2", "make_int2", "make_uint2", "make_long2", "make_ulong2", "make_Word2"
      };
      bool matched = false;
      for (int vi = 0; vecTypes[vi]; vi++) {
        size_t tlen = strlen(vecTypes[vi]);
        if (i + 1 + tlen <= source.size() && source.compare(i + 1, tlen, vecTypes[vi]) == 0) {
          // Check what follows: should be whitespace then '(' for a cast constructor
          size_t after = i + 1 + tlen;
          while (after < source.size() && source[after] == ' ') after++;
          if (after < source.size() && source[after] == '(') {
            // It's (type2) (args) — replace with make_type2(args)
            result += makeNames[vi];
            i = after;  // now pointing at '(' of args
            matched = true;
            break;
          }
          // Also handle (type2){args} — less common but possible
          if (after < source.size() && source[after] == '{') {
            result += makeNames[vi];
            result += '(';
            i = after + 1;  // skip '{'
            // Find matching '}' and replace with ')'
            int depth = 1;
            while (i < source.size() && depth > 0) {
              if (source[i] == '{') depth++;
              else if (source[i] == '}') { depth--; if (depth == 0) { result += ')'; i++; break; } }
              else result += source[i];
              i++;
            }
            matched = true;
            break;
          }
        }
      }
      if (matched) continue;
    }

    // Convert "local TYPE NAME[" to "__shared__ TYPE NAME[" for shared memory declarations
    // This handles: "  local T2 lds[WIDTH / 4];" → "  __shared__ T2 lds[WIDTH / 4];"
    // Also handles: "  local T lds[IN_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];"
    // But NOT: "local T2 *lds" in function params (which becomes just "T2 *lds" via macro)
    if (i + 6 <= source.size() && source.compare(i, 6, "local ") == 0) {
      bool preceded = (i > 0 && (isalnum(source[i-1]) || source[i-1] == '_'));
      if (!preceded) {
        // Check if this "local" is followed by a type then a name then '['
        // i.e., it's a shared memory array declaration
        size_t lineEnd = source.find('\n', i);
        if (lineEnd == std::string::npos) lineEnd = source.size();
        std::string line = source.substr(i, lineEnd - i);
        // Match: "local TYPE IDENT[" pattern — indicates array declaration
        // Array declarations have '[' and end with ';'. They may also contain '(' in
        // the array size expression (e.g., ternary operators). The key distinction is
        // that function parameters don't have '['.
        if (line.find('[') != std::string::npos) {
          result += "__shared__ ";
          i += 6;  // skip "local "
          continue;
        }
        // For everything else (params, casts), just skip "local " → empty
        i += 6;
        continue;
      }
    }
    // Same for __local
    if (i + 8 <= source.size() && source.compare(i, 8, "__local ") == 0) {
      bool preceded = (i > 0 && (isalnum(source[i-1]) || source[i-1] == '_'));
      if (!preceded) {
        size_t lineEnd = source.find('\n', i);
        if (lineEnd == std::string::npos) lineEnd = source.size();
        std::string line = source.substr(i, lineEnd - i);
        if (line.find('[') != std::string::npos) {
          result += "__shared__ ";
          i += 8;
          continue;
        }
        i += 8;
        continue;
      }
    }

    // Replace "n" constraint with "r" in PTX asm (NVRTC requires true constants for "n")
    // Match: "n"( → "r"(
    if (i + 3 <= source.size() && source.compare(i, 3, "\"n\"") == 0) {
      // Check we're inside an asm statement context (look for preceding ':')
      // Simple heuristic: if the previous non-whitespace char is ':', ':' + space, or ','
      size_t back = i;
      while (back > 0 && (source[back-1] == ' ' || source[back-1] == '\t')) back--;
      if (back > 0 && (source[back-1] == ':' || source[back-1] == ',')) {
        result += "\"r\"";
        i += 3;
        continue;
      }
    }

    // Match __global NOT followed by _
    if (i + 8 <= source.size() && source.compare(i, 8, "__global") == 0) {
      if (i + 8 < source.size() && source[i + 8] == '_') {
        result += source[i++];
      } else {
        bool preceded = (i > 0 && (isalnum(source[i-1]) || source[i-1] == '_'));
        if (preceded) {
          result += source[i++];
        } else {
          i += 8;
        }
      }
    }
    // Match standalone "global" (not inside a word or PTX instruction)
    else if (i + 6 <= source.size() && source.compare(i, 6, "global") == 0) {
      bool preceded = (i > 0 && (isalnum(source[i-1]) || source[i-1] == '_' || source[i-1] == '.'));
      bool followed = (i + 6 < source.size() && (isalnum(source[i + 6]) || source[i + 6] == '_'));
      if (!preceded && !followed) {
        i += 6;
      } else {
        result += source[i++];
      }
    } else {
      result += source[i++];
    }
  }
  return result;
}

std::string NvrtcProgram::compile(const std::string& source, const std::string& name,
                                   const std::vector<std::string>& options,
                                   const std::vector<std::pair<std::string, std::string>>& headers) {
  // Prepare header arrays
  std::vector<const char*> headerSources, headerNames;
  for (auto& [hName, hSource] : headers) {
    headerNames.push_back(hName.c_str());
    headerSources.push_back(hSource.c_str());
  }

  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog, source.c_str(), name.c_str(),
                                  (int)headers.size(),
                                  headerSources.empty() ? nullptr : headerSources.data(),
                                  headerNames.empty() ? nullptr : headerNames.data()));

  // Convert options to char*
  std::vector<const char*> opts;
  for (auto& o : options) opts.push_back(o.c_str());

  nvrtcResult compileResult = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());

  // Get compilation log
  size_t logSize;
  nvrtcGetProgramLogSize(prog, &logSize);
  if (logSize > 1) {
    std::string compileLog(logSize, '\0');
    nvrtcGetProgramLog(prog, compileLog.data());
    if (compileResult != NVRTC_SUCCESS) {
      cuda_log("NVRTC compile error for '%s':\n%s\n", name.c_str(), compileLog.c_str());
    }
  }

  if (compileResult != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    throw std::runtime_error("NVRTC compilation failed for " + name);
  }

  // Get PTX
  size_t ptxSize;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
  std::string ptx(ptxSize, '\0');
  NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));

  nvrtcDestroyProgram(&prog);
  return ptx;
}

// ---- Kernel launcher ----

void CudaKernelLauncher::launch(CUstream stream, u32 gridSize, void** args, u32 sharedMem) {
  CU_CHECK(cuLaunchKernel(func,
                           gridSize, 1, 1,   // grid dimensions
                           blockSize, 1, 1,   // block dimensions
                           sharedMem,         // shared memory bytes
                           stream,            // stream
                           args,              // kernel arguments
                           nullptr));         // extra
}
