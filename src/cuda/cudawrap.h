// CUDA Driver API wrappers, parallel to clwrap.h for OpenCL.
// Uses the CUDA Driver API (cu*) for maximum control.

#pragma once

#include <cuda.h>
#include <nvrtc.h>
#include <string>
#include <vector>
#include <memory>
#include <array>
#include <cassert>

// Types — when building standalone tests, define minimal types.
// When building as part of PRPLL, common.h is already included.
#if __has_include("../common.h") && !defined(CUDAWRAP_STANDALONE)
#include "../common.h"
#else
#ifndef CUDAWRAP_TYPES_DEFINED
#define CUDAWRAP_TYPES_DEFINED
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned char u8;
#endif
#endif

// ---- Error checking ----
void checkCuda(CUresult err, const char* file, int line, const char* func, const char* expr);
void checkNvrtc(nvrtcResult err, const char* file, int line, const char* func, const char* expr);

#define CU_CHECK(expr) checkCuda((expr), __FILE__, __LINE__, __func__, #expr)
#define NVRTC_CHECK(expr) checkNvrtc((expr), __FILE__, __LINE__, __func__, #expr)

// ---- Device management ----
std::vector<CUdevice> getAllDevices();
std::string getDeviceName(CUdevice dev);
std::string getDriverVersion();
float getGpuRamGB(CUdevice dev);
u64 getFreeMem(CUdevice dev);
std::string getShortInfo(CUdevice dev);

// ---- Context ----
class CudaContext {
  CUcontext ctx{};
  CUdevice device{};
public:
  explicit CudaContext(CUdevice dev);
  ~CudaContext();

  CUcontext get() const { return ctx; }
  CUdevice getDevice() const { return device; }
  void makeCurrent();
};

// ---- Module (compiled kernels) ----
class CudaModule {
  CUmodule module{};
public:
  CudaModule() = default;
  explicit CudaModule(const std::string& ptx, const std::string& name = "");
  ~CudaModule();

  CudaModule(CudaModule&& rhs) noexcept : module(rhs.module) { rhs.module = nullptr; }
  CudaModule& operator=(CudaModule&& rhs) noexcept {
    if (this != &rhs) { if (module) cuModuleUnload(module); module = rhs.module; rhs.module = nullptr; }
    return *this;
  }

  CUmodule get() const { return module; }
  CUfunction getFunction(const char* name) const;
};

// ---- Stream (equivalent to cl_command_queue) ----
class CudaStream {
  CUstream stream{};
public:
  CudaStream();
  ~CudaStream();

  CUstream get() const { return stream; }
  void sync();

  CudaStream(CudaStream&& rhs) noexcept : stream(rhs.stream) { rhs.stream = nullptr; }
  CudaStream& operator=(CudaStream&&) = delete;
};

// ---- Memory buffer ----
class CudaBuffer {
  CUdeviceptr ptr{};
  size_t bytes{};
public:
  CudaBuffer() = default;
  explicit CudaBuffer(size_t bytes);
  ~CudaBuffer();

  CudaBuffer(CudaBuffer&& rhs) noexcept : ptr(rhs.ptr), bytes(rhs.bytes) { rhs.ptr = 0; rhs.bytes = 0; }
  CudaBuffer& operator=(CudaBuffer&& rhs) noexcept {
    if (this != &rhs) { if (ptr) cuMemFree(ptr); ptr = rhs.ptr; bytes = rhs.bytes; rhs.ptr = 0; rhs.bytes = 0; }
    return *this;
  }

  CUdeviceptr get() const { return ptr; }
  size_t size() const { return bytes; }

  void readSync(void* dst, size_t n) const;
  void writeSync(const void* src, size_t n);
  void zero();
  void copyFrom(const CudaBuffer& src);
  void fillPattern(const void* pattern, size_t patternSize);
};

// ---- NVRTC Compilation ----
struct NvrtcProgram {
  // Preprocess OpenCL source for CUDA: strip __global/global pointer qualifiers
  // without breaking __global__ (CUDA kernel qualifier).
  static std::string preprocessOpenCL(const std::string& source);

  static std::string compile(const std::string& source, const std::string& name,
                              const std::vector<std::string>& options,
                              const std::vector<std::pair<std::string, std::string>>& headers = {});
};

// ---- Kernel launcher ----
class CudaKernelLauncher {
  CUfunction func{};
  std::string name;
  u32 blockSize{};

public:
  CudaKernelLauncher() = default;
  CudaKernelLauncher(CUfunction f, const std::string& name, u32 blockSize)
    : func(f), name(name), blockSize(blockSize) {}

  void launch(CUstream stream, u32 gridSize, void** args, u32 sharedMem = 0);

  CUfunction get() const { return func; }
  const std::string& getName() const { return name; }
};
