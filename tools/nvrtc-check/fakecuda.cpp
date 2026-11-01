// A stand-in for the CUDA driver library (libcuda) with one fake nVidia GPU and no hardware behind it.  A PRPLL CUDA build run
// against it (LD_PRELOAD) does everything it does before the first iteration, with the real NVRTC compiling every kernel; kernels
// are never run, memory is not backed, and a residue read returns 3 so that a PRP gets past its on-load check.  See nvrtc-check.py.
//
// FAKECUDA_CC picks the GPU: a compute capability as digits (89 = sm_89), "min" or "max" for the oldest or newest architecture the
// NVRTC in use supports.
//
// No CUDA headers: the functions are defined under their exported (versioned) names, and the attribute numbers are the driver API's
// (ABI-stable).  Functions with an unlisted prototype in an older or newer toolkit simply have no caller here.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <dlfcn.h>

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;

static int fakeCC() {
  static int const cc = [] {
    const char* s = getenv("FAKECUDA_CC");
    if (!s || !*s) { return 89; }
    if (strcmp(s, "min") && strcmp(s, "max")) { return atoi(s); }
    // The architectures the loaded NVRTC accepts (NVRTC 11.2 and up; older ones fall back to a fixed guess)
    auto count = (int (*)(int*)) dlsym(RTLD_DEFAULT, "nvrtcGetNumSupportedArchs");
    auto list = (int (*)(int*)) dlsym(RTLD_DEFAULT, "nvrtcGetSupportedArchs");
    int n = 0;
    if (count && list && count(&n) == 0 && n > 0) {
      int* a = (int*) malloc(n * sizeof(int));
      if (list(a) == 0) { return *s == 'm' && s[1] == 'i' ? a[0] : a[n - 1]; }
    }
    return *s == 'm' && s[1] == 'i' ? 52 : 89;
  }();
  return cc;
}

extern "C" {

CUresult cuInit(unsigned) { return 0; }
CUresult cuDriverGetVersion(int* v) { *v = 12090; return 0; }
CUresult cuDeviceGetCount(int* n) { *n = 1; return 0; }
CUresult cuDeviceGet(CUdevice* d, int i) { *d = i; return 0; }
CUresult cuDeviceGetName(char* name, int len, CUdevice) { snprintf(name, len, "Fake NVIDIA GPU (sm_%d)", fakeCC()); return 0; }
CUresult cuDeviceTotalMem_v2(size_t* b, CUdevice) { *b = 16ull << 30; return 0; }
CUresult cuDeviceGetPCIBusId(char* s, int len, CUdevice) { snprintf(s, len, "0000:01:00.0"); return 0; }
CUresult cuMemGetInfo_v2(size_t* f, size_t* t) { *f = 15ull << 30; *t = 16ull << 30; return 0; }

CUresult cuDeviceGetAttribute(int* v, int attr, CUdevice) {
  switch (attr) {
    case 1:   *v = 1024; break;                       // MAX_THREADS_PER_BLOCK
    case 8:   *v = 49152; break;                      // MAX_SHARED_MEMORY_PER_BLOCK
    case 12:  *v = 65536; break;                      // MAX_REGISTERS_PER_BLOCK
    case 13:  *v = 1800000; break;                    // CLOCK_RATE (kHz)
    case 16:  *v = 64; break;                         // MULTIPROCESSOR_COUNT
    case 32:  *v = 0; break;                          // ECC_ENABLED
    case 39:  *v = 1536; break;                       // MAX_THREADS_PER_MULTIPROCESSOR
    case 75:  *v = fakeCC() / 10; break;              // COMPUTE_CAPABILITY_MAJOR
    case 76:  *v = fakeCC() % 10; break;              // COMPUTE_CAPABILITY_MINOR
    case 81:  *v = 102400; break;                     // MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
    case 82:  *v = 65536; break;                      // MAX_REGISTERS_PER_MULTIPROCESSOR
    case 106: *v = 16; break;                         // MAX_BLOCKS_PER_MULTIPROCESSOR
    case 108: *v = 0; break;                          // MAX_PERSISTING_L2_CACHE_SIZE
    case 109: *v = 0; break;                          // MAX_ACCESS_POLICY_WINDOW_SIZE
    case 111: *v = 1024; break;                       // RESERVED_SHARED_MEMORY_PER_BLOCK
    default:  *v = 0;
  }
  return 0;
}

// Contexts, streams, events, modules, functions, graphs: any non-null handle
static int handles[1 << 12];
static std::atomic<unsigned> nextHandle{0};
static void* newHandle() { return &handles[nextHandle++ % (1 << 12)]; }

CUresult cuCtxCreate_v2(void** c, unsigned, CUdevice) { *c = newHandle(); return 0; }
CUresult cuCtxCreate_v4(void** c, void*, unsigned, CUdevice) { *c = newHandle(); return 0; }
CUresult cuCtxDestroy_v2(void*) { return 0; }
CUresult cuCtxSetCurrent(void*) { return 0; }
CUresult cuCtxGetDevice(CUdevice* d) { *d = 0; return 0; }
CUresult cuCtxSetLimit(int, size_t) { return 0; }
CUresult cuCtxSetCacheConfig(int) { return 0; }

CUresult cuStreamCreate(void** s, unsigned) { *s = newHandle(); return 0; }
CUresult cuStreamDestroy_v2(void*) { return 0; }
CUresult cuStreamSynchronize(void*) { return 0; }
CUresult cuStreamWaitEvent(void*, void*, unsigned) { return 0; }
CUresult cuStreamSetAttribute(void*, int, const void*) { return 0; }
CUresult cuStreamBeginCapture_v2(void*, int) { return 0; }
CUresult cuStreamEndCapture(void*, void** g) { *g = newHandle(); return 0; }
CUresult cuGraphInstantiateWithFlags(void** e, void*, unsigned long long) { *e = newHandle(); return 0; }
CUresult cuGraphInstantiate_v2(void** e, void*, void*, char*, size_t) { *e = newHandle(); return 0; }
CUresult cuGraphLaunch(void*, void*) { return 0; }
CUresult cuGraphDestroy(void*) { return 0; }
CUresult cuGraphExecDestroy(void*) { return 0; }
CUresult cuEventCreate(void** e, unsigned) { *e = newHandle(); return 0; }
CUresult cuEventDestroy_v2(void*) { return 0; }
CUresult cuEventRecord(void*, void*) { return 0; }
CUresult cuEventQuery(void*) { return 0; }
CUresult cuEventSynchronize(void*) { return 0; }
CUresult cuEventElapsedTime(float* ms, void*, void*) { *ms = 0; return 0; }

// The module is the PTX or CUBIN NVRTC made: nothing here loads or checks it
CUresult cuModuleLoadDataEx(void** m, const void*, unsigned, void*, void*) { *m = newHandle(); return 0; }
CUresult cuModuleLoadData(void** m, const void*) { *m = newHandle(); return 0; }
CUresult cuModuleUnload(void*) { return 0; }
CUresult cuModuleGetFunction(void** f, void*, const char*) { *f = newHandle(); return 0; }
CUresult cuFuncSetAttribute(void*, int, int) { return 0; }
CUresult cuFuncGetAttribute(int* v, int attr, void*) {
  switch (attr) {
    case 0:  *v = 1024; break;                        // MAX_THREADS_PER_BLOCK
    case 4:  *v = 64; break;                          // NUM_REGS
    default: *v = 0;                                  // SHARED_SIZE_BYTES, LOCAL_SIZE_BYTES
  }
  return 0;
}

CUresult cuLaunchKernel(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**) { return 0; }
CUresult cuLaunchKernelEx(const void*, void*, void**, void**) { return 0; }

// Memory is not backed.  Reads are made to look sane enough to get a run past its checks and on to the kernels after them: a PRP
// resumed at iteration 0 reads back the residue 3 (FAKECUDA_RES, 4 for an LL test) from the first word of the 256-word output
// buffer; the GPU's checksum of a buffer (8 bytes) and the checksum the host computes from the buffer itself agree on 3; other
// data are zeros.
static std::atomic<CUdeviceptr> nextPtr{1ull << 40};
CUresult cuMemAlloc_v2(CUdeviceptr* p, size_t n) { *p = nextPtr.fetch_add((n + 0xffff) & ~size_t(0xffff)); return 0; }
CUresult cuMemFree_v2(CUdeviceptr) { return 0; }
static void fakeRead(void* dst, size_t n) {
  static long long const res = getenv("FAKECUDA_RES") ? atoll(getenv("FAKECUDA_RES")) : 3;
  memset(dst, 0, n);
  if (n == 8) { *(long long*) dst = 3; }
  else if (n == 2048) { ((int*) dst)[32] = res; ((long long*) dst)[32] = res; }
  else if (n > 8) { *(int*) dst = 3; }
}
CUresult cuMemcpyDtoH_v2(void* dst, CUdeviceptr, size_t n) { fakeRead(dst, n); return 0; }
CUresult cuMemcpyDtoHAsync_v2(void* dst, CUdeviceptr, size_t n, void*) { fakeRead(dst, n); return 0; }
CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void*, size_t) { return 0; }
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr, const void*, size_t, void*) { return 0; }
CUresult cuMemcpyDtoD_v2(CUdeviceptr, CUdeviceptr, size_t) { return 0; }
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr, CUdeviceptr, size_t, void*) { return 0; }
CUresult cuMemsetD8_v2(CUdeviceptr, unsigned char, size_t) { return 0; }
CUresult cuMemsetD32_v2(CUdeviceptr, unsigned, size_t) { return 0; }
CUresult cuMemsetD8Async(CUdeviceptr, unsigned char, size_t, void*) { return 0; }
CUresult cuMemsetD16Async(CUdeviceptr, unsigned short, size_t, void*) { return 0; }
CUresult cuMemsetD32Async(CUdeviceptr, unsigned, size_t, void*) { return 0; }
CUresult cuMemsetD2D32Async(CUdeviceptr, size_t, unsigned, size_t, size_t, void*) { return 0; }

CUresult cuGetErrorName(CUresult r, const char** s) { *s = r ? "CUDA_ERROR_FAKE" : "CUDA_SUCCESS"; return 0; }
CUresult cuGetErrorString(CUresult r, const char** s) { *s = r ? "error from the fake CUDA driver" : "no error"; return 0; }

}
