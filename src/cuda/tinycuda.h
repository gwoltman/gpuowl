// CUDA type shim — replaces tinycl.h for the native CUDA backend.
// Maps OpenCL types and constants to CUDA Driver API equivalents.
// Used together with clwrap_cuda.cpp which implements cl* functions via cu*.

#pragma once

#include "../common.h"
#include <cuda.h>
#include <nvrtc.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <cstring>

// ---- Opaque handle wrappers ----
// OpenCL uses opaque pointer types (struct _cl_foo*).
// CUDA uses different representations (int, pointer, u64).
// We wrap CUDA handles in structs so they're pointer-like opaque types.

// cl_device_id wraps CUdevice (int)
struct _cl_device_id { CUdevice dev; };
typedef _cl_device_id* cl_device_id;

// cl_context wraps CUcontext
struct _cl_context { CUcontext ctx; CUdevice dev; };
typedef _cl_context* cl_context;

// cl_command_queue wraps CUstream
struct _cl_command_queue {
  CUstream stream;
  cl_context context;
  bool profiling;
};
typedef _cl_command_queue* cl_command_queue;

// cl_mem wraps CUdeviceptr + size
struct _cl_mem {
  CUdeviceptr ptr;
  size_t size;
};
typedef _cl_mem* cl_mem;

// cl_program: dual-purpose — stores either source string or compiled PTX/module
struct _cl_program {
  std::string source;     // OpenCL source (before NVRTC compilation)
  std::string preprocessedSource; // CUDA source after preprocessOpenCL (for parsing __launch_bounds__)
  std::string ptx;        // Compiled PTX (after NVRTC compilation)
  CUmodule module;        // Loaded module (after cuModuleLoadData)
  bool compiled;
  bool moduleLoaded;

  _cl_program() : module{}, compiled{false}, moduleLoaded{false} {}
};
typedef _cl_program* cl_program;

// cl_kernel wraps CUfunction + accumulated arguments
struct _cl_kernel {
  CUfunction func;
  std::string name;
  CUmodule parentModule;  // Keep reference so module isn't unloaded
  int numArgs;
  int reqWorkGroupSize;   // From __launch_bounds__(N) in source, matches OpenCL reqd_work_group_size

  // Argument accumulator for setArg/launch pattern
  static constexpr int MAX_ARGS = 32;
  static constexpr int MAX_ARG_BYTES = 512;
  char argData[MAX_ARG_BYTES];
  size_t argSizes[MAX_ARGS];
  size_t argOffsets[MAX_ARGS];

  _cl_kernel() : func{}, parentModule{}, numArgs{0}, reqWorkGroupSize{0} {
    memset(argData, 0, sizeof(argData));
    memset(argSizes, 0, sizeof(argSizes));
    memset(argOffsets, 0, sizeof(argOffsets));
  }

  void setArg(int pos, size_t size, const void* value) {
    if (pos >= MAX_ARGS) return;
    if (pos >= numArgs) numArgs = pos + 1;

    // Fixed 8-byte slots per arg position. CUDA kernel args are pointers (8 bytes)
    // or small scalars (4 bytes). Using fixed slots avoids data corruption when
    // args are set out of order (e.g., setFixedArgs(2,3) then operator()(0,1)).
    size_t offset = pos * 8;
    argOffsets[pos] = offset;
    argSizes[pos] = size;
    if (offset + size <= MAX_ARG_BYTES && value) {
      memcpy(argData + offset, value, size);
    }
  }

  // Build void* args[] array for cuLaunchKernel
  void buildArgPointers(void** ptrs) const {
    for (int i = 0; i < numArgs; i++) {
      ptrs[i] = const_cast<char*>(argData + argOffsets[i]);
    }
  }
};
typedef _cl_kernel* cl_kernel;

// cl_event wraps CUevent pair (start + end for profiling)
struct _cl_event {
  CUevent start;
  CUevent end;
  bool hasTimings;
  u32 commandType;

  _cl_event() : start{}, end{}, hasTimings{false}, commandType{0} {}
  ~_cl_event() {
    if (start) cuEventDestroy(start);
    if (end) cuEventDestroy(end);
  }
};
typedef _cl_event* cl_event;

// Unused types — just need to exist for compilation
typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_sampler*     cl_sampler;

typedef unsigned cl_bool;
typedef unsigned cl_program_build_info;
typedef unsigned cl_program_info;
typedef unsigned cl_device_info;
typedef unsigned cl_kernel_info;
typedef unsigned cl_kernel_arg_info;
typedef unsigned cl_kernel_work_group_info;
typedef unsigned cl_profiling_info;
typedef unsigned cl_event_info;
typedef unsigned cl_command_queue_info;

typedef u64 cl_mem_flags;
typedef u64 cl_svm_mem_flags;
typedef u64 cl_device_type;
typedef u64 cl_queue_properties;

using cl_queue = cl_command_queue;

// ---- Constants ----
#define CL_SUCCESS              0
#define CL_DEVICE_TYPE_GPU      (1 << 2)
#define CL_DEVICE_TYPE_ALL      0xFFFFFFFF

#define CL_DEVICE_VENDOR_ID             0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS     0x1002
#define CL_DEVICE_MAX_CLOCK_FREQUENCY   0x100C
#define CL_DEVICE_GLOBAL_MEM_SIZE       0x101F
#define CL_DEVICE_ERROR_CORRECTION_SUPPORT 0x1024
#define CL_DEVICE_NAME          0x102B
#define CL_DRIVER_VERSION       0x102D
#define CL_DEVICE_VERSION       0x102F
#define CL_DEVICE_BUILT_IN_KERNELS 0x103F
#define CL_PLATFORM_VERSION     0x0901

#define CL_PROGRAM_BINARY_SIZES 0x1165
#define CL_PROGRAM_BINARIES     0x1166
#define CL_PROGRAM_BUILD_LOG    0x1183

#define CL_MEM_READ_WRITE       (1 << 0)
#define CL_MEM_WRITE_ONLY       (1 << 1)
#define CL_MEM_READ_ONLY        (1 << 2)
#define CL_MEM_USE_HOST_PTR     (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR   (1 << 4)
#define CL_MEM_COPY_HOST_PTR    (1 << 5)
#define CL_MEM_HOST_WRITE_ONLY  (1 << 7)
#define CL_MEM_HOST_READ_ONLY   (1 << 8)
#define CL_MEM_HOST_NO_ACCESS   (1 << 9)
#define CL_MEM_SVM_FINE_GRAIN_BUFFER (1 << 10)
#define CL_MEM_SVM_ATOMICS           (1 << 11)

#define CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE (1 << 0)
#define CL_QUEUE_PROFILING_ENABLE              (1 << 1)
#define CL_QUEUE_ON_DEVICE                     (1 << 2)
#define CL_QUEUE_ON_DEVICE_DEFAULT             (1 << 3)

#define CL_QUEUE_CONTEXT            0x1090
#define CL_QUEUE_DEVICE             0x1091
#define CL_QUEUE_REFERENCE_COUNT    0x1092
#define CL_QUEUE_PROPERTIES         0x1093

#define CL_PROFILING_COMMAND_QUEUED  0x1280
#define CL_PROFILING_COMMAND_SUBMIT  0x1281
#define CL_PROFILING_COMMAND_START   0x1282
#define CL_PROFILING_COMMAND_END     0x1283
#define CL_PROFILING_COMMAND_COMPLETE 0x1284

#define CL_EVENT_COMMAND_QUEUE              0x11D0
#define CL_EVENT_COMMAND_TYPE               0x11D1
#define CL_EVENT_REFERENCE_COUNT            0x11D2
#define CL_EVENT_COMMAND_EXECUTION_STATUS   0x11D3
#define CL_EVENT_CONTEXT                    0x11D4

#define CL_COMMAND_NDRANGE_KERNEL   0x11F0
#define CL_COMMAND_READ_BUFFER      0x11F3
#define CL_COMMAND_WRITE_BUFFER     0x11F4
#define CL_COMMAND_COPY_BUFFER      0x11F5
#define CL_COMMAND_FILL_BUFFER      0x1207
#define CL_COMMAND_MARKER           0x11FE

#define CL_COMPLETE     0x0
#define CL_RUNNING      0x1
#define CL_SUBMITTED    0x2
#define CL_QUEUED       0x3

#define CL_KERNEL_NUM_ARGS       0x1191
#define CL_KERNEL_ARG_NAME       0x119A
#define CL_KERNEL_ATTRIBUTES     0x1195
#define CL_KERNEL_COMPILE_WORK_GROUP_SIZE 0x11B1

// AMD-specific (unused but must exist for compilation)
#define CL_DEVICE_PCIE_ID_AMD           0x4034
#define CL_DEVICE_TOPOLOGY_AMD          0x4037
#define CL_DEVICE_BOARD_NAME_AMD        0x4038
#define CL_DEVICE_GLOBAL_FREE_MEMORY_AMD 0x4039

typedef union {
  struct { u32 type; u32 data[5]; } raw;
  struct { u32 type; char unused[17]; char bus; char device; char function; } pcie;
} cl_device_topology_amd;

// Error codes
#define CL_DEVICE_NOT_FOUND             -1
#define CL_DEVICE_NOT_AVAILABLE         -2
#define CL_COMPILER_NOT_AVAILABLE       -3
#define CL_MEM_OBJECT_ALLOCATION_FAILURE -4
#define CL_OUT_OF_RESOURCES             -5
#define CL_OUT_OF_HOST_MEMORY           -6
#define CL_PROFILING_INFO_NOT_AVAILABLE -7
#define CL_BUILD_PROGRAM_FAILURE        -11
#define CL_COMPILE_PROGRAM_FAILURE      -15
#define CL_LINK_PROGRAM_FAILURE         -17
#define CL_INVALID_VALUE                -30
#define CL_INVALID_DEVICE               -33
#define CL_INVALID_CONTEXT              -34
#define CL_INVALID_MEM_OBJECT           -38
#define CL_INVALID_BINARY               -42
#define CL_INVALID_BUILD_OPTIONS        -43
#define CL_INVALID_PROGRAM              -44
#define CL_INVALID_KERNEL_NAME          -46
#define CL_INVALID_KERNEL               -48
#define CL_INVALID_ARG_INDEX            -49
#define CL_INVALID_ARG_VALUE            -50
#define CL_INVALID_ARG_SIZE             -51
#define CL_INVALID_WORK_GROUP_SIZE      -54
#define CL_INVALID_GLOBAL_WORK_SIZE     -63

// ---- OpenCL API function declarations ----
// These are implemented in clwrap_cuda.cpp using CUDA Driver API.
// They match the signatures from tinycl.h so clwrap.h compiles unchanged.

extern "C" {

unsigned clGetPlatformIDs(unsigned, cl_platform_id*, unsigned*);
int clGetDeviceIDs(cl_platform_id, cl_device_type, unsigned, cl_device_id*, unsigned*);
cl_context clCreateContext(const intptr_t*, unsigned, const cl_device_id*,
                           void (*)(const char*, const void*, size_t, void*), void*, int*);
int clReleaseContext(cl_context);
int clReleaseProgram(cl_program);
int clReleaseCommandQueue(cl_command_queue);
int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, unsigned, const size_t*,
                            const size_t*, const size_t*, unsigned, const cl_event*, cl_event*);

cl_program clCreateProgramWithSource(cl_context, unsigned, const char**, const size_t*, int*);
cl_program clCreateProgramWithBinary(cl_context, unsigned, const cl_device_id*, const size_t*,
                                      const unsigned char**, int*, int*);

int clBuildProgram(cl_program, unsigned, const cl_device_id*, const char*,
                   void (*)(cl_program, void*), void*);
int clCompileProgram(cl_program, unsigned, const cl_device_id*, const char*,
                     unsigned numHeaders, const cl_program* headers, const char* const* headerNames,
                     void (*)(cl_program, void*), void*);
cl_program clLinkProgram(cl_context, unsigned, const cl_device_id*, const char*,
                          unsigned nProgs, const cl_program* progs,
                          void (*)(cl_program, void*), void*, int* err);

int clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
int clGetProgramInfo(cl_program, cl_program_info, size_t, void*, size_t*);
int clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void*, size_t*);
int clGetPlatformInfo(cl_platform_id, cl_device_info, size_t, void*, size_t*);
int clGetCommandQueueInfo(cl_command_queue, cl_command_queue_info, size_t, void*, size_t*);

cl_kernel clCreateKernel(cl_program, const char*, int*);
int clReleaseKernel(cl_kernel);
cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void*, int*);
int clReleaseMemObject(cl_mem);
cl_command_queue clCreateCommandQueueWithProperties(cl_context, cl_device_id,
                                                     const cl_queue_properties*, int*);

int clEnqueueMarkerWithWaitList(cl_command_queue, unsigned, const cl_event*, cl_event*);
int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*,
                         unsigned, const cl_event*, cl_event*);
int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*,
                          unsigned, const cl_event*, cl_event*);
int clEnqueueCopyBuffer(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t,
                         unsigned, const cl_event*, cl_event*);
int clEnqueueFillBuffer(cl_command_queue, cl_mem, const void*, size_t, size_t, size_t,
                         unsigned, const cl_event*, cl_event*);

int clFlush(cl_command_queue);
int clFinish(cl_command_queue);
int clSetKernelArg(cl_kernel, unsigned, size_t, const void*);

int clReleaseEvent(cl_event);
int clWaitForEvents(unsigned, const cl_event*);

int clGetKernelInfo(cl_kernel, cl_kernel_info, size_t, void*, size_t*);
int clGetKernelArgInfo(cl_kernel, unsigned, cl_kernel_arg_info, size_t, void*, size_t*);
int clGetKernelWorkGroupInfo(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*);

int clGetEventInfo(cl_event, cl_event_info, size_t, void*, size_t*);
int clGetEventProfilingInfo(cl_event, cl_profiling_info, size_t, void*, size_t*);

void* clSVMAlloc(cl_context, cl_svm_mem_flags, size_t, unsigned);
void clSVMFree(cl_context, void*);
int clSetKernelArgSVMPointer(cl_kernel, unsigned, const void*);

}
