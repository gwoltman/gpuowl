// Copyright Mihai Preda.

#pragma once

#ifdef CUDA_BACKEND
#include "tinycuda.h"
#else
#include "tinycl.h"
#endif

#include <string>
#include <string_view>
#include <vector>
#include <memory>

using cl_queue = cl_command_queue;

void release(cl_context context);
void release(cl_kernel k);
void release(cl_mem buf);
void release(cl_program program);
void release(cl_queue queue);
void release(cl_event event);
void release(cl_graph graph);

template<typename T>
struct Deleter {
  using pointer = T;
  void operator()(T t) const { release(t); }
};

namespace std {
template<> struct default_delete<cl_context> : public Deleter<cl_context> {};
template<> struct default_delete<cl_kernel> : public Deleter<cl_kernel> {};
template<> struct default_delete<cl_mem> : public Deleter<cl_mem> {};
template<> struct default_delete<cl_program> : public Deleter<cl_program> {};
template<> struct default_delete<cl_queue> : public Deleter<cl_queue> {};
template<> struct default_delete<cl_event> : public Deleter<cl_event> {};
template<> struct default_delete<cl_graph> : public Deleter<cl_graph> {};
}

template<typename T> using Holder = std::unique_ptr<T, Deleter<T> >;

using QueueHolder = std::unique_ptr<cl_queue>;
using KernelHolder = std::unique_ptr<cl_kernel>;
using EventHolder = std::unique_ptr<cl_event>;
using GraphHolder = std::unique_ptr<cl_graph>;
using Program = std::unique_ptr<cl_program>;

class Context;

std::string getUUID(int seqId);

std::string errMes(int err);
void check(int err, const char *file, int line, const char *func, string_view mes);

#define CHECK1(err) check(err, __FILE__, __LINE__, __func__, #err)
#define CHECK2(err, mes) check(err, __FILE__, __LINE__, __func__, mes)

vector<cl_device_id> getAllDeviceIDs();
string getShortInfo(cl_device_id device);

string getDeviceName(cl_device_id id);
string getBoardName(cl_device_id id);
float getGpuRamGB(cl_device_id id);

// Get GPU free memory in bytes.
u64 getFreeMem(cl_device_id id);
bool hasFreeMemInfo(cl_device_id id);
bool isAmdGpu(cl_device_id id);
bool isNvidiaGpu(cl_device_id id);
bool hasFP64(cl_device_id id);
bool isRusticl(cl_device_id id);
u32 getNvidiaComputeCapability(cl_device_id id);
u32 getMaxWorkGroupSize(cl_device_id id);
u64 getLocalMemSize(cl_device_id id);
// AMD only (cl_amd_device_attribute_query); 0 when not available (e.g. not AMD, or an older driver).
u32 getAmdSimdPerComputeUnit(cl_device_id id);
u32 getAmdWavefrontWidth(cl_device_id id);
// CDNA2/CDNA3 (gfx90a, gfx94x/gfx95x): natively wave64 like gfx906, but with a "back-off" barrier that
// does not wait for outstanding LDS traffic -- see amdFastBarrierUnsafe.
bool isAmdCdna2Plus(cl_device_id id);
// True if a bare s_barrier (what -use FAST_BARRIER turns bar() into) cannot be trusted to wait for LDS on
// this device: false for non-AMD, true when the compiled wavefront isn't 64 (RDNA under ROCm's OpenCL
// compiler always picks 32, even though the hardware supports 64 too) or the device is CDNA2/CDNA3.
bool amdFastBarrierUnsafe(cl_device_id id);
string getDriverVersion(cl_device_id id);
string getDriverVersionByPos(int pos);

string getBdfFromDevice(cl_device_id id);

cl_context createContext(cl_device_id id);

string getBuildLog(cl_program program, cl_device_id deviceId);

Program loadBinary(cl_context context, cl_device_id deviceId, string_view fileName);
Program loadSource(cl_context context, const string& source);
bool hasAmdBcastBuiltins(cl_context context, cl_device_id deviceId);
cl_kernel loadKernel(cl_program program, const char *name);
void saveBinary(cl_program program, string_view fileName);

template<typename T>
void setArg(cl_kernel k, int pos, const T &value, const string& name) {
  CHECK2(clSetKernelArg(k, pos, sizeof(value), &value), name + '[' + to_string(pos) + "] size " + to_string(sizeof(value)));
}

/*
template<>
void setArg<int>(cl_kernel k, int pos, const int &value, const string& name);
*/

cl_mem makeBuf_(cl_context context, unsigned kind, size_t size, const void *ptr = nullptr);
cl_queue makeQueue(cl_device_id d, cl_context c, bool enableProfile);

void flush( cl_queue q);
void finish(cl_queue q);

EventHolder run(cl_queue queue, cl_kernel kernel, size_t groupSizeX, size_t workSizeX, size_t workSizeY,
                vector<cl_event>&& waits, const string &name, bool genEvent);

EventHolder read(cl_queue queue, vector<cl_event>&& waits,
                 bool blocking, cl_mem buf, size_t size, void *data, bool genEvent);

EventHolder write(cl_queue queue, vector<cl_event>&& waits,
                  bool blocking, cl_mem buf, size_t size, const void *data, bool genEvent);

EventHolder copyBuf(cl_queue queue, vector<cl_event>&& waits, const cl_mem src, cl_mem dst, size_t size, bool genEvent);

EventHolder fillBuf(cl_queue q, vector<cl_event>&& waits, cl_mem buf, const void *pat, size_t patSize, size_t size, bool genEvent);

EventHolder enqueueMarker(cl_queue q);
EventHolder enqueueMarkerWithWaits(cl_queue q, vector<cl_event>&& waits);

void waitForEvents(vector<cl_event>&& waits);


int getKernelNumArgs(cl_kernel k);
int getWorkGroupSize(cl_kernel k, cl_device_id device, const char *name);
int getKernelMaxWorkGroupSize(cl_kernel k, cl_device_id device, const char *name);
std::string getKernelArgName(cl_kernel k, int pos);

cl_device_id getDevice(u32 argsDevId);

// Returns the 3 intervals: queued, submit, run
std::array<i64, 3> getEventNanos(cl_event event);

// The command execution status: a state (CL_QUEUED..CL_COMPLETE), or, when the command terminated
// abnormally, the negative error code that terminated it.  Signed: as u32 an error compares equal to
// nothing and a waiter polling for CL_COMPLETE never stops.
int getEventInfo(cl_event event);

cl_context getQueueContext(cl_command_queue q);

#ifdef CUDA_BACKEND
// Set L1 cache configuration - 4 possibilities
void cudaSetL1Config(int x);

// Set L2 cache persistence for multiple read-only buffers on the given stream.
// Computes the address span covering all buffers and sets a single access policy window.
// Buffers that are nullptr or zero-size are skipped.
void cudaSetL2Persistent(cl_command_queue q, const std::vector<cl_mem>& buffers);

// Reserve a fraction (0-100%) of the device's max persisting L2 cache size for the current context.
// Without this call the driver uses its own (usually small) default, which limits how much of an
// access-policy window's "persisting" hint actually takes effect.
void cudaSetL2PersistLimit(int pct);

// Limit the registers of one kernel of a built program: maxRegs > 0 is a maximum register count, otherwise minBlocks > 0 is the
// launch bounds minimum blocks per SM.  The program's module is rebuilt from its PTX.  Returns false on failure.
bool cudaSetKernelRegLimit(cl_program prog, const char* kernelName, int maxRegs, int minBlocks);

// A compiled kernel's resource use, as it limits occupancy
struct CudaKernelResources {
  int regs;            // registers per thread
  int localBytes;      // local memory per thread, i.e. spilled registers
  int sharedBytes;     // static shared memory per block
  int threads;         // threads per block
};
CudaKernelResources cudaKernelResources(cl_kernel k);

// The current device's per-SM limits
struct CudaSmLimits {
  int regsPerSM;
  int regsPerBlock;
  int maxThreadsPerSM;
  int maxBlocksPerSM;
  int sharedPerSM;
  int reservedSharedPerBlock;
};
CudaSmLimits cudaSmLimits();
#endif


