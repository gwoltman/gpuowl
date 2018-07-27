// Copyright (C) 2017 Mihai Preda.

#pragma once

#include "tinycl.h"
#include "timeutil.h"

#include <cassert>
#include <cstdio>
#include <cstdarg>

#include <string>
#include <vector>

using std::string;
using std::vector;

typedef cl_command_queue cl_queue;

template<typename T>
struct ReleaseDelete {
  using pointer = T;
  
  void operator()(T t) {
    // fprintf(stderr, "Release %s %llx\n", typeid(T).name(), u64(t));
    release(t);
  }
};

template<typename T> using Holder = std::unique_ptr<T, ReleaseDelete<T> >;

using Buffer  = Holder<cl_mem>;
using Context = Holder<cl_context>;
using QueueHolder = Holder<cl_queue>;

static_assert(sizeof(Buffer) == sizeof(cl_mem), "size Buffer");

bool check(int err, const char *mes = nullptr) {
  bool ok = (err == CL_SUCCESS);
  if (!ok) {
    if (mes) {
      log("error %d (%s)\n", err, mes);
    } else {
      log("error %d\n", err);
    }
  }
  return ok;
}

#define CHECK(what) assert(check(what));
#define CHECK2(what, mes) assert(check(what, mes));

vector<cl_device_id> getDeviceIDs(bool onlyGPU) {
  cl_platform_id platforms[16];
  int nPlatforms = 0;
  CHECK(clGetPlatformIDs(16, platforms, (unsigned *) &nPlatforms));
  vector<cl_device_id> ret;
  cl_device_id devices[64];
  for (int i = 0; i < nPlatforms; ++i) {
    unsigned n = 0;
    CHECK(clGetDeviceIDs(platforms[i], onlyGPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL, 64, devices, &n));
    for (unsigned k = 0; k < n; ++k) { ret.push_back(devices[k]); }
  }
  return ret;
}

int getNumberOfDevices() {
  cl_platform_id platforms[8];
  unsigned nPlatforms;
  CHECK(clGetPlatformIDs(8, platforms, &nPlatforms));
  
  unsigned n = 0;
  for (int i = 0; i < (int) nPlatforms; ++i) {
    unsigned delta = 0;
    CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &delta));
    n += delta;
  }
  return n;
}

void getInfo(cl_device_id id, int what, size_t bufSize, void *buf) { CHECK(clGetDeviceInfo(id, what, bufSize, buf, NULL)); }

bool getInfoMaybe(cl_device_id id, int what, size_t bufSize, void *buf) {
  return clGetDeviceInfo(id, what, bufSize, buf, NULL) == CL_SUCCESS;
}

static string getTopology(cl_device_id id) {
  char topology[64];
  cl_device_topology_amd top;
  if (!getInfoMaybe(id, CL_DEVICE_TOPOLOGY_AMD, sizeof(top), &top)) { return ""; }
  snprintf(topology, sizeof(topology), "@%x:%u.%u",
           (unsigned) (unsigned char) top.pcie.bus, (unsigned) top.pcie.device, (unsigned) top.pcie.function);
  return topology;
}

static string getBoardName(cl_device_id id) {
  char boardName[64];
  return getInfoMaybe(id, CL_DEVICE_BOARD_NAME_AMD, sizeof(boardName), boardName) ? boardName : "";
}

static string getHwName(cl_device_id id) {
  char name[64];
  getInfo(id, CL_DEVICE_NAME, sizeof(name), name);
  return name;
}

static string getFreq(cl_device_id device) {
  unsigned computeUnits, frequency;
  getInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits);
  getInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(frequency), &frequency);

  char info[64];
  snprintf(info, sizeof(info), "%ux%4u", computeUnits, frequency);
  return info;
}

string getShortInfo(cl_device_id device) { return getHwName(device) + "-" + getFreq(device) + "-" + getTopology(device); }
string getLongInfo(cl_device_id device) { return getShortInfo(device) + " " + getBoardName(device); }

cl_context createContext(cl_device_id device) {
  int err;
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK2(err, "clCreateContext");
  return context;
}

void release(cl_context context) { CHECK(clReleaseContext(context)); }
void release(cl_program program) { CHECK(clReleaseProgram(program)); }
void release(cl_mem buf)         { CHECK(clReleaseMemObject(buf)); }
void release(cl_queue queue)     { CHECK(clReleaseCommandQueue(queue)); }
void release(cl_kernel k)        { CHECK(clReleaseKernel(k)); }

bool dumpBinary(cl_program program, const string &fileName) {
  if (auto fo = open(fileName.c_str(), "wb")) {
    size_t size;
    CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size), &size, NULL));
    char *buf = new char[size + 1];
    CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(&buf), &buf, NULL));
    fwrite(buf, 1, size, fo.get());
    delete[] buf;
    return true; 
  }
  return false;
}

static string readFile(const string &name) {
  string ret;
  if (auto fi = open(name, "rb", false)) {
    char buf[1024];
    while (true) {
      size_t n = fread(buf, 1, sizeof(buf), fi.get());
      ret.append(buf, n);
      if (n < sizeof(buf)) { break; }
    }
  }
  return ret;
}

static cl_program loadBinary(cl_device_id device, cl_context context, const string &name, const string &config) {
  string binFile = string("precompiled/") + name + "_" + config + ".so";
  string binary = readFile(binFile);
  cl_program program = 0;
  if (!binary.empty()) {  
    cl_device_id devices[] = {device};
    size_t sizes[] = {binary.size()};
    const unsigned char *binaries[] = {(const unsigned char *) binary.c_str()};
    int binStatus[] = {0};
    int err = 0;
    program = clCreateProgramWithBinary(context, 1, devices, sizes, binaries, binStatus, &err);
    if (err != CL_SUCCESS) {
      log("Error loading pre-compiled kernel from '%s' (error %d, %d)\n", binFile.c_str(), err, binStatus[0]);
    } else {
      log("Loaded pre-compiled kernel from '%s'\n", binFile.c_str());
    }
  }
  return program;
}

static cl_program loadSource(cl_context context, const string &name) {
  string stub = string("#include \"") + name + ".cl\"\n";
  
  const char *ptr = stub.c_str();
  size_t size = stub.size();
  int err;
  cl_program program = clCreateProgramWithSource(context, 1, &ptr, &size, &err);
  CHECK2(err, "clCreateProgramWithSource");
  return program;  
}

static bool build(cl_program program, cl_device_id device, const string &args) {
  Timer timer;
  int err = clBuildProgram(program, 1, &device, args.c_str(), NULL, NULL);
  bool ok = (err == CL_SUCCESS);
  if (!ok) { log("OpenCL compilation error %d (args %s)\n", err, args.c_str()); }
  
  size_t logSize;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
  if (logSize > 1) {
    std::unique_ptr<char> buf(new char[logSize + 1]);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buf.get(), &logSize);
    buf.get()[logSize] = 0;
    log("%s\n", buf.get());
  }
  if (ok) { log("OpenCL compilation in %d ms, with \"%s\"\n", timer.deltaMillis(), args.c_str()); }
  return ok;
}

string join(const string &prefix, const vector<string> &elems) {
  string big = "";
  for (auto s : elems) { big += prefix + s; }
  return big;
}

cl_program compile(cl_device_id device, cl_context context, const string &name, const string &extraArgs,
                   const vector<string> &defVect, const string &config = "") {
  string args = join(" -D", defVect) + " " + extraArgs + " " + "-I. -cl-fast-relaxed-math -cl-std=CL2.0 ";
  // -cl-kernel-arg-info

  cl_program program = 0;
  
  if (!config.empty() && (program = loadBinary(device, context, name, config))) {
    if (build(program, device, args)) {
      return program;
    } else {
      release(program);
    }
  }

  if ((program = loadSource(context, name))) {
    if (build(program, device, args)) {
      return program;
    } else {
      release(program);
    }
  }
  
  return 0;
}
  // Other options:
  // * -cl-uniform-work-group-size
  // * -fno-bin-llvmir
  // * various: -fno-bin-source -fno-bin-amdil

cl_kernel makeKernel(cl_program program, const char *name) {
  int err;
  cl_kernel k = clCreateKernel(program, name, &err);
  CHECK2(err, name);
  return k;
}

template<typename T>
void setArg(cl_kernel k, int pos, const T &value) { CHECK(clSetKernelArg(k, pos, sizeof(value), &value)); }

// Special-case Buffer argument: pass the wrapped cl_mem.
void setArg(cl_kernel k, int pos, const Buffer &buf) { setArg(k, pos, buf.get()); }

cl_mem makeBuf(cl_context context, unsigned kind, size_t size, const void *ptr = 0) {
  int err;
  cl_mem buf = clCreateBuffer(context, kind, size, (void *) ptr, &err);
  CHECK2(err, "clCreateBuffer");
  return buf;
}

cl_mem makeBuf(Context &context, unsigned kind, size_t size, const void *ptr = 0) { return makeBuf(context.get(), kind, size, ptr); }

cl_queue makeQueue(cl_device_id d, cl_context c) {
  int err;
  cl_queue q = clCreateCommandQueue(c, d, 0, &err);
  CHECK2(err, "clCreateCommandQueue");
  return q;
}

void flush( cl_queue q) { CHECK(clFlush(q)); }
void finish(cl_queue q) { CHECK(clFinish(q)); }

void run(cl_queue queue, cl_kernel kernel, size_t groupSize, size_t workSize, const string &name) {
  CHECK2(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &workSize, &groupSize, 0, NULL, NULL), name.c_str());
}

void read(cl_queue queue, bool blocking, cl_mem buf, size_t size, void *data, size_t start = 0) {
  CHECK(clEnqueueReadBuffer(queue, buf, blocking, start, size, data, 0, NULL, NULL));
}

void read(cl_queue queue, bool blocking, Buffer &buf, size_t size, void *data, size_t start = 0) {
  CHECK(clEnqueueReadBuffer(queue, buf.get(), blocking, start, size, data, 0, NULL, NULL));
}

void write(cl_queue queue, bool blocking, cl_mem buf, size_t size, const void *data, size_t start = 0) {
  CHECK(clEnqueueWriteBuffer(queue, buf, blocking, start, size, data, 0, NULL, NULL));
}

void write(cl_queue queue, bool blocking, Buffer &buf, size_t size, const void *data, size_t start = 0) {
  CHECK(clEnqueueWriteBuffer(queue, buf.get(), blocking, start, size, data, 0, NULL, NULL));
}

void copyBuf(cl_queue queue, Buffer &src, Buffer &dst, size_t size) {
  CHECK(clEnqueueCopyBuffer(queue, src.get(), dst.get(), 0, 0, size, 0, NULL, NULL));
}

int getKernelNumArgs(cl_kernel k) {
  int nArgs = 0;
  CHECK(clGetKernelInfo(k, CL_KERNEL_NUM_ARGS, sizeof(nArgs), &nArgs, NULL));
  return nArgs;
}

int getWorkGroupSize(cl_kernel k, cl_device_id device, const char *name) {
  size_t size[3];
  CHECK2(clGetKernelWorkGroupInfo(k, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(size), &size, NULL), name);
  return size[0];
}

std::string getKernelArgName(cl_kernel k, int pos) {
  char buf[128];
  size_t size = 0;
  CHECK(clGetKernelArgInfo(k, pos, CL_KERNEL_ARG_NAME, sizeof(buf), buf, &size));
  assert(size >= 0 && size < sizeof(buf));
  buf[size] = 0;
  return buf;
}

class Queue {
  QueueHolder queue;
  
public:
  Queue(cl_queue queue) : queue(queue) {}

  template<typename T> vector<T> read(Buffer &buf, size_t nItems) {
    vector<T> ret(nItems);
    ::read(queue.get(), true, buf, nItems * sizeof(T), ret.data());
    return ret;
  }

  template<typename T> void write(Buffer &buf, const vector<T> &vect) {
    ::write(queue.get(), true, buf, vect.size() * sizeof(T), vect.data());
  }

  template<typename T> void copy(Buffer &src, Buffer &dst, size_t nItems) {
    ::copyBuf(queue.get(), src, dst, nItems * sizeof(T));
  }
  
  void run(cl_kernel kernel, size_t groupSize, size_t workSize, const string &name) {
    ::run(queue.get(), kernel, groupSize, workSize, name);
  }

  void finish() { ::finish(queue.get()); }

  cl_queue get() { return queue.get(); }

  void zero(Buffer &buf, size_t size) {
    assert(size % sizeof(int) == 0);
    int zero = 0;
    CHECK(clEnqueueFillBuffer(queue.get(), buf.get(), &zero, sizeof(zero), 0, size, 0, 0, 0));
    // finish();
  }
};
