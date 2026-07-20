// Copyright (C) Mihai Preda.

#pragma once

#include "Queue.h"
#include "Buffer.h"
#include "common.h"

#include <future>
#include <string>
#include <vector>
#include <utility>

class KernelCompiler;
class TimeInfo;

class Kernel {
  const string name;
  KernelCompiler* compiler;
  const string fileName;
  const string nameInFile;
  const string defines;
  
  TimeInfo *timeInfo;

  Queue* queue;
  size_t workSizeX;
  size_t workSizeY;
  u32 groupSize = 0;
  
  KernelHolder kernel{};
  std::future<KernelHolder> pendingKernel;
  cl_device_id deviceId;
  std::vector<std::pair<u32, cl_mem>> pendingArgs;

public:
  Kernel(string_view name, KernelCompiler* compiler,
         TimeInfo* timeInfo, Queue* queue,
         string_view fileName, string_view nameInFile,
         size_t workSize, string_view defines = "");

  ~Kernel();

  void startLoad(KernelCompiler* compiler);
  void finishLoad();

  // Change which queue is used to run a kernel
  void setQueue(Queue *q) { if (q != NULL) queue = q; }

  // Change number of kernels to execute.  Usually this is set by Gpu.cpp at object creation (by setting the total number of work-items in workSizeX).
  // L2 striping requires the ability to change this setting on-the-fly.  One and two dimensional kernels are supported.
  void setKernelsToExecute(size_t nX, size_t nY = 1);

  template<typename... Args> void setFixedArgs(int pos, const Args &...tail) { setArgs(pos, tail...); }

  template<typename... Args> void operator()(const Args &...args) {
    if (!kernel) {
      startLoad(compiler);
      finishLoad();
    }
    if (!kernel) { throw std::runtime_error("OpenCL kernel "s + name + " not found"); }
    setArgs(0, args...);
    run();
  }

private:
  template<typename T> void setArgs(int pos, const shared_ptr<Buffer<T>>& buf) { setArgs(pos, buf->get()); }
  template<typename T> void setArgs(int pos, const Buffer<T>* buf) { setArgs(pos, buf->get()); }
  template<typename T> void setArgs(int pos, const Buffer<T>& buf) { setArgs(pos, buf.get()); }

  void setArgs(int pos, cl_mem arg) {
    if (kernel) {
      ::setArg(kernel.get(), pos, arg, name);
    } else {
      pendingArgs.push_back({pos, arg});
    }
  }

  template<typename T> void setArgs(int pos, const T &arg) { ::setArg(kernel.get(), pos, arg, name); }
  
  template<typename T, typename... Args> void setArgs(int pos, const T &arg, const Args &...tail) {
    setArgs(pos, arg);
    setArgs(pos + 1, tail...);
  }
  
  void run();
};
