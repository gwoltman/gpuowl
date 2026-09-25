// Copyright (C) Mihai Preda

#include "Kernel.h"
#include "KernelCompiler.h"


Kernel::Kernel(string_view name, KernelCompiler* compiler, TimeInfo* timeInfo, Queue* queue,
       string_view fileName, string_view nameInFile,
       size_t workSize, string_view defines):
  name{name},
  compiler{compiler},
  fileName{fileName},
  nameInFile{nameInFile},
  defines{defines},
  timeInfo{timeInfo},
  queue{queue},
  workSizeX{workSize},
  workSizeY{1}
{
  compiler->declare(this->nameInFile, this->defines);
}

Kernel::~Kernel() = default;

void Kernel::startLoad(KernelCompiler* compiler) {
  assert(!kernel);
  assert(!pendingKernel.valid());
  pendingKernel = compiler->load(fileName, nameInFile, defines);
  deviceId = compiler->deviceId;
}

void Kernel::finishLoad() {
  pendingKernel.wait();
  kernel = pendingKernel.get();
  assert(kernel);
  groupSize = getWorkGroupSize(kernel.get(), deviceId, name.c_str());
  assert(groupSize);
  assert(workSizeX % groupSize == 0);
  // The compiler may have produced code that cannot run as many threads as reqd_work_group_size asks for
  // (e.g. too many registers).  Say so here rather than fail with INVALID_WORK_GROUP_SIZE at the first launch.
  if (int maxSize = getKernelMaxWorkGroupSize(kernel.get(), deviceId, name.c_str()); maxSize > 0 && u32(maxSize) < groupSize) {
    log("%s needs a workgroup of %u but the compiled kernel supports at most %d%s\n", name.c_str(), groupSize, maxSize,
        name.starts_with("kCarryFused") ? "; try a lower -use WMUL" : "");
    throw std::runtime_error("workgroup size too large for kernel "s + name);
  }

  for (auto [pos, arg] : pendingArgs) { setArgs(pos, arg); }
}

void Kernel::setKernelsToExecute(size_t nX, size_t nY) {  // The rare two-dimensional kernel execution
  if (nX == 0) return;  // Use the default work size set at object creation.

  // Make sure kernel is loaded so that we have the groupSize
  if (!kernel) {
    startLoad(compiler);
    finishLoad();
  }
  if (!kernel) { throw std::runtime_error("OpenCL kernel "s + name + " not found"); }

  // For 2D kernels, we only support groupSizeY of 1.  Setting workSizeY to more than one indicates a 2D kernel execution.
  workSizeX = nX * groupSize;
  workSizeY = nY;
}

void Kernel::run() {
  assert(kernel);
  queue->run(kernel.get(), groupSize, workSizeX, workSizeY, timeInfo);
}
