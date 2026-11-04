// Copyright (C) Mihai Preda and George Woltman

#include "Memlock.h"
#include "Signal.h"
#include "log.h"

#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace {

// Returns the handle of the locked file, or -1 if another holder has it.
intptr_t tryLock(const fs::path& path) {
#ifdef _WIN32
  HANDLE h = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0 /* no sharing */, nullptr, OPEN_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
  return h == INVALID_HANDLE_VALUE ? -1 : intptr_t(h);
#else
  int const fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) {
    log("Can't open the memory lock '%s'\n", path.string().c_str());
    throw "memory lock";
  }
  if (flock(fd, LOCK_EX | LOCK_NB) == 0) { return fd; }
  close(fd);
  return -1;
#endif
}

}

Memlock::Memlock(const fs::path& dir, u32 device) : path{dir / ("memlock-"s + to_string(device))} {
  handle = tryLock(path);
  if (handle == -1) {
    log("Waiting for the memory lock '%s'\n", path.string().c_str());
    Signal signal;
    do {
      std::this_thread::sleep_for(std::chrono::seconds(5));
      if (signal.stopRequested()) { throw "stop requested"; }
    } while ((handle = tryLock(path)) == -1);
  }
  log("Acquired the memory lock '%s'\n", path.string().c_str());
}

Memlock::~Memlock() {
#ifdef _WIN32
  CloseHandle(HANDLE(handle));
#else
  close(int(handle));      // releases the flock; the file stays (removing it would race with a waiter that has it open)
#endif
  log("Released the memory lock '%s'\n", path.string().c_str());
}
