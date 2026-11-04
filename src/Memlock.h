// Copyright (C) Mihai Preda and George Woltman

#pragma once

#include "common.h"

#include <filesystem>

// Holds the lock file <dir>/memlock-<device> while alive, so that only one memory-hungry phase (P-1 stage 2) runs per device at a time,
// across workers and processes.  The lock belongs to the open file (flock, or an exclusive open on Windows), so it is released when the
// process ends however it ends.  Waits while another holder has it; throws "stop requested" if a stop is requested while waiting.
class Memlock {
  fs::path path;
  intptr_t handle = -1;

public:
  Memlock(const fs::path& dir, u32 device);
  ~Memlock();

  Memlock(const Memlock&) = delete;
  Memlock& operator=(const Memlock&) = delete;
};
