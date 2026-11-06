// Copyright (C) Mihai Preda.

#pragma once

#include "log.h"
#include <map>
#include <mutex>
#include <new>

using namespace std::string_literals;

// Tracks the GPU memory allocated by Buffers, per device (keyed by the OpenCL / CUDA context the buffer belongs to), and refuses an
// allocation that would take a device over its limit:  the -maxAlloc cap (default 15 GB), the same for every device, plus the
// allowances that are held on that device (P-1 stage 2 without a cap, see Allowance).  Nothing is shared between devices.
class AllocTrac {
  struct Device {
    size_t total = 0;
    size_t allowance = 0;
  };
  static std::mutex mut;
  static std::map<const void*, Device> devices;
  static size_t maxAlloc;

  const void* key{};
  size_t size{};

public:
  AllocTrac() = default;
  AllocTrac(size_t size, const void* key) : key{key}, size(size) {
    if (size) {
      std::lock_guard lock{mut};
      Device& d = devices[key];
      size_t const limit = maxAlloc + d.allowance;
      if (d.total + size >= limit) {
        log("Reached GPU maxAlloc limit %.1f GB\n", float(limit) / (1024 * 1024 * 1024));
        throw std::bad_alloc();
      }
      d.total += size;
    }
  }
  ~AllocTrac() {
    if (size) {
      std::lock_guard lock{mut};
      devices[key].total -= size;
    }
  }

  AllocTrac(const AllocTrac&) = delete;
  void operator=(const AllocTrac&) = delete;

  AllocTrac(AllocTrac&& rhs) noexcept : key(rhs.key), size(rhs.size) { rhs.size = 0; }
  AllocTrac& operator=(AllocTrac&& rhs) noexcept {
    AllocTrac tmp{std::move(rhs)};
    swap(*this, tmp);
    return *this;
  }

  friend void swap(AllocTrac& a, AllocTrac& b) noexcept {
    using std::swap;
    swap(a.key, b.key);
    swap(a.size, b.size);
  }

  static void setMaxAlloc(size_t m) { std::lock_guard lock{mut}; maxAlloc = m; }
  static size_t totalAllocBytes(const void* key) { std::lock_guard lock{mut}; return devices[key].total; }

  // Raises one device's limit by `bytes` while alive (P-1 stage 2 without a -maxAlloc cap:  its budget comes from the device's free
  // memory, not from the default limit).  Allowances of a device add up and are taken back in any order; other devices are unaffected.
  class Allowance {
    const void* key;
    size_t bytes;
  public:
    Allowance(const void* key, size_t bytes) : key{key}, bytes{bytes} { std::lock_guard lock{mut}; devices[key].allowance += bytes; }
    ~Allowance() { std::lock_guard lock{mut}; devices[key].allowance -= bytes; }
    Allowance(const Allowance&) = delete;
    void operator=(const Allowance&) = delete;
  };
};
