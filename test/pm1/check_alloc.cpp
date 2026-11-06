// Checks of AllocTrac's per-device accounting (src/AllocTrac.h):  two devices (two context keys) with their own totals, limits and
// allowances.  No GPU needed.
#include "AllocTrac.h"

#include <cstdio>
#include <optional>

static int fails = 0;
static void check(bool ok, const char* what) { printf("%s: %s\n", ok ? "OK" : "FAIL", what); fails += !ok; }

static bool tryAlloc(std::optional<AllocTrac>& slot, size_t size, const void* key) {
  try { slot.emplace(size, key); return true; } catch (const std::bad_alloc&) { return false; }
}

int main() {
  constexpr size_t GB = size_t(1) << 30;
  int devA = 0, devB = 0;      // their addresses are the keys
  AllocTrac::setMaxAlloc(15 * GB);
  std::optional<AllocTrac> a1, a2, b1, b2, b3;

  check(tryAlloc(a1, 10 * GB, &devA), "device A: 10 GB under the 15 GB limit");
  check(tryAlloc(b1, 10 * GB, &devB), "device B: 10 GB, not counted against A");
  check(AllocTrac::totalAllocBytes(&devA) == 10 * GB && AllocTrac::totalAllocBytes(&devB) == 10 * GB, "totals are per device");
  check(!tryAlloc(a2, 6 * GB, &devA), "device A: 16 GB refused at the default limit");
  {
    AllocTrac::Allowance allowA{&devA, 8 * GB};
    check(tryAlloc(a2, 6 * GB, &devA), "device A: 16 GB accepted with an 8 GB allowance on A");
    check(!tryAlloc(b2, 6 * GB, &devB), "device B: still refused at 16 GB (A's allowance does not apply to B)");
    {
      AllocTrac::Allowance allowB{&devB, 2 * GB};
      check(tryAlloc(b2, 6 * GB, &devB), "device B: 16 GB with its own 2 GB allowance, while A's is held");
    }
    check(!tryAlloc(b3, 1 * GB, &devB), "device B: back to its 15 GB limit after its allowance ends (A's still held)");
    std::optional<AllocTrac> a3;
    check(tryAlloc(a3, 1 * GB, &devA), "device A: its allowance is still there after B's ended");
  }
  std::optional<AllocTrac> a4;
  check(!tryAlloc(a4, 1 * GB, &devA), "device A: back to 15 GB (17 GB held) after its allowance ends");
  a2.reset();
  check(AllocTrac::totalAllocBytes(&devA) == 10 * GB, "release lowers A's total only");
  check(AllocTrac::totalAllocBytes(&devB) == 16 * GB, "B's total unchanged");
  AllocTrac moved{std::move(*b2)};
  b2.reset();
  check(AllocTrac::totalAllocBytes(&devB) == 16 * GB, "a moved tracker keeps its device and size");
  printf("%d failure(s)\n", fails);
  return fails != 0;
}
