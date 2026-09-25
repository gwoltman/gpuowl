#include "TuneEntry.h"
#include "Args.h"
#include "CycleFile.h"

#include <cassert>
#include <cinttypes>

// Returns whether *results* was updated.
bool TuneEntry::update(vector<TuneEntry>& results) const {
  u64 const maxExp = fft.maxExp();
  [[maybe_unused]] bool didErase = false;

  int i{};
  for (i = int(results.size()) - 1; i >= 0 && results[i].cost > cost; --i) {
    if (results[i].fft.maxExp() <= maxExp) {
      results.erase(std::next(results.begin(), i));
      didErase = true;
    }
  }

  if (i >= 0 && results[i].fft.maxExp() >= maxExp) {
    assert(!didErase);
    return false;
  }

  results.insert(std::next(results.begin(), i + 1), *this);
  return true;
}

// Returns whether entry *e* represents an improvement over *results* (i.e. would update the results).
bool TuneEntry::willUpdate(const vector<TuneEntry>& results) const {
  u64 const maxExp = fft.maxExp();
  for (const auto& r : results) {
    if (r.cost > cost) {
      break;
    } if (r.fft.maxExp() >= maxExp) {
      return false;
    }
  }
  return true;
}

vector<TuneEntry> TuneEntry::readTuneFile(const Args& args) {
  fs::path tuneFile = "tune.txt";
  if (!fs::exists(tuneFile)) {
    tuneFile = args.masterDir / "tune.txt";
  }

  // if (!fs::exists(tuneFile)) { log("Tune file %s not found\n", tuneFile.string().c_str()); }

  vector<TuneEntry> results;
  File fi = File::openRead(tuneFile);
  if (!fi) { return {}; }

  for (const string& line : fi) {
    char specBuf[32];
    double cost{};
    if (sscanf(line.c_str(), "%lf %31s", &cost, specBuf) < 2) {
      log("tune.txt line '%s' ignored\n", line.c_str());
      continue;   // otherwise specBuf below is uninitialised
    }
    try {
      FFTConfig const fft{specBuf};
      // Insert through update() so the list is a proper cost/maxExp frontier whatever order the file is in.  The file
      // was written sorted, but maxExp comes from the bits-per-word tables of the build that reads it, so rows written
      // by an older build can be out of order or dominated by a cheaper row; those are dropped here.
      if (!TuneEntry{cost, fft}.update(results) && args.verbose) {
        log("tune.txt line '%s' ignored, a cheaper FFT covers its exponents\n", rstripNewline(line).c_str());
      }
    } catch (const char*) {
      // e.g. a row from an older build whose variant encoding is no longer valid: skip it, keep the other rows
      log("tune.txt line '%s' ignored\n", rstripNewline(line).c_str());
    }
  }
  if (args.verbose && !results.empty()) { log("Read %u entries from %s\n", u32(results.size()), tuneFile.string().c_str()); }
  return results;
}

void TuneEntry::writeTuneFile(const vector<TuneEntry>& results) {
  [[maybe_unused]] u64 prevMaxExp{};
  [[maybe_unused]] double prevCost{};
  CycleFile tune{"tune.txt"};
  for (const TuneEntry& r : results) {
    u64 const maxExp = r.fft.maxExp();
    assert(r.cost >= prevCost && maxExp > prevMaxExp);
    prevCost = r.cost;
    prevMaxExp = maxExp;
    tune->printf("%6.1f %14s # %" PRIu64 "\n", r.cost, r.fft.spec().c_str(), maxExp);
  }
}
