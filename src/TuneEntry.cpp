#include "TuneEntry.h"
#include "Args.h"
#include "CycleFile.h"

#include <cassert>
#include <cmath>
#include <cinttypes>
#include <algorithm>
#include <cctype>
#include <optional>

// Returns whether *results* was updated.
bool TuneEntry::update(vector<TuneEntry>& results) const {
  if (!std::isfinite(cost)) { return false; }   // a failed timing (see Gpu::timePRP) must never be recorded
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

static fs::path tuneFilePath(const Args& args) {
  fs::path tuneFile = "tune.txt";
  if (!fs::exists(tuneFile)) {
    tuneFile = args.masterDir / "tune.txt";
  }
  return tuneFile;
}

// A tune.txt line is "<cost> <fft spec> # <maxExp>[, KEY=VAL]...".  The KEY=VALs are -use settings for that FFT alone,
// taking priority over config.txt (but not over the command line).
// Returns nothing for a line that is not an entry.  Throws (const char*) for an FFT spec this build can't parse.
static optional<TuneEntry> parseTuneLine(const string& line) {
  char specBuf[32];
  double cost{};
  if (sscanf(line.c_str(), "%lf %31s", &cost, specBuf) < 2) { return {}; }
  vector<KeyVal> uses;
  if (auto pos = line.find('#'); pos != string::npos) {
    for (auto& kv : Args::splitUses(line.substr(pos + 1))) {
      // The first item is the maxExp comment, not a setting
      if (!kv.first.empty() && std::ranges::all_of(kv.first, [](char c) { return isdigit(c); })) { continue; }
      uses.push_back(std::move(kv));
    }
  }
  return TuneEntry{cost, FFTConfig{specBuf}, std::move(uses)};
}

vector<TuneEntry> TuneEntry::readTuneFile(const Args& args) {
  fs::path const tuneFile = tuneFilePath(args);

  // if (!fs::exists(tuneFile)) { log("Tune file %s not found\n", tuneFile.string().c_str()); }

  vector<TuneEntry> results;
  File fi = File::openRead(tuneFile);
  if (!fi) { return {}; }
  fi.allowUnterminatedLastLine();

  for (const string& line : fi) {
    try {
      optional<TuneEntry> const e = parseTuneLine(line);
      if (!e) {
        log("tune.txt line '%s' ignored\n", line.c_str());
        continue;
      }
      // Insert through update() so the list is a proper cost/maxExp frontier whatever order the file is in.  The file
      // was written sorted, but maxExp comes from the bits-per-word tables of the build that reads it, so rows written
      // by an older build can be out of order or dominated by a cheaper row; those are dropped here.
      if (!e->update(results) && args.verbose) {
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

// The settings of the tune.txt line for exactly this FFT, for an FFT given with -fft.  Unlike readTuneFile() this also looks at
// lines a cheaper FFT dominates, so that a user can try settings on any FFT by adding them to its tune.txt line.
vector<KeyVal> TuneEntry::usesFor(const Args& args, const FFTConfig& fft) {
  File fi = File::openRead(tuneFilePath(args));
  if (!fi) { return {}; }
  fi.allowUnterminatedLastLine();

  string const spec = fft.spec();
  for (const string& line : fi) {
    try {
      if (optional<TuneEntry> const e = parseTuneLine(line); e && e->fft.spec() == spec) { return e->uses; }
    } catch (const char*) {}
  }
  return {};
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
    string uses;
    for (const auto& [key, val] : r.uses) { uses += ", " + key + '=' + val; }
    tune->printf("%6.1f %14s # %" PRIu64 "%s\n", r.cost, r.fft.spec().c_str(), maxExp, uses.c_str());
  }
}
