// Tune file round trips (no GPU):  TuneEntry read and written back through tune.txt (PRP) and tune-pm1.txt (P-1 stage 2), from the
// current directory.  Links prpll's objects (build prpll first).  Also builds against an older build's objects (the PRP part only uses
// the API both have), so that a tune.txt round trip can be compared byte for byte between the two builds.
//
//   check_tune prp       reads ./tune.txt and writes it back
//   check_tune carry <spec> <E>...   the spec's carry limits and the FFT stage 2 runs for each E (Stage2::configFor)
//   check_tune pm1 <E>   reads ./tune-pm1.txt, prints each entry's max exponent with stage 2's margin and the entry chosen for E,
//                        then writes the file back

#include "TuneEntry.h"
#include "Args.h"
#include "Stage2.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: check_tune prp | pm1 <E>\n"); return 2; }
  Args args{true};
  string const what = argv[1];
  if (what == "prp") {
    vector<TuneEntry> const r = TuneEntry::readTuneFile(args);
    printf("%u entries\n", u32(r.size()));
    TuneEntry::writeTuneFile(r);
    return 0;
  }
#ifndef OLD_API
  if (what == "pm1" && argc > 2) {
    u64 const E = strtoull(argv[2], nullptr, 10);
    vector<TuneEntry> const r = TuneEntry::readTuneFile(args, PM1_TUNE);
    for (const TuneEntry& e : r) {
      string uses;
      for (const auto& [k, v] : e.uses) { uses += ' ' + k + '=' + v; }
      printf("%8.1f %-20s maxExp %" PRIu64 " stage2 %" PRIu64 "%s\n", e.cost, e.fft.spec().c_str(), e.fft.maxExp(), PM1_TUNE.maxExp(e.fft),
             uses.c_str());
    }
    for (const TuneEntry& e : r) {
      if (PM1_TUNE.maxExp(e.fft) >= E) { printf("for %" PRIu64 ": %s\n", E, e.fft.spec().c_str()); break; }
    }
    TuneEntry::writeTuneFile(r, PM1_TUNE);
    return 0;
  }
  if (what == "carry" && argc > 2) {      // the carry limits of an FFT spec, and stage 2's choice for the given exponents
    FFTConfig const fft{argv[2]};
    printf("%s: maxBpw %.3f narrowCarryBPW %.3f\n", fft.spec().c_str(), fft.maxBpw(), fft.shape.narrowCarryBPW());
    for (int i = 3; i < argc; ++i) {
      u64 const E = strtoull(argv[i], nullptr, 10);
      printf("  E %" PRIu64 " (%.3f bpw): stage 2 %s\n", E, E / double(fft.size()), Stage2::configFor(fft, E).spec().c_str());
    }
    return 0;
  }
#endif
  fprintf(stderr, "usage: check_tune prp | pm1 <E>\n");
  return 2;
}
