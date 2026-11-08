// Copyright (C) Mihai Preda.

#include "Task.h"

#include "Gpu.h"
#include "Args.h"
#include "File.h"
#include "Worktodo.h"
#include "Saver.h"
#include "version.h"
#include "Primes.h"
#include "Pm1.h"
#include "Gcd.h"
#include "Stage2.h"
#include "Pm1Bounds.h"
#include "Pm1Prp.h"
#include "KernelCompiler.h"
#include "Proof.h"
#include "log.h"
#include "Signal.h"
#include "timeutil.h"

#include <cmath>
#include <cassert>
#include <cstdio>
#include <utility>

namespace {

/*
0  => 'Windows16', // 16-bit
1  => 'Windows',   // 32-bit
2  => 'Linux',     // 32-bit
3  => 'Solaris',   // never happened
4  => 'Windows64',
5  => 'WindowsService',
6  => 'FreeBSD',
7  => 'OS/2',
8  => 'Linux64',
9  => 'Mac OS X',
10 => 'Mac OS X 64-bit',
11 => 'Haiku',
12 => 'FreeBSD64',
*/

enum {WIN_16=0, WIN_32, LINUX_32, SOLARIS, WIN_64, WIN_SERV, FREEBSD_32, OS2, LINUX_64, MACOSX_32, MACOSX_64, HAIKU, FREEBSD_64, OS_COUNT};
static_assert(OS_COUNT == 13);

constexpr int platform() {

  const constexpr bool IS_32BIT = (sizeof(void*) == 4);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || defined(__MINGW32__) || defined(__MINGW64__) || defined(__MSYS__)
  return IS_32BIT ? WIN_32 : WIN_64;

#elif __APPLE__
  return IS_32BIT ? MACOSX_32 : MACOSX_64;

#elif __linux__
  return IS_32BIT ? LINUX_32 : LINUX_64;

#else
#error "Unknown OS platform"
#endif

}

struct OsInfo {
  string os;
  string release;
  string arch;
};

[[maybe_unused]] OsInfo getOsInfoMinimum() {
  int const plat = platform();
  string const os = plat == LINUX_64 || plat == LINUX_32 ? "Linux" : plat == 4 ? "Windows" : plat == MACOSX_64 ? "MacOS" : "";
  return {.os=os, .release="", .arch=""};
}

#if __has_include(<sys/utsname.h>)

#include <sys/utsname.h>

OsInfo getOsInfo() {
  utsname buf{};
  uname(&buf);
  return OsInfo{.os=buf.sysname, .release=buf.release, .arch=buf.machine};
}

#else

OsInfo getOsInfo() { return getOsInfoMinimum(); }

#endif

string ffttype(FFTConfig fft) {
  return fft.shape.fft_type == FFT64 ? "FP64" : fft.shape.fft_type == FFT3161 ? "M31+M61" :
         fft.shape.fft_type == FFT61 ? "M61"  : fft.shape.fft_type == FFT3261 ? "FP32+M61" :
         fft.shape.fft_type == FFT31 ? "M31"  : fft.shape.fft_type == FFT3231 ? "FP32+M31" :
         fft.shape.fft_type == FFT32 ? "FP32" : fft.shape.fft_type == FFT6431 ? "FP64+M31" : fft.shape.fft_type == FFT323161 ? "FP32+M31+M61" : "unknown";
}

string json(const vector<string>& v) {
  bool isFirst = true;
  string s = "{";
  for (const std::string& e : v) {
    if (e.empty()) { continue; }
    if (!isFirst) { s += ", "; } else { isFirst = false; }
    s += e;
  }
  return {isFirst ? ""s : (s + '}')};
}

// JSON string literal: escape the quote, the backslash and control characters, so a user name or OS string
// containing them cannot produce an unparseable results line.
string json(const string& s) {
  string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
    else if (c < 0x20) { char buf[8]; snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf; }
    else { out += char(c); }
  }
  return out + '"';
}
string json(int x) { return to_string(x); }
string json(u32 x) { return to_string(x); }
string json(u64 x) { return to_string(x); }

template<typename T> string json(const string& key, const T& value) { return json(key) + ':' + json(value); }

string maybe(const string& key, const string& value) { return value.empty() ? ""s : json(key, value); }

template<typename T> void operator+=(vector<T>& a, const vector<T>& b) { a.insert(a.end(), b.begin(), b.end()); }


vector<string> commonFields(u64 E, const char *worktype, const string &status) {
  return {
    json("status", status),
    json("exponent", E),
    json("worktype", worktype),
  };
}

vector<string> tailFields(const std::string &AID, const Args &args) {
  assert(*VERSION); // version string isn't empty
  OsInfo const os = getOsInfo();
  return {json("program", vector<string>{
                 json("name", "prpll"),
                 json("version", (VERSION[0] == 'v') ? VERSION + 1 : VERSION), // skip leading "v" from version
                 json("port", platform()),
                 json("os", vector<string>{
                   json("os", os.os),
                   maybe("version", os.release),
                   maybe("architecture", os.arch)
                 })
               }),
          maybe("user", args.user),
          maybe("aid", AID),
          maybe("uid", args.uid.empty() ? args.tailDir() : args.uid),
          json("timestamp", timeStr())
  };
}

void writeResult(u32 instance, u64 E, const char *workType, const string &status, const std::string &AID, const Args &args,
                 const vector<string>& extras) {
  fs::path const resultsFile = "results-" + to_string(instance) + ".txt";
  vector<string> fields = commonFields(E, workType, status);
  fields += extras;
  fields += tailFields(AID, args);
  string const s = json(fields);
  log("%s\n", s.c_str());
  File::append(resultsFile, s + '\n');
}

}

void Task::writeResultPRP(FFTConfig fft, const Args &args, u32 instance, bool isPrime, u64 res64, const string& res2048, u32 nErrors, const fs::path& proofPath) const {
  vector<string> fields{json("res64", hex(res64)),
                        json("res2048", res2048),
                        json("residue-type", 1),
                        json("errors", vector<string>{json("gerbicz", nErrors)}),
                        json("fft-type", ffttype(fft)),
                        json("fft-length", fft.size())
  };

  // "proof":{"version":1, "power":6, "hashsize":64, "md5":"0123456789ABCDEF"}, 
  if (!proofPath.empty()) {
    ProofInfo const info = proof::getInfo(proofPath);
    if (info.power > 0) {
      fields.push_back(json("proof", vector<string>{
                              json("version", 1),
                              json("power", info.power),
                              json("hashsize", 64),
                              json("md5", info.md5)
                            }));
    }
  }
  
  writeResult(instance, exponent, "PRP-3", isPrime ? "P" : "C", AID, args, fields);
}

void Task::writeResultLL(FFTConfig fft, const Args &args, u32 instance, bool isPrime, u64 res64, u32 nJacobi) const {
  vector<string> fields{json("res64", hex(res64)),
                        json("fft-type", ffttype(fft)),
                        json("fft-length", fft.size()),
                        json("shift-count", 0),
                        json("error-code", "00000000"), // I don't know the meaning of this
  };
  if (nJacobi) { fields.push_back(json("errors", vector<string>{json("jacobi", nJacobi)})); }

  writeResult(instance, exponent, "LL", isPrime ? "P" : "C", AID, args, fields);
}

void Task::writeResultPM1(FFTConfig fft, const Args &args, u32 instance, u64 B1, u64 B2, const vector<string>& factors, u32 nErrors,
                          u32 nJacobi, bool integrated) const {
  // {"status":"NF", "exponent":45581713, "worktype":"P-1", "b1":50000, "b2":5000000, "fft-length":5120, ..., "factors":["..."]}
  vector<string> fields{json("b1", B1)};
  if (B2 > B1) { fields.push_back(json("b2", B2)); }
  fields.push_back(json("fft-length", fft.size()));
  vector<string> errors{json("gerbicz", nErrors)};
  if (nJacobi) { errors.push_back(json("jacobi", nJacobi)); }
  fields.push_back(json("errors", errors));
  if (integrated) { fields.push_back(json("integrated") + ":true"); }    // stage 1 from the squarings of a PRP test (Pm1Prp)
  if (!factors.empty()) {
    string list = "[";
    for (size_t i = 0; i < factors.size(); ++i) { list += (i ? ", " : "") + json(factors[i]); }
    fields.push_back(json("factors") + ':' + list + ']');
  }
  writeResult(instance, exponent, "P-1", factors.empty() ? "NF" : "F", AID, args, fields);
}

namespace {

// The bounds of a Pfactor line (pm1::chooseBounds), chosen once:  when stage 1 has a savefile, the bounds in it, so that a resume keeps
// them (a version 1 savefile has B1 alone:  B2 is chosen for it).  False when P-1 is not worth its cost for this line even with stage 2's
// largest table (e.g. testsSaved = 0); a line that would pay with a stage 2 this GPU has no room for runs stage 1 alone.
bool pfactorBounds(Task& task, GpuCommon shared, const Gpu& gpu) {
  u64 const E = task.exponent;
  P1State saved{};
  if (fs::exists(fs::current_path() / (string(P1State::KIND) + '-' + to_string(E)))) {
    saved = Saver<P1State>{E, 1000, shared.args->nSavefiles}.load();
  }
  if (!saved.data.empty() && saved.B1 && saved.B2) {
    task.B1 = saved.B1;
    task.B2 = saved.B2;
    log("P-1 bounds from the stage 1 savefile: B1=%" PRIu64 " B2=%" PRIu64 "\n", task.B1, task.B2);
    return true;
  }
  u64 const maxR = Stage2::tableCapacity(shared, gpu, E);
  bool const haveB1 = !saved.data.empty() && saved.B1;
  pm1::Bounds const b = haveB1 ? pm1::chooseB2(E, task.tfBits, task.testsSaved, saved.B1, maxR)
                               : pm1::chooseBounds(E, task.tfBits, task.testsSaved, maxR);
  constexpr u64 maxTable = 1u << 16;      // "stage 2's largest table":  as many entries as P-1 bounds can use
  if (!haveB1 && b.savings <= 0 && (maxR >= maxTable || pm1::chooseBounds(E, task.tfBits, task.testsSaved, maxTable).savings <= 0)) {
    log("P-1 is not worth its cost for %" PRIu64 " (no factor below 2^%g, %g tests saved): nothing to do\n", E, task.tfBits,
        task.testsSaved);
    return false;
  }
  task.B1 = b.B1;
  task.B2 = b.B2;
  log("P-1 bounds%s (no factor below 2^%g, %g tests saved): B1=%" PRIu64 " B2=%" PRIu64 ", chance of a factor %.2f%% (stage 1 %.2f%%), "
      "expected cost %.2f%% of a PRP test, saving %.2f%% of one\n", haveB1 ? " (B1 of the stage 1 savefile)" : "", task.tfBits,
      task.testsSaved, b.B1, b.B2, 100 * b.prob.total(), 100 * b.prob.stage1, 100 * b.cost / double(E), 100 * b.savings / double(E));
  if (b.B2 == b.B1) {
    bool const noRoom = maxR < pm1::stage2MinTable();
    log("P-1 without stage 2 (B2 = B1): %s%s\n", noRoom ? "too little GPU memory for stage 2's smallest table" : "a stage 2 does not pay "
        "for itself here", b.savings <= 0 ? "; stage 1 alone does not pay for itself either, but P-1 with a stage 2 would" : "");
  }
  return true;
}

// Whether the PRP test of E has a savefile (it has started)
bool prpStarted(u64 E) {
  std::error_code ec;
  fs::path const dir = fs::current_path() / to_string(E);
  if (!fs::is_directory(dir, ec)) { return false; }
  string const prefix = to_string(E) + '-';
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    string const name = entry.path().filename().string();
    if (entry.is_regular_file(ec) && name.starts_with(prefix) && name.ends_with(".prp")) { return true; }
  }
  return false;
}

// A -use key's value as a number (0 when absent or not a number)
u64 useValue(const Args& args, const string& key) {
  auto it = args.flags.find(key);
  return it == args.flags.end() ? 0 : strtoull(it->second.c_str(), nullptr, 10);
}

// The bounds of the P-1 inside a PRP test (Pm1Prp), into task.B1 and task.B2 (B1 = 0:  none).  It runs for a worktodo line that asks
// for it (tests saved above 0, or no tf and tests-saved fields at all).
// The bounds are chosen once, as for a Pfactor line, but for stage 1 inside the PRP (its cost, its buffers) and with -pm1TestsSaved,
// not the line's value.  A test with accumulator savefiles keeps their bounds; a test started without P-1 can't have it.  P-1 never stops the PRP test:  when it can't be done (no memory for
// its buffers, no saving), the PRP goes on without it.  Hidden -use keys, for debugging and tests:  PRPPM1=0, no P-1;
// PRPPM1_B1=<B1>[,PRPPM1_B2=<B2>], these bounds in every PRP test whatever its line says.
void prpPm1Bounds(Task& task, GpuCommon shared, const Gpu& gpu) {
  const Args& args = *shared.args;
  task.B1 = task.B2 = 0;
  if (args.value("PRPPM1", 1) == 0) { return; }
  if (u64 const b1 = useValue(args, "PRPPM1_B1")) {
    u64 const b2 = useValue(args, "PRPPM1_B2");
    if (b1 < 2 || (b2 && b2 <= b1) || b2 >= pm1::MAX_B2) {
      log("-use PRPPM1_B1=%" PRIu64 ",PRPPM1_B2=%" PRIu64 ": B1 must be at least 2 and B2, when given, above B1 and at most %" PRIu64 "\n",
          b1, b2, pm1::MAX_B2);
      throw "bad PRPPM1_B1/PRPPM1_B2";
    }
    task.B1 = b1;
    task.B2 = b2;
    return;
  }
  u64 const E = task.exponent;
  if (task.line.empty()) { return; }    // -prp <exponent> from the command line (often a timing run):  no P-1
  // A P-1 under way (or done) in this test keeps its bounds, whatever the line, the memory or -pm1TestsSaved say now
  if (fs::exists(fs::current_path() / (string(P1PrpState::KIND) + '-' + to_string(E)))) {
    Saver<P1PrpState> const saver{E, 1000, args.nSavefiles + 2};
    vector<u64> const ks = saver.listK();
    for (auto it = ks.rbegin(); it != ks.rend(); ++it) {
      if (std::optional<P1PrpState> const s = saver.loadAt(*it)) {
        task.B1 = s->B1;
        task.B2 = s->B2;
        log("P-1 in the PRP test: bounds from its savefile: B1=%" PRIu64 " B2=%" PRIu64 "\n", task.B1, task.B2);
        return;
      }
    }
  }
  if (!task.pm1Fields) {
    // A line without the fields (e.g. PRP=<p> or PRP=AID,1,2,p,-1):  P-1 is taken as needed, with an assumed trial factoring depth
    task.tfBits = pm1::assumedTfBits(E);
    task.testsSaved = 1;
    log("P-1 in the PRP test: the line has no trial factoring depth and tests-saved fields: P-1 is taken as needed, with no factor "
        "assumed below 2^%g\n", task.tfBits);
  }
  if (task.testsSaved <= 0) {
    log("P-1 in the PRP test: the line's tests-saved field is 0 (P-1 is done or not wanted): no P-1\n");
    return;
  }
  if (prpStarted(E)) {
    log("P-1 in the PRP test: the PRP test was started without P-1 (it needs the PRP residues from iteration 1 on): no P-1\n");
    return;
  }
  Pm1Prp::Budget const mem = Pm1Prp::budget(gpu);
  if (mem.maxBufs < pm1::inlineBuffers(pm1::INLINE_MIN_L)) {
    log("P-1 in the PRP test: %s: no P-1\n", mem.tooLittle.c_str());
    return;
  }
  u64 const maxR = Stage2::tableCapacity(shared, gpu, E);
  pm1::Stage1Model const model{.maxBufs = mem.maxBufs, .checkStep = double(Pm1Prp::checkStep(gpu))};
  double const ts = args.pm1TestsSaved;
  pm1::Bounds const b = pm1::chooseBounds(E, task.tfBits, ts, maxR, model);
  if (b.savings <= 0) {
    log("P-1 in the PRP test is not worth its cost for %" PRIu64 " (no factor below 2^%g, %g tests saved; at best B1=%" PRIu64 " B2=%"
        PRIu64 " would save %.2f%% of a PRP test): no P-1\n", E, task.tfBits, ts, b.B1, b.B2,
        100 * b.savings / double(E));
    return;
  }
  task.B1 = b.B1;
  task.B2 = b.B2;
  log("P-1 in the PRP test, bounds (no factor below 2^%g, %g tests saved, room for %" PRIu64 " residue buffers): B1=%" PRIu64 " B2=%"
      PRIu64 ", chance of a factor %.2f%% (stage 1 %.2f%%), expected cost %.2f%% of the PRP test, saving %.2f%% of one\n", task.tfBits,
      ts, mem.maxBufs, b.B1, b.B2, 100 * b.prob.total(), 100 * b.prob.stage1, 100 * b.cost / double(E),
      100 * b.savings / double(E));
  if (b.B2 == b.B1) {
    log("P-1 in the PRP test without stage 2 (B2 = B1): %s\n", maxR < pm1::stage2MinTable() ?
        "too little GPU memory for stage 2's smallest table, or no stage 2 FFT for this exponent" : "a stage 2 does not pay for itself here");
  }
}

}

// P-1:  stage 1 on the GPU, then the stage 1 GCD on the GPU.  A factor ends the assignment, and so does B2 <= B1.  Otherwise stage 2
// (Stage2) and its GCD run on the GPU too, and the result has "b2".  When stage 2 declines (too little GPU memory, an FFT type it
// does not support), -mprimeDir hands the stage 1 residue to mprime for it (and mprime writes the result); without -mprimeDir the
// result is written for B1 alone (no "b2") and the stage 1 savefile is kept.  A stop (signal, -iters) in stage 2 keeps the worktodo
// line and the savefiles:  the next run loads the finished stage 1 and resumes stage 2 from its savefile.
void Task::executePM1(GpuCommon shared, u32 instance, FFTConfig fft, unique_ptr<Gpu> gpu) {
  const Args& args = *shared.args;
  fs::path const& dir = args.mprimeDir;
  if (pfactor && !B1 && !pfactorBounds(*this, shared, *gpu)) {
    Worktodo::deleteTask(*this, instance);
    return;
  }
  Gcd gcd{shared};
  // Named members rather than a structured binding:  clang before 16 cannot capture a structured binding in the lambdas below
  PM1Result const stage1 = gpu->pm1Stage1(*this, gcd);
  const vector<u32>& x = stage1.x;
  u32 const nErrors = stage1.nErrors;
  u32 const nJacobi = stage1.nJacobi;
  gpu.reset();    // the stage 1 buffers are not needed any more

  // x = 1 makes the GCD 2^p - 1:  no information (and not a correct stage 1 for any p that can run here)
  if (pm1::isOne(x)) {
    log("P-1 stage 1 residue is 1: the GCD would be 2^%" PRIu64 " - 1 itself\n", exponent);
    throw "P-1 stage 1 residue is 1";
  }

  // mprime runs the stage 1 GCD only on its way into stage 2: only if the line's B2 is above the B1 in the savefile (otherwise it
  // reports "already tested" and stops), so a hand-off for the GCD alone asks for B2 = B1 + 1.
  auto handOff = [&](u64 handB2, const char* why) {
    pm1::writeMprimeStage1(dir, exponent, B1, x);
    string const work = pm1::mprimePminus1Line(AID, exponent, B1, handB2);
    pm1::appendMprimeWork(dir, work);
    log("P-1 stage 1 residue handed to mprime for %s: %s in '%s', %u Gerbicz errors; %s", why, pm1::mprimeSaveName(exponent).c_str(),
        dir.string().c_str(), nErrors, work.c_str());
    if (B1 < 50000) { log("Note: mprime rejects P-1 lines with B1 < 50000 unless its prime.txt has AllowLowB1=1\n"); }
    Worktodo::deleteTask(*this, instance);
    if (args.clean) { Saver<P1State>::clear(exponent); }
  };

  // The GCD, as a factor (empty when it is 1)
  vector<string> factors;
  auto addFactor = [&](const Words& g, int stage) { factors = pm1::gcdFactor(g, stage, exponent); };

  // A stage 2 savefile for this residue means that this stage 1 GCD already ran and found nothing (stage 2 starts only then)
  bool const stage2Started = B2 > B1 && Stage2::hasSavefile(exponent, B1, B2, x);
  if (stage2Started) {
    log("P-1 stage 1 GCD skipped: stage 2 was started from this residue, so it found no factor\n");
  } else {
    // There is no CPU GCD:  if the GPU one can't run (e.g. its kernels don't build), mprime can still run it
    Words g;
    string gcdError;
    try {
      g = gcd.gcdMinus1(x, u32(exponent));
    } catch (const std::exception& e) {
      if (dir.empty()) { throw; }
      gcdError = e.what();
    } catch (const string& mes) {
      if (dir.empty()) { throw; }
      gcdError = mes;
    }
    if (!gcdError.empty()) {
      log("P-1 stage 1 GCD failed on the GPU (%s)\n", gcdError.c_str());
      handOff(B2 > B1 ? B2 : B1 + 1, "the GCD and stage 2");
      return;
    }
    log("P-1 stage 1 GCD: %.1f s\n", gcd.lastStats().seconds);
    addFactor(g, 1);
    if (factors.empty()) { log("P-1 stage 1 found no factor (B1=%" PRIu64 ")\n", B1); }
  }
  if (nJacobi) { log("P-1 stage 1 had %u Jacobi check failures\n", nJacobi); }

  u64 doneB2 = 0;     // the B2 of the result:  0 for a result of stage 1 alone
  if (factors.empty() && B2 > B1) {
    Stage2Result const r = Stage2::run(shared, exponent, B1, B2, x);
    if (r.status == Stage2Result::STOPPED) {
      // As in stage 1:  the worktodo line stays, and so do the savefiles
      throw "stop requested";
    }
    if (r.status == Stage2Result::DONE) {
      factors = Stage2::accFactor(gcd, r.acc, exponent, B1, B2);
      doneB2 = B2;
    } else if (!dir.empty()) {
      log("P-1 stage 2 is not run here (%s)\n", r.why.c_str());
      handOff(B2, "stage 2");
      return;
    } else {
      log("P-1 stage 2 (B2=%" PRIu64 ") is not run (%s) and there is no -mprimeDir to hand it to.  The result is for B1=%" PRIu64
          " only; the stage 1 residue is kept in '%s'\n", B2, r.why.c_str(), B1,
          (fs::current_path() / (string(P1State::KIND) + '-' + to_string(exponent))).string().c_str());
    }
  }
  writeResultPM1(fft, args, instance, B1, doneB2, factors, nErrors, nJacobi);
  Worktodo::deleteTask(*this, instance);
  if (args.clean && (B2 <= B1 || doneB2 || !factors.empty())) {
    Saver<P1State>::clear(exponent);
    Saver<P2State>::clear(exponent);
  }
}

void Task::writeResultCERT(FFTConfig fft, const Args &args, u32 instance, array <u64, 4> hash, u32 squarings) const {
  string const hexhash = hex(hash[3]) + hex(hash[2]) + hex(hash[1]) + hex(hash[0]);
  vector<string> fields{json("worktype", "Cert"),
                        json("exponent", exponent),
                        json("sha3-hash", hexhash.c_str()),
                        json("squarings", squarings),
                        json("fft-type", ffttype(fft)),
                        json("fft-length", fft.size()),
                        json("shift-count", 0),
                        json("error-code", "00000000"), // I don't know the meaning of this
  };
  fields += tailFields(AID, args);
  string const s = json(fields);
  log("%s\n", s.c_str());
  fs::path const resultsFile = "results-" + to_string(instance) + ".txt";
  File::append(resultsFile, s + '\n');
}

void Task::execute(GpuCommon shared, u32 instance) {
  if (kind == VERIFY) { exponent = proof::getInfo(verifyPath).exp; }

  assert(exponent);

  // Testing exponent 140000001 using FFT 512:15:512 fails with severe round off errors.
  // I'm guessing this is because both the exponent and FFT size are divisible by 3.
  // Here we make sure the exponent is prime.  If not we do not raise an error because it
  // is very common to use command line argument "-prp some-random-exponent" to get a quick
  // timing.  Instead, we output a warning and test a smaller prime exponent.
  {
    Primes const primes;
    if (!primes.isPrime(exponent)) {
      u64 const new_exponent = primes.prevPrime(exponent);
      log("Warning: Exponent %" PRIu64 " is not prime.  Using exponent %" PRIu64 " instead.\n", exponent, new_exponent);
      exponent = new_exponent;
    }
  }

  LogContext const pushContext(std::to_string(exponent));

  // A -mprimeDir that is not there would only be found at the end of stage 1:  fail before any GPU work
  if (kind == PM1) {
    fs::path const& dir = shared.args->mprimeDir;
    std::error_code ec;
    if (!dir.empty() && !fs::is_directory(dir, ec)) {
      log("-mprimeDir '%s' is not a directory: \"%s\"\n", dir.string().c_str(), rstripNewline(line).c_str());
      throw "-mprimeDir is not a directory";
    }
  }

  vector<Pm1Fft> ffts;      // (the FFTs with their -use settings)
  if (kind == PM1) {
    ffts = pm1Ffts(*shared.args, exponent);
  } else {
    vector<KeyVal> fftUses;
    for (const FFTConfig& fft : FFTConfig::bestFits(*shared.args, exponent, shared.args->fftSpec, &fftUses, kind != LL && kind != CERT)) {
      ffts.push_back({fft, fftUses});
    }
  }

  // Without -fft or a tune.txt line there are several untuned candidates.  If the kernels of one can't be built (e.g. an OpenCL
  // compiler that can't build the FP32 code), move on to the next.  Kernels compile on first use, so this can surface after the
  // work has started; the savefiles hold plain residues, so the next FFT resumes from them.  Other errors are not retried.
  // P-1 with an -fft never moves to an FFT the user did not give.
  for (size_t i = 0; ; ++i) {
    bool const last = i + 1 == ffts.size();
    auto const failed = [&](const string& mes) {
      if (!isBuildFailure(mes)) { return false; }
      if (last && kind == PM1 && !shared.args->fftSpec.empty()) {
        log("P-1 stage 1: the kernels of the -fft FFT %s can't be built (%s), and P-1 does not replace it\n",
            ffts[i].fft.spec().c_str(), mes.c_str());
      }
      if (last) { return false; }
      log("Can't build FFT %s (%s), trying the next untuned FFT\n", ffts[i].fft.spec().c_str(), mes.c_str());
      return true;
    };
    try {
      executeWith(shared, instance, ffts[i].fft, ffts[i].uses);
      return;
    } catch (const string& mes) {
      if (!failed(mes)) { throw; }
    } catch (const std::exception& e) {
      if (!failed(e.what())) { throw; }
    }
  }
}

void Task::executeWith(GpuCommon shared, u32 instance, FFTConfig fft, const vector<KeyVal>& fftUses) {
  auto gpu = Gpu::make(exponent, shared, fft, {}, true, fftUses);

  if (kind == VERIFY) {
    Proof const proof{Proof::load(verifyPath)};
    assert(proof.E == exponent);
    bool const ok = proof.verify(gpu.get());
    log("proof '%s' %s\n", verifyPath.c_str(), ok ? "verified" : "failed");
    // -verify is a one-shot job (VERIFY tasks come only from the command line), so a proof that does not
    // check out is a failed run and has to be visible as one: without this the process exits 0 and a
    // caller cannot tell a good proof from a bad one except by reading the log.
    if (!ok) { throw "proof verification failed"; }

  } else if (kind == PRP || kind == LL) {
    bool isPrime;
    if (kind == PRP) {
      // The P-1 inside the PRP test reports through this; a factor ends the test (it is composite) with no PRP result
      Pm1Report const pm1Report = [&](u64 b1, u64 b2, const vector<string>& factors, u32 nErr) {
        writeResultPM1(fft, *shared.args, instance, b1, b2, factors, nErr, 0, true);
      };
      prpPm1Bounds(*this, shared, *gpu);
      auto [tmpIsPrime, res64, nErrors, proofPath, res2048, pm1Factor] = gpu->isPrimePRP(*this, pm1Report);
      if (pm1Factor) {
        log("%" PRIu64 " has a factor: the PRP test is not needed\n", exponent);
        Worktodo::deleteTask(*this, instance);
        if (shared.args->clean) { gpu->clear(true); }
        return;
      }
      isPrime = tmpIsPrime;
      writeResultPRP(fft, *shared.args, instance, isPrime, res64, res2048, nErrors, proofPath);
    } else { // LL
      // The Jacobi check, on by default:  its GCD kernels are built only when it runs
      unique_ptr<Gcd> gcd;
      if (shared.args->value("LL_JACOBI", 1) > 0) {
        // Without them (there is no CPU Jacobi) the test runs unchecked, as before the check existed
        string error;
        try {
#ifdef PRPLL_LL_FAULTS
          if (const char* f = getenv("PRPLL_LL_JACOBI_FAIL"); f && string(f) == "build") { throw "TEST FAULT: the GCD kernels do not build"s; }
#endif
          gcd = make_unique<Gcd>(shared);
        } catch (const std::exception& e) {
          if (Signal::stopRequested()) { throw; }
          error = e.what();
        } catch (const string& e) {
          if (Signal::stopRequested()) { throw; }
          error = e;
        } catch (const char* e) {
          if (Signal::stopRequested()) { throw; }
          error = e;
        }
        if (!error.empty()) { log("Warning: the LL Jacobi check can't run on this GPU (%s); the test goes on without it\n", error.c_str()); }
      }
      LLResult const r = gpu->isPrimeLL(*this, gcd.get());
      isPrime = r.isPrime;
      if (r.nJacobi) { log("LL had %u Jacobi check failures\n", r.nJacobi); }
      writeResultLL(fft, *shared.args, instance, isPrime, r.res64, r.nJacobi);
    }

    Worktodo::deleteTask(*this, instance);

    if (isPrime) {
      log("%" PRIu64 " is PRIME!\n", exponent);
    } else if (shared.args->clean) {
      gpu->clear(kind == PRP);
    }
  } else if (kind == PM1) {
    executePM1(shared, instance, fft, std::move(gpu));
  } else if (kind == CERT) {
    CERTResult const r = gpu->isCERT(*this);
    writeResultCERT(fft, *shared.args, instance, r.hash, squarings);
    if (r.checked && r.nErrors) { log("CERT had %u Gerbicz check failures, all redone\n", r.nErrors); }
    Worktodo::deleteTask(*this, instance);
  } else {
    throw "Unexpected task kind " + to_string(kind);
  }
}
