// Copyright Mihai Preda.

#pragma once

#include "common.h"
#include "FFTConfig.h"

#include <string>
#include <map>
#include <set>
#include <filesystem>

namespace fs = std::filesystem;

using KeyVal = std::pair<std::string, std::string>;

class Args {
private:
  int proofPow = -1;
  bool readingConfig = false;          // true while readConfig() parses config.txt lines, so -use keys there aren't taken as command line
  std::set<std::string> cmdlineUses;   // -use keys given on the command line; these beat a tune.txt line's settings

public:
  static vector<KeyVal> splitArgLine(const std::string& inputLine);
  static vector<KeyVal> splitUses(std::string ss);
  static std::string mergeArgs(int argc, char **argv);

  explicit Args(bool silent = false) : silent{silent} {}

  void parse(const string& line);
  void setDefaults();
  [[nodiscard]] bool uses(const std::string& key) const { return flags.contains(key); }
  [[nodiscard]] int value(const std::string& key, int valNotFound = -1) const;
  // The value of a -use key as the kernels compiled for FFT shape fftSpec see it (see clDefines): these flags
  // first, then a "! <fftSpec> ..." line from config.txt.  value() above never sees the per-FFT line.
  [[nodiscard]] int valueFor(const std::string& key, int valNotFound, const std::string& fftSpec) const;
  void readConfig(const fs::path& path);
  // The REGxxxx register limit keys (see Gpu::numRegisters) and NOREG.  The C++ code reads them, never the .cl code.
  [[nodiscard]] static bool isRegisterKey(const string& k) {
    return k == "NOREG" || k.starts_with("REGCF") || k.starts_with("REGMI") || k.starts_with("REGMO") || k.starts_with("REGTS");
  }
  // A copy of these args with a tune.txt line's -use settings for one FFT applied on top of config.txt, but not over the command line.
  [[nodiscard]] Args withFftUses(const string& fftSpec, const vector<KeyVal>& uses) const;
  [[nodiscard]] u32 getProofPow(u64 exponent) const;
  [[nodiscard]] string tailDir() const;

  [[nodiscard]] bool hasFlag(const string& key) const;

  bool silent;
  string user;
  string dir;
  
  string uid;
  string verifyPath;

  string tune;
  vector<string> ctune;

  bool doCtune{};
  bool doTune{};
  bool doZtune{};
  bool carryTune{};
  bool logROE{};

  std::map<std::string, std::string> flags;
  std::map<std::string, vector<KeyVal>> perFftConfig;

  int device = 0;

  bool safeMath = true;
  bool clean = true;
  int verbose = 0;
  bool useCache = false;
  bool profile = false;
  bool smallest = false;

  fs::path masterDir;
  fs::path proofResultDir = "proof";
  fs::path proofToVerifyDir = "proof-tmp";
  fs::path cacheDir = "kernel-cache";
  // fs::path tuneFile = "tune.txt";

  bool keepProof = false;
  u32 proofVerify = 0;      // self-verify a generated proof only if its power is at least this; 0 verifies every proof.

  enum CARRY_KIND carry = CARRY_AUTO;
  u32 workers = 1;
  u32 blockSize = 1000;
  u32 logStep = 20000;
  string fftSpec;

  u64 prpExp = 0;
  u64 llExp = 0;
  
  size_t maxAlloc = 0;

  u32 iters = 0;
  u32 nSavefiles = 4;

  // Extend the range of the FFTs beyond what's safe WRT ROE and CARRY32.
  // The FFT will handle up to fft.maxExp() * fftOverdrive
  // May also take values <1 to lower the max E handled.
  double fftOverdrive = 1;
  
  void printHelp();
};
