// Copyright (C) Mihai Preda.

#pragma once

#include "File.h"
#include "common.h"
#include <cinttypes>

namespace fs = std::filesystem;

class Gpu;

struct ProofInfo {
  u32 power;
  u64 exp;
  string md5;
};

namespace proof {

array<u64, 4> hashWords(u64 E, const Words& words);

array<u64, 4> hashWords(u64 E, array<u64, 4> prefix, const Words& words);

string fileHash(const fs::path& filePath);

ProofInfo getInfo(const fs::path& proofFile);

}

class Proof {  
public:
  const u64 E;
  const Words B;
  const vector<Words> middles;

  /*Example header:
    PRP PROOF\n
    VERSION=2\n
    HASHSIZE=64\n
    POWER=8\n
    NUMBER=M216091\n
  */
  static const constexpr char* HEADER_v2 = "PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=%u\nNUMBER=M%" PRIu64 "%c";

  static Proof load(const fs::path& path);

  void save(const fs::path& proofResultDir) const;

  fs::path file(const fs::path& proofDir) const;
  
  bool verify(Gpu *gpu, const vector<u64>& hashes = {}) const;
};

class ProofSet {
public:
  const u64 E;
  const u32 power;
  
private:  
  vector<u64> points;  
  
  bool isValidTo(u64 limitK) const;

  static bool canDo(u64 E, u32 power, u64 currentK);

  mutable decltype(points)::const_iterator cacheIt{};

  bool fileExists(u64 k) const;

  static fs::path proofPath(u64 E) { return fs::path(to_string(E)) / "proof"; }
public:
  
  static u32 bestPower(u64 E);
  static u32 effectivePower(u64 E, u32 power, u64 currentK);
  static double diskUsageGB(u64 E, u32 power);
  static bool isInPoints(u64 E, u32 power, u64 k);
  
  ProofSet(u64 E, u32 power);

  u64 next(u64 k) const;

  static void save(u64 E, u32 power, u64 k, const Words& words);
  static Words load(u64 E, u32 power, u64 k);

  void save(u64 k, const Words& words) const { return save(E, power, k, words); }
  Words load(u64 k) const { return load(E, power, k); }

  std::pair<Proof, vector<u64>> computeProof(Gpu *gpu) const;
};
