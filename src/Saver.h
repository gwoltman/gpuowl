// Copyright (C) Mihai Preda

#pragma once

#include "common.h"

#include <filesystem>
#include <optional>

class File;
class SaveMan;

struct PRPState {
  static const constexpr char* KIND = "prp";

  u64 exponent;
  u64 k;
  u32 blockSize;
  u64 res64;
  vector<u32> check;
  u32 nErrors;
  double elapsed{};
};

struct LLState {
  static const constexpr char* KIND = "ll";

  u64 exponent;
  u64 k;
  vector<u32> data;
  double elapsed{};
};

// A CERT (certification) in progress: the current residue after k of `squarings` squarings.  An empty `data`
// means "no checkpoint": start from the M<E>.cert file.
struct CERTState {
  static const constexpr char* KIND = "cert";

  u64 exponent;
  u64 k;
  u64 squarings;
  vector<u32> data;
  double elapsed{};
};

// P-1 stage 1 after k of the nBits iterations of x <- x^2 * 3^bit (the bits of the exponent of B1, from the top), with x in data.
// Only residues that passed the Gerbicz check are saved.  An empty `data` means "no checkpoint": start from x = 3 at k = 1.
struct P1State {
  static const constexpr char* KIND = "p1";

  u64 exponent;
  u64 B1;
  u64 B2;       // the B2 of the assignment (0 in a savefile of version 1):  a Pfactor line's chosen bounds are kept with the residue
  u64 k;
  u64 nBits;
  vector<u32> data;
  u32 nErrors;
  double elapsed{};
};

// P-1 stage 2 in progress: the accumulator after the k below `k` (see pm1::Stage2Plan).  x64 is the res64 of the stage 1 residue it
// started from.
struct P2State {
  static const constexpr char* KIND = "p2";

  u64 exponent;
  u64 k;
  u64 B1, B2;
  u32 D;
  u64 x64;
  u64 nMul;
  vector<u32> acc;
  double elapsed{};
};

// P-1 stage 1 inside a PRP test (Pm1Prp, -prpPm1):  written beside the verified PRP savefile of iteration k, whose res64 is prp64.  data
// is the product of the PRP residues 3^(2^j) over the set bits j < next of the stage 1 exponent of B1.  stage 0:  accumulating;
// 1:  stage 1 done without a factor, stage 2 to run;  2:  the P-1 result is written;  3:  given up (accumulator or GCD errors).
struct P1PrpState {
  static const constexpr char* KIND = "p1prp";

  u64 exponent;
  u64 B1, B2;
  u64 k;
  u64 prp64;
  u64 next;
  u32 stage;
  u32 nErrors;
  vector<u32> data;
};

template<typename State>
class Saver {
  u64 exponent;
  u32 blockSize;
  fs::path base;
  string prefix;
  u32 nSavefiles;

  State initState();
  void moveToTrash(const fs::path& file);
  void trimFiles();
  fs::path mostRecentSavefile();

public:
  Saver(u64 exponent, u32 blockSize, u32 nSavefiles);
  ~Saver();

  State load();
  void save(const State& s);

  void dropMostRecent();

  // The iterations of the savefiles (not the unverified one), increasing;  the savefile of iteration k, if it can be read;  its removal
  vector<u64> listK() const;
  std::optional<State> loadAt(u64 k) const;
  void removeAt(u64 k) const;

  // The folder of the savefiles
  const fs::path& dir() const { return base; }

  static void clear(u64 exponent);

  // For PRP, we can save a verified save (see save() above) or an unverified save.
  void saveUnverified(const PRPState& s) const;
};
