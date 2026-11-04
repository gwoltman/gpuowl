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
  u64 k;
  u64 nBits;
  vector<u32> data;
  u32 nErrors;
  double elapsed{};
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

  // The folder of the savefiles
  const fs::path& dir() const { return base; }

  static void clear(u64 exponent);

  // For PRP, we can save a verified save (see save() above) or an unverified save.
  void saveUnverified(const PRPState& s) const;
};
