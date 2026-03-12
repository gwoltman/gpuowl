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
  double elapsed;
};

struct LLState {
  static const constexpr char* KIND = "ll";

  u64 exponent;
  u64 k;
  vector<u32> data;
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
  void moveToTrash(fs::path file);
  void trimFiles();
  fs::path mostRecentSavefile();

public:
  Saver(u64 exponent, u32 blockSize, u32 nSavefiles);
  ~Saver();

  State load();
  void save(const State& s);

  void dropMostRecent();

  static void clear(u64 exponent);

  // For PRP, we can save a verified save (see save() above) or an unverified save.
  void saveUnverified(const PRPState& s) const;
};
