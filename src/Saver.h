// Copyright (C) Mihai Preda

#pragma once

#include "common.h"

#include <memory>

class File;
class SaveMan;

struct PRPState {
  static const constexpr char* KIND = "prp";

  u32 exponent;
  u32 k;
  u32 blockSize;
  u64 res64;
  vector<u32> check;
  u32 nErrors;
};

struct LLState {
  static const constexpr char* KIND = "ll";

  u32 exponent;
  u32 k;
  vector<u32> data;
};

template<typename State>
class Saver {
  std::unique_ptr<SaveMan> man;

public:
  Saver(u32 exponent);
  ~Saver();

  State load();
  void save(const State& s);
  void clear();
};
