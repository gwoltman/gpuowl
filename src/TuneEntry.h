// Copyright (C) Mihai Preda

#pragma once

#include "FFTConfig.h"

#include <vector>

class Args;

class TuneEntry {
public:
  double cost;
  FFTConfig fft;
  vector<KeyVal> uses;     // -use settings for this FFT alone, from the end of its tune.txt line

  bool update(std::vector<TuneEntry>&) const;
  [[nodiscard]] bool willUpdate(const vector<TuneEntry>&) const;

  static vector<TuneEntry> readTuneFile(const Args& args);
  static vector<KeyVal> usesFor(const Args& args, const FFTConfig& fft);
  static void writeTuneFile(const vector<TuneEntry>&);
};
