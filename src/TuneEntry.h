// Copyright (C) Mihai Preda

#pragma once

#include "FFTConfig.h"

#include <vector>

class Args;

// A file of TuneEntry lines:  PRP's tune.txt (cost = us per squaring), or P-1 stage 2's tune-pm1.txt (cost = us per stage 2
// multiplication).  An entry's max exponent is its FFT's, less `margin` bits per word of the FFT's size (stage 2's margin).
struct TuneFile {
  const char* name;
  double margin;

  [[nodiscard]] u64 maxExp(const FFTConfig& fft) const { return margin ? u64(double(fft.maxExp()) - margin * fft.size()) : fft.maxExp(); }
};

inline constexpr TuneFile PRP_TUNE{"tune.txt", 0};
// P-1 stage 2 (-tune pm1):  the difference of two residues has words up to twice as large, so stage 2 runs at least 0.5 bits per word
// below the FFT's limit
inline constexpr TuneFile PM1_TUNE{"tune-pm1.txt", 0.5};

class TuneEntry {
public:
  double cost;
  FFTConfig fft;
  vector<KeyVal> uses;     // -use settings for this FFT alone, from the end of its tune.txt line

  bool update(std::vector<TuneEntry>&, const TuneFile& tf = PRP_TUNE) const;
  [[nodiscard]] bool willUpdate(const vector<TuneEntry>&, const TuneFile& tf = PRP_TUNE) const;

  static vector<TuneEntry> readTuneFile(const Args& args, const TuneFile& tf = PRP_TUNE);
  // Whether the file has a line for exactly this FFT, and that line's settings
  static bool hasLineFor(const Args& args, const FFTConfig& fft, const TuneFile& tf, vector<KeyVal>* uses);
  static vector<KeyVal> usesFor(const Args& args, const FFTConfig& fft, const TuneFile& tf = PRP_TUNE);
  static void writeTuneFile(const vector<TuneEntry>&, const TuneFile& tf = PRP_TUNE);
};
