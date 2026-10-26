// Copyright (C) Mihai Preda and George Woltman

#pragma once

#include "common.h"

#include <string>
#include <tuple>
#include <vector>
#include <array>
#include <algorithm>

// We pre-calculate the maximum BPW for a number of fft specs.  From these entries we can either look up or interpolate to get the
// maximum BPW for all variants of an FFT spec.  The variants for which maximum bpw are precomputed are 000, 101, 202, 010, 111, 212.
enum {
NUM_BPW_ENTRIES = 6
};

class Args;

// Format 'n' with a K or M suffix if multiple of 1024 or 1024*1024
string numberK(u64 n);

using KeyVal = std::pair<std::string, std::string>;

enum FFT_TYPES {FFT64=0, FFT3161=1, FFT3261=2, FFT61=3, FFT323161=4, FFT3231=50, FFT6431=51, FFT31=52, FFT32=53};

class FFTShape {
public:
  static std::vector<FFTShape> allShapes(u32 from=0, u32 to = -1);

  static vector<FFTShape> multiSpec(const string& spec);

  enum FFT_TYPES fft_type;
  u32 width  = 0;
  u32 middle = 0;
  u32 height = 0;
  array<float, NUM_BPW_ENTRIES> bpw;

  FFTShape(enum FFT_TYPES t = FFT64, u32 w = 1, u32 m = 1, u32 h = 1);
  FFTShape(enum FFT_TYPES t, const string& w, const string& m, const string& h);
  explicit FFTShape(const string& spec);

  [[nodiscard]] u32 size() const { return width * height * middle * 2; }
  // Radix 8, except radix 4 at 256 unless radix8At256 (-use 256RADIX8_W / 256RADIX8_H, see Args::radix8At256) selects the 4 * 8 * 8
  // radix-8 code.  Radix 8 at 256 (32 threads per line) does less math.  It is faster for FP64 on Radeon VII and TitanV and for GF61 on
  // Radeon VII, but slower for the NTT types on TitanV (more registers, fewer warps, more instruction fetch stalls).
  [[nodiscard]] u32 nW(bool radix8At256) const { return width == 256 && !radix8At256 ? 4 : 8; }
  [[nodiscard]] u32 nH(bool radix8At256) const { return height == 256 && !radix8At256 ? 4 : 8; }
  // Variant 0 (BCAST) is implemented for sizes up to 1K, but not for radix 8 at 256
  [[nodiscard]] bool hasVariant0W(bool radix8At256) const { return width <= 1024 && !(width == 256 && radix8At256); }
  [[nodiscard]] bool hasVariant0H(bool radix8At256) const { return height <= 1024 && !(height == 256 && radix8At256); }

  [[nodiscard]] float minBpw() const { return fft_type != FFT32 ? 3.0f : 1.0f; }
  [[nodiscard]] float maxBpw() const { return *std::ranges::max_element(bpw); }
  [[nodiscard]] u64 maxExp() const { return u64(maxBpw() * size()); }
  [[nodiscard]] std::string spec() const { return (fft_type ? to_string(fft_type) + ':' : "") + numberK(width) + ':' + numberK(middle) + ':' + numberK(height); }

  // The highest bpw for 32-bit carries in carryFused; mul3: in its MUL3 version
  [[nodiscard]] float carry32BPW(bool mul3 = false) const;
  // The same for FFT323161's 64-bit carries (its wider carry is 96 bits)
  [[nodiscard]] float carry64BPW(bool mul3 = false) const;
  // The highest bpw for this FFT type's narrower carry (32 bits, or 64 bits for FFT323161)
  [[nodiscard]] float narrowCarryBPW(bool mul3 = false) const { return fft_type == FFT323161 ? carry64BPW(mul3) : carry32BPW(mul3); }
  // True if the exponent needs the wider carry (64 bits, or 96 bits for FFT323161)
  [[nodiscard]] bool needsLargeCarry(u64 E, bool mul3 = false) const;
  [[nodiscard]] bool isFavoredShape() const;
  // A MIDDLE with an odd factor R times a power of two on an NTT or hybrid FFT/NTT type is done as a Good-Thomas prime-factor
  // transform: R rows of a power-of-two transform, the radix-R in fftMiddleIn/Out uses only an R-th root of unity (in Z/pZ for M31
  // and M61, complex for the FP part of a hybrid) and no twiddles.  R = 3, 7, 9, 11 and 21 divide both M31 - 1 and M61 - 1.  R = 5 and 15
  // divide only M61 - 1 (GF(M31^2) has no 5th roots of unity), so they are for the types with GF61 and without GF31.  The pure FP
  // types use ordinary odd radices.
  [[nodiscard]] static bool pfaType(enum FFT_TYPES t) { return t != FFT64 && t != FFT32; }
  [[nodiscard]] static u32 pfaRadix(u32 m) { return m / (m & (~m + 1)); }     // The odd part of m
  [[nodiscard]] static bool pfaMiddle(u32 m, enum FFT_TYPES t) {
    u32 const r = pfaRadix(m);
    return r == 3 || r == 7 || r == 9 || r == 11 || r == 21 || ((r == 5 || r == 15) && (t == FFT61 || t == FFT3261));
  }
  [[nodiscard]] u32 pfaRadix() const { return pfaRadix(middle); }
  // MIDDLE 1..16, plus the odd middles above 16 that have a radix (fft21.cl).  Those are only run when asked for by an explicit
  // -fft spec: allShapes() leaves them out until their BPW is measured (see the FFTShape constructor).
  [[nodiscard]] static bool validMiddle(u32 m) { return (m >= 1 && m <= 16) || m == 21; }
  [[nodiscard]] bool isPfa() const { return pfaMiddle(middle, fft_type) && pfaType(fft_type); }
};

static const u32 N_VARIANT_W = 3;
static const u32 N_VARIANT_M = 2;
static const u32 N_VARIANT_H = 3;
static const u32 LAST_VARIANT = (N_VARIANT_W - 1) * 100 + (N_VARIANT_M - 1) * 10 + N_VARIANT_H - 1;
inline u32 variant_WMH(u32 v_W, u32 v_M, u32 v_H) { return v_W * 100 + v_M * 10 + v_H; }
inline u32 variant_W(u32 v) { return v / 100; }
inline u32 variant_M(u32 v) { return v % 100 / 10; }
inline u32 variant_H(u32 v) { return v % 10; }
inline u32 next_variant(u32 v) {
  u32 new_v = v + 1; if (variant_H (new_v) < N_VARIANT_H) return (new_v);
  new_v = (v / 10 + 1) * 10; if (variant_M (new_v) < N_VARIANT_M) return (new_v);
  new_v = (v / 100 + 1) * 100; return (new_v);
}

enum CARRY_KIND {CARRY_32=0, CARRY_64=1, CARRY_AUTO=2};

struct FFTConfig {
public:
  // uses, if given, receives the -use settings from the chosen FFT's tune.txt line
  // errorChecked:  the work catches its own errors (PRP's Gerbicz check, proof verification), so an -fft that is too small for E is used anyway
  static FFTConfig bestFit(const Args& args, u64 E, const std::string& spec, vector<KeyVal>* uses = nullptr, bool errorChecked = true);
  // The same choice as a list:  one FFT for an -fft spec or a tune.txt line, else the untuned candidates in order of preference
  // (Task::execute tries the next when a GPU can't be built for one).  Throws "No FFT" if none can handle E.
  static vector<FFTConfig> bestFits(const Args& args, u64 E, const std::string& spec, vector<KeyVal>* uses = nullptr, bool errorChecked = true);

  // Which FP and NTT primes are involved in the FFT
  bool FFT_FP64;
  bool FFT_FP32;
  bool NTT_GF31;
  bool NTT_GF61;
  // bool NTT_NCW;	// Nick Craig-Wood prime not supported (yet?)

  // Size (in bytes) of integer data passed to FFTs/NTTs on the GPU
  u32 WordSize;

  FFTShape shape;
  u32 variant;
  enum CARRY_KIND carry;

  explicit FFTConfig(const string& spec);
  FFTConfig(FFTShape shape, u32 variant, enum CARRY_KIND carry);

  [[nodiscard]] std::string spec() const;
  [[nodiscard]] u32 size() const { return shape.size(); }
  [[nodiscard]] u64 maxExp() const { return u64(maxBpw() * shape.size()); }   // this config's variant and carry, not the shape's best

  [[nodiscard]] float minBpw() const { return shape.minBpw(); }
  [[nodiscard]] float maxBpw() const;
};
