// Copyright Mihai Preda

#pragma once

#include "common.h"
#include <array>

struct TrigCoefs {
  u32 scale;
  std::array<double, 8> sinCoefs;
  std::array<double, 8> cosCoefs;
};

TrigCoefs trigCoefs(u32 N);

// The FP32 variant: x = k * scale is kept in [0, 2) and the coefficients are not scaled (see trigCoefsFP32)
struct TrigCoefsFP32 {
  double scale;
  std::array<double, 8> sinCoefs;
  std::array<double, 8> cosCoefs;
};

TrigCoefsFP32 trigCoefsFP32(u32 N);
