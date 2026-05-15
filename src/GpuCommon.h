// Copyright (C) Mihai Preda

#pragma once

class Context;
class Args;
class TrigBufCache;
class Background;

// Data that's normally shared between Gpu instances
class GpuCommon {
public:
  Context* context;
  Args* args;
  TrigBufCache* bufCache;
  Background* background;
};
