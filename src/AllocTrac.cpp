// Copyright (C) Mihai Preda.

#include "AllocTrac.h"

std::mutex AllocTrac::mut;
std::map<const void*, AllocTrac::Device> AllocTrac::devices;
size_t AllocTrac::maxAlloc = size_t(15) * 1024 * 1024 * 1024; // 15 GB, per device
