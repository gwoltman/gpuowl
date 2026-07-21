// Copyright Mihai Preda

#pragma once

#include "common.h"
#include <array>

template <typename H>
class Hash {
  H h;
  
public:
  template <typename... Ts>
  static auto hash(const Ts&... data) {
    Hash hash;
    (hash.update(data),...);
    return std::move(hash).finish();
  }

  Hash& update(const void* data, u32 size) { h.update(data, size); return *this; }

  template<typename T, std::size_t N>
  Hash&& update(const array<T, N>& v) && { h.update(v.data(), u32(N * sizeof(T))); return std::move(*this); }

  void update(u32 x) { h.update(&x, u32(sizeof(x))); }
  void update(u64 x) { h.update(&x, u32(sizeof(x))); }

  template<typename T>
  void update(const vector<T>& v) { h.update(v.data(), u32(v.size() * sizeof(T))); }

  void update(const string& s) {h.update(s.c_str(), u32(s.size())); }
  
  auto finish() && { return std::move(h).finish(); }
};
