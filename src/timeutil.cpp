// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "timeutil.h"

#include <algorithm>
#include <cstdio>
#include <ctime>

// localtime() and gmtime() share one static result buffer, and every worker thread logs,
// so convert into a local struct tm instead.
static std::tm localTm(time_t t) {
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  return tm;
}

static std::tm utcTm(time_t t) {
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  return tm;
}

std::string timeStr(const char *format) {
  std::tm tm = localTm(time(nullptr));
  char buf[64];
  strftime(buf, sizeof(buf), format, &tm);
  return buf;
}

std::string timeStr() {
  std::tm tm = utcTm(time(nullptr));
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);   // equivalent to: "%F %T"
  return buf;
}

std::string formatETA(double secs) {
  u64 const s = u64(std::max(0.0, secs));
  u32 const days = u32(s / 86400), hours = u32(s / 3600 % 24), mins = u32(s / 60 % 60), sec = u32(s % 60);
  char buf[64];
  if (days) { snprintf(buf, sizeof(buf), "%ud %02u:%02u:%02u", days, hours, mins, sec); }
  else      { snprintf(buf, sizeof(buf), "%02u:%02u:%02u", hours, mins, sec); }
  return buf;
}
