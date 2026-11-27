// Copyright (C) Mihai Preda

#include "log.h"
#include "File.h"
#include "timeutil.h"

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <system_error>

thread_local string context;
thread_local vector<string> contextParts;

thread_local File logFile;
// A worker's log file adopted by a helper thread (LogLinkScope); borrowed, never owned.
thread_local File* linkedLogFile = nullptr;

static File stdoutFile{stdout, "stdout"};

string logContext() { return context; }

void initLog(const char *logName) {
  assert(!logFile);
  logFile = File::openAppend(logName);
}

thread_local string progressName;
thread_local bool progressFailed = false;

thread_local bool progressPaused = false;

void initProgress(const string& name) { progressName = name; }

ProgressPause::ProgressPause() : previous{progressPaused} { progressPaused = true; }
ProgressPause::~ProgressPause() { progressPaused = previous; }

void writeProgress(double fraction) {
  if (progressName.empty() || progressPaused) { return; }
  char buf[32];
  // "%.10f" gives fixed notation (never 1e-05) with a '.' separator:  PRPLL never calls setlocale, so printf uses the C locale
  int const n = snprintf(buf, sizeof(buf), "%.10f", std::clamp(fraction, 0.0, 1.0));
  string const tmp = progressName + ".new";
  bool ok = false;
  if (FILE* f = fopen(tmp.c_str(), "wb")) {
    ok = fwrite(buf, 1, n, f) == size_t(n);
    ok = fclose(f) == 0 && ok;
  }
  std::error_code ec;
  if (ok) {
    fs::rename(tmp, progressName, ec);
    if (ec) {        // Some Windows runtimes do not replace an existing file;  a reader holding it open can also block the rename
      fs::remove(progressName, ec);
      fs::rename(tmp, progressName, ec);
    }
  }
  if (!ok || ec) {
    fs::remove(tmp, ec);
    if (!progressFailed) { log("Could not write %s (it is written again at the next log)\n", progressName.c_str()); }
    progressFailed = true;
  } else {
    progressFailed = false;
  }
}

[[maybe_unused]] static string longTimeStr()  { return timeStr("%Y-%m-%d %H:%M:%S %Z"); }
string shortTimeStr() { return timeStr("%Y%m%d %H:%M:%S"); }

static char logBuf[32 * 1024];

void log(const char *fmt, ...) {
  static std::mutex logMutex;

  string const prefix = shortTimeStr() + ' ' + context;

  std::unique_lock const lock(logMutex);
  int pos = 0;
  snprintf(logBuf, sizeof(logBuf), "%s %n", prefix.c_str(), &pos);

  va_list va;
  va_start(va, fmt);
  vsnprintf(logBuf + pos, sizeof(logBuf) - pos, fmt, va);
  va_end(va);
  string_view const s{logBuf};

  if (logFile) { logFile.write(s); } else if (linkedLogFile && *linkedLogFile) { linkedLogFile->write(s); }
  stdoutFile.write(s);
}

LogLink logLink() { return {logFile ? &logFile : linkedLogFile, context}; }

LogLinkScope::LogLinkScope(const LogLink& link) : previous{linkedLogFile}, context{link.context} { linkedLogFile = link.file; }

LogLinkScope::~LogLinkScope() { linkedLogFile = previous; }

LogContext::LogContext(const string& s) : part{s} {
  contextParts.push_back(s);
  context += s;
}

LogContext::~LogContext() {
  assert(!contextParts.empty());
  assert(context.size() >= contextParts.back().size());
  context = context.substr(0, context.size() - contextParts.back().size());
  contextParts.pop_back();
}
