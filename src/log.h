// Copyright (C) Mihai Preda

#pragma once

#include <string>

#ifdef __GNUC__
void log(const char *fmt, ...) __attribute__ ((format(printf, 1, 2)));
#else
void log(const char *fmt, ...);
#endif

void initLog();
void initLog(const char *);
std::string logContext();
std::string shortTimeStr();

struct LogContext {
  explicit LogContext(const std::string& s);
  ~LogContext();

private:
  std::string part;
};

// The log file and context are thread-local (one log per worker instance),
// so a helper thread starts with neither and its log() lines reach stdout
// alone, without the exponent prefix. A thread that works on a worker's
// behalf — KernelCompiler's parallel compile under CUDA — takes a LogLink
// from the thread that starts it and adopts it for the task's duration.
class File;
struct LogLink {
  File* file;           // the starting thread's log file, borrowed — it outlives the task
  std::string context;
};
LogLink logLink();

struct LogLinkScope {
  explicit LogLinkScope(const LogLink& link);
  ~LogLinkScope();

private:
  File* previous;
  LogContext context;
};
