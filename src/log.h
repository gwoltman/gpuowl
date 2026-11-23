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

// progress.txt (progress-<instance>.txt with -workers 2 or more):  the fraction of the current task (or P-1 stage) done, 0 to 1, e.g. 0.1234567890.  Written at each log
// point of the worker thread that called initProgress (other threads write nothing), for a wrapper that polls it (BOINC's
// fraction_done_filename).  Written to a temporary file and renamed into place, so a reader never sees part of a value;  not
// synced to disk, since it is only a progress display.
void initProgress(const std::string& name);
void writeProgress(double fraction);

// While one exists, writeProgress does nothing on this thread.  Held around the stage 2 of the P-1 inside a PRP test, whose own
// progress would otherwise show the PRP test at 1 (done) before it goes on:  the file keeps the PRP test's fraction instead.
struct ProgressPause {
  ProgressPause();
  ~ProgressPause();
  ProgressPause(const ProgressPause&) = delete;
  ProgressPause& operator=(const ProgressPause&) = delete;

private:
  bool previous;
};
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
