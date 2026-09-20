// Copyright (C) Mihai Preda

#include "Args.h"
#include "Background.h"
#include "Queue.h"
#include "Signal.h"
#include "Task.h"
#include "Worktodo.h"
#include "version.h"
#include "AllocTrac.h"
#include "typeName.h"
#include "log.h"
#include "Context.h"
#include "TrigBufCache.h"
#include "GpuCommon.h"
#include "Gpu.h"
#include "tune.h"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <thread>
#include <utility>
// #include <format> from GCC-13 onwards

// Set when a worker dies on an exception, so that main() can report the failure through the exit code
// even though the other workers (and the process) carry on to a normal end.
static std::atomic<bool> workerFailed{false};

static bool isCleanExit(const char* reason);

static void gpuWorker(GpuCommon shared, i32 instance) {
  // LogContext context{(instance ? shared.args->tailDir() : ""s) + to_string(instance) + ' '};
  // log("Starting worker %d\n", instance);
  if (instance > 0) {
    initLog(("gpuowl-"s + to_string(instance) + ".log").c_str());
    log("PRPLL %s, instance %d\n", VERSION, instance);
  }

  try {
    while (auto task = Worktodo::getTask(*shared.args, instance)) { task->execute(shared, instance); }
  } catch (const char *mes) {
    log("Exception \"%s\"\n", mes);
    if (!isCleanExit(mes)) { workerFailed = true; }
  } catch (const string& mes) {
    log("Exception \"%s\"\n", mes.c_str());
    if (!isCleanExit(mes.c_str())) { workerFailed = true; }
  } catch (const std::exception& e) {
    log("Exception %s: %s\n", typeName(e), e.what());
    workerFailed = true;
  }
}


#if defined(__MINGW32__) || defined(__MINGW64__) || defined(__MSYS__) // for Windows
extern int putenv(char *);
#endif

// The exceptions that end a run on purpose: the user's stop, and the two
// flags that only print. Everything else thrown to main() is a failure.
static bool isCleanExit(const char *reason) {
  return !strcmp(reason, "stop requested") || !strcmp(reason, "help") || !strcmp(reason, "version");
}

int main(int argc, char **argv) {
//!MSVC version support
#ifdef _MSC_VER
  _set_printf_count_output(1);    // I'm not sure what this does (it's from CrazeTheDragon)
#endif

#ifdef __MSYS__
  // I was unable to get putenv to link in MSYS2
#elif defined(__MINGW32__) || defined(__MINGW64__)
  putenv("ROC_SIGNAL_POOL_SIZE=32");
#elif defined(_WIN32)
  _putenv_s("ROC_SIGNAL_POOL_SIZE", "32");  // For MSVC
#else
  // Required to work around a ROCm bug when using multiple queues
  setenv("ROC_SIGNAL_POOL_SIZE", "32", 0);
#endif

  // 0 for a normal end — the queue ran dry, a stop was requested, -h or
  // -version — and 1 for an exception nobody else classified (a kernel that
  // would not compile, a missing device, a bad argument), so a supervisor
  // can tell "out of work" from "cannot run" without parsing the log.
  int exitCode = 0;

  try {
    string const mainLine = Args::mergeArgs(argc, argv);
    {
      Args args{true};
      args.parse(mainLine);
      if (!args.dir.empty()) {
        fs::current_path(args.dir);
      }
    }

    fs::path poolDir;
    {
      Args args{true};
      args.readConfig("config.txt");
      args.parse(mainLine);
      poolDir = args.masterDir;
    }

    initLog("gpuowl-0.log");
    log("PRPLL %s starting\n", VERSION);

    Args args;

    if (!poolDir.empty()) { args.readConfig(poolDir / "config.txt"); }
    args.readConfig("config.txt");
    args.parse(mainLine);
    args.setDefaults();

    if (args.maxAlloc) { AllocTrac::setMaxAlloc(args.maxAlloc); }

    Context context(getDevice(args.device));
    Signal const signal;
    Background background;
    GpuCommon shared;
    shared.context = &context;
    shared.args = &args;
    TrigBufCache bufCache{&context};
    shared.bufCache = &bufCache;
    shared.background = &background;

    if (args.doCtune || args.doTune || args.doZtune || args.carryTune) {
      Tune tune{shared};

      if (args.doCtune) {
        tune.ctune();
      } else if (args.doTune) {
        tune.tune();
      } else if (args.doZtune) {
        tune.ztune();
      } else if (args.carryTune) {
        tune.carryTune();
      }
    } else {
      {
        vector<jthread> threads;
        for (int i = 1; std::cmp_less(i, args.workers); ++i) {
          threads.emplace_back(gpuWorker, shared, i);
        }
        gpuWorker(shared, 0);
      }

      // log("No more work. Add work to worktodo.txt , see -h for details.\n");
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
    exitCode = isCleanExit(mes) ? 0 : 1;
  } catch (const string& mes) {
    log("Exiting because \"%s\"\n", mes.c_str());
    exitCode = isCleanExit(mes.c_str()) ? 0 : 1;
  } catch (const std::exception& e) {
    log("Exiting because of exception %s: %s\n", typeName(e), e.what());
    exitCode = 1;
  }

  if (workerFailed && exitCode == 0) { exitCode = 1; }

  log("Bye\n");
  return exitCode;
}
