// Copyright (C) Mihai Preda.

#include "Signal.h"
#include "log.h"

#include <csignal>
#include <filesystem>
#include <system_error>

using namespace std;

static volatile sig_atomic_t signalled = 0;

static void (* volatile oldIntHandler)(int) = nullptr;
static void (* volatile oldTermHandler)(int) = nullptr;

static void signalHandler(int signal) { signalled = signal; }

// A file named "stop" in the run directory asks for the same graceful stop
// a SIGINT does: finish the block, verify it, write the savefile, exit.
// It is the stop a launcher can request where no signal reaches the
// process — a hidden Windows child has no console to deliver a Ctrl-C to,
// and TerminateProcess forfeits the work since the last savefile. The file
// is removed once seen, so the next start is not a stop.
static const char *STOP_FILE = "stop";

static bool stopFileSeen() {
  error_code ec;
  if (!filesystem::exists(STOP_FILE, ec) || ec) { return false; }
  filesystem::remove(STOP_FILE, ec);
  log("Stop requested by the '%s' file\n", STOP_FILE);
  return true;
}

Signal::Signal() {
  if (!oldIntHandler) {
    oldIntHandler = signal(SIGINT, signalHandler);
    // SIGTERM is what service managers and supervisors send first; unhandled
    // it ends the process at once.
    oldTermHandler = signal(SIGTERM, signalHandler);
    isOwner = true;
  }
}

Signal::~Signal() { release(); }

unsigned Signal::stopRequested() {
  if (!signalled && stopFileSeen()) { signalled = SIGTERM; }
  return signalled;
}

void Signal::release() {
  if (isOwner) {
    isOwner = false;
    signal(SIGINT, oldIntHandler);
    signal(SIGTERM, oldTermHandler);
    oldIntHandler = nullptr;
    oldTermHandler = nullptr;
  }
}
