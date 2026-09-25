// Copyright (C) Mihai Preda

#include "Event.h"
#include "TimeInfo.h"
#include "log.h"

#include <atomic>
#include <cassert>
#include <exception>
#include <utility>

Event::Event(EventHolder&& e, TimeInfo* tInfo) :
  event{std::move(e)},
  tInfo{tInfo}
{
  assert(tInfo);
}

Event::~Event() {
  // A destructor must not throw (that is std::terminate): if the driver fails the status query here,
  // log it and drop the event.  A failing GPU also fails the queue's next finish/read, which reports it.
  try {
    [[maybe_unused]] bool const done = isComplete();
    assert(done);
  } catch (const std::exception& e) {
    // Log only the first: a lost device fails the query for every event still in the queue.
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) { log("Event: %s\n", e.what()); }
  }
}

bool Event::isComplete() {
  if (!event) { return true; }
  int const status = getEventInfo(event.get());
  if (status == CL_COMPLETE) {
    tInfo->add(getEventNanos(get()));
    event.reset();
  } else if (status < 0) {
    // Terminated abnormally: terminal, and with no valid profiling timestamps to collect.  Retire it
    // rather than keep it in the queue's list for ever; the queue's own wait reports the error.
    event.reset();
  }
  return !event;
}

bool Event::isRunning() { return event && getEventInfo(event.get()) == CL_RUNNING; }
bool Event::isQueued() { return event && getEventInfo(event.get()) == CL_QUEUED; }
bool Event::isSubmitted() { return event && getEventInfo(event.get()) == CL_SUBMITTED; }
