// Copyright (C) Mihai Preda

#include "Event.h"
#include "TimeInfo.h"

#include <cassert>
#include <utility>

Event::Event(EventHolder&& e, TimeInfo* tInfo) :
  event{std::move(e)},
  tInfo{tInfo}
{
  assert(tInfo);
}

Event::~Event() {
  [[maybe_unused]] bool const done = isComplete();
  assert(done);
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
