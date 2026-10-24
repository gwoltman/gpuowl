// Copyright (C) Mihai Preda

#include "Queue.h"
#include "TimeInfo.h"
#include "log.h"

#include <cassert>
#include <chrono>
#include <thread>
#include <utility>

void Events::clearCompleted() { while (!empty() && front().isComplete()) { pop_front(); } }

void Events::synced() {
  clearCompleted();
  assert(empty());
}

Queue::Queue(const Context& context, bool profile, bool auxQueue, int sleepWait) :
  QueueHolder{makeQueue(context.deviceId(), context.get(), profile)},
  hasEvents{profile},
  isAuxQueue(auxQueue),
  context{&context},
  markerEvent{},
  markerQueued(false),
  queueCount(0),
  squareCount(0),
  squareTime(50),
  firstSetTime(true),
  graphRecording(false)
{
  // AMD's finish does not busy wait -- unless ROCm runs out of interrupt events (4096 per device; ROCm/rocm-systems#277), when the
  // waiting thread spins at 100% CPU.  -use SLEEP_WAIT=1 gives AMD the marker and sleep wait too.
  bool const isAmd = isAmdGpu(context.deviceId());
  this->sleepWait = sleepWait < 0 ? !isAmd : sleepWait != 0;

  // Formerly a constant (thus the CAPS).  nVidia is 3% CPU load at 400 or 500, and 35% load at 800 on my Linux machine.
  // AMD is just over 2% load at 1600 and 3200 on the same Linux machine.  Marginally better timings(?) at 3200.
  // With markers up to twice MAX_QUEUE_COUNT entries are outstanding, so AMD's sleep wait uses half its finish count.
  MAX_QUEUE_COUNT = !isAmd ? 500 : this->sleepWait ? 1600 : 3200;          // Queue size for 800 or 125 squarings (if squareKernels = 4)
}

void Queue::writeTE(cl_mem buf, u64 size, const void* data, TimeInfo* tInfo) {
  add(::write(get(), {}, true, buf, size, data, hasEvents), tInfo);
  events.synced();
}

void Queue::fillBufTE(cl_mem buf, size_t patSize, const void* pattern, size_t size, TimeInfo* tInfo) {
  add(::fillBuf(get(), {}, buf, pattern, patSize, size, hasEvents), tInfo);
}

static string status(Events& events) {
  if (events.empty()) { return ""; }
  Event& f = events.front();
  return f.isComplete() ? "C" : f.isQueued() ? "Q" : f.isRunning() ? "R" : f.isSubmitted() ? "S" : "?";
}

void Queue::print() {
  events.clearCompleted();
  log("%s\n", status(events).c_str());
}

void Queue::add(EventHolder&& e, TimeInfo* ti) {
  if (hasEvents) { events.emplace_back(std::move(e), ti); }
  if (isAuxQueue || graphRecording) return;
  queueCount++;
  if (queueCount >= MAX_QUEUE_COUNT) queueMarkerEvent();
}

void Queue::readSync(cl_mem buf, size_t size, void* out, TimeInfo* tInfo) {
  add(read(get(), {}, true, buf, size, out, hasEvents), tInfo);
  events.synced();
}

void Queue::readAsync(cl_mem buf, size_t size, void* out, TimeInfo* tInfo) {
  add(read(get(), {}, false, buf, size, out, hasEvents), tInfo);
}

void Queue::copyBuf(cl_mem src, cl_mem dst, size_t size, TimeInfo* tInfo) {
  add(::copyBuf(get(), {}, src, dst, size, hasEvents), tInfo);
}

void Queue::run(cl_kernel kernel, size_t groupSizeX, size_t workSizeX, size_t workSizeY, TimeInfo* tInfo) {
  add(::run(get(), kernel, groupSizeX, workSizeX, workSizeY, {}, tInfo->name, hasEvents), tInfo);
}

void Queue::finish() {
  assert(!isAuxQueue);
  waitForMarkerEvent();
  ::finish(get());
  events.synced();
  queueCount = 0;
  squareCount = 0;
}

void Queue::queueMarkerEvent() {
  assert(!isAuxQueue);
  waitForMarkerEvent();
  if (queueCount) {
    // AMD GPUs normally have no trouble waiting for a finish without a CPU busy wait.  So, instead of markers and events, simply run finish every now and then.
    if (!sleepWait) {
      finish();
    }
    // Enqueue a marker (nVidia GPUs, and AMD with SLEEP_WAIT=1)
    else {
      markerEvent = enqueueMarker(get());
      ::flush(get());                 // Submit the queued work before sleeping: without it ROCm runs about 7% slower
      markerQueued = true;
      queueCount = 0;
      squareCount = 0;
    }
  }
}

void Queue::waitForMarkerEvent() {
  assert(!isAuxQueue);
  if (!markerQueued) return;
  // By default, nVidia finish causes a CPU busy wait.  Instead, sleep for a while.  Since we know how many items are enqueued after the marker we can make an
  // educated guess of how long to sleep to keep CPU overhead low.
  while (true) {
    int const status = getEventInfo(markerEvent.get());
    if (status == CL_COMPLETE) { break; }
    // A negative status is the error code of a command that terminated abnormally.  It is terminal:
    // waiting for it to reach CL_COMPLETE would sleep for ever.  Report it like any other CL error,
    // so the worker fails instead of going quiet with the GPU idle.
    if (status < 0) { CHECK1(status); }
    // There are usually 4, 7, or 10 kernels per squaring.  Use a rolling average to create a very accurate kernel count.
    // Don't overestimate sleep time.  Divide by much more than the number of kernels.
    std::this_thread::sleep_for(std::chrono::microseconds((squareCount + 1) * squareTime / 2));
  }
  markerQueued = false;
}

void Queue::setSquareTime(int time) {
  assert(!isAuxQueue);
  if (firstSetTime) {                 // Ignore first setSquareTime call.  First measured times are wrong because of startup costs
    firstSetTime = false;
    return;
  }
  time = std::max(time, 30);           // Assume a minimum square time of 30us
  time = std::min(time, 3000);       // Assume a maximum square time of 3000us
  squareTime = time;
}
