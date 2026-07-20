// Copyright (C) Mihai Preda

#pragma once

#include "common.h"
#include "clwrap.h"
#include "Context.h"
#include "Args.h"
#include "Event.h"

#include <deque>
#include <vector>

class Args;
class TimeInfo;

class Events : public std::deque<Event> {
public:
  void clearCompleted();
  void synced();
};

class Queue : public QueueHolder {
  Events events;
  bool hasEvents;
  bool isAuxQueue;

  void writeTE(cl_mem buf, u64 size, const void* data, TimeInfo *tInfo);
  void fillBufTE(cl_mem buf, u32 patSize, const void* pattern, u64 size, TimeInfo* tInfo);
  void flush();
  void print();
  void add(EventHolder &&e, TimeInfo* ti);

public:
  const Context* context;

  Queue(const Context& context, bool profile, bool auxQueue = false);

  static int registerThread();
  static int tid();

  template<typename T>
  void write(cl_mem buf, const vector<T>& v, TimeInfo* tInfo) { writeTE(buf, v.size() * sizeof(T), v.data(), tInfo); }

  template<typename T>
  void fillBuf(cl_mem buf, T pattern, u32 size, TimeInfo* tInfo) { fillBufTE(buf, sizeof(T), &pattern, size, tInfo); }

  void run(cl_kernel kernel, size_t groupSizeX, size_t workSizeX, size_t workSizeY, TimeInfo* tInfo);
  void readSync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo);
  void readAsync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo);
  void copyBuf(cl_mem src, cl_mem dst, u32 size, TimeInfo* tInfo);
  void finish();

  EventHolder createSyncEvent() { if (!isAuxQueue && !graphRecording) queueCount++; return enqueueMarker(get()); }  // Enqueue a synchronization event.  Used to sync work among multiple queues.
  void waitForSyncEvent(EventHolder* e) { if (!isAuxQueue && !graphRecording) queueCount++; enqueueMarkerWithWaits(get(), {e->get()}); }  // Wait for a synchronization event to complete.

  void incSquareCount(int n = 1) { squareCount += n; }
  void setSquareTime(int);          // Update the time to do one squaring (in microseconds)

  void beginRecording(void) { graphRecording = true; CHECK1(clGraphBeginRecording(get())); }
  void endRecording(cl_graph *graph) { graphRecording = false; CHECK1(clGraphEndRecording(get(), graph)); }
  void playRecording(cl_graph graph) { CHECK1(clGraphLaunch(graph)); add(EventHolder{}, NULL); }

private:                            // This replaces the "call queue->finish every 400 squarings" code in Gpu.cpp.  Solves the busy wait on nVidia GPUs.
  int MAX_QUEUE_COUNT;              // Queue size before a marker will be enqueued.  Typically, 100 to 1000 squarings.
  EventHolder markerEvent;          // Event associated with an enqueued marker placed in the queue every MAX_QUEUE_COUNT entries and before r/w operations.
  bool markerQueued{false};                // TRUE if a marker and event have been queued
  int queueCount{0};                   // Count of items added to the queue since last marker
  int squareCount{0};                  // Count of squarings/multiplies since last marker queued
  int squareTime{50};                   // Time to do one squaring (in microseconds)
  bool firstSetTime{true};                // Flag so we can ignore first setSquareTime call (which is inaccurate because of all the initial openCL compiles)
  bool graphRecording{false};              // Graph recording in progress.  waitForMarkerEvent and enqueueMarker must be avoided.
  void queueMarkerEvent();          // Queue the marker event
  void waitForMarkerEvent();        // Wait for marker event to complete
};



// Wrapper class for our OpenCL-like extensions invented to provide a clean interface to some nVidia CUDA graphs feature

class Graph {

public:
  Graph() : graph{} {}
  ~Graph() { if (graph) release(graph); }

  bool isSupported(cl_device_id id) { return clIsGraphSupported(id); }
  void beginRecording(Queue *q) { q->beginRecording(); }
  void endRecording(Queue *q) { q->endRecording(&graph); }
  bool isRecorded() { return graph != NULL; }
  void launch(Queue *q) { q->playRecording(graph); }

private:
  cl_graph graph;
};

