// Copyright 2018 Mihai Preda

#pragma once

class Signal {
  bool isOwner;
  
public:
  Signal();
  ~Signal();
  
  static unsigned stopRequested();
  static void requestStop();          // Ask every worker for the same graceful stop as a SIGTERM
  void release();
};
