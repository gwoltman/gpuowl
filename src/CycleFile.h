// Copyright (C) Mihai Preda

#pragma once

#include <exception>

#include "File.h"
#include <filesystem>

/* CycleFile writes the new file to "name.new".
   When done writing, it renames "name.new" to "name".
*/
class CycleFile {
  const fs::path name;
  optional<File> f;

public:
  explicit CycleFile(const fs::path& name);
  ~CycleFile();

  File* operator->();
  File& operator*();

  // Cancel the rename
  void reset();

  // Exceptions in flight when this object was created; if more are in flight when it is destroyed, the write
  // is being unwound (e.g. disk full) and the partial .new file must not replace the previous good file.
  int uncaughtAtStart = std::uncaught_exceptions();
};
