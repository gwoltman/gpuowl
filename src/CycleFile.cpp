// Copyright (C) Mihai Preda

#include "CycleFile.h"
#include "fs.h"

#include <exception>
#include <system_error>

CycleFile::CycleFile(const fs::path& name) :
  name{name},
  f{File::openWrite(name + ".new")}
{}


CycleFile::~CycleFile() {
  if (!f) { return; }
  f.reset();
  if (std::uncaught_exceptions() > uncaughtAtStart) {
    // The write threw (WriteError on a full disk, say): keep the previous file, drop the partial one.
    std::error_code ec;
    fs::remove(name + ".new", ec);
    return;
  }
  fancyRename(name + ".new", name);
}

File* CycleFile::operator->() { return f.operator->(); }
File& CycleFile::operator*() { return f.operator*(); }

void CycleFile::reset() { f.reset(); }
