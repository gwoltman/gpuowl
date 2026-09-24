// Copyright (C) Mihai Preda.

#include "Worktodo.h"

#include "Task.h"
#include "File.h"
#include "common.h"
#include "Args.h"
#include "fs.h"
#include "Primes.h"

#include <cassert>
#include <string>
#include <optional>
#include <charconv>
#include <cinttypes>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <mutex>

namespace {

bool isHex(const string& s) {
  u32 dummy{};
  const char *end = s.c_str() + s.size();
  auto[ptr, ec] = std::from_chars(s.c_str(), end, dummy, 16);
  return (ptr == end);
}

// Examples:
// PRP=FEEE9DCD59A0855711265C1165C4C693,1,2,124647911,-1,77,0
// DoubleCheck=E0F583710728343C61643028FBDBA0FB,70198703,75,1
// Cert=B2EE67DC0A514753E488794C9DD6F6BD,1,2,82997591,-1,162105
std::optional<Task> parse(const std::string& line) {
  if (line.empty() || line[0] == '#') { return {}; }

  vector<string> topParts = split(line, '=');

  bool isPRP = false;
  bool isLL = false;
  bool isCERT = false;

  if (topParts.size() == 2) {
    const string& kind = topParts.front();
    if (kind == "PRP" || kind == "PRPDC") {
      isPRP = true;
    } else if (kind == "Test" || kind == "DoubleCheck") {
      isLL = true;
    } else if (kind == "Cert") {
      isCERT = true;
    }
  }

  if (isPRP || isLL) {
    vector<string> parts = split(topParts.back(), ',');
    if (!parts.empty() && (parts.front() == "N/A" || parts.front().empty())) {
      parts.erase(parts.begin()); // skip empty AID
    }

    string AID;
    if (!parts.empty() && parts.front().size() == 32 && isHex(parts.front())) {
      AID = parts.front();
      parts.erase(parts.begin());
    }

    // PRP lines are "k,b,n,c,..." and only k=1, b=2, c=-1 is a Mersenne number; anything else (k*2^n-1, 2^n+1, base 3)
    // is not ours and must not be run as exponent k.  The bare "E,..." form is only used by Test=/DoubleCheck= lines.
    bool const mersenne = parts.size() >= 4 && parts[0] == "1" && parts[1] == "2" && (parts[3] == "-1" || parts[3] == "-1\n");
    string const s = mersenne ? parts[2] : ((isLL && !parts.empty()) ? parts[0] : "");

    const char *end = s.c_str() + s.size();
    u64 exp{};
    auto [ptr, _] = from_chars(s.c_str(), end, exp, 10);
    if (ptr != end) { exp = 0; }
    // Task::execute silently retargets a composite exponent to the previous prime.  That is a convenience for
    // "-prp <random>" timing runs; an assignment line with a composite exponent is a mistake, and running a
    // different exponent under its AID would report the wrong result.  Ignore the line instead.
    if (exp > 1000 && !Primes{}.isPrime(exp)) {
      log("worktodo.txt line ignored, exponent %" PRIu64 " is not prime: \"%s\"\n", exp, rstripNewline(line).c_str());
      return {};
    }
    if (exp > 1000) { return {{.kind=isPRP ? Task::PRP : Task::LL, .exponent=exp, .AID=AID, .line=line, .squarings=0}}; }
  }
  if (isCERT) {
    vector<string> parts = split(topParts.back(), ',');
    if (!parts.empty() && parts.front().size() == 32 && isHex(parts.front())) {
      string AID;
      AID = parts.front();
      parts.erase(parts.begin());

      if (parts.size() == 5 && parts[0] == "1" && parts[1] == "2" && parts[3] == "-1") {
	string s = parts[2];
	const char *end = s.c_str() + s.size();
	u64 exp{0};
	from_chars(s.c_str(), end, exp, 10);
	s = parts[4];
	end = s.c_str() + s.size();
	u64 squarings{0};
	from_chars(s.c_str(), end, squarings, 10);
//printf ("Exec cert %d %d \n", (int) exp, (int) squarings);
	if (exp > 1000 && squarings > 100) { return {{.kind=Task::CERT, .exponent=exp, .AID=AID, .line=line, .squarings=u32(squarings) }}; }
      }
    }
  }
  log("worktodo.txt line ignored: \"%s\"\n", rstripNewline(line).c_str());
  return {};
}

// Among the valid tasks from fileName, return the "best" which means the smallest CERT, or otherwise the exponent PRP/LL
static std::optional<Task> bestTask(const fs::path& fileName, bool smallest) {
  optional<Task> best;
  File fi = File::openRead(fileName);
  // A missing file means no work; any other failure to open it (permissions, I/O error) must not look like an empty
  // queue, which ends the run with exit code 0 as if all the work were done.
  if (!fi && errno != ENOENT) {
    log("Can't read '%s': %s\n", fileName.string().c_str(), strerror(errno));
    throw "worktodo file unreadable";
  }
  for (const string& line : fi) {
    optional<Task> task = parse(line);
    // A Cert line whose start-value file is not here cannot run: isCERT would throw and end the worker, and since
    // Cert lines take priority over PRP/LL the worker would be wedged for good.  Skip the line until the file appears.
    if (task && task->kind == Task::CERT && !std::filesystem::exists("M" + to_string(task->exponent) + ".cert")) {
      log("Cert start file M%" PRIu64 ".cert not found; skipping that worktodo line for now\n", task->exponent);
      continue;
    }
    if (task && (!best
                 || (best->kind != Task::CERT && task->kind == Task::CERT)
                 || ((best->kind != Task::CERT || task->kind == Task::CERT) && smallest && task->exponent < best->exponent))) {
      best = task;
    }
  }
  return best;
}

string workName(i32 instance) { return "worktodo-" + to_string(instance) + ".txt"; }

optional<Task> getWork(Args& args, i32 instance) {
  string filename = workName(instance);           // Used for printf statements.  Using fd::path is problematic because it 8-bit char in Linux and 16-bit char in Windows.
  fs::path const localWork = filename;

  // Try to get a task from the local worktodo-<N> file.
  if (optional<Task> task = bestTask(localWork, args.smallest)) { return task; }

  if (args.masterDir.empty()) { log("No work to do found.  Add work to %s.\n", filename.c_str()); return {}; }

  filename = "worktodo.txt";
  fs::path const worktodo = args.masterDir / filename;

  /*
    We need to aquire a task from the global worktodo.txt, and "atomically"
    add the task to the local worktodo-N.txt and remove it from worktodo.txt

    Below we call the global worktodo.txt "global worktodo", and worktodo-N.txt "local worktodo".

    We want to avoid filesystem-based locking, so we approximate it this way:
    1. read the file-size of the global worktodo
    2. read one task from the global worktodo
    3. append the task to the local worktodo
    4. write the new content of the global worktodo without the task to a temporary file
    5. compare the size of the global worktodo with its initial size (as an heuristic to detect modifications to it)
    6a. if the size is not changed, rename the temporary file to global worktodo and done
    6b. if the size is changed (i.e. global worktodo was modified in the meantime):
       7. remove the task from the local worktodo (undo the local task add)
       8. start again (from step 1)
  */

  // The size heuristic below guards against other processes.  Within this process the workers start together and
  // would all read the same file and pick the same task, so serialize the claim itself.
  static std::mutex claimMutex;
  std::lock_guard<std::mutex> const claimLock(claimMutex);

  for (int retry = 0; retry < 2; ++retry) {
    u64 const initialSize = fileSize(worktodo);
    if (!initialSize) { return {}; }

    optional<Task> task = bestTask(worktodo, args.smallest);
    if (!task) { return {}; }

    string const workLine = task->line;
    File::append(localWork, workLine);

    if (deleteLine(worktodo, workLine, initialSize)) {
      return task;
    }

    // Undo add to local worktodo. Attempt twice.
    bool const found = deleteLine(localWork, workLine) || deleteLine(localWork, workLine);
    assert(found);
    if (!found) { return {}; }
  }

  log("Could not extract a task from '%s'\n", filename.c_str());
  // must be tough luck to be preempted twice while mutating the global worktodo
  assert(false);
  return {};
}

} // namespace

std::optional<Task> Worktodo::getTask(Args &args, i32 instance) {
  if (instance == 0) {
    if (args.prpExp) {
      u64 const exp = args.prpExp;
      args.prpExp = 0;
      return Task{.kind=Task::PRP, .exponent=exp};
    } if (args.llExp) {
      u64 const exp = args.llExp;
      args.llExp = 0;
      return Task{.kind=Task::LL, .exponent=exp};
    } if (!args.verifyPath.empty()) {
      auto path = args.verifyPath;
      args.verifyPath.clear();
      return Task{.kind=Task::VERIFY, .verifyPath=path};
    }
  }
  return getWork(args, instance);
}

bool Worktodo::deleteTask(const Task &task, i32 instance) {
  // Some tasks don't originate in worktodo.txt and thus don't need deleting.
  if (task.line.empty()) { return true; }
  return deleteLine(workName(instance), task.line);
}
