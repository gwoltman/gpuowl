// Copyright (C) Mihai Preda

#include "Saver.h"
#include "CycleFile.h"
#include "File.h"
#include "fs.h"

#include <charconv>
#include <cmath>
#include <type_traits>
#include <cinttypes>
#include <filesystem>
#include <algorithm>
#include <utility>

namespace {

// E, k, block-size, res64, nErrors, CRC
static constexpr const char *PRP_v12 = "OWL PRP 12 %" PRIu64 " %" PRIu64 " %u %016" SCNx64 " %u %u\n";

// Anticipated next version of the header.
// Has general number form N=k*b^E+c, and labels for values.
static constexpr const char *PRP_v13 = "OWL PRP 13 N=1*2^%" PRIu64 "-1 k=%" PRIu64 " block=%u res64=%016" SCNx64 " err=%u time=%lf\n";
// static constexpr const char *PRP_v13_PRI = "OWL PRP 13 N=1*2^%u-1 k=%u block=%u res64=%016" PRIx64 " err=%u time=%.0lf\n";

// E, k, CRC
static constexpr const char *LL_v1 = "OWL LL 1 E=%" PRIu64 " k=%" PRIu64 " CRC=%u\n";

// Anticipated next version.
// Push version number to sync it with PRP.
static constexpr const char *LL_v13 = "OWL LL 13 N=1*2^%" PRIu64 "-1 k=%" PRIu64 " time=%lf\n";
static constexpr const char *P1_v1 = "OWL P1 1 N=1*2^%" PRIu64 "-1 B1=%" PRIu64 " k=%" PRIu64 " bits=%" PRIu64 " err=%u time=%lf\n";
static constexpr const char *P1_v2 = "OWL P1 2 N=1*2^%" PRIu64 "-1 B1=%" PRIu64 " B2=%" PRIu64 " k=%" PRIu64 " bits=%" PRIu64 " err=%u time=%lf\n";
static constexpr const char *P2_v1 = "OWL P2 1 N=1*2^%" PRIu64 "-1 k=%" PRIu64 " B1=%" PRIu64 " B2=%" PRIu64 " D=%u x64=%016" SCNx64 " muls=%" PRIu64 " time=%lf\n";
static constexpr const char *P1PRP_v1 = "OWL P1PRP 1 N=1*2^%" PRIu64 "-1 B1=%" PRIu64 " B2=%" PRIu64 " k=%" PRIu64 " prp64=%016" SCNx64 " next=%" PRIu64 " stage=%u err=%u\n";
static constexpr const char *CERT_v1 = "OWL CERT 1 N=1*2^%" PRIu64 "-1 k=%" PRIu64 " squarings=%" PRIu64 " time=%lf\n";

struct BadHeaderError : std::runtime_error {
  string name;
  explicit BadHeaderError(string n) : std::runtime_error("bad savefile header: " + n), name(std::move(n)) {}
};

// The header's exponent sizes the residue read that follows; a corrupt header must not be allowed to size it.
void checkExponent(u64 got, u64 want, const string& header) {
  if (got != want) {
    log("savefile header exponent %" PRIu64 " does not match %" PRIu64 ": \"%s\"\n", got, want, rstripNewline(header).c_str());
    throw BadHeaderError{header};
  }
}

bool startsWith(const string& s, const string& prefix) {
  return s.starts_with(prefix);
}

vector<u64> savefiles(const fs::path& dir, const string& prefix, const string& kind) {
  vector<u64> v;
  for (const auto& entry: fs::directory_iterator(dir)) {
    if (entry.is_regular_file()) {
      string const filename = entry.path().filename().string();
      auto dot = filename.find('.');
      if (dot != string::npos && startsWith(filename, prefix) && filename.substr(dot + 1) == kind) {
        assert(dot > prefix.size());
        string id = filename.substr(prefix.size(), dot - prefix.size());
        if (id == "unverified") { continue; }
        u64 k = 0;
        const char* first = id.data();
        const char* end   = first + id.size();
        auto res = from_chars(first, end, k);
        if (res.ptr != end) {
          log("Savefile ignored: '%s' '%s' %p %p\n", filename.c_str(), id.c_str(), end, res.ptr);
        } else {
          v.push_back(k);
        }
      }
    }
  }
  std::ranges::sort(v);
  return v;
}

string str9(u64 k) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%09" PRIu64, k);
  return buf;
}

fs::path pathFor(const fs::path& base, const string& prefix, const string& kind, u64 k) {
  return base / (prefix + str9(k) + '.' + kind);
}

fs::path pathUnverified(const fs::path& base, const string& prefix) {
  return base / (prefix + "unverified.prp");
}

// find the "most advanced" file in dir with a name of the form
// <prefix><id>.<kind>
// e.g.: 125784077-010000000.prp
fs::path findLast(const fs::path& dir, const string& prefix, const string& kind) {
  vector<u64> v = savefiles(dir, prefix, kind);
  if (v.empty()) { return {}; }
  u64 const lastK = v.back();
  fs::path path = pathFor(dir, prefix, kind, lastK);
  assert(is_regular_file(path));
  return path;
}

PRPState readState(const PRPState& expected, File fi) {
  u64 exponent{}, k{};
  u32 blockSize{}, nErrors{};
  u64 res64{};
  double elapsed{};

  string const header = fi.readLine();

  if (sscanf(header.c_str(), PRP_v13, &exponent, &k, &blockSize, &res64, &nErrors, &elapsed) == 6) {
    checkExponent(exponent, expected.exponent, header);
    return {.exponent=exponent, .k=k, .blockSize=blockSize, .res64=res64, .check=fi.readChecked<u32>(nWords(exponent)), .nErrors=nErrors, .elapsed=elapsed};
  }

  u32 crc{};
  if (sscanf(header.c_str(), PRP_v12, &exponent, &k, &blockSize, &res64, &nErrors, &crc) == 6) {
    checkExponent(exponent, expected.exponent, header);
    return {.exponent=exponent, .k=k, .blockSize=blockSize, .res64=res64, .check=fi.readWithCRC<u32>(nWords(exponent), crc), .nErrors=nErrors, .elapsed=0};
  }

  log("Loading PRP from '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
  throw BadHeaderError{fi.name};
}

LLState readState(const LLState& expected, File fi) {
  u64 exponent{}, k{};
  double elapsed{};

  string const header = fi.readLine();

  if (sscanf(header.c_str(), LL_v13, &exponent, &k, &elapsed) == 3) {
    checkExponent(exponent, expected.exponent, header);
    return {.exponent=exponent, .k=k, .data=fi.readChecked<u32>(nWords(exponent)), .elapsed=elapsed};
  }

  u32 crc{};
  if (sscanf(header.c_str(), LL_v1, &exponent, &k, &crc) == 3) {
    checkExponent(exponent, expected.exponent, header);
    return {.exponent=exponent, .k=k, .data=fi.readWithCRC<u32>(nWords(exponent), crc), .elapsed=0};
  }

  log("Loading LL from '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
  throw BadHeaderError{fi.name};
}

void writeState(const File& fo, const PRPState& state) {
  assert(state.check.size() == nWords(state.exponent));
  if (fo.printf(PRP_v13, state.exponent, state.k, state.blockSize, state.res64, state.nErrors, state.elapsed) <= 0) {
    throw WriteError{fo.name};
  }
  fo.writeChecked(state.check);
}

CERTState readState([[maybe_unused]] const CERTState& dummy, File fi) {
  u64 exponent{}, k{}, squarings{};
  double elapsed{};

  string const header = fi.readLine();

  if (sscanf(header.c_str(), CERT_v1, &exponent, &k, &squarings, &elapsed) == 4) {
    return {.exponent=exponent, .k=k, .squarings=squarings, .data=fi.readChecked<u32>(nWords(exponent)), .elapsed=elapsed};
  }

  log("Loading CERT from '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
  throw BadHeaderError{fi.name};
}

void writeState(const File& fo, const CERTState& state) {
  assert(state.data.size() == nWords(state.exponent));
  if (fo.printf(CERT_v1, state.exponent, state.k, state.squarings, state.elapsed) <= 0) {
    throw WriteError{fo.name};
  }
  fo.writeChecked(state.data);
}

P2State readState(const P2State& expected, File fi) {
  P2State s{};
  string const header = fi.readLine();
  if (sscanf(header.c_str(), P2_v1, &s.exponent, &s.k, &s.B1, &s.B2, &s.D, &s.x64, &s.nMul, &s.elapsed) == 8) {
    checkExponent(s.exponent, expected.exponent, header);
    s.acc = fi.readChecked<u32>(nWords(s.exponent));
    return s;
  }
  log("Loading P2 from '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
  throw BadHeaderError{fi.name};
}

void writeState(const File& fo, const P2State& state) {
  assert(state.acc.size() == nWords(state.exponent));
  if (fo.printf(P2_v1, state.exponent, state.k, state.B1, state.B2, state.D, state.x64, state.nMul, state.elapsed) <= 0) {
    throw WriteError{fo.name};
  }
  fo.writeChecked(state.acc);
}

void writeState(const File& fo, const LLState& state) {
  assert(state.data.size() == nWords(state.exponent));
  if (fo.printf(LL_v13, state.exponent, state.k, state.elapsed) <= 0) {
    throw WriteError{fo.name};
  }
  fo.writeChecked(state.data);
}

P1State readState(const P1State& expected, File fi) {
  u64 exponent{}, B1{}, B2{}, k{}, nBits{};
  u32 nErrors{};
  double elapsed{};

  string const header = fi.readLine();

  if (sscanf(header.c_str(), P1_v2, &exponent, &B1, &B2, &k, &nBits, &nErrors, &elapsed) == 7
      || sscanf(header.c_str(), P1_v1, &exponent, &B1, &k, &nBits, &nErrors, &elapsed) == 6) {
    checkExponent(exponent, expected.exponent, header);
    return {.exponent=exponent, .B1=B1, .B2=B2, .k=k, .nBits=nBits, .data=fi.readChecked<u32>(nWords(exponent)), .nErrors=nErrors,
            .elapsed=elapsed};
  }

  log("Loading P-1 from '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
  throw BadHeaderError{fi.name};
}

void writeState(const File& fo, const P1State& state) {
  assert(state.data.size() == nWords(state.exponent));
  if (fo.printf(P1_v2, state.exponent, state.B1, state.B2, state.k, state.nBits, state.nErrors, state.elapsed) <= 0) {
    throw WriteError{fo.name};
  }
  fo.writeChecked(state.data);
}

P1PrpState readState(const P1PrpState& expected, File fi) {
  P1PrpState s{};
  string const header = fi.readLine();
  if (sscanf(header.c_str(), P1PRP_v1, &s.exponent, &s.B1, &s.B2, &s.k, &s.prp64, &s.next, &s.stage, &s.nErrors) == 8) {
    checkExponent(s.exponent, expected.exponent, header);
    s.data = fi.readChecked<u32>(nWords(s.exponent));
    return s;
  }
  log("Loading P-1 (in PRP) from '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
  throw BadHeaderError{fi.name};
}

void writeState(const File& fo, const P1PrpState& state) {
  assert(state.data.size() == nWords(state.exponent));
  if (fo.printf(P1PRP_v1, state.exponent, state.B1, state.B2, state.k, state.prp64, state.next, state.stage, state.nErrors) <= 0) {
    throw WriteError{fo.name};
  }
  fo.writeChecked(state.data);
}

double roundNumberScore(u64 x) {
  if (x == 0) { return 1; }

  double score = 0;
  while (x % 10 == 0) {
    x /= 10;
    ++score;
  }
  if (x % 5 == 0) { score += .7; }
  if (x % 2 == 0) { score += .3; }
  return score;
}

} // namespace

template<> PRPState Saver<PRPState>::initState() {
  return {.exponent=exponent, .k=0, .blockSize=blockSize, .res64=3, .check=makeWords(exponent, 1), .nErrors=0, .elapsed=0};
}

template<> LLState Saver<LLState>::initState() {
  return {.exponent=exponent, .k=0, .data=makeWords(exponent, 4), .elapsed=0};
}

template<> P2State Saver<P2State>::initState() {
  return {.exponent=exponent, .k=0, .B1=0, .B2=0, .D=0, .x64=0, .nMul=0, .acc={}, .elapsed=0};   // no checkpoint
}

template<> CERTState Saver<CERTState>::initState() {
  return {.exponent=exponent, .k=0, .squarings=0, .data={}, .elapsed=0};   // no checkpoint: caller starts from the .cert file
}

template<> P1State Saver<P1State>::initState() {
  return {.exponent=exponent, .B1=0, .B2=0, .k=0, .nBits=0, .data={}, .nErrors=0, .elapsed=0};   // no checkpoint: caller starts from 3
}

template<> P1PrpState Saver<P1PrpState>::initState() {
  return {.exponent=exponent, .B1=0, .B2=0, .k=0, .prp64=0, .next=0, .stage=0, .nErrors=0, .data={}};   // no checkpoint
}

// ---- Saver ----

template<typename State>
Saver<State>::Saver(u64 exponent, u32 blockSize, u32 nSavefiles) :
  exponent{exponent},
  blockSize{blockSize},
  prefix{to_string(exponent) + '-'},
  nSavefiles{nSavefiles}
{
  assert(blockSize && (blockSize % 100 == 0) && (10'000 % blockSize == 0));

  base = std::is_same_v<State, PRPState> ?
        fs::current_path() / to_string(exponent)
      : fs::current_path() / (string(State::KIND) + '-' + to_string(exponent));

  if (!fs::exists(base)) { fs::create_directories(base); }
}

template<typename State>
Saver<State>::~Saver() = default;

template<typename State>
void Saver<State>::clear(u64 exponent) {
  error_code dummy;
  fs::path const base = std::is_same_v<State, PRPState> ?
        fs::current_path() / to_string(exponent)
      : fs::current_path() / (string(State::KIND) + '-' + to_string(exponent));
  fs::remove_all(base, dummy);
}

template<typename State>
void Saver<State>::moveToTrash(const fs::path& src) {
  log("Removing bad savefile '%s'\n", src.string().c_str());
  fancyRename(src, src + ".bad"s);
}

template<typename State>
fs::path Saver<State>::mostRecentSavefile() {
  fs::path path = pathUnverified(base, prefix);
  error_code dummy;
  if (!fs::is_regular_file(path, dummy)) {
    path = findLast(base, prefix, State::KIND);
  }
  return path;
}

template<typename State>
State Saver<State>::load() {
  for (int i = 0; i < 2; ++i) {
    fs::path const path = mostRecentSavefile();

    if (path.empty()) {
      // no savefiles at all
      return initState();
    }

    if (File fi{File::openRead(path)}; fi) {
      try {
        State expected{};
        expected.exponent = exponent;
        State state = readState(expected, std::move(fi));
        assert(state.exponent == exponent);
        if (state.exponent == exponent) {
          return state;
        }
      } catch (const CRCError& e) {
      } catch (const BadHeaderError& e) {
      } catch (const ReadError& e) {
      }
    }

    // if for any reason reading the savefile failed, move it out of the way and retry
    log("Bad savefile '%s'\n", path.string().c_str());
    moveToTrash(path);
  }
  throw "bad savefile retry";
}

template<typename State>
void Saver<State>::trimFiles() {
  vector<u64> v = savefiles(base, prefix, State::KIND);

  assert(nSavefiles > 0);
  while (v.size() > nSavefiles) {
    int bestIdx = -1;
    double bestSpan = 1e20;
    u64 prevK = 0;

    for (u32 i = 0; i < v.size() - 1; ++i) {
      u64 const k = v[i];
      double const niceBias = std::min(1.0, roundNumberScore(k) - 4);
      double const span = (v[i + 1] - prevK) * niceBias;
      prevK = k;
      if (span < bestSpan) {
        bestSpan = span;
        bestIdx = i;
      }
    }
    assert(bestIdx >= 0);
    u64 const k = v[bestIdx];
    // log("Deleting savefile %" PRIu64 "\n", k);
    fs::path const path = pathFor(base, prefix, State::KIND, k);
    fs::remove(path);
    v.erase(v.begin() + bestIdx);
  }
}

template<typename State>
void Saver<State>::save(const State& state) {
  fs::path const path = pathFor(base, to_string(exponent) + '-', State::KIND, state.k);
  ::writeState(*CycleFile{path}, state);
  trimFiles();
  // log("rm '%s'\n", pathUnverified(base, prefix).string().c_str());
  fs::remove(pathUnverified(base, prefix));
}

template<>
void Saver<PRPState>::saveUnverified(const PRPState& state) const {
  ::writeState(*CycleFile{pathUnverified(base, prefix)}, state);
}

template<typename State>
void Saver<State>::dropMostRecent() {
  fs::path const path = mostRecentSavefile();
  assert(!path.empty());
  if (!path.empty()) { moveToTrash(path); }
}

template<typename State>
vector<u64> Saver<State>::listK() const { return savefiles(base, prefix, State::KIND); }

template<typename State>
std::optional<State> Saver<State>::loadAt(u64 k) const {
  File fi{File::openRead(pathFor(base, prefix, State::KIND, k))};
  if (!fi) { return {}; }
  try {
    State expected{};
    expected.exponent = exponent;
    State state = readState(expected, std::move(fi));
    if (state.exponent == exponent && state.k == k) { return state; }
  } catch (const CRCError& e) {
  } catch (const BadHeaderError& e) {
  } catch (const ReadError& e) {
  }
  return {};
}

template<typename State>
void Saver<State>::removeAt(u64 k) const {
  error_code dummy;
  fs::remove(pathFor(base, prefix, State::KIND, k), dummy);
}

template class Saver<PRPState>;
template class Saver<LLState>;
template class Saver<CERTState>;
template class Saver<P1State>;
template class Saver<P2State>;
template class Saver<P1PrpState>;
