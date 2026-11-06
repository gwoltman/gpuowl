// Test tool:  flips one bit of the residue in one of prpll's P-1 savefiles (p1-<p>/<p>-<k>.p1: a header line, the CRC, the words) and
// writes the CRC again, so that the file still loads and only a check of the residue itself (the Jacobi symbol) can tell.
//
//   p1flip <savefile> <bit>

#include "common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s <savefile> <bit>\n", argv[0]); return 2; }
  std::vector<char> b;
  {
    std::ifstream in(argv[1], std::ios::binary);
    b.assign(std::istreambuf_iterator<char>(in), {});
  }
  auto const nl = std::find(b.begin(), b.end(), '\n');
  if (nl == b.end()) { fprintf(stderr, "no header line\n"); return 1; }
  size_t const crcAt = nl - b.begin() + 1, wordsAt = crcAt + 4;
  if ((b.size() - wordsAt) % 4) { fprintf(stderr, "bad size\n"); return 1; }
  vector<u32> words((b.size() - wordsAt) / 4);
  memcpy(words.data(), b.data() + wordsAt, 4 * words.size());
  u64 const bit = strtoull(argv[2], nullptr, 10);
  if (bit >= 32 * words.size()) { fprintf(stderr, "bit out of range\n"); return 1; }
  words[bit / 32] ^= u32(1) << (bit % 32);
  u32 const crc = crc32(words);
  memcpy(b.data() + crcAt, &crc, 4);
  memcpy(b.data() + wordsAt, words.data(), 4 * words.size());
  std::ofstream out(argv[1], std::ios::binary | std::ios::trunc);
  out.write(b.data(), b.size());
  return out.good() ? 0 : 1;
}
