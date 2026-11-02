// Copyright (C) 2017-2024 Mihai Preda.

#include "gpuid.h"
#include <utility>
#include <algorithm>
#include "clwrap.h"
#include "File.h"

using namespace std;

static bool startsWith(string_view a, string_view b) {
  return a.starts_with(b);
}

string getBdfFromSysfs(int pos) {
  assert(pos >= 0);
  
  const string PREFIX = "PCI_SLOT_NAME=";
  const string SKIP = "0000:";

  if (File f = File::openRead("/sys/class/drm/card"s + std::to_string(pos) + "/device/uevent")) {
    for (const auto& line : f) {
      if (startsWith(line, PREFIX)) {
        string bdf = line.substr(PREFIX.length());
        if (startsWith(bdf, SKIP)) { bdf = bdf.substr(SKIP.length()); }
        if (!bdf.empty() && bdf.back() == '\n') { bdf.pop_back(); }
        return bdf;
      }
    }
  }
  return "";
}

/* BDF is PCIe Bus:Device.Function e.g. "6a:00.0" */
int getSysfsFromBdf(const string& wantBdf) {  
  for (u32 pos = 0; !pos || filesystem::exists("/sys/class/drm/card"s + std::to_string(pos)); ++pos) {
    if (wantBdf == getBdfFromSysfs(pos)) { return pos; }
  }
  return -1; // not found
}

int getSysfsFromUid(const string& wantUid) {
  for (u32 pos = 0; !pos || filesystem::exists("/sys/class/drm/card"s + std::to_string(pos)); ++pos) {
    if (File f = File::openRead("/sys/class/drm/card"s + std::to_string(pos) + "/device/unique_id")) {
      string uid = f.readLine();
      if (!uid.empty() && uid.back() == '\n') { uid.pop_back(); }
      if (wantUid == uid) { return pos; }
    }
  }
  log("Sysfs UID '%s' not found\n", wantUid.c_str());
  return -1;
}

string getUidFromSysfs(int pos) {
  File f = File::openRead("/sys/class/drm/card"s + std::to_string(pos) + "/device/unique_id");
  std::string uuid = f ? f.readLine() : "";
  if (!uuid.empty() && uuid.back() == '\n') { uuid.pop_back(); }
  return uuid;
}

/* BDF is PCIe Bus:Device.Function e.g. "6a:00.0" */
string getUidFromBdf(const string& bdf) {
  int const pos = getSysfsFromBdf(bdf);
  return pos >= 0 ? getUidFromSysfs(pos) : "";
}

string getBdfFromUid(const string& uid) {
  int const pos = getSysfsFromUid(uid);
  return (pos >= 0) ? getBdfFromSysfs(pos) : "";
}

int getPosFromBdf(const string& bdf) {
  auto openclIds = getAllDeviceIDs();
  for (int pos = 0; std::cmp_less(pos, openclIds.size()); ++pos) {
    auto bdfAtPos = getBdfFromDevice(openclIds[pos]);
    // log("BDF '%s' at %d\n", bdfAtPos.c_str(), pos);
    if (bdf == bdfAtPos) { return pos; }
  }
  
  log("OpenCL device with BDF '%s' not found\n", bdf.c_str());
  return -1;
}

string getBdfFromPos(int pos) {
  assert(pos >= 0);
  return getBdfFromDevice(getDevice(pos));
}

int getPosFromUid(const string& uid) {
  string const bdf = getBdfFromUid(uid);
  if (bdf.empty()) { return -1; }
  return getPosFromBdf(bdf);
}

string getUidFromPos(int pos) {
  return getUidFromBdf(getBdfFromPos(pos));
}

namespace {

// The GPU devices of each OpenCL platform, in clGetPlatformIDs() order.
vector<vector<cl_device_id>> getGpusByPlatform() {
  cl_platform_id platforms[16];
  unsigned nPlatforms = 0;
  if (clGetPlatformIDs(16, platforms, &nPlatforms) != CL_SUCCESS) { nPlatforms = 0; }
  nPlatforms = min(nPlatforms, 16u);

  vector<vector<cl_device_id>> ret(nPlatforms);
  for (unsigned p = 0; p < nPlatforms; ++p) {
    cl_device_id devices[64];
    unsigned n = 0;
    if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 64, devices, &n) != CL_SUCCESS) { continue; }
    ret[p].assign(devices, devices + min(n, 64u));
  }
  return ret;
}

}

int getPosFromPlatformDevice(int platform, int device) {
  auto const gpus = getGpusByPlatform();
  if (platform < 0 || std::cmp_greater_equal(platform, gpus.size())) {
    log("OpenCL platform %d not found (%u platforms)\n", platform, unsigned(gpus.size()));
    return -1;
  }
  auto const& devs = gpus[platform];
  if (device < 0 || std::cmp_greater_equal(device, devs.size())) {
    log("OpenCL platform %d has no GPU %d (%u GPUs)\n", platform, device, unsigned(devs.size()));
    return -1;
  }

  // Positions used by -device N, and everywhere else, are positions in getAllDeviceIDs().
  auto const all = getAllDeviceIDs();
  auto const it = find(all.begin(), all.end(), devs[device]);
  return it == all.end() ? -1 : int(it - all.begin());
}

string getPlatformDeviceFromPos(int pos) {
  auto const all = getAllDeviceIDs();
  if (pos < 0 || std::cmp_greater_equal(pos, all.size())) { return ""; }
  auto const gpus = getGpusByPlatform();
  for (size_t p = 0; p < gpus.size(); ++p) {
    auto const it = find(gpus[p].begin(), gpus[p].end(), all[pos]);
    if (it != gpus[p].end()) { return to_string(p) + ":" + to_string(it - gpus[p].begin()); }
  }
  return "";
}
