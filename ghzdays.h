#pragma once

struct TimeEntry {
  int fftSize;
  float factor;
};

TimeEntry timeInfo[] = {
{33554432,  2.3832E+0},
{33030144,  2.3577E+0},
{32768000,  2.3449E+0},
{31457280,  2.2812E+0},
{29360128,  2.1792E+0},
{28311552,  2.0898E+0},
{27525120,  2.0227E+0},
{26214400,  1.9110E+0},
{25165824,  1.8216E+0},
{23592960,  1.6974E+0},
{22937600,  1.6456E+0},
{22020096,  1.5732E+0},
{20971520,  1.4904E+0},
{19660800,  1.3621E+0},
{18874368,  1.2852E+0},
{18350080,  1.2339E+0},
{16777216,  1.0800E+0},
{16515072,  1.0680E+0},
{16384000,  1.0620E+0},
{15728640,  1.0320E+0},
{14680064,  9.8400E-1},
{14155776,  9.4080E-1},
{13762560,  9.0840E-1},
{13107200,  8.5440E-1},
{12582912,  8.1120E-1},
{11796480,  7.5900E-1},
{11468800,  7.3725E-1},
{11010048,  7.0680E-1},
{10485760,  6.7200E-1},
{ 9830400,  6.1950E-1},
{ 9437184,  5.8800E-1},
{ 9175040,  5.6700E-1},
{ 8388608,  5.0400E-1},
{ 8257536,  4.9830E-1},
{ 8192000,  4.9545E-1},
{ 7864320,  4.8120E-1},
{ 7340032,  4.5840E-1},
{ 7077888,  4.3860E-1},
{ 6881280,  4.2375E-1},
{ 6553600,  3.9900E-1},
{ 6291456,  3.7920E-1},
{ 5898240,  3.5436E-1},
{ 5734400,  3.4401E-1},
{ 5505024,  3.2952E-1},
{ 5242880,  3.1296E-1},
{ 4915200,  2.9128E-1},
{ 4718592,  2.7828E-1},
{ 4587520,  2.6961E-1},
{ 4194304,  2.4360E-1},
{ 4128768,  2.4045E-1},
{ 4096000,  2.3887E-1},
{ 3932160,  2.3100E-1},
{ 3670016,  2.1840E-1},
{ 3538944,  2.0958E-1},
{ 3440640,  2.0296E-1},
{ 3276800,  1.9194E-1},
{ 3145728,  1.8312E-1},
{ 2949120,  1.7070E-1},
{ 2867200,  1.6552E-1},
{ 2752512,  1.5828E-1},
{ 2621440,  1.5000E-1},
{ 2457600,  1.3867E-1},
{ 2359296,  1.3188E-1},
{ 2293760,  1.2735E-1},
{ 2097152,  1.1376E-1},
{ 2064384,  1.1232E-1},
{ 2048000,  1.1160E-1},
{ 1966080,  1.0800E-1},
{ 1835008,  1.0224E-1},
{ 1769472,  9.8160E-2},
{ 1720320,  9.5100E-2},
{ 1638400,  9.0000E-2},
{ 1572864,  8.5920E-2},
{ 1474560,  7.9980E-2},
{ 1376256,  7.4040E-2},
{ 1310720,  7.0080E-2},
{ 1228800,  6.5655E-2},
{ 1179648,  6.3000E-2},
{ 1146880,  6.1230E-2},
{ 1048576,  5.5920E-2},
{ 1032192,  5.5080E-2},
{  983040,  5.2560E-2},
{  917504,  4.9200E-2},
{  884736,  4.7244E-2},
{  819200,  4.3332E-2},
{  786432,  4.1376E-2},
{  737280,  3.8334E-2},
{  688128,  3.5292E-2},
{  655360,  3.3264E-2},
{  589824,  2.9232E-2},
{  573440,  2.8224E-2},
{  524288,  2.5200E-2},
{  491520,  2.3880E-2},
{  458752,  2.2560E-2},
{  409600,  1.9770E-2},
{  393216,  1.8840E-2},
{  344064,  1.6194E-2},
{  327680,  1.5312E-2},
{  294912,  1.3644E-2},
{  262144,  1.1976E-2},
{  245760,  1.1376E-2},
{  229376,  1.0776E-2},
{  196608,  9.0744E-3},
{  163840,  7.3536E-3},
{  147456,  6.6732E-3},
{  131072,  5.9928E-3},
{  122880,  5.8368E-3},
{  114688,  5.6808E-3},
{   98304,  4.6872E-3},
{   86016,  4.0662E-3},
{   81920,  3.8592E-3},
{   73728,  3.2988E-3},
{   65536,  2.7384E-3},
{   61440,  2.6496E-3},
{   57344,  2.5608E-3},
{   49152,  2.0952E-3},
{   40960,  1.7256E-3},
{   32768,  1.3128E-3},
{   28672,  1.2600E-3},
{   24576,  1.0176E-3},
{   20480,  8.3760E-4},
{   16384,  6.1440E-4},
{   14336,  5.9040E-4},
{   12288,  4.8480E-4},
{   10240,  3.9840E-4},
{    8192,  2.8560E-4},
{    7168,  2.7600E-4},
{    6144,  2.2560E-4},
{    5120,  1.9200E-4},
{    4096,  1.2960E-4},
{    3584,  1.2648E-4},
{    3072,  1.0248E-4},
{    2560,  8.5200E-5},
{    2048,  6.1680E-5},
{    1792,  5.6880E-5},
{    1536,  4.7592E-5},
{    1280,  3.9120E-5},
{    1024,  2.7648E-5},
{     896,  2.5416E-5},
{     768,  2.0592E-5},
{     640,  1.7040E-5},
{     512,  1.1328E-5},
{     448,  1.0536E-5},
{     384,  8.7840E-6},
{     320,  7.3440E-6},
{     256,  5.4744E-6},
{     224,  5.2512E-6},
{     192,  4.2936E-6},
{     160,  3.5640E-6},
{     128,  2.5272E-6},
{     112,  2.4624E-6},
{      96,  2.0520E-6},
{      80,  1.7592E-6},
{      64,  1.4088E-6},
{      48,  1.1400E-6},
{      32,  8.7840E-7},
};

float ghzSecsPerIt(int fftSize) {
  for (auto entry : timeInfo) { if (fftSize == entry.fftSize) { return entry.factor; } }
  return 0;
}

float ghzDays(u32 exp, int fftSize) {
  float ghzSecs = ghzSecsPerIt(fftSize) * exp;
  return ghzSecs / (24 * 3600);
  // for (auto entry : timeInfo) { if (fftSize == entry.fftSize) { return exp * entry.factor / (24 * 3600);} }
}
