// Copyright (C) Mihai Preda

// #defines that allow fft_height and fft_width share common code in fftbase.cl
#define WG            G_H
#define RADIX         NH
#define VARIANT       FFT_VARIANT_H
#define LDSPAD        LDSPAD_H
#define LDSSWIZ       LDSSWIZ_H
#define SHUFL_BYTES   SHUFL_BYTES_H
#define UNROLL        UNROLL_H
#define SAVE_ONE_MUL  1          // Radeon VII weirdness where saving one mul was slower (needs retesting!)
#define DOING_HEIGHT  1          // Flags to work around any optimizer weirdness where common code performs better in fft_WIDTH and worse in fft_HEIGHT or vice versa
#define DOING_WIDTH   0

#include "math.cl"
#include "trig.cl"
#include "fftbase.cl"

#if SMALL_HEIGHT != 256 && SMALL_HEIGHT != 512 && SMALL_HEIGHT != 1024
#error SMALL_HEIGHT must be one of: 256, 512, 1024
#endif

#if !INPLACE
u32 transPos(u32 k, u32 middle, u32 width) { return k / width + k % width * middle; }
#else
u32 transPos(u32 k, u32 middle, u32 width) { return k; }
#endif

#if FFT_FP64

// Three versions.  fft_HEIGHT1 and fft_HEIGHT2 are for the two tailSquare calls where a future version might save some data from call 1 for use in call 2.
void fft_HEIGHT(local T2 *lds, T2 *u, Trig trig, T2 w, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, w, numWG, lowMe, 0); }
void fft_HEIGHT1(local T2 *lds, T2 *u, Trig trig, T2 w, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, w, numWG, lowMe, 1); }
void fft_HEIGHT2(local T2 *lds, T2 *u, Trig trig, T2 w, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, w, numWG, lowMe, 2); }

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// Three versions.  fft_HEIGHT1 and fft_HEIGHT2 are for the two tailSquare calls where a future version might save some data from call 1 for use in call 2.
void fft_HEIGHT(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe, 0); }
void fft_HEIGHT1(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe, 1); }
void fft_HEIGHT2(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe, 2); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

// Three versions.  fft_HEIGHT1 and fft_HEIGHT2 are for the two tailSquare calls where a future version might save some data from call 1 for use in call 2.
void OVERLOAD fft_HEIGHT(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }
void OVERLOAD fft_HEIGHT1(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }
void OVERLOAD fft_HEIGHT2(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

// Three versions.  fft_HEIGHT1 and fft_HEIGHT2 are for the two tailSquare calls where a future version might save some data from call 1 for use in call 2.
void OVERLOAD fft_HEIGHT(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }
void OVERLOAD fft_HEIGHT1(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }
void OVERLOAD fft_HEIGHT2(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }

#endif
