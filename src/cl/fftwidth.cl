// Copyright (C) Mihai Preda

// #defines that allow fft_height and fft_width share common code in fftbase.cl
#define WG            G_W
#define RADIX         NW
#define VARIANT       FFT_VARIANT_W
#define LDSPAD        LDSPAD_W
#define LDSSWIZ       LDSSWIZ_W
#define SHUFL_BYTES   SHUFL_BYTES_W
#define UNROLL        UNROLL_W
#define SAVE_ONE_MUL  0          // Radeon VII weirdness where saving one mul in width variant 2 was slower
#define DOING_HEIGHT  0          // Flags to work around any optimizer weirdness where common code performs better in fft_WIDTH and worse in fft_HEIGHT or vice versa
#define DOING_WIDTH   1

#include "math.cl"
#include "trig.cl"
#include "fftbase.cl"

#if WIDTH != 256 && WIDTH != 512 && WIDTH != 1024 && WIDTH != 4096 && WIDTH != 625
#error WIDTH must be one of: 256, 512, 1024, 4096, 625
#endif

#if FFT_FP64

// Three versions.  fft_WIDTH1 and fft_WIDTH2 are for the two carryFused calls where a future version might save some data from call 1 for use in call 2.
void OVERLOAD fft_WIDTH(local T2 *lds, T2 *u, Trig trig, u32 numWG, u32 lowMe) { T2 dummy; fft_common(lds, u, trig, dummy, numWG, lowMe, 0); }
void OVERLOAD fft_WIDTH1(local T2 *lds, T2 *u, Trig trig, u32 numWG, u32 lowMe) { T2 dummy; fft_common(lds, u, trig, dummy, numWG, lowMe, 1); }
void OVERLOAD fft_WIDTH2(local T2 *lds, T2 *u, Trig trig, u32 numWG, u32 lowMe) { T2 dummy; fft_common(lds, u, trig, dummy, numWG, lowMe, 2); }

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// Three versions.  fft_WIDTH1 and fft_WIDTH2 are for the two carryFused calls where a future version might save some data from call 1 for use in call 2.
void OVERLOAD fft_WIDTH(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, u32 lowMe) { fft_common(lds , u, trig, numWG, lowMe, 0); }
void OVERLOAD fft_WIDTH1(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe, 1); }
void OVERLOAD fft_WIDTH2(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe, 2); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

// Three versions.  fft_WIDTH1 and fft_WIDTH2 are for the two carryFused calls where a future version might save some data from call 1 for use in call 2.
void OVERLOAD fft_WIDTH(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }
void OVERLOAD fft_WIDTH1(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }
void OVERLOAD fft_WIDTH2(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

// Three versions.  fft_WIDTH1 and fft_WIDTH2 are for the two carryFused calls where a future version might save some data from call 1 for use in call 2.
void OVERLOAD fft_WIDTH(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }
void OVERLOAD fft_WIDTH1(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }
void OVERLOAD fft_WIDTH2(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, u32 lowMe) { fft_common(lds, u, trig, numWG, lowMe); }

#endif
