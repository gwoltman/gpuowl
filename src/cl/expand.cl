// Copyright (C) Mihai Preda

// Some routines can be written for any 64-bit data type (T2 or GF61).  Same for 32-bit data types (F2 or GF31).
// Some routines can be written to work 32-bit and 64-bit data types.
// These #defines make it easy to write those routines.  This used to be done with type-casting, but
// this method generates better PTX code (not sure if that results in any better run times).

#if FFT_FP64
#define T_Z61 T
#define T2_GF61 T2
#define T2_F2_GF31_GF61 T2
#define as_T2_GF61 as_double2
#include INCLUDE_FILE
#undef T_Z61
#undef T2_GF61
#undef T2_F2_GF31_GF61
#undef as_T2_GF61
#endif 

#if NTT_GF61
#define T_Z61 Z61
#define T2_GF61 GF61
#define T2_F2_GF31_GF61 GF61
#define as_T2_GF61 as_ulong2
#include INCLUDE_FILE
#undef T_Z61
#undef T2_GF61
#undef T2_F2_GF31_GF61
#undef as_T2_GF61
#endif 

#if FFT_FP32
#define F_Z31 F
#define F2_GF31 F2
#define T2_F2_GF31_GF61 F2
#include INCLUDE_FILE
#undef F_Z31
#undef F2_GF31
#undef T2_F2_GF31_GF61
#endif 

#if NTT_GF31
#define F_Z31 Z31
#define F2_GF31 GF31
#define T2_F2_GF31_GF61 GF31
#include INCLUDE_FILE
#undef F_Z31
#undef F2_GF31
#undef T2_F2_GF31_GF61
#endif 

#undef INCLUDE_FILE

