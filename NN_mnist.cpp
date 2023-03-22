// NN_mnist.cpp
//
// JHelas (c) 09/20
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Delay.h"
#include "File.h"


typedef unsigned char   BYTE;
typedef float           float32_t;

///////////////////////////////////////////////////////////

const int           HNODES              = 192;
const int           ONODES              =  10;
const int           EPOCHS              =   4;

const float         LEARN_RATE          = .107;


const char          FILENAME_LD[]       = "train-images-idx3-ubyte";
const char          FILENAME_LL[]       = "train-labels-idx1-ubyte";
const char          FILENAME_TD[]       = "t10k-images-idx3-ubyte";
const char          FILENAME_TL[]       = "t10k-labels-idx1-ubyte";

///////////////////////////////////////////////////////////

float               g_Outputs[ONODES][ONODES];

float               *g_pLNodes          = NULL;
float               *g_pTNodes          = NULL;
float               *g_pINodes;
float               *g_pHNodes          = NULL;
float               g_ONodes[ONODES];
float               *g_pWIH             = NULL;
float               *g_pWHO             = NULL;

float               g_fLearnRate;
int                 g_nINodes;
int                 g_nHNodes;
int                 g_nLearnRecords;
int                 g_nTestRecords;


#if defined _NEON

typedef unsigned int    uint32_t;

float               g_Constants0[4];
const float         g_Constants1[4]   = {         // Sigmoid function
                        0.03138777F, 0.276281267F, 1.442695022F, 1.442695022F * .5 };

////////////////////////////////////////////////////////////

void VectorMultiplyMatrix(float32_t *pV, float32_t *pM, float32_t *pT, uint32_t nVectorSize, uint32_t nTargetSize) {
  asm volatile (
    "ld1  { V12.4S }, [%[ptr_constants]]\n\t"     // Sigmoid()
    "dup  V13.4S, V12.4S[2]\n\t"
    "mov  x5, %[TSIZE]\n\t"
    "mov  x10, %[TSIZE], lsl #2\n\t"
    "mov  x11, %[target]\n\t"
    "mov  x8, #0\n\t"
    "VMM0:\n\t"
    "fsub V0.4S, V0.4S, V0.4S\n\t"
    "mov  x6, %[VSIZE]\n\t"
    "dup V1.4S, V0.4S[0]\n\t"
    "mov  x7, %[vector]\n\t"
    "dup V2.4S, V0.4S[0]\n\t"
    "add  x9, %[matrix], x8, lsl #2\n\t"
    "dup V3.4S, V0.4S[0]\n\t"
    "VMM1:\n\t"
    "subs x6, x6, 16\n\t"
    "ld1  { V4.4S,  V5.4S,  V6.4S,  V7.4S } , [x7],#64\n\t"

    //
    "ld1  { V8.4S,  V9.4S, V10.4S, V11.4S }, [x9], x10\n\t"
    "ld1  { V20.4S,  V21.4S, V22.4S, V23.4S }, [x9], x10\n\t"
    "fmla V0.4S,  V8.4S, V4.4S[0]\n\t"
    "fmla V1.4S,  V9.4S, V4.4S[0]\n\t"
    "ld1  { V24.4S,  V25.4S, V26.4S, V27.4S }, [x9], x10\n\t"
    "fmla V2.4S, V10.4S, V4.4S[0]\n\t"
    "fmla V3.4S, V11.4S, V4.4S[0]\n\t"
    "ld1  { V28.4S,  V29.4S, V30.4S, V31.4S }, [x9], x10\n\t"
    "fmla V0.4S, V20.4S, V4.4S[1]\n\t"
    "fmla V1.4S, V21.4S, V4.4S[1]\n\t"
    "fmla V2.4S, V22.4S, V4.4S[1]\n\t"
    "fmla V3.4S, V23.4S, V4.4S[1]\n\t"
    "ld1  { V8.4S,  V9.4S, V10.4S, V11.4S }, [x9], x10\n\t"
    "fmla V0.4S, V24.4S, V4.4S[2]\n\t"
    "fmla V1.4S, V25.4S, V4.4S[2]\n\t"
    "fmla V2.4S, V26.4S, V4.4S[2]\n\t"
    "fmla V3.4S, V27.4S, V4.4S[2]\n\t"
    "fmla V0.4S, V28.4S, V4.4S[3]\n\t"
    "fmla V1.4S, V29.4S, V4.4S[3]\n\t"
    "fmla V2.4S, V30.4S, V4.4S[3]\n\t"
    "fmla V3.4S, V31.4S, V4.4S[3]\n\t"

    "ld1  { V20.4S,  V21.4S, V22.4S, V23.4S }, [x9], x10\n\t"
    "fmla V0.4S,  V8.4S, V5.4S[0]\n\t"
    "fmla V1.4S,  V9.4S, V5.4S[0]\n\t"
    "ld1  { V24.4S,  V25.4S, V26.4S, V27.4S }, [x9], x10\n\t"
    "fmla V2.4S, V10.4S, V5.4S[0]\n\t"
    "fmla V3.4S, V11.4S, V5.4S[0]\n\t"
    "ld1  { V28.4S,  V29.4S, V30.4S, V31.4S }, [x9], x10\n\t"
    "fmla V0.4S, V20.4S, V5.4S[1]\n\t"
    "fmla V1.4S, V21.4S, V5.4S[1]\n\t"
    "fmla V2.4S, V22.4S, V5.4S[1]\n\t"
    "fmla V3.4S, V23.4S, V5.4S[1]\n\t"
    "ld1  { V8.4S,  V9.4S, V10.4S, V11.4S }, [x9], x10\n\t"
    "fmla V0.4S, V24.4S, V5.4S[2]\n\t"
    "fmla V1.4S, V25.4S, V5.4S[2]\n\t"
    "fmla V2.4S, V26.4S, V5.4S[2]\n\t"
    "fmla V3.4S, V27.4S, V5.4S[2]\n\t"
    "fmla V0.4S, V28.4S, V5.4S[3]\n\t"
    "fmla V1.4S, V29.4S, V5.4S[3]\n\t"
    "fmla V2.4S, V30.4S, V5.4S[3]\n\t"
    "fmla V3.4S, V31.4S, V5.4S[3]\n\t"

    "ld1  { V20.4S,  V21.4S, V22.4S, V23.4S }, [x9], x10\n\t"
    "fmla V0.4S,  V8.4S, V6.4S[0]\n\t"
    "fmla V1.4S,  V9.4S, V6.4S[0]\n\t"
    "ld1  { V24.4S,  V25.4S, V26.4S, V27.4S }, [x9], x10\n\t"
    "fmla V2.4S, V10.4S, V6.4S[0]\n\t"
    "fmla V3.4S, V11.4S, V6.4S[0]\n\t"
    "ld1  { V28.4S,  V29.4S, V30.4S, V31.4S }, [x9], x10\n\t"
    "fmla V0.4S, V20.4S, V6.4S[1]\n\t"
    "fmla V1.4S, V21.4S, V6.4S[1]\n\t"
    "fmla V2.4S, V22.4S, V6.4S[1]\n\t"
    "fmla V3.4S, V23.4S, V6.4S[1]\n\t"
    "ld1  { V8.4S,  V9.4S, V10.4S, V11.4S }, [x9], x10\n\t"
    "fmla V0.4S, V24.4S, V6.4S[2]\n\t"
    "fmla V1.4S, V25.4S, V6.4S[2]\n\t"
    "fmla V2.4S, V26.4S, V6.4S[2]\n\t"
    "fmla V3.4S, V27.4S, V6.4S[2]\n\t"
    "fmla V0.4S, V28.4S, V6.4S[3]\n\t"
    "fmla V1.4S, V29.4S, V6.4S[3]\n\t"
    "fmla V2.4S, V30.4S, V6.4S[3]\n\t"
    "fmla V3.4S, V31.4S, V6.4S[3]\n\t"

    "ld1  { V20.4S,  V21.4S, V22.4S, V23.4S }, [x9], x10\n\t"
    "fmla V0.4S,  V8.4S, V7.4S[0]\n\t"
    "fmla V1.4S,  V9.4S, V7.4S[0]\n\t"
    "ld1  { V24.4S,  V25.4S, V26.4S, V27.4S }, [x9], x10\n\t"
    "fmla V2.4S, V10.4S, V7.4S[0]\n\t"
    "fmla V3.4S, V11.4S, V7.4S[0]\n\t"
    "ld1  { V28.4S,  V29.4S, V30.4S, V31.4S }, [x9], x10\n\t"
    "fmla V0.4S, V20.4S, V7.4S[1]\n\t"
    "fmla V1.4S, V21.4S, V7.4S[1]\n\t"
    "fmla V2.4S, V22.4S, V7.4S[1]\n\t"
    "fmla V3.4S, V23.4S, V7.4S[1]\n\t"
    "fmla V0.4S, V24.4S, V7.4S[2]\n\t"
    "fmla V1.4S, V25.4S, V7.4S[2]\n\t"
    "fmla V2.4S, V26.4S, V7.4S[2]\n\t"
    "fmla V3.4S, V27.4S, V7.4S[2]\n\t"
    "fmla V0.4S, V28.4S, V7.4S[3]\n\t"
    "fmla V1.4S, V29.4S, V7.4S[3]\n\t"
    "fmla V2.4S, V30.4S, V7.4S[3]\n\t"
    "fmla V3.4S, V31.4S, V7.4S[3]\n\t"

    "bne  VMM1\n\t"

    ////////////////////////////////////////////////////////////
    // Sigmoid()

    "fmul V4.4S, V0.4S, V12.4S[3]\n\t"            // v *= c_log2f * .5
    "fcvtzs V5.4S, V4.4S\n\t"                     // to int
    "scvtf  V7.4S, V5.4S\n\t"
    "fsub V6.4S, V4.4S, V7.4S\n\t"                // x = v - intPart
    "fmul V7.4S, V6.4S, V6.4S\n\t"                // xx = x * x
    "fmul V8.4S, V7.4S, V12.4S[1]\n\t"            // v1 = c2 * xx
    "fadd V8.4S, V8.4S, V13.4S\n\t"               // v1 += c_log2f
    "fmul V9.4S, V6.4S, V12.4S[0]\n\t"            // v2 = x *c1
    "fmul V9.4S, V9.4S, V7.4S\n\t"                // v2 *= xx
    "fadd V9.4S, V9.4S, V6.4S\n\t"                // v2 += x
    "fadd V10.4S, V9.4S, V8.4S\n\t"               // v3 = v2 + v1
    "fsub V11.4S, V9.4S, V8.4S\n\t"               // v4 = v2 - v1
    "shl  V5.4S, V5.4S, #24\n\t"
    "add  V7.4S, V10.4S, V5.4S\n\t"               // v3 += intPart << 24
    "fsub V6.4S, V7.4S, V11.4S\n\t"               //  V3 - V4
    "fdiv V0.4S, V7.4S, V6.4S\n\t"                // res = v3 / (V3 - V4)

    "fmul V4.4S, V1.4S, V12.4S[3]\n\t"            // v *= c_log2f * .5
    "fcvtzs V5.4S, V4.4S\n\t"                     // to int
    "scvtf  V7.4S, V5.4S\n\t"
    "fsub V6.4S, V4.4S, V7.4S\n\t"                // x = v - intPart
    "fmul V7.4S, V6.4S, V6.4S\n\t"                // xx = x * x
    "fmul V8.4S, V7.4S, V12.4S[1]\n\t"            // v1 = c2 * xx
    "fadd V8.4S, V8.4S, V13.4S\n\t"               // v1 += c_log2f
    "fmul V9.4S, V6.4S, V12.4S[0]\n\t"            // v2 = x *c1
    "fmul V9.4S, V9.4S, V7.4S\n\t"                // v2 *= xx
    "fadd V9.4S, V9.4S, V6.4S\n\t"                // v2 += x
    "fadd V10.4S, V9.4S, V8.4S\n\t"               // v3 = v2 + v1
    "fsub V11.4S, V9.4S, V8.4S\n\t"               // v4 = v2 - v1
    "shl  V5.4S, V5.4S, #24\n\t"
    "add  V7.4S, V10.4S, V5.4S\n\t"               // v3 += intPart << 24
    "fsub V6.4S, V7.4S, V11.4S\n\t"               //  V3 - V4
    "fdiv V1.4S, V7.4S, V6.4S\n\t"                // res = v3 / (V3 - V4)

    "fmul V4.4S, V2.4S, V12.4S[3]\n\t"            // v *= c_log2f * .5
    "fcvtzs V5.4S, V4.4S\n\t"                     // to int
    "scvtf  V7.4S, V5.4S\n\t"
    "fsub V6.4S, V4.4S, V7.4S\n\t"                // x = v - intPart
    "fmul V7.4S, V6.4S, V6.4S\n\t"                // xx = x * x
    "fmul V8.4S, V7.4S, V12.4S[1]\n\t"            // v1 = c2 * xx
    "fadd V8.4S, V8.4S, V13.4S\n\t"               // v1 += c_log2f
    "fmul V9.4S, V6.4S, V12.4S[0]\n\t"            // v2 = x *c1
    "fmul V9.4S, V9.4S, V7.4S\n\t"                // v2 *= xx
    "fadd V9.4S, V9.4S, V6.4S\n\t"                // v2 += x
    "fadd V10.4S, V9.4S, V8.4S\n\t"               // v3 = v2 + v1
    "fsub V11.4S, V9.4S, V8.4S\n\t"               // v4 = v2 - v1
    "shl  V5.4S, V5.4S, #24\n\t"
    "add  V7.4S, V10.4S, V5.4S\n\t"               // v3 += intPart << 24
    "fsub V6.4S, V7.4S, V11.4S\n\t"               //  V3 - V4
    "fdiv V2.4S, V7.4S, V6.4S\n\t"                // res = v3 / (V3 - V4)

    "fmul V4.4S, V3.4S, V12.4S[3]\n\t"            // v *= c_log2f * .5
    "fcvtzs V5.4S, V4.4S\n\t"                     // to int
    "scvtf  V7.4S, V5.4S\n\t"
    "fsub V6.4S, V4.4S, V7.4S\n\t"                // x = v - intPart
    "fmul V7.4S, V6.4S, V6.4S\n\t"                // xx = x * x
    "fmul V8.4S, V7.4S, V12.4S[1]\n\t"            // v1 = c2 * xx
    "fadd V8.4S, V8.4S, V13.4S\n\t"               // v1 += c_log2f
    "fmul V9.4S, V6.4S, V12.4S[0]\n\t"            // v2 = x *c1
    "fmul V9.4S, V9.4S, V7.4S\n\t"                // v2 *= xx
    "fadd V9.4S, V9.4S, V6.4S\n\t"                // v2 += x
    "fadd V10.4S, V9.4S, V8.4S\n\t"               // v3 = v2 + v1
    "fsub V11.4S, V9.4S, V8.4S\n\t"               // v4 = v2 - v1
    "shl  V5.4S, V5.4S, #24\n\t"
    "add  V7.4S, V10.4S, V5.4S\n\t"               // v3 += intPart << 24
    "fsub V6.4S, V7.4S, V11.4S\n\t"               //  V3 - V4
    "fdiv V3.4S, V7.4S, V6.4S\n\t"                // res = v3 / (V3 - V4)

    // \Sigmoid()
    ////////////////////////////////////////////////////////////

    "st1  {V0.4S, V1.4S, V2.4S, V3.4S}, [x11], #64\n\t"
    "subs x5, x5, 16\n\t"
    "add  x8, x8, 16\n\t"
    "bne  VMM0\n\t"
    :
    : [vector] "r" (pV), [matrix] "r" (pM), [VSIZE] "r" (nVectorSize),
                [target] "r" (pT), [TSIZE] "r" (nTargetSize), [ptr_constants] "r" (g_Constants1)
    : "memory", "cc", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
      "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
      "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",  "v28", "v29", "v30", "v31"
   );
 }


void VectorMultiplyMatrix10(float32_t *pV, float32_t *pM, float32_t *pT, uint32_t nVectorSize) {
  asm volatile (
    "ld1  { V12.4S }, [%[ptr_constants]]\n\t"   // Sigmoid()
    "dup  V13.4S, V12.4S[2]\n\t"
    "fsub V0.4S, V0.4S, V0.4S\n\t"
    "mov  x6, %[VSIZE]\n\t"
    "dup V1.4S, V0.4S[0]\n\t"
    "mov  x7, %[vector]\n\t"
    "dup V2.4S, V0.4S[0]\n\t"
    "mov  x9, %[matrix]\n\t"
    "VMM2%=:\n\t"
    "ld1  { V4.4S,  V5.4S,  V6.4S,  V7.4S }, [x7],#64\n\t"

    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V4.4S[0]\n\t"
    "fmla V1.4S,  V9.4S, V4.4S[0]\n\t"
    "fmla V2.4S, V10.4S, V4.4S[0]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V4.4S[1]\n\t"
    "fmla V1.4S,  V9.4S, V4.4S[1]\n\t"
    "fmla V2.4S, V10.4S, V4.4S[1]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V4.4S[2]\n\t"
    "fmla V1.4S,  V9.4S, V4.4S[2]\n\t"
    "fmla V2.4S, V10.4S, V4.4S[2]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V4.4S[3]\n\t"
    "fmla V1.4S,  V9.4S, V4.4S[3]\n\t"
    "fmla V2.4S, V10.4S, V4.4S[3]\n\t"

    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V5.4S[0]\n\t"
    "fmla V1.4S,  V9.4S, V5.4S[0]\n\t"
    "fmla V2.4S, V10.4S, V5.4S[0]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V5.4S[1]\n\t"
    "fmla V1.4S,  V9.4S, V5.4S[1]\n\t"
    "fmla V2.4S, V10.4S, V5.4S[1]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V5.4S[2]\n\t"
    "fmla V1.4S,  V9.4S, V5.4S[2]\n\t"
    "fmla V2.4S, V10.4S, V5.4S[2]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V5.4S[3]\n\t"
    "fmla V1.4S,  V9.4S, V5.4S[3]\n\t"
    "fmla V2.4S, V10.4S, V5.4S[3]\n\t"

    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V6.4S[0]\n\t"
    "fmla V1.4S,  V9.4S, V6.4S[0]\n\t"
    "fmla V2.4S, V10.4S, V6.4S[0]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V6.4S[1]\n\t"
    "fmla V1.4S,  V9.4S, V6.4S[1]\n\t"
    "fmla V2.4S, V10.4S, V6.4S[1]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V6.4S[2]\n\t"
    "fmla V1.4S,  V9.4S, V6.4S[2]\n\t"
    "fmla V2.4S, V10.4S, V6.4S[2]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V6.4S[3]\n\t"
    "fmla V1.4S,  V9.4S, V6.4S[3]\n\t"
    "fmla V2.4S, V10.4S, V6.4S[3]\n\t"

    "subs x6, x6, 16\n\t"

    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V7.4S[0]\n\t"
    "fmla V1.4S,  V9.4S, V7.4S[0]\n\t"
    "fmla V2.4S, V10.4S, V7.4S[0]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V7.4S[1]\n\t"
    "fmla V1.4S,  V9.4S, V7.4S[1]\n\t"
    "fmla V2.4S, V10.4S, V7.4S[1]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V7.4S[2]\n\t"
    "fmla V1.4S,  V9.4S, V7.4S[2]\n\t"
    "fmla V2.4S, V10.4S, V7.4S[2]\n\t"
    "ld1  { V8.4S,  V9.4S }, [x9], #32\n\t"
    "ld1  { V10.2S }, [x9], #8\n\t"
    "fmla V0.4S,  V8.4S, V7.4S[3]\n\t"
    "fmla V1.4S,  V9.4S, V7.4S[3]\n\t"
    "fmla V2.4S, V10.4S, V7.4S[3]\n\t"
    "bne  VMM2%=\n\t"

    ////////////////////////////////////////////////////////////
    // Sigmoid()

    "fmul V4.4S, V0.4S, V12.4S[3]\n\t"            // v *= c_log2f * .5
    "fcvtzs V5.4S, V4.4S\n\t"                     // to int
    "scvtf  V7.4S, V5.4S\n\t"
    "fsub V6.4S, V4.4S, V7.4S\n\t"                // x = v - intPart
    "fmul V7.4S, V6.4S, V6.4S\n\t"                // xx = x * x
    "fmul V8.4S, V7.4S, V12.4S[1]\n\t"            // v1 = c2 * xx
    "fadd V8.4S, V8.4S, V13.4S\n\t"               // v1 += c_log2f
    "fmul V9.4S, V6.4S, V12.4S[0]\n\t"            // v2 = x *c1
    "fmul V9.4S, V9.4S, V7.4S\n\t"                // v2 *= xx
    "fadd V9.4S, V9.4S, V6.4S\n\t"                // v2 += x
    "fadd V10.4S, V9.4S, V8.4S\n\t"               // v3 = v2 + v1
    "fsub V11.4S, V9.4S, V8.4S\n\t"               // v4 = v2 - v1
    "shl  V5.4S, V5.4S, #24\n\t"
    "add  V7.4S, V10.4S, V5.4S\n\t"               // v3 += intPart << 24
    "fsub V6.4S, V7.4S, V11.4S\n\t"               //  V3 - V4
    "fdiv V0.4S, V7.4S, V6.4S\n\t"                // res = v3 / (V3 - V4)

    "fmul V4.4S, V1.4S, V12.4S[3]\n\t"            // v *= c_log2f * .5
    "fcvtzs V5.4S, V4.4S\n\t"                     // to int
    "scvtf  V7.4S, V5.4S\n\t"
    "fsub V6.4S, V4.4S, V7.4S\n\t"                // x = v - intPart
    "fmul V7.4S, V6.4S, V6.4S\n\t"                // xx = x * x
    "fmul V8.4S, V7.4S, V12.4S[1]\n\t"            // v1 = c2 * xx
    "fadd V8.4S, V8.4S, V13.4S\n\t"               // v1 += c_log2f
    "fmul V9.4S, V6.4S, V12.4S[0]\n\t"            // v2 = x *c1
    "fmul V9.4S, V9.4S, V7.4S\n\t"                // v2 *= xx
    "fadd V9.4S, V9.4S, V6.4S\n\t"                // v2 += x
    "fadd V10.4S, V9.4S, V8.4S\n\t"               // v3 = v2 + v1
    "fsub V11.4S, V9.4S, V8.4S\n\t"               // v4 = v2 - v1
    "shl  V5.4S, V5.4S, #24\n\t"
    "add  V7.4S, V10.4S, V5.4S\n\t"               // v3 += intPart << 24
    "fsub V6.4S, V7.4S, V11.4S\n\t"               //  V3 - V4
    "fdiv V1.4S, V7.4S, V6.4S\n\t"                // res = v3 / (V3 - V4)

    "fmul V4.4S, V2.4S, V12.4S[3]\n\t"            // v *= c_log2f * .5
    "fcvtzs V5.4S, V4.4S\n\t"                     // to int
    "scvtf  V7.4S, V5.4S\n\t"
    "fsub V6.4S, V4.4S, V7.4S\n\t"                // x = v - intPart
    "fmul V7.4S, V6.4S, V6.4S\n\t"                // xx = x * x
    "fmul V8.4S, V7.4S, V12.4S[1]\n\t"            // v1 = c2 * xx
    "fadd V8.4S, V8.4S, V13.4S\n\t"               // v1 += c_log2f
    "fmul V9.4S, V6.4S, V12.4S[0]\n\t"            // v2 = x *c1
    "fmul V9.4S, V9.4S, V7.4S\n\t"                // v2 *= xx
    "fadd V9.4S, V9.4S, V6.4S\n\t"                // v2 += x
    "fadd V10.4S, V9.4S, V8.4S\n\t"               // v3 = v2 + v1
    "fsub V11.4S, V9.4S, V8.4S\n\t"               // v4 = v2 - v1
    "shl  V5.4S, V5.4S, #24\n\t"
    "add  V7.4S, V10.4S, V5.4S\n\t"               // v3 += intPart << 24
    "fsub V6.4S, V7.4S, V11.4S\n\t"               //  V3 - V4
    "fdiv V2.4S, V7.4S, V6.4S\n\t"                // res = v3 / (V3 - V4)

    // \Sigmoid()
    ////////////////////////////////////////////////////////////

    "st1  {V0.4S, V1.4S }, [%[target]], #32\n\t"
    "st1  {V2.2S }, [%[target]]\n\t"
    : [target] "+r" (pT)
    : [vector] "r" (pV), [matrix] "r" (pM), [VSIZE] "r" (nVectorSize), [ptr_constants] "r" (g_Constants1)
    : "memory", "cc", "x6", "x7", "x9", "v0", "v1", "v2", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13"
   );
 }

///////////////////////////////////////////////////////////

void Query() {
  // Multiply matrix i to h
  VectorMultiplyMatrix(g_pINodes, g_pWIH, g_pHNodes, g_nINodes, g_nHNodes);

  // Multiply matrix h to o
  VectorMultiplyMatrix10(g_pHNodes, g_pWHO, g_ONodes, g_nHNodes);
 }

#else


void Query() {
  int       x, y, n;
  float     f;

  // Multiply matrix i to h
  for (y = 0; y < g_nHNodes; y++) {
    for (x = f = 0, n = y; x < g_nINodes; x++, n += g_nHNodes) f -= g_pINodes[x] * g_pWIH[n];

    g_pHNodes[y] = 1.f / (1.f + exp(f));  // Activation
   }


  // Multiply matrix h to o
  for (y = 0; y < ONODES; y++) {
    for (x = f = 0, n = y; x < g_nHNodes; x++, n += ONODES) f -= g_pHNodes[x] * g_pWHO[n];

    g_ONodes[y] = 1.f / (1.f + exp(f)); // Activation
   }
 }

#endif


int Swap(int nVal) {
  return (nVal >> 24) | ((nVal >> 8) & 0xFF00) | ((nVal & 0xFF00) << 8) | (nVal << 24);
 }


void BackProp(BYTE nVal) {
  int         x, y, n;
  float       f;
  float       ErrOut[ONODES];
  float       ErrHid[g_nHNodes];

#if defined _NEON

  ///////////////////////////////////////////////////////////////////

  for (x = 0; x < ONODES; x++) ErrOut[x] = g_Outputs[nVal][x] - g_ONodes[x];

  asm volatile (
    "mov  x5, %[errout]\n\t"
    "ld1  { V7.4S,  V8.4S }, [x5], #32\n\t"
    "ld1  { V9.2S }, [x5]\n\t"
    "mov  x6, %[matrix]\n\t"
    "mov  x7, %[errhid]\n\t"
    "mov  x5, %[hnodes]\n\t"

    "BP0:\n\t"
    "ld1  { V4.4S,  V5.4S }, [x6], #32\n\t"
    "ld1  { V6.2S }, [x6], #8\n\t"
    "fmul V0.2S, V6.2S, V9.2S\n\t"
    "ld1  { V24.4S,  V25.4S }, [x6], #32\n\t"
    "fmla V0.4S, V4.4S, V7.4S\n\t"
    "subs x5, x5, 2\n\t"
    "ld1  { V26.2S }, [x6], #8\n\t"
    "fmla V0.4S, V5.4S, V8.4S\n\t"
    "faddp V0.4S, V0.4S, V0.4S\n\t"
    "fmul V20.2S, V26.2S, V9.2S\n\t"
    "faddp V0.4S, V0.4S, V0.4S\n\t"
    "fmla V20.4S, V24.4S, V7.4S\n\t"
    "st1 { V0.S } [0], [x7], #4\n\t"
    "fmla V20.4S, V25.4S, V8.4S\n\t"
    "faddp V20.4S, V20.4S, V20.4S\n\t"
    "faddp V20.4S, V20.4S, V20.4S\n\t"
    "st1 { V20.S } [0], [x7], #4\n\t"
    "bne  BP0\n\t"

    "ld1  { V12.4S }, [%[ptr_constants]]\n\t"
    "dup V13.4S, V12.4S[1]\n\t"
    "mov  x5, %[hnodes]\n\t"
    "mov  x6, %[ptr_hnodes]\n\t"
    "mov  x7, %[ptr_errhid]\n\t"
    "mov  x12, xzr\n\t"
    "lsl  x13, %[hnodes], 2\n\t"

    "BP1:\n\t"
    "ld1  { V4.4S,  V5.4S,  V6.4S,  V7.4S }, [x6], #64\n\t"
    "ld1  { V8.4S,  V9.4S,  V10.4S,  V11.4S }, [x7], #64\n\t"
    "fsub V0.4S, V13.4S, V4.4S\n\t"               // (1 - g_pHNodes[y])
    "fsub V1.4S, V13.4S, V5.4S\n\t"
    "fsub V2.4S, V13.4S, V6.4S\n\t"
    "fsub V3.4S, V13.4S, V7.4S\n\t"
    "fmul V0.4S, V0.4S, V4.4S\n\t"                // * g_pHNodes[y]
    "fmul V1.4S, V1.4S, V5.4S\n\t"
    "fmul V2.4S, V2.4S, V6.4S\n\t"
    "fmul V3.4S, V3.4S, V7.4S\n\t"
    "fmul V0.4S, V0.4S, V8.4S\n\t"                // * ErrHid[y]
    "fmul V1.4S, V1.4S, V9.4S\n\t"
    "fmul V2.4S, V2.4S, V10.4S\n\t"
    "fmul V3.4S, V3.4S, V11.4S\n\t"
    "fmul V0.4S, V0.4S, V12.4S[2]\n\t"            // * LEARN_RATE
    "fmul V1.4S, V1.4S, V12.4S[2]\n\t"
    "fmul V2.4S, V2.4S, V12.4S[2]\n\t"
    "fmul V3.4S, V3.4S, V12.4S[2]\n\t"

    "mov  x11,  %[ptr_tnodes]\n\t"                // Prepare inner loop
    "add  x8,   %[ptr_matrix], x12\n\t"
    "mov  x9, x8\n\t"
    "mov  x10, %[inodes]\n\t"

    "BP2:\n\t"
    "ld1  { V4.4S }, [x11], #16\n\t"
    "ld1  { V16.4S, V17.4S, V18.4S, V19.4S }, [x8], x13\n\t"
    "ld1  { V20.4S, V21.4S, V22.4S, V23.4S }, [x8], x13\n\t"

    "fmla V16.4S,  V0.4S, V4.4S[0]\n\t"
    "fmla V17.4S,  V1.4S, V4.4S[0]\n\t"
    "fmla V18.4S,  V2.4S, V4.4S[0]\n\t"
    "fmla V19.4S,  V3.4S, V4.4S[0]\n\t"

    "ld1  { V24.4S, V25.4S, V26.4S, V27.4S }, [x8], x13\n\t"
    "ld1  { V28.4S, V29.4S, V30.4S, V31.4S }, [x8], x13\n\t"

    "fmla V20.4S,  V0.4S, V4.4S[1]\n\t"
    "fmla V21.4S,  V1.4S, V4.4S[1]\n\t"
    "fmla V22.4S,  V2.4S, V4.4S[1]\n\t"
    "fmla V23.4S,  V3.4S, V4.4S[1]\n\t"

    "st1  { V16.4S, V17.4S, V18.4S, V19.4S }, [x9], x13\n\t"

    "fmla V24.4S,  V0.4S, V4.4S[2]\n\t"
    "fmla V25.4S,  V1.4S, V4.4S[2]\n\t"
    "fmla V26.4S,  V2.4S, V4.4S[2]\n\t"
    "fmla V27.4S,  V3.4S, V4.4S[2]\n\t"

    "st1  { V20.4S, V21.4S, V22.4S, V23.4S }, [x9], x13\n\t"

    "subs x10, x10, 4\n\t"
    "fmla V28.4S,  V0.4S, V4.4S[3]\n\t"
    "fmla V29.4S,  V1.4S, V4.4S[3]\n\t"
    "st1  { V24.4S, V25.4S, V26.4S, V27.4S }, [x9], x13\n\t"
    "fmla V30.4S,  V2.4S, V4.4S[3]\n\t"
    "fmla V31.4S,  V3.4S, V4.4S[3]\n\t"
    "st1  { V28.4S, V29.4S, V30.4S, V31.4S }, [x9], x13\n\t"
    "bne  BP2\n\t"

    "subs x5, x5, 16\n\t"
    "add  x12, x12, 64\n\t"
    "bne  BP1\n\t"
    :
    : [errout] "r" (ErrOut), [errhid] "r" (ErrHid), [matrix] "r" (g_pWHO),
      [ptr_errhid] "r" (ErrHid), [ptr_matrix] "r" (g_pWIH), [hnodes] "r" (g_nHNodes),
      [ptr_hnodes] "r" (g_pHNodes), [inodes] "r" (g_nINodes), [ptr_constants] "r" (g_Constants0),
      [ptr_tnodes] "r" (g_pINodes)
    : "memory", "cc", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13",
      "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
      "v8", "v9", "v10", "v11", "v12", "v13",
      "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
      "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
   );

#else

  ///////////////////////////////////////////////////////////////////

  for (x = 0; x < ONODES; x++) ErrOut[x] = g_Outputs[nVal][x] - g_ONodes[x];

  for (y = 0; y < g_nHNodes; y++) {
    for (ErrHid[y] = x = 0, n = y * ONODES; x < ONODES; x++) ErrHid[y] += g_pWHO[n++] * ErrOut[x];

    f = g_fLearnRate * ErrHid[y] * g_pHNodes[y] * (1.f - g_pHNodes[y]);

    for (x = 0, n = y; x < g_nINodes; x++, n += g_nHNodes)  g_pWIH[n] += f * g_pINodes[x];
   }

#endif

  ///////////////////////////////////////////////////////////////////
  // H-O
  for (y = 0; y < ONODES; y++) {
    f = g_fLearnRate * ErrOut[y] * g_ONodes[y] * (1.f - g_ONodes[y]);

    for (x = 0, n = y; x < g_nHNodes; x++, n += ONODES) g_pWHO[n] += f * g_pHNodes[x];
   }
 }


///////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  int       nEpochs(EPOCHS);
  BYTE      *pLearnVal  = NULL;
  BYTE      *pTestVal   = NULL;

  printf("START\n");

  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "-e:", 3) == 0) {
      nEpochs = atoi(&(argv[i])[3]);
     }

    if (strncmp(argv[i], "-h:", 3) == 0) {
      g_nHNodes = atoi(&(argv[i])[3]);
     }

    if (strncmp(argv[i], "-l:", 3) == 0) {
      g_fLearnRate = atof(&(argv[i])[3]);
     }
   }

  if (nEpochs   <  1) nEpochs   = EPOCHS;
  if (g_nHNodes < 10) g_nHNodes = HNODES;
  if (g_fLearnRate <= 0 || g_fLearnRate > 1) g_fLearnRate = LEARN_RATE;


#if defined _NEON

  g_nHNodes = (g_nHNodes + 15) / 16 * 16;

  g_Constants0[0] = 0;
  g_Constants0[1] = 1;
  g_Constants0[2] = g_fLearnRate;
  g_Constants0[3] = 0;

#endif

  try {
    int         e, i, j, k, m, nHit(0);
    float       Inputs[256];
    BYTE        Temp[16];

    for (i = 0; i < 256; i++) Inputs[i] = i * .99 / 255. + .01;

    printf("Loading training data ... ");
    ::fflush(stdout);

    {
      CFile File(FILENAME_LD, O_RDONLY);

      File.Read(Temp, 16);
      g_nLearnRecords = Swap(*(int *) &Temp[4]);
      g_nINodes       = Swap(*(int *) &Temp[8]) * Swap(*(int *) &Temp[12]);
      g_pLNodes       = new float[g_nLearnRecords * g_nINodes];
      pLearnVal       = new BYTE[g_nLearnRecords];
      g_pINodes       = g_pLNodes;

      BYTE  ReadBuff[g_nINodes];

      for (i = 0; i < g_nLearnRecords; i++) {
        File.Read(ReadBuff, g_nINodes);

        for (j = 0; j < g_nINodes; j++) *g_pINodes++ = Inputs[ReadBuff[j]];
       }

      File.Close();
     }

    if (pLearnVal != NULL) {
      CFile File(FILENAME_LL, O_RDONLY);

      File.Seek(8, SEEK_CUR);

      if (g_nLearnRecords != File.Read(pLearnVal, g_nLearnRecords)) puts("ERR1");

      File.Close();
     }

    printf("done\n");

    printf("Loading test data ... ");   ::fflush(stdout);

    {
      CFile File(FILENAME_TD, O_RDONLY);

      File.Read(Temp, 16);
      g_nTestRecords  = Swap(*(int *) &Temp[4]);
      g_pTNodes       = new float[g_nTestRecords * g_nINodes];
      pTestVal        = new BYTE[g_nTestRecords];
      g_pINodes       = g_pTNodes;

      BYTE  ReadBuff[g_nINodes];

      for (i = 0; i < g_nTestRecords; i++) {
        File.Read(ReadBuff, g_nINodes);

        for (j = 0; j < g_nINodes; j++) *g_pINodes++ = Inputs[ReadBuff[j]];
       }

      File.Close();
     }

    if (pTestVal != NULL) {
      CFile File(FILENAME_TL, O_RDONLY);

      File.Seek(8, SEEK_CUR);

      if (g_nTestRecords != File.Read(pTestVal, g_nTestRecords)) puts("ERR3");

      File.Close();
     }

    printf("done\n");

    ///////////////////////////////////////////////////////////////////

    g_pWIH      = new float[g_nINodes * g_nHNodes];
    g_pWHO      = new float[g_nHNodes * ONODES];
    g_pHNodes   = new float[g_nHNodes];

    printf( "Layers: %d, %d, %d - Records: %d, %d, LearnRate: %.5f, RAM usage %d MByte\n",
          g_nINodes, g_nHNodes, ONODES, g_nLearnRecords, g_nTestRecords, g_fLearnRate,
          (g_nLearnRecords * g_nINodes * 4 + g_nLearnRecords * g_nINodes +
          g_nTestRecords * g_nINodes * 4 + g_nTestRecords * g_nINodes +
          g_nINodes * g_nHNodes * 4 + g_nHNodes * ONODES * 4) / (1 << 20));

    ///////////////////////////////////////////////////////////////////
    // INIT

    for (i = 0; i < g_nINodes; i++) {
      for (j = 0; j < g_nHNodes; j++) g_pWIH[i * g_nHNodes + j] = (::rand() % 100) / 100. - .5;
     }

    for (i = 0; i < g_nHNodes; i++) {
      for (j = 0; j < ONODES; j++) g_pWHO[i * ONODES + j] = (::rand() % 100) / 100. - .5;
     }

    for (i = 0; i < ONODES; i++) {
      for (j = 0; j < ONODES; j++) g_Outputs[i][j] = .01;

      g_Outputs[i][i] = .99;
     }


    ///////////////////////////////////////////////////////////////////

    for (e = 0; e < nEpochs; e++) {
      ///////////////////////////////////////////////////////////////////
      // Training
      {
        CDelay D("%2.2fs - Testing ... ");

        printf("Training epoch %2d ... ", e);
        ::fflush(stdout);

        for (i = k = 0, g_pINodes = g_pLNodes; i < g_nLearnRecords; i++, g_pINodes += g_nINodes) {
          Query();
          BackProp(pLearnVal[i]);
         }
       }

      ::fflush(stdout);

      ///////////////////////////////////////////////////////////////////
      // Test

      for (i = nHit = 0,  g_pINodes = g_pTNodes; i < g_nTestRecords; i++,  g_pINodes += g_nINodes) {
        Query();

        for (j = 0, m = 1; m < ONODES; m++) {
          if (g_ONodes[m] > g_ONodes[j]) j = m;
         }

        if (j == pTestVal[i]) nHit++;
       }

      printf("Hits: %5d, Error rate: %5.2f%%\n", nHit, (g_nTestRecords - nHit) * 100. / g_nTestRecords);
     }

    ///////////////////////////////////////////////////////////////////

   }
  catch (CFileException *e) {
    e->PrintError("\nError reading file");
    e->Delete();
   }

  if (g_pWIH    != NULL) delete [] g_pWIH;
  if (g_pWHO    != NULL) delete [] g_pWHO;
  if (g_pLNodes != NULL) delete [] g_pLNodes;
  if (g_pTNodes != NULL) delete [] g_pTNodes;
  if (g_pHNodes != NULL) delete [] g_pHNodes;
  if (pLearnVal != NULL) delete [] pLearnVal;
  if (pTestVal  != NULL) delete [] pTestVal;

  return 0;
 }
