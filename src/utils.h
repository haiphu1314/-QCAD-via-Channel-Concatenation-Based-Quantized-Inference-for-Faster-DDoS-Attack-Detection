#include "conv.h"
#include <stdint.h>
#ifndef UTILS_H
#define UTILS_H
#define MAX_CHARS_LINE 1024

#define USE_LONG

#ifdef USE_LONG
#define SIZEQUANT 64
#define qtype long
#define QFSCAN 0x%lx\n
#else
#define SIZEQUANT 32
#define qtype int
#define QFSCAN 0x%x\n

#endif

// #define qt int


typedef struct {
    qtype bit_0;
    qtype bit_1;
} ttype;

typedef enum {
    BNN,
    TBN,
    TNN,
    FP
} quant_type;

int bitCount(qtype n);
int sign(int x);
int count_layers(const char* filename);
float *flatto1d(float *input, int input_channel, int input_height, int input_width);

#endif // UTILS_H