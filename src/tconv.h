#ifndef TCONV_H
#define TCONV_H
#include "utils.h"

typedef struct {
    int input_channel;
    int output_channel;
    int kernel_size;
    int stride;
    int padding;
    int output_padding;
    int dilation;
    float input_thres; 
    quant_type quant;
    union {
        int ****weights;    // For BNN and TBN layer
        struct {
            int ****weights_0; //(output channel, input channel, kernelsize, kernelsize)
            int ****weights_1;
        };              // For TNN layer
    };
} tconv_layer;

typedef union {
    ttype ***t;
    int ***b;
} tconv_input;

void conv_transpose_2d(float *input, float *output, float *weights, float *bias, 
                       int in_channels, int out_channels, 
                       int in_height, int in_width, 
                       int kernel_size, int stride, int padding, int output_padding);

#endif // TCONV_H