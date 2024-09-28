#ifndef CONV_H
#define CONV_H
#include "utils.h"
#include <math.h>
typedef struct {
    int input_channel;
    int output_channel;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    float input_thres; 
    quant_type quant;
    union {
        qtype *weights_b;    // For BNN and TBN layer
        struct {
            qtype *weights_t0; //(output channel, input channel, kernelsize, kernelsize)
            qtype *weights_t1;
        };              // For TNN layer
        float *weights_f;
    };
} conv2d_layer;

typedef struct {
    int input_channel;
    int output_channel;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    float input_thres;
    quant_type quant;
    union {
        qtype ***weights_b; // For BNN and TBN layer
        struct {
            qtype ***weights_t0; //(output channel, input channel, kernelsize)
            qtype ***weights_t1;
        }; // For TNN layer
        float ***weights_f; // For FP layer
    };
} conv1d_layer;

typedef union {
    ttype **t;
    qtype **b;
} conv1d_input;

typedef union {
    ttype t;
    qtype b;

} conv2d_input;

conv2d_layer* create_conv2d_layer(int input_channel, int output_channel, int kernel_size, int stride, int padding, int dilation, quant_type quant);
float *conv2d_forward(conv2d_layer *layer, float *input, int input_height, int input_width);

float *max_pooling_2d(float *input, int input_channels, int input_height, int input_width);
float *max_pooling_2d_k(float *input, int input_channels, int input_height, int input_width, int kernel_size, int stride);
#endif // CONV_H