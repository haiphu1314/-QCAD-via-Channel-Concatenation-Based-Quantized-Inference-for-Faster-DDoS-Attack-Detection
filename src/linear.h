#ifndef LINEAR_H
#define LINEAR_H
#include "utils.h"

typedef struct {
    int input_channel;
    int output_channel;
    float input_thres; 
    union {
        qtype *weights_b;    // For BNN and TBN layer
        struct {
            qtype *weights_t0;
            qtype *weights_t1;
        };              // For TNN layer
        float *weights_f;
    };
    quant_type quant;
} linear_layer;

typedef union {
    ttype t;
    qtype b;
} linear_input;

linear_layer* create_linear_layer(int input_channel, int output_channel, quant_type quant);
float* linear_forward(linear_layer* layer, float* input);
#endif // LINEAR_H