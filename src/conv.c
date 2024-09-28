

/**
 * @ Author: Hai Phu
 * @ Email:  haiphu@hcmut.edu.vn
 * @ Create Time: 2024-07-07 21:00:35
 * @ Modified time: 2024-09-28 12:50:57
 * @ Description:
 */

#include "conv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "float.h"
#include <stdint.h>

#include <time.h>
// #define RAND
// #define MC
/**
 * @brief Creates and initializes a convolutional layer with specified parameters.
 *
 * This function allocates memory for a convolutional layer structure and initializes its parameters
 * based on the given input channels, output channels, kernel size, stride, padding, dilation, and
 * quantization type. Depending on the quantization type, it allocates memory for the appropriate weights.
 *
 * @param input_channel The number of input channels for the convolutional layer.
 * @param output_channel The number of output channels for the convolutional layer.
 * @param kernel_size The size of the convolutional kernel.
 * @param stride The stride of the convolution.
 * @param padding The padding added to the input.
 * @param dilation The dilation rate of the convolutional kernel.
 * @param quant The quantization type for the convolutional layer. Possible values include:
 *              - BNN: Binary Neural Network
 *              - TBN: Ternary Binary Neural Network
 *              - TNN: Ternary Neural Network
 *
 * @return A pointer to the initialized conv_layer structure. If memory allocation fails or an unknown
 *         quantization type is specified, the function prints an error message and terminates the program.
 */
conv2d_layer *create_conv2d_layer(int input_channel, int output_channel, int kernel_size, int stride, int padding, int dilation, quant_type quant)
{
    conv2d_layer *layer = (conv2d_layer *)malloc(sizeof(conv2d_layer));
    layer->input_channel = input_channel;
    layer->output_channel = output_channel;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->dilation = dilation;

    layer->input_thres = 0.0;
    layer->quant = quant;
    int inputq_size = (input_channel % SIZEQUANT) == 0 ? (input_channel / SIZEQUANT) : (input_channel / SIZEQUANT + 1);
    int dim1 = output_channel;
    int dim2 = inputq_size;
    int dim3 = kernel_size;
    int dim4 = kernel_size;
    srand(time(0));
    switch (quant)
    {
    case BNN:
    case TBN:
        // layer->weights_b = allocate_4d_qtype_array(dim1, dim2, dim3, dim4);

        layer->weights_b = (qtype *)malloc(dim1 * dim2 * dim3 * dim4 * sizeof(qtype *));
        if (layer->weights_b == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights\n");
            exit(1);
        }
#ifdef RAND
        for (int i = 0; i < dim1 * dim2 * dim3 * dim4; ++i)
        {
            layer->weights_b[i] = (qtype)rand(); 
        }
#endif
        break;

    case TNN:
        layer->weights_t0 = (qtype *)malloc(dim1 * dim2 * dim3 * dim4 * sizeof(qtype *));
        if (layer->weights_t0 == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights_0\n");
            exit(1);
        }

        layer->weights_t1 = (qtype *)malloc(dim1 * dim2 * dim3 * dim4 * sizeof(qtype *));
        if (layer->weights_t1 == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights_1\n");
            exit(1);
        }
#ifdef RAND
        for (int i = 0; i < dim1 * dim2 * dim3 * dim4; ++i)
        {
            layer->weights_t0[i] = (qtype)rand();
            layer->weights_t1[i] = (qtype)rand();
        }
#endif
        break;

    case FP:
        // printf("%d \n", dim1 * dim2 * dim3 * dim4);
        layer->weights_f = (float *)malloc(dim1 * input_channel * dim3 * dim4 * sizeof(float *));
        if (layer->weights_f == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights\n");
            exit(1);
        }
#ifdef RAND
        for (int i = 0; i < dim1 * dim2 * dim3 * dim4; ++i)
        {
            layer->weights_f[i] = (float)rand();
        }
#endif
        break;
    default:
        fprintf(stderr, "create_conv_layer: Unknown quantization type \n");
        exit(1);
    }
    return layer;
}

/**
 * @brief Performs the forward pass for a convolutional layer with quantized inputs.
 *
 * This function computes the output of a convolutional layer given the input data, layer parameters,
 * and the dimensions of the input. It handles different quantization types (BNN, TBN, TNN) by quantizing
 * the input data appropriately and then performing the forward pass computation.
 *
 * @param layer Pointer to the conv_layer structure containing the layer parameters.
 * @param input Pointer to the input data array.
 * @param input_height The height of the input data.
 * @param input_width The width of the input data.
 *
 * @return A pointer to the output data array.
 */
float *conv2d_forward(conv2d_layer *layer, float *input, int input_height, int input_width)
{
    int input_channel = layer->input_channel;
    int output_channel = layer->output_channel;
    int kernel_size = layer->kernel_size;
    int stride = layer->stride;
    int padding = layer->padding;
    int dilation = layer->dilation;
    float input_thres = layer->input_thres;
    quant_type quant = layer->quant;

    int inputq_size = (input_channel % SIZEQUANT) ? (input_channel / SIZEQUANT + 1) : (input_channel / SIZEQUANT);
    int output_height = (int)((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    int output_width = (int)((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    float *output = (float *)malloc(output_channel * output_height * output_width * sizeof(float *));
    int dim1_input = inputq_size;
    int dim2_input = input_height;
    int dim3_input = input_width;
    conv2d_input input_quant[dim1_input * dim2_input * dim3_input];

    switch (quant)
    {
    case BNN:
        for (int i = 0; i < dim1_input * dim2_input * dim3_input; i++)
        {
            input_quant[i].b = 0;
        }
        for (int c = 0; c < input_channel; c++)
        {

            for (int h = 0; h < dim2_input; h++)
            {
                for (int w = 0; w < dim3_input; w++)
                {
                    int index_inq = ((c / SIZEQUANT) * input_height * input_width) + (h * input_width) + w;
                    int index_in = (c * input_height * input_width) + (h * input_width) + w;

                    if (input[index_in] < input_thres)
                    {
                        input_quant[index_inq].b |= (qtype)1 << (c % SIZEQUANT);
                    }
                }
            }
        }

        break;
    case TBN:
    case TNN:
        for (int i = 0; i < dim1_input * dim2_input * dim3_input; i++)
        {
            input_quant[i].t.bit_0 = 0;
            input_quant[i].t.bit_1 = 0;
        }
        for (int c = 0; c < input_channel; c++)
        {
            for (int h = 0; h < dim2_input; h++)
            {
                for (int w = 0; w < dim3_input; w++)
                {
                    int index_inq = ((c / SIZEQUANT) * input_height * input_width) + (h * input_width) + w;
                    int index_in = (c * input_height * input_width) + (h * input_width) + w;
                    if (input[index_in] >= input_thres)
                    {
                        input_quant[index_inq].t.bit_1 |= (qtype)1 << (c % SIZEQUANT);
                    }
                    else if (input[index_in] <= -input_thres)
                    {
                        input_quant[index_inq].t.bit_0 |= (qtype)1 << (c % SIZEQUANT);
                    }
                }
            }
        }
        break;
    case FP:
        break;

    default:
        fprintf(stderr, "conv_forward: Unknown quantization type\n");
        exit(1);
    }

    switch (quant)
    {
    case BNN:

#ifdef MC
#pragma omp parallel for collapse(3)
#endif
        for (int co = 0; co < output_channel; co++)
        {
            for (int y = 0; y < output_height; y++)
            {
                for (int x = 0; x < output_width; x++)
                {
                    int cnt_minus_one = 0;
                    int cnt_zero = 0;
                    int base_x = x * stride - padding;
                    int base_y = y * stride - padding;

                    for (int kc = 0; kc < inputq_size; kc++)
                    {
                        for (int ky = 0; ky < kernel_size; ky++)
                        {
                            int padded_y = base_y + ky * dilation;

                            for (int kx = 0; kx < kernel_size; kx++)
                            {
                                int padded_x = base_x + kx * dilation;
                                if (padded_x < 0 || padded_y < 0 || padded_x >= input_width || padded_y >= input_height)
                                {
                                    cnt_zero += 1;
                                }
                                else
                                {
                                    qtype input_b = input_quant[(kc * input_height * input_width) + (padded_y * input_width) + padded_x].b;
                                    qtype weight_b = layer->weights_b[(co * inputq_size * kernel_size * kernel_size) + (kc * kernel_size * kernel_size) + (ky * kernel_size) + kx];

                                    qtype result_bit = input_b ^ weight_b;
                                    cnt_minus_one += bitCount(result_bit);
                                }
                            }
                        }
                    }
                    qtype cnt_one = (kernel_size * kernel_size - cnt_zero) * input_channel - cnt_minus_one;
                    output[(co * output_height * output_width) + (y * output_width) + x] = (float)(cnt_one - cnt_minus_one);
                }
            }
        }

        break;
    case TBN:

#ifdef MC
#pragma omp parallel for collapse(3)
#endif
        for (int co = 0; co < output_channel; co++)
        {
            for (int y = 0; y < output_height; y++)
            {
                for (int x = 0; x < output_width; x++)
                {
                    int cnt_minus_one = 0;
                    int cnt_one = 0;
                    int cnt_zero = 0;
                    int base_x = x * stride - padding;
                    int base_y = y * stride - padding;

                    for (int kc = 0; kc < inputq_size; kc++)
                    {
                        for (int ky = 0; ky < kernel_size; ky++)
                        {
                            int padded_y = base_y + ky * dilation;

                            for (int kx = 0; kx < kernel_size; kx++)
                            {
                                int padded_x = base_x + kx * dilation;
                                if (padded_x < 0 || padded_y < 0 || padded_x >= input_width || padded_y >= input_height)
                                {
                                    cnt_zero += 1;
                                }
                                else
                                {
                                    qtype weight = layer->weights_b[(co * inputq_size * kernel_size * kernel_size) + (kc * kernel_size * kernel_size) + (ky * kernel_size) + kx];
                                    qtype i_weight = ~weight;
                                    ttype input_t = input_quant[(kc * input_height * input_width) + (padded_y * input_width) + padded_x].t;
                                    qtype bit_1 = input_t.bit_1;
                                    qtype bit_0 = input_t.bit_0;
                                    qtype result_bit1 = (bit_1 & i_weight) | (bit_0 & weight);
                                    qtype result_bit0 = (bit_1 & weight) | (bit_0 & i_weight);
                                    cnt_minus_one += bitCount(result_bit0);
                                    cnt_one += bitCount(result_bit1);
                                }
                            }
                        }
                    }
                    output[(co * output_height * output_width) + (y * output_width) + x] = ((float)cnt_one - (float)cnt_minus_one);
                }
            }
        }
        break;
    case TNN:
#ifdef MC
#pragma omp parallel for collapse(3)
#endif
        for (int co = 0; co < output_channel; co++)
        {
            for (int y = 0; y < output_height; y++)
            {
                for (int x = 0; x < output_width; x++)
                {
                    int cnt_minus_one = 0;
                    int cnt_one = 0;
                    int cnt_zero = 0;

                    int base_x = x * stride - padding;
                    int base_y = y * stride - padding;

                    for (int kc = 0; kc < inputq_size; kc++)
                    {
                        for (int ky = 0; ky < kernel_size; ky++)
                        {
                            int padded_y = base_y + ky * dilation;

                            for (int kx = 0; kx < kernel_size; kx++)
                            {
                                int padded_x = base_x + kx * dilation;
                                if (padded_x < 0 || padded_y < 0 || padded_x >= input_width || padded_y >= input_height)
                                {
                                    cnt_zero += 1;
                                }
                                else
                                {
                                    ttype input_t = input_quant[(kc * input_height * input_width) + (padded_y * input_width) + padded_x].t;
                                    qtype bit_1 = input_t.bit_1;
                                    qtype bit_0 = input_t.bit_0;
                                    qtype weight_t0 = layer->weights_t0[(co * inputq_size * kernel_size * kernel_size) + (kc * kernel_size * kernel_size) + (ky * kernel_size) + kx];
                                    qtype weight_t1 = layer->weights_t1[(co * inputq_size * kernel_size * kernel_size) + (kc * kernel_size * kernel_size) + (ky * kernel_size) + kx];

                                    qtype result_bit1 = (bit_1 & weight_t1) | (bit_0 & weight_t0);
                                    qtype result_bit0 = (bit_1 & weight_t0) | (bit_0 & weight_t1);
                                    cnt_minus_one += bitCount(result_bit0);
                                    cnt_one += bitCount(result_bit1);
                                }
                            }
                        }
                    }
                    output[(co * output_height * output_width) + (y * output_width) + x] = (float)(cnt_one - cnt_minus_one);
                }
            }
        }

        break;
    case FP:
#ifdef MC
#pragma omp parallel for collapse(3)
#endif
        for (int co = 0; co < output_channel; co++)
        {
            for (int y = 0; y < output_height; y++)
            {
                for (int x = 0; x < output_width; x++)
                {
                    float sum = 0.0;
                    for (int kc = 0; kc < input_channel; kc++)
                    {
                        for (int ky = 0; ky < kernel_size; ky++)
                        {
                            for (int kx = 0; kx < kernel_size; kx++)
                            {
                                int ih = y * layer->stride - layer->padding + ky * layer->dilation;
                                int iw = x * layer->stride - layer->padding + kx * layer->dilation;
                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width)
                                {
                                    sum += input[kc * ih * iw] * layer->weights_f[co * kc * ky * kx];
                                }
                            }
                        }
                    }
                    output[co * y * x] = sum;
                }
            }
        }
        break;
    }

    free(input);
    return output;
}

float *max_pooling_2d(float *input, int input_channels, int input_height, int input_width)
{
    int pool_size = 2;
    int stride = 2;
    int output_height = input_height / pool_size;
    int output_width = input_width / pool_size;

    float *output = (float *)malloc(input_channels * output_height * output_width * sizeof(float));

    for (int c = 0; c < input_channels; c++)
    {
        for (int i = 0; i < output_height; i++)
        {
            for (int j = 0; j < output_width; j++)
            {
                float max_val = -__FLT_MAX__;
                for (int x = 0; x < pool_size; x++)
                {
                    for (int y = 0; y < pool_size; y++)
                    {
                        int input_row = i * stride + x;
                        int input_col = j * stride + y;
                        float val = input[c * input_height * input_width + input_row * input_width + input_col];
                        if (val > max_val)
                        {
                            max_val = val;
                        }
                    }
                }
                output[c * output_height * output_width + i * output_width + j] = max_val;
            }
        }
    }
    free(input);
    return output;
}

float *max_pooling_2d_k(float *input, int input_channels, int input_height, int input_width, int kernel_size, int stride)
{
    int output_height = (input_height - kernel_size) / stride + 1;
    int output_width = (input_width - kernel_size) / stride + 1;
    float *output = (float *)malloc(input_channels * output_height * output_width * sizeof(float));
    for (int c = 0; c < input_channels; ++c)
    {
        for (int i = 0; i < output_height; ++i)
        {
            for (int j = 0; j < output_width; ++j)
            {
                float max_val = -__FLT_MAX__;
                for (int ki = 0; ki < kernel_size; ++ki)
                {
                    for (int kj = 0; kj < kernel_size; ++kj)
                    {
                        int input_row = i * stride + ki;
                        int input_col = j * stride + kj;
                        int input_index = c * input_height * input_width + input_row * input_width + input_col;

                        if (input[input_index] > max_val)
                        {
                            max_val = input[input_index];
                        }
                    }
                }
                int output_index = c * output_height * output_width + i * output_width + j;
                output[output_index] = max_val;
            }
        }
    }
    free(input);
    return output;
}
