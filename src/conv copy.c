

/**
 * @ Author: Hai Phu
 * @ Email:  haiphu@hcmut.edu.vn
 * @ Create Time: 2024-07-07 21:00:35
 * @ Modified time: 2024-08-01 23:47:12
 * @ Description:
 */

#include "conv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "float.h"
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
    switch (quant)
    {
    case BNN:
    case TBN:
        layer->weights_b = allocate_4d_qtype_array(dim1, dim2, dim3, dim4);
        if (layer->weights_b == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights\n");
            exit(1);
        }
        break;
    case TNN:
        layer->weights_t0 = allocate_4d_qtype_array(dim1, dim2, dim3, dim4);
        if (layer->weights_t0 == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights_0\n");
            exit(1);
        }
        layer->weights_t1 = allocate_4d_qtype_array(dim1, dim2, dim3, dim4);
        if (layer->weights_t1 == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights_1\n");
            exit(1);
        }
        break;
    case FP:
        layer->weights_f = allocate_4d_float_array(dim1, input_channel, dim3, dim4);
        if (layer->weights_f == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights\n");
            exit(1);
        }
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
float ***conv2d_forward(conv2d_layer *layer, float ***input, int input_height, int input_width)
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
    int output_height = (int)((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1; // height
    int output_width = (int)((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;   // width
    // float ***output;
    float ***output = allocate_3d_float_array(output_channel, output_height, output_width);
    int dim1_input = inputq_size;
    int dim2_input = input_height; // height
    int dim3_input = input_width;  // width
    conv2d_input input_quant[dim1_input][dim2_input][dim3_input];
    
    switch (quant)
    {
    case BNN:
        for (int c = 0; c < input_channel; c++)
        {
            for (int h = 0; h < dim2_input; h++)
            {
                for (int w = 0; w < dim3_input; w++)
                {
                    if (input[c][h][w] < input_thres)
                    {
                        input_quant[c / SIZEQUANT][h][w].b |= 1 << (c % SIZEQUANT);
                    }
                }
            }
        }
        break;
    case TBN:
    case TNN:
        for (int c = 0; c < input_channel; c++)
        {
            for (int h = 0; h < dim2_input; h++)
            {
                for (int w = 0; w < dim3_input; w++)
                {
                    if (input[c][h][w] >= input_thres)
                    {
                        input_quant[c / SIZEQUANT][h][w].t.bit_1 |= 1 << (c % SIZEQUANT);
                    }
                    else if (input[c][h][w] <= -input_thres)
                    {
                        input_quant[c / SIZEQUANT][h][w].t.bit_0 |= 1 << (c % SIZEQUANT);
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
        for (int co = 0; co < output_channel; co++)
        {
            for (int y = 0; y < output_height; y++)
            {
                for (int x = 0; x < output_width; x++)
                {
                    int cnt_minus_one = 0;
                    int cnt_zero = 0;
                    
                    for (int kc = 0; kc < inputq_size; kc++)
                    {
                        for (int ky = 0; ky < kernel_size; ky++)
                        {
                            for (int kx = 0; kx < kernel_size; kx++)
                            {
                                int padded_x = (x * stride + kx * dilation - padding) * dilation;
                                int padded_y = (y * stride + ky * dilation - padding) * dilation;
                                if (padded_x < 0 || padded_y < 0 || padded_x >= input_width || padded_y >= input_height)
                                {
                                    cnt_zero += 1;
                                }
                                else
                                {
                                    
                                    int result_bit = input_quant[kc][padded_y][padded_x].b ^ layer->weights_b[co][kc][ky][kx];
                                    
                                    cnt_minus_one += bitCount(result_bit);
                                }
                            }
                        }
                    }
                    int cnt_one = (kernel_size * kernel_size - cnt_zero) * input_channel - cnt_minus_one;
                    output[co][y][x] = (float)(cnt_one - cnt_minus_one);
                }
            }
        }
        break;
    case TBN:
        for (int co = 0; co < output_channel; co++)
        {
            for (int y = 0; y < output_height; y++)
            {
                for (int x = 0; x < output_width; x++)
                {
                    int cnt_minus_one = 0;
                    int cnt_one = 0;
                    int cnt_zero = 0;
                    for (int kc = 0; kc < inputq_size; kc++)
                    {
                        for (int ky = 0; ky < kernel_size; ky++)
                        {
                            for (int kx = 0; kx < kernel_size; kx++)
                            {
                                int padded_x = (x * stride + kx * dilation - padding) * dilation;
                                int padded_y = (y * stride + ky * dilation - padding) * dilation;
                                if (padded_x < 0 || padded_y < 0 || padded_x >= input_width || padded_y >= input_height)
                                {
                                    cnt_zero += 1;
                                }
                                else
                                {
                                    int weight = layer->weights_b[co][kc][ky][kx];
                                    int i_weight = ~weight;
                                    int result_bit0 = (input_quant[kc][padded_y][padded_x].t.bit_1 & i_weight) | (input_quant[kc][padded_y][padded_x].t.bit_0 & weight);
                                    int result_bit1 = (input_quant[kc][padded_y][padded_x].t.bit_1 & weight) | (input_quant[kc][padded_y][padded_x].t.bit_0 & i_weight);
                                    cnt_minus_one += bitCount(result_bit0);
                                    cnt_one += bitCount(result_bit1);
                                }
                            }
                        }
                    }
                    output[co][y][x] = (float)(cnt_one - cnt_minus_one);
                }
            }
        }
        break;

    case TNN:
        // int check = 0;
        for (int co = 0; co < output_channel; co++)
        {
            for (int y = 0; y < output_height; y++)
            {
                for (int x = 0; x < output_width; x++)
                {
                    int cnt_minus_one = 0;
                    int cnt_one = 0;
                    int cnt_zero = 0;
                    // printf('dsadsa\n');
                    for (int kc = 0; kc < inputq_size; kc++)
                    {
                        for (int ky = 0; ky < kernel_size; ky++)
                        {
                            for (int kx = 0; kx < kernel_size; kx++)
                            {
                                int padded_x = (x * stride + kx * dilation - padding) * dilation;
                                int padded_y = (y * stride + ky * dilation - padding) * dilation;
                                if (padded_x < 0 || padded_y < 0 || padded_x >= input_width || padded_y >= input_height)
                                {
                                    cnt_zero += 1;
                                }
                                else
                                {
                                    int result_bit0 = (input_quant[kc][padded_y][padded_x].t.bit_1 & layer->weights_t0[co][kc][ky][kx]) | (input_quant[kc][padded_y][padded_x].t.bit_0 & layer->weights_t1[co][kc][ky][kx]);
                                    int result_bit1 = (input_quant[kc][padded_y][padded_x].t.bit_1 & layer->weights_t1[co][kc][ky][kx]) | (input_quant[kc][padded_y][padded_x].t.bit_0 & layer->weights_t0[co][kc][ky][kx]);
                                    cnt_minus_one += bitCount(result_bit0);
                                    cnt_one += bitCount(result_bit1);
                                }
                            }
                        }
                    }
                    output[co][y][x] = (float)(cnt_one - cnt_minus_one);
                }
            }
        }
        break;
    case FP:
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
                                    sum += input[kc][ih][iw] * layer->weights_f[co][kc][ky][kx];
                                }
                            }
                        }
                    }
                    output[co][y][x] = sum;
                }
            }
        }
        break;
    }
    // free_3d_float_array(input, input_channel, input_height);
    return output;
}
/**
 * @brief Creates and initializes a 1D convolutional layer with specified parameters and quantization type.
 *
 * This function allocates memory for a 1D convolutional layer structure and initializes its parameters
 * based on the given input channels, output channels, kernel size, stride, padding, dilation, and
 * quantization type. It also allocates memory for the layer's weights according to the quantization type.
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
 *              - FP: Floating Point
 *
 * @return A pointer to the initialized conv1d_layer structure. If memory allocation fails for the layer or weights,
 *         the function prints an error message and terminates the program.
 */
conv1d_layer *create_conv1d_layer(int input_channel, int output_channel, int kernel_size, int stride, int padding, int dilation, quant_type quant) {
    conv1d_layer *layer = (conv1d_layer *)malloc(sizeof(conv1d_layer));
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

    switch (quant) {
    case BNN:
    case TBN:
        layer->weights_b = allocate_3d_qtype_array(dim1, dim2, dim3);
        if (layer->weights_b == NULL) {
            fprintf(stderr, "Memory allocation failed for weights\n");
            exit(1);
        }
        break;
    case TNN:
        layer->weights_t0 = allocate_3d_qtype_array(dim1, dim2, dim3);
        if (layer->weights_t0 == NULL) {
            fprintf(stderr, "Memory allocation failed for weights_t0\n");
            exit(1);
        }
        layer->weights_t1 = allocate_3d_qtype_array(dim1, dim2, dim3);
        if (layer->weights_t1 == NULL) {
            fprintf(stderr, "Memory allocation failed for weights_t1\n");
            exit(1);
        }
        break;
    case FP:
        layer->weights_f = allocate_3d_float_array(dim1, input_channel, dim3);
        if (layer->weights_f == NULL) {
            fprintf(stderr, "Memory allocation failed for weights\n");
            exit(1);
        }
        break;
    default:
        fprintf(stderr, "create_conv1d_layer: Unknown quantization type\n");
        exit(1);
    }
    return layer;
}

/**
 * @brief Performs the forward pass for a 1D convolutional layer with quantized or floating-point inputs.
 *
 * This function computes the output of a 1D convolutional layer given the input data, layer parameters,
 * and the length of the input. It handles different quantization types (BNN, TBN, TNN, FP) by quantizing
 * the input data appropriately and then performing the convolution operation.
 *
 * @param layer Pointer to the conv1d_layer structure containing the layer parameters.
 * @param input Pointer to the input data array (2D array).
 * @param input_length The length of the input data.
 *
 * @return A pointer to the output data array (2D array). The memory for the output array is allocated within
 *         this function and should be freed by the caller when no longer needed.
 */
float **conv1d_forward(conv1d_layer *layer, float **input, int input_length) {
    int input_channel = layer->input_channel;
    int output_channel = layer->output_channel;
    int kernel_size = layer->kernel_size;
    int stride = layer->stride;
    int padding = layer->padding;
    int dilation = layer->dilation;
    float input_thres = layer->input_thres;
    quant_type quant = layer->quant;

    int inputq_size = (input_channel % SIZEQUANT) ? (input_channel / SIZEQUANT + 1) : (input_channel / SIZEQUANT);
    int output_length = (int)((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    float **output = allocate_2d_float_array(output_channel, output_length);

    // Quantize input
    conv1d_input input_quant;
    int dim1_input = inputq_size;
    int dim2_input = input_length;

    switch (quant) {
    case BNN:
        input_quant.b = allocate_2d_qtype_array(dim1_input, dim2_input);
        for (int c = 0; c < input_channel; c++) {
            for (int l = 0; l < dim2_input; l++) {
                if (input[c][l] < input_thres) {
                    input_quant.b[c / SIZEQUANT][l] |= 1 << (c % SIZEQUANT);
                }
            }
        }
        break;
    case TBN:
    case TNN:
        input_quant.t = allocate_2d_ttype_array(dim1_input, dim2_input);
        for (int c = 0; c < input_channel; c++) {
            for (int l = 0; l < dim2_input; l++) {
                if (input[c][l] >= input_thres) {
                    input_quant.t[c / SIZEQUANT][l].bit_1 |= 1 << (c % SIZEQUANT);
                } else if (input[c][l] <= -input_thres) {
                    input_quant.t[c / SIZEQUANT][l].bit_0 |= 1 << (c % SIZEQUANT);
                }
            }
        }
        break;
    case FP:
        break;
    default:
        fprintf(stderr, "conv1d_forward: Unknown quantization type\n");
        exit(1);
    }

    // Convolution operation
    switch (quant) {
    case BNN:
        for (int co = 0; co < output_channel; co++) {
            for (int l = 0; l < output_length; l++) {
                int cnt_minus_one = 0;
                int cnt_zero = 0;
                for (int kc = 0; kc < inputq_size; kc++) {
                    for (int kl = 0; kl < kernel_size; kl++) {
                        int padded_l = l * stride + kl * dilation - padding;
                        if (padded_l < 0 || padded_l >= input_length) {
                            cnt_zero += 1;
                        } else {
                            qtype result_bit = input_quant.b[kc][padded_l] ^ layer->weights_b[co][kc][kl];
                            cnt_minus_one += bitCount(result_bit);
                        }
                    }
                }
                int cnt_one = (kernel_size - cnt_zero) * input_channel - cnt_minus_one;
                output[co][l] = (float)(cnt_one - cnt_minus_one);
            }
        }
        break;
    case TBN:
        for (int co = 0; co < output_channel; co++) {
            for (int l = 0; l < output_length; l++) {
                int cnt_minus_one = 0;
                int cnt_one = 0;
                int cnt_zero = 0;
                for (int kc = 0; kc < inputq_size; kc++) {
                    for (int kl = 0; kl < kernel_size; kl++) {
                        int padded_l = l * stride + kl * dilation - padding;
                        if (padded_l < 0 || padded_l >= input_length) {
                            cnt_zero += 1;
                        } else {
                            qtype weight = layer->weights_b[co][kc][kl];
                            qtype i_weight = ~weight;
                            qtype result_bit0 = (input_quant.t[kc][padded_l].bit_1 & i_weight) | (input_quant.t[kc][padded_l].bit_0 & weight);
                            qtype result_bit1 = (input_quant.t[kc][padded_l].bit_1 & weight) | (input_quant.t[kc][padded_l].bit_0 & i_weight);
                            cnt_minus_one += bitCount(result_bit0);
                            cnt_one += bitCount(result_bit1);
                        }
                    }
                }
                output[co][l] = (float)(cnt_one - cnt_minus_one);
            }
        }
        break;
    case TNN:
        for (int co = 0; co < output_channel; co++) {
            for (int l = 0; l < output_length; l++) {
                int cnt_minus_one = 0;
                int cnt_one = 0;
                int cnt_zero = 0;
                for (int kc = 0; kc < inputq_size; kc++) {
                    for (int kl = 0; kl < kernel_size; kl++) {
                        int padded_l = l * stride + kl * dilation - padding;
                        if (padded_l < 0 || padded_l >= input_length) {
                            cnt_zero += 1;
                        } else {
                            qtype result_bit0 = (input_quant.t[kc][padded_l].bit_1 & layer->weights_t0[co][kc][kl]) | (input_quant.t[kc][padded_l].bit_0 & layer->weights_t1[co][kc][kl]);
                            qtype result_bit1 = (input_quant.t[kc][padded_l].bit_1 & layer->weights_t1[co][kc][kl]) | (input_quant.t[kc][padded_l].bit_0 & layer->weights_t0[co][kc][kl]);
                            cnt_minus_one += bitCount(result_bit0);
                            cnt_one += bitCount(result_bit1);
                        }
                    }
                }
                output[co][l] = (float)(cnt_one - cnt_minus_one);
            }
        }
        break;
    case FP:
        for (int co = 0; co < output_channel; co++) {
            for (int l = 0; l < output_length; l++) {
                float sum = 0.0;
                for (int kc = 0; kc < input_channel; kc++) {
                    for (int kl = 0; kl < kernel_size; kl++) {
                        int il = l * layer->stride - layer->padding + kl * layer->dilation;
                        if (il >= 0 && il < input_length) {
                            sum += input[kc][il] * layer->weights_f[co][kc][kl];
                        }
                    }
                }
                output[co][l] = sum;
            }
        }
        break;
    }

    return output;
}

float ***max_pooling_2d(float ***input, int input_channels, int input_height, int input_width) {
    int output_height = input_height/2;
    int output_width = input_width/2;
    float ***output = allocate_3d_float_array(input_channels, output_height, output_width);
    for (int c = 0; c < input_channels; c++) {
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                float max_value = -FLT_MAX;
                for (int m = 0; m < 2; m++) {
                    for (int n = 0; n < 2; n++) {
                        int input_x = i * 2 + m;
                        int input_y = j * 2 + n;
                        if (input[c][input_x][input_y] > max_value) {
                            max_value = input[c][input_x][input_y];
                        }
                    }
                }
                output[c][i][j] = max_value;
            }
        }
    }
    // free_3d_float_array(input, input_channels, input_height);
    return output;
}