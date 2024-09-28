/**
 * @ Author: Hai Phu
 * @ Email:  haiphu@hcmut.edu.vn
 * @ Create Time: 2024-06-30 01:36:42
 * @ Modified time: 2024-09-28 12:50:42
 * @ Description:
 */

#include "utils.h"
#include "linear.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
/**
 * create_linear_layer
 * @brief Creates and initializes a linear layer with specified input and output channels, and quantization type.
 *
 * This function allocates memory for a linear layer structure and initializes its parameters based on
 * the given input channels, output channels, and quantization type. Depending on the quantization type,
 * it allocates memory for the appropriate weights.
 *
 * @param input_channel The number of input channels for the linear layer.
 * @param output_channel The number of output channels for the linear layer.
 * @param quant The quantization type for the linear layer. Possible values include:
 *              - BNN: Binary Neural Network
 *              - TBN: Ternary Binary Neural Network
 *              - TNN: Ternary Neural Network
 *
 * @return A pointer to the initialized linear_layer structure.
 */

linear_layer *create_linear_layer(int input_channel, int output_channel, quant_type quant)
{
    linear_layer *layer = (linear_layer *)malloc(sizeof(linear_layer));
    layer->input_channel = input_channel;
    layer->output_channel = output_channel;
    layer->quant = quant;
    layer->input_thres = 0.0;

    int weight_size = (input_channel % SIZEQUANT) == 0 ? (input_channel / SIZEQUANT) * output_channel : (input_channel / SIZEQUANT + 1) * output_channel;
    switch (quant)
    {
    case BNN:
    case TBN:
        layer->weights_b = (qtype *)malloc(weight_size * sizeof(qtype));
        if (layer->weights_b == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights\n");
            exit(1);
        }
        break;
    case TNN:
        layer->weights_t0 = (qtype *)malloc(weight_size * sizeof(qtype));
        if (layer->weights_t0 == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights_0\n");
            exit(1);
        }
        layer->weights_t1 = (qtype *)malloc(weight_size * sizeof(qtype));
        if (layer->weights_t1 == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights_1\n");
            exit(1);
        }
        break;
    case FP:
        // printf("%d \n", input_channel * output_channel);
        layer->weights_f = (float *)malloc(input_channel * output_channel * sizeof(float));
        if (layer->weights_f == NULL)
        {
            fprintf(stderr, "Memory allocation failed for weights\n");
            exit(1);
        }
        break;

    default:
        fprintf(stderr, "create_linear_layer: Unknown quantization type \n");
        exit(1);
    }
    return layer;
}

/**
 * @brief Performs the forward pass for a linear layer with quantized inputs.
 *
 * This function computes the output of a linear layer given the input data and the layer parameters.
 * It handles different quantization types (BNN, TBN, TNN) by quantizing the input data appropriately
 * and then performing the forward pass computation.
 *
 * @param layer Pointer to the linear_layer structure containing the layer parameters.
 * @param input Pointer to the input data array.
 *
 * @return A pointer to the output data array.
 */
float *linear_forward(linear_layer *layer, float *input)
{
    int input_channel = layer->input_channel;
    int output_channel = layer->output_channel;
    float input_thres = layer->input_thres;
    quant_type quant = layer->quant;
    
    float *output = (float *)malloc(output_channel * sizeof(float));
    int inputq_size = (input_channel % SIZEQUANT) ? (input_channel / SIZEQUANT + 1) : (input_channel / SIZEQUANT);
    linear_input input_quant[inputq_size];
    switch (quant)
    {
    case BNN:
       for(int i = 0; i < inputq_size; i++) {
            input_quant[i].b = 0;
        }    
        for (int k = 0; k < input_channel; k++)
        {
            if (input[k] < input_thres)
            {
                input_quant[k / SIZEQUANT].b |= (qtype)1 << (k % SIZEQUANT);
            }
        }      
        break;
    case TBN:
    case TNN:
       for(int i = 0; i < inputq_size; i++) {
            input_quant[i].t.bit_0 = 0;
            input_quant[i].t.bit_1 = 0;
        }        
        for (int k = 0; k < input_channel; k++)
        {
            if (input[k] > input_thres)
            {
                input_quant[k / SIZEQUANT].t.bit_1 |= (qtype)1 << (k % SIZEQUANT);
            }
            else if (input[k] < -input_thres)
            {
                input_quant[k / SIZEQUANT].t.bit_0 |= (qtype)1 << (k % SIZEQUANT);
            }
        }
        break;
    case FP:
        break;
    default:
        fprintf(stderr, "linear_forward: Unknown quantization type\n");
        exit(1);
    }
    
    switch (quant)
    {
    case BNN:
        for (int i = 0; i < output_channel; ++i)
        {
            int cnt_minus_one = 0;
            for (int j = 0; j < inputq_size; ++j)
            {
                qtype result_bit = (qtype)(input_quant[j].b) ^ (qtype)(layer->weights_b[i * inputq_size + j]);
                cnt_minus_one += bitCount(result_bit);
            }
            int cnt_one = input_channel - cnt_minus_one;
            output[i] = (float)(cnt_one - cnt_minus_one);
        }
        // printf("%f %f %f\n", output[0], output[1], output[2]);
        break;
    case TBN:
    
        for (int i = 0; i < output_channel; ++i)
        {
            int cnt_minus_one = 0;
            int cnt_one = 0;
            for (int j = 0; j < inputq_size; ++j)
            {
                qtype weight = layer->weights_b[i * inputq_size + j];
                qtype i_weight = ~weight;
                ttype input_t = input_quant[j].t;
                qtype bit_1 = input_quant[j].t.bit_1;
                qtype bit_0 = input_quant[j].t.bit_0;

                qtype result_bit1 = (bit_1 & i_weight) | (bit_0 & weight);
                qtype result_bit0 = (bit_1 & weight) | (bit_0 & i_weight);
                cnt_minus_one += bitCount(result_bit0);
                cnt_one += bitCount(result_bit1);
            }
            output[i] = (float)(cnt_one - cnt_minus_one);
        }
        break;
    case TNN:
        for (int i = 0; i < output_channel; ++i)
        {
            int cnt_minus_one = 0;
            int cnt_one = 0;
            for (int j = 0; j < inputq_size; ++j)
            {
                ttype input_t = input_quant[j].t;
                qtype bit_1 = input_quant[j].t.bit_1;
                qtype bit_0 = input_quant[j].t.bit_0;
                qtype weights_t1 = layer->weights_t1[i * inputq_size + j];
                qtype weights_t0 = layer->weights_t0[i * inputq_size + j];
                
                qtype result_bit1 = (bit_1 & weights_t1) | (bit_0 & weights_t0);
                qtype result_bit0 = (bit_1 & weights_t0) | (bit_0 & weights_t1);
                
                cnt_minus_one += bitCount(result_bit0);
                cnt_one += bitCount(result_bit1);
            }
            output[i] = (float)(cnt_one - cnt_minus_one);
        }
        break;
    case FP:
    
        for (int i = 0; i < output_channel; ++i)
        {
            float sum = 0.0;
            
            for (int j = 0; j < input_channel; ++j)
            {
                sum += input[j] * layer->weights_f[i * input_channel + j];
            }
            output[i] = sum;
        }
        break;
    }
    free(input);
    return output;
}
