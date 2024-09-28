/**
 * @ Author: Hai Phu
 * @ Email:  haiphu@hcmut.edu.vn
 * @ Create Time: 2024-06-27 16:23:39
 * @ Modified time: 2024-09-28 12:56:31
 * @ Description:
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "src/model.h"
// #include "testcase.h"
#include <time.h>
#include <nmmintrin.h>
#include <sys/time.h>
#define NUM_TESTCASES 84000
// #define NO_TESTS 1
#define NO_TESTS 10
#define NUM_CPU 24

int main()
{
    printf("TESTCASE\n");

    int input_height = 32;
    int input_width = 9;
    int input_channel = 1;

    layer_node *model = NULL;
    conv2d_layer *conv1 = create_conv2d_layer(input_channel, 64, 3, 1, 1, 1, BNN);
    model = add_layer(model, CONV, "conv1", conv1);
    conv2d_layer *conv2 = create_conv2d_layer(64, 64, 3, 1, 1, 1, BNN);
    model = add_layer(model, CONV, "conv2", conv2);
    linear_layer *linear1 = create_linear_layer(64 * (input_height / 2 / 2) * (input_width / 2 / 2), 128, BNN);
    model = add_layer(model, LINEAR, "linear1", linear1);
    linear_layer *linear2 = create_linear_layer(128, 2, BNN);
    model = add_layer(model, LINEAR, "linear2", linear2);

    FILE *file;
    int layer, inc, outc, kx, ky;
    int size;
    file = fopen("model_bnn.bin", "rb");
    if (file == NULL)
    {
        printf("Error opening WEIGHT file\n");
        return 1;
    }
    // LAYER CONV1 ##########################################################################################################################################################################################################################################################
    fread(&layer, sizeof(int), 1, file);
    fread(&outc, sizeof(int), 1, file);
    fread(&inc, sizeof(int), 1, file);
    fread(&kx, sizeof(int), 1, file);
    fread(&ky, sizeof(int), 1, file);
    fread(&conv1->input_thres, sizeof(float), 1, file);
    size = inc * outc * kx * ky;
    conv1->weights_b = (long *)malloc(size * sizeof(long));
    fread(conv1->weights_b, sizeof(long), size, file);
    // LAYER CONV2 ##########################################################################################################################################################################################################################################################
    fread(&layer, sizeof(int), 1, file);
    fread(&outc, sizeof(int), 1, file);
    fread(&inc, sizeof(int), 1, file);
    fread(&kx, sizeof(int), 1, file);
    fread(&ky, sizeof(int), 1, file);
    fread(&conv2->input_thres, sizeof(float), 1, file);
    size = inc * outc * kx * ky;
    fread(conv2->weights_b, sizeof(long), size, file);
    // LAYER LINEAR1 ##########################################################################################################################################################################################################################################################
    fread(&layer, sizeof(int), 1, file);
    fread(&outc, sizeof(int), 1, file);
    fread(&inc, sizeof(int), 1, file);
    fread(&linear1->input_thres, sizeof(float), 1, file);
    size = inc * outc;
    fread(linear1->weights_b, sizeof(long), size, file);
    // LAYER LINEAR2 ##########################################################################################################################################################################################################################################################
    fread(&layer, sizeof(int), 1, file);
    fread(&outc, sizeof(int), 1, file);
    fread(&inc, sizeof(int), 1, file);
    fread(&linear2->input_thres, sizeof(float), 1, file);
    size = inc * outc;
    fread(linear2->weights_b, sizeof(long), size, file);

    int num_tesecase;
    int num_correct = 0;
    file = fopen("testcase_bnn.bin", "rb");
    if (file == NULL)
    {
        printf("Error opening file\n");
        return 1;
    }
    fread(&num_tesecase, sizeof(int), 1, file);
    printf("\n BEGIN %d TESCASES FOR BNN MODEL ....\n", num_tesecase);
    for (int i = 0; i < num_tesecase; i++)
    {
        float *input = (float *)malloc(32 * 9 * sizeof(float));
        float *label = (float *)malloc(2 * sizeof(float));

        fread(input, sizeof(float), 32 * 9, file);
        fread(label, sizeof(float), 2, file);
        float *x1 = conv2d_forward(conv1, input, input_height, input_width);

        
        float *x1_mp = max_pooling_2d(x1, conv1->output_channel, input_height, input_width);
        float *x2 = conv2d_forward(conv2, x1_mp, input_height / 2, input_width / 2);
        float *x2_mp = max_pooling_2d(x2, conv2->output_channel, (input_height / 2), (input_width / 2));
        float *input_linear = flatto1d(x2_mp, 64, (input_height / 2) / 2, (input_width / 2) / 2);
        float *x3 = linear_forward(linear1, input_linear);
        float *x4 = linear_forward(linear2, x3);
        // printf("Predict: %f %f\n", x4[0], x4[1]);
        if (x4[0] == label[0] && x4[1] == label[1])
            num_correct += 1;
        free(x4);
    }
    printf("\nNumber of correct BNN test cases: %d out of %d (%.2f\%)\n\n", num_correct, num_tesecase, (float)num_correct/num_tesecase*100);


    return 0;
    // free_layer_nodes(model);
}
