/**
 * @ Author: Hai Phu
 * @ Email:  haiphu@hcmut.edu.vn
 * @ Create Time: 2024-06-27 16:23:39
 * @ Modified time: 2024-09-28 13:35:09
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

int power(int base, int exponent)
{
    int result = 1;
    for (int i = 0; i < exponent; i++)
    {
        result *= base;
    }
    return result;
}
float getRandomNumber()
{
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}
int main()
{
    printf("QCAD_S\n");

    // omp_set_num_threads(NUM_CPU);
    quant_type typ[4] = {FP, TNN, TBN, BNN};
    int len_typ = 4;
    double time_FP;
    for (int t = 0; t < len_typ; t++)
    // int t = 1;
    {
        int input_height = 32;
        int input_width = 9;
        int input_channel = 1;

        layer_node *model = NULL;
        conv2d_layer *conv1 = create_conv2d_layer(1, 32, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv1", conv1);
        conv2d_layer *conv2 = create_conv2d_layer(32, 32, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv2", conv2);


        linear_layer *linear1 = create_linear_layer(32 * (input_height/2/2) * (input_width/2/2), 128, typ[t]);
        model = add_layer(model, LINEAR, "linear1", linear1);
        linear_layer *linear2 = create_linear_layer(128, 2, typ[t]);
        model = add_layer(model, LINEAR, "linear2", linear2);

        srand(time(NULL));
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int ct = 0; ct < NO_TESTS; ct++)
        {
            float *input = (float*)malloc(input_channel * input_height * input_width * sizeof(float*));
            for (int c = 0; c < 1; c++)
            {
                for (int h = 0; h < input_height; h++)
                {
                    for (int w = 0; w < input_width; w++)
                    {
                        input[c*h*w] = power(-1, w);
                    }
                }
            }
            float *x1 = conv2d_forward(conv1, input, input_height, input_width);
            float *x1_mp = max_pooling_2d(x1, conv1->output_channel, input_height, input_width);
            float *x2 = conv2d_forward(conv2, x1_mp, input_height/2, input_width/2);
            float *x2_mp = max_pooling_2d(x2, conv2->output_channel, (input_height/2), (input_width/2));
            float *input_linear = flatto1d(x2_mp, 64, (input_height/2)/2, (input_height/2)/2);
            float *x3 = linear_forward(linear1, input_linear);
            float *x4 = linear_forward(linear2, x3);    
            free(x4);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        if(typ[t] == FP){
            time_FP = time_taken;
        }
        printf("Model execution time  %d: %.6f seconds, %.4f\n\n", t, time_taken/NO_TESTS, time_FP/time_taken);
        free_layer_nodes(model);

    }

}
