/**
 * @ Author: Hai Phu
 * @ Email:  haiphu@hcmut.edu.vn
 * @ Create Time: 2024-06-27 14:22:38
 * @ Modified time: 2024-09-19 22:24:21
 * @ Description:
 */


#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include "utils.h"
#include "linear.h"
#include "conv.h"

/**
 * @brief Adds a layer to the model.
 *
 * This function creates a new layer node and adds it to the end of the model's layer list.
 * The layer can be either a linear layer or a convolutional layer.
 *
 * @param model Pointer to the head of the model's layer list.
 * @param type The type of the layer being added (LINEAR or CONV).
 * @param layer_name The name of the layer being added.
 * @param layer Pointer to the layer structure being added (either linear_layer or conv_layer).
 *
 * @return A pointer to the head of the updated model's layer list.
 */
layer_node* add_layer(layer_node *model, layer_type type, char* layer_name, void *layer) {
    layer_node *new_node = (layer_node*) malloc(sizeof(layer_node));
    new_node->layer_type = type;
    strcpy(new_node->layer_name, layer_name);
    switch (type)
    {
    case LINEAR:
        new_node->linear = (linear_layer*) layer;
        new_node->conv = NULL;
        break;

    case CONV:
        new_node->conv = (conv2d_layer*) layer;
        new_node->linear = NULL;
        break;
        
    default:
        break;
    }

    if (model == NULL) {
        return new_node;
    }
    layer_node *current = model;
    while (current->next != NULL) {
        current = current->next;
    }
    current->next = new_node;
    new_node->next = NULL;
    return model;
}

void free_layer_nodes(layer_node* head) {
    layer_node* temp;

    // Duyệt qua danh sách liên kết và giải phóng từng node
    while (head != NULL) {
        temp = head;  // Lưu node hiện tại
        // Nếu linear_layer được cấp phát động, giải phóng nó
        if (temp->linear != NULL) {
            free(temp->linear);
        }

        // Nếu conv2d_layer được cấp phát động, giải phóng nó
        if (temp->conv != NULL) {
            free(temp->conv);
        }

        // Tiếp theo chuyển tới node tiếp theo
        head = head->next;

        // Giải phóng node hiện tại
        free(temp);
    }
}

/**
 * @brief Retrieves a layer from the model by its name.
 *
 * This function searches through the model's layer list to find a layer with the specified name.
 * If the layer is found, it returns a pointer to the layer structure (either a linear_layer or conv_layer).
 * If the layer is not found, it prints an error message and terminates the program.
 *
 * @param model Pointer to the head of the model's layer list.
 * @param layer_name The name of the layer to retrieve.
 *
 * @return A pointer to the layer structure (either linear_layer or conv_layer). 
 */

void* get_layer(layer_node *model, char* layer_name){
    layer_node *current = model;
    while(current!=NULL){
        if(strcmp(current->layer_name, layer_name) == 0){
            if(current->conv != NULL){
                return current->conv;
            }
            else if (current->linear != NULL){
                return current->linear;
            }
        }
        current = current->next;
    }
    fprintf(stderr, "layer %s does not exist!\n",layer_name);
    exit(EXIT_FAILURE);       
}

/**
 * @brief Loads weights from a text file and assigns them to the corresponding layers in the model.
 *
 * This function reads weights from a specified text file and assigns them to the appropriate
 * layers in the model. The file should have a specific format for linear and convolutional layers,
 * including details such as layer name, input channels, output channels, quantization type, and
 * input threshold. If there is a mismatch between the file data and the model, an error is raised.
 *
 * @param model Pointer to the head of the model's layer list.
 * @param filename The name of the text file containing the weights.
 */
// void load_weight_from_txt(layer_node *model, const char* filename) {
//     FILE* file = fopen(filename, "r");
//     if (!file) {
//         perror("Error opening file");
//         exit(EXIT_FAILURE);
//     }
//     char line[MAX_CHARS_LINE];
//     while (fgets(line, sizeof(line), file)) {
//         if (strstr(line, "linear")) {
//             int input_channel;
//             int output_channel;
//             int quant_i;
//             quant_type quant;
//             char layer_name[20];
//             float thres;
//             fscanf(file, "layer_name: %s\n", layer_name);
//             fscanf(file, "input_channel: %d\n", &input_channel);
//             fscanf(file, "output_channel: %d\n", &output_channel);
//             fscanf(file, "quant_type: %d\n", &quant_i);
//             fscanf(file, "input_thres: %f\n", &thres);
//             switch(quant_i){
//                 case 0:
//                     quant = BNN;
//                     break;
//                 case 1:
//                     quant = TBN;
//                     break;
//                 case 2:
//                     quant = TNN;
//                     break;
//             }
//             int found_layer = 0; 
//             layer_node *current = model;
//             printf("load layer %s ... \n", layer_name);
//             while (current != NULL) {
//                 if(strcmp(current->layer_name, layer_name) == 0){
//                     found_layer = 1;                    
//                     if(input_channel != current->linear->input_channel){
//                         fprintf(stderr, "input channel of %s mismatch!\n",current->layer_name);
//                         exit(EXIT_FAILURE);                        
//                     }
//                     if(output_channel != current->linear->output_channel){
//                         fprintf(stderr, "output channel of %s mismatch!\n",current->layer_name);
//                         exit(EXIT_FAILURE);                        
//                     }
//                     if(quant != current->linear->quant){
//                         fprintf(stderr, "quantization method of %s mismatch!\n",current->layer_name);
//                         exit(EXIT_FAILURE);                        
//                     }
//                     // if(thres != current->linear->input_thres){
//                     //     fprintf(stderr, "input threshold of %s mismatch!\n",current->layer_name);
//                     //     exit(EXIT_FAILURE);                        
//                     // }                    
//                     break;
//                 }
//                 current = current->next;
                
//             }
//             if (!found_layer) {
//                 printf("Layer %s does not exist", layer_name);
//                 exit(EXIT_FAILURE);
//             }
//             current->linear->input_thres = thres;
//             int weight_size = (input_channel % SIZEQUANT) == 0 ? (input_channel/SIZEQUANT)*output_channel : (input_channel/SIZEQUANT+1)*output_channel;
//             if (quant == TNN){
//                 for (int i = 0; i < weight_size; ++i) {
//                     fscanf(file, "0x%lx\n", &current->linear->weights_t0[i]);
//                     fscanf(file, "0x%lx\n", &current->linear->weights_t1[i]);
//                 }
//             }
//             else if (quant == TBN || quant == BNN){
//                 for (int i = 0; i < weight_size; ++i) {
//                     fscanf(file, "0x%lx\n", &current->linear->weights_b[i]);
//                 }
//             }

//         }
//         else if (strstr(line, "conv")) {
//             char layer_name[20];
//             int input_channel;
//             int output_channel;
//             int kernel_size;
//             int stride;
//             int padding;
//             int dilation;
//             int quant_i;
//             float thres;
//             quant_type quant;
            
//             fscanf(file, "layer_name: %s\n", layer_name);
//             fscanf(file, "input_channel: %d\n", &input_channel);
//             fscanf(file, "output_channel: %d\n", &output_channel);
//             fscanf(file, "kernel_size: %d\n", &kernel_size);
//             fscanf(file, "stride: %d\n", &stride);
//             fscanf(file, "padding: %d\n", &padding);
//             fscanf(file, "dilation: %d\n", &dilation);
//             fscanf(file, "quant_type: %d\n", &quant_i);
//             fscanf(file, "input_thres: %f\n", &thres);
//             switch(quant_i){
//                 case 0:
//                     quant = BNN;
//                     break;
//                 case 1:
//                     quant = TBN;
//                     break;
//                 case 2:
//                     quant = TNN;
//                     break;
//             }
//             int found_layer = 0; 
//             layer_node *current = model;
//             printf("load layer %s ... \n", layer_name);
//             while (current != NULL) {
//                 if(strcmp(current->layer_name, layer_name) == 0){
//                     found_layer = 1;
//                     if(input_channel != current->conv->input_channel){
//                         fprintf(stderr, "Input channel of layer %s is %d, but in the .txt file, it is %d.\n",current->layer_name, current->conv->input_channel, input_channel);
//                         exit(EXIT_FAILURE);                        
//                     }
//                     if(output_channel != current->conv->output_channel){
//                         fprintf(stderr, "Output channel of layer %s is %d, but in the .txt file, it is %d.\n",current->layer_name, current->conv->output_channel, output_channel);
//                         exit(EXIT_FAILURE);                        
//                     }
//                     if(quant != current->conv->quant){
//                         fprintf(stderr, "quantization method of %s mismatch!\n",current->layer_name);
//                         exit(EXIT_FAILURE);                        
//                     }
//                     // if(thres != current->conv->input_thres){
//                     //     fprintf(stderr, "Input threshold of layer %s is %f, but in the .txt file, it is %f.\n",current->layer_name, current->conv->input_thres, thres);
//                     //     exit(EXIT_FAILURE);                        
//                     // }                    
//                     break;
//                 }
                
//                 current = current->next;
//             }
//             if (!found_layer) {
//                 printf("Layer %s does not exist", layer_name);
//                 exit(EXIT_FAILURE);
//             }
//             current->conv->input_thres = thres;
//             int weight_size = (input_channel % SIZEQUANT) == 0 ? (input_channel/SIZEQUANT) : (input_channel/SIZEQUANT+1);
//             int qb = (quant == TNN) ? 2 : 1;
//             for (int oc = 0; oc < output_channel; oc++) {
//                 for(int w = 0; w < weight_size; w++){
//                     for (int q = 0; q < qb; q++){
//                         for (int kh = 0; kh < kernel_size; kh++) {
//                             qtype values[10];
//                             switch (kernel_size){
//                                 case 1:
//                                     fscanf(file, "0x%lx\n", &values[0]);
//                                     break;
//                                 case 2:
//                                     fscanf(file, "0x%lx"", 0x%lx\n", &values[0], &values[1]);
//                                     break;
//                                 case 3:
//                                     fscanf(file, "0x%lx, 0x%lx, 0x%lx\n", &values[0], &values[1], &values[2]);
//                                     break;
//                                 case 4:
//                                     fscanf(file, "0x%lx, 0x%lx, 0x%lx, 0x%lx\n", &values[0], &values[1], &values[2], &values[3]);
//                                     break;
//                                 case 5:
//                                     fscanf(file, "0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx\n", &values[0], &values[1], &values[2], &values[3], &values[4]);
//                                     break;
//                                 case 6:
//                                     fscanf(file, "0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5]);
//                                     break;
//                                 case 7:
//                                     fscanf(file, "0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6]);
//                                     break;
//                                 case 8:
//                                     fscanf(file, "0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7]);
//                                     break;
//                                 case 9:
//                                     fscanf(file, "0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8]);
//                                     break;
//                                 case 10:
//                                     fscanf(file, "0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx, 0x%lx\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8], &values[9]);
//                                     break;
//                                 default:
//                                     printf("Unsupported kernel_size: %d\n", kernel_size);
//                                     break;
//                             }
//                             if(quant == TBN || quant == BNN){
//                                 for(int kw = 0; kw<kernel_size; kw++){
//                                     current->conv->weights_b[oc][w][kh][kw] = values[kw];
//                                 }
//                             }
//                             else if(quant == TNN){
//                                 if(q == 0){
//                                     for(int kw = 0; kw<kernel_size; kw++){
//                                         current->conv->weights_t0[oc][w][kh][kw] = values[kw];
//                                     }
//                                 }
//                                 else if(q == 1){
//                                     for(int kw = 0; kw<kernel_size; kw++){
//                                         current->conv->weights_t1[oc][w][kh][kw] = values[kw];
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }