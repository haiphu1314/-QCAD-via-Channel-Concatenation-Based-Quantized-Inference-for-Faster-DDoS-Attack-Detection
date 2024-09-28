#ifndef MODEL_H
#define MODEL_H
#include "utils.h"
#include "linear.h"
#include "conv.h"

typedef enum {
    LINEAR,
    CONV
} layer_type;

typedef struct layer_node {
    layer_type layer_type;
    char layer_name[50];
    linear_layer* linear;
    conv2d_layer* conv;
    struct layer_node *next;
} layer_node;

// layer_node* create_layer(layer_type layer_type, void* layer);
layer_node* add_layer(layer_node *model, layer_type type, char* layer_name, void *layer);
void free_layer_nodes(layer_node* head);
// float* model_forward(layer_node* network, float* input);
// void load_weight_from_txt(layer_node *model, const char* filename);
void* get_layer(layer_node *model, char* layer_name);
// void model_summary(layer_node *head);

#endif // MODEL_H