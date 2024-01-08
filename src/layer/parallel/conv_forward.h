#ifndef SRC_LAYER_PARALLEL_CONV_FORWARD_H
#define SRC_LAYER_PARALLEL_CONV_FORWARD_H
#pragma once

#include "./gpu_support.h"
struct ConvParams
{
    int num_samples;
    int output_channel;
    int input_channel;
    int height;
    int width;
    int kernel_size;
};

class GPUInterface
{
public:
    void conv_forward_gpu_full(float *output_data, const float *input_data, const float *weight_data, const ConvParams params);
};

#endif