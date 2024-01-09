#ifndef PARALLEL_CONV_FORWARD_H
#define PARALLEL_CONV_FORWARD_H
#pragma once
#include "./gpu_support.h"
class GPUInterface
{
public:
    void conv_forward_optimize(float *output_data, const float *input_data, const float *weight_data,
                               const int num_samples, const int output_channel, const int input_channel,
                               const int height, const int width, const int kernel_size);
};

#endif