#ifndef CONV_GPU_H_
#define CONV_GPU_H_
#pragma once

#include <vector>
#include <chrono>
#include "../layer.h"
#include "./parallel/conv_forward.h"
struct ConvParams
{
    int dim_in;
    int dim_out;
    int channel_in;
    int height_in;
    int width_in;
    int height_kernel;
    int width_kernel;
    int stride;
    int pad_h;
    int pad_w;
    int height_out;
    int width_out;
    int channel_out;
};

class Conv_GPU : public Layer
{
private:
    // Convolution parameters
    ConvParams params;

    // weight and bias
    Matrix weight;      // weight param, size=channel_in*h_kernel*w_kernel*channel_out
    Vector bias;        // bias param, size = channel_out
    Matrix grad_weight; // gradient w.r.t weight
    Vector grad_bias;   // gradient w.r.t bias

    std::vector<Matrix> data_cols;

    // Custom by hhman
    GPU_Support gpu_support;

    void init();

public:
    Conv_GPU(int channel_in, int height_in, int width_in, int channel_out,
             int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,
             int pad_h = 0)
    {
        params.dim_in = channel_in * height_in * width_in;
        params.channel_in = channel_in;
        params.height_in = height_in;
        params.width_in = width_in;
        params.channel_out = channel_out;
        params.height_kernel = height_kernel;
        params.width_kernel = width_kernel;
        params.stride = stride;
        params.pad_w = pad_w;
        params.pad_h = pad_h;

        init();
    }

    void forward(const Matrix &bottom);
    void backward(const Matrix &bottom, const Matrix &grad_top);
    void update(Optimizer &opt);
    void im2col(const Vector &image, Matrix &data_col);
    void col2im(const Matrix &data_col, Vector &image);
    int output_dim() { return params.dim_out; }
    std::vector<float> get_parameters() const;
    std::vector<float> get_derivatives() const;
    void set_parameters(const std::vector<float> &param);
};
#endif