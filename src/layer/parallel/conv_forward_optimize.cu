#include "./conv_forward.h"
#include <cmath>
#include <iostream>

#define TILE_WIDTH 16

__constant__ float dc_filter[2400];

__global__ void conv_forward_kernel(float *output, const float *input, const int num_samples,
                                    const int output_channel, const int input_channel,
                                    const int height, const int width, const int kernel_size)
{
    const int height_out = height - kernel_size + 1;
    const int width_out = width - kernel_size + 1;

    int width_grid = ceil(1.0 * width_out / TILE_WIDTH);

    int batch_idx = blockIdx.x;                                         // batch number
    int output_feature_idx = blockIdx.y;                                // output feature
    int row_idx = (blockIdx.z / width_grid) * TILE_WIDTH + threadIdx.y; // row of the image matrix
    int col_idx = (blockIdx.z % width_grid) * TILE_WIDTH + threadIdx.x; // col of the image matrix

    float accumulator = 0.0f;

    if (row_idx < height_out && col_idx < width_out)
    {
        for (int input_channel_idx = 0; input_channel_idx < input_channel; input_channel_idx++) // sum over all input features
        {
            for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) // kernel_size x kernel_size filter
            {
                for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++)
                {
                    int input_row = row_idx + kernel_row;
                    int input_col = col_idx + kernel_col;

                    float input_base_idx = batch_idx * (input_channel * height * width) + input_channel_idx * (height * width);
                    float input_idx = input_base_idx + input_row * width + input_col;
                    float filter_base_idx = output_feature_idx * (input_channel * kernel_size * kernel_size) + input_channel_idx * (kernel_size * kernel_size);
                    float filter_idx = filter_base_idx + kernel_row * kernel_size + kernel_col;

                    accumulator += input[input_idx] * dc_filter[filter_idx];
                }
            }
        }
        if (row_idx < height_out && col_idx < width_out)
        {

            float output_batch_offset = batch_idx * output_channel * height_out * width_out;
            float output_channel_offset = output_feature_idx * height_out * width_out;
            float output_row_offset = row_idx * width_out;
            float output_index = output_batch_offset + output_channel_offset + output_row_offset + col_idx;

            output[output_index] = accumulator;
        }
    }
}

__host__ void GPUInterface::conv_forward_optimize(float *output_data, const float *input_data, const float *weight_data,
                                                  const int num_samples, const int output_channel, const int input_channel,
                                                  const int height_in, const int width_in, const int kernel_height)
{
    std::cout << ". Optimization 1 - Const memory:\n";
    const int height_out = height_in - kernel_height + 1;
    const int width_out = width_in - kernel_height + 1;

    // Allocate device memory
    float *device_input, *device_output;
    cudaMalloc((void **)&device_input, num_samples * input_channel * height_in * width_in * sizeof(float));     // input features map is input_channel
    cudaMalloc((void **)&device_output, num_samples * output_channel * height_out * width_out * sizeof(float)); // output feature map is output_channel

    // Copy input and mask data to device
    cudaMemcpy(device_input, input_data, num_samples * input_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dc_filter, weight_data, output_channel * input_channel * kernel_height * kernel_height * sizeof(float));

    // Set the kernel dimensions and call the kernel
    int Z = ceil(1.0 * height_out / TILE_WIDTH) * ceil(1.0 * width_out / TILE_WIDTH);
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(num_samples, output_channel, Z);

    // Launch the kernel
    GpuTimer time_kernel;
    time_kernel.Start();
    conv_forward_kernel<<<num_blocks_in_grid, num_threads_per_block>>>(device_output, device_input, num_samples, output_channel, input_channel, height_in, width_in, kernel_height);
    time_kernel.Stop();
    float time_kernel_ms = time_kernel.Elapsed();
    std::cout << "\t - Kernel Time: " << time_kernel_ms << " ms" << std::endl;

    // Copy the output back to host
    cudaMemcpy(output_data, device_output, num_samples * output_channel * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
}