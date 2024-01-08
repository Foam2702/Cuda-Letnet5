#include "./conv_forward.h"
#include <cmath>
#include <iostream>

#define TILE_WIDTH 16

__device__ int calculateIndex(int batch_idx, int channel_idx, int row_idx, int col_idx, int height, int width)
{
    return (batch_idx * (channel_idx * height * width)) +
           (channel_idx * (height * width)) +
           (row_idx * width) +
           col_idx;
}

__global__ void conv_forward_kernel(float *output, const float *input, const float *kernel, const ConvParams params)
{
    const int height_out = params.height - params.kernel_size + 1;
    const int width_out = params.width - params.kernel_size + 1;

    int width_grid = ceil(1.0 * width_out / TILE_WIDTH);

    int batch_idx = blockIdx.x;                                         // batch number
    int output_feature_idx = blockIdx.y;                                // output feature
    int row_idx = (blockIdx.z / width_grid) * TILE_WIDTH + threadIdx.y; // row of the image matrix
    int col_idx = (blockIdx.z % width_grid) * TILE_WIDTH + threadIdx.x; // col of the image matrix

    float accumulator = 0.0f;

    if (row_idx < height_out && col_idx < width_out)
    {
        for (int input_channel_idx = 0; input_channel_idx < params.input_channel; input_channel_idx++) // sum over all input features
        {
            for (int kernel_row = 0; kernel_row < params.kernel_size; kernel_row++) // kernel_size x kernel_size filter
            {
                for (int kernel_col = 0; kernel_col < params.kernel_size; kernel_col++)
                {
                    int input_row = row_idx + kernel_row;
                    int input_col = col_idx + kernel_col;
                    int input_idx = calculateIndex(batch_idx, input_channel_idx, input_row, input_col, params.height, params.width);
                    int kernel_idx = calculateIndex(output_feature_idx, input_channel_idx, kernel_row, kernel_col, params.kernel_size, params.kernel_size);
                    accumulator += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
        int output_idx = calculateIndex(batch_idx, output_feature_idx, row_idx, col_idx, height_out, width_out);
        output[output_idx] = accumulator;
    }
}

void GPUInterface::conv_forward_gpu_full(float *output_data, const float *input_data, const float *weight_data, const ConvParams params)
{
    std::cout << ". Not Optimize:\n";
    const int height_out = params.height - params.kernel_size + 1;
    const int width_out = params.width - params.kernel_size + 1;

    // Allocate device memory
    float *device_input, *device_output, *device_weight;
    cudaMalloc((void **)&device_input, params.num_samples * params.input_channel * params.height * params.width * sizeof(float));                // input features map is input_channel
    cudaMalloc((void **)&device_output, params.num_samples * params.output_channel * height_out * width_out * sizeof(float));                    // output feature map is output_channel
    cudaMalloc((void **)&device_weight, params.output_channel * params.input_channel * params.kernel_size * params.kernel_size * sizeof(float)); // input_channel * output_channel filter Maps of size kernel_height * kernel_height

    // Copy input and mask data to device
    cudaMemcpy(device_input, input_data, params.num_samples * params.input_channel * params.height * params.width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight_data, params.output_channel * params.input_channel * params.kernel_size * params.kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Set the kernel dimensions and call the kernel
    int Z = ceil(1.0 * height_out / TILE_WIDTH) * ceil(1.0 * width_out / TILE_WIDTH);
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(params.num_samples, params.output_channel, Z);

    // Launch the kernel
    GpuTimer time_kernel;
    time_kernel.Start();
    conv_forward_kernel<<<num_blocks_in_grid, num_threads_per_block>>>(device_output, device_input, device_weight, params);
    time_kernel.Stop();
    float time_kernel_ms = time_kernel.Elapsed();
    std::cout << "\t - Kernel Time: " << time_kernel_ms << " ms" << std::endl;
    // Copy the output back to host
    cudaMemcpy(output_data, device_output, params.num_samples * params.output_channel * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_weight);
}