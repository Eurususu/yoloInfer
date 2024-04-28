#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"

namespace anktech{
void cuda_preprocess(uint8_t* src, int src_width, int src_height,
                     float* dst, int dst_width, int dst_height, cudaStream_t &cudaStream);

void cuda_preprocess_init(int max_image_size);

void cuda_preprocess_destroy();

}