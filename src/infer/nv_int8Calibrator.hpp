#pragma once
#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>
#include "NvInferRuntimeCommon.h"
#include "ImageBatch.hpp"
#include <memory>
#include "cuda_utils.h"
namespace anktech{
class EngineCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    EngineCalibrator(const std::string& cache_file) : cache_file_(cache_file)
    {
        // Initialize other members if needed
    }

    int32_t getBatchSize() const noexcept override
    {
        if (image_batcher_)
        {
            return image_batcher_->getBatchSize();
        }
        return 1;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        if (!image_batcher_)
        {
            return false;
        }

        
        if (image_batcher_->getBatchIndex() < image_batcher_->getNumBatch()){
            auto batch = image_batcher_->getBatch();
            std::cout << "Calibrating image " << image_batcher_->getImageIndex() << " / " << image_batcher_->getNumImages() << std::endl;

            // Assuming batch is a vector<float> representing image data
            for (int i=0; i<batch.size(); i++){
                cudaMemcpy(batch_allocation_ + i*(image_batcher_->getOffset()), batch[i].data, image_batcher_->getImageSize(), cudaMemcpyHostToDevice);  
            };
            for (int j = 0; j < nbBindings; j++){
                if (!strcmp(names[j], kInputTensorName))
                {
                    bindings[j] = batch_allocation_;
                }
            }
            // bindings[0] = batch_allocation_;
            return true;
        }
        else{
        
            std::cout << "Finished calibration batches" << std::endl;
            return false;
        }
        
    }

    const void* readCalibrationCache(std::size_t& length) noexcept override
    {
        calibrationCache.clear();
        std::ifstream input(cache_file_, std::ios::binary);
        input >> std::noskipws;
        if (input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                      std::back_inserter(calibrationCache));
        }
        length = calibrationCache.size();

        return length ? calibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, std::size_t length) noexcept override
    {
        std::ofstream file(cache_file_, std::ios::binary);
        if (file)
        {
            file.write(reinterpret_cast<const char*>(cache), length);
            file.close();
            std::cout << "Writing calibration cache data to: " << cache_file_ << std::endl;
        }
    }

    void setImageBatcher(std::shared_ptr<ImageBatcher> imgBatcher)
    {
        image_batcher_ = imgBatcher;
        size_t size = imgBatcher->getSizeOfBatch();
        CUDA_CHECK(cudaMalloc((void **)&batch_allocation_, size));
    }

private:
    std::string cache_file_;
    std::vector<char> calibrationCache;
    std::shared_ptr<ImageBatcher> image_batcher_;
    float* batch_allocation_{nullptr};
};
}