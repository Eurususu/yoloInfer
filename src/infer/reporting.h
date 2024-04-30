#pragma once
#include "NvInfer.h"
#include "sampleOptions.h"
#include "sampleUtils.h"

namespace anktech{
struct LayerProfile
{
    std::string name;
    float timeMs{0};
};

class Profiler : public nvinfer1::IProfiler
{

public:
    void reportLayerTime(const char* layerName, float timeMs) noexcept override;

    void print(std::ostream& os) const noexcept;

    //!
    //! \brief Export a profile to JSON file
    //!
    void exportJSONProfile(const std::string& fileName) const noexcept;

private:
    float getTotalTime() const noexcept
    {
        const auto plusLayerTime = [](float accumulator, const LayerProfile& lp) { return accumulator + lp.timeMs; };
        return std::accumulate(mLayers.begin(), mLayers.end(), 0.0, plusLayerTime);
    }

    std::vector<LayerProfile> mLayers;
    std::vector<LayerProfile>::iterator mIterator{mLayers.begin()};
    int mUpdatesCount{0};
};
}