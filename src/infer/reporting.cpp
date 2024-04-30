#include "reporting.h"

namespace anktech{
void Profiler::reportLayerTime(const char* layerName, float timeMs) noexcept
{
    if (mIterator == mLayers.end())
    {
        const bool first = !mLayers.empty() && mLayers.begin()->name == layerName;
        mUpdatesCount += mLayers.empty() || first;
        if (first)
        {
            mIterator = mLayers.begin();
        }
        else
        {
            mLayers.emplace_back();
            mLayers.back().name = layerName;
            mIterator = mLayers.end() - 1;
        }
    }

    mIterator->timeMs += timeMs;
    ++mIterator;
}

void Profiler::print(std::ostream& os) const noexcept
{
    const std::string nameHdr("Layer");
    const std::string timeHdr("   Time (ms)");
    const std::string avgHdr("   Avg. Time (ms)");
    const std::string percentageHdr("   Time %");

    const float totalTimeMs = getTotalTime();

    const auto cmpLayer = [](const LayerProfile& a, const LayerProfile& b)
    {
        return a.name.size() < b.name.size();
    };
    const auto longestName = std::max_element(mLayers.begin(), mLayers.end(), cmpLayer);
    const auto nameLength = std::max(longestName->name.size() + 1, nameHdr.size());
    const auto timeLength = timeHdr.size();
    const auto avgLength = avgHdr.size();
    const auto percentageLength = percentageHdr.size();

    os << std::endl
       << "=== Profile (" << mUpdatesCount << " iterations ) ===" << std::endl
       << std::setw(nameLength) << nameHdr << timeHdr << avgHdr << percentageHdr << std::endl;

    for (const auto& p : mLayers)
    {
        // clang-format off
        os << std::setw(nameLength) << p.name << std::setw(timeLength) << std::fixed << std::setprecision(2) << p.timeMs
           << std::setw(avgLength) << std::fixed << std::setprecision(4) << p.timeMs / mUpdatesCount
           << std::setw(percentageLength) << std::fixed << std::setprecision(1) << p.timeMs / totalTimeMs * 100
           << std::endl;
    }
    {
        os << std::setw(nameLength) << "Total" << std::setw(timeLength) << std::fixed << std::setprecision(2)
           << totalTimeMs << std::setw(avgLength) << std::fixed << std::setprecision(4) << totalTimeMs / mUpdatesCount
           << std::setw(percentageLength) << std::fixed << std::setprecision(1) << 100.0 << std::endl;
        // clang-format on
    }
    os << std::endl;
}

void Profiler::exportJSONProfile(const std::string& fileName) const noexcept
{
    std::ofstream os(fileName, std::ofstream::trunc);
    os << "[" << std::endl << "  { \"count\" : " << mUpdatesCount << " }" << std::endl;

    const auto totalTimeMs = getTotalTime();

    for (const auto& l : mLayers)
    {
        // clang-format off
        os << ", {" << " \"name\" : \""      << l.name << "\""
                       ", \"timeMs\" : "     << l.timeMs
           <<          ", \"averageMs\" : "  << l.timeMs / mUpdatesCount
           <<          ", \"percentage\" : " << l.timeMs / totalTimeMs * 100
           << " }"  << std::endl;
        // clang-format on
    }
    os << "]" << std::endl;
}
}