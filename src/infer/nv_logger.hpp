
#ifndef _NV_LOGGER_H_
#define _NV_LOGGER_H_

#include "NvInfer.h"
#include <iostream>
#include <utils/log.hpp>

class TrtLogger : public nvinfer1::ILogger
{
public:
    explicit TrtLogger(Severity severity = Severity::kWARNING) : m_reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > m_reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            INFOF(msg);
            break;
        case Severity::kERROR:
            INFOF(msg);
            break;
        case Severity::kWARNING:
            INFOW(msg);
            break;
        case Severity::kVERBOSE:
            INFOS(msg);
            break;
        case Severity::kINFO:
            INFO(msg);
            break;
        default:
            INFO(msg);
            break;
        }
        // std::cerr << msg << std::endl;
    }
    Severity m_reportableSeverity;
};

#endif