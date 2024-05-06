#pragma once
#include <opencv2/opencv.hpp>

namespace anktech{
enum Prec_t
{
    FP32 = 0,
    FP16 = 1,
    INT8 = 2
};

enum Error_t
{
    OK = 0,
    FAIL = -1,
    TIME_OUT = -2,
    INVALID_PARAM = -3,
    NOT_INITIALIZED = -4
};

struct BBox
{
    float prob;
    int label;
    cv::Rect_<float> rect; // lf x, lf y, w, h
};

struct Out_boxes
{
    std::vector<std::vector<BBox>> bboxes;
};

const static char* kInputTensorName = "images";

}