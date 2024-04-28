#include <opencv2/opencv.hpp>

cv::Mat letterbox(cv::Mat &src, int width, int height)
{
  float scale = std::min(height / (float)src.rows, width / (float)src.cols);

  int offsetx = (width - src.cols * scale) / 2;
  int offsety = (height - src.rows * scale) / 2;

  cv::Point2f srcTri[3]; // 计算原图的三个点：左上角、右上角、左下角
  srcTri[0] = cv::Point2f(0.f, 0.f);
  srcTri[1] = cv::Point2f(src.cols - 1.f, 0.f);
  srcTri[2] = cv::Point2f(0.f, src.rows - 1.f);
  cv::Point2f dstTri[3]; // 计算目标图的三个点：左上角、右上角、左下角
  dstTri[0] = cv::Point2f(offsetx, offsety);
  dstTri[1] = cv::Point2f(src.cols * scale - 1.f + offsetx, offsety);
  dstTri[2] = cv::Point2f(offsetx, src.rows * scale - 1.f + offsety);
  cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);       // 计算仿射变换矩阵
  cv::Mat warp_dst = cv::Mat::zeros(height, width, src.type()); // 创建目标图
  cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());        // 进行仿射变换
  return warp_dst;
}

// Define a function to preprocess the image
cv::Mat preprocess_image(const std::string& image_path, int width, int height, const std::string& preprocessor = "keep_aspect_ratio_resizer") {
    // Load the image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        exit(EXIT_FAILURE);
    }
    // resize image
    cv::Mat warp_dst;
    if (preprocessor == "fixed_shape_resizer") {
        // Get image dimensions
        int img_width = image.cols;
        int img_height = image.rows;
        // Calculate scaling factors
        float width_scale = static_cast<float>(width) / img_width;
        float height_scale = static_cast<float>(height) / img_height;
        // Resize the image
        cv::resize(image, warp_dst, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    }
    else if (preprocessor == "keep_aspect_ratio_resizer")
    {
        warp_dst = letterbox(image, width, height);
    }
    else{
        std::cerr << "Preprocessing method not supported" << std::endl;
    }

    
    warp_dst.convertTo(warp_dst, CV_32FC3, 1.0 / 255.0); // normalization
    cv::cvtColor(warp_dst, warp_dst, cv::COLOR_BGR2RGB); // BGR2RGB
    // NHWC to NCHW：rgbrgbrgb to rrrgggbbb：
    std::vector<cv::Mat> warp_dst_nchw_channels;
    cv::split(warp_dst, warp_dst_nchw_channels); // 将输入图像分解成三个单通道图像：rrrrr、ggggg、bbbbb

    // Convert channels vector to CHW format
    cv::Mat chw_image;
    cv::merge(warp_dst_nchw_channels, chw_image);


    return chw_image;
}

