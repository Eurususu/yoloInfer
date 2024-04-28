#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
namespace anktech{
class ImageBatcher
{
public:
    ImageBatcher(const std::string& input, const std::vector<int>& shape, nvinfer1::DataType dtype, int max_num_images = -1, 
    bool exact_batches = false, const std::string& preprocessor = "fixed_shape_resizer")
    {
        // Find images in the given input path
        input_ = std::filesystem::absolute(input).string();
        findImages(input_);

        // Handle Tensor Shape
        dtype_ = dtype;
        shape_ = shape;
        batch_size_ = shape[0];
        format_ = (shape[1] == 3) ? "NCHW" : "NHWC";
        height_ = (shape[1] == 3) ? shape[2] : shape[1];
        width_ = (shape[1] == 3) ? shape[3] : shape[2];

        // Adapt the number of images as needed
        if (max_num_images > 0 && max_num_images < images_.size())
        {
            num_images_ = max_num_images;
        }
        if (exact_batches)
        {
            num_images_ = batch_size_ * (num_images_ / batch_size_);
        }
        if (num_images_ < 1)
        {
            std::cerr << "Not enough images to create batches" << std::endl;
            exit(1);
        }

        // Subdivide the list of images into batches
        num_batches_ = 1 + ((num_images_ - 1) / batch_size_);
        for (int i = 0; i < num_batches_; ++i)
        {
            int start = i * batch_size_;
            int end = std::min(start + batch_size_, num_images_);
            batches_.emplace_back(images_.begin() + start, images_.begin() + end);
        }

        // Indices
        image_index_ = 0;
        batch_index_ = 0;

        preprocessor_ = preprocessor;
    }

    void findImages(const std::filesystem::path& path)
    {
        if (std::filesystem::is_directory(path))
        {
            for (const auto& entry : std::filesystem::directory_iterator(path))
            {
                if (entry.is_regular_file() && isImage(entry.path().extension().string()))
                {
                    images_.push_back(entry.path().string());
                }
            }
        }
        else if (std::filesystem::is_regular_file(path) && isImage(path))
        {
            images_.push_back(path);
        }
        else
        {
            std::cerr << "No valid images found in the given directory or file." << std::endl;
            exit(1);
        }

        num_images_ = images_.size();
        if (num_images_ < 1)
        {
            std::cerr << "No valid images found in the given directory or file." << std::endl;
            exit(1);
        }
    }

    bool isImage(const std::string& extension)
    {
        static const std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};
        return std::find(extensions.begin(), extensions.end(), extension) != extensions.end();
    }

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
    cv::Mat preprocess_image(const std::string& image_path) {
        // Load the image
        cv::Mat image = cv::imread(image_path);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // BGR2RGB  
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            exit(EXIT_FAILURE);
        }
        // resize image
        cv::Mat warp_dst;
        if (preprocessor_ == "fixed_shape_resizer") {
            // Get image dimensions
            int img_width = image.cols;
            int img_height = image.rows;
            // Calculate scaling factors
            float width_scale = static_cast<float>(width_) / img_width;
            float height_scale = static_cast<float>(height_) / img_height;
            // Resize the image
            cv::resize(image, warp_dst, cv::Size(width_, height_), 0, 0, cv::INTER_LINEAR);
        }
        else if (preprocessor_ == "keep_aspect_ratio_resizer")
        {
            warp_dst = letterbox(image, width_, height_);
        }
        else{
            std::cerr << "Preprocessing method not supported" << std::endl;
        }

              
        warp_dst.convertTo(warp_dst, CV_32FC3); // normalization
        
        cv::Mat warp_dst_nchw;
        // NHWC to NCHW：rgbrgbrgb to rrrgggbbb
        if (format_ == "NCHW"){
            std::vector<cv::Mat> warp_dst_nchw_channels;
            cv::split(warp_dst, warp_dst_nchw_channels); // 将输入图像分解成三个单通道图像：rrrrr、ggggg、bbbbb
            for (auto &img : warp_dst_nchw_channels) {
                img = img.reshape(1, 1);
            }
            cv::hconcat(warp_dst_nchw_channels, warp_dst_nchw);
        }
        else if(format_ == "NHWC"){
            warp_dst_nchw = warp_dst;
        }
        warp_dst_nchw /= 255.;

        return warp_dst_nchw;
    }
    

    std::vector<cv::Mat> getBatch()
    {
        if (batch_index_ >= num_batches_)
        {
            return {};
        }

        const auto& batch_images = batches_[batch_index_];
        ++batch_index_;
        
        

        std::vector<cv::Mat> batch_data;
        batch_data.reserve(batch_images.size());

        for (const auto& image_path : batch_images)
        {
            ++image_index_;
            auto image = preprocess_image(image_path);

            // Add the image's label to the batch
            batch_data.push_back(image);
        }

        return batch_data;
    }

    int32_t getBatchSize()
    {
        return batch_size_;
    }

    size_t getSizeOfBatch(){
        return batch_size_ * shape_[1] * shape_[2] * shape_[3] * sizeof(float);
    }

    size_t getImageSize(){
        return shape_[1] * shape_[2] * shape_[3] * sizeof(float);
    }

    size_t getOffset(){
        return shape_[1] * shape_[2] * shape_[3];
    }

    int getBatchIndex(){
        return batch_index_;
    }

    int getImageIndex(){
        return image_index_;
    }

    int getNumImages(){
        return num_images_;
    }

    int getNumBatch(){
        return num_batches_;
    }

private:
    std::string input_;
    std::vector<std::string> images_;
    int num_images_ = 0;

    std::vector<int> shape_;
    nvinfer1::DataType dtype_;
    int batch_size_;
    std::string format_;
    int height_;
    int width_;

    int num_batches_ = 0;
    std::vector<std::vector<std::string>> batches_;

    int image_index_ = 0;
    int batch_index_ = 0;

    std::string preprocessor_;
};
}