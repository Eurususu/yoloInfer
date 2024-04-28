#include "yoloInfer.h"
#include "nv_infer_engine.h"
#include <string>
#include "common.h"
#include "buffers.h"
#include "cuda_runtime_api.h"
using namespace anktech;
int main(){
    auto infer = std::make_shared<YoloInfer>();
    std::string onnx_path {"/home/jia/project/trt_infer/model/best_tiny_converted_fp16.onnx"};
    std::string trt_path {"/home/jia/project/trt_infer/model/best_tiny_converted_int8.trt"};
    std::string calibCache {"/home/jia/project/trt_infer/model/3classes.cache"};
    std::string dataDir {"/home/jia/project/trt_infer/data/val"};
    // std::string fileList {"/home/jia/project/trt_infer/data/fileList.txt"};
    int calib_batch_size {8}; 
    int calib_num_images {139};
    uint32_t batchSize {1};
    Prec_t prec = Prec_t::INT8;
    uint32_t gpuID {0};
    bool end2end {false};
    bool V8 {false};

    cudaStream_t cudaStream;
    CUDA_CHECK(cudaStreamCreate(&cudaStream));

    if(infer->initialize(onnx_path, calibCache,dataDir, calib_batch_size, calib_num_images, trt_path, prec, batchSize , gpuID, end2end, V8)){
        std::cout << "failed" << std::endl;
        return FAIL;
    }
    std::cout << "success" << std::endl;

    // std::string input_video_path = "/home/jia/project/trt_infer/videos/test.mp4";
    // auto cap = cv::VideoCapture(input_video_path);
    // int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    // int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    // int fps = int(cap.get(cv::CAP_PROP_FPS));
    // cv::VideoWriter writer("/home/jia/project/trt_infer/output/record.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(width, height));
    // cv::Mat frame;
    // int frame_index{0};
    // int img_size = width * height;
    // cuda_preprocess_init(img_size);
    // while (cap.isOpened())
    // {
    //     std::vector<std::vector<float>> scale;
    //     std::shared_ptr<Out_boxes> out_boxes {new Out_boxes};
    //     // 统计运行时间
    //     auto start = std::chrono::high_resolution_clock::now();
    //     cap >> frame;
    //     if (frame.empty())
    //     {
    //         std::cout << "文件处理完毕" << std::endl;
    //         break;
    //     }
    //     frame_index++;
    //     infer->preprocess(frame, scale, cudaStream);
    //     infer->Inference(out_boxes, scale, cudaStream);
    //     // 结束时间
    //     auto end = std::chrono::high_resolution_clock::now();
    //     // microseconds 微秒，milliseconds 毫秒，seconds 秒，1微妙=0.001毫秒 = 0.000001秒
    //     auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;
    //     auto time_str = std::to_string(elapsed) + "ms";
    //     auto fps = 1000.0f / elapsed;
    //     auto fps_str = std::to_string(fps) + "fps";
    //     std::cout << "完成第" << frame_index << "帧" << std::endl;

    //     for (auto& boxes : out_boxes->bboxes){
    //         for (auto &box : boxes){
    //             if (box.prob > 0.4 && box.label == 0)
    //             cv::rectangle(frame, {box.rect.x, box.rect.y, box.rect.width, box.rect.height}, {150, 150, 180}, 2);
    //             else if(box.prob > 0.4 && box.label == 1)
    //             cv::rectangle(frame, {box.rect.x, box.rect.y, box.rect.width, box.rect.height}, {150, 0, 180}, 2);
    //             else if(box.prob > 0.4 && box.label == 2)
    //             cv::rectangle(frame, {box.rect.x, box.rect.y, box.rect.width, box.rect.height}, {150, 250, 180}, 2);
    //             else
    //             continue;
    //         }
    //     }
    //     writer.write(frame);

    // }

}