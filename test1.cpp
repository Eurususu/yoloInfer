#include "p2pInfer.h"
#include "nv_infer_engine.h"
#include <string>
#include "common.h"
#include "buffers.h"
#include "cuda_runtime_api.h"

using namespace anktech;

int main(){
    auto infer = std::make_shared<P2PInfer>();
    std::string onnx_path {"/home/jia/PycharmProjects/crowd count/CrowdCounting-P2PNet/p2p_net.onnx"};
    std::string trt_path {"/home/jia/PycharmProjects/crowd count/CrowdCounting-P2PNet/p2p_net_dy.engine"};
    std::string calibCache {""};
    std::string dataDir {""};
    int calib_batch_size {2}; 
    int calib_num_images {250};
    uint32_t batchSize {16};
    Prec_t prec = Prec_t::FP16;
    uint32_t gpuID {0};
    bool end2end {false};
    bool V8 {false};

    cudaStream_t cudaStream;
    CUDA_CHECK(cudaStreamCreate(&cudaStream));

    if(infer->initialize(onnx_path, calibCache,dataDir, calib_batch_size, calib_num_images, trt_path, prec, batchSize , gpuID, end2end, V8)){
        std::cout << "failed" << std::endl;
        return FAIL;
    }
    std::cout << "initialize success" << std::endl;
    std::string input_video_path = "/home/jia/anktechDrive/12 原始数据/虹桥火车站录像/2024-08-01/1003.mp4";
    auto cap = cv::VideoCapture(input_video_path);
    int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = int(cap.get(cv::CAP_PROP_FPS));
    cv::VideoWriter writer("/home/jia/project/trt_infer/output/record.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(width, height));
    cv::Mat frame;
    int frame_index{0};
    int img_size = width * height;
    cuda_preprocess_init(img_size);
    while (cap.isOpened())
    {
        std::vector<std::vector<float>> scale;
        std::shared_ptr<Scores> score {new Scores};
        std::shared_ptr<Points> pts {new Points};
        // 统计运行时间
        auto start = std::chrono::high_resolution_clock::now();
        cap >> frame;
        if (frame.empty())
        {
            // std::cout << "文件处理完毕" << std::endl;
            // break;
            continue;
        }
        frame_index++;
        infer->preprocess(frame, scale, cudaStream);
        infer->Inference(score, pts, scale, cudaStream);
        // 结束时间
        auto end = std::chrono::high_resolution_clock::now();
        // microseconds 微秒，milliseconds 毫秒，seconds 秒，1微妙=0.001毫秒 = 0.000001秒
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;
        auto time_str = std::to_string(elapsed) + "ms";
        auto fps = 1000.0f / elapsed;
        auto fps_str = std::to_string(fps) + "fps";
        cv::Mat outputFrame = frame.clone();
        for (const auto& point:pts->points[0]) {
            cv::circle(outputFrame, point, 5, cv::Scalar(0, 0, 255), -1); // 绘制红色圆点
        }
        auto count = std::to_string(pts->points[0].size());
        cv::putText(outputFrame, time_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(outputFrame, fps_str, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(outputFrame, count, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        writer << outputFrame;
        cv::imshow("output", outputFrame);
        cv::waitKey(1);
        std::cout << "完成第" << frame_index << "帧" << std::endl;



    }

}