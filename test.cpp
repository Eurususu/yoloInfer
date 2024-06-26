#include "yoloInfer.h"
#include "nv_infer_engine.h"
#include <string>
#include "common.h"
#include "buffers.h"
#include "cuda_runtime_api.h"
#include "BYTETracker.h"

using namespace anktech;
int main(){
    auto infer = std::make_shared<YoloInfer>();
    auto tracker0 = std::make_shared<BYTETracker>();
    auto tracker1 = std::make_shared<BYTETracker>();
    auto tracker2 = std::make_shared<BYTETracker>();
    std::string onnx_path {"/home/jia/PycharmProjects/yolov8-pytorch_quantization/weights/yolov8s-basketball.onnx"};
    std::string trt_path {"/home/jia/PycharmProjects/yolov8-pytorch_quantization/weights/yolov8s-basketball.engine"};
    std::string calibCache {"/home/jia/project/trt_infer/model/football.cache"};
    std::string dataDir {"/home/jia/project/trt_infer/data/calibrate"};
    int calib_batch_size {2}; 
    int calib_num_images {250};
    uint32_t batchSize {8};
    Prec_t prec = Prec_t::INT8;
    uint32_t gpuID {0};
    bool end2end {true};
    bool V8 {true};

    cudaStream_t cudaStream;
    CUDA_CHECK(cudaStreamCreate(&cudaStream));

    if(infer->initialize(onnx_path, calibCache,dataDir, calib_batch_size, calib_num_images, trt_path, prec, batchSize , gpuID, end2end, V8)){
        std::cout << "failed" << std::endl;
        return FAIL;
    }
    // std::cout << "success" << std::endl;

    std::string input_video_path = "/home/jia/PycharmProjects/data/1213.mp4";
    auto cap = cv::VideoCapture(input_video_path);
    int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = int(cap.get(cv::CAP_PROP_FPS));
    cv::VideoWriter writer("/home/jia/project/trt_infer/output/record.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(width, height));
    cv::Mat frame;
    int frame_index{0};
    int img_size = width * height;
    cuda_preprocess_init(img_size);
    while (cap.isOpened())
    {
        std::vector<std::vector<float>> scale;
        std::shared_ptr<Out_boxes> out_boxes {new Out_boxes};
        // 统计运行时间
        auto start = std::chrono::high_resolution_clock::now();
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "文件处理完毕" << std::endl;
            break;
        }
        frame_index++;
        infer->preprocess(frame, scale, cudaStream);
        infer->Inference(out_boxes, scale, cudaStream);
        // 结束时间
        auto end = std::chrono::high_resolution_clock::now();
        // microseconds 微秒，milliseconds 毫秒，seconds 秒，1微妙=0.001毫秒 = 0.000001秒
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;
        auto time_str = std::to_string(elapsed) + "ms";
        auto fps = 1000.0f / elapsed;
        auto fps_str = std::to_string(fps) + "fps";
        std::cout << "完成第" << frame_index << "帧" << std::endl;

    //     // 检测加跟踪
    //     for (auto &batch: out_boxes->bboxes){
    //         std::vector<BBox> bboxes0;
    //         std::vector<BBox> bboxes1;
    //         std::vector<BBox> bboxes2;
    //         for (auto& box : batch){
    //             if (box.label == 0){
    //                 bboxes0.push_back(box);
    //             }
    //             else if(box.label == 1){
    //                 bboxes1.push_back(box);
    //             }
    //             else if(box.label == 2){
    //                 bboxes2.push_back(box);
    //             }
    //         };
    //         auto out_boxes0 = tracker0->update(bboxes0);
    //         for (auto &box0 : out_boxes0){
    //             cv::rectangle(frame, cv::Rect(box0.tlwh[0], box0.tlwh[1], box0.tlwh[2], box0.tlwh[3]), cv::Scalar(150, 150, 180), 2);
    //             cv::putText(frame, format("%d", box0.track_id), Point(box0.tlwh[0], box0.tlwh[1] - 5),0, 0.6, cv::Scalar(150, 150, 180), 2, cv::LINE_AA);
                
    //         }
    //         auto out_boxes1 = tracker1->update(bboxes1);
    //         for (auto &box1 : out_boxes1){
    //             cv::rectangle(frame, cv::Rect(box1.tlwh[0], box1.tlwh[1], box1.tlwh[2], box1.tlwh[3]), cv::Scalar(150, 0, 180), 2);
    //             cv::putText(frame, format("%d", box1.track_id), Point(box1.tlwh[0], box1.tlwh[1] - 5),0, 0.6, cv::Scalar(150, 0, 180), 2, cv::LINE_AA);
    //         }
    //         auto out_boxes2 = tracker2->update(bboxes2);
    //         for (auto &box2 : out_boxes2){
    //             cv::rectangle(frame, cv::Rect(box2.tlwh[0], box2.tlwh[1], box2.tlwh[2], box2.tlwh[3]), cv::Scalar(150, 250, 180), 2);
    //             cv::putText(frame, format("%d", box2.track_id), Point(box2.tlwh[0], box2.tlwh[1] - 5),0, 0.6, cv::Scalar(150, 250, 180), 2, cv::LINE_AA);
    //         }
    //     }
    //     writer.write(frame);

        //只是检测
        for (auto& boxes : out_boxes->bboxes){
            for (auto &box : boxes){
                if (box.prob > 0.4 && box.label == 0)
                cv::rectangle(frame, {box.rect.x, box.rect.y, box.rect.width, box.rect.height}, {150, 150, 180}, 2);
                else if(box.prob > 0.4 && box.label == 1)
                cv::rectangle(frame, {box.rect.x, box.rect.y, box.rect.width, box.rect.height}, {150, 0, 180}, 2);
                else if(box.prob > 0.4 && box.label == 2)
                cv::rectangle(frame, {box.rect.x, box.rect.y, box.rect.width, box.rect.height}, {150, 250, 180}, 2);
                else
                continue;
            }
        }
        writer.write(frame);

    }

}