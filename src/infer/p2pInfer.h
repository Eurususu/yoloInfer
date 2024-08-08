#pragma
#include "nv_infer_engine.h"
#include <string>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include <memory>
#include "preprocess.h"
#include "common.h"
#include "buffers.h"

namespace anktech
{
    class P2PInfer
    {
    public:
        P2PInfer() = default;
        ~P2PInfer();
        int initialize(const std::string &strOnnx, const std::string &calibCache,const std::string &dataDir, const int &calib_batch_size, const int &calib_num_images,
                    const std::string &strTrtModel, Prec_t precisionType,
                    const uint32_t &batchSize, const uint32_t &gpuID, bool end2end, bool V8);
        void preprocess(cv::Mat &src, std::vector<std::vector<float>> &scale, cudaStream_t &cudaStream);
        int Inference(std::shared_ptr<Scores> score, std::shared_ptr<Points> pts, const std::vector<std::vector<float>> &scale, cudaStream_t &cudaStream);
        void postprocess(std::shared_ptr<Scores> score,std::shared_ptr<Points> pts, const std::vector<std::vector<float>> &scale, cudaStream_t &cudaStream);
    private:
        // 构造函数
        std::shared_ptr<InferenceEngine> m_pTrtInferenceEngine{nullptr};
        // 初始化
        bool m_bInitialized{false};
        std::string m_onnxFile;
        std::string m_calibCache;
        std::string m_dataDir;
        int mCalib_batch_size; 
        int mCalib_num_images;
        std::string m_trtModel;
        Prec_t m_precisionType;
        uint32_t m_batchSize;
        uint32_t m_gpuID;
        bool m_end2end;
        bool m_V8;
        // 推理
        const std::string m_iname{"images"};
        const std::string m_oname1{"scores"};
        const std::string m_oname2{"points"};

        float *m_outputCPU1{nullptr};
        float *m_outputCPU2{nullptr};

        // 前处理
        uint32_t m_inputWidth;
        uint32_t m_inputHeight;
};

}