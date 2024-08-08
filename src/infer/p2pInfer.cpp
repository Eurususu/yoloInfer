#include "p2pInfer.h"
#include <algorithm>

namespace anktech{

void P2PInfer::preprocess(cv::Mat &src, std::vector<std::vector<float>> &scale, cudaStream_t &cudaStream){
    cuda_preprocess(src.ptr(), src.cols, src.rows, (float *)m_pTrtInferenceEngine->getInputBufferGpu(m_iname), m_inputWidth, m_inputHeight, cudaStream);
    float temp1 = std::min(m_inputWidth * 1.0f /src.cols , m_inputHeight * 1.0f / src.rows);
    float temp2 = (m_inputWidth - src.cols*temp1) / 2;
    float temp3 = (m_inputHeight - src.rows*temp1) / 2;
    std::vector<float> temp;
    temp.push_back(temp1);
    temp.push_back(temp2);
    temp.push_back(temp3);
    scale.push_back(temp);
}

int P2PInfer::initialize(const std::string &strOnnx, const std::string &calibCache,const std::string &dataDir, const int &calib_batch_size, const int &calib_num_images,
                const std::string &strTrtModel, Prec_t precisionType,
                const uint32_t &batchSize, const uint32_t &gpuID, bool end2end, bool V8)
                {
                    if (m_pTrtInferenceEngine == nullptr){
                        m_pTrtInferenceEngine = std::make_shared<InferenceEngine>();
                    }
                    if (!m_bInitialized){
                        m_onnxFile = strOnnx;
                        m_calibCache = calibCache;
                        m_dataDir = dataDir;
                        mCalib_batch_size = calib_batch_size;
                        mCalib_num_images = calib_num_images;
                        m_trtModel = strTrtModel;
                        m_precisionType = precisionType;
                        m_batchSize = batchSize;
                        m_gpuID = gpuID;
                        m_end2end = end2end;
                        m_V8 = V8;
                        m_bInitialized = (OK == m_pTrtInferenceEngine->initialize(m_onnxFile, m_calibCache, m_dataDir, mCalib_batch_size, mCalib_num_images, m_trtModel, m_precisionType, m_batchSize, m_gpuID, m_end2end, m_V8));
                        if (!m_bInitialized){
                            return NOT_INITIALIZED;
                        }
                        uint32_t outputSize = m_pTrtInferenceEngine->getOutputBufferSize(m_oname1);
                        CUDA_CHECK(cudaMallocHost(&m_outputCPU1, outputSize));
                        outputSize = m_pTrtInferenceEngine->getOutputBufferSize(m_oname2);
                        CUDA_CHECK(cudaMallocHost(&m_outputCPU2, outputSize));

                        uint32_t nInputWidth;
                        uint32_t nInputHeight;
                        m_pTrtInferenceEngine->getInputShape(m_iname, nInputWidth, nInputHeight);
                        m_inputWidth = nInputWidth;
                        m_inputHeight = nInputHeight;
                    }
                    return OK;
                }

int P2PInfer::Inference(std::shared_ptr<Scores> score,std::shared_ptr<Points> pts,const std::vector<std::vector<float>> &scale, cudaStream_t &cudaStream){
    if (!m_pTrtInferenceEngine){
        return FAIL;
    }
    int batchSize = scale.size();
    if (m_pTrtInferenceEngine->doInference(cudaStream, batchSize)){
        return FAIL;
    }
    score->scores.clear();
    score->scores.resize(batchSize);
    pts->points.clear();
    pts->points.resize(batchSize);
    postprocess(score, pts, scale, cudaStream);
    return OK;
}

void P2PInfer::postprocess(std::shared_ptr<Scores> score,std::shared_ptr<Points> pts,const std::vector<std::vector<float>> &scale, cudaStream_t &cudaStream){
    auto pOutputGpu1 = m_pTrtInferenceEngine->getOutputBufferGpu(m_oname1);
    auto pOutputSize1 = m_pTrtInferenceEngine->getOutputBufferSize(m_oname1);
    CUDA_CHECK(cudaMemcpyAsync(m_outputCPU1, pOutputGpu1, pOutputSize1, cudaMemcpyDeviceToHost, cudaStream));
    auto pOutputGpu2 = m_pTrtInferenceEngine->getOutputBufferGpu(m_oname2);
    auto pOutputSize2 = m_pTrtInferenceEngine->getOutputBufferSize(m_oname2);
    CUDA_CHECK(cudaMemcpyAsync(m_outputCPU2, pOutputGpu2, pOutputSize2, cudaMemcpyDeviceToHost, cudaStream));
    CUDA_CHECK(cudaStreamSynchronize(cudaStream));
    float threshold = 0.5;
    int nPoints = 28672;
    int batchSize = scale.size();
    for (int b = 0; b < batchSize; ++b){
        std::vector<float> s = scale[b];
        for (size_t i = 0; i < nPoints; ++i){
            float sc = m_outputCPU1[b*nPoints + i];
            if (sc < threshold){
                continue;
            }
            else{
                float x = (m_outputCPU2[(b * nPoints + i) * 2] - s[1]) / s[0];
                float y = (m_outputCPU2[(b * nPoints + i) * 2 + 1] - s[2]) / s[0];
                pts->points[b].push_back(cv::Point2f(x, y));
            }
        }
}
}

P2PInfer::~P2PInfer(){
    if (m_outputCPU1){
        CUDA_CHECK(cudaFreeHost(m_outputCPU1));
    }

    if (m_outputCPU2){
        CUDA_CHECK(cudaFreeHost(m_outputCPU2));
    }
}

}