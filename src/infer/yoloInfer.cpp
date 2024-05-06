#include <algorithm>
#include "yoloInfer.h"



namespace anktech{
YoloInfer::YoloInfer(){
    
};

YoloInfer::~YoloInfer(){
    if (m_outputCPU0){
        CUDA_CHECK(cudaFreeHost(m_outputCPU0));
    }

    if (m_outputCPU1){
        CUDA_CHECK(cudaFreeHost(m_outputCPU1));
    }

    if (m_outputCPU2){
        CUDA_CHECK(cudaFreeHost(m_outputCPU2));
    }

    if (m_outputCPU3){
        CUDA_CHECK(cudaFreeHost(m_outputCPU3));
    }
}


int YoloInfer::initialize(const std::string &strOnnx, const std::string &calibCache,
                    const std::string &dataDir, const int &calib_batch_size, const int &calib_num_images,
                    const std::string &strTrtModel, Prec_t precisionType,
                    const uint32_t &batchSize, const uint32_t &gpuID, bool end2end, bool V8){
    if (m_pTrtInferenceEngine == nullptr)
        m_pTrtInferenceEngine = std::make_shared<InferenceEngine>();
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

        uint32_t outputSize = m_pTrtInferenceEngine->getOutputBufferSize(m_oname0);
        CUDA_CHECK(cudaMallocHost(&m_outputCPU0, outputSize));
        outputSize = m_pTrtInferenceEngine->getOutputBufferSize(m_oname1);
        CUDA_CHECK(cudaMallocHost(&m_outputCPU1, outputSize));
        outputSize = m_pTrtInferenceEngine->getOutputBufferSize(m_oname2);
        CUDA_CHECK(cudaMallocHost(&m_outputCPU2, outputSize));
        outputSize = m_pTrtInferenceEngine->getOutputBufferSize(m_oname3);
        CUDA_CHECK(cudaMallocHost(&m_outputCPU3, outputSize));

        uint32_t nInputWidth;
        uint32_t nInputHeight;
        m_pTrtInferenceEngine->getInputShape(m_iname, nInputWidth, nInputHeight);
        m_inputWidth = nInputWidth;
        m_inputHeight = nInputHeight;

    }
    return OK;
}

void YoloInfer::preprocess(cv::Mat &src, std::vector<std::vector<float>> &scale, cudaStream_t &cudaStream){
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

int YoloInfer::Inference(std::shared_ptr<Out_boxes> out_boxes, const std::vector<std::vector<float>> &scale, cudaStream_t &cudaStream){
    if (!m_pTrtInferenceEngine){
        return FAIL;
    }
    int batchSize = scale.size();
    if (m_pTrtInferenceEngine->doInference(cudaStream, batchSize)){
        return FAIL;
    }
    out_boxes->bboxes.clear();
    out_boxes->bboxes.resize(batchSize);
    postprocess(out_boxes, scale, cudaStream);

    return OK;
}

void YoloInfer::postprocess(std::shared_ptr<Out_boxes> out_boxes, const std::vector<std::vector<float>> &scale, cudaStream_t &cudaStream){
    auto pOutputGpu = m_pTrtInferenceEngine->getOutputBufferGpu(m_oname0);
    auto pOutputSize = m_pTrtInferenceEngine->getOutputBufferSize(m_oname0);
    CUDA_CHECK(cudaMemcpyAsync(m_outputCPU0, pOutputGpu, pOutputSize, cudaMemcpyDeviceToHost, cudaStream));
    pOutputGpu = m_pTrtInferenceEngine->getOutputBufferGpu(m_oname1);
    pOutputSize = m_pTrtInferenceEngine->getOutputBufferSize(m_oname1);
    CUDA_CHECK(cudaMemcpyAsync(m_outputCPU1, pOutputGpu, pOutputSize, cudaMemcpyDeviceToHost, cudaStream));
    pOutputGpu = m_pTrtInferenceEngine->getOutputBufferGpu(m_oname2);
    pOutputSize = m_pTrtInferenceEngine->getOutputBufferSize(m_oname2);
    CUDA_CHECK(cudaMemcpyAsync(m_outputCPU2, pOutputGpu, pOutputSize, cudaMemcpyDeviceToHost, cudaStream));
    pOutputGpu = m_pTrtInferenceEngine->getOutputBufferGpu(m_oname3);
    pOutputSize = m_pTrtInferenceEngine->getOutputBufferSize(m_oname3);
    CUDA_CHECK(cudaMemcpyAsync(m_outputCPU3, pOutputGpu, pOutputSize, cudaMemcpyDeviceToHost, cudaStream));

    int batchSize = scale.size();
    for (int b = 0; b < batchSize; ++b){
        std::vector<float> s = scale[b];
        size_t numDets = m_outputCPU0[b];
        for (size_t i = 0; i < numDets; ++i){
            BBox box;
            box.rect.x = std::max(0.f, ((m_outputCPU1[b*100*4 + i*4 + 0]) - s[1])) / s[0];
            box.rect.y = std::max(0.f, ((m_outputCPU1[b*100*4 + i*4 + 1]) - s[2])) / s[0];
            float x2 = std::min(static_cast<float>(m_inputWidth), ((m_outputCPU1[b*100*4 + i*4 + 2]) - s[1])) / s[0];
            float y2 = std::min(static_cast<float>(m_inputHeight), ((m_outputCPU1[b*100*4 + i*4 + 3]) - s[2])) / s[0];
            box.rect.width = x2 - box.rect.x;
            box.rect.height = y2 - box.rect.y;
            box.prob = m_outputCPU2[b*100 + i];
            box.label = m_outputCPU3[b*100 + i];
            out_boxes->bboxes[b].push_back(box);
        }

    }
}

}