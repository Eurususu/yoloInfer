
#ifndef _NV_INFER_ENGINE_H_
#define _NV_INFER_ENGINE_H_

#include "NvInfer.h"
#include "nv_logger.hpp"
#include <yoloInferData_types.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include "common.h"
#include "buffers.h"

namespace anktech
{

class InferenceEngine
{
  public:
    typedef std::shared_ptr<InferenceEngine> ptr;
    // 禁止InferenceEngine类进行隐式转化
    explicit InferenceEngine();
    // 这些声明指示编译器不生成默认的拷贝构造函数和拷贝赋值运算符。通过将它们标记为 delete，
    //禁止了对象的拷贝和赋值操作。这可能是因为 InferenceEngine 类的设计不允许被拷贝或赋值，
    //通过基类指针用于管理资源。
    InferenceEngine(const InferenceEngine &) = delete; //禁用拷贝构造函数
    InferenceEngine(const InferenceEngine &&) = delete; //禁用移动构造函数
    InferenceEngine &operator=(const InferenceEngine &) = delete; // 禁用赋值
    InferenceEngine &operator=(const InferenceEngine &&) = delete; // 禁用移动赋值

    virtual ~InferenceEngine(); // 虚析构函数通常用于基类，以确保在删除基类指针时正确调用派生类的析构函数。这也暗示着 InferenceEngine 类可能会被用作基类，允许通过基类指针删除对象。

  public:
    int32_t initialize(const std::string &strOnnx, const std::string &calibCache, const std::string &dataDir, const int &calib_batch_size, const int &calib_num_images, const std::string &strTrtModel,
                       Prec_t precisionType, const uint32_t &batchSize, const uint32_t &gpuID = 0, bool end2end=false, bool V8=false);

    uint32_t getInputNum() const;
    uint32_t getOutputNum() const;

    uint32_t getInputBufferSize(const std::string &blobName) const;
    float *getInputBufferGpu(const std::string &blobName) const;

    uint32_t getOutputBufferSize(const std::string &blobName) const;
    float *getOutputBufferGpu(const std::string &blobName) const;

    uint32_t getBatchSize() const
    {
        return m_batchSize;
    }

    void getInputShape(const std::string &blobName, uint32_t &width, uint32_t &height);
    void getOutputShape(const std::string &blobName, uint32_t &channel, uint32_t &height, uint32_t &width);

    int32_t doInference(cudaStream_t &stream, int batchsize);

    



  protected:
    void allocateGpuMemory();
    void releaseGpuMemory();

    nvinfer1::IHostMemory *convertOnnxToTrtModel(const std::string &strOnnx, Prec_t precisionType, 
                                                 const uint32_t &batchSize, const std::string &calibCache, const std::string &dataDir,
                                                 const int &calib_batch_size, const int &calib_num_images,
                                                 bool end2end, bool V8);

    std::unique_ptr<nvinfer1::INetworkDefinition> addEfficientNMS(std::unique_ptr<nvinfer1::INetworkDefinition> network, 
    float conf_thres=0.45, float iou_thres=0.5, int max_det=100, bool V8=false);
    uint64_t getElementSize(nvinfer1::DataType type);
    uint64_t volume(const nvinfer1::Dims &d);

  protected:
    static bool ReadBuffer(const std::string strTrtModel, std::string &strValue);
    static void WriteBuffer(void *buffer, size_t size, const std::string &strTrtModel);

  protected:
    std::vector<char> weight_buffer;
    bool m_debugMode{false};
    bool m_dumped{false}; 
    uint32_t m_batchSize; // batch size
    uint32_t m_gpuID; // GPU ID
    bool m_trtEngineReady{false}; // 初始化完成标志位：创建runtime以及反序列化
    uint32_t m_iBlobConunt{0}; // 模型输入数量

    TrtLogger gLogger; // trt日志 TrtLogger类继承自nvinfer1::ILogger，实现虚函数log

    std::vector<void *> m_gpuBuffers; // Bindings组成的vector 这里是void*类型,主要用来接受cudaMalloc申请内存
    std::vector<uint32_t> m_gpuBufferSizes; // Bindings大小数值组成的vector，用于给Bindings分配GPU内存
    std::vector<std::string> m_inputNames; 

    std::unordered_map<std::string, uint32_t> m_iNameIdxMap; //模型输入的名称和索引的字典
    std::unordered_map<std::string, uint32_t> m_oNameIdxMap; //模型输出的名称和索引的字典

    std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> m_iShapes; //模型输入的名称和形状的字典
    nvinfer1::IExecutionContext* m_trtExecContext{nullptr}; // 推理线程执行上下文
    nvinfer1::ICudaEngine* m_trtEngine{nullptr}; // 反序列化后的结果
    nvinfer1::IRuntime* m_trtRunTime{nullptr}; // 根据trt创建的runtime引擎

    std::unordered_map<std::string, std::tuple<uint32_t, uint32_t, uint32_t>> m_oShapes; //模型输出的名称和形状的字典
    Prec_t m_curPrecisionType;
};
} // namespace anktech
#endif