
#include "nv_infer_engine.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"
#include "cuda_runtime.h"
#include <utils/log.hpp>
#include <fstream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <cassert>
#include "nv_int8Calibrator.hpp"
#include "ImageBatch.hpp"
#include "reporting.h"

namespace anktech
{

InferenceEngine::InferenceEngine()
{
    m_trtEngineReady = false;
}

InferenceEngine::~InferenceEngine()
{
    releaseGpuMemory();
    if (m_trtExecContext)
    {
        delete m_trtExecContext;
        m_trtExecContext = nullptr;
    }

    if (m_trtEngine)
    {
        delete m_trtEngine;
        m_trtEngine = nullptr;
    }

    if (m_trtRunTime)
    {
        delete m_trtRunTime;
        m_trtRunTime = nullptr;
    }
}

nvinfer1::IHostMemory *InferenceEngine::convertOnnxToTrtModel(const std::string &strOnnx, Prec_t precisionType,
                                                              const uint32_t &batchSize, const std::string &calibCache, 
                                                              const std::string &dataDir, const int &calib_batch_size, const int &calib_num_images,
                                                              bool end2end, bool V8)
{
    CHECK_CUDA_ERROR(cudaSetDevice(m_gpuID));

    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING); // 转枚举类为int型，Severity本身为int型枚举类

    // 创建构建器 builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger)); 
    if (!builder)
        return nullptr;
    // 1这个数字左移kEXPLICIT_BATCH位，默认kEXPLICIT_BATCH=0，所以explicitBatch默认为1
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 创建网络定义
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
        return nullptr;
    // 创建网络配置
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
        return nullptr;

    // 创建onnx解析器
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
        return nullptr;
    std::ifstream f(strOnnx, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    f.close();
    std::string buff = ss.str();
    // std::string key("anktechSH");
    // int i = 0;
    // for (auto &v : buff)
    // {
    //     v ^= key[i % 9];
    //     i++;
    // }
    // 解析onnx,返回是否解析成功
    auto parsed = parser->parse(buff.c_str(), buff.size());

    if (!parsed)
        return nullptr;
    // 创建 托管在主机内存中的数据的指针
    nvinfer1::IHostMemory *trtModelStream = nullptr;
    // 获取输入层
    auto inputLayer = network->getInput(0);
    if (!inputLayer)
    {
        INFOE("[stitcher] No input layer found.\n");
        return nullptr;
    }
    std::string name(inputLayer->getName());
    auto dim = inputLayer->getDimensions();

    // Set up optimization profile
    if (dim.d[0] == -1)
    {   //创建可选配置profile
        nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
        dim.d[0] = 1;
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, dim);
        dim.d[0] = batchSize;
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, dim);
        dim.d[0] = batchSize;
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, dim);
        config->addOptimizationProfile(profile);
    }

    // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, (std::size_t)1 << 33);
    config->setMaxWorkspaceSize((uint64_t)1 << 33);
    // 添加efficient nms
    if (end2end) network = addEfficientNMS(std::move(network),0.25, 0.65, 100, V8);
    if (precisionType == FP16 && builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else if (precisionType == INT8 && builder->platformHasFastInt8())
    {
        if (builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        gLogger.log(nvinfer1::ILogger::Severity::kWARNING, "building INT8 engine.");
        // 检查模型是否包含Q/DQ节点
        bool hasQDQ = false;
        for (int i = 0; i < network->getNbLayers(); ++i)
        {
            auto layer = network->getLayer(i);
            if (layer->getType() == nvinfer1::LayerType::kQUANTIZE || layer->getType() == nvinfer1::LayerType::kDEQUANTIZE)
            {
                hasQDQ = true;
                break;
            }
        }

        if (hasQDQ)
        {
            gLogger.log(nvinfer1::ILogger::Severity::kINFO, "Detected Q/DQ nodes in the model, no calibrator needed.");
        }
        else
        {
        auto calibrator = new EngineCalibrator(calibCache);
        config->setInt8Calibrator(calibrator);
        if (!std::filesystem::exists(calibCache)){
            std::vector<int> calib_shape = {calib_batch_size};
            calib_shape.push_back(inputLayer->getDimensions().d[1]);
            calib_shape.push_back(inputLayer->getDimensions().d[2]);
            calib_shape.push_back(inputLayer->getDimensions().d[3]);
            // std::string calib_dtype = inputs_dtype;  // Convert dtype if necessary
            auto imgBatcher = std::make_shared<ImageBatcher>(dataDir, calib_shape, nvinfer1::DataType::kFLOAT, calib_num_images,true);
            calibrator->setImageBatcher(imgBatcher);
        }    
        }
    }
    trtModelStream = builder->buildSerializedNetwork(*network, *config);
    return trtModelStream;
}

int32_t InferenceEngine::initialize(const std::string &strOnnx, const std::string &calibCache,const std::string &dataDir,
                                    const int &calib_batch_size, const int &calib_num_images,
                                    const std::string &strTrtModel, Prec_t precisionType,
                                    const uint32_t &batchSize, const uint32_t &gpuID, bool end2end, bool V8)
{

    if (m_trtEngineReady) // m_trtEngineReady 初始化完成标志位
        return true;

    m_batchSize = batchSize;
    m_gpuID = gpuID;

    m_curPrecisionType = precisionType; //设置精度类型 fp32 fp16 int8

    CHECK_CUDA_ERROR(cudaSetDevice(m_gpuID));

    initLibNvInferPlugins(&gLogger, ""); // initLibNvInferPlugins "NvInferPlugin.h"中初始化trt日志

    std::string buffer;
    if (ReadBuffer(strTrtModel, buffer)) // 读取trt模型到buffer
    {
        if (buffer.size()) // 如果buffer不为空,则进行创建runtime和反序列化
        {
            m_trtRunTime = nvinfer1::createInferRuntime(gLogger);
            m_trtEngine = m_trtRunTime->deserializeCudaEngine(buffer.data(), buffer.size());
        }
    }
    else //如果没有trt文件，则根据onnx创建
    {
        std::string w = "Cannot find valid trt file \"" + strTrtModel + "\". Now converting from \"" + strOnnx +
                        "\". This will take a while...";
        gLogger.log(nvinfer1::ILogger::Severity::kWARNING, w.c_str());
        nvinfer1::IHostMemory *trtModelStream = convertOnnxToTrtModel(strOnnx, precisionType, batchSize, calibCache, dataDir, calib_batch_size, calib_num_images, end2end, V8);

        if (trtModelStream == nullptr)
        {
            INFOE("[stitcher] failed to convert onnx to trt model.\n");
            return FAIL;
        }

        // Save TRT Engine
        WriteBuffer(trtModelStream->data(), trtModelStream->size(), strTrtModel);

        m_trtRunTime = nvinfer1::createInferRuntime(gLogger);
        if (m_trtRunTime == nullptr)
        {
            INFOE("[stitcher] failed to create inference runtime.\n");
            return FAIL;
        }

        m_trtEngine = m_trtRunTime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size());
        if (m_trtEngine == nullptr)
        {
            INFOE("[stitcher] failed to deserialize tensorrt engine.\n");
            return FAIL;
        }

        delete trtModelStream;
    }

    if (m_trtEngine)
    {
        allocateGpuMemory();
        // m_DeviceBuffer = std::make_shared<samplesCommon::BufferManager>(m_trtEngine); // 初始化device数据缓存
        m_trtEngineReady = true;
        return OK;
    }

    m_trtEngineReady = false;
    return FAIL;
}

void InferenceEngine::allocateGpuMemory()
{
    CHECK_CUDA_ERROR(cudaSetDevice(m_gpuID));
    m_iBlobConunt = 0;

    if (m_trtExecContext == nullptr)
        m_trtExecContext = m_trtEngine->createExecutionContext();

    int nbBindings = m_trtEngine->getNbBindings(); // 获取输入输出的数量
    m_gpuBuffers.resize(nbBindings); // m_gpuBuffers 存储输入输出数据
    m_gpuBufferSizes.resize(nbBindings); // m_gpuBufferSizes 存储输入输出数据大小

    for (int i = 0; i < nbBindings; i++)
    {
        nvinfer1::Dims dims = m_trtEngine->getBindingDimensions(i);
        // 如果是动态batch的话，通过外部传入参数设置batchsize
        if (dims.d[0] == -1){
            dims.d[0] = m_batchSize;
        }
        // dims.d[0] = m_batchSize; // 设置输入输出的batch大小
        nvinfer1::DataType dataType = m_trtEngine->getBindingDataType(i);
        auto pName = m_trtEngine->getBindingName(i);

        //计算buffer大小
        int64_t bufferSize = volume(dims) * getElementSize(dataType);
        m_gpuBufferSizes[i] = bufferSize;
        // 更具buffer size分配内存
        CHECK_CUDA_ERROR(cudaMalloc(&m_gpuBuffers[i], bufferSize));

        if (m_trtEngine->bindingIsInput(i))
        {
            m_iNameIdxMap.insert(std::make_pair(std::string(pName), i)); //模型输入的名称和索引的字典
            uint32_t iWidth = static_cast<uint32_t>(dims.d[3]);
            uint32_t iHeight = static_cast<uint32_t>(dims.d[2]);
            m_iShapes.insert(std::make_pair(std::string(pName), std::make_pair(iWidth, iHeight))); //模型输入的名称和形状的字典
            m_iBlobConunt++; // 模型输入数量
        }
        else
        {
            m_oNameIdxMap.insert(std::make_pair(std::string(pName), i)); //模型输出的名称和索引的字典
            uint32_t oWidth = static_cast<uint32_t>(dims.d[3]);
            uint32_t oHeight = static_cast<uint32_t>(dims.d[2]);
            uint32_t oChannel = static_cast<uint32_t>(dims.d[1]);
            m_oShapes.insert(std::make_pair(std::string(pName), std::make_tuple(oChannel, oHeight, oWidth))); //模型输出的名称和形状的字典
        }
    }
}

void InferenceEngine::releaseGpuMemory()
{
    CHECK_CUDA_ERROR(cudaSetDevice(m_gpuID));
    for (auto item : m_gpuBuffers)
    {
        if (item)
            CHECK_CUDA_ERROR(cudaFree(item));
    }
    m_gpuBuffers.clear();
}

uint32_t InferenceEngine::getInputNum() const
{
    return m_iNameIdxMap.size();
}
uint32_t InferenceEngine::getOutputNum() const
{
    return m_oNameIdxMap.size();
}

uint32_t InferenceEngine::getInputBufferSize(const std::string &blobName) const
{
    auto iter = m_iNameIdxMap.find(blobName);
    if (iter == m_iNameIdxMap.end())
        return 0;
    else
    {
        auto index = iter->second;
        return m_gpuBufferSizes[index];
    }
}

// size_t InferenceEngine::getBufferSize(const std::string &blobName) const
// {
//     return m_DeviceBuffer->size(blobName);
// }

// void* InferenceEngine::getBufferGpu(const std::string &blobName) const
// {
//     return m_DeviceBuffer->getDeviceBuffer(blobName);
// }

float *InferenceEngine::getInputBufferGpu(const std::string &blobName) const
{
    auto iter = m_iNameIdxMap.find(blobName);
    if (iter == m_iNameIdxMap.end())
        return nullptr;
    else
    {
        auto index = iter->second;
        return reinterpret_cast<float *>(m_gpuBuffers[index]);
    }
}

uint32_t InferenceEngine::getOutputBufferSize(const std::string &blobName) const
{
    auto iter = m_oNameIdxMap.find(blobName);
    if (iter == m_oNameIdxMap.end())
        return 0;
    else
    {
        auto index = iter->second;
        return m_gpuBufferSizes[index];
    }
}

void InferenceEngine::getInputShape(const std::string &blobName, uint32_t &width, uint32_t &height)
{
    auto iter = m_iShapes.find(blobName);
    if (iter == m_iShapes.end())
    {
        width = 0;
        height = 0;
    }
    else
    {
        width = iter->second.first;
        height = iter->second.second;
    }
}

float *InferenceEngine::getOutputBufferGpu(const std::string &blobName) const
{
    auto iter = m_oNameIdxMap.find(blobName);
    if (iter == m_oNameIdxMap.end())
        return nullptr;
    else
    {
        auto index = iter->second;
        return reinterpret_cast<float *>(m_gpuBuffers[index]);
    }
}

void InferenceEngine::getOutputShape(const std::string &blobName, uint32_t &channel, uint32_t &height, uint32_t &width)
{
    if (m_oShapes.find(blobName) == m_oShapes.end())
    {
        channel = 0;
        height = 0;
        width = 0;
    }
    else
    {
        auto oShape = m_oShapes[blobName];
        channel = std::get<0>(oShape) == 0 ? 1 : std::get<0>(oShape);
        height = std::get<1>(oShape) == 0 ? 1 : std::get<1>(oShape);
        width = std::get<2>(oShape) == 0 ? 1 : std::get<2>(oShape);
    }
}

int32_t InferenceEngine::doInference(cudaStream_t &stream, int batchsize)
{
    if (m_trtExecContext)
    {
        // 打开计算统计会耗费时间，真正运行时，建议关闭
        // auto mProfile = new Profiler(); // 用于统计每层计算量的profile
        // m_trtExecContext->setProfiler(mProfile); // 添加profile到context中
        m_trtExecContext->setOptimizationProfileAsync(0, stream);

        // 根据输入的batch size重新指定bindingds对应的batch size, batchsize为8的模型也可以推理一张图片
        auto dim = m_trtExecContext->getBindingDimensions(0);
        if (dim.d[0] == -1){
            dim.d[0] = batchsize;
            m_trtExecContext->setBindingDimensions(0, dim); 
        }
        
        
        m_trtExecContext->enqueueV2(&m_gpuBuffers[0], stream, nullptr);
        // const std::string filename {"./report.txt"}; // 计算量结果
        // mProfile->exportJSONProfile(filename); // 写入
        // delete mProfile; // 释放profile
    }
    else
        return FAIL;
    return OK;
}

uint64_t InferenceEngine::getElementSize(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    default:
        throw 4;
    }
}

uint64_t InferenceEngine::volume(const nvinfer1::Dims &d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

bool InferenceEngine::ReadBuffer(const std::string strTrtModel, std::string &buffer)
{
    std::string value;
    std::ifstream stream(strTrtModel.c_str(), std::ios::binary);

    if (stream)
    {
        stream >> std::noskipws;
        std::copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(value));
    }
    else
        return false;
    // int i = 0;
    // std::string key = "anktechSH"; // 解密部分
    // for (auto &v : value)
    // {
    //     v ^= key[i % 9];
    //     i++;
    // }
    buffer = value;
    return true;
}

void InferenceEngine::WriteBuffer(void *buffer, size_t size, const std::string &strTrtModel)
{
    std::ofstream stream(strTrtModel.c_str(), std::ios::binary);

    if (stream)
    {
        // std::string key = "anktechSH";
        // for (int i = 0; i < size; i++)
        // {
        //     stream.put(((char *)buffer)[i] ^ key[i % 9]);
        // }
        stream.write(static_cast<char *>(buffer), size);
    }
    stream.close();
}

std::unique_ptr<nvinfer1::INetworkDefinition> InferenceEngine::addEfficientNMS(std::unique_ptr<nvinfer1::INetworkDefinition> network,
                                    float conf_thres=0.45, float iou_thres=0.5, int max_det=100, bool V8=false)
{
    auto previous_output = network->getOutput(0);
    if (previous_output == nullptr) {
    // 创建插件层失败，进行相应的错误处理
    throw std::runtime_error("previous output is null!");
    }
    network->unmarkOutput(*previous_output);
    nvinfer1::ITensor* boxes;
    nvinfer1::ITensor* obj_score;
    nvinfer1::ITensor* scores;
    if (not V8)
    {
        // output [1, 8400, 85]
        // slice boxes, obj_score, class_scores
        nvinfer1::Dims strides;
        strides.nbDims = 3;
        strides.d[0] = 1;
        strides.d[1] = 1;
        strides.d[2] = 1;
        nvinfer1::Dims starts;
        starts.nbDims = 3;
        starts.d[0] = 0;
        starts.d[1] = 0;
        starts.d[2] = 0;
        nvinfer1::Dims previousShape = previous_output->getDimensions();
        int bs = previousShape.d[0];
        if (bs==-1) bs = 1;
        int num_boxes = previousShape.d[1];
        int temp = previousShape.d[2];
        nvinfer1::Dims shapes;
        shapes.nbDims = 3;
        shapes.d[0] = bs;
        shapes.d[1] = num_boxes;
        shapes.d[2] = 4;

        boxes = network->addSlice(*previous_output, starts, shapes, strides)->getOutput(0);
        int num_clasees = temp - 5;
        starts.d[2] = 4;
        shapes.d[2] = 1;
        obj_score = network->addSlice(*previous_output, starts, shapes, strides)->getOutput(0);
        starts.d[2] = 5;
        shapes.d[2] = num_clasees;
        scores = network->addSlice(*previous_output, starts, shapes, strides)->getOutput(0);
        scores = network->addElementWise(*obj_score, *scores, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    }
    else
    {
        nvinfer1::Dims strides;
        strides.nbDims = 3;
        strides.d[0] = 1;
        strides.d[1] = 1;
        strides.d[2] = 1;
        nvinfer1::Dims starts;
        starts.nbDims = 3;
        starts.d[0] = 0;
        starts.d[1] = 0;
        starts.d[2] = 0;
        nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*previous_output);
        shuffle->setSecondTranspose({0, 2, 1});
        nvinfer1::ITensor* shuffled_output = shuffle->getOutput(0);
        nvinfer1::Dims previousShape = shuffled_output->getDimensions();
        int bs = previousShape.d[0];
        if (bs == -1) bs = 1;
        int num_boxes = previousShape.d[1];
        int temp = previousShape.d[2];
        nvinfer1::Dims shapes;
        shapes.nbDims = 3;
        shapes.d[0] = bs;
        shapes.d[1] = num_boxes;
        shapes.d[2] = 4;

        boxes = network->addSlice(*shuffled_output, starts, shapes, strides)->getOutput(0);
        int num_classes = temp - 4;
        starts.d[2] = 4;
        shapes.d[2] = num_classes;
        scores = network->addSlice(*shuffled_output, starts, shapes, strides)->getOutput(0);         
    }
    // Create EfficientNMS plugin layer
    /*
    "plugin_version": "1",
    "background_class": -1,  # no background class
    "max_output_boxes": detections_per_img,
    "score_threshold": score_thresh,
    "iou_threshold": nms_thresh,
    "score_activation": False,
    "box_coding": 1,
    */
    nvinfer1::IPluginRegistry* registry = getPluginRegistry();
    assert(registry);
    nvinfer1::IPluginCreator* creator = registry->getPluginCreator("EfficientNMS_TRT", "1");
    assert(creator);
    nvinfer1::PluginFieldCollection pluginData;
    int backgroundClass = -1;
    int boxCoding = 1;
    int scoreActivation = 0;
    pluginData.nbFields = 6;
    pluginData.fields = new nvinfer1::PluginField[pluginData.nbFields]{
        nvinfer1::PluginField("background_class", &backgroundClass, nvinfer1::PluginFieldType::kINT32, sizeof(int)),
        nvinfer1::PluginField("max_output_boxes", &max_det, nvinfer1::PluginFieldType::kINT32, sizeof(int)),
        nvinfer1::PluginField("score_threshold", &conf_thres, nvinfer1::PluginFieldType::kFLOAT32, sizeof(float)),
        nvinfer1::PluginField("iou_threshold", &iou_thres, nvinfer1::PluginFieldType::kFLOAT32, sizeof(float)),
        nvinfer1::PluginField("box_coding", &boxCoding, nvinfer1::PluginFieldType::kINT32, sizeof(int)),
        nvinfer1::PluginField("score_activation", &scoreActivation, nvinfer1::PluginFieldType::kINT32, sizeof(int))
    };

    nvinfer1::IPluginV2* nmsLayer = creator->createPlugin("nms_layer", &pluginData);
    if (nmsLayer == nullptr) {
    // 创建插件层失败，进行相应的错误处理
    throw std::runtime_error("Failed to create EfficientNMS plugin layer");
    }
    nvinfer1::ITensor* inputTensors[] = {boxes, scores};
    auto layer = network->addPluginV2(inputTensors, 2, *nmsLayer);
    if (layer == nullptr) {
    // 添加插件层失败，进行相应的错误处理
    throw std::runtime_error("Failed to add EfficientNMS plugin layer to network");
    }

    // Set output names
    layer->getOutput(0)->setName("num_dets");
    layer->getOutput(1)->setName("det_boxes");
    layer->getOutput(2)->setName("det_scores");
    layer->getOutput(3)->setName("det_classes");

    // Mark outputs
    for (int i = 0; i < 4; ++i)
    {
        network->markOutput(*layer->getOutput(i));
    }

    delete[] pluginData.fields;
    delete nmsLayer;
    return std::move(network);
}
} // namespace anktech
