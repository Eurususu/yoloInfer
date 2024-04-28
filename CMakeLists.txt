cmake_minimum_required(VERSION 3.10)
project(sports_camera)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fno-use-cxa-atexit -fexceptions -Wno-unused-function -Wno-unused-variable
                    -Wno-sign-compare -Wno-reorder -Wno-unused-but-set-variable -Wno-misleading-indentation
                    -fPIC -fpermissive)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# set(OpenCV_DIR 3rdparty/OpenCV-4.4/lib/cmake/opencv4)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)
                    
                    
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_86)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --default-stream=per-thread)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(/usr/local/cuda/lib64)

file(GLOB SRC_infer src/infer/*.cpp src/infer/*.cu src/infer/*.hpp)
file(GLOB SRC_track src/track/*.cpp src/track/*.cu)

cuda_add_library(infer_algo SHARED ${SRC_infer} ${SRC_track})
target_include_directories(infer_algo PRIVATE ${TensorRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common 3rdparty/Eigen3)
target_include_directories(infer_algo PUBLIC include src src/utils)
target_link_libraries(infer_algo ${TensorRT_LIBRARIES} ${OpenCV_LIBRARIES})

# cuda_add_library(track_algo SHARED ${SRC_track})

message(STATUS ${OpenCV_INCLUDE_DIRS})

cuda_add_executable(test test.cpp)
target_include_directories(test PRIVATE ${OpenCV_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common src/infer)
target_link_libraries(test infer_algo cuda nvcuvid nppicc nppig nvidia-encode)

# add_executable(test1 test1.cpp)
# target_include_directories(test1 PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(test1 ${OpenCV_LIBRARIES})

