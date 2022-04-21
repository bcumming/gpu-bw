cmake_minimum_required(VERSION 3.17)
project(gpu-bandwidth)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit 10 REQUIRED)

add_executable(bandwidth driver.cpp)

target_include_directories(bandwidth PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(bandwidth PRIVATE CUDA::cudart)

