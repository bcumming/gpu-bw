#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

void cuda_check_call(cudaError_t status) {
    if(status != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
                  << cudaGetErrorString(status) << std::endl;
        exit(1);
    }
}

