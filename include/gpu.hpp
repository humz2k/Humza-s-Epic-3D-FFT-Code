#define CUDA

#ifdef CUDA

#include <cuda_runtime.h>
#include <cufft.h>

#define complexDouble cufftDoubleComplex
#define complexFloat cufftComplex
//#define complexFFT_t cufftDoubleComplex

#define gpufftHandle cufftHandle

#define gpufftPlan1d cufftPlan1d

#define gpufftDestroy cufftDestroy

#define GPUFFT_Z2Z CUFFT_Z2Z
#define GPUFFT_C2C CUFFT_C2C
#define GPUFFT_SUCCESS CUFFT_SUCCESS

#define gpufftExecZ2Z cufftExecZ2Z
#define gpufftExecC2C cufftExecC2C

#define gpuStream_t cudaStream_t

#define gpuMalloc cudaMalloc

#define gpuMemset cudaMemset

#define gpuMemcpy cudaMemcpy

#define gpuDeviceSynchronize cudaDeviceSynchronize

#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice

#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost

#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#define gpuFree cudaFree

#define gpuLaunch(kernel,numBlocks,blockSize,...) kernel<<<numBlocks,blockSize>>>(__VA_ARGS__)

#else 

#ifdef HIP

#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"

#define gpuStream_t hipStream_t

#define gpuMalloc hipMalloc

#define gpuMemcpy hipMemcpy

#define gpuDeviceSynchronize hipDeviceSynchronize

#define gpuMemcpyHostToDevice hipMemcpyHostToDevice

#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost

#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define gpuFree hipFree

#define gpuLaunch(kernel,numBlocks,blockSize,...) hipLaunchKernel(kernel,dim3(numBlocks),dim3(blockSize),0,0,__VA_ARGS__)

#endif

#endif