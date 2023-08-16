export DFFT_PLATFORM="summit"

module load gcc
module load cuda/11.4.0
module load hip-cuda

export DFFT_CUDA_CC="hipcc -D__HIP_PLATFORM_NVIDIA__"
export DFFT_MPI_CXX="mpic++ -O3 -D__HIP_PLATFORM_NVIDIA__"
export DFFT_CUDA_FLAGS="-I/sw/summit/spectrum_mpi/10.1.0.2-20161130/include/"
export DFFT_CUDA_ARCH=""

export DFFT_CUDA_MPI="-Dcudampi"

export DFFT_CUDA_LIB="/sw/summit/hip-cuda/5.1.0/lib"
export DFFT_CUDA_INC="/sw/summit/hip-cuda/5.1.0/include"

export DFFT_CUDA_LD="-lhipfft"

export DFFT_GPU="HIP"