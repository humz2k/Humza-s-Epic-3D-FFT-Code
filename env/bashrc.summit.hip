export DFFT_PLATFORM="summit"

module load gcc
module load cuda
module load hip-cuda

export DFFT_CUDA_CC="hipcc"
export DFFT_MPI_CXX="mpic++ -O3"
export DFFT_CUDA_FLAGS="-I/sw/summit/spectrum_mpi/10.1.0.2-20161130/include/"
export DFFT_CUDA_ARCH=""

export DFFT_CUDA_MPI="-Dcudampi"

export DFFT_CUDA_LIB="/sw/summit/cuda/11.0.3/lib64"
export DFFT_CUDA_INC="/sw/summit/cuda/11.0.3/include"

export DFFT_CUDA_LD="-lcufft -lcudart"

export DFFT_GPU="HIP"