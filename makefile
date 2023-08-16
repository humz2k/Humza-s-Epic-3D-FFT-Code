DFFT_BUILD_DIR ?= build
DFFT_LIB_DIR ?= lib
DFFT_CPU_LIB_DIR ?= $(DFFT_LIB_DIR)/cpu
DFFT_GPU_LIB_DIR ?= $(DFFT_LIB_DIR)/gpu
DFFT_CUDA_LIB_DIR ?= $(DFFT_GPU_LIB_DIR)/cuda

DFFT_PLATFORM ?= unknown

DFFT_GPU_AR ?= lib/fbfftgpu.a

DFFT_CUDA_LIB ?= /usr/local/cuda/lib64
DFFT_CUDA_INC ?= /usr/local/cuda/include

DFFT_INCLUDE ?= -Iinclude -I$(DFFT_CUDA_INC)
DFFT_LD ?= -L$(DFFT_CUDA_LIB)

#DFFT_CUDA_ARCH ?= -arch=sm_60 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_87,code=sm_87 -gencode=arch=compute_86,code=compute_86
DFFT_CUDA_ARCH ?= -gencode=arch=compute_60,code=sm_60

DFFT_CUDA_LD ?= -lcufft -lcudart

DFFT_CUDA_FLAGS ?= -lineinfo -Xptxas -v -Xcompiler="-fPIC" $(DFFT_CUDA_ARCH)

DFFT_CUDA_MPI ?=

# MPI C compiler
DFFT_MPI_CC ?= mpicc -O3

# MPI C++ compiler
DFFT_MPI_CXX ?= mpicxx -O3

DFFT_CUDA_CC ?= nvcc -O3

main: gpu

gpu: $(DFFT_BUILD_DIR)/testdfft

.PHONY: clean
clean:
	rm -rf $(DFFT_BUILD_DIR)
	rm -rf $(DFFT_LIB_DIR)

$(DFFT_LIB_DIR):
	mkdir -p $(DFFT_LIB_DIR)
	mkdir -p $(DFFT_CPU_LIB_DIR)
	mkdir -p $(DFFT_GPU_LIB_DIR)
	mkdir -p $(DFFT_CUDA_LIB_DIR)

$(DFFT_BUILD_DIR):
	mkdir -p $(DFFT_BUILD_DIR)

$(DFFT_GPU_LIB_DIR)/%.o: src/%.cpp | $(DFFT_LIB_DIR)
	$(DFFT_MPI_CXX) $(DFFT_CUDA_MPI) $(DFFT_INCLUDE) -c -o $@ $<

$(DFFT_CUDA_LIB_DIR)/%.o: src/%.cu | $(DFFT_LIB_DIR)
	$(DFFT_CUDA_CC) $(DFFT_CUDA_MPI) $(DFFT_INCLUDE) $(DFFT_CUDA_FLAGS) -c -o $@ $<

$(DFFT_GPU_AR): $(DFFT_GPU_LIB_DIR)/collective_communicator.o $(DFFT_GPU_LIB_DIR)/distribution.o $(DFFT_GPU_LIB_DIR)/distribution_sends.o $(DFFT_CUDA_LIB_DIR)/reshape.o $(DFFT_GPU_LIB_DIR)/dfft.o
	ar cr $@ $^

$(DFFT_BUILD_DIR)/testdfft: test/testdfft.cpp $(DFFT_GPU_AR) | $(DFFT_BUILD_DIR)
	$(DFFT_MPI_CXX) $(DFFT_CUDA_MPI) -DDFFT_PLATFORM=$(DFFT_PLATFORM) $(DFFT_INCLUDE) $^ $(DFFT_LD) $(DFFT_CUDA_LD) -o $@
