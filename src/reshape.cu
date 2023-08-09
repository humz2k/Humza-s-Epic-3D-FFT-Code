#include "reshape.hpp"

__global__ void reshape_kernel(const complexFFT_t* __restrict buff1, complexFFT_t* __restrict buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal){
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= nlocal)return;

    int mini_pencil_id = i / mini_pencil_size;

    int rank = i / send_per_rank;

    int rank_offset = rank * mini_pencil_size;

    int pencil_offset = (mini_pencil_id % pencils_per_rank) * mini_pencil_size * n_recvs;

    int local_offset = i % mini_pencil_size;

    int new_idx = rank_offset + pencil_offset + local_offset;

    buff2[new_idx] = __ldg(&buff1[i]);
}

void launch_reshape(complexFFT_t* buff1, complexFFT_t* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    int numBlocks = (nlocal + (blockSize - 1))/blockSize;
    reshape_kernel<<<numBlocks,blockSize>>>(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal);
}

__global__ void unreshape_kernel(const complexFFT_t* __restrict buff1, complexFFT_t* __restrict buff2, int z_dim, int x_dim, int y_dim, int nlocal){
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= nlocal)return;

    int x = i / (y_dim * z_dim);
    int y = (i - (x * y_dim * z_dim)) / z_dim;
    int z = (i - (x * y_dim * z_dim)) - y * z_dim;
    int new_idx = z * x_dim * y_dim + x * y_dim + y;

    buff2[new_idx] = __ldg(&buff1[i]);
}

void launch_unreshape(complexFFT_t* buff1, complexFFT_t* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    int numBlocks = (nlocal + (blockSize - 1))/blockSize;
    unreshape_kernel<<<numBlocks,blockSize>>>(buff1,buff2,z_dim,x_dim,y_dim,nlocal);
}