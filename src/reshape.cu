#include "reshape.hpp"
#include <cassert>

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
    //cudaDeviceSynchronize();
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
    //cudaDeviceSynchronize();
}

__global__ void reshape_final_kernel(const complexFFT_t* __restrict buff1, complexFFT_t* __restrict buff2, int ny, int nz, int3 local_grid_size, int nlocal){
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= nlocal)return;

    int3 local_dims = make_int3(local_grid_size.x,local_grid_size.y / ny,local_grid_size.z / nz); //per rank dims

    int n_recvs = ny * nz; //where we recieve from in each direction.
    int per_rank = nlocal / n_recvs; //how many per rank we have recieved
    int rank = i / per_rank; //which rank I am from

    int i_local = i % per_rank; //my idx local to the rank I am from

    int3 local_coords;

    local_coords.x = i_local / (local_dims.y * local_dims.z);
    local_coords.y = (i_local - local_coords.x * local_dims.y * local_dims.z) / local_dims.z;
    local_coords.z = (i_local - local_coords.x * local_dims.y * local_dims.z) - local_coords.y * local_dims.z;

    int z_coord = rank / ny; //z is slow index for sends

    int y_coord = rank - z_coord * ny; //y is fast index for sends

    int z_offset = (local_grid_size.z / nz) * z_coord;

    int y_offset = (local_grid_size.y / ny) * y_coord;

    int3 global_coords = make_int3(local_coords.x,local_coords.y + y_offset,local_coords.z + z_offset);

    int new_idx = global_coords.x * local_grid_size.y * local_grid_size.z + global_coords.y * local_grid_size.z + global_coords.z;

    buff2[new_idx] = __ldg(&buff1[i]);
    
}

void launch_reshape_final(complexFFT_t* buff1, complexFFT_t* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize){
    int numBlocks = (nlocal + (blockSize - 1))/blockSize;
    int3 local_grid_size_vec = make_int3(local_grid_size[0],local_grid_size[1],local_grid_size[2]);
    //assert(local_grid_size_vec.y % ny == 0);
    //assert(local_grid_size_vec.z % nz == 0);
    reshape_final_kernel<<<numBlocks,blockSize>>>(buff1,buff2,ny,nz,local_grid_size_vec,nlocal);
    //cudaDeviceSynchronize();
    
}