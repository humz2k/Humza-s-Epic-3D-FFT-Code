#include "complex-type.hpp"

void launch_reshape(complexFFT_t* buff1, complexFFT_t* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);

void launch_unreshape(complexFFT_t* buff1, complexFFT_t* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);

void launch_reshape_final(complexFFT_t* buff1, complexFFT_t* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);

void launch_reshape_stride(complexFFT_t* buff1, complexFFT_t* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int stride, int offset, int nlocal, int blockSize);