#include "complex-type.hpp"

template<class T>
void launch_reshape(T* buff1, T* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);

template<class T>
void launch_unreshape(T* buff1, T* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);

template<class T>
void launch_reshape_final(T* buff1, T* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);