#include "distribution.hpp"
#include <cassert>

Distribution::Distribution(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize_) : world_comm(comm), ng {ngx,ngy,ngz} , blockSize(blockSize_), dims {0,0,0}, tests(0) {
    MPI_Comm_rank(world_comm,&world_rank);
    MPI_Comm_size(world_comm,&world_size);

    MPI_Dims_create(world_size,3,dims);

    local_grid_size[0] = ng[0] / dims[0];
    local_grid_size[1] = ng[1] / dims[1];
    local_grid_size[2] = ng[2] / dims[2];

    nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

    coords[0] = world_rank / (dims[1] * dims[2]);
    coords[1] = (world_rank - coords[0] * (dims[1] * dims[2])) / dims[2];
    coords[2] = (world_rank - coords[0] * (dims[1] * dims[2])) - coords[1] * dims[2];

    local_coords_start[0] = local_grid_size[0] * coords[0];
    local_coords_start[1] = local_grid_size[1] * coords[1];
    local_coords_start[2] = local_grid_size[2] * coords[2];

    if (world_rank == 0){
        printf("Distribution:\n");
        printf("   ng              = [%d %d %d]\n",ng[0],ng[1],ng[2]);
        printf("   dims            = [%d %d %d]\n",dims[0],dims[1],dims[2]);
        printf("   local_grid_size = [%d %d %d]\n",local_grid_size[0],local_grid_size[1],local_grid_size[2]);
        printf("   nlocal          = %d\n",nlocal);
        printf("   blockSize       = %d\n",blockSize);
        printf("   world_size      = %d\n",world_size);
        
    }

    MPI_Comm z_col_comm;

    int z_col_idx = coords[0] * dims[1] + coords[1];
    int z_col_rank = coords[2];

    MPI_Comm_split(world_comm,z_col_idx,z_col_rank,&z_col_comm);

    distcomms[0] = z_col_comm;

    MPI_Comm y_col_comm;

    int y_col_idx = coords[0] * dims[2] + coords[2];
    int y_col_rank = coords[1];

    MPI_Comm_split(world_comm,y_col_idx,y_col_rank,&y_col_comm);

    distcomms[1] = y_col_comm;

    MPI_Comm x_col_comm;

    int x_col_idx = coords[1];
    int x_col_rank = coords[0] * dims[2] + coords[2];

    MPI_Comm_split(world_comm,x_col_idx,x_col_rank,&x_col_comm);

    distcomms[2] = x_col_comm;

    assert(local_grid_size[0] % dims[2] == 0);
    assert(local_grid_size[2] % dims[1] == 0);
    assert(local_grid_size[2] % (dims[0] * dims[2]) == 0);
    assert(local_grid_size[2] % (dims[0] * dims[2]) == 0);
    assert(ng[2] % (dims[0] * dims[2]) == 0);
}

int Distribution::buffSize(){
    return nlocal;
}

Distribution::~Distribution(){

}

void Distribution::alltoall(complexFFT_t* src, size_t n_send, complexFFT_t* dest, size_t n_recv, MPI_Comm comm){
    MPI_Alltoall(src,n_send,MPI_BYTE,dest,n_recv,MPI_BYTE,comm);
}

void Distribution::reshape_1(complexFFT_t* buff1, complexFFT_t* buff2){
    int n_recvs = dims[2];
    int mini_pencil_size = local_grid_size[2];
    int send_per_rank = nlocal / n_recvs;
    int pencils_per_rank = send_per_rank / mini_pencil_size;
    for (int i = 0; i < nlocal; i++){
        int mini_pencil_id = i / mini_pencil_size;
        int rank = i / send_per_rank;

        int rank_offset = rank * mini_pencil_size;

        int pencil_offset = (mini_pencil_id % pencils_per_rank) * mini_pencil_size * n_recvs;

        int local_offset = i % mini_pencil_size;

        int new_idx = rank_offset + pencil_offset + local_offset;

        buff2[new_idx] = buff1[i];
    }
}

void Distribution::unreshape_1(complexFFT_t* buff1, complexFFT_t* buff2){
    int n_sends = dims[1];
    //int n_recvs = dims[2];
    int mini_pencil_size = local_grid_size[2];
    int send_per_rank = nlocal / n_sends;

    int z_dim = ng[2];
    int x_dim = local_grid_size[0] / dims[2];
    int y_dim = (nlocal / z_dim) / x_dim;

    //int pencils_per_rank = send_per_rank / mini_pencil_size;
    for (int i = 0; i < nlocal; i++){
        int x = i / (y_dim * z_dim);
        int y = (i - (x * y_dim * z_dim)) / z_dim;
        int z = (i - (x * y_dim * z_dim)) - y * z_dim;
        //if (world_rank == 0){
        //    printf("%d = [%d %d %d]\n",i,x,y,z);
        //}
        int new_idx = z * x_dim * y_dim + x * y_dim + y;

        buff2[new_idx] = buff1[i];
    }
}

void Distribution::reshape_2(complexFFT_t* buff1, complexFFT_t* buff2){
    int n_recvs = dims[1];
    int mini_pencil_size = local_grid_size[1];
    int send_per_rank = nlocal / n_recvs;
    int pencils_per_rank = send_per_rank / mini_pencil_size;
    for (int i = 0; i < nlocal; i++){
        int mini_pencil_id = i / mini_pencil_size;
        int rank = i / send_per_rank;

        int rank_offset = rank * mini_pencil_size;

        int pencil_offset = (mini_pencil_id % pencils_per_rank) * mini_pencil_size * n_recvs;

        int local_offset = i % mini_pencil_size;

        int new_idx = rank_offset + pencil_offset + local_offset;

        buff2[new_idx] = buff1[i];
    }
}

void Distribution::unreshape_2(complexFFT_t* buff1, complexFFT_t* buff2){
    //int n_sends = dims[0] * dims[2];
    //int n_recvs = dims[2];
    //int mini_pencil_size = local_grid_size[2];
    //int send_per_rank = nlocal / n_sends;

    int z_dim = ng[1];
    int x_dim = local_grid_size[2] / dims[1];
    int y_dim = (nlocal / z_dim) / x_dim;

    //int pencils_per_rank = send_per_rank / mini_pencil_size;
    for (int i = 0; i < nlocal; i++){
        int x = i / (y_dim * z_dim);
        int y = (i - (x * y_dim * z_dim)) / z_dim;
        int z = (i - (x * y_dim * z_dim)) - y * z_dim;
        //if (world_rank == 0){
        //    printf("%d = [%d %d %d]\n",i,x,y,z);
        //}
        int new_idx = z * x_dim * y_dim + x * y_dim + y;

        buff2[new_idx] = buff1[i];
    }
}

void Distribution::reshape_3(complexFFT_t* buff1, complexFFT_t* buff2){
    int n_recvs = dims[0] * dims[2];
    int mini_pencil_size = ng[0] / n_recvs;
    int send_per_rank = nlocal / n_recvs;
    int pencils_per_rank = send_per_rank / mini_pencil_size;
    for (int i = 0; i < nlocal; i++){
        int mini_pencil_id = i / mini_pencil_size;
        int rank = i / send_per_rank;

        int rank_offset = rank * mini_pencil_size;

        int pencil_offset = (mini_pencil_id % pencils_per_rank) * mini_pencil_size * n_recvs;

        int local_offset = i % mini_pencil_size;

        int new_idx = rank_offset + pencil_offset + local_offset;

        buff2[new_idx] = buff1[i];
    }
}

/*void Distribution::unreshape_3(complexFFT_t* buff1, complexFFT_t* buff2){
    int n_recvs = dims[0] * dims[2];
    int mini_pencil_size = ng[0] / n_recvs;
    int send_per_rank = nlocal / n_recvs;
    int pencils_per_rank = send_per_rank / mini_pencil_size;
    for (int i = 0; i < nlocal; i++){
        int mini_pencil_id = i / mini_pencil_size;
        int rank = i / send_per_rank;

        int rank_offset = rank * mini_pencil_size;

        int pencil_offset = (mini_pencil_id % pencils_per_rank) * mini_pencil_size * n_recvs;

        int local_offset = i % mini_pencil_size;

        int new_idx = rank_offset + pencil_offset + local_offset;

        buff2[i] = buff1[new_idx];
    }
}*/

void Distribution::unreshape_3(complexFFT_t* buff1, complexFFT_t* buff2){
    //int n_sends = dims[0] * dims[2];
    //int n_recvs = dims[2];
    //int mini_pencil_size = local_grid_size[2];
    //int send_per_rank = nlocal / n_sends;

    int z_dim = ng[0];
    int x_dim = local_grid_size[1] / dims[0];
    int y_dim = (nlocal / z_dim) / x_dim;

    //int pencils_per_rank = send_per_rank / mini_pencil_size;
    for (int i = 0; i < nlocal; i++){
        int x = i / (y_dim * z_dim);
        int y = (i - (x * y_dim * z_dim)) / z_dim;
        int z = (i - (x * y_dim * z_dim)) - y * z_dim;
        //if (world_rank == 0){
        //    printf("%d = [%d %d %d]\n",i,x,y,z);
        //}
        int new_idx = z * x_dim * y_dim + x * y_dim + y;

        buff2[new_idx] = buff1[i];
    }
}

void Distribution::fillTest(complexFFT_t* buff){
    int i = 0;
    for (int x = local_coords_start[0]; x < local_grid_size[0] + local_coords_start[0]; x++){
        for (int y = local_coords_start[1]; y < local_grid_size[1] + local_coords_start[1]; y++){
            for (int z = local_coords_start[2]; z < local_grid_size[2] + local_coords_start[2]; z++){
                int idx = x * ng[1] * ng[2] + y * ng[2] + z;
                buff[i].x = idx;
                buff[i].y = 0;
                i++;
            }
        }
    }
}

void Distribution::printTest(complexFFT_t* buff){
    MPI_Barrier(world_comm);
    
    for (int rank = 0; rank < world_size; rank++){
        if (rank == world_rank){
            for (int i = 0; i < nlocal; i++){
                int idx = buff[i].x;
                int x = idx / (ng[1] * ng[2]);
                int y = (idx - (x * ng[1] * ng[2])) / ng[2];
                int z = (idx - (x * ng[1] * ng[2])) - (y * ng[2]);
                printf("TEST -> world_rank %d, test %d : idx %d = [%d %d %d]\n",world_rank,tests,i,x,y,z);
            }
        }
        MPI_Barrier(world_comm);
    }
    MPI_Barrier(world_comm);
    tests++;
}