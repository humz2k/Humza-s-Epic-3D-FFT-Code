#include "dfft.hpp"
#include "reshape.hpp"
#include <cassert>

template<class T, template<class> class Communicator>
Distribution<T,Communicator>::Distribution(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize_) : world_comm(comm), ng {ngx,ngy,ngz} , blockSize(blockSize_), dims {0,0,0}, tests(0) {
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
        #ifdef cudampi
        printf("   using cuda mpi\n");
        #else
        printf("   not using cuda mpi (!!)\n");
        #endif
        CollectiveComm.query();
        
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

    CollectiveComm.init(buffSize());

    #ifdef GPU
    #ifndef cudampi
    h_buff1 = (T*)malloc(sizeof(T) * buffSize());
    h_buff2 = (T*)malloc(sizeof(T) * buffSize());
    #endif
    #endif

    numBlocks = (nlocal + (blockSize - 1))/blockSize;
}

template<class T, template<class> class Communicator>
int Distribution<T,  Communicator>::buffSize(){
    return nlocal;
}

template<class T, template<class> class Communicator>
Distribution<T, Communicator>::~Distribution(){
    #ifdef GPU
    #ifndef cudampi
    free(h_buff1);
    free(h_buff2);
    #endif
    #endif
}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::pencils_1(T* buff1, T* buff2){

    //cudaDeviceSynchronize();
    #if DFFT_TIMING == 1
    double comm_start = MPI_Wtime();
    #endif

    CollectiveComm.alltoall(buff1,buff2,(nlocal / dims[2]),distcomms[0]);
    
    #if DFFT_TIMING == 1
    double comm_end = MPI_Wtime();
    pencil_1_comm_time = comm_end - comm_start;
    #endif

    reshape_1(buff2,buff1);
}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::pencils_2(T* buff1, T* buff2){
    unreshape_1(buff1,buff2);

    cudaDeviceSynchronize();

    #if DFFT_TIMING == 1
    double comm_start = MPI_Wtime();
    #endif

    CollectiveComm.alltoall(buff2,buff1,(nlocal / dims[1]),distcomms[1]);

    #if DFFT_TIMING == 1
    double comm_end = MPI_Wtime();
    pencil_2_comm_time = comm_end - comm_start;
    #endif

    reshape_2(buff1,buff2);
}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::pencils_3(T* buff1, T* buff2){
    unreshape_2(buff1,buff2);

    cudaDeviceSynchronize();

    #if DFFT_TIMING == 1
    double comm_start = MPI_Wtime();
    #endif

    CollectiveComm.alltoall(buff2,buff1,(nlocal / (dims[2] * dims[0])),distcomms[2]);

    #if DFFT_TIMING == 1
    double comm_end = MPI_Wtime();
    pencil_3_comm_time = comm_end - comm_start;
    #endif

    reshape_3(buff1,buff2);
}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::return_pencils_sm(T* buff1, T* buff2){
    unreshape_3(buff1,buff2);

    int dest_x_start = 0;
    int dest_x_end = dims[0] - 1;

    int y = ((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2]))) / local_grid_size[1];

    int y_send = local_grid_size[1] / (ng[1] / (dims[0] * dims[2]));
    int y_send_id = (((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2]))) % local_grid_size[1]) / (local_grid_size[1] / y_send);

    int z = (coords[1] * (ng[2] / dims[1])) / local_grid_size[2];

    int n_recvs = dims[0];

    cudaDeviceSynchronize();

    #if DFFT_TIMING == 1
    double comm_start = MPI_Wtime();
    #endif

    

    T* src_buff = buff2;
    T* dest_buff = buff1;
    #ifdef GPU
    #ifndef cudampi
    src_buff = h_buff2;
    dest_buff = h_buff1;
    cudaMemcpy(h_buff2,buff2,sizeof(T)*nlocal,cudaMemcpyDeviceToHost);
    #endif
    #endif

    MPI_Win win;
    MPI_Win_create(dest_buff,sizeof(T)*nlocal,sizeof(T),MPI_INFO_NULL,world_comm,&win);

    MPI_Win_fence(0,win);

    int count = 0;
    for (int x = dest_x_start; x < dest_x_end+1; x++){
        int dest = x*dims[1]*dims[2] + y*dims[2] + z;

        int reqidx = x - dest_x_start;
        int nx = (dest_x_end+1) - dest_x_start;
        int ny = n_recvs / nx;

        int xsrc = x * local_grid_size[0];
        int ysrc = ((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2])));
        int zsrc = (coords[1] * (ng[2] / dims[1]));
        int id = xsrc * local_grid_size[1] * local_grid_size[2] + ysrc * local_grid_size[2] + zsrc;

        int tmp1 = count / y_send;

        int xoff = 0;
        int yoff = (count - tmp1 * y_send) * (ng[1] / (dims[0] * dims[2]));
        int zoff = tmp1 * (ng[2] / dims[1]);

        int xrec = local_grid_size[0] * coords[0] + xoff;
        int yrec = local_grid_size[1] * coords[1] + yoff;
        int zrec = local_grid_size[2] * coords[2] + zoff;
        int recid = xrec * local_grid_size[1] * local_grid_size[2] + yrec * local_grid_size[2] + zrec;
        printf("AHH: %d %d %d %d\n",y_send_id,y_send,local_grid_size[1],(y_send_id / (local_grid_size[1]/y_send)) * (nlocal/n_recvs));
        MPI_Put(&src_buff[count*(nlocal/n_recvs)],(nlocal/n_recvs)*sizeof(T),MPI_BYTE,dest,y_send_id * (nlocal/n_recvs),(nlocal/n_recvs)*sizeof(T),MPI_BYTE,win);

        //MPI_Isend(&src_buff[count*(nlocal/n_recvs)],(nlocal/n_recvs) * sizeof(T),MPI_BYTE,dest,id,world_comm,&req);
        //MPI_Request_free(&req);
        //MPI_Irecv(&dest_buff[count*(nlocal/n_recvs)],(nlocal/n_recvs) * sizeof(T),MPI_BYTE,MPI_ANY_SOURCE,recid,world_comm,&reqs[count]);
        count++;
    }

    MPI_Win_fence(0,win);

    MPI_Win_free(&win);

    #ifdef GPU
    #ifndef cudampi
    cudaMemcpy(buff1,h_buff1,sizeof(T)*nlocal,cudaMemcpyHostToDevice);
    #endif
    #endif

    #if DFFT_TIMING == 1
    double comm_end = MPI_Wtime();
    return_comm_time = comm_end - comm_start;
    #endif

    reshape_final(buff1,buff2,y_send,n_recvs / y_send);

}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::return_pencils(T* buff1, T* buff2){
    unreshape_3(buff1,buff2);

    int dest_x_start = 0;
    int dest_x_end = dims[0] - 1;

    int y = ((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2]))) / local_grid_size[1];

    int y_send = local_grid_size[1] / (ng[1] / (dims[0] * dims[2]));
    int y_send_id = (((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2]))) % local_grid_size[1]) / y_send;

    int z = (coords[1] * (ng[2] / dims[1])) / local_grid_size[2];

    int n_recvs = dims[0];

    cudaDeviceSynchronize();

    #if DFFT_TIMING == 1
    double comm_start = MPI_Wtime();
    #endif

    T* src_buff = buff2;
    T* dest_buff = buff1;
    #ifdef GPU
    #ifndef cudampi
    src_buff = h_buff2;
    dest_buff = h_buff1;
    cudaMemcpy(h_buff2,buff2,sizeof(T)*nlocal,cudaMemcpyDeviceToHost);
    #endif
    #endif

    MPI_Request reqs[n_recvs];
    int count = 0;
    for (int x = dest_x_start; x < dest_x_end+1; x++){
        int dest = x*dims[1]*dims[2] + y*dims[2] + z;
        MPI_Request req;

        int reqidx = x - dest_x_start;
        int nx = (dest_x_end+1) - dest_x_start;
        int ny = n_recvs / nx;

        int xsrc = x * local_grid_size[0];
        int ysrc = ((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2])));
        int zsrc = (coords[1] * (ng[2] / dims[1]));
        int id = xsrc * local_grid_size[1] * local_grid_size[2] + ysrc * local_grid_size[2] + zsrc;

        int tmp1 = count / y_send;

        int xoff = 0;
        int yoff = (count - tmp1 * y_send) * (ng[1] / (dims[0] * dims[2]));
        int zoff = tmp1 * (ng[2] / dims[1]);

        int xrec = local_grid_size[0] * coords[0] + xoff;
        int yrec = local_grid_size[1] * coords[1] + yoff;
        int zrec = local_grid_size[2] * coords[2] + zoff;
        int recid = xrec * local_grid_size[1] * local_grid_size[2] + yrec * local_grid_size[2] + zrec;

        MPI_Isend(&src_buff[count*(nlocal/n_recvs)],(nlocal/n_recvs) * sizeof(T),MPI_BYTE,dest,id,world_comm,&req);
        MPI_Request_free(&req);
        MPI_Irecv(&dest_buff[count*(nlocal/n_recvs)],(nlocal/n_recvs) * sizeof(T),MPI_BYTE,MPI_ANY_SOURCE,recid,world_comm,&reqs[count]);
        count++;
    }

    for (int i = 0; i < n_recvs; i++){
        MPI_Wait(&reqs[i],MPI_STATUS_IGNORE);
    }

    #ifdef GPU
    #ifndef cudampi
    cudaMemcpy(buff1,h_buff1,sizeof(T)*nlocal,cudaMemcpyHostToDevice);
    #endif
    #endif

    #if DFFT_TIMING == 1
    double comm_end = MPI_Wtime();
    return_comm_time = comm_end - comm_start;
    #endif

    reshape_final(buff1,buff2,y_send,n_recvs / y_send);
}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::reshape_1(T* buff1, T* buff2){
    int n_recvs = dims[2];
    int mini_pencil_size = local_grid_size[2];
    int send_per_rank = nlocal / n_recvs;
    int pencils_per_rank = send_per_rank / mini_pencil_size;
    #ifdef GPU
    launch_reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
    #else
    for (int i = 0; i < nlocal; i++){
        int mini_pencil_id = i / mini_pencil_size;
        int rank = i / send_per_rank;

        int rank_offset = rank * mini_pencil_size;

        int pencil_offset = (mini_pencil_id % pencils_per_rank) * mini_pencil_size * n_recvs;

        int local_offset = i % mini_pencil_size;

        int new_idx = rank_offset + pencil_offset + local_offset;

        buff2[new_idx] = buff1[i];
    }
    #endif
}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::unreshape_1(T* buff1, T* buff2){

    int z_dim = ng[2];
    int x_dim = local_grid_size[0] / dims[2];
    int y_dim = (nlocal / z_dim) / x_dim;

    #ifdef GPU
    launch_unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
    #else
    for (int i = 0; i < nlocal; i++){
        int x = i / (y_dim * z_dim);
        int y = (i - (x * y_dim * z_dim)) / z_dim;
        int z = (i - (x * y_dim * z_dim)) - y * z_dim;
        int new_idx = z * x_dim * y_dim + x * y_dim + y;

        buff2[new_idx] = buff1[i];
    }
    #endif
}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::reshape_2(T* buff1, T* buff2){
    int n_recvs = dims[1];
    int mini_pencil_size = local_grid_size[1];
    int send_per_rank = nlocal / n_recvs;
    int pencils_per_rank = send_per_rank / mini_pencil_size;
    #ifdef GPU
    launch_reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
    #else
    for (int i = 0; i < nlocal; i++){
        int mini_pencil_id = i / mini_pencil_size;
        int rank = i / send_per_rank;

        int rank_offset = rank * mini_pencil_size;

        int pencil_offset = (mini_pencil_id % pencils_per_rank) * mini_pencil_size * n_recvs;

        int local_offset = i % mini_pencil_size;

        int new_idx = rank_offset + pencil_offset + local_offset;

        buff2[new_idx] = buff1[i];
    }
    #endif
}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::unreshape_2(T* buff1, T* buff2){
    //int n_sends = dims[0] * dims[2];
    //int n_recvs = dims[2];
    //int mini_pencil_size = local_grid_size[2];
    //int send_per_rank = nlocal / n_sends;

    int z_dim = ng[1];
    int x_dim = local_grid_size[2] / dims[1];
    int y_dim = (nlocal / z_dim) / x_dim;

    #ifdef GPU
    launch_unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
    #else
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
    #endif
}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::reshape_3(T* buff1, T* buff2){
    int n_recvs = dims[0] * dims[2];
    int mini_pencil_size = ng[0] / n_recvs;
    int send_per_rank = nlocal / n_recvs;
    int pencils_per_rank = send_per_rank / mini_pencil_size;
    #ifdef GPU
    launch_reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
    #else
    for (int i = 0; i < nlocal; i++){
        int mini_pencil_id = i / mini_pencil_size;
        int rank = i / send_per_rank;

        int rank_offset = rank * mini_pencil_size;

        int pencil_offset = (mini_pencil_id % pencils_per_rank) * mini_pencil_size * n_recvs;

        int local_offset = i % mini_pencil_size;

        int new_idx = rank_offset + pencil_offset + local_offset;

        buff2[new_idx] = buff1[i];
    }
    #endif
}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::unreshape_3(T* buff1, T* buff2){

    int z_dim = ng[0];
    int x_dim = local_grid_size[1] / dims[0];
    int y_dim = (nlocal / z_dim) / x_dim;

    #ifdef GPU
    launch_unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
    #else
    for (int i = 0; i < nlocal; i++){
        int x = i / (y_dim * z_dim);
        int y = (i - (x * y_dim * z_dim)) / z_dim;
        int z = (i - (x * y_dim * z_dim)) - y * z_dim;
        int new_idx = z * x_dim * y_dim + x * y_dim + y;

        buff2[new_idx] = buff1[i];
    }
    #endif
}

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::reshape_final(T* buff1, T* buff2, int ny, int nz){
    #ifdef GPU
    launch_reshape_final(buff1,buff2,ny,nz,local_grid_size,nlocal,blockSize);
    #else
    int3 local_dims = make_int3(local_grid_size.x,local_grid_size.y / ny,local_grid_size.z / nz); //per rank dims

    int n_recvs = ny * nz; //where we recieve from in each direction.
    int per_rank = nlocal / n_recvs; //how many per rank we have recieved

    for (int i = 0; i < nlocal; i++){

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

        buff2[new_idx] = buff1[i];
    }
    #endif
}

#ifndef cudampi
template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::fillTest(T* buff){
    int i = 0;
    for (int x = local_coords_start[0]; x < local_grid_size[0] + local_coords_start[0]; x++){
        for (int y = local_coords_start[1]; y < local_grid_size[1] + local_coords_start[1]; y++){
            for (int z = local_coords_start[2]; z < local_grid_size[2] + local_coords_start[2]; z++){
                int idx = x * ng[1] * ng[2] + y * ng[2] + z;
                #ifdef GPU
                h_buff1[i].x = idx;
                h_buff1[i].y = 0;
                #else
                buff[i].x = idx;
                buff[i].y = 0;
                #endif
                i++;
            }
        }
    }
    #ifdef GPU
    cudaMemcpy(buff,h_buff1,sizeof(complexFFT_t)*nlocal,cudaMemcpyHostToDevice);
    #endif
}
#endif

template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::printTest(T* buff){
    MPI_Barrier(world_comm);
    T* printBuff = buff;
    #ifdef GPU
    printBuff = (T*)malloc(sizeof(T)*nlocal);
    cudaMemcpy(printBuff,buff,sizeof(T)*nlocal,cudaMemcpyDeviceToHost);
    #endif

    for (int rank = 0; rank < world_size; rank++){
        if (rank == world_rank){
            for (int i = 0; i < nlocal; i++){
                int idx = printBuff[i].x;
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
    #ifdef GPU
    free(printBuff);
    #endif
}
#ifndef cudampi
template<class T, template<class> class Communicator>
void Distribution<T, Communicator>::runTest(T* buff1, T* buff2){

    fillTest(buff1);

    printTest(buff1);

    pencils_1(buff1,buff2);

    printTest(buff1);

    pencils_2(buff1,buff2);

    printTest(buff2);

    pencils_3(buff2,buff1);

    printTest(buff1);

    return_pencils(buff1,buff2);

    printTest(buff2);

}
#endif

template class Distribution<complexFloat,AllToAll>;
template class Distribution<complexDouble,AllToAll>;
template class Distribution<complexFloat,PairSends>;
template class Distribution<complexDouble,PairSends>;