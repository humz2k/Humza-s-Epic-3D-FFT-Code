#include "dfft.hpp"
#include "reshape.hpp"
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

    #ifdef GPU
    #ifndef cudampi
    h_buff1 = (complexFFT_t*)malloc(sizeof(complexFFT_t) * buffSize());
    h_buff2 = (complexFFT_t*)malloc(sizeof(complexFFT_t) * buffSize());
    #endif
    #endif

    numBlocks = (nlocal + (blockSize - 1))/blockSize;
}

int Distribution::buffSize(){
    return nlocal;
}

Distribution::~Distribution(){
    #ifdef GPU
    #ifndef cudampi
    free(h_buff1);
    free(h_buff2);
    #endif
    #endif
}

template<class T>
void Distribution::alltoall(T* src, T* dest, int n, MPI_Comm comm){
    #if defined(GPU) && !defined(cudampi)
    cudaMemcpy(h_buff1,src,nlocal * sizeof(T),cudaMemcpyDeviceToHost);
    MPI_Alltoall(h_buff1,n * sizeof(T),MPI_BYTE,h_buff2,n * sizeof(T),MPI_BYTE,comm);
    cudaMemcpy(dest,h_buff2,nlocal * sizeof(T),cudaMemcpyHostToDevice);
    #else
    MPI_Alltoall(src,n * sizeof(T),MPI_BYTE,dest,n * sizeof(T),MPI_BYTE,comm);
    #endif
}

template void Distribution::alltoall<complexFFT_t>(complexFFT_t*, complexFFT_t*, int, MPI_Comm);

void Distribution::pencils_1(complexFFT_t* buff1, complexFFT_t* buff2){
    alltoall(buff1,buff2,(nlocal / dims[2]),distcomms[0]);

    reshape_1(buff2,buff1);
}

void Distribution::pencils_2(complexFFT_t* buff1, complexFFT_t* buff2){
    unreshape_1(buff1,buff2);

    alltoall(buff2,buff1,(nlocal / dims[1]),distcomms[1]);

    reshape_2(buff1,buff2);
}

void Distribution::pencils_3(complexFFT_t* buff1, complexFFT_t* buff2){
    unreshape_2(buff1,buff2);

    alltoall(buff2,buff1,(nlocal / (dims[2] * dims[0])),distcomms[2]);

    reshape_3(buff1,buff2);
}

void Distribution::return_pencils(complexFFT_t* buff1, complexFFT_t* buff2){
    unreshape_3(buff1,buff2);

    int dest_x_start = 0;
    int dest_x_end = dims[0] - 1;

    int y = ((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2]))) / local_grid_size[1];
    //int dest_y_end = ((coords[0] * dims[2] + coords[2] + 1) * (ng[1] / (dims[0] * dims[2])) - 1) / local_grid_size[1];

    int y_send = local_grid_size[1] / (ng[1] / (dims[0] * dims[2]));
    int y_send_id = (((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2]))) % local_grid_size[1]) / y_send;

    int z = (coords[1] * (ng[2] / dims[1])) / local_grid_size[2];
    //int dest_z_end = ((coords[1] + 1) * (ng[2] / dims[1]) - 1) / local_grid_size[2];

    int n_recvs = dims[0];

    if (world_rank == 0)printf("n_recvs = %d\n",n_recvs);

    cudaMemcpy(h_buff2,buff2,sizeof(complexFFT_t)*nlocal,cudaMemcpyDeviceToHost);

    MPI_Request reqs[n_recvs];
    int count = 0;
    for (int x = dest_x_start; x < dest_x_end+1; x++){
        //for (int y = dest_y_start; y < dest_y_end+1; y++){
        //    for (int z = dest_z_start; z < dest_z_end+1; z++){
                int dest = x*dims[1]*dims[2] + y*dims[2] + z;
                MPI_Request req;

                int reqidx = x - dest_x_start;
                //int reqidy = y - dest_y_start;
                int nx = (dest_x_end+1) - dest_x_start;
                int ny = n_recvs / nx;
                //printf("nx %d,ny %d\n",nx,ny);

                //int reqid = reqidx * ny + (count % ny);

                //int reqid1 = reqidx * ny + y_send_id;
                int xsrc = x * local_grid_size[0];
                int ysrc = ((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2])));
                int zsrc = (coords[1] * (ng[2] / dims[1]));
                int id = xsrc * local_grid_size[1] * local_grid_size[2] + ysrc * local_grid_size[2] + zsrc;

                int tmp1 = count / y_send;

                int xoff = 0;
                //int yoff = (count - (count / dims[0])*dims[0]) * (ng[1] / (dims[0] * dims[2]));
                int yoff = (count - tmp1 * y_send) * (ng[1] / (dims[0] * dims[2]));
                int zoff = tmp1 * (ng[2] / dims[1]);//(count - (count / dims[2])*dims[2]) ;

                int xrec = local_grid_size[0] * coords[0] + xoff;
                int yrec = local_grid_size[1] * coords[1] + yoff;
                int zrec = local_grid_size[2] * coords[2] + zoff;
                int recid = xrec * local_grid_size[1] * local_grid_size[2] + yrec * local_grid_size[2] + zrec;

                printf("rank %d [%d %d %d] send rank %d :: %d %d %d :: %d %d %d :: %d %d %d -> (%d) (%d)\n",world_rank,coords[0],coords[1],coords[2],dest,xsrc,ysrc,zsrc,xoff,yoff,zoff,xrec,yrec,zrec,id,recid);

                MPI_Isend(&h_buff2[count*(nlocal/n_recvs)],(nlocal/n_recvs) * sizeof(complexFFT_t),MPI_BYTE,dest,id,world_comm,&req);
                MPI_Request_free(&req);
                MPI_Irecv(&h_buff1[count*(nlocal/n_recvs)],(nlocal/n_recvs) * sizeof(complexFFT_t),MPI_BYTE,MPI_ANY_SOURCE,recid,world_comm,&reqs[count]);
                count++;
            //}
        //}
    }

    for (int i = 0; i < n_recvs; i++){
        MPI_Wait(&reqs[i],MPI_STATUS_IGNORE);
    }

    cudaMemcpy(buff1,h_buff1,sizeof(complexFFT_t)*nlocal,cudaMemcpyHostToDevice);

    reshape_final(buff1,buff2,y_send,n_recvs / y_send);
}

void Distribution::reshape_1(complexFFT_t* buff1, complexFFT_t* buff2){
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

void Distribution::unreshape_1(complexFFT_t* buff1, complexFFT_t* buff2){

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

void Distribution::reshape_2(complexFFT_t* buff1, complexFFT_t* buff2){
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

void Distribution::unreshape_2(complexFFT_t* buff1, complexFFT_t* buff2){
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

void Distribution::reshape_3(complexFFT_t* buff1, complexFFT_t* buff2){
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

void Distribution::unreshape_3(complexFFT_t* buff1, complexFFT_t* buff2){

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

void Distribution::reshape_final(complexFFT_t* buff1, complexFFT_t* buff2, int ny, int nz){
    #ifdef GPU
    launch_reshape_final(buff1,buff2,ny,nz,local_grid_size,nlocal,blockSize);
    #else

    #endif
}

void Distribution::fillTest(complexFFT_t* buff){
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

void Distribution::printTest(complexFFT_t* buff){
    MPI_Barrier(world_comm);
    complexFFT_t* printBuff = buff;
    #ifdef GPU
    printBuff = (complexFFT_t*)malloc(sizeof(complexFFT_t)*nlocal);
    cudaMemcpy(printBuff,buff,sizeof(complexFFT_t)*nlocal,cudaMemcpyDeviceToHost);
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

void Distribution::runTest(complexFFT_t* buff1, complexFFT_t* buff2){

    fillTest(buff1);

    printTest(buff1);


    pencils_1(buff1,buff2);


    printTest(buff1);


    pencils_2(buff1,buff2);


    printTest(buff2);


    pencils_3(buff2,buff1);

    printTest(buff1);

    return_pencils(buff1,buff2);

    //printTest(buff1);

    

    printTest(buff2);

    /*unreshape_3(buff1,buff2);

    //printTest(buff2);

    int dest_x_start = 0;
    int dest_x_end = dims[0] - 1;

    int dest_y_start = ((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2]))) / local_grid_size[1];
    int dest_y_end = ((coords[0] * dims[2] + coords[2] + 1) * (ng[1] / (dims[0] * dims[2])) - 1) / local_grid_size[1];

    int y_send = local_grid_size[1] / (ng[1] / (dims[0] * dims[2]));
    int y_send_id = (((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2]))) % local_grid_size[1]) / y_send;

    int dest_z_start = (coords[1] * (ng[2] / dims[1])) / local_grid_size[2];
    int dest_z_end = ((coords[1] + 1) * (ng[2] / dims[1]) - 1) / local_grid_size[2];

    int cy = coords[2] * local_grid_size[2] / (ng[2] / dims[1]);
    int cy2 = ((coords[2] - (((ng[2] / dims[1]) - 1) / local_grid_size[2])) * local_grid_size[2]) / (ng[2] / dims[1]);

    int cx = ((coords[1] * local_grid_size[1]) / (ng[1] / (dims[0] * dims[2]))) / dims[2];// / dims[2];
    int cz = ((coords[1] * local_grid_size[1]) / (ng[1] / (dims[0] * dims[2]))) - cx * dims[2];//(coords[1]) / ((ng[1] / (dims[0] * dims[2])) / local_grid_size[1]) - cx * dims[2];
    
    double tmp;
    //(tmp * (ng[1]/(dims[0]*dims[2])) - 1) / local_grid_size[1];
    tmp = ceil(((double)(coords[1] * local_grid_size[1] + 1)) / ((double)ng[1] / (double)(dims[0] * dims[2])));//(coords[1] * local_grid_size[1] + 1) / (ng[1] / (dims[0] * dims[2]));
    int cx2 = (tmp - 1) / dims[2];
    //(((coords[1] - (((ng[1] / (dims[0] * dims[2])) - 1) / local_grid_size[1])) * local_grid_size[1]) / (ng[1] / (dims[0] * dims[2]))) / dims[2];
    
    int n_recvs = dims[0];
    int cz2 = cz + (dims[0] / (cx2+1 - cx) * (cy2+1 - cy)) - 1;

    //;(((coords[1] * local_grid_size[1] + 1) / (ng[1] / (dims[0] * dims[2]))) - 1) / dims[2];
   //int cz2 = cz + 1;//
    //(((coords[1] - (((ng[1] / (dims[0] * dims[2])) - 1) / local_grid_size[1])) * local_grid_size[1]) / (ng[1] / (dims[0] * dims[2])));//(((coords[1] - (((ng[1] / (dims[0] * dims[2])) - 1) / local_grid_size[1])) * local_grid_size[1]) / (ng[1] / (dims[0] * dims[2]))) - cx2 * dims[2];//((((coords[1] * local_grid_size[1] + 1) / (ng[1] / (dims[0] * dims[2])))) - cx2 * dims[2]);
    /*
    (coords[1] * local_grid_size[1]) / (ng[1] / (dims[0] * dims[2])) = (cx * dims[2] + cz);
    world_rank = 

    

    int n_y_send = (ng[1] / (dims[0] * dims[2]));

    //if(world_rank == 0)printf("n_y_send %d\n",n_y_send);

    //if (world_rank == 2){
        //printf("[%d %d], [%d %d], [%d %d]\n",dest_x_start,dest_x_end,dest_y_start,dest_y_end,dest_z_start,dest_z_end);
        //printf("%d\n",y_send_id);
    //MPI_Barrier(MPI_COMM_WORLD);
    for (int rank = 0; rank < world_size; rank++){
        if (rank == world_rank){
            for (int x = dest_x_start; x < dest_x_end+1; x++){
                for (int y = dest_y_start; y < dest_y_end+1; y++){
                    for (int z = dest_z_start; z < dest_z_end+1; z++){
                        int dest = x*dims[1]*dims[2] + y*dims[2] + z;
                        //printf("rank %d [%d %d %d]: send %d [%d %d %d] %d\n",world_rank,coords[0],coords[1],coords[2],x*dims[1]*dims[2] + y*dims[2] + z,x,y,z,y_send_id);
                    }
                }
            }
            for (int x = cx; x < cx2+1; x++){
                for (int y = cy; y < cy2 + 1; y++){
                    for (int z = cz; z < cz2 + 1; z++){
                        y_send_id = (((x * dims[2] + z) * (ng[1] / (dims[0] * dims[2]))) % local_grid_size[1]) / y_send;
                        int src = x*dims[1]*dims[2] + y*dims[2] + z;
                        //printf("rank %d [%d %d %d]: recv %d [%d %d %d] %d\n",world_rank,coords[0],coords[1],coords[2],x*dims[1]*dims[2] + y*dims[2] + z,x,y,z,y_send_id);
                    }
                }
            }
            //printf("rank %d [%d %d %d]: recv [%d %d] [%d %d] [%d %d]\n",world_rank,coords[0],coords[1],coords[2],cx,cx2,cy,cy2,cz,cz2);
        }
        //MPI_Barrier(MPI_COMM_WORLD);
    }*/
    //}

}