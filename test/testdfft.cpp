#include "distribution.hpp"

void test(int ngx, int ngy, int ngz, int blockSize){
    Distribution dist(MPI_COMM_WORLD,ngx,ngy,ngz,blockSize);

    complexFFT_t* buff1 = (complexFFT_t*)malloc(sizeof(complexFFT_t) * dist.buffSize());
    complexFFT_t* buff2 = (complexFFT_t*)malloc(sizeof(complexFFT_t) * dist.buffSize());

    dist.fillTest(buff1);
    dist.printTest(buff1);

    MPI_Alltoall(buff1,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),MPI_BYTE,buff2,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[0]);

    dist.printTest(buff2);

    dist.reshape_1(buff2,buff1);

    dist.printTest(buff1);

    dist.unreshape_1(buff1,buff2);

    dist.printTest(buff2);

    MPI_Alltoall(buff2,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),MPI_BYTE,buff1,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[1]);

    dist.printTest(buff1);

    dist.reshape_2(buff1,buff2);

    dist.printTest(buff2);

    dist.unreshape_2(buff2,buff1);

    dist.printTest(buff1);

    MPI_Alltoall(buff1,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),MPI_BYTE,buff2,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[2]);

    dist.printTest(buff2);

    dist.reshape_3(buff2,buff1);

    dist.printTest(buff1);

    dist.unreshape_3(buff1,buff2);

    dist.printTest(buff2);

    MPI_Alltoall(buff2,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),MPI_BYTE,buff1,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[2]);

    //MPI_Alltoall(buff2,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),MPI_BYTE,buff1,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[1]);

    dist.printTest(buff1);

    MPI_Alltoall(buff1,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),MPI_BYTE,buff2,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[1]);

    dist.printTest(buff2);

    MPI_Alltoall(buff2,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),MPI_BYTE,buff1,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[0]);

    dist.printTest(buff1);

    //MPI_Barrier(MPI_COMM_WORLD);
    //int transfer_rank = dist.coords[2] * dist.dims[0] * dist.dims[1] + dist.coords[0] * dist.dims[1] + dist.coords[1];
    
    //int recv_from_x = dist.world_rank / (dist.dims[0] * dist.dims[1]);
    //int recv_from_y = (dist.world_rank - recv_from_x * (dist.dims[0] * dist.dims[1])) / dist.dims[1];
    //int recv_from_z = (dist.world_rank - recv_from_x * (dist.dims[0] * dist.dims[1])) - recv_from_y * dist.dims[1];
    
    //int recv_from_rank = recv_from_y * dist.dims[0] * dist.dims[2] + recv_from_z * dist.dims[0] + recv_from_x;
    
    //MPI_Sendrecv(buff1,dist.nlocal * sizeof(complexFFT_t),MPI_BYTE,transfer_rank,0,buff2,dist.nlocal*sizeof(complexFFT_t),MPI_BYTE,recv_from_rank,0,dist.world_comm,MPI_STATUS_IGNORE);



    //dist.printTest(buff2);
    
    //printf("Rank %d send to %d recv from %d %d %d -> %d\n",dist.world_rank,transfer_rank,recv_from_x,recv_from_y,recv_from_z,recv_from_rank);
    //MPI_Alltoall(buff1,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),MPI_BYTE,buff2,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[2]);

    //dist.printTest(buff2);

    free(buff1);
    free(buff2);
}

int main(){

    MPI_Init(NULL,NULL);

    test(48,48,48,64);

    MPI_Finalize();
    return 0;
}