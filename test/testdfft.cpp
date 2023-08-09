#include "distribution.hpp"

void test(int ngx, int ngy, int ngz, int blockSize){
    Distribution dist(MPI_COMM_WORLD,ngx,ngy,ngz,blockSize);

    complexFFT_t* buff1 = (complexFFT_t*)malloc(sizeof(complexFFT_t) * dist.buffSize());
    complexFFT_t* buff2 = (complexFFT_t*)malloc(sizeof(complexFFT_t) * dist.buffSize());

    dist.fillTest(buff1);
    dist.printTest(buff1);

    //MPI_Alltoall(buff1,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),MPI_BYTE,buff2,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[0]);

    //dist.alltoall(buff1,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),buff2,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),dist.distcomms[0]);

    dist.alltoall(buff1,buff2,(dist.nlocal / dist.dims[2]),dist.distcomms[0]);

    dist.printTest(buff2);

    dist.reshape_1(buff2,buff1);

    dist.printTest(buff1);

    dist.unreshape_1(buff1,buff2);

    dist.printTest(buff2);

    //MPI_Alltoall(buff2,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),MPI_BYTE,buff1,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[1]);

    //dist.alltoall(buff2,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),buff1,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),dist.distcomms[1]);

    dist.alltoall(buff2,buff1,(dist.nlocal / dist.dims[1]),dist.distcomms[1]);

    dist.printTest(buff1);

    dist.reshape_2(buff1,buff2);

    dist.printTest(buff2);

    dist.unreshape_2(buff2,buff1);

    dist.printTest(buff1);

    //MPI_Alltoall(buff1,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),MPI_BYTE,buff2,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[2]);

    //dist.alltoall(buff1,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),buff2,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),dist.distcomms[2]);

    dist.alltoall(buff1,buff2,(dist.nlocal / (dist.dims[1] * dist.dims[0])),dist.distcomms[2]);

    dist.printTest(buff2);

    dist.reshape_3(buff2,buff1);

    dist.printTest(buff1);

    dist.unreshape_3(buff1,buff2);

    dist.printTest(buff2);

    //MPI_Alltoall(buff2,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),MPI_BYTE,buff1,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[2]);

    //dist.alltoall(buff2,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),buff1,(dist.nlocal / (dist.dims[1] * dist.dims[0])) * sizeof(complexFFT_t),dist.distcomms[2]);

    dist.alltoall(buff2,buff1,(dist.nlocal / (dist.dims[1] * dist.dims[0])),dist.distcomms[2]);

    dist.printTest(buff1);

    //MPI_Alltoall(buff1,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),MPI_BYTE,buff2,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[1]);

    //dist.alltoall(buff1,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),buff2,(dist.nlocal / dist.dims[1]) * sizeof(complexFFT_t),dist.distcomms[1]);

    dist.alltoall(buff1,buff2,(dist.nlocal / dist.dims[1]),dist.distcomms[1]);

    dist.printTest(buff2);

    //MPI_Alltoall(buff2,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),MPI_BYTE,buff1,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),MPI_BYTE,dist.distcomms[0]);

    //dist.alltoall(buff2,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),buff1,(dist.nlocal / dist.dims[2]) * sizeof(complexFFT_t),dist.distcomms[0]);

    dist.alltoall(buff2,buff1,(dist.nlocal / dist.dims[2]),dist.distcomms[0]);

    dist.printTest(buff1);


    free(buff1);
    free(buff2);
}

int main(){

    MPI_Init(NULL,NULL);

    test(8,8,8,64);

    MPI_Finalize();
    return 0;
}