#include "dfft.hpp"
#include <iostream>
#include <iomanip>
#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


uint64_t double_to_uint64_t(double d) {
  uint64_t i;
  memcpy(&i, &d, 8);
  return i;
}


void check_kspace(Distribution &dist, complexFFT_t *a_){

    complexFFT_t* a = (complexFFT_t*)malloc(sizeof(complexFFT_t) * dist.nlocal);
    cudaMemcpy(a,a_,sizeof(complexFFT_t)*dist.nlocal,cudaMemcpyDeviceToHost);

    double LocalRealMin, LocalRealMax, LocalImagMin, LocalImagMax;
    LocalRealMin = LocalRealMax = a[1].x;
    LocalImagMin = LocalImagMax = a[1].y;

    for(int local_indx=0; local_indx<dist.nlocal; local_indx++) {
        double re = a[local_indx].x;
        double im = a[local_indx].y;

        LocalRealMin = re < LocalRealMin ? re : LocalRealMin;
        LocalRealMax = re > LocalRealMax ? re : LocalRealMax;
        LocalImagMin = im < LocalImagMin ? im : LocalImagMin;
        LocalImagMax = im > LocalImagMax ? im : LocalImagMax;
    }

    const MPI_Comm comm = dist.world_comm;

    double GlobalRealMin, GlobalRealMax, GlobalImagMin, GlobalImagMax;
    MPI_Allreduce(&LocalRealMin, &GlobalRealMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&LocalRealMax, &GlobalRealMax, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&LocalImagMin, &GlobalImagMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&LocalImagMax, &GlobalImagMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if(dist.world_rank == 0) {
    std::cout << std::endl << "k-space:" << std::endl
            << "real in " << std::scientific
            << "[" << GlobalRealMin << "," << GlobalRealMax << "]"
            << " = " << std::hex
            << "[" << double_to_uint64_t(GlobalRealMin) << ","
            << double_to_uint64_t(GlobalRealMax) << "]"
            << std::endl
            << "imag in " << std::scientific
            << "[" << GlobalImagMin << "," << GlobalImagMax << "]"
            << " = " << std::hex
                << "[" << double_to_uint64_t(GlobalImagMin) << ","
            << double_to_uint64_t(GlobalImagMax) << "]"
            << std::endl << std::endl << std::fixed;
    }

    free(a);

}

void check_rspace(Distribution &dist, complexFFT_t *a_){

    complexFFT_t* a = (complexFFT_t*)malloc(sizeof(complexFFT_t) * dist.nlocal);
    cudaMemcpy(a,a_,sizeof(complexFFT_t)*dist.nlocal,cudaMemcpyDeviceToHost);

    const int *self = dist.coords;

    double LocalRealMin, LocalRealMax, LocalImagMin, LocalImagMax;
    LocalRealMin = LocalRealMax = a[1].x;
    LocalImagMin = LocalImagMax = a[1].y;

    const MPI_Comm comm = dist.world_comm;

    if(dist.world_rank == 0){
        std::cout << std::endl << "r-space:" << std::endl;
    }

    MPI_Barrier(comm);

    int start = 0;

    if ((self[0] == 0) && (self[1] == 0) && (self[2] == 0)) {
        start = 1;
        std::cout << "a[0,0,0] = " << std::fixed << a[0].x << "," << a[0].y
            << std::hex << " = ("
            << a[0].x
            << ","
            << a[0].y
            << ")" << std::endl;
    }

    for (int local_indx = start; local_indx < dist.nlocal; local_indx++){
        double re = a[local_indx].x;
        double im = a[local_indx].y;
        LocalRealMin = re < LocalRealMin ? re : LocalRealMin;
        LocalRealMax = re > LocalRealMax ? re : LocalRealMax;
        LocalImagMin = im < LocalImagMin ? im : LocalImagMin;
        LocalImagMax = im > LocalImagMax ? im : LocalImagMax;
    }

    double GlobalRealMin, GlobalRealMax, GlobalImagMin, GlobalImagMax;
    MPI_Allreduce(&LocalRealMin, &GlobalRealMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&LocalRealMax, &GlobalRealMax, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&LocalImagMin, &GlobalImagMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&LocalImagMax, &GlobalImagMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if(dist.world_rank == 0) {
    std::cout << "real in " << std::scientific
            << "[" << GlobalRealMin << "," << GlobalRealMax << "]"
            << " = " << std::hex
            << "[" << double_to_uint64_t(GlobalRealMin) << ","
            << double_to_uint64_t(GlobalRealMax) << "]"
            << std::endl
            << "imag in " << std::scientific
            << "[" << GlobalImagMin << "," << GlobalImagMax << "]"
            << " = " << std::hex
                << "[" << double_to_uint64_t(GlobalImagMin) << ","
            << double_to_uint64_t(GlobalImagMax) << "]"
            << std::endl << std::endl << std::fixed;
    }

    free(a);

}


template<class T>
void cpy(T* buff1, T* buff2, int n){
    for (int i = 0; i < n; i++){
        buff2[i] = buff1[i];
    }
}

void test(int ngx, int ngy, int ngz, int blockSize, int reps){
    Distribution dist(MPI_COMM_WORLD,ngx,ngy,ngz,blockSize);
    Dfft dfft(dist);

    complexFFT_t* buff1; cudaMalloc(&buff1,sizeof(complexFFT_t)*dist.buffSize());
    complexFFT_t* buff2; cudaMalloc(&buff2,sizeof(complexFFT_t)*dist.buffSize());

    dfft.makePlans(buff1,buff2);

    //dist.runTest(buff1,buff2);

    for (int i = 0; i < reps; i++){
        if(dist.world_rank == 0)printf("\n\nRep %d/%d\n\n",i+1,reps);
        dfft.fillDelta();

        dfft.forward();

        check_kspace(dist,buff1);

        dfft.backward();

        check_rspace(dist,buff1);
    }

    cudaFree(buff1);
    cudaFree(buff2);
}

int main(int argc, char** argv){

    #ifdef GPU
    cudaFree(0);
    #endif

    MPI_Init(NULL,NULL);
    int ng;
    int reps;
    int blockSize;
    if (argc == 4){
        ng = atoi(argv[3]);
        reps = atoi(argv[1]);
        blockSize = atoi(argv[2]);
    } else{
        printf("USAGE: %s <reps> <blockSize> <ng>\n",argv[0]);
        return 0;
    }

    test(ng,ng,ng,blockSize,reps);

    MPI_Finalize();
    return 0;
}