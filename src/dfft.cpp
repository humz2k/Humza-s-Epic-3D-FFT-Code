#include "dfft.hpp"

Dfft::Dfft(Distribution& dist_) : dist(dist_){
    ng = dist.ng[0];
    nlocal = dist.nlocal;
    plansMade = false;
}

Dfft::~Dfft(){
    if (plansMade)cufftDestroy(plan);
}

void Dfft::makePlans(complexFFT_t* buff1_, complexFFT_t* buff2_){
    buff1 = buff1_;
    buff2 = buff2_;

    #ifdef GPU
    int nFFTs = nlocal / ng;
    if (cufftPlan1d(&plan, ng, CUFFT_Z2Z, nFFTs) != CUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;	
    }
    #endif

    plansMade = true;

}

void Dfft::forward(){
    double start = MPI_Wtime();
    fft(CUFFT_FORWARD);
    double stop = MPI_Wtime();
    printTimingStats(MPI_COMM_WORLD,"FORWARD ",stop-start);
}

void Dfft::backward(){
    double start = MPI_Wtime();
    fft(CUFFT_INVERSE);
    double stop = MPI_Wtime();
    printTimingStats(MPI_COMM_WORLD,"BACKWARD ",stop-start);
}

void Dfft::fft(int direction){
    //printf("pencils_1()\n");
    dist.pencils_1(buff1,buff2);
    
    //dist.printTest(buff1);
    #ifdef GPU
    //printf("exec()\n");
    if (cufftExecZ2Z(plan, buff1, buff2, direction) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Forward failed\n");
        return;	
    }
    //cudaDeviceSynchronize();
    #endif

    //printf("pencils_2()\n");
    dist.pencils_2(buff2,buff1);

    //dist.printTest(buff2);
    #ifdef GPU
    //printf("exec()\n");
    if (cufftExecZ2Z(plan, buff1, buff2, direction) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Forward failed\n");
        return;	
    }
    //cudaDeviceSynchronize();
    #endif

    //printf("pencils_3()\n");
    dist.pencils_3(buff2,buff1);

    //dist.printTest(buff1);
    #ifdef GPU
    //printf("exec()\n");
    if (cufftExecZ2Z(plan, buff1, buff2, direction) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Forward failed\n");
        return;	
    }
    //cudaDeviceSynchronize();
    #endif

    //printf("return_pencils()\n");
    dist.return_pencils(buff2,buff1);
    //dist.printTest(buff1);
}

void Dfft::fillDelta(){
    //complexFFT_t* fillArray = buff1;
    #ifdef GPU
    cudaMemset(buff1,0,sizeof(complexFFT_t)*dist.nlocal);
    complexFFT_t origin;
    origin.x = 1;
    origin.y = 0;
    if (dist.world_rank == 0){
        cudaMemcpy(buff1,&origin,sizeof(complexFFT_t),cudaMemcpyHostToDevice);
    }
    #endif
}