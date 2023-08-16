#include "dfft.hpp"

template<class T, class Dist>
Dfft<T, Dist>::Dfft(Dist& dist_) : dist(dist_){
    ng = dist.ng[0];
    nlocal = dist.nlocal;
    plansMade = false;
    //#ifdef HIP
    //rocfft_setup();
    //#endif
}

template<class T, class Dist>
Dfft<T, Dist>::~Dfft(){
    if (plansMade)gpufftDestroy(plan);
    //#ifdef HIP
    //rocfft_cleanup();
    //#endif
}

template<>
void Dfft<complexDouble,Distribution<complexDouble,AllToAll>>::makePlans(complexDouble* buff1_, complexDouble* buff2_){
    buff1 = buff1_;
    buff2 = buff2_;

    #ifdef GPU
    int nFFTs = nlocal / ng;
    //#ifdef CUDA
    if (gpufftPlan1d(&plan, ng, GPUFFT_Z2Z, nFFTs) != GPUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;	
    }
    //#else
    //#ifdef HIP
    //rocfft_plan_create(&plan_forward,rocfft_placement_notinplace,rocfft_transform_type_complex_forward,rocfft_precision_double,1,&ng,&nFFTs,nullptr);
    //rocfft_plan_create(&plan_inverse,rocfft_placement_notinplace,rocfft_transform_type_complex_inverse,rocfft_precision_double,1,&ng,&nFFTs,nullptr);
    //#endif
    //#endif
    #endif

    plansMade = true;

}

template<>
void Dfft<complexFloat,Distribution<complexFloat,AllToAll>>::makePlans(complexFloat* buff1_, complexFloat* buff2_){
    buff1 = buff1_;
    buff2 = buff2_;

    #ifdef GPU
    int nFFTs = nlocal / ng;
    //#ifdef CUDA
    if (gpufftPlan1d(&plan, ng, GPUFFT_C2C, nFFTs) != GPUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;	
    }
    //#else
    //#ifdef HIP
    //rocfft_plan_create(&plan_forward,rocfft_placement_notinplace,rocfft_transform_type_complex_forward,rocfft_precision_single,1,&ng,&nFFTs,nullptr);
    //rocfft_plan_create(&plan_inverse,rocfft_placement_notinplace,rocfft_transform_type_complex_inverse,rocfft_precision_single,1,&ng,&nFFTs,nullptr);
    //#endif
    //#endif
    #endif

    plansMade = true;

}

template<>
void Dfft<complexDouble,Distribution<complexDouble,PairSends>>::makePlans(complexDouble* buff1_, complexDouble* buff2_){
    buff1 = buff1_;
    buff2 = buff2_;

    #ifdef GPU
    int nFFTs = nlocal / ng;
    //#ifdef CUDA
    if (gpufftPlan1d(&plan, ng, GPUFFT_Z2Z, nFFTs) != GPUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;	
    }
    //#else
    //#ifdef HIP
    //rocfft_plan_create(&plan_forward,rocfft_placement_notinplace,rocfft_transform_type_complex_forward,rocfft_precision_double,1,&ng,&nFFTs,nullptr);
    //rocfft_plan_create(&plan_inverse,rocfft_placement_notinplace,rocfft_transform_type_complex_inverse,rocfft_precision_double,1,&ng,&nFFTs,nullptr);
    //#endif
    //#endif
    #endif

    plansMade = true;

}

template<>
void Dfft<complexFloat,Distribution<complexFloat,PairSends>>::makePlans(complexFloat* buff1_, complexFloat* buff2_){
    buff1 = buff1_;
    buff2 = buff2_;

    #ifdef GPU
    int nFFTs = nlocal / ng;
    //#ifdef CUDA
    if (gpufftPlan1d(&plan, ng, GPUFFT_C2C, nFFTs) != GPUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;	
    }
    //#else
    //#ifdef HIP
    //rocfft_plan_create(&plan_forward,rocfft_placement_notinplace,rocfft_transform_type_complex_forward,rocfft_precision_single,1,&ng,&nFFTs,nullptr);
    //rocfft_plan_create(&plan_inverse,rocfft_placement_notinplace,rocfft_transform_type_complex_inverse,rocfft_precision_single,1,&ng,&nFFTs,nullptr);
    //#endif
    //#endif
    #endif

    plansMade = true;

}

template<class T, class Dist>
void Dfft<T,Dist>::forward(){
    #ifdef DFFT_TIMING
    double start = MPI_Wtime();
    #endif
    fft(GPUFFT_FORWARD);
    #ifdef DFFT_TIMING
    double stop = MPI_Wtime();
    printTimingStats(MPI_COMM_WORLD,"FORWARD  ",stop-start);
    #endif
    #if DFFT_TIMING == 1
    printTimingStats(MPI_COMM_WORLD,"P1Comm   ",dist.pencil_1_comm_time);
    printTimingStats(MPI_COMM_WORLD,"P2Comm   ",dist.pencil_2_comm_time);
    printTimingStats(MPI_COMM_WORLD,"P3Comm   ",dist.pencil_3_comm_time);
    printTimingStats(MPI_COMM_WORLD,"RetComm  ",dist.return_comm_time);
    printTimingStats(MPI_COMM_WORLD,"TotComm  ",dist.return_comm_time + dist.pencil_3_comm_time + dist.pencil_2_comm_time + dist.pencil_1_comm_time);
    #endif
}

template<class T, class Dist>
void Dfft<T,Dist>::backward(){
    #ifdef DFFT_TIMING
    double start = MPI_Wtime();
    #endif
    fft(GPUFFT_INVERSE);
    #ifdef DFFT_TIMING
    double stop = MPI_Wtime();
    printTimingStats(MPI_COMM_WORLD,"BACKWARD ",stop-start);
    #endif
    #if DFFT_TIMING == 1
    printTimingStats(MPI_COMM_WORLD,"P1Comm   ",dist.pencil_1_comm_time);
    printTimingStats(MPI_COMM_WORLD,"P2Comm   ",dist.pencil_2_comm_time);
    printTimingStats(MPI_COMM_WORLD,"P3Comm   ",dist.pencil_3_comm_time);
    printTimingStats(MPI_COMM_WORLD,"RetComm  ",dist.return_comm_time);
    printTimingStats(MPI_COMM_WORLD,"TotComm  ",dist.return_comm_time + dist.pencil_3_comm_time + dist.pencil_2_comm_time + dist.pencil_1_comm_time);
    #endif
}

template<>
void Dfft<complexDouble,Distribution<complexDouble,AllToAll>>::exec1d(complexDouble* buff1_, complexDouble* buff2_, int direction){
    #ifdef GPU
    //#ifdef CUDA
    if (gpufftExecZ2Z(plan, buff1_, buff2_, direction) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Forward failed\n");
        return;	
    }
    //#else
    //#ifdef HIP
    //if (direction == GPUFFT_FORWARD){
    //    rocfft_execute(plan_forward,(void**)&buff1_,(void**)&buff2_,nullptr);
    //} else {
    //    rocfft_execute(plan_inverse,(void**)&buff1_,(void**)&buff2_,nullptr);
    //}
    //#endif
    //#endif
    //cudaDeviceSynchronize();
    #endif
}
template<>
void Dfft<complexFloat,Distribution<complexFloat,AllToAll>>::exec1d(complexFloat* buff1_, complexFloat* buff2_, int direction){
    #ifdef GPU
    //#ifdef CUDA
    if (gpufftExecC2C(plan, buff1_, buff2_, direction) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Forward failed\n");
        return;	
    }
    //cudaDeviceSynchronize();
    //#else
    //#ifdef HIP
    //if (direction == GPUFFT_FORWARD){
    //    rocfft_execute(plan_forward,(void**)&buff1_,(void**)&buff2_,nullptr);
    //} else {
    //    rocfft_execute(plan_inverse,(void**)&buff1_,(void**)&buff2_,nullptr);
    //}
    //#endif
    //#endif
    //cudaDeviceSynchronize();
    #endif
}

template<>
void Dfft<complexDouble,Distribution<complexDouble,PairSends>>::exec1d(complexDouble* buff1_, complexDouble* buff2_, int direction){
    #ifdef GPU
    //#ifdef CUDA
    if (gpufftExecZ2Z(plan, buff1_, buff2_, direction) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Forward failed\n");
        return;	
    }
    //cudaDeviceSynchronize();
    //#else
    //#ifdef HIP
    //if (direction == GPUFFT_FORWARD){
    //    rocfft_execute(plan_forward,(void**)&buff1_,(void**)&buff2_,nullptr);
    //} else {
    //    rocfft_execute(plan_inverse,(void**)&buff1_,(void**)&buff2_,nullptr);
    //}
    //#endif
    //#endif
    //cudaDeviceSynchronize();
    #endif
}
template<>
void Dfft<complexFloat,Distribution<complexFloat,PairSends>>::exec1d(complexFloat* buff1_, complexFloat* buff2_, int direction){
    #ifdef GPU
    //#ifdef CUDA
    if (gpufftExecC2C(plan, buff1_, buff2_, direction) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Forward failed\n");
        return;	
    }
    //cudaDeviceSynchronize();
    //#else
    //#ifdef HIP
    //if (direction == GPUFFT_FORWARD){
    //    rocfft_execute(plan_forward,(void**)&buff1_,(void**)&buff2_,nullptr);
    //} else {
    //    rocfft_execute(plan_inverse,(void**)&buff1_,(void**)&buff2_,nullptr);
    //}
    //#endif
    //#endif
    //cudaDeviceSynchronize();
    #endif
}

template<class T, class Dist>
void Dfft<T,Dist>::fft(int direction){
    dist.pencils_1(buff1,buff2);

    exec1d(buff1,buff2,direction);

    dist.pencils_2(buff2,buff1);

    exec1d(buff1,buff2,direction);

    dist.pencils_3(buff2,buff1);

    exec1d(buff1,buff2,direction);

    dist.return_pencils(buff2,buff1);

    gpuDeviceSynchronize();
}

template<class T, class Dist>
void Dfft<T,Dist>::fillDelta(){
    //complexFFT_t* fillArray = buff1;
    #ifdef GPU
    gpuMemset(buff1,0,sizeof(T)*dist.nlocal);
    T origin;
    origin.x = 1;
    origin.y = 0;
    if ((dist.coords[0] == 0) && (dist.coords[1] == 0) && (dist.coords[2] == 0)){
        gpuMemcpy(buff1,&origin,sizeof(T),gpuMemcpyHostToDevice);
    }
    #endif
}

template class Dfft<complexFloat,Distribution<complexFloat,AllToAll>>;
template class Dfft<complexDouble,Distribution<complexDouble,AllToAll>>;
template class Dfft<complexFloat,Distribution<complexFloat,PairSends>>;
template class Dfft<complexDouble,Distribution<complexDouble,PairSends>>;