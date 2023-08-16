#include "dfft.hpp"

template<class T>
CollectiveCommunicator<T>::CollectiveCommunicator() : set(false){}

template<class T>
void CollectiveCommunicator<T>::init(int buff_sz_){
    #ifdef GPU
    #ifndef cudampi
    //printf("Allocating swap!\n");
    buff_sz = buff_sz_;
    h_buff1 = (T*)malloc(sizeof(T) * buff_sz);
    h_buff2 = (T*)malloc(sizeof(T) * buff_sz);
    set = true;
    #endif
    #endif
}

template<class T>
CollectiveCommunicator<T>::~CollectiveCommunicator(){
    #ifdef GPU
    #ifndef cudampi
    if(set){
        //printf("Freeing swap!\n");
        free(h_buff1);
        free(h_buff2);
    }
    #endif
    #endif
}

template<class T>
void AllToAll<T>::query(){
    printf("   Using MPI_Alltoall Communication!\n");
}

template<class T>
void PairSends<T>::query(){
    printf("   Using MPI_Isend/MPI_Irecv Communication!\n");
}

template<class T>
void AllToAll<T>::alltoall(T* src, T* dest, int n, MPI_Comm comm){
    #if defined(GPU) && !defined(cudampi)
    //printf("buff_sz = %d\n",this->buff_sz);
    gpuMemcpy(this->h_buff1,src,this->buff_sz * sizeof(T),gpuMemcpyDeviceToHost);
    MPI_Alltoall(this->h_buff1,n * sizeof(T),MPI_BYTE,this->h_buff2,n * sizeof(T),MPI_BYTE,comm);
    gpuMemcpy(dest,this->h_buff2,this->buff_sz * sizeof(T),gpuMemcpyHostToDevice);
    #else
    MPI_Alltoall(src,n * sizeof(T),MPI_BYTE,dest,n * sizeof(T),MPI_BYTE,comm);
    #endif
}

template<class T>
void PairSends<T>::alltoall(T* src, T* dest, int n, MPI_Comm comm){
    int comm_rank; MPI_Comm_rank(comm,&comm_rank);
    int comm_size; MPI_Comm_size(comm,&comm_size);
    #if defined(GPU) && !defined(cudampi)
    T* src_buff = this->h_buff1;
    T* dest_buff = this->h_buff2;
    gpuMemcpy(src_buff,src,this->buff_sz * sizeof(T),gpuMemcpyDeviceToHost);
    #else
    T* src_buff = src;
    T* dest_buff = dest;
    #endif

    if (comm_size == 2){
        //printf("SEND_RECV!\n");
        MPI_Sendrecv(&src_buff[((comm_rank + 1)%comm_size) * n],n*sizeof(T),MPI_BYTE,(comm_rank + 1)%comm_size,0,&dest_buff[((comm_rank+1)%comm_size) * n],n*sizeof(T),MPI_BYTE,(comm_rank + 1)%comm_size,0,comm,MPI_STATUS_IGNORE);

        #if defined(GPU) && !defined(cudampi)
        for (int j = 0; j < n; j++){
            dest_buff[comm_rank * n + j] = src_buff[comm_rank * n + j];
        }
        #else
        gpuMemcpy(&dest_buff[comm_rank * n],&src_buff[comm_rank * n],n * sizeof(T),gpuMemcpyDeviceToDevice);
        #endif
        
    } else {
        
        MPI_Request reqs[comm_size*2];
        for (int i = 0; i < comm_size; i++){
            if (i == comm_rank){
                #if defined(GPU) && !defined(cudampi)
                for (int j = 0; j < n; j++){
                    dest_buff[comm_rank * n + j] = src_buff[comm_rank * n + j];
                }
                #else
                gpuMemcpy(&dest_buff[comm_rank * n],&src_buff[comm_rank * n],n * sizeof(T),gpuMemcpyDeviceToDevice);
                #endif
                continue;
            } else {
                MPI_Isend(&src_buff[i * n],n * sizeof(T),MPI_BYTE,i,0,comm,&reqs[i*2]);
                MPI_Irecv(&dest_buff[i * n],n * sizeof(T),MPI_BYTE,i,0,comm,&reqs[i*2 + 1]);
            }
        }

        for (int i = 0; i < comm_size; i++){
            if (i == comm_rank)continue;
            MPI_Wait(&reqs[i*2],MPI_STATUS_IGNORE);
            MPI_Wait(&reqs[i*2 + 1],MPI_STATUS_IGNORE);
        }
    }

    #if defined(GPU) && !defined(cudampi)
    gpuMemcpy(dest,this->h_buff2,this->buff_sz * sizeof(T),gpuMemcpyHostToDevice);
    #endif
    
}

template class CollectiveCommunicator<complexFloat>;
template class CollectiveCommunicator<complexDouble>;
template class AllToAll<complexFloat>;
template class AllToAll<complexDouble>;
template class PairSends<complexFloat>;
template class PairSends<complexDouble>;
