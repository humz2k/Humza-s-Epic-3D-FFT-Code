#include <mpi.h>
#include "complex-type.hpp"

class Distribution{
    public:
        int ng[3];
        int nlocal;
        int world_size;
        int world_rank;
        int local_grid_size[3];
        int dims[3];
        int coords[3];
        int local_coords_start[3];
        MPI_Comm world_comm;
        MPI_Comm distcomms[4];

        cudaStream_t memcpystream;

        int blockSize;

        Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_);
        ~Distribution();

        void pencils_1(complexFFT_t* buff1, complexFFT_t* buff2);
        void pencils_2(complexFFT_t* buff1, complexFFT_t* buff2);
        void pencils_3(complexFFT_t* buff1, complexFFT_t* buff2);

        int tests;
        void fillTest(complexFFT_t* buff);
        void printTest(complexFFT_t* buff);

        void reshape_1(complexFFT_t* buff1, complexFFT_t* buff2);
        void unreshape_1(complexFFT_t* buff1, complexFFT_t* buff2);

        void reshape_2(complexFFT_t* buff1, complexFFT_t* buff2);
        void unreshape_2(complexFFT_t* buff1, complexFFT_t* buff2);

        void reshape_3(complexFFT_t* buff1, complexFFT_t* buff2);
        void unreshape_3(complexFFT_t* buff1, complexFFT_t* buff2);

        void alltoall(complexFFT_t* src, size_t n_send, complexFFT_t* dest, size_t n_recv, MPI_Comm comm);

        template<class T>
        void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

        int buffSize();
};