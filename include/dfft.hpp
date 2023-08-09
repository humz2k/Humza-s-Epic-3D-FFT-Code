#include <mpi.h>
#include "complex-type.hpp"
#include <stdio.h>

#define GPU
//#define cudampi

inline void printTimingStats(MPI_Comm comm,        // comm for MPI_Allreduce()
		      const char *preamble, // text at beginning of line
		      double dt)            // delta t in seconds
{
  int myrank, nranks;
  double max, min, sum, avg, var, stdev;

  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &nranks);

  MPI_Allreduce(&dt, &max, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&dt, &min, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&dt, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);
  avg = sum/nranks;

  dt -= avg;
  dt *= dt;
  MPI_Allreduce(&dt, &var, 1, MPI_DOUBLE, MPI_SUM, comm);
  var *= 1.0/nranks;
  stdev = sqrt(var);

  if(myrank==0) {
    printf("%s  max %.3es  avg %.3es  min %.3es  dev %.3es\n",
	   preamble, max, avg, min, stdev);
  }

  MPI_Barrier(comm);

  return;
}


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

        #ifdef GPU
            #ifndef cudampi
            complexFFT_t* h_buff1;
            complexFFT_t* h_buff2;
            #endif
        #endif

        cudaStream_t memcpystream;

        int blockSize;
        int numBlocks;

        Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_);
        ~Distribution();

        void pencils_1(complexFFT_t* buff1, complexFFT_t* buff2);
        void pencils_2(complexFFT_t* buff1, complexFFT_t* buff2);
        void pencils_3(complexFFT_t* buff1, complexFFT_t* buff2);
        void return_pencils(complexFFT_t* buff1, complexFFT_t* buff2);

        int tests;
        void fillTest(complexFFT_t* buff);
        void printTest(complexFFT_t* buff);

        void reshape_1(complexFFT_t* buff1, complexFFT_t* buff2);
        void unreshape_1(complexFFT_t* buff1, complexFFT_t* buff2);

        void reshape_2(complexFFT_t* buff1, complexFFT_t* buff2);
        void unreshape_2(complexFFT_t* buff1, complexFFT_t* buff2);

        void reshape_3(complexFFT_t* buff1, complexFFT_t* buff2);
        void unreshape_3(complexFFT_t* buff1, complexFFT_t* buff2);

        void runTest(complexFFT_t* buff1, complexFFT_t* buff2);

        template<class T>
        void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

        int buffSize();
};

class Dfft{
    public:
        Distribution& dist;
        int ng;
        int nlocal;
        complexFFT_t* buff1;
        complexFFT_t* buff2;
        #ifdef GPU
        cufftHandle plan;
        #endif
        bool plansMade;

        Dfft(Distribution& dist_);
        ~Dfft();

        void makePlans(complexFFT_t* buff1_, complexFFT_t* buff2_);
        void forward();
        void backward();
        void fft(int direction);
        void fillDelta();
};