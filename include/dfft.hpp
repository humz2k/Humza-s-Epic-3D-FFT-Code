#include <mpi.h>
#include "complex-type.hpp"
#include <stdio.h>

#define DFFT_TIMING 1

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

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

template<class T>
class CollectiveCommunicator{
    public:
        #ifdef GPU
        #ifndef cudampi
        T* h_buff1;
        T* h_buff2;
        int buff_sz;
        bool set;
        #endif
        #endif
        CollectiveCommunicator();
        //CollectiveCommunicator(int buff_sz);
        ~CollectiveCommunicator();

        void init(int buff_sz);

        void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

        void query();
};

template<class T>
class AllToAll : public CollectiveCommunicator<T>{
    public:
        void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);
        void query();
};

template<class T>
class PairSends : public CollectiveCommunicator<T>{
    public:
        void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);
        void query();
};

template <class T, template<class> class Communicator>
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

        Communicator<T> CollectiveComm;

        double pencil_1_comm_time;
        double pencil_1_calc_time;
        double pencil_2_comm_time;
        double pencil_2_calc_time;
        double pencil_3_comm_time;
        double pencil_3_calc_time;
        double return_comm_time;
        double return_calc_time;

        #ifdef GPU
            #ifndef cudampi
            T* h_buff1;
            T* h_buff2;
            #endif
        #endif

        cudaStream_t memcpystream;

        int blockSize;
        int numBlocks;

        Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_);
        ~Distribution();

        void pencils_1(T* buff1, T* buff2);
        void pencils_2(T* buff1, T* buff2);
        void pencils_3(T* buff1, T* buff2);
        void return_pencils(T* buff1, T* buff2);
        void return_pencils_sm(T* buff1, T* buff2);

        int tests;
        void fillTest(T* buff);
        void printTest(T* buff);

        void reshape_1(T* buff1, T* buff2);
        void unreshape_1(T* buff1, T* buff2);

        void reshape_2(T* buff1, T* buff2);
        void unreshape_2(T* buff1, T* buff2);

        void reshape_3(T* buff1, T* buff2);
        void unreshape_3(T* buff1, T* buff2);

        void reshape_final(T* buff1, T* buff2, int ny, int nz);

        void runTest(T* buff1, T* buff2);

        void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

        int buffSize();
};

template <class T, class Dist>
class Dfft{
    public:
        Dist& dist;
        int ng;
        int nlocal;
        T* buff1;
        T* buff2;
        #ifdef GPU
        cufftHandle plan;
        #endif
        bool plansMade;

        Dfft(Dist& dist_);
        ~Dfft();

        void makePlans(T* buff1_, T* buff2_);
        void forward();
        void backward();
        void exec1d(T* buff1_, T* buff2_, int direction);
        void fft(int direction);
        void fillDelta();
};