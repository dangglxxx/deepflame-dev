#pragma once
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "nccl.h"
#include "mpi.h"

#define checkMpiErrors(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define checkNcclErrors(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void ncclInit(MPI_Comm mpi_comm, ncclComm_t& nccl_comm, ncclUniqueId& nccl_id,
		int *nRanks, int *myRank, int *localRank, int *mpi_init_flag);

void ncclDestroy(ncclComm_t nccl_comm);

// TODO: for temp
void ncclTest(ncclComm_t nccl_comm);
