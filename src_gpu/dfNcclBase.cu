#include "dfNcclBase.H"
#include "dfMatrixDataBase.H"

static uint64_t getHostHash(const char* string) {
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

void ncclInit(MPI_Comm mpi_comm, ncclComm_t& nccl_comm, ncclUniqueId& nccl_id,
        int *pnRanks, int *pmyRank, int *plocalRank, int *p_mpi_init_flag)
{
    // check mpi initialized
    int mpi_init_flag;
    checkMpiErrors(MPI_Initialized(&mpi_init_flag));
    if(mpi_init_flag) MPI_Barrier(mpi_comm);
    else {
        fprintf(stderr, "MPI is not yet initialized!\n");
        exit(EXIT_FAILURE);
    }

    //initializing MPI info
    int nRanks, myRank, localRank = 0;
    checkMpiErrors(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    checkMpiErrors(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    //calculating localRank based on hostname which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    checkMpiErrors(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, mpi_comm));
    for (int p=0; p<nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    //get NCCL unique ID at rank 0 and broadcast it to all others
    if (myRank == 0) ncclGetUniqueId(&nccl_id);
    checkMpiErrors(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, mpi_comm));

    //picking a GPU based on localRank, allocate device buffers
    checkCudaErrors(cudaSetDevice(localRank));

    //initializing NCCL
    checkNcclErrors(ncclCommInitRank(&nccl_comm, nRanks, nccl_id, myRank));

    *pnRanks = nRanks;
    *pmyRank = myRank;
    *plocalRank = localRank;
    *p_mpi_init_flag = mpi_init_flag;
}

void ncclDestroy(ncclComm_t nccl_comm)
{
    //finalizing NCCL
    ncclCommDestroy(nccl_comm);
}

void ncclTest(ncclComm_t nccl_comm)
{
    int size = 32*1024*1024;

    // create buf and stream
    float *sendbuff, *recvbuff;
    cudaStream_t s;
    checkCudaErrors(cudaMalloc(&sendbuff, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&recvbuff, size * sizeof(float)));
    checkCudaErrors(cudaStreamCreate(&s));

    //communicating using NCCL
    checkNcclErrors(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
                nccl_comm, s));

    //completing NCCL operation by synchronizing on the CUDA stream
    checkCudaErrors(cudaStreamSynchronize(s));
    usleep(3 * 1000 * 1000);

    //free device buffers
    checkCudaErrors(cudaFree(sendbuff));
    checkCudaErrors(cudaFree(recvbuff));
    checkCudaErrors(cudaStreamDestroy(s));
}

