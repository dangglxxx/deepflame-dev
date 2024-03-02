/**
 * \file AmgXSolver.cpp
 * \brief Definition of member functions of the class AmgXSolver.
 * \author Pi-Yueh Chuang (pychuang@gwu.edu)
 * \author Matt Martineau (mmartineau@nvidia.com)
 * \date 2015-09-01
 * \copyright Copyright (c) 2015-2019 Pi-Yueh Chuang, Lorena A. Barba.
 * \copyright Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *            This project is released under MIT License.
 */

// AmgXWrapper
#include "AmgXSolver.H"
#include <numeric>
#include <limits>
#include <mpi.h>
#include "dfMatrixDataBase.H"

// initialize AmgXSolver::count to 0
int AmgXSolver::count = 0;

// initialize AmgXSolver::rsrc to nullptr;
AMGX_resources_handle AmgXSolver::rsrc = nullptr;


/* \implements AmgXSolver::AmgXSolver */
AmgXSolver::AmgXSolver(const std::string &modeStr, const std::string &cfgFile, const int devID)
{
    initialize(modeStr, cfgFile, devID);
}


/* \implements AmgXSolver::~AmgXSolver */
AmgXSolver::~AmgXSolver()
{
    if (isInitialised) finalize();
}


/* \implements AmgXSolver::initialize */
void AmgXSolver::initialize(const std::string &modeStr, const std::string &cfgFile, int devID)
{
    
    // if this instance has already been initialized, skip
    if (isInitialised) {
        fprintf(stderr,
                "This AmgXSolver instance has been initialized on this process.\n");
        exit(0);
    }

    // increase the number of AmgXSolver instances
    count += 1;

    // get the mode of AmgX solver
    setMode(modeStr);  

    // check if MPI has been initialized
    MPI_Initialized(&isMPIEnabled);
    if (isMPIEnabled) {
        MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
        mpiWorld = MPI_COMM_WORLD;
    }

    // initialize AmgX
    initAmgX(cfgFile, devID);

    // a bool indicating if this instance is initialized
    isInitialised = true;

    return;
}

/* \implements AmgXSolver::setMode */
void AmgXSolver::setMode(const std::string &modeStr)
{
    if (modeStr == "dDDI")
        mode = AMGX_mode_dDDI;
    else if (modeStr == "dDFI")
        mode = AMGX_mode_dDFI;
    else if (modeStr == "dFFI")
        mode = AMGX_mode_dFFI;
    else if (modeStr[0] == 'h') {
        printf("CPU mode, %s, is not supported in this wrapper!",
                modeStr.c_str());
        exit(0);
    }
    else {
        printf("%s is not an available mode! Available modes are: "
                "dDDI, dDFI, dFFI.\n", modeStr.c_str());
        exit(0);
    }
}


/* \implements AmgXSolver::initAmgX */
 void AmgXSolver::initAmgX(const std::string &cfgFile, int devID)
{
    // only the first instance (AmgX solver) is in charge of initializing AmgX
    if (count == 1)
    {
        // initialize AmgX
        AMGX_SAFE_CALL(AMGX_initialize());

        // intialize AmgX plugings
        AMGX_SAFE_CALL(AMGX_initialize_plugins());

        // let AmgX to handle errors returned
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
    }

    // create an AmgX configure object
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, cfgFile.c_str()));

    // let AmgX handle returned error codes internally
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    // create an AmgX resource object, only the first instance is in charge
    if (count == 1) {
        if (isMPIEnabled) {
            AMGX_resources_create(&rsrc, cfg, &mpiWorld, 1, &devID);
        } else {
            AMGX_resources_create_simple(&rsrc, cfg);
        }
    }

    // create AmgX vector object for unknowns and RHS
    AMGX_vector_create(&AmgXP, rsrc, mode);
    AMGX_vector_create(&AmgXRHS, rsrc, mode);

    // create AmgX matrix object for unknowns and RHS
    AMGX_matrix_create(&AmgXA, rsrc, mode);

    // create an AmgX solver object
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    // obtain the default number of rings based on current configuration
    AMGX_config_get_default_number_of_rings(cfg, &ring);
}

/* \implements AmgXSolver::finalize */
void AmgXSolver::finalize()
{
    // skip if this instance has not been initialised
    if (!isInitialised)
    {
        fprintf(stderr,
                "This AmgXWrapper has not been initialised. "
                "Please initialise it before finalization.\n");
        exit(0);
    }

    // destroy solver instance
    AMGX_solver_destroy(solver);

    // destroy matrix instance
    AMGX_matrix_destroy(AmgXA);

    // destroy RHS and unknown vectors
    AMGX_vector_destroy(AmgXP);
    AMGX_vector_destroy(AmgXRHS);

    // only the last instance need to destroy resource and finalizing AmgX
    if (count == 1)
    {
        AMGX_resources_destroy(rsrc);
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

        AMGX_SAFE_CALL(AMGX_finalize_plugins());
        AMGX_SAFE_CALL(AMGX_finalize());
    }
    else
    {
        AMGX_config_destroy(cfg);
    }

    // decrease the number of instances
    count -= 1;

    // change status
    isInitialised = false;
}

/* \implements AmgXSolver::setOperator */
void AmgXSolver::setOperator
(
    const int nRows,
    const int nGlobalRows,
    const int nNz,
    const int *rowIndex,
    const int *colIndex,
    const double *value
)
{

    // Check the matrix size is not larger than tolerated by AmgX
    if(nRows > std::numeric_limits<int>::max())
    {
        fprintf(stderr,
                "AmgX does not support a global number of rows greater than "
                "what can be stored in 32 bits (nGlobalRows = %d).\n",
                nRows);
        exit(0);
    }

    if (nNz > std::numeric_limits<int>::max())
    {
        fprintf(stderr,
                "AmgX does not support non-zeros per (consolidated) rank greater than"
                "what can be stored in 32 bits (nLocalNz = %d).\n",
                nNz);
        exit(0);
    }

    // check if mpi initialize
    if (!isMPIEnabled)
    {
        // upload matrix A to AmgX
        AMGX_matrix_upload_all(
            AmgXA, nRows, nNz, 1, 1, rowIndex, colIndex, value, nullptr);

        // bind the matrix A to the solver
        AMGX_solver_setup(solver, AmgXA);

        // connect (bind) vectors to the matrix
        AMGX_vector_bind(AmgXP, AmgXA);
        AMGX_vector_bind(AmgXRHS, AmgXA);
    } else {
        MPI_Barrier(MPI_COMM_WORLD);

        AMGX_distribution_handle dist;
        AMGX_distribution_create(&dist, cfg);

        // Must persist until after we call upload
        std::vector<int> offsets(mpiSize + 1, 0);

        // Determine the number of rows per GPU
        std::vector<int> nRowsPerGPU(mpiSize);
        MPI_Allgather(&nRows, 1, MPI_INT, nRowsPerGPU.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Calculate the global offsets
        std::partial_sum(nRowsPerGPU.begin(), nRowsPerGPU.end(), offsets.begin() + 1);
        
        AMGX_distribution_set_partition_data(
            dist, AMGX_DIST_PARTITION_OFFSETS, offsets.data());
        
        // Set the column indices size, 32- / 64-bit
        AMGX_distribution_set_32bit_colindices(dist, true);

        AMGX_matrix_upload_distributed(
            AmgXA, nGlobalRows, nRows, nNz, 1, 1, rowIndex,
            colIndex, value, nullptr, dist);
        
        AMGX_distribution_destroy(dist);

        // bind the matrix A to the solver
        AMGX_solver_setup(solver, AmgXA);

        // connect (bind) vectors to the matrix
        AMGX_vector_bind(AmgXP, AmgXA);
        AMGX_vector_bind(AmgXRHS, AmgXA);

        MPI_Barrier(MPI_COMM_WORLD); 
    }
}


/* \implements AmgXSolver::updateOperator */
void AmgXSolver::updateOperator
(
    const int nRows,
    const int nNz,
    const double *value
)
{

    // Replace the coefficients for the CSR matrix A within AmgX
    AMGX_matrix_replace_coefficients(AmgXA, nRows, nNz, value, nullptr);

    // Re-setup the solver (a reduced overhead setup that accounts for consistent matrix structure)
    AMGX_solver_resetup(solver, AmgXA);
}

/* \implements AmgXSolver::solve */
// void AmgXSolver::solve(
//     int nLocalRows, Vec& p, Vec& b, AmgXCSRMatrix& matrix)
// {
//     double* pscalar;
//     double* bscalar;

//     // get pointers to the raw data of local vectors
//     VecGetArray(p, &pscalar);
//     VecGetArray(b, &bscalar);

//     solve(nLocalRows, pscalar, bscalar, matrix);

//     VecRestoreArray(p, &pscalar);
//     VecRestoreArray(b, &bscalar);
// }


/* \implements AmgXSolver::solve */
void AmgXSolver::solve(
    int nRows, double* psi, const double* rhs)
{
    // Upload potentially consolidated vectors to AmgX
    AMGX_vector_upload(AmgXP, nRows, 1, psi);
    AMGX_vector_upload(AmgXRHS, nRows, 1, rhs);

    if (isMPIEnabled) MPI_Barrier(MPI_COMM_WORLD); 

    // Solve
    AMGX_solver_solve(solver, AmgXRHS, AmgXP);

    // Get the status of the solver
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(solver, &status);

    // Check whether the solver successfully solved the problem
    if (status != AMGX_SOLVE_SUCCESS)
    {
        fprintf(stderr, "AmgX solver failed to solve the system! "
                        "The error code is %d.\n",
                status);
    }

    // Download data from device
    AMGX_vector_download(AmgXP, psi);

    if (isMPIEnabled) MPI_Barrier(MPI_COMM_WORLD);

    // get norm and iteration number
    double irnorm = 0., rnorm = 0.;
    int nIters = 0;
    getResidual(0, irnorm);
    getIters(nIters);
    getResidual(nIters, rnorm);
    if (!isMPIEnabled || myRank == 0)
        printf("Initial residual = %.10lf, Final residual = %.5e, No Iterations %d\n", irnorm, rnorm, nIters);

}


/* \implements AmgXSolver::getIters */
void AmgXSolver::getIters(int &iter)
{
    // only processes using AmgX will try to get # of iterations
    AMGX_solver_get_iterations_number(solver, &iter);
}


/* \implements AmgXSolver::getResidual */
void AmgXSolver::getResidual(const int &iter, double &res)
{
    // only processes using AmgX will try to get residual
    AMGX_solver_get_iteration_residual(solver, iter, 0, &res);
}

