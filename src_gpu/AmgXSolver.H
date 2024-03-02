/**
 * \file AmgXSolver.hpp
 * \brief Definition of class AmgXSolver.
 * \author Pi-Yueh Chuang (pychuang@gwu.edu)
 * \author Matt Martineau (mmartineau@nvidia.com)
 * \date 2015-09-01
 * \copyright Copyright (c) 2015-2019 Pi-Yueh Chuang, Lorena A. Barba.
 * \copyright Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *            This project is released under MIT License.
 */


#ifndef __AMGX_SOLVER_H__
#define __AMGX_SOLVER_H__

// CUDA
#include <cuda_runtime.h>

// STL
# include <string>
# include <vector>
# include <ctime>

// AmgX
# include <amgx_c.h>

// PETSc
// # include <petscvec.h>

// mpi
#include <mpi.h>

/** \brief A wrapper class for coupling PETSc and AmgX.
 *
 * This class is a wrapper of AmgX library for PETSc. PETSc users only need to
 * pass a PETSc matrix and vectors into an AmgXSolver instance to solve their
 * linear systems. The class is designed specifically for the situation where
 * the number of MPI processes is more than the number of GPU devices.
 *
 * Eaxmple usage:
 * \code
 * int main(int argc, char **argv)
 * {
 *     // initialize matrix A, RHS, etc using PETSc
 *     ...
 *
 *     // create an instance of the solver wrapper
 *     AmgXSolver    solver;
 *     // initialize the instance with communicator, executation mode, and config file
 *     solver.initialize(comm, mode, file);
 *     // set matrix A. Currently it only accept PETSc AIJ matrix
 *     solver.setA(A);
 *     // solve. x and rhs are PETSc vectors. unkns will be the final result in the end
 *     solver.solve(unks, rhs);
 *     // get number of iterations
 *     int         iters;
 *     solver.getIters(iters);
 *     // get residual at the last iteration
 *     double      res;
 *     solver.getResidual(iters, res);
 *     // finalization
 *     solver.finalize();
 *
 *     // other codes
 *     ....
 *
 *     return 0;
 * }
 * \endcode
 */
class AmgXSolver
{
    public:

        /** \brief Default constructor. */
        AmgXSolver() = default;

        /** \brief Construct a AmgXSolver instance.
         *
         * \param modeStr [in] A string; target mode of AmgX (e.g., dDDI).
         * \param cfgFile [in] A string; the path to AmgX configuration file.
         * \param devID [in] An integer; the ID of the GPU device to be used.
         */
        AmgXSolver
        (
            const std::string &modeStr,
            const std::string &cfgFile,
            int devID
        );

        /** \brief Destructor. */
        ~AmgXSolver();

        /** \brief Initialize a AmgXSolver instance.
         *
         * \param modeStr [in] A string; target mode of AmgX (e.g., dDDI).
         * \param cfgFile [in] A string; the path to AmgX configuration file.
         * \param devID [in] An integer; the ID of the GPU device to be used.
         *
         */
        void initialize
        (
            const std::string &modeStr,
            const std::string &cfgFile,
            const int devID
        );


        /** \brief Finalize this instance.
         *
         * This function destroys AmgX data. When there are more than one
         * AmgXSolver instances, the last one destroyed is also in charge of
         * destroying the shared resource object and finalizing AmgX.
         *
         */
        void finalize();

        /** \brief Set up the matrix used by AmgX.
         *
         * This function sets up the AmgX matrix from the provided CSR data
         * structures and partition data.
         *
         * \param nGlobalRows [in] The number of global rows.
         * \param nLocalRows [in] The number of local rows on this rank.
         * \param nLocalNz [in] The total number of non zero entries locally.
         * \param rowOffsets [in] The local CSR matrix row offsets.
         * \param colIndicesGlobal [in] The global CSR matrix column indices.
         * \param values [in] The local CSR matrix values.
         * id of the owning rank for each row.
         *
         */
        void setOperator
        (
            const int nRows,
            const int nGlobalRows,
            const int nNz,
            const int *rowIndex,
            const int *colIndex,
            const double *value
        );

        /** \brief Re-sets up an existing AmgX matrix.
         *
         * Replaces the matrix coefficients with the provided values and performs
         * a resetup for the AmgX matrix.
         *
         * \param nLocalRows [in] The number of local rows on this rank.
         * \param nLocalNz [in] The total number of non zero entries locally.
         * \param values [in] The local CSR matrix values.
         *
         */
        void updateOperator
        (
            const int nRows,
            const int nNz,
            const double *value
        );

        /** \brief Solve the linear system.
         *
         * \p p vector will be used as an initial guess and will be updated to the
         * solution by the end of solving.
         *
         * For cases that use more MPI processes than the number of GPUs, this
         * function will do data gathering before solving and data scattering
         * after the solving.
         *
         * \param nLocalRows [in] The number of rows owned by this rank.
         * \param pscalar [in, out] The unknown array.
         * \param bscalar [in] The RHS array.
         * \param matrix [in,out] The AmgX CSR matrix, A.
         *
         */
        void solve
        (
            int nRows,
            double* psi,
            const double* rhs
        );

	/** \brief Solve the linear system.
         *
         * \p p vector will be used as an initial guess and will be updated to the
         * solution by the end of solving.
         *
         * For cases that use more MPI processes than the number of GPUs, this
         * function will do data gathering before solving and data scattering
         * after the solving.
         *
         * \param nLocalRows [in] The number of rows owned by this rank.
         * \param p [in, out] The unknown vector.
         * \param b [in] The RHS vector.
         * \param matrix [in,out] The AmgX CSR matrix, A.
         *
         */
        // void solve
        // (
        //     int nLocalRows,
        //     Vec& p,
        //     Vec& b,
        //     AmgXCSRMatrix& matrix
        // );


        /** \brief Get the number of iterations of the last solving.
         *
         * \param iter [out] Number of iterations.
         *
         */
        void getIters
        (
            int &iter
        );

        /** \brief Get the residual at a specific iteration during the last solving.
         *
         * \param iter [in] Target iteration.
         * \param res [out] Returned residual.
         *
         */
        void getResidual
        (
            const int &iter,
            double &res
        );


    private:

        /** \brief Current count of AmgXSolver instances.
         *
         * This static variable is used to count the number of instances. The
         * fisrt instance is responsable for initializing AmgX library and the
         * resource instance.
         */
        static int              count;

        /** \brief A flag indicating if this instance has been initialized. */
        bool                    isInitialised = false;

        /** \brief A flag indicating whether MPI is enabled */
        int                     isMPIEnabled;
        
        /** \brief A parameter used by AmgX. */
        int                     ring;

        /** \brief rank size in global communicator */
        int                     mpiSize;

        /** \brief rank size in global communicator */
        MPI_Comm                mpiWorld;

        /** \brief AmgX solver mode. */
        AMGX_Mode               mode;

        /** \brief AmgX config object. */
        AMGX_config_handle      cfg = nullptr;

        /** \brief AmgX matrix object. */
        AMGX_matrix_handle      AmgXA = nullptr;

        /** \brief AmgX vector object representing unknowns. */
        AMGX_vector_handle      AmgXP = nullptr;

        /** \brief AmgX vector object representing RHS. */
        AMGX_vector_handle      AmgXRHS = nullptr;

        /** \brief AmgX solver object. */
        AMGX_solver_handle      solver = nullptr;

        /** \brief AmgX resource object.
         *
         * Due to the design of AmgX library, using more than one resource
         * instance may cause some problems. So we make the resource instance
         * as a static member to keep only one instance.
         */
        static AMGX_resources_handle   rsrc;

        /** \brief Set AmgX solver mode based on the user-provided string.
         *
         * Available modes are: dDDI, dDFI, dFFI, hDDI, hDFI, hFFI.
         *
         * \param modeStr [in] a std::string.
         */
        void setMode(const std::string &modeStr);

        /** \brief Perform necessary initialization of AmgX.
         *
         * This function initializes AmgX for current instance. Based on
         * \ref AmgXSolver::count "count", only the instance initialized first
         * is in charge of initializing AmgX and the resource instance.
         *
         * \param cfgFile [in] Path to AmgX solver configuration file.
         * \param devID [in] ID of the GPU device to be used.
         */
        void initAmgX(const std::string &cfgFile, int devID);
};

#endif

