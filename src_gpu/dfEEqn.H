#pragma once

#include "AmgXSolver.H"
#include <amgx_c.h>
#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"
#include "dfThermo.H"

class dfEEqn
{
    dfMatrixDataBase &dataBase_;
    dfThermo &thermo_;

    // cuda resource
    cudaStream_t stream;
#ifdef USE_GRAPH
    // one graph for one eqn before using self-developed solver
    cudaGraph_t graph_pre, graph_post;
    cudaGraphExec_t graph_instance_pre, graph_instance_post;
    bool pre_graph_created=false;
    bool post_graph_created=false;
#endif

	// constant values -- basic
	std::string mode_string;
	std::string setting_path;

	// constant values -- amgx solvers
    AmgXSolver *ESolver = nullptr;
    int num_iteration = 0;

    // constant fields - boundary
	std::vector<int> patch_type_he;
    std::vector<int> patch_type_k;

    // non-constant fields - internal
    // dpdt
    double *d_dpdt = nullptr;
    double *h_dpdt = nullptr;
    
    // non-constant fields - boundary
    // gradient
    double *d_boundary_heGradient = nullptr;
    double *h_boundary_heGradient = nullptr;
    int num_gradientEnergy_boundary_surfaces = 0;
    // boundary coeff fields
    double *d_value_internal_coeffs = nullptr;
	double *d_value_boundary_coeffs= nullptr;
	double *d_gradient_internal_coeffs= nullptr;
	double *d_gradient_boundary_coeffs= nullptr;

    // non-constant fields - ldu
    double *d_ldu = nullptr;
	double *d_lower = nullptr;
	double *d_upper = nullptr;
    double *d_extern = nullptr;
	double *d_diag = nullptr;
	double *d_source = nullptr;
	double *d_internal_coeffs = nullptr;
	double *d_boundary_coeffs = nullptr;

    // non-constant fields - csr
	double *d_A = nullptr;
	double *d_b = nullptr; // TODO: needless

    // field pointer map
    std::unordered_map<std::string, double*> fieldPointerMap;

public:
    // constructor
    dfEEqn(dfMatrixDataBase &dataBase, dfThermo & thermo)
        : dataBase_(dataBase), thermo_(thermo) {}

	// destructor
	~dfEEqn(){}

    // member function

    // getter function
    double* getFieldPointer(const char* fieldAlias, location loc, position pos);

	// initialization
	void setConstantValues(const std::string &mode_string, const std::string &setting_path); 
	void setConstantFields(const std::vector<int> patch_type_he, const std::vector<int> patch_type_k);
    void createNonConstantFieldsInternal();
    void createNonConstantFieldsBoundary();
    void createNonConstantLduAndCsrFields();
    void initNonConstantFields(const double *he, const double *boundary_he);

    void cleanCudaResources();

    // run equation
    // volScalarField diffAlphaD, volVectorField hDiffCorrFlux, 
    void preProcessForYEqn();
    void preProcess(const double *h_he, const double *h_k, const double *h_k_old, const double *h_dpdt, const double *h_boundary_k, const double *h_boundary_heGradient);
	void process();
	void postProcess(double *h_he, double *h_boundary_he);

    void eeqn_calculate_energy_gradient(dfThermo& GPUThermo, int num_cells, int num_species, 
            int num_boundary_surfaces, const int *face2Cells, double *T, double *p, double *y,
            int num_patches, const int *patch_size, const int *patch_type,
            const double *boundary_delta_coeffs, const double *boundary_p, const double* boundary_y, 
            double *boundary_thermo_gradient);

    void solve();
    void compareResult(const double *lower, const double *upper, const double *diag, const double *source, const double *internal_coeffs, const double *boundary_coeffs, bool printFlag);
    void compareHe(const double *he, const double *boundary_he, bool printFlag);
    void sync();

};
