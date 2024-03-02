#pragma once

#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"

void init_const_coeff_ptr(std::vector<std::vector<double>>& nasa_coeffs, std::vector<std::vector<double>>& viscosity_coeffs,
        std::vector<std::vector<double>>& thermal_conductivity_coeffs, std::vector<std::vector<double>>& binary_diffusion_coeffs,
        std::vector<double>& molecular_weights);

class dfThermo
{
    dfMatrixDataBase &dataBase_;

    // private data members
    std::string mechanism_file;
    std::string thermo_coeff_file;

    // private member functions
    void readCoeffsBinary(FILE* fp, int dimension, std::vector<std::vector<double>>& coeffs);
    void initCoeffsfromBinaryFile(FILE* fp);

public:
    // cuda resource
    cudaStream_t stream;

    // public data members
    int num_species;
    int num_cells;

    // thermo coeffs
    std::vector<std::vector<double>> nasa_coeffs;
    std::vector<std::vector<double>> viscosity_coeffs;
    std::vector<std::vector<double>> thermal_conductivity_coeffs;
    std::vector<std::vector<double>> binary_diffusion_coeffs;
    std::vector<double> molecular_weights;

    // species info
    std::vector<double> mass_fraction;
    std::vector<double> mole_fraction;
    double meanMolecularWeight;

    double *d_mole_fraction, *d_mean_mole_weight;
    double *d_boundary_mole_fraction, *d_boundary_mean_mole_weight;

    // intermediate variables
    std::vector<double> T_poly;

    double *d_T_poly, *d_boundary_T_poly;
    double *d_species_viscosities, *d_boundary_species_viscosities;
    double *d_species_thermal_conductivities, *d_boundary_species_thermal_conductivities;
    double *d_psip0, *d_boundary_psip0;

    // constructor
    dfThermo(dfMatrixDataBase &dataBase)
        : dataBase_(dataBase) {};

    // destructor
    ~dfThermo(){};

    void cleanCudaResources();

    // public member functions
    void setConstantValue(std::string mechanism_file, int num_cells, int num_species);
    void setConstantFields(const std::vector<int> patch_type);
    void initNonConstantFields(const double *h_T, const double *h_he, const double *h_psi, const double *h_alpha, 
            const double *h_mu, const double *h_k, const double *h_dpdt, const double *h_rhoD, const double *h_boundary_T, 
            const double *h_boundary_he, const double *h_boundary_psi, const double *h_boundary_alpha, const double *h_boundary_mu, 
            const double *h_boundary_k, const double *h_boundary_rhoD);

    // set mass fraction
    void setMassFraction(const double *d_y, const double *d_boundary_y);

    // *** GPU functions ***
    void calculateTPolyGPU(int threads_per_block, int num_thread, int num_total, const double *T, double *T_poly, int offset = 0);
    void calculatePsiGPU(int threads_per_block, int num_thread, const double *T, const double *mean_mole_weight, 
            double *d_psi, int offset = 0);
    void calculateRhoGPU(int thread_per_block, int num_thread, const double *p, const double *psi, double *rho, int offset = 0);
    void calculateViscosityGPU(int num_thread, int num_total, const double *T, const double *mole_fraction,
            const double *T_poly, double *species_viscosities, double *viscosity, int offset = 0);
    void calculateThermoConductivityGPU(int thread_per_block, int num_thread, int num_total, const double *T, const double *T_poly,
            const double *d_y, const double *mole_fraction, double *species_thermal_conductivities,
            double *thermal_conductivity, int offset = 0);
    void calculateRhoDGPU(int threads_per_block, int num_thread, int num_total, const double *T, 
            const double *T_poly, const double *p, const double *mole_fraction, 
            const double *mean_mole_weight, const double *rho, double *rhoD, int offset = 0);
    void calculateEnthalpyGPU(int thread_per_block, int num_thread, int num_total, const double *T, double *enthalpy, const double *d_mass_fraction, int offset = 0);
    void calculateTemperatureGPU(int thread_per_block, int num_thread, int num_total, const double *T_init, const double *target_h, 
            double *T, const double *d_mass_fraction, int offset = 0,
            double atol = 1e-7, double rtol = 1e-7, int max_iter = 20);
    void calculateEnergyGradient(int num_thread, int num_cells, int num_species, 
            int num_boundary_surfaces, int bou_offset, int gradient_offset, const int *face2Cells, 
            const double *T, const double *p, const double *y, const double *boundary_delta_coeffs,
            const double *boundary_p, const double *boundary_y, double *boundary_thermo_gradient);
    void setPsip0(int thread_per_block, int num_thread, const double *p, const double *psi, double *psip0, int offset = 0);
    void addPsipRho(int thread_per_block, int num_thread, const double *p, const double *psi, const double *psip0, 
            double *rho, int offset = 0);
    void updateCPUT(double *h_T, double *h_boundary_T);

    void compareT(const double *T, const double *boundary_T, bool printFlag);
    void compareRho(const double *rho, const double *boundary_rho, bool printFlag);
    void comparePsi(const double *psi, const double *boundary_psi, bool printFlag);
    void compareMu(const double *mu, const double *boundary_mu, bool printFlag);
    void compareAlpha(const double *alpha, const double *boundary_alpha, bool printFlag);
    void compareHe(const double *he, const double *boundary_he, bool printFlag);
    void compareRhoD(const double *rhoD, const double *boundary_rhoD, int species_index, bool printFlag);

    void correctHe(const double *he, const double *boundary_he);
    void correctPsi(const double *psi, const double *boundary_psi);
    void correctAlpha(const double *alpha, const double *boundary_alpha);
    void correctMu(const double *mu, const double *boundary_mu);
    void correctRho(const double *rho, const double *boundary_rho);

    void sync();

    // outer API
    void updateEnergy();
    void correctThermo();
    void updateRho();
    void psip0();
    void correctPsipRho();

    // getter functions
};
