#include "dfThermo.H"
#include <filesystem>
#include <cmath>
#include <numeric>
#include <cassert>
#include <cstring>
#include "device_launch_parameters.h"

#define GAS_CANSTANT 8314.46261815324
#define SQRT8 2.8284271247461903
#define NUM_SPECIES 7

// constant memory
__constant__ __device__ double d_nasa_coeffs[NUM_SPECIES*15];
__constant__ __device__ double d_viscosity_coeffs[NUM_SPECIES*5];
__constant__ __device__ double d_thermal_conductivity_coeffs[NUM_SPECIES*5];
__constant__ __device__ double d_binary_diffusion_coeffs[NUM_SPECIES*NUM_SPECIES*5];
__constant__ __device__ double d_molecular_weights[NUM_SPECIES];
__constant__ __device__ double d_viscosity_conatant1[NUM_SPECIES*NUM_SPECIES];
__constant__ __device__ double d_viscosity_conatant2[NUM_SPECIES*NUM_SPECIES];

void init_const_coeff_ptr(std::vector<std::vector<double>>& nasa_coeffs, std::vector<std::vector<double>>& viscosity_coeffs,
        std::vector<std::vector<double>>& thermal_conductivity_coeffs, std::vector<std::vector<double>>& binary_diffusion_coeffs,
        std::vector<double>& molecular_weights)
{
    //double *d_tmp;
    //checkCudaErrors(cudaMalloc((void**)&d_tmp, sizeof(double) * NUM_SPECIES * 15));
    double nasa_coeffs_tmp[NUM_SPECIES*15];
    double viscosity_coeffs_tmp[NUM_SPECIES*5];
    double thermal_conductivity_coeffs_tmp[NUM_SPECIES*5];
    double binary_diffusion_coeffs_tmp[NUM_SPECIES*NUM_SPECIES*5];
    double viscosity_conatant1_tmp[NUM_SPECIES*NUM_SPECIES];
    double viscosity_conatant2_tmp[NUM_SPECIES*NUM_SPECIES];

    for (int i = 0; i < NUM_SPECIES; i++) {
        std::copy(nasa_coeffs[i].begin(), nasa_coeffs[i].end(), nasa_coeffs_tmp + i * 15);
        std::copy(viscosity_coeffs[i].begin(), viscosity_coeffs[i].end(), viscosity_coeffs_tmp + i * 5);
        std::copy(thermal_conductivity_coeffs[i].begin(), thermal_conductivity_coeffs[i].end(), thermal_conductivity_coeffs_tmp + i * 5);
        std::copy(binary_diffusion_coeffs[i].begin(), binary_diffusion_coeffs[i].end(), binary_diffusion_coeffs_tmp + i * 5 * NUM_SPECIES);
        for (int j = 0; j < NUM_SPECIES; j++) {
            viscosity_conatant1_tmp[i * NUM_SPECIES + j] = pow((1 + molecular_weights[i] / molecular_weights[j]), -0.5);
            viscosity_conatant2_tmp[i * NUM_SPECIES + j] = pow(molecular_weights[j] / molecular_weights[i], 0.25);
        }
    }
    checkCudaErrors(cudaMemcpyToSymbol(d_nasa_coeffs, nasa_coeffs_tmp, sizeof(double) * 15 * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_viscosity_coeffs, viscosity_coeffs_tmp, sizeof(double) * 5 * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_thermal_conductivity_coeffs, thermal_conductivity_coeffs_tmp, sizeof(double) * 5 * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_binary_diffusion_coeffs, binary_diffusion_coeffs_tmp, sizeof(double) * 5 * NUM_SPECIES * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_molecular_weights, molecular_weights.data(), sizeof(double) * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_viscosity_conatant1, viscosity_conatant1_tmp, sizeof(double) * NUM_SPECIES * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_viscosity_conatant2, viscosity_conatant2_tmp, sizeof(double) * NUM_SPECIES * NUM_SPECIES));
}

__global__ void get_mole_fraction_mean_mole_weight(int num_cells, int num_species, const double *d_Y, 
        double *mole_fraction, double *mean_mole_weight)
{   
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double sum = 0.;
    for (int i = 0; i < num_species; i++) {
        sum += d_Y[i * num_cells + index] / d_molecular_weights[i];
    }
    double meanMoleWeight = 0.;
    for (int i = 0; i < num_species; i++) {
        mole_fraction[i * num_cells + index] = d_Y[i * num_cells + index] / (d_molecular_weights[i] * sum);
        meanMoleWeight += mole_fraction[i * num_cells + index] * d_molecular_weights[i];
    }
    mean_mole_weight[index] = meanMoleWeight;
}

__global__ void calculate_TPoly_kernel(int num_thread, int num_total, const double *T, double *d_T_poly, int offset)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;
    
    int startIndex = index + offset;
    
    d_T_poly[num_total * 0 + startIndex] = 1.0;
    d_T_poly[num_total * 1 + startIndex] = log(T[startIndex]);
    d_T_poly[num_total * 2 + startIndex] = d_T_poly[num_total * 1 + startIndex] * d_T_poly[num_total * 1 + startIndex];
    d_T_poly[num_total * 3 + startIndex] = d_T_poly[num_total * 1 + startIndex] * d_T_poly[num_total * 2 + startIndex];
    d_T_poly[num_total * 4 + startIndex] = d_T_poly[num_total * 2 + startIndex] * d_T_poly[num_total * 2 + startIndex];
}

__global__ void calculate_psi_kernel(int num_cells, int offset, const double *T, const double *mean_mole_weight,
        double *psi)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    int startIndex = index + offset;
    
    psi[startIndex] = mean_mole_weight[startIndex] / (GAS_CANSTANT * T[startIndex]);
}

__global__ void calculate_rho_kernel(int num_thread, int offset, const double *p, const double *psi, double *rho)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;
    
    int startIndex = index + offset;
    
    rho[startIndex] = p[startIndex] * psi[startIndex];
}

__global__ void calculate_viscosity_kernel(int num_thread, int num_total, int num_species, int offset,
        const double *T_poly, const double *T, const double *mole_fraction,
        double *species_viscosities, double *mu)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;
    
    extern __shared__ double sdata[];
    double* sv = sdata;
    double* mf = &sdata[blockDim.x * num_species];

    int startIndex = index + offset;

    double sqrt_local_T = sqrt(T[startIndex]);

    double poly[5];
    for (int j = 0; j < 5; j++) {
        poly[j] = T_poly[num_total * j + startIndex];
    }

    for (int i = 0; i < num_species; i++) {
        double dot_product = 0.;
        for (int j = 0; j < 5; j++) {
            dot_product += d_viscosity_coeffs[i * 5 + j] * poly[j];
        }
        sv[threadIdx.x * num_species + i] = dot_product;
        mf[threadIdx.x * num_species + i] = mole_fraction[num_total * i + startIndex];
    }
    double mu_mix = 0.;
    for (int i = 0; i < num_species; i++) {
        double sum = 0.;
        for (int j = 0; j < num_species; j++) {
            double temp = 1.0 + (sv[threadIdx.x * num_species + i] / sv[threadIdx.x * num_species + j]) *
                          d_viscosity_conatant2[i * NUM_SPECIES + j];
            sum += mf[threadIdx.x * num_species + j] / SQRT8 * d_viscosity_conatant1[i * NUM_SPECIES + j] * (temp * temp);
        }
        mu_mix += mf[threadIdx.x * num_species + i] * (sv[threadIdx.x * num_species + i] * sv[threadIdx.x * num_species + i]) / sum;
    }
    mu[startIndex] = mu_mix * sqrt_local_T;
}

__device__ double calculate_cp_device_kernel(int num_total, int num_species, int index, 
        const double local_T, const double *mass_fraction)
{   
    double cp = 0.;

    for (int i = 0; i < num_species; i++) {
        if (local_T > d_nasa_coeffs[i * 15 + 0]) {
            cp += mass_fraction[i * num_total + index] * (d_nasa_coeffs[i * 15 + 1] + d_nasa_coeffs[i * 15 + 2] * local_T + d_nasa_coeffs[i * 15 + 3] * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 4] * local_T * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 5] * local_T * local_T * local_T * local_T) * GAS_CANSTANT / d_molecular_weights[i];
        } else {
            cp += mass_fraction[i * num_total + index] * (d_nasa_coeffs[i * 15 + 8] + d_nasa_coeffs[i * 15 + 9] * local_T + d_nasa_coeffs[i * 15 + 10] * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 11] * local_T * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 12] * local_T * local_T * local_T * local_T) * GAS_CANSTANT / d_molecular_weights[i];
        }
    }
    return cp;
}

__global__ void calculate_thermoConductivity_kernel(int num_thread, int num_total, int num_species, 
        int offset, const double *nasa_coeffs, const double *mass_fraction,
        const double *T_poly, const double *T, const double *mole_fraction,
        double *species_thermal_conductivities, double *thermal_conductivity)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;

    int startIndex = offset + index;

    double dot_product;
    double local_T = T[startIndex];

    for (int i = 0; i < num_species; i++) {
        dot_product = 0.;
        for (int j = 0; j < 5; j++) {
            dot_product += d_thermal_conductivity_coeffs[i * 5 + j] * T_poly[num_total * j + startIndex];
        }
        species_thermal_conductivities[i * num_total + startIndex] = dot_product * sqrt(local_T);
    }

    double sum_conductivity = 0.;
    double sum_inv_conductivity = 0.;

    for (int i = 0; i < num_species; i++) {
        sum_conductivity += mole_fraction[num_total * i + startIndex] * species_thermal_conductivities[i * num_total + startIndex];
        sum_inv_conductivity += mole_fraction[num_total * i + startIndex] / species_thermal_conductivities[i * num_total + startIndex];
    }
    double lambda_mix = 0.5 * (sum_conductivity + 1.0 / sum_inv_conductivity);

    double cp = calculate_cp_device_kernel(num_total, num_species, startIndex, local_T, mass_fraction);

    thermal_conductivity[startIndex] = lambda_mix / cp;
}

__global__ void calculate_diffusion_kernel(int num_thread, int num_total, int num_species,
        int offset, const double *T_poly, const double *mole_fraction, const double *p,
        const double *mean_mole_weight, const double *rho, const double *T, double *d)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;
    
    extern __shared__ double shared_data[];
    double *mole_fraction_shared = shared_data;
    
    int startIndex = offset + index;

    for (int i = 0; i < num_species; i++) {
        mole_fraction_shared[i * blockDim.x + threadIdx.x] = mole_fraction[i * num_total + startIndex];
    }

    double poly[5];
    for (int j = 0; j < 5; j++) {
        poly[j] = T_poly[num_total * j + startIndex];
    }
    
    double powT = T[startIndex] * sqrt(T[startIndex]);

    double local_mean_mole_weight = mean_mole_weight[startIndex];
    double local_rho_div_p = rho[startIndex] / p[startIndex];
    for (int i = 0; i < num_species; i++) {
        if (mole_fraction_shared[i * blockDim.x + threadIdx.x] + 1e-10 > 1.) {
            d[num_total * i + startIndex] = 0.;
            continue;
        }
        double sum1 = 0.;
        double sum2 = 0.;
        for (int j = 0; j < num_species; j++) {
            if (i == j) continue;
            // calculate D
            double tmp = 0.;
            for (int k = 0; k < 5; k++)
                tmp += (d_binary_diffusion_coeffs[i * num_species * 5 + j * 5 + k] * poly[k]);
            double local_D = tmp * powT;
            sum1 += mole_fraction_shared[j * blockDim.x + threadIdx.x] / local_D;
            sum2 += mole_fraction_shared[j * blockDim.x + threadIdx.x] * d_molecular_weights[j] / local_D;
        }
        sum2 *= mole_fraction_shared[i * blockDim.x + threadIdx.x] / 
                (local_mean_mole_weight - mole_fraction_shared[i * blockDim.x + threadIdx.x] * d_molecular_weights[i]);
        d[num_total * i + startIndex] = 1 / (sum1 + sum2) * local_rho_div_p;
    }
}

__device__ double calculate_enthalpy_device_kernel(int num_total, int num_species, int index, const double local_T,
        const double *mass_fraction)
{
    double h = 0.;

    for (int i = 0; i < num_species; i++) {
        if (local_T > d_nasa_coeffs[i * 15 + 0]) {
            h += (d_nasa_coeffs[i * 15 + 1] + d_nasa_coeffs[i * 15 + 2] * local_T / 2 + d_nasa_coeffs[i * 15 + 3] * local_T * local_T / 3 + 
                    d_nasa_coeffs[i * 15 + 4] * local_T * local_T * local_T / 4 + d_nasa_coeffs[i * 15 + 5] * local_T * local_T * local_T * local_T / 5 + 
                    d_nasa_coeffs[i * 15 + 6] / local_T) * GAS_CANSTANT * local_T / d_molecular_weights[i] * mass_fraction[i * num_total + index];
        } else {
            h += (d_nasa_coeffs[i * 15 + 8] + d_nasa_coeffs[i * 15 + 9] * local_T / 2 + d_nasa_coeffs[i * 15 + 10] * local_T * local_T / 3 + 
                    d_nasa_coeffs[i * 15 + 11] * local_T * local_T * local_T / 4 + d_nasa_coeffs[i * 15 + 12] * local_T * local_T * local_T * local_T / 5 + 
                    d_nasa_coeffs[i * 15 + 13] / local_T) * GAS_CANSTANT * local_T / d_molecular_weights[i] * mass_fraction[i * num_total + index];
        }
    }
    return h;
}

__global__ void calculate_energy_gradient_kernel(int num_thread, int num_cells, int num_species, 
        int num_boundary_surfaces, int bou_offset, int gradient_offset,
        const int *face2Cells, const double *T, const double *p, const double *y,
        const double *boundary_p, const double *boundary_y, const double *boundary_delta_coeffs,
        double *boundary_energy_gradient)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;
    
    int bou_start_index = index + bou_offset;
    int gradient_start_index = index + gradient_offset;
    int cellIndex = face2Cells[bou_start_index];
    
    double local_T = T[cellIndex];
    double h_internal = calculate_enthalpy_device_kernel(num_cells, num_species, cellIndex, local_T, y);
    double h_boundary = calculate_enthalpy_device_kernel(num_boundary_surfaces, num_species, bou_start_index, local_T, boundary_y);
    boundary_energy_gradient[gradient_start_index] = (h_boundary - h_internal) * boundary_delta_coeffs[bou_start_index];
}

__global__ void calculate_temperature_kernel(int num_thread, int num_total, int num_species, int offset,
        const double *T_init, const double *h_target, const double *mass_fraction,
        double *T_est, double atol, double rtol, int max_iter)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;
    
    int startIndex = index + offset;

    double local_T = T_init[startIndex];
    double local_h_target = h_target[startIndex];
    double h, cp, delta_T;
    for (int n = 0; n < max_iter; ++n) {
        h = calculate_enthalpy_device_kernel(num_total, num_species, startIndex, local_T, mass_fraction);
        cp = calculate_cp_device_kernel(num_total, num_species, startIndex, local_T, mass_fraction);
        delta_T = (h - local_h_target) / cp;
        local_T -= delta_T;
        if (fabs(h - local_h_target) < atol || fabs(delta_T / local_T) < rtol) {
            break;
        }
    }
    
    T_est[startIndex] = local_T;
}

extern void __global__ correct_internal_boundary_field_scalar(int num, int offset,
        const double *vf_internal, const int *face2Cells, double *vf_boundary);

__global__ void calculate_enthalpy_kernel(int num_thread, int offset, int num_total, int num_species, 
        const double *T, const double *mass_fraction, double *enthalpy)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;
    
    int startIndex = index + offset;
    
    enthalpy[startIndex] = calculate_enthalpy_device_kernel(num_total, num_species, startIndex, T[startIndex], mass_fraction);
}

__global__ void calculate_psip0_kernel(int num_thread, int offset, const double *p, const double *psi, double *psip0)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;
    
    int startIndex = index + offset;
    
    psip0[startIndex] = p[startIndex] * psi[startIndex];
}

__global__ void add_psip_rho_kernel(int num_thread, int offset, const double *p, const double *psi, const double *psip0, double *rho)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;
    
    int startIndex = index + offset;

    rho[startIndex] += p[startIndex] * psi[startIndex] - psip0[startIndex];
}

void dfThermo::cleanCudaResources() {}

void dfThermo::setConstantValue(std::string mechanism_file, int num_cells, int num_species)
{
    this->mechanism_file = mechanism_file;
    this->num_cells = num_cells;
    this->num_species = num_species;
    // get thermo_coeff_file from mechanism_file
    std::string prefix = "thermo_";
    std::string suffix = ".txt";
    std::string baseName = std::filesystem::path(mechanism_file).stem().string();
    thermo_coeff_file = prefix + baseName + suffix;

    // check if thermo_coeff_file exists
    if (!std::filesystem::exists(thermo_coeff_file))
    {
        std::cout << "Thermo coefficient file does not exist!" << std::endl;
        exit(1);
    }

    // read binary file
    FILE *fp = NULL;
    char *c_thermo_file = new char[thermo_coeff_file.length() + 1];
    strcpy(c_thermo_file, thermo_coeff_file.c_str());

    fp = fopen(c_thermo_file, "rb+");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open input file: %s!\n", c_thermo_file);
        exit(EXIT_FAILURE);
    }

    fread(&num_species, sizeof(int), 1, fp);

    molecular_weights.resize(num_species);
    fread(molecular_weights.data(), sizeof(double), num_species, fp);

    mass_fraction.resize(num_species);
    mole_fraction.resize(num_species);

    initCoeffsfromBinaryFile(fp);

    stream = dataBase_.stream;
#ifndef STREAM_ALLOCATOR
    checkCudaErrors(cudaMalloc((void**)&d_mole_fraction, sizeof(double) * num_species * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_mole_fraction, sizeof(double) * num_species * dataBase_.num_boundary_surfaces));
    checkCudaErrors(cudaMalloc((void**)&d_mean_mole_weight, sizeof(double) * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_mean_mole_weight, sizeof(double) * dataBase_.num_boundary_surfaces));
    checkCudaErrors(cudaMalloc((void**)&d_T_poly, sizeof(double) * 5 * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_T_poly, sizeof(double) * 5 * dataBase_.num_boundary_surfaces));

    checkCudaErrors(cudaMalloc((void**)&d_species_viscosities, sizeof(double) * num_species * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_species_viscosities, sizeof(double) * num_species * dataBase_.num_boundary_surfaces));
    checkCudaErrors(cudaMalloc((void**)&d_species_thermal_conductivities, sizeof(double) * num_species * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_species_thermal_conductivities, sizeof(double) * num_species * dataBase_.num_boundary_surfaces));
#endif
    
    checkCudaErrors(cudaMalloc((void**)&d_psip0, sizeof(double) * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_psip0, sizeof(double) * dataBase_.num_boundary_surfaces));
    std::cout << "dfThermo initialized" << std::endl;
}

void dfThermo::readCoeffsBinary(FILE* fp, int dimension, std::vector<std::vector<double>>& coeffs)
{
    coeffs.resize(num_species);
    for (int i = 0; i < num_species; i++) {
        coeffs[i].resize(dimension);
        fread(coeffs[i].data(), sizeof(double), dimension, fp);
    }
}

void dfThermo::initCoeffsfromBinaryFile(FILE* fp)
{
    readCoeffsBinary(fp, 15, nasa_coeffs);
    readCoeffsBinary(fp, 5, viscosity_coeffs);
    readCoeffsBinary(fp, 5, thermal_conductivity_coeffs);
    readCoeffsBinary(fp, num_species * 5, binary_diffusion_coeffs);
}

void dfThermo::sync()
{
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void dfThermo::setMassFraction(const double *d_y, const double *d_boundary_y)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;

    get_mole_fraction_mean_mole_weight<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            d_y, d_mole_fraction, d_mean_mole_weight);
    
    blocks_per_grid = (dataBase_.num_boundary_surfaces + threads_per_block - 1) / threads_per_block;
    get_mole_fraction_mean_mole_weight<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.num_boundary_surfaces, num_species, 
            d_boundary_y, d_boundary_mole_fraction, d_boundary_mean_mole_weight);
}

void dfThermo::setConstantFields(const std::vector<int> patch_type) 
{
    dataBase_.patch_type_T = patch_type;
}

void dfThermo::initNonConstantFields(const double *h_T, const double *h_he, const double *h_psi, const double *h_alpha, 
        const double *h_mu, const double *h_k, const double *h_dpdt, const double *h_rhoD, const double *h_boundary_T, 
        const double *h_boundary_he, const double *h_boundary_psi, const double *h_boundary_alpha, const double *h_boundary_mu, 
        const double *h_boundary_k, const double *h_boundary_rhoD)
{
    checkCudaErrors(cudaMemcpy(dataBase_.d_T, h_T, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_he, h_he, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_thermo_psi, h_psi, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_thermo_alpha, h_alpha, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_mu, h_mu, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_k, h_k, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_dpdt, h_dpdt, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_thermo_rhoD, h_rhoD, dataBase_.cell_value_bytes * dataBase_.num_species, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_T, h_boundary_T, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_he, h_boundary_he, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_thermo_psi, h_boundary_psi, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_thermo_alpha, h_boundary_alpha, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_mu, h_boundary_mu, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_k, h_boundary_k, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_thermo_rhoD, h_boundary_rhoD, dataBase_.boundary_surface_value_bytes * dataBase_.num_species, cudaMemcpyHostToDevice));
}

void dfThermo::calculateTPolyGPU(int threads_per_block, int num_thread, int num_total, const double *T, double *T_poly, int offset)
{
    size_t blocks_per_grid = (num_thread + threads_per_block - 1) / threads_per_block;

    calculate_TPoly_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_thread, num_total, T, T_poly, offset);
}

void dfThermo::calculatePsiGPU(int threads_per_block, int num_thread, const double *T, const double *mean_mole_weight, 
        double *d_psi, int offset)
{
    size_t blocks_per_grid = (num_thread + threads_per_block - 1) / threads_per_block;
    calculate_psi_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_thread, offset, T, mean_mole_weight, d_psi);
}

void dfThermo::calculateRhoGPU(int threads_per_block, int num_thread, const double *p, const double *psi, double *rho, int offset)
{
    size_t blocks_per_grid = (num_thread + threads_per_block - 1) / threads_per_block;
    calculate_rho_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_thread, offset, p, psi, rho);
}

void dfThermo::calculateViscosityGPU(int num_thread, int num_total, const double *T, const double *mole_fraction,
        const double *T_poly, double *species_viscosities, double *viscosity, int offset)
{
    TICK_INIT_EVENT;
    TICK_START_EVENT;
    int threads_per_block = 32;
    size_t blocks_per_grid = (num_thread + threads_per_block - 1) / threads_per_block;
    size_t num_bytes_dyn_shm = threads_per_block * num_species * 2 * sizeof(double);
    calculate_viscosity_kernel<<<blocks_per_grid, threads_per_block, num_bytes_dyn_shm, stream>>>(num_thread, num_total, num_species, offset,
            T_poly, T, mole_fraction, species_viscosities, viscosity);
    TICK_END_EVENT(calculate_viscosity_kernel);
}

void dfThermo::calculateThermoConductivityGPU(int threads_per_block, int num_thread, int num_total, const double *T, 
        const double *T_poly, const double *d_y, const double *mole_fraction, double *species_thermal_conductivities,
        double *thermal_conductivity, int offset)
{
    size_t blocks_per_grid = (num_thread + threads_per_block - 1) / threads_per_block;
    calculate_thermoConductivity_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_thread, num_total, num_species, 
            offset, d_nasa_coeffs, d_y, T_poly, T, mole_fraction, species_thermal_conductivities, thermal_conductivity);
}

void dfThermo::calculateRhoDGPU(int threads_per_block, int num_thread, int num_total, const double *T,
        const double *T_poly, const double *p, const double *mole_fraction,
        const double *mean_mole_weight, const double *rho, double *rhoD, int offset)
{
    threads_per_block = 32;
    size_t blocks_per_grid = (num_thread + threads_per_block - 1) / threads_per_block;
    size_t sharedMemSize = sizeof(double) * threads_per_block * num_species;

    TICK_INIT_EVENT;
    TICK_START_EVENT;
    calculate_diffusion_kernel<<<blocks_per_grid, threads_per_block, sharedMemSize, stream>>>(num_thread, num_total, num_species, offset,
            T_poly, mole_fraction, p, mean_mole_weight, rho, T, rhoD);
    TICK_END_EVENT("calculate_diffusion_kernel");
}

void dfThermo::calculateTemperatureGPU(int threads_per_block, int num_thread, int num_total, const double *T_init, const double *target_h, double *T, 
        const double *d_mass_fraction, int offset, double atol, double rtol, int max_iter)
{
    size_t blocks_per_grid = (num_thread + threads_per_block - 1) / threads_per_block;

    calculate_temperature_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_thread, num_total, num_species, offset,
            T_init, target_h, d_mass_fraction, T, atol, rtol, max_iter);
}

void dfThermo::calculateEnthalpyGPU(int threads_per_block, int num_thread, int num_total, const double *T, double *enthalpy, const double *d_mass_fraction, int offset)
{
    size_t blocks_per_grid = (num_thread + threads_per_block - 1) / threads_per_block;

    calculate_enthalpy_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_thread, offset, num_total, num_species, 
            T, d_mass_fraction, enthalpy);
}

void dfThermo::updateEnergy()
{
    calculateEnthalpyGPU(1024, num_cells, num_cells, dataBase_.d_T, dataBase_.d_he, dataBase_.d_y);
    
    // int offset = 0;
    // for (int i = 0; i < dataBase_.num_patches; i++) {
    //     calculateEnthalpyGPU(dataBase_.patch_size[i], dataBase_.d_boundary_T + offset, dataBase_.d_he + offset, dataBase_.d_y, offset);
    //     if (dataBase_.patch_type_T[i] == boundaryConditions::processor) {
    //         offset += 2 * dataBase_.patch_size[i];
    //     } else {
    //         offset += dataBase_.patch_size[i];
    //     }
    // }
}

void dfThermo::correctThermo()
{
#ifdef STREAM_ALLOCATOR
    checkCudaErrors(cudaMallocAsync((void**)&d_mole_fraction, sizeof(double) * num_species * num_cells, stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_boundary_mole_fraction, sizeof(double) * num_species * dataBase_.num_boundary_surfaces, stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_mean_mole_weight, sizeof(double) * num_cells, stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_boundary_mean_mole_weight, sizeof(double) * dataBase_.num_boundary_surfaces, stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_T_poly, sizeof(double) * 5 * num_cells, stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_boundary_T_poly, sizeof(double) * 5 * dataBase_.num_boundary_surfaces, stream));

    checkCudaErrors(cudaMallocAsync((void**)&d_species_viscosities, sizeof(double) * num_species * num_cells, stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_boundary_species_viscosities, sizeof(double) * num_species * dataBase_.num_boundary_surfaces, stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_species_thermal_conductivities, sizeof(double) * num_species * num_cells, stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_boundary_species_thermal_conductivities, sizeof(double) * num_species * dataBase_.num_boundary_surfaces, stream));
#endif
 
    setMassFraction(dataBase_.d_y, dataBase_.d_boundary_y);
    // internal field
    int cell_thread = 512, boundary_thread = 32;
    fprintf(stderr, "\n\n");
    calculateTemperatureGPU(cell_thread, dataBase_.num_cells, dataBase_.num_cells, dataBase_.d_T, dataBase_.d_he, dataBase_.d_T, dataBase_.d_y); // calculate temperature
    calculateTPolyGPU(cell_thread, dataBase_.num_cells, dataBase_.num_cells, dataBase_.d_T, d_T_poly); // calculate T_poly
    calculatePsiGPU(cell_thread, dataBase_.num_cells, dataBase_.d_T, d_mean_mole_weight, dataBase_.d_thermo_psi); // calculate psi
    calculateRhoGPU(cell_thread, dataBase_.num_cells, dataBase_.d_p, dataBase_.d_thermo_psi, dataBase_.d_rho); // calculate rho
    calculateViscosityGPU(dataBase_.num_cells, dataBase_.num_cells, dataBase_.d_T, d_mole_fraction,
            d_T_poly, d_species_viscosities, dataBase_.d_mu); // calculate viscosity
    calculateThermoConductivityGPU(cell_thread, dataBase_.num_cells, dataBase_.num_cells, dataBase_.d_T, d_T_poly, dataBase_.d_y, d_mole_fraction, 
            d_species_thermal_conductivities, dataBase_.d_thermo_alpha); // calculate thermal conductivity
    calculateRhoDGPU(cell_thread, dataBase_.num_cells, dataBase_.num_cells, dataBase_.d_T, d_T_poly, dataBase_.d_p, d_mole_fraction, 
            d_mean_mole_weight, dataBase_.d_rho, dataBase_.d_thermo_rhoD); 
    fprintf(stderr, "\n\n");
    // boundary field
    int offset = 0;
    for (int i = 0; i < dataBase_.num_patches; i++) {
        if (dataBase_.patch_size[i] == 0) continue;
        if (dataBase_.patch_type_T[i] == boundaryConditions::fixedValue) {
            calculateTPolyGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.num_boundary_surfaces, dataBase_.d_boundary_T, d_boundary_T_poly, offset);
            calculateEnthalpyGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.num_boundary_surfaces, dataBase_.d_boundary_T, dataBase_.d_boundary_he, 
                    dataBase_.d_boundary_y, offset);
            calculatePsiGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.d_boundary_T, d_boundary_mean_mole_weight, dataBase_.d_boundary_thermo_psi, offset);
            calculateRhoGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.d_boundary_p, dataBase_.d_boundary_thermo_psi, dataBase_.d_boundary_rho, offset);
            calculateViscosityGPU(dataBase_.patch_size[i], dataBase_.num_boundary_surfaces, dataBase_.d_boundary_T, d_boundary_mole_fraction,
                    d_boundary_T_poly, d_boundary_species_viscosities, dataBase_.d_boundary_mu, offset);
            calculateThermoConductivityGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.num_boundary_surfaces, dataBase_.d_boundary_T, d_boundary_T_poly, 
                    dataBase_.d_boundary_y, d_boundary_mole_fraction, d_boundary_species_thermal_conductivities, dataBase_.d_boundary_thermo_alpha, offset);
            calculateRhoDGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.num_boundary_surfaces, dataBase_.d_boundary_T, d_boundary_T_poly,
                    dataBase_.d_boundary_p, d_boundary_mole_fraction, d_boundary_mean_mole_weight, dataBase_.d_boundary_rho, dataBase_.d_boundary_thermo_rhoD, offset);
        } else {
            calculateTemperatureGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.num_boundary_surfaces, dataBase_.d_boundary_T, dataBase_.d_boundary_he, 
                    dataBase_.d_boundary_T, dataBase_.d_boundary_y, offset);
            calculateTPolyGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.num_boundary_surfaces, dataBase_.d_boundary_T, d_boundary_T_poly, offset);
            calculatePsiGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.d_boundary_T, d_boundary_mean_mole_weight, dataBase_.d_boundary_thermo_psi, offset);
            calculateRhoGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.d_boundary_p, dataBase_.d_boundary_thermo_psi, dataBase_.d_boundary_rho, offset);
            calculateViscosityGPU(dataBase_.patch_size[i], dataBase_.num_boundary_surfaces, dataBase_.d_boundary_T, d_boundary_mole_fraction,
                    d_boundary_T_poly, d_boundary_species_viscosities, dataBase_.d_boundary_mu, offset);
            calculateThermoConductivityGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.num_boundary_surfaces, dataBase_.d_boundary_T, d_boundary_T_poly,
                    dataBase_.d_boundary_y, d_boundary_mole_fraction, d_boundary_species_thermal_conductivities, dataBase_.d_boundary_thermo_alpha, offset);
            calculateRhoDGPU(boundary_thread, dataBase_.patch_size[i], dataBase_.num_boundary_surfaces, dataBase_.d_boundary_T, d_boundary_T_poly,
                    dataBase_.d_boundary_p, d_boundary_mole_fraction, d_boundary_mean_mole_weight, dataBase_.d_boundary_rho, dataBase_.d_boundary_thermo_rhoD, offset);
        }
        // correct internal field of processor boundary
        if (dataBase_.patch_type_T[i] == boundaryConditions::processor
            || dataBase_.patch_type_T[i] == boundaryConditions::processorCyclic) {
            size_t threads_per_block = 32;
            size_t blocks_per_grid = (dataBase_.patch_size[i] + threads_per_block - 1) / threads_per_block;
            correct_internal_boundary_field_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.patch_size[i], offset,
                    dataBase_.d_T, dataBase_.d_boundary_face_cell, dataBase_.d_boundary_T);
            correct_internal_boundary_field_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.patch_size[i], offset,
                    dataBase_.d_he, dataBase_.d_boundary_face_cell, dataBase_.d_boundary_he);
            correct_internal_boundary_field_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.patch_size[i], offset,
                    dataBase_.d_thermo_psi, dataBase_.d_boundary_face_cell, dataBase_.d_boundary_thermo_psi);
            correct_internal_boundary_field_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.patch_size[i], offset,
                    dataBase_.d_thermo_alpha, dataBase_.d_boundary_face_cell, dataBase_.d_boundary_thermo_alpha);
            correct_internal_boundary_field_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.patch_size[i], offset,
                    dataBase_.d_mu, dataBase_.d_boundary_face_cell, dataBase_.d_boundary_mu);
            correct_internal_boundary_field_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.patch_size[i], offset,
                    dataBase_.d_rho, dataBase_.d_boundary_face_cell, dataBase_.d_boundary_rho);
            for (int j = 0; j < num_species; j++) {
                correct_internal_boundary_field_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.patch_size[i], offset,
                        dataBase_.d_thermo_rhoD + j * dataBase_.num_cells, dataBase_.d_boundary_face_cell, 
                        dataBase_.d_boundary_thermo_rhoD + j * dataBase_.num_boundary_surfaces);
            }
            offset += 2 * dataBase_.patch_size[i];
        } else {
            offset += dataBase_.patch_size[i]; }
    }
#ifdef STREAM_ALLOCATOR
    checkCudaErrors(cudaFreeAsync(d_mole_fraction, stream));
    checkCudaErrors(cudaFreeAsync(d_boundary_mole_fraction, stream));
    checkCudaErrors(cudaFreeAsync(d_mean_mole_weight, stream));
    checkCudaErrors(cudaFreeAsync(d_boundary_mean_mole_weight, stream));
    checkCudaErrors(cudaFreeAsync(d_T_poly, stream));
    checkCudaErrors(cudaFreeAsync(d_boundary_T_poly, stream));

    checkCudaErrors(cudaFreeAsync(d_species_viscosities, stream));
    checkCudaErrors(cudaFreeAsync(d_boundary_species_viscosities, stream));
    checkCudaErrors(cudaFreeAsync(d_species_thermal_conductivities, stream));
    checkCudaErrors(cudaFreeAsync(d_boundary_species_thermal_conductivities, stream));
#endif
}

void dfThermo::updateRho()
{
    int num_thread = 1024;
    calculateRhoGPU(num_thread, dataBase_.num_cells, dataBase_.d_p, dataBase_.d_thermo_psi, dataBase_.d_rho);
    calculateRhoGPU(num_thread, dataBase_.num_boundary_surfaces, dataBase_.d_boundary_p, 
            dataBase_.d_boundary_thermo_psi, dataBase_.d_boundary_rho);
}

void dfThermo::psip0()
{
    int num_thread = 1024;
    setPsip0(num_thread, dataBase_.num_cells, dataBase_.d_p, dataBase_.d_thermo_psi, d_psip0);
    setPsip0(num_thread, dataBase_.num_boundary_surfaces, dataBase_.d_boundary_p, 
            dataBase_.d_boundary_thermo_psi, d_boundary_psip0);
}

void dfThermo::correctPsipRho()
{
    int num_thread = 1024;
    addPsipRho(num_thread, dataBase_.num_cells, dataBase_.d_p, dataBase_.d_thermo_psi, d_psip0, dataBase_.d_rho);
    addPsipRho(num_thread, dataBase_.num_boundary_surfaces, dataBase_.d_boundary_p, 
            dataBase_.d_boundary_thermo_psi, d_boundary_psip0, dataBase_.d_boundary_rho);
}

void dfThermo::calculateEnergyGradient(int num_thread, int num_cells, int num_species, 
        int num_boundary_surfaces, int bou_offset, int gradient_offset, const int *face2Cells, 
        const double *T, const double *p, const double *y, const double *boundary_delta_coeffs,
        const double *boundary_p, const double *boundary_y, double *boundary_thermo_gradient)
{
    size_t threads_per_block = 256;
    size_t blocks_per_grid = (num_thread + threads_per_block - 1) / threads_per_block;
    calculate_energy_gradient_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_thread, num_cells, num_species, num_boundary_surfaces,
            bou_offset, gradient_offset, face2Cells, T, p, y, boundary_p, boundary_y, boundary_delta_coeffs, boundary_thermo_gradient);
}

void dfThermo::setPsip0(int thread_per_block, int num_thread, const double *p, const double *psi, double *psip0, int offset)
{
    size_t blocks_per_grid = (num_thread + thread_per_block - 1) / thread_per_block;
    
    calculate_psip0_kernel<<<blocks_per_grid, thread_per_block, 0, stream>>>(num_thread, offset, p, psi, psip0);
}

void dfThermo::addPsipRho(int thread_per_block, int num_thread, const double *p, const double *psi, const double *psip0, 
        double *rho, int offset)
{
    size_t blocks_per_grid = (num_thread + thread_per_block - 1) / thread_per_block;
    
    add_psip_rho_kernel<<<blocks_per_grid, thread_per_block, 0, stream>>>(num_thread, offset, p, psi, psip0, rho);
}

void dfThermo::updateCPUT(double *h_T, double *h_boundary_T)
{
    checkCudaErrors(cudaMemcpyAsync(h_T, dataBase_.d_T, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(h_boundary_T, dataBase_.d_boundary_T, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
    sync();
}

void dfThermo::compareT(const double *T, const double *boundary_T, bool printFlag)
{
    double *h_T = new double[dataBase_.num_cells];
    double *h_boundary_T = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_T, dataBase_.d_T, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_T, dataBase_.d_boundary_T, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_T\n");
    checkVectorEqual(dataBase_.num_cells, T, h_T, 1e-10, printFlag);
    fprintf(stderr, "check h_boundary_T\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_T, h_boundary_T, 1e-10, printFlag);

    delete h_T;
    delete h_boundary_T;
}

void dfThermo::compareHe(const double *he, const double *boundary_he, bool printFlag)
{
    double *h_he = new double[dataBase_.num_cells];
    double *h_boundary_he = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_he, dataBase_.d_he, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_he, dataBase_.d_boundary_he, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_he\n");
    checkVectorEqual(dataBase_.num_cells, he, h_he, 1e-10, printFlag);
    fprintf(stderr, "check h_boundary_he\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_he, h_boundary_he, 1e-10, printFlag);

    delete h_he;
    delete h_boundary_he;
}

void dfThermo::compareRho(const double *rho, const double *boundary_rho, bool printFlag)
{
    double *h_rho = new double[dataBase_.num_cells];
    double *h_boundary_rho = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_rho, dataBase_.d_rho, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_rho, dataBase_.d_boundary_rho, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_thermo_rho\n");
    checkVectorEqual(dataBase_.num_cells, rho, h_rho, 1e-10, printFlag);
    fprintf(stderr, "check h_thermo_boundary_rho\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_rho, h_boundary_rho, 1e-10, printFlag);

    delete h_rho;
    delete h_boundary_rho;
}

void dfThermo::comparePsi(const double *psi, const double *boundary_psi, bool printFlag)
{
    double *h_psi = new double[dataBase_.num_cells];
    double *h_boundary_psi = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_psi, dataBase_.d_thermo_psi, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_psi, dataBase_.d_boundary_thermo_psi, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_thermo_psi\n");
    checkVectorEqual(dataBase_.num_cells, psi, h_psi, 1e-10, printFlag);
    fprintf(stderr, "check h_thermo_boundary_psi\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_psi, h_boundary_psi, 1e-10, printFlag);

    delete h_psi;
    delete h_boundary_psi;
}

void dfThermo::compareMu(const double *mu, const double *boundary_mu, bool printFlag)
{
    double *h_mu = new double[dataBase_.num_cells];
    double *h_boundary_mu = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_mu, dataBase_.d_mu, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_mu, dataBase_.d_boundary_mu, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_thermo_mu\n");
    checkVectorEqual(dataBase_.num_cells, mu, h_mu, 1e-10, printFlag);
    fprintf(stderr, "check h_thermo_boundary_mu\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_mu, h_boundary_mu, 1e-10, printFlag);

    delete h_mu;
    delete h_boundary_mu;
}

void dfThermo::compareAlpha(const double *alpha, const double *boundary_alpha, bool printFlag)
{
    double *h_alpha = new double[dataBase_.num_cells];
    double *h_boundary_alpha = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_alpha, dataBase_.d_thermo_alpha, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_alpha, dataBase_.d_boundary_thermo_alpha, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_thermo_alpha\n");
    checkVectorEqual(dataBase_.num_cells, alpha, h_alpha, 1e-10, printFlag);
    fprintf(stderr, "check h_thermo_boundary_alpha\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_alpha, h_boundary_alpha, 1e-10, printFlag);

    delete h_alpha;
    delete h_boundary_alpha;
}

void dfThermo::compareRhoD(const double *rhoD, const double *boundary_rhoD, int species_index, bool printFlag)
{
    double *h_rhoD = new double[dataBase_.num_cells];
    double *h_boundary_rhoD = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_rhoD, dataBase_.d_thermo_rhoD + species_index * dataBase_.num_cells, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_rhoD, dataBase_.d_boundary_thermo_rhoD + species_index * dataBase_.num_boundary_surfaces, 
                dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_thermo_rhoD\n");
    checkVectorEqual(dataBase_.num_cells, rhoD, h_rhoD, 1e-10, printFlag);
    fprintf(stderr, "check h_thermo_boundary_rhoD\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_rhoD, h_boundary_rhoD, 1e-10, printFlag);

    delete h_rhoD;
    delete h_boundary_rhoD;
}

void dfThermo::correctHe(const double *he, const double *boundary_he)
{
    checkCudaErrors(cudaMemcpy(dataBase_.d_he, he, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_he, boundary_he, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
}

void dfThermo::correctRho(const double *rho, const double *boundary_rho)
{
    checkCudaErrors(cudaMemcpy(dataBase_.d_rho, rho, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_rho, boundary_rho, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
}

void dfThermo::correctPsi(const double *psi, const double *boundary_psi)
{
    checkCudaErrors(cudaMemcpy(dataBase_.d_thermo_psi, psi, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_thermo_psi, boundary_psi, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
}

void dfThermo::correctMu(const double *mu, const double *boundary_mu)
{
    checkCudaErrors(cudaMemcpy(dataBase_.d_mu, mu, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_mu, boundary_mu, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
}

void dfThermo::correctAlpha(const double *alpha, const double *boundary_alpha)
{
    checkCudaErrors(cudaMemcpy(dataBase_.d_thermo_alpha, alpha, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_thermo_alpha, boundary_alpha, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
}
