#pragma once

#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include "dfMatrixDataBase.H"

class dfChemistrySolver
{
private:
    dfMatrixDataBase &dataBase_;
    cudaStream_t stream;
    std::vector<torch::jit::script::Module> modules_;
    torch::Device device_;
    double *Xmu_, *Xstd_, *Ymu_, *Ystd_;

    double *init_input_, *y_input_BCT, *NN_output_; 
    int *d_reactCellIndex;
    int dim_input_, num_cells_, num_species_, num_modules_;
    int batch_size_;
    double unReactT_;
    int inputsize_;
public:
    dfChemistrySolver(dfMatrixDataBase &dataBase)
        : device_(torch::kCUDA), dataBase_(dataBase) {};
    ~dfChemistrySolver();

    void setConstantValue(int num_cells, int num_species, int batch_size);
    void loadModels(const std::string dir);
    void loadNormalization(const std::string dir);
    void Inference(const double *h_T, const double *d_T,const double *p, const double *y,
            const double *rho, double *RR);
    
    void sync();
};