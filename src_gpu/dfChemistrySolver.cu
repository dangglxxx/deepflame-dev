#include "dfChemistrySolver.H"
#include "dfMatrixOpBase.H"

__global__ void construct_init_input(int num_thread, int num_cells, int dim, const double *T, const double *p,
        const int *reactCellIndex, const double *y, double *y_input_BCT, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;

    int cellIndex = reactCellIndex[index];

    output[index * dim] = T[cellIndex];
    // output[index * dim + 1] = p[cellIndex];
    output[index * dim + 1] = 101325.;
    double y_BCT;
    for (int i = 0; i < dim - 2; ++i) {
        y_BCT = (pow(y[i * num_cells + cellIndex], 0.1) - 1) * 10; // BCT: lambda = 0.1
        output[index * dim + 2 + i] = y_BCT;
        y_input_BCT[i * num_thread + index] = y_BCT;
    }
}

__global__ void normalize_input(int num_thread, int num_cells, int dim, const double *input, 
        const double *Xmu, const double *Xstd, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;

    for (int i = 0; i < dim; ++i) {
        output[index * dim + i] = (input[index * dim + i] - Xmu[i]) / Xstd[i];
    }
}

__global__ void calculate_y_new(int num_thread, int num_modules, const double *output_init, 
        const double *y_input_BCT, const double *Ymu, const double *Ystd, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;
    
    double RR_tmp;
    for (int i = 0; i < num_modules; ++i) {
        RR_tmp = output_init[i * num_thread + index] * Ystd[i] + Ymu[i] + y_input_BCT[i * num_thread + index];
        RR_tmp = pow((RR_tmp * 0.1 + 1), 10); // rev-BCT: lambda = 0.1
        output[i * num_thread + index] = RR_tmp;
    }
}

__global__ void calculate_RR(int num_thread, int num_cells, int num_species, double delta_t,
        const int *reactCellIndex, const double *rho, const double *y_old, const double *p, 
        double *y_NN, double *RR)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_thread)
        return;

    int cellIndex = reactCellIndex[index];
    
    // normalize
    double y_ave = 0.;
    for (int i = 0; i < num_species - 1; ++i) {
        y_ave += y_NN[i * num_thread + index];
    }
    y_ave += y_old[(num_species - 1) * num_cells + cellIndex];
    for (int i = 0; i < num_species - 1; ++i) {
        y_NN[i * num_thread + index] = y_NN[i * num_thread + index] / y_ave;
        RR[i * num_cells + cellIndex] = (y_NN[i * num_thread + index] - y_old[i * num_cells + cellIndex]) * rho[cellIndex]
                * (p[cellIndex] / 101325.) / delta_t; // correction
    }
}

dfChemistrySolver::~dfChemistrySolver() {
    cudaFree(init_input_);
}

void dfChemistrySolver::setConstantValue(int num_cells, int num_species, int batch_size) {
    this->num_cells_ = num_cells;
    this->num_species_ = num_species;
    this->batch_size_ = batch_size;
    this->stream = dataBase_.stream;

    dim_input_ = num_species + 2; // p, T, y
    num_modules_ = num_species_ - 1;
    unReactT_ = 610;
    cudaMalloc(&Xmu_, sizeof(double) * dim_input_);
    cudaMalloc(&Xstd_, sizeof(double) * dim_input_);
    cudaMalloc(&Ymu_, sizeof(double) * num_modules_);
    cudaMalloc(&Ystd_, sizeof(double) * num_modules_);
    modules_.reserve(num_modules_);

    // now norm paras are set in constructor manually
    at::TensorOptions opts = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA);
    std::vector<double> Xmu_vec = {1.2996375154e+03,  1.4349643303e+05, -4.3678815323e+00,
            -5.8949183472e+00, -3.8840763486e+00, -5.5436246211e+00,
            -6.0178199636e+00, -2.1469850084e+00, -6.9828365432e+00,
            -7.7747568654e+00, -1.8571483828e-01};
    std::vector<double> Xstd_vec = {3.9612732767e+02, 1.8822821412e+04, 1.1226048640e+00, 6.8397462420e-01,
            1.8879462146e+00, 1.2433158499e+00, 1.3169176600e+00, 4.3600457243e-01,
            8.1820904505e-01, 8.0471805333e-01, 6.1020187522e-02};
    std::vector<double> Ymu_vec = {-0.0101101322, -0.0138129078, -0.0146349442, -0.0088870325,
            -0.0075195178,  0.0020506931, -0.0103104668, -0.0192603020};
    std::vector<double> Ystd_vec = {0.0297933161, 0.0802139099, 0.0230954310, 0.1541940427, 
            0.1316836678, 0.0042975580, 0.1476416977, 0.0860471308};
    
    cudaMemcpy(Xmu_, Xmu_vec.data(), sizeof(double) * dim_input_, cudaMemcpyHostToDevice);
    cudaMemcpy(Xstd_, Xstd_vec.data(), sizeof(double) * dim_input_, cudaMemcpyHostToDevice);
    cudaMemcpy(Ymu_, Ymu_vec.data(), sizeof(double) * num_modules_, cudaMemcpyHostToDevice);
    cudaMemcpy(Ystd_, Ystd_vec.data(), sizeof(double) * num_modules_, cudaMemcpyHostToDevice);

    // input modules
    std::string prefix = "new_Temporary_Chemical_";
    std::string suffix = ".pt";
    for (int i = 0; i < num_modules_; ++i) {
        std::string model_path = prefix + std::to_string(i) + suffix;
        try {
            modules_.push_back(torch::jit::load(model_path));
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            exit(-1);
        }
        // modules_[i].to(device_);
        modules_[i].to(device_, torch::kHalf);
    }
}

void dfChemistrySolver::Inference(const double *h_T, const double *d_T,const double *p, const double *y,
        const double *rho, double *RR) {
    // construct input
    clock_t start = clock();
    inputsize_ = 0;
    std::vector<int> reactCellIndex;
    for (int i = 0; i < num_cells_; i++) {
        if (h_T[i] >= unReactT_) {
            reactCellIndex.push_back(i);
        }
    }
    inputsize_ = reactCellIndex.size();
    clock_t end = clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "construct input time: " << elapsed_secs << std::endl;

#ifdef STREAM_ALLOCATOR
    checkCudaErrors(cudaMallocAsync((void**)&init_input_, sizeof(double) * inputsize_ * dim_input_, stream));
    checkCudaErrors(cudaMallocAsync((void**)&y_input_BCT, sizeof(double) * inputsize_ * num_species_, stream));
    checkCudaErrors(cudaMallocAsync((void**)&NN_output_, sizeof(double) * inputsize_ * num_species_, stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_reactCellIndex, sizeof(int) * inputsize_, stream));
    checkCudaErrors(cudaMemcpyAsync(d_reactCellIndex, reactCellIndex.data(), sizeof(int) * inputsize_, cudaMemcpyHostToDevice, stream));
#else
    cudaMalloc(&init_input_, sizeof(double) * inputsize_ * dim_input_);
    cudaMalloc(&y_input_BCT, sizeof(double) * inputsize_ * num_species_);
    cudaMalloc(&d_reactCellIndex, sizeof(int) * inputsize_);
    cudaMemcpy(d_reactCellIndex, reactCellIndex.data(), sizeof(int) * inputsize_, cudaMemcpyHostToDevice);
#endif
    // construct input
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (inputsize_ + threads_per_block - 1) / threads_per_block;
    construct_init_input<<<blocks_per_grid, threads_per_block, 0, stream>>>(inputsize_, num_cells_, dim_input_, d_T, p, 
            d_reactCellIndex, y, y_input_BCT, init_input_);
    normalize_input<<<blocks_per_grid, threads_per_block, 0, stream>>>(inputsize_, num_cells_, dim_input_, init_input_, 
            Xmu_, Xstd_, init_input_);

    // inference by torch
    TICK_INIT_EVENT;
    TICK_START_EVENT;
    double *d_output;
    for (int sample_start = 0; sample_start < inputsize_; sample_start += batch_size_) {
        int sample_end = std::min(sample_start + batch_size_, inputsize_);
        int sample_len = sample_end - sample_start;
        at::Tensor torch_input = torch::from_blob(init_input_ + sample_start * dim_input_, {sample_len, dim_input_}, 
                torch::TensorOptions().device(device_).dtype(torch::kDouble));
        // torch_input = torch_input.to(at::kFloat);
        torch_input = torch_input.to(at::kHalf);
        std::vector<torch::jit::IValue> INPUTS;
        INPUTS.push_back(torch_input);
        std::vector<at::Tensor> output(num_modules_);

        for (int i = 0; i < num_modules_; ++i) {
            output[i] = modules_[i].forward(INPUTS).toTensor();
            output[i] = output[i].to(at::kDouble);
            d_output = output[i].data_ptr<double>();
            cudaMemcpy(NN_output_ + (i * inputsize_ + sample_start), d_output, sizeof(double) * sample_len, cudaMemcpyDeviceToDevice);
        }
    }
    TICK_END_EVENT(Inference);

    calculate_y_new<<<blocks_per_grid, threads_per_block, 0, stream>>>(inputsize_, num_modules_, NN_output_, 
            y_input_BCT, Ymu_, Ystd_, NN_output_);
    calculate_RR<<<blocks_per_grid, threads_per_block, 0, stream>>>(inputsize_, num_cells_, num_species_, 1e-6, 
            d_reactCellIndex, rho, y, p, NN_output_, RR);

#ifdef STREAM_ALLOCATOR
    checkCudaErrors(cudaFreeAsync(init_input_, dataBase_.stream));
    checkCudaErrors(cudaFreeAsync(y_input_BCT, dataBase_.stream));
    checkCudaErrors(cudaFreeAsync(NN_output_, dataBase_.stream));
    checkCudaErrors(cudaFreeAsync(d_reactCellIndex, dataBase_.stream)); 
#else
    cudaFree(init_input_);
    cudaFree(y_input_BCT);
    cudaFree(NN_output_);
    cudaFree(d_reactCellIndex);
#endif

}

void dfChemistrySolver::sync() {
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));
}