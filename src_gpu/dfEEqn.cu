#include "dfEEqn.H"

double* dfEEqn::getFieldPointer(const char* fieldAlias, location loc, position pos) {
    char mergedName[256];
    if (pos == position::internal) {
        sprintf(mergedName, "%s_%s", (loc == location::cpu) ? "h" : "d", fieldAlias);
    } else if (pos == position::boundary) {
        sprintf(mergedName, "%s_boundary_%s", (loc == location::cpu) ? "h" : "d", fieldAlias);
    }

    double *pointer = nullptr;
    if (fieldPointerMap.find(std::string(mergedName)) != fieldPointerMap.end()) {
        pointer = fieldPointerMap[std::string(mergedName)];
    }
    if (pointer == nullptr) {
        fprintf(stderr, "Warning! getFieldPointer of %s returns nullptr!\n", mergedName);
    }
    //fprintf(stderr, "fieldAlias: %s, mergedName: %s, pointer: %p\n", fieldAlias, mergedName, pointer);

    return pointer;
}

void dfEEqn::setConstantValues(const std::string &mode_string, const std::string &setting_path) {
    this->stream = dataBase_.stream;
    this->mode_string = mode_string;
    this->setting_path = setting_path;
    ESolver = new AmgXSolver(mode_string, setting_path, dataBase_.localRank);
}

void dfEEqn::setConstantFields(const std::vector<int> patch_type_he, const std::vector<int> patch_type_k) {
    this->patch_type_he = patch_type_he;
    this->patch_type_k = patch_type_k;
    // calculate num_gradientEnergy_boundary_surfaces
    for (int i = 0; i < dataBase_.num_patches; i++) {
        if (patch_type_he[i] == boundaryConditions::gradientEnergy) {
            num_gradientEnergy_boundary_surfaces += dataBase_.patch_size[i];
        }
    }
}

void dfEEqn::createNonConstantFieldsInternal() {
#ifndef STREAM_ALLOCATOR
    // thermophysical fields
    checkCudaErrors(cudaMalloc((void**)&d_dpdt, dataBase_.cell_value_bytes));
    // boundary coeffs
    checkCudaErrors(cudaMalloc((void**)&d_value_internal_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_value_boundary_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_gradient_internal_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_gradient_boundary_coeffs, dataBase_.boundary_surface_value_bytes));
#endif
    // computed on CPU, used on GPU, need memcpyh2d
    checkCudaErrors(cudaMallocHost((void**)&h_dpdt, dataBase_.cell_value_bytes));

    // getter for h_dpdt
    fieldPointerMap["h_dpdt"] = h_dpdt;
}

void dfEEqn::createNonConstantFieldsBoundary() {
#ifndef STREAM_ALLOCATOR
    checkCudaErrors(cudaMalloc((void**)&d_boundary_heGradient, sizeof(double) * num_gradientEnergy_boundary_surfaces));
#endif
    // computed on CPU, used on GPU, need memcpyh2d
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_heGradient, sizeof(double) * num_gradientEnergy_boundary_surfaces));

    // getter for h_boundary_heGradient
    fieldPointerMap["h_boundary_heGradient"] = h_boundary_heGradient;
}


void dfEEqn::createNonConstantLduAndCsrFields() {
    checkCudaErrors(cudaMalloc((void**)&d_ldu, dataBase_.csr_value_bytes));
    d_lower = d_ldu;
    d_diag = d_ldu + dataBase_.num_surfaces;
    d_upper = d_ldu + dataBase_.num_cells + dataBase_.num_surfaces;
    d_extern = d_ldu + dataBase_.num_cells + 2 * dataBase_.num_surfaces;
#ifndef STREAM_ALLOCATOR
    checkCudaErrors(cudaMalloc((void**)&d_source, dataBase_.cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_A, dataBase_.csr_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_b, dataBase_.cell_value_bytes));
#endif
}

void dfEEqn::initNonConstantFields(const double *he, const double *boundary_he)
{
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_he, he, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_he, boundary_he, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
}

void dfEEqn::cleanCudaResources() {
#ifdef USE_GRAPH
    if (pre_graph_created) {
        checkCudaErrors(cudaGraphExecDestroy(graph_instance_pre));
        checkCudaErrors(cudaGraphDestroy(graph_pre));
    }
    if (post_graph_created) {
        checkCudaErrors(cudaGraphExecDestroy(graph_instance_post));
        checkCudaErrors(cudaGraphDestroy(graph_post));
    }
#endif
}

void dfEEqn::preProcess(const double *h_he, const double *h_k, const double *h_k_old, const double *h_dpdt, const double *h_boundary_k, const double *h_boundary_heGradient)
{
}

void dfEEqn::process() {
    TICK_INIT_EVENT;
    TICK_START_EVENT;
#ifdef USE_GRAPH
    if(!pre_graph_created) {
        DEBUG_TRACE;
        checkCudaErrors(cudaStreamBeginCapture(dataBase_.stream, cudaStreamCaptureModeGlobal));
#endif

#ifdef STREAM_ALLOCATOR
    // thermophysical fields
    checkCudaErrors(cudaMallocAsync((void**)&d_dpdt, dataBase_.cell_value_bytes, dataBase_.stream));
    // fiv weight fields
    //checkCudaErrors(cudaMallocAsync((void**)&d_phi_special_weight, dataBase_.cell_value_bytes, dataBase_.stream));
    // boundary coeffs
    checkCudaErrors(cudaMallocAsync((void**)&d_value_internal_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_value_boundary_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_gradient_internal_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_gradient_boundary_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
 
    checkCudaErrors(cudaMallocAsync((void**)&d_boundary_heGradient, sizeof(double) * num_gradientEnergy_boundary_surfaces, dataBase_.stream));

    checkCudaErrors(cudaMallocAsync((void**)&d_source, dataBase_.cell_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_internal_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_boundary_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_A, dataBase_.csr_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_b, dataBase_.cell_value_bytes, dataBase_.stream));
#endif
    // checkCudaErrors(cudaMemcpyAsync(dataBase_.d_k, dataBase_.h_k, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    // checkCudaErrors(cudaMemcpyAsync(dataBase_.d_k_old, dataBase_.h_k_old, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    // checkCudaErrors(cudaMemcpyAsync(d_dpdt, h_dpdt, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    // checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_k, dataBase_.h_boundary_k, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));

    checkCudaErrors(cudaMemsetAsync(d_ldu, 0, dataBase_.csr_value_bytes, dataBase_.stream)); // d_ldu contains d_lower, d_diag, and d_upper
    checkCudaErrors(cudaMemsetAsync(d_source, 0, dataBase_.cell_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_internal_coeffs, 0, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_boundary_coeffs, 0, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_A, 0, dataBase_.csr_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_b, 0, dataBase_.cell_value_bytes, dataBase_.stream));

    eeqn_calculate_energy_gradient(thermo_, dataBase_.num_cells, dataBase_.num_species, dataBase_.num_boundary_surfaces, 
            dataBase_.d_boundary_face_cell, dataBase_.d_T, dataBase_.d_p, dataBase_.d_y,
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_he.data(),
            dataBase_.d_boundary_delta_coeffs, dataBase_.d_boundary_p, dataBase_.d_boundary_y,
            d_boundary_heGradient);
    correct_boundary_conditions_scalar(dataBase_.stream, dataBase_.nccl_comm, dataBase_.neighbProcNo.data(),
            dataBase_.num_boundary_surfaces, dataBase_.num_patches, dataBase_.patch_size.data(),
            patch_type_he.data(), dataBase_.d_boundary_delta_coeffs, dataBase_.d_boundary_face_cell,
            dataBase_.d_he, dataBase_.d_boundary_he, dataBase_.cyclicNeighbor.data(), 
            dataBase_.patchSizeOffset.data(), dataBase_.d_boundary_weight,
            dataBase_.d_boundary_T, dataBase_.d_boundary_y, d_boundary_heGradient, &thermo_);
    update_boundary_coeffs_scalar(dataBase_.stream,
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_he.data(),
            dataBase_.d_boundary_delta_coeffs, dataBase_.d_boundary_he, dataBase_.d_boundary_weight,
            d_value_internal_coeffs, d_value_boundary_coeffs,
            d_gradient_internal_coeffs, d_gradient_boundary_coeffs, d_boundary_heGradient);
    fvm_ddt_vol_scalar_vol_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.rdelta_t, dataBase_.d_rho, dataBase_.d_rho_old, 
            dataBase_.d_he, dataBase_.d_volume, d_diag, d_source);
    // NOTE: fvm_div_scalar use d_phi_weight, which is computed in YEqn_GPU by compute_upwind_weight()
    // thus we need open YEqn_GPU before UEqn_GPU
    fvm_div_scalar(dataBase_.stream, dataBase_.num_surfaces, dataBase_.d_owner, dataBase_.d_neighbor,
            dataBase_.d_phi, dataBase_.d_phi_weight,
            d_lower, d_upper, d_diag, // end for internal
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_he.data(),
            dataBase_.d_boundary_phi,
            d_value_internal_coeffs, d_value_boundary_coeffs,
            d_internal_coeffs, d_boundary_coeffs, 1.);
    fvc_ddt_vol_scalar_vol_scalar(dataBase_.stream, dataBase_.num_cells,
            dataBase_.rdelta_t, dataBase_.d_rho, dataBase_.d_rho_old, dataBase_.d_k,
            dataBase_.d_k_old, dataBase_.d_volume, d_source, -1.);
    fvc_div_surface_scalar_vol_scalar(dataBase_.stream, dataBase_.num_surfaces, dataBase_.d_owner, dataBase_.d_neighbor, dataBase_.d_weight, 
            dataBase_.d_k, dataBase_.d_phi, d_source, // end for internal
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_k.data(), 
            dataBase_.d_boundary_face_cell, dataBase_.d_boundary_k, dataBase_.d_boundary_phi, -1);
    fvm_laplacian_scalar(dataBase_.stream, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, dataBase_.d_owner, dataBase_.d_neighbor,
            dataBase_.d_weight, dataBase_.d_mag_sf, dataBase_.d_delta_coeffs, dataBase_.d_thermo_alpha, 
            d_lower, d_upper, d_diag, // end for internal
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_he.data(), dataBase_.d_boundary_mag_sf, dataBase_.d_boundary_thermo_alpha,
            d_gradient_internal_coeffs, d_gradient_boundary_coeffs, d_internal_coeffs, d_boundary_coeffs, -1);
    fvc_div_cell_vector(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, 
            dataBase_.d_owner, dataBase_.d_neighbor, 
            dataBase_.d_weight, dataBase_.d_sf, dataBase_.d_hDiff_corr_flux, d_source,
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_he.data(), dataBase_.d_boundary_face_cell,
            dataBase_.d_boundary_weight, dataBase_.d_boundary_hDiff_corr_flux, dataBase_.d_boundary_sf, dataBase_.d_volume);
    fvc_to_source_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.d_volume, dataBase_.d_dpdt, d_source);
    fvc_to_source_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.d_volume, dataBase_.d_diff_alphaD, d_source, -1);
#ifndef DEBUG_CHECK_LDU
    ldu_to_csr_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
            dataBase_.num_Nz, dataBase_.d_boundary_face_cell, dataBase_.d_ldu_to_csr_index, dataBase_.num_patches,
            dataBase_.patch_size.data(), patch_type_he.data(), d_ldu, d_source, d_internal_coeffs, d_boundary_coeffs, d_A);
#endif
#ifdef USE_GRAPH
        checkCudaErrors(cudaStreamEndCapture(dataBase_.stream, &graph_pre));
        checkCudaErrors(cudaGraphInstantiate(&graph_instance_pre, graph_pre, NULL, NULL, 0));
        pre_graph_created = true;
    }
    DEBUG_TRACE;
    checkCudaErrors(cudaGraphLaunch(graph_instance_pre, dataBase_.stream));
#endif
    TICK_END_EVENT(EEqn assembly);

    TICK_START_EVENT;
#ifndef DEBUG_CHECK_LDU
    solve();
#endif
    TICK_END_EVENT(EEqn solve);

#ifdef USE_GRAPH
    if(!post_graph_created) {
        checkCudaErrors(cudaStreamBeginCapture(dataBase_.stream, cudaStreamCaptureModeGlobal));
#endif

        TICK_START_EVENT;
        correct_boundary_conditions_scalar(dataBase_.stream, dataBase_.nccl_comm, dataBase_.neighbProcNo.data(),
                dataBase_.num_boundary_surfaces, dataBase_.num_patches, dataBase_.patch_size.data(),
                patch_type_he.data(), dataBase_.d_boundary_delta_coeffs, dataBase_.d_boundary_face_cell,
                dataBase_.d_he, dataBase_.d_boundary_he, dataBase_.cyclicNeighbor.data(), 
                dataBase_.patchSizeOffset.data(), dataBase_.d_boundary_weight,
                dataBase_.d_boundary_T, dataBase_.d_boundary_y, d_boundary_heGradient, &thermo_);
        TICK_END_EVENT(EEqn post process correctBC);

        TICK_START_EVENT;
        // copy he to host
        // checkCudaErrors(cudaMemcpyAsync(dataBase_.h_he, dataBase_.d_he, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
        // checkCudaErrors(cudaMemcpyAsync(dataBase_.h_boundary_he, dataBase_.d_boundary_he, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
        TICK_END_EVENT(EEqn post process copy back);

        TICK_START_EVENT;
#ifdef STREAM_ALLOCATOR
        // thermophysical fields
        checkCudaErrors(cudaFreeAsync(d_dpdt, dataBase_.stream));
        // fiv weight fieldsFree
        //checkCudaErrors(cudaFreeAsync(d_phi_special_weight, dataBase_.stream));
        // boundary coeffs
        checkCudaErrors(cudaFreeAsync(d_value_internal_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_value_boundary_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_gradient_internal_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_gradient_boundary_coeffs, dataBase_.stream));
     
        checkCudaErrors(cudaFreeAsync(d_boundary_heGradient, dataBase_.stream));
    
        checkCudaErrors(cudaFreeAsync(d_source, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_internal_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_boundary_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_A, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_b, dataBase_.stream));
#endif
        TICK_END_EVENT(EEqn post process free);
#ifdef USE_GRAPH
        checkCudaErrors(cudaStreamEndCapture(dataBase_.stream, &graph_post));
        checkCudaErrors(cudaGraphInstantiate(&graph_instance_post, graph_post, NULL, NULL, 0));
        post_graph_created = true;
    }
    checkCudaErrors(cudaGraphLaunch(graph_instance_post, dataBase_.stream));
#endif
    sync();
}

void dfEEqn::eeqn_calculate_energy_gradient(dfThermo& GPUThermo, int num_cells, int num_species, 
        int num_boundary_surfaces, const int *face2Cells, double *T, double *p, double *y,
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_delta_coeffs, const double *boundary_p, const double* boundary_y, 
        double *boundary_thermo_gradient)
{
    int bou_offset = 0, gradient_offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        if (patch_type[i] == boundaryConditions::gradientEnergy) {
            GPUThermo.calculateEnergyGradient(patch_size[i], num_cells, num_species, num_boundary_surfaces, bou_offset, gradient_offset,
                    face2Cells, T, p, y, boundary_delta_coeffs, boundary_p, boundary_y, boundary_thermo_gradient);
            bou_offset += patch_size[i];
            gradient_offset += patch_size[i];
        } else if (patch_type[i] == boundaryConditions::processor
                    || patch_type[i] == boundaryConditions::processorCyclic) {
            bou_offset += 2 * patch_size[i];
        } else {
            bou_offset += patch_size[i];
        }
    }
}

// #if defined DEBUG_
void dfEEqn::compareResult(const double *lower, const double *upper, const double *diag, 
        const double *source, const double *internal_coeffs, const double *boundary_coeffs, bool printFlag)
{
    DEBUG_TRACE;
    std::vector<double> h_lower;
    h_lower.resize(dataBase_.num_surfaces);
    checkCudaErrors(cudaMemcpy(h_lower.data(), d_lower, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_lower\n");
    checkVectorEqual(dataBase_.num_surfaces, lower, h_lower.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_upper;
    h_upper.resize(dataBase_.num_surfaces);
    checkCudaErrors(cudaMemcpy(h_upper.data(), d_upper, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_upper\n");
    checkVectorEqual(dataBase_.num_surfaces, upper, h_upper.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_diag;
    h_diag.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_diag.data(), d_diag, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_diag\n");
    checkVectorEqual(dataBase_.num_cells, diag, h_diag.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_source;
    h_source.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_source.data(), d_source, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_source\n");
    checkVectorEqual(dataBase_.num_cells, source, h_source.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_internal_coeffs;
    h_internal_coeffs.resize(dataBase_.num_boundary_surfaces);
    checkCudaErrors(cudaMemcpy(h_internal_coeffs.data(), d_internal_coeffs, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_internal_coeffs\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, internal_coeffs, h_internal_coeffs.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_boundary_coeffs;
    h_boundary_coeffs.resize(dataBase_.num_boundary_surfaces);
    checkCudaErrors(cudaMemcpy(h_boundary_coeffs.data(), d_boundary_coeffs, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_boundary_coeffs\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_coeffs, h_boundary_coeffs.data(), 1e-14, printFlag);
    DEBUG_TRACE;
}

void dfEEqn::compareHe(const double *he, const double *boundary_he, bool printFlag)
{
    double *h_he = new double[dataBase_.num_cells];
    double *h_boundary_he = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_he, dataBase_.d_he, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_he, dataBase_.d_boundary_he, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_he\n");
    checkVectorEqual(dataBase_.num_cells, he, h_he, 1e-14, printFlag);
    fprintf(stderr, "check h_boundary_he\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_he, h_boundary_he, 1e-14, printFlag);
}
// #endif

void dfEEqn::sync()
{
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));
}

void dfEEqn::solve()
{
    dataBase_.solve(num_iteration, AMGXSetting::u_setting, d_A, dataBase_.d_he, d_source);
    num_iteration++;
}

void dfEEqn::postProcess(double *h_he, double *h_boundary_he) {}
