#include <iostream>
#include <vector>

#include "../common/utils.h"
#include "include/inplace_fnnls.h"
#include "include/inplace_fnnls_option0.h"

template<typename T>
std::vector<vector_t<T>> run_cpu(std::vector<matrix_t<T>> const& As, 
                                 std::vector<vector_t<T>> const& bs) {
    // init the results
    std::vector<vector_t<T>> result(As.size());
    for (auto& v: result)
        v = vector_t<T>::Zero();

    // compute
    for (unsigned int i=0; i<As.size(); i++) {
        cpubased_inplace_fnnls(As[i], bs[i], result[i]);
    }

    return result;
}

template<typename T>
__global__
void kernel_cpubased_inplace_fnnls(matrix_t<T> const* As, 
                          vector_t<T> const* bs,
                          vector_t<T>* results,
                          unsigned int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // compute per thread
        cpubased_inplace_fnnls(As[idx], bs[idx], results[idx]);
    }
}

template<typename T>
std::vector<vector_t<T>> run_gpu_option0(std::vector<matrix_t<T>> const& As,
                                         std::vector<vector_t<T>> const& bs) {
    std::cout << "*** running gpu option0 ***\n";
    std::vector<vector_t<T>> result(As.size());
    for (auto& v : result)
        v = vector_t<T>::Zero();

    // device ptrs
    matrix_t<T> *d_As;
    vector_t<T> *d_bs, *d_result;
    unsigned int n = As.size();

    // allocate on teh device
    cuda::cuda_malloc(d_As, n);
    cuda::cuda_malloc(d_bs, n);
    cuda::cuda_malloc(d_result, n);
    cuda::assert_if_error("\tcuda mallocs");

    // copy
    cuda::copy_to_dev(d_As, As);
    cuda::copy_to_dev(d_bs, bs);
    cuda::assert_if_error("\tcuda memcpy to device");

    {
        dim3 threads{10, 10};
        int blocks{n};
        gpu::option0::kernel_inplace_fnnls<T><<<blocks, threads>>>(d_As, d_bs, 
            d_result);
        cuda::assert_if_error("\tkernel inplace fnnls option0");
    }

    cuda::copy_to_host(result, d_result);
    cuda::assert_if_error("\tcuda memcpy back to host");

    cudaFree(d_As);
    cudaFree(d_bs);
    cudaFree(d_result);

    return result;
}

template<typename T>
std::vector<vector_t<T>> run_gpu_cpubased(std::vector<matrix_t<T>> const& As,
                                       std::vector<vector_t<T>> const& bs) {
    std::cout << "*** running gpu cpubased option ***\n";
    std::vector<vector_t<T>> result(As.size());
    for (auto& v : result)
        v = vector_t<T>::Zero();

    // device ptrs
    matrix_t<T> *d_As;
    vector_t<T> *d_bs, *d_result;
    unsigned int n = As.size();

    // allocate on teh device
    cuda::cuda_malloc(d_As, n);
    cuda::cuda_malloc(d_bs, n);
    cuda::cuda_malloc(d_result, n);
    cuda::assert_if_error("\tcuda mallocs");

    // copy
    cuda::copy_to_dev(d_As, As);
    cuda::copy_to_dev(d_bs, bs);
    cuda::assert_if_error("\tcuda memcpy to device");

    {
        int threads{256};
        int blocks{(n + threads - 1) / threads};
        kernel_cpubased_inplace_fnnls<T><<<blocks, threads>>>(d_As, d_bs, 
            d_result, n);
        cuda::assert_if_error("\tkernel cpubased inplace fnnls");
    }

    cuda::copy_to_host(result, d_result);
    cuda::assert_if_error("\tcuda memcpy back to host");

    cudaFree(d_As);
    cudaFree(d_bs);
    cudaFree(d_result);

    return result;
}

int main(int argc, char** argv) {
    if (argc<=1) {
        std::cout << "run with './main <number of matrices>'\n";
        exit(0);
    }
    unsigned int n = std::atoi(argv[1]);
    using type = float;
    
    // create the input matrices
    std::vector<matrix_t<type>> As(n);
    std::vector<vector_t<type>> bs(n);

    // randomize
    for (auto& m : As)
        m = matrix_t<type>::Random();
    for (auto& v: bs)
        v = vector_t<type>::Random();
    
    // run on cpu 
    auto results = run_cpu(As, bs);
#ifdef DEBUG
        std::cout 
            << results[0] << std::endl;
#endif // DEBUG
    auto results_gpu = run_gpu_cpubased(As, bs);
    auto results_gpu_option0 = run_gpu_option0(As, bs);
#ifdef DEBUG
    std::cout << results_gpu[0] << std::endl;
#endif 

    auto cpu_vs_gpu_valid = validation::validate_eigen_vectors(results, results_gpu);
    if (cpu_vs_gpu_valid.size()==0)
        std::cout << "+++ cpu vs cpubased impl gpu is valid +++\n";
    else {
        std::cout << "--- cpu vs cpubased impl gpu is not valid ---\n";
        std::cout << "incorrect list { ";
        for (auto i : cpu_vs_gpu_valid)
            std::cout << i << ", ";
        std::cout << " }\n";
        for (auto i : cpu_vs_gpu_valid)
            std::cout 
                << "\n************\n"
                << results[i]
                << "\n****** vs ******\n"
                << results_gpu[i] << std::endl;
    }

    auto gpu_option0_vs_cpu = validation::validate_eigen_vectors(results, results_gpu_option0);
    if (gpu_option0_vs_cpu.size()==0)
        std::cout << "+++ cpu vs option0 impl gpu is valid +++\n";
    else {
        std::cout << "--- cpu vs option0 impl gpu is not valid ---\n";
        std::cout << "incorrect list { ";
        for (auto i : gpu_option0_vs_cpu)
            std::cout << i << ", ";
        std::cout << " }\n";
        for (auto i : gpu_option0_vs_cpu)
            std::cout 
                << "\n************\n"
                << results[i]
                << "\n******* vs *****\n"
                << results_gpu_option0[i] << std::endl;
    }

    return 0;
}
