#include <iostream>
#include <vector>

#include "../common/utils.h"
#include "include/inplace_fnnls.h"

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
std::vector<vector_t<T>> run_gpu_cpubased(std::vector<matrix_t<T>> const& As,
                                       std::vector<vector_t<T>> const& bs) {
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
    cuda::assert_if_error("cuda mallocs");

    // copy
    cuda::copy_to_dev(d_As, As);
    cuda::copy_to_dev(d_bs, bs);
    cuda::assert_if_error("cuda memcpy to device");

    {
        int threads{256};
        int blocks{(n + threads - 1) / threads};
        kernel_cpubased_inplace_fnnls<T><<<blocks, threads>>>(d_As, d_bs, 
            d_result, n);
        cuda::assert_if_error("kernel cpubased inplace fnnls");
    }

    cuda::copy_to_host(result, d_result);
    cuda::assert_if_error("cuda memcpy back to host");

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
                << results[i]
                << "\n************\n"
                << results_gpu[i] << std::endl;
    }

    return 0;
}
