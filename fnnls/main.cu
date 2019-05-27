#include <iostream>
#include <chrono>
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
    auto t1 = std::chrono::high_resolution_clock::now();
    for (unsigned int i=0; i<As.size(); i++) {
        v1::fnnls(As[i], bs[i], result[i]);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
    std::cout << "*** cpu runtime = " << duration.count() << " (ms) ***\n";

    return result;
}

template<typename T>
__global__
void kernel_fnnls(matrix_t<T> const* As, 
                          vector_t<T> const* bs,
                          vector_t<T>* results,
                          unsigned int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        v1::fnnls(As[idx], bs[idx], results[idx]);
    }
}

template<typename T>
std::vector<vector_t<T>> run_gpu_cpubased(std::vector<matrix_t<T>> const& As,
                                       std::vector<vector_t<T>> const& bs, 
                                       int nthreads) {
    std::cout << "*** running gpu cpubased option ***\n";
    std::vector<vector_t<T>> result(As.size());
    for (auto& v : result)
        v = vector_t<T>::Zero();

   cudaEvent_t startE;
   cudaEvent_t endE;

   cudaEventCreate(&startE);
   cudaEventCreate(&endE);

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
        int threads{nthreads};
        int blocks{(n + threads - 1) / threads};
        // warm up
        std::cout << "*** warming up ***\n";
        kernel_fnnls<T><<<blocks, threads>>>(d_As, d_bs, 
            d_result, n);

        // measure
        std::cout << "*** running ****\n";
        for (unsigned int i=0; i<10; i++) {
            cudaEventRecord(startE, 0);
            kernel_fnnls<T><<<blocks, threads>>>(d_As, d_bs, 
                d_result, n);
            cudaEventRecord(endE, 0);
            cudaEventSynchronize(endE);
            float ms;
            cudaEventElapsedTime(&ms, startE, endE);
            printf("\truntime = %f (ms)\n", ms);
            cuda::assert_if_error("\tkernel cpubased inplace fnnls");
        }
    }

    cuda::copy_to_host(result, d_result);
    cuda::assert_if_error("\tcuda memcpy back to host");

    cudaEventDestroy(startE);
    cudaEventDestroy(endE);
    cudaFree(d_As);
    cudaFree(d_bs);
    cudaFree(d_result);

    return result;
}

int main(int argc, char** argv) {
    if (argc<=1) {
        std::cout << "run with './main <number of channels> <nthreads per block>'\n";
        exit(0);
    }
    unsigned int n = std::atoi(argv[1]);
    unsigned int nthreads = std::atoi(argv[2]);
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
    auto results_gpu = run_gpu_cpubased(As, bs, nthreads);

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
            std::cerr 
                << "\n************\n"
                << results[i]
                << "\n****** vs ******\n"
                << results_gpu[i] << std::endl
                << "***** diff *****\n"
                << results[i] - results_gpu[i]
                << std::endl;
    }

    return 0;
}
