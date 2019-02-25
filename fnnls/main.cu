#include <iostream>
#include <vector>

#include "../common/utils.h"
#include "include/inplace_fnnls.h"
#include "include/inplace_fnnls_option0.h"
#include "include/inplace_fnnls_option1.h"

template<typename T>
std::vector<vector_t<T>> run_cpu(std::vector<matrix_t<T>> const& As, 
                                 std::vector<vector_t<T>> const& bs) {
    // init the results
    std::vector<vector_t<T>> result(As.size());
    for (auto& v: result)
        v = vector_t<T>::Zero();

    // compute
    for (unsigned int i=0; i<As.size(); i++) {
//        if (i==113)
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
//        if (idx==113)
            cpubased_inplace_fnnls(As[idx], bs[idx], results[idx]);
    }
}

namespace gpu { namespace option1 {

template<typename T>
__global__
void kernel_inplace_fnnls(matrix_t<T> const* As,
                          vector_t<T> const* bs,
                          vector_t<T>* results,
                          unsigned int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        inplace_fnnls(As[idx], bs[idx], results[idx],
                      threadIdx.x, blockDim.x);
    }
}

}}

namespace gpu { namespace option0 {

template<typename T>
__global__
void kernel_inplace_fnnls(matrix_t<T> const* As,
                                  vector_t<T> const* bs,
                                  vector_t<T>* xs) {
    int imatrix = blockIdx.x;
//    if (imatrix==113)
        inplace_fnnls(As[imatrix], bs[imatrix], xs[imatrix]);
}

}}

template<typename T>
std::vector<vector_t<T>> run_gpu_option0(std::vector<matrix_t<T>> const& As,
                                         std::vector<vector_t<T>> const& bs) {
    std::cout << "*** running gpu option0 ***\n";
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
        dim3 threads{10, 10};
        int blocks{n};
        // warm up
 //       gpu::option0::kernel_inplace_fnnls<T><<<blocks, threads>>>(d_As, d_bs, 
 //           d_result);

        // measure
//        for (unsigned int i=0; i<10; i++) {
            cudaEventRecord(startE, 0);
            gpu::option0::kernel_inplace_fnnls<T><<<blocks, threads>>>(d_As, d_bs, 
                d_result);
            cudaEventRecord(endE, 0);
            cudaEventSynchronize(endE);
            float ms;
            cudaEventElapsedTime(&ms, startE, endE);
            printf("\truntime = %f (ms)\n", ms);
            cuda::assert_if_error("\tkernel inplace fnnls option0");
//        }
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

template<typename T>
std::vector<vector_t<T>> run_gpu_option1(std::vector<matrix_t<T>> const& As,
                                       std::vector<vector_t<T>> const& bs, 
                                       int nthreads) {
    std::cout << "*** running gpu option1 ***\n";
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
        int shared_mem_bytes = sizeof(T) * nthreads *
            ( MATRIX_SIZE*MATRIX_SIZE // AtA
            + MATRIX_SIZE             // Atb
            + MATRIX_SIZE             // x
            + MATRIX_SIZE             // s
            + MATRIX_SIZE             // w
            );
        std::cout << "shared memory per block = " << shared_mem_bytes << std::endl;
        // warm up
        gpu::option1::kernel_inplace_fnnls<T><<<blocks, threads, shared_mem_bytes>>>(
            d_As, d_bs, 
            d_result, n);

        // measure
//        for (unsigned int i=0; i<10; i++) {
            cudaEventRecord(startE, 0);
            gpu::option1::kernel_inplace_fnnls<T><<<blocks, threads, shared_mem_bytes>>>(
                d_As, d_bs, 
                d_result, n);
            cudaEventRecord(endE, 0);
            cudaEventSynchronize(endE);
            float ms;
            cudaEventElapsedTime(&ms, startE, endE);
            printf("\truntime = %f (ms)\n", ms);
            cuda::assert_if_error("\tkernel option1 inplace fnnls");
//        }
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
        kernel_cpubased_inplace_fnnls<T><<<blocks, threads>>>(d_As, d_bs, 
            d_result, n);

        // measure
//        for (unsigned int i=0; i<10; i++) {
            cudaEventRecord(startE, 0);
            kernel_cpubased_inplace_fnnls<T><<<blocks, threads>>>(d_As, d_bs, 
                d_result, n);
            cudaEventRecord(endE, 0);
            cudaEventSynchronize(endE);
            float ms;
            cudaEventElapsedTime(&ms, startE, endE);
            printf("\truntime = %f (ms)\n", ms);
            cuda::assert_if_error("\tkernel cpubased inplace fnnls");
//        }
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
        std::cout << "run with './main <number of matrices>'\n";
        exit(0);
    }
    unsigned int n = std::atoi(argv[1]);
    int nthreads = std::atoi(argv[2]);
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
#ifdef FNNLS_DEBUG_MAIN
        std::cout 
            << results[0] << std::endl;
#endif // DEBUG
    auto results_gpu = run_gpu_cpubased(As, bs, nthreads);
    auto results_gpu_option1 = run_gpu_option1(As, bs, nthreads);
    auto results_gpu_option0 = run_gpu_option0(As, bs);
#ifdef FNNLS_DEBUG_MAIN
    std::cout << results_gpu[0] << std::endl;
    std::cout << "******\n";
    std::cout << results_gpu_option0[0] << std::endl;
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
            std::cerr 
                << "\n************\n"
                << results[i]
                << "\n****** vs ******\n"
                << results_gpu[i] << std::endl
                << "***** diff *****\n"
                << results[i] - results_gpu[i]
                << std::endl;
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
            std::cerr 
                << "\n****** " << i << " ******\n"
                << results[i]
                << "\n******* vs *****\n"
                << results_gpu_option0[i] << std::endl
                << "**** diff ****\n"
                << results[i] - results_gpu_option0[i]
                << std::endl;
    }

    return 0;
}
