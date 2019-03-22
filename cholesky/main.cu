#include <iostream>
#include <chrono>
#include <vector>

#include "Eigen/Dense"

#include "../common/utils.h"

constexpr unsigned long MATRIX_SIZE = 10;

template<typename T>
using matrix_t = Eigen::Matrix<T, MATRIX_SIZE, MATRIX_SIZE, Eigen::ColMajor>;

namespace cpu_based { 

template<typename T>
__global__
void kernel_cholesky(matrix_t<T> const* As,
                     matrix_t<T> *Ls,
                     unsigned int n) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < n) {
        for (unsigned int i=0; i<100; i++)
            Ls[idx] = As[idx].llt().matrixL();
    }
}

}

/*
namespace gpu_v0 {

template<typename T>
__global__
void kernel_cholesky(...) {
    int ltx = threadIdx.x;
    int gtx = threadIdx.x + blockIdx.x*blockDim.x;
    int gch = gtx / matrix_size;
    int lch = ltx / matrix_size;

    extern __shared__ char sdata[];


    if (gch < nchannels) {

    }
}

}
*/

template<typename T>
std::vector<matrix_t<T>> cpu_based_gpu(std::vector<matrix_t<T>> const& As) {
    std::vector<matrix_t<T>> results(As.size());
    
    // devcie ptrs
    matrix_t<T> *d_As;
    matrix_t<T> *d_Ls;
    unsigned int n = As.size();

    cudaEvent_t start;
    cudaEvent_t end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // allocate
    cuda::cuda_malloc(d_As, n);
    cuda::cuda_malloc(d_Ls, n); 
    cuda::assert_if_error("\tcuda mallocs");

    // copy
    cuda::copy_to_dev(d_As, As);
    cuda::assert_if_error("\tcuda memcpy to device");

    {
        int nthreads = 256;
        int blocks{(n + nthreads - 1) / nthreads};
        cpu_based::kernel_cholesky<T><<<blocks, nthreads>>>(
            d_As, d_Ls, n);
        cudaDeviceSynchronize();
        cuda::assert_if_error("\tkernel cpubased cholesky decomposition");

        cudaEventRecord(start, 0);
        cpu_based::kernel_cholesky<T><<<blocks, nthreads>>>(
            d_As, d_Ls, n);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float ms;
        cudaEventElapsedTime(&ms, start, end);
        cuda::assert_if_error("\tkernel cpubased cholesky decomposition");
        printf("\telapsed time = %f (ms)\n", ms);
    }
    cuda::copy_to_host(results, d_Ls);
    cuda::assert_if_error("\tcuda memcpy back to host");

    cudaFree(d_As);
    cudaFree(d_Ls);

    return results;
}

int main(int argc, char** argv) {
    if (argc<=1) {
        std::cout << "run with './main <number of matrices>'\n";
        exit(0);
    }
    unsigned int n = std::atoi(argv[1]);
//    int nthreads = std::atoi(argv[2]);
    using type = float;
    
    // create the input matrices
    std::vector<matrix_t<type>> As(n);
    std::vector<matrix_t<type>> cpu(n);

    // randomize
    int i = 0; 
    for (auto& m : As) {
        auto tmp = matrix_t<type>::Random();
        m = tmp.transpose() * tmp;

        cpu[i]= m.llt().matrixL();
        i++;
    }

    auto result_cpubased = cpu_based_gpu(As);

    auto vals_cpu_cpubased = validation::validate_eigen_vectors(cpu, result_cpubased) ;
    if (vals_cpu_cpubased.size() > 0) {
        std::cout << "--- cpu vs gpu cpu based ---\n";
    } else {
        std::cout << "+++ cpu vs gpu cpu based +++\n";
    }
    
    return 0;
}
