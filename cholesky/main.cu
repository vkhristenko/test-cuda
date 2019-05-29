#include <iostream>
#include <chrono>
#include <vector>

#include "Eigen/Dense"

#include "../common/utils.h"
#include "parse_matrices.hpp"

/*
constexpr unsigned long MATRIX_SIZE = 3;

template<typename T>
using matrix_t = Eigen::Matrix<T, MATRIX_SIZE, MATRIX_SIZE, Eigen::ColMajor>;
*/

namespace mymath {

template<typename T>
struct FusedCholeskyForwardSubst {
    __forceinline__
    __device__ static void compute(
            matrix_t<T> const& M, 
            vector_t<T> const& b,
            matrix_t<T> &L,
            vector_t<T> &intermediate,
            int view) {
        // compute element 0,0 for L
        auto const sqrtm_0_0 = std::sqrt(M(0, 0));
        L(0, 0) = sqrtm_0_0;

        // compute solution for forward subst for element 0
        auto const interm_0 = b(0) / sqrtm_0_0;
        intermediate(0) = interm_0;

        for (int i=1; i<view; ++i) {
            // load the value to sub from
            T total = b(i);

            // first compute elements to the left of the diagoanl
            T sumsq{static_cast<T>(0)};
            for (int j=0; j<i; ++j) {
                T sumsq2{static_cast<T>(0)};
                auto const m_i_j = M(i, j);
                for (int k=0; k<j; ++k)
                    sumsq2 += L(i, k) * L(j, k);

                // comput the i,j : i>j, elements to the left of the diagonal
                auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
                L(i, j) = value_i_j;

                // needed to compute diagonal element
                sumsq += value_i_j * value_i_j;

                total -= value_i_j * intermediate(j);
            }

            // second, compute the diagonal element
            auto const l_i_i = std::sqrt(M(i, i) - sumsq);
            L(i, i) = l_i_i;

            intermediate(i) = total / l_i_i;
        }
    }
};

//
// note, we need to use transpose of M
//
template<typename T>
struct BackwardSubst {
    __forceinline__
    __device__ static void compute(
            matrix_t<T> const& M,
            vector_t<T> const& b,
            vector_t<T> &x,
            int view) {
        // first element
        x(view - 1) = b(view - 1) / M(view-1, view-1);

        // the rest
        for (int i=view-2; i>=0; --i) {
            T total{static_cast<T>(0)};
            for (int j=i+1; j<view; ++j) 
                total += M(j, i) * x(j);

            x(i) = (b(i) - total) / M(i, i);
        }
    }
};

}

namespace cpu_based { 

template<typename T>
__global__
void kernel_cholesky(matrix_t<T> const* As,
                     matrix_t<T> *Ls,
                     unsigned int n) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < n) {
        Ls[idx] = As[idx].llt().matrixL();
    }
}

}

namespace gpu_v0 {

template<typename T>
__global__
void kernel_cholesky(matrix_t<T> const* As, matrix_t<T> *Ls) {
    // indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ch = blockIdx.x;

    // 
    // configure shared mem + load from global
    // 
    __shared__ T __sL[MATRIX_SIZE * MATRIX_SIZE];
    Eigen::Map<matrix_t<T>> sL{__sL};
    __shared__ T __sLSums[MATRIX_SIZE * MATRIX_SIZE];
    Eigen::Map<matrix_t<T>> sLSums{__sLSums};

    // initialize shared mem for L sums
    // note, no synching - this is used from the first iteration only
    sLSums(ty, tx) = 0;
   
    // load M(ty, tx) into thread (ty, tx)
    // load M(tx, tx) into thread (ty, tx)
    // load M(0, 0), into thread (ty, tx)
    auto const m_0_0 = As[ch](0, 0);
    auto const sqrt_m_0_0 = std::sqrt(m_0_0);
    auto const m_i_j = As[ch](ty, tx);
    auto const m_j_j = As[ch](tx, tx);

    //
    // column 0
    //
    // compute L(ty, 0) for ty>=1
    if (tx == 0 && ty >= 1)
        sL(ty, 0) = m_i_j / sqrt_m_0_0; // L(i, 0) = M(i, 0) / M(0, 0)
    else if (tx == 0 && ty == 0) 
        sL(0, 0) = sqrt_m_0_0;
    __syncthreads();

    // TODO: verify that the loop is completely unrolled
#pragma unroll
    for (int column=1; column<MATRIX_SIZE; column++) {
        if (tx == column && ty>=column) {
            // compute L(j, j) = sqrt(M(j, j) - Sum[k](L(j, k) * L(j, k)))
            auto const sumsq = sLSums(column, column) + 
                sL(column, column-1) * sL(column, column-1);
            auto const l_j_j = std::sqrt(m_j_j - sumsq);
            if (ty==column)
                sL(column, column) = l_j_j;
            else {
                // compute L(i, column) = 
                //   (M(i, column) - Sum[k](L(i, k) * L(column, k))) / L(col, col)
                auto const sumsq_i_j = sLSums(ty, column) + 
                    sL(ty, column-1) * sL(column, column-1);
                auto const tmp = m_i_j - sumsq_i_j;
                auto const l_i_column = tmp / l_j_j;
                sL(ty, column) = l_i_column;
            }
        }
        if (tx>=column && ty>=column)
            sLSums(ty, tx) += sL(ty, column-1) * sL(tx, column-1);
        __syncthreads();
    }

    // store back to global mem
    Ls[ch](ty, tx) = sL(ty, tx);

    //
    // column 1
    //
    /*
    if (tx == 1 && ty>=1) {
        // compute L(1, 1) = sqrt(M(1, 1) - L(1, 0) * L(1, 0))
        auto const l_1_1 = std::sqrt(m_j_j - sLSums(1, 1));
        if (ty>1) {
            // compute L(i, 1) = (M(i,1) - L(i, 0) * L(1, 0)) / L(1, 1)
            auto const tmp = m_i_j - sLSums(ty, 1);
            auto const l_i_1 = tmp / l_1_1;
            sL(ty, 1) = l_i_1;
        } else // ty == 1
            sL(1, 1) = l_1_1;
    }
    __syncthreads();
    // accumulate sum of products
    if (tx>=2 && ty>=2)
        sLSums(ty, tx) += sL(ty, 1) * sL(tx, 1);
    __syncthreads();
    */
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
std::vector<matrix_t<T>> wrapper_gpu_v0(std::vector<matrix_t<T>> const& As) {
    std::vector<matrix_t<T>> results(As.size());
    std::cout << "gpu v0\n";
    
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
        dim3 nthreads {MATRIX_SIZE, MATRIX_SIZE};
        int blocks{As.size()};
        gpu_v0::kernel_cholesky<T><<<blocks, nthreads>>>(
            d_As, d_Ls);
        cudaDeviceSynchronize();
        cuda::assert_if_error("\tkernel cpubased cholesky decomposition");

        cudaEventRecord(start, 0);
        gpu_v0::kernel_cholesky<T><<<blocks, nthreads>>>(
            d_As, d_Ls);
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


template<typename T>
std::vector<matrix_t<T>> cpu_based_gpu(std::vector<matrix_t<T>> const& As) {
    std::vector<matrix_t<T>> results(As.size());

    std::cout << "gpu cpu based\n";
    
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

    // template example matrices - symmetric positive definite
    std::cout << "getting template matrices\n";
    std::string name {"matrices.in"};
    auto template_matrices = parse_matrices<type>(name);

    // randomize
    std::cout << "setting up the input matrices\n";
    int i = 0; 
    for (auto& m : As) {
        m = template_matrices[i % template_matrices.size()];

        cpu[i]= m.llt().matrixL();
        i++;
    }

    std::cout << "start computing\n";
    auto result_cpubased = cpu_based_gpu(As);
    auto result_gpu_v0 = wrapper_gpu_v0(As);

    std::cout << "comparison\n";
    /*
    for (int i=0; i<result_cpubased.size(); i++) {
        std::cout 
            << "original matrix\n"
            << As[i]
            << "\ncpu result\n"
            << cpu[i]
            << "\nstandard gpu result\n"
            << result_cpubased[i]
            << "\n"
            << "gpu v0 result\n"
            << result_gpu_v0[i]
            << std::endl;
    }*/

    auto vals_cpu_cpubased = validation::validate_eigen_vectors(cpu, result_cpubased) ;
    auto vals_cpu_gpuv0 = validation::validate_eigen_vectors(cpu, result_gpu_v0);
    if (vals_cpu_cpubased.size() > 0) {
        std::cout << "--- cpu vs gpu cpu based ---\n";
    } else {
        std::cout << "+++ cpu vs gpu cpu based +++\n";
    }
    if (vals_cpu_gpuv0.size() > 0) {
        std::cout << "--- cpu vs gpu v0 ---\n";
    } else 
        std::cout << "+++ cpu vs gpu v0 +++\n";
    
    return 0;
}
