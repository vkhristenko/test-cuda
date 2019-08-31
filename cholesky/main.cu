#include <iostream>
#include <chrono>
#include <vector>

#include "Eigen/Dense"

#include "../common/utils.h"
#include "parse_matrices.hpp"

#include <cooperative_groups.h>
using namespace cooperative_groups;

/*
constexpr unsigned long MATRIX_SIZE = 3;

template<typename T>
using matrix_t = Eigen::Matrix<T, MATRIX_SIZE, MATRIX_SIZE, Eigen::ColMajor>;
*/

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

namespace gpu_v1 {

constexpr int threadTileSize = 16;

template<typename T>
__global__
void kernel_cholesky(matrix_t<T> const* As, matrix_t<T> *Ls) {
    // assume that we 
    constexpr auto N = matrix_t<T>::RowsAtCompileTime;
    static_assert(N <= threadTileSize);

    // x <- [0, 15]
    // y <- [0, nchannels per block)
    int const ch = threadIdx.y;

    // partition all threads into groups of 16(threadTileSize)
    auto this_tile = tiled_partition<threadTileSize>(this_thread_block());
    int const rank = this_tile.thread_rank();

    // load M(i, 0)
    T m_i_0;
    if (rank<N) m_i_0 = As[ch](rank, 0);

    // compute L(0, 0)
    T l_0_0;
    if (rank==0) {
        l_0_0 = std::sqrt(m_i_0);

        // store this guy to global
        Ls[ch](0, 0) = l_0_0;
    }
    // retrieve L(0, 0) from rank 0 thread 
    l_0_0 = this_tile.shfl(l_0_0, 0);
    // compute L(i, 0) for i>=1
    T l_i_0;
    if (rank>0 && rank<N) {
        l_i_0 = m_i_0 / l_0_0;

        // store this guy to global L(i, 0) for i >= 1
        Ls[ch](rank, 0) = l_i_0;
    }

    // accumulators
    // FIXME: this must be in registers
    T reg_sumsq;
    T reg_accum[N-2];

    // accumulate per row
    if (rank>=1 && rank<N)
        reg_sumsq = l_i_0 * l_i_0;

    // accumulate over column 0
    #pragma unroll
    for (int ineighbor=1; ineighbor<N-1; ineighbor++) {
        auto value_l_i_0 = this_tile.shfl_up(l_i_0, ineighbor);
        if (rank>ineighbor && rank<N)
            reg_accum[rank-ineighbor-1] = value_l_i_0 * l_i_0;
    }

    // iterate until the last one:
    // for icol>=1
    // - load icol
    // - compute L(icol, icol)
    // - compute L(i, icol) for i>icol
    // - accumulate for icol
    #pragma unroll
    for (int icol=1; icol<N-1; icol++) {
        // load icol
        T m_i_icol;
        if (rank>=icol && rank<N)
            m_i_icol = As[ch](rank, icol);

        // compute L(icol, icol)
        T l_icol_icol;
        if (rank==icol) {
            l_icol_icol = std::sqrt(m_i_icol - reg_sumsq);

            // store to global
            Ls[ch](icol, icol) = l_icol_icol;
        }

        // broadcast icol thread value and sync
        l_icol_icol = this_tile.shfl(l_icol_icol, icol);

        // compute L(i, icol) for i>icol
        T l_i_icol;
        if (rank>icol && rank<N) {
            l_i_icol = (m_i_icol - reg_accum[icol-1]) / l_icol_icol;

            // store to global
            Ls[ch](rank, icol) = l_i_icol;
        }

        // accumulate per row
        if (rank>=icol+1 && rank<N)
            reg_sumsq += l_i_icol*l_i_icol;

        // accumulate over column icol
        auto Nneighbors = N-icol-1;
        #pragma unroll
        for (int delta=1; delta<Nneighbors; delta++) {
            auto value_l_i_icol = this_tile.shfl_up(l_i_icol, delta);
            if (rank>icol+delta && rank<N)
                reg_accum[rank-delta-1] += value_l_i_icol*l_i_icol;
        }
    }

    // compute the L(N-1, N-1)
    if (rank==N-1) {
        auto m_i_i = As[ch](N-1, N-1);
        auto l_i_i = std::sqrt(m_i_i - reg_sumsq);

        // store to global
        Ls[ch](N-1, N-1) = l_i_i;
    }
}

template<typename T>
std::vector<matrix_t<T>> launcher(std::vector<matrix_t<T>> const& As) {
    std::vector<matrix_t<T>> results(As.size());
    std::cout << "gpu v1\n";
    
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
        int nchannelsPerBlock = 32;
        dim3 nthreads = {threadTileSize, nchannelsPerBlock};
        int blocks{(As.size() + nchannelsPerBlock - 1) / nchannelsPerBlock};
        kernel_cholesky<T><<<blocks, nthreads>>>(
            d_As, d_Ls);
        cudaDeviceSynchronize();
        cuda::assert_if_error("\tkernel gpu v1 cholesky decomposition");

        cudaEventRecord(start, 0);
        kernel_cholesky<T><<<blocks, nthreads>>>(
            d_As, d_Ls);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float ms;
        cudaEventElapsedTime(&ms, start, end);
        cuda::assert_if_error("\tkernel gpu v1 cholesky decomposition");
        printf("\telapsed time = %f (ms)\n", ms);
    }
    cuda::copy_to_host(results, d_Ls);
    cuda::assert_if_error("\tcuda memcpy back to host");

    cudaFree(d_As);
    cudaFree(d_Ls);

    return results;
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
    auto result_gpu_coopgroups = gpu_v1::launcher(As);

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
    auto vals_cpu_gpuv1 = validation::validate_eigen_vectors(cpu, result_gpu_v0);
    if (vals_cpu_cpubased.size() > 0) {
        std::cout << "--- cpu vs gpu cpu based ---\n";
    } else {
        std::cout << "+++ cpu vs gpu cpu based +++\n";
    }
    if (vals_cpu_gpuv0.size() > 0) {
        std::cout << "--- cpu vs gpu v0 ---\n";
    } else 
        std::cout << "+++ cpu vs gpu v0 +++\n";
    if (vals_cpu_gpuv1.size() > 0) {
        std::cout << "--- cpu vs gpu v1 ---\n";
    } else 
        std::cout << "+++ cpu vs gpu v1 +++\n";
    
    return 0;
}
