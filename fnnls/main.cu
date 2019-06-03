#include <iostream>
#include <chrono>
#include <vector>

#include "../common/utils.h"
#include "include/inplace_fnnls.h"
//#include "include/fnnls_no_mm_mv.h"

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
void kernel_inplace_fnnls(matrix_t<T> const* As, 
                          vector_t<T> const* bs,
                          vector_t<T>* results,
                          unsigned int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        v1::fnnls(As[idx], bs[idx], results[idx]);
    }
}

namespace v2 {

//
// perform AtA and Atb mm and mv mults
//
template<typename T, int MSIZE>
__global__
void kernel_mults(
        matrix_t<T> const* __restrict__ As,
        vector_t<T> const* __restrict__ bs,
        matrix_t<T> * __restrict__ AtAs,
        vector_t<T> * __restrict__ Atbs,
        vector_t<T> * __restrict__ xs,
        char * __restrict__ mapping) {
    int const ch = blockIdx.x;
    int const tx = threadIdx.x;
    int const ty = threadIdx.y;

    // shared mem - static alloc
    __shared__ T shrA[MSIZE][MSIZE];
    __shared__ T shrb[MSIZE];

    // load into shared mem
    shrA[ty][tx] = As[ch](ty, tx);
    if (ty==0)
        shrb[tx] = bs[ch](tx);

    // make sure things get loaded
    __syncthreads();

    auto result_mm{static_cast<T>(0)};
    auto result_mv{static_cast<T>(0)};
    #pragma unroll
    for (int i=0; i<MSIZE; i++) {
        result_mm += shrA[i][ty] * shrA[i][tx];
        if (ty==0)
            result_mv += shrA[i][tx] * shrb[i];
    }

    // store back to global
    AtAs[ch](ty, tx) = result_mm;
    if (ty==0) {
        Atbs[ch](tx) = result_mv;
        
        // initialize the result vector
        xs[ch](tx) = 0;
        // init the mapping
        mapping[ch*MSIZE + tx] = tx;
    }
}

using namespace Eigen;

//
// default simple version - for any N
//
template<typename T>
struct FusedCholeskyForwardSubst {
    __forceinline__
    __device__ static void compute(
            my_matrix_t<T> const& M, 
            my_vector_t<T> const& b,
            my_matrix_t<T> &L,
            my_vector_t<T> &intermediate,
            char const* mapping,
            int view) {
        // compute element 0,0 for L
        auto const real_0 = mapping[0];
        auto const sqrtm_0_0 = std::sqrt(M(real_0, real_0));
        L(0, 0) = sqrtm_0_0;

        // compute solution for forward subst for element 0
        auto const interm_0 = b(real_0) / sqrtm_0_0;
        intermediate(0) = interm_0;

        for (int i=1; i<view; ++i) {
            // load the value to sub from
            auto const real_i = mapping[i];
            T total = b(real_i);

            // first compute elements to the left of the diagoanl
            T sumsq{static_cast<T>(0)};
            for (int j=0; j<i; ++j) {
                T sumsq2{static_cast<T>(0)};
                auto const real_j = mapping[j];
                auto const m_i_j = M(real_i, real_j);
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
            auto const l_i_i = std::sqrt(M(real_i, real_i) - sumsq);
            L(i, i) = l_i_i;

            intermediate(i) = total / l_i_i;
        }
    }
};

template<typename T, int N>
struct FusedCholeskyForwardSubstUnrolled {
    __forceinline__
    __device__ static void compute(
            my_matrix_t<T> const& M, 
            my_vector_t<T> const& b,
            my_matrix_t<T> &L,
            my_vector_t<T> &intermediate,
            char const* mapping) {
        // compute element 0,0 for L
        auto const real_0 = mapping[0];
        auto const sqrtm_0_0 = std::sqrt(M(real_0, real_0));
        L(0, 0) = sqrtm_0_0;

        // compute solution for forward subst for element 0
        auto const interm_0 = b(real_0) / sqrtm_0_0;
        intermediate(0) = interm_0;

        #pragma unroll
        for (int i=1; i<N; ++i) {
            // load the value to sub from
            auto const real_i = mapping[i];
            T total = b(real_i);

            // first compute elements to the left of the diagoanl
            T sumsq{static_cast<T>(0)};
            for (int j=0; j<i; ++j) {
                T sumsq2{static_cast<T>(0)};
                auto const real_j = mapping[j];
                auto const m_i_j = M(real_i, real_j);
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
            auto const l_i_i = std::sqrt(M(real_i, real_i) - sumsq);
            L(i, i) = l_i_i;

            intermediate(i) = total / l_i_i;
        }
    }
};

template<typename T, int N>
struct FusedCholeskySolver;

template<typename T>
struct FusedCholeskySolver<T, 1> {
    __forceinline__
    __device__ static void compute(
            my_matrix_t<T> const& M,
            my_vector_t<T> const& b,
            my_vector_t<T> &x,
            char const* mapping) {
        auto const real_0 = mapping[0];
        auto const x_0 = b(real_0) / M(real_0, real_0);
        x(0) = x_0;
    }
};

template<typename T>
struct FusedCholeskySolver<T, 2> {
    __forceinline__
    __device__ static void compute(
            my_matrix_t<T> const& M,
            my_vector_t<T> const& b,
            my_vector_t<T> &x,
            char const* mapping) {
        // element 0
        auto const real_0 = mapping[0];
        auto const real_1 = mapping[1];
        auto const l_0_0 = std::sqrt(M(real_0, real_0));
        auto const interm_0 = b(real_0) / l_0_0;

        // element 1
        auto const l_1_0 = M(real_1, real_0) / l_0_0;
        auto const l_1_1 = std::sqrt(M(real_1, real_1) - l_1_0*l_1_0);
        auto const interm_1 = (b(real_1) - interm_0 * l_1_0) / l_1_1;
        auto const x_1 = interm_1 / l_1_1;
        x(1) = x_1;
        auto const x_0 = (interm_0 - l_1_0 * x_1) / l_0_0;
        x(0) = x_0;
    }
};

template<typename T>
struct FusedCholeskySolver<T, 3> {
    __forceinline__
    __device__ static void compute(
            my_matrix_t<T> const& M,
            my_vector_t<T> const& b,
            my_vector_t<T> &x,
            char const* mapping) {
        // element 0
        auto const real_0 = mapping[0];
        auto const l_0_0 = std::sqrt(M(real_0, real_0));
        auto const interm_0 = b(real_0) / l_0_0;

        // row 1
        auto const real_1 = mapping[1];
        auto const l_1_0 = M(real_1, real_0) / l_0_0;
        auto const l_1_1 = std::sqrt(M(real_1, real_1) - l_1_0*l_1_0);
        auto const interm_1 = (b(real_1) - interm_0 * l_1_0) / l_1_1;

        // row 2
        auto const real_2 = mapping[2];
        auto const l_2_0 = M(real_2, real_0) / l_0_0;
        auto const l_2_1 = (M(real_2, real_1) - l_2_0 * l_1_0) / l_1_1;
        auto const l_2_2 = std::sqrt(M(real_2, real_2) - l_2_0 * l_2_0 - l_2_1*l_2_1);
        auto const interm_2 = (b(real_2) - interm_0 * l_2_0 - interm_1 * l_2_1) / l_2_2;

        auto const x_2 = interm_2 / l_2_2;
        x(2) = x_2;
        auto const x_1 = (interm_1 - l_2_1 * x_2) / l_1_1;
        x(1) = x_1;
        auto const x_0 = (interm_0 - l_1_0 * x_1 - l_2_0 * x_2) / l_0_0;
        x(0) = x_0;
    }
};

//
// note, we need to use transpose of M
//  default simple version
// note, we do not need to do mapping of indices for backward substitution
//
template<typename T>
struct BackwardSubst {
    __forceinline__
    __device__ static void compute(
            my_matrix_t<T> const& M,
            my_vector_t<T> const& b,
            my_vector_t<T> &x,
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

template<typename T, int N>
struct BackwardSubstUnrolled {
    __forceinline__
    __device__ static void compute(
            my_matrix_t<T> const& M,
            my_vector_t<T> const& b,
            my_vector_t<T> &x) {
        // first element
        x(N - 1) = b(N - 1) / M(N-1, N-1);

        // the rest
        #pragma unroll
        for (int i=N-2; i>=0; --i) {
            T total{static_cast<T>(0)};
            for (int j=i+1; j<N; ++j) 
                total += M(j, i) * x(j);

            x(i) = (b(i) - total) / M(i, i);
        }
    }
};

// j > i
template<typename T>
__forceinline__
__device__
void swap_rows_cols(
        my_matrix_t<T> &M,
        int i,
        int j) {
    // diagonal 
    auto const diag = M(i, i);
    M(i, i) = M(j, j);
    M(j, j) = diag;

    #pragma unroll
    for (int k=0; k<matrix_t<T>::RowsAtCompileTime; ++k) {
        if (k==i || k==j)
            continue;

        auto const tmp = M(i, k);
        M(i, k) = M(j, k);
        M(j, k) = tmp;
        M(k, i) = M(k, j);
        M(k, j) = tmp;
    }
}

/*
template<typename T>
__global__
void kernel_fnnls_mult() {
    auto const ty = threadIdx.y;
    auto const tx = threadIdx.x;
    if (tid >= blockDim.x) return;

    __shared__ T shrAtA[VECTOR_SIZE][VECTOR_SIZE];
    // load only the necessary rows
    if (ty>=npassive)
        shrAtA[ty][tx] = AtA(ty, tx);
    if (ty==0)
        shrx[tx] = x(tx);
    __syncthreads();

    // for now use a single thread per row
    if (tx==0 && ty>=npassive) {
        auto const atb = Atb(ty);
        auto sum{static_cast<T>(0)};
        #pragma unroll
        for (unsigned int k=0; k<VECTOR_SIZE; k++)
            sum += shrAtA(ty, k) * shrx(k);

        auto const wvalue = atb - sum;
    }
}
*/

template<typename T>
__global__
void kernel_fnnls(
        matrix_t<T> * __restrict__ AtAs,
        matrix_t<T> * __restrict__ Ls,
        vector_t<T> * __restrict__ Atbs,
        vector_t<T> * __restrict__ xs,
        char * __restrict__ mapping,
        unsigned int n) {
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    int const offset = tid * VECTOR_SIZE;

    if (tid >= n) return;

    constexpr double eps = 1e-11;
    constexpr unsigned int max_iterations = 1000;
    my_matrix_t<T> AtA{AtAs[tid].data()};
    my_matrix_t<T> L{Ls[tid].data()};
    my_vector_t<T> Atb{Atbs[tid].data()};
    my_vector_t<T> x{xs[tid].data()};
    using data_type = T;

    auto nPassive = 0;
    // main loop
    for (auto iter = 0; iter < max_iterations; ++iter) {
        const auto nActive = VECTOR_SIZE - nPassive;

        if(!nActive)
          break;

        //  
        unsigned int w_max_idx = -1;
        auto max_w {static_cast<T>(-1)};
        for (unsigned int i=VECTOR_SIZE-nActive; i<VECTOR_SIZE; i++) {
            auto sum_per_row{static_cast<T>(0)};
            auto const real_i = mapping[offset + i];
            auto const atb = Atb(real_i);
            #pragma unroll
            for (unsigned int k=0; k<VECTOR_SIZE; k++)
                // note, we do not need to look up k in the mapping
                // both AtA and x have swaps applied -> therefore dot product will 
                // not change per row
                sum_per_row += AtA(real_i, k) * x(k);

            // compute gradient value and check if it is greater than max
            auto const wvalue = atb - sum_per_row;
            if (max_w < wvalue) {
                max_w = wvalue;
                w_max_idx = i;
            }
        }

        // check for convergence
        if (max_w < eps)
          break;

        Eigen::numext::swap(
                mapping[offset + nPassive], 
                mapping[offset + w_max_idx]);
        ++nPassive;

        // inner loop
        data_type __s[matrix_t<T>::RowsAtCompileTime], __tmp[matrix_t<T>::RowsAtCompileTime];
        my_vector_t<data_type> s{__s}, tmp{__tmp};
        while (nPassive > 0) {
          char const* current_mapping = mapping + offset;
          switch (nPassive) {
          case 1:
              FusedCholeskySolver<T, 1>::compute(AtA, Atb, s, current_mapping);
              break;
          case 2:
              FusedCholeskySolver<T, 2>::compute(AtA, Atb, s, current_mapping);
              break;
          case 3:
              FusedCholeskySolver<T, 3>::compute(AtA, Atb, s, current_mapping);
              break;
          case 4:
              FusedCholeskyForwardSubstUnrolled<T, 4>::compute(AtA, Atb, L, tmp, 
                current_mapping);
              BackwardSubstUnrolled<T, 4>::compute(L, tmp, s);
              break;
          case 5:
              FusedCholeskyForwardSubstUnrolled<T, 5>::compute(AtA, Atb, L, tmp,
                current_mapping);
              BackwardSubstUnrolled<T, 5>::compute(L, tmp, s);
              break;
          case 6:
              FusedCholeskyForwardSubstUnrolled<T, 6>::compute(AtA, Atb, L, tmp,
                current_mapping);
              BackwardSubstUnrolled<T, 6>::compute(L, tmp, s);
              break;
          case 7:
              FusedCholeskyForwardSubstUnrolled<T, 7>::compute(AtA, Atb, L, tmp,
                current_mapping);
              BackwardSubstUnrolled<T, 7>::compute(L, tmp, s);
              break;
          case 8:
              FusedCholeskyForwardSubstUnrolled<T, 8>::compute(AtA, Atb, L, tmp,
                current_mapping);
              BackwardSubstUnrolled<T, 8>::compute(L, tmp, s);
              break;
          default:
              FusedCholeskyForwardSubst<T>::compute(AtA, Atb, L, tmp,
                current_mapping, nPassive);
              BackwardSubst<T>::compute(L, tmp, s, nPassive);
          }

          bool hasNegative = false;
          for (int ii=0; ii<nPassive; ++ii) {
              hasNegative |= s(ii) <= 0;
          }
          if (!hasNegative) {
              for (int i=0; i<nPassive; ++i) {
                  // note, s contains passive/active set layout
                  // and x contains unpermuted final values in their repective pos
                  auto const real_i = mapping[offset + i];
                  x(real_i) = s(i);
              }
              break;
          }

          auto alpha = std::numeric_limits<T>::max();
          char alpha_idx=0, real_alpha_idx=0;

          for (auto i = 0; i < nPassive; ++i) {
            if (s(i) <= 0.) {
              auto const real_i = mapping[offset + i];
              auto const x_i = x(real_i);
              auto const ratio = x_i / (x_i - s(i));
              if (ratio < alpha) {
                alpha = ratio;
                alpha_idx = i;
                real_alpha_idx = real_i;
              }
            }
          }

          if (std::numeric_limits<T>::max() == alpha) {
            for (int i=0; i<nPassive; ++i) {
                auto const real_i = mapping[offset + i];
                x(real_i) = s(i);
            }
            break;
          }

          for (int ii=0; ii<nPassive; ++ii) {
            auto const real_i = mapping[offset+ii];
            auto const x_ii = x(real_i);
            x(real_i) += alpha * (s(ii) - x_ii);
          }
          x(real_alpha_idx) = 0;
          --nPassive;

          Eigen::numext::swap(
                mapping[offset + nPassive], 
                mapping[offset + alpha_idx]);
    }
  }
}

template<typename T, int NCHANNELS>
__global__
void kernel_permute(
        vector_t<T> *xs,
        char const* mapping,
        int const n) {
    // indices 
    int const gtid = threadIdx.x + blockDim.x*blockIdx.x;
    int const gch = gtid / 10;
    int const ltid = threadIdx.x % 10;
    int const lch = threadIdx.x / 10;

    if (gch >= n) return;

    // configure shared mem
    __shared__ T values[NCHANNELS * 10];

    // copy to local
    values[lch*10 + ltid] = xs[gch](ltid);
    char const sample = mapping[gtid];
    __syncthreads();

    // write back to global
    xs[gch](ltid) = values[lch*10 + sample];
}

template<typename T>
std::vector<vector_t<T>> run(
        std::vector<matrix_t<T>> const& As,
        std::vector<vector_t<T>> const& bs,
        int nthreads) {
    std::cout << "*** testing v2 ***\n";
    constexpr unsigned int nrows = matrix_t<T>::RowsAtCompileTime;

    // init results
    std::vector<vector_t<T>> results(As.size());
    for (auto& v : results)
        v = vector_t<T>::Zero();

    // create cuda events
    cudaEvent_t eStart, eFinish;
    cudaEventCreate(&eStart);
    cudaEventCreate(&eFinish);

    // allocate device mem
    matrix_t<T> *d_As, *d_AtAs, *d_L;
    vector_t<T> *d_bs, *d_Atbs, *d_xs;
    char *d_mapping;
    unsigned int n = As.size();

    // allocate on device
    cuda::cuda_malloc(d_As, n);
    cuda::cuda_malloc(d_AtAs, n);
    cuda::cuda_malloc(d_L, n);
    cuda::cuda_malloc(d_bs, n);
    cuda::cuda_malloc(d_Atbs, n);
    cuda::cuda_malloc(d_xs, n);
    cuda::cuda_malloc(d_mapping, n * matrix_t<T>::RowsAtCompileTime);
    cuda::assert_if_error("\tcuda mallocs");

    // copy input
    cuda::copy_to_dev(d_As, As);
    cuda::copy_to_dev(d_bs, bs);
    cuda::assert_if_error("\tcuda memcpy to device");

    {
        std::cout << "warming up...\n";
        dim3 nthreadsMult{nrows, nrows};
        dim3 blocksMult{n};
        cudaEventRecord(eStart, 0);
        kernel_mults<T, nrows><<<blocksMult, nthreadsMult>>>(
            d_As, d_bs, d_AtAs, d_Atbs, d_xs, d_mapping);
        cudaEventRecord(eFinish, 0);
        cudaEventSynchronize(eFinish);
        float ms;
        cudaEventElapsedTime(&ms, eStart, eFinish);
        printf("runtime = %f (ms)\n", ms);
        cuda::assert_if_error("checking 'kernel_mults' kernel");

        unsigned int threadsFnnls{nthreads};
        unsigned int blocksFnnls{(n+threadsFnnls-1)/threadsFnnls};
        cudaEventRecord(eStart, 0);
        kernel_fnnls<T><<<blocksFnnls, threadsFnnls>>>(
            d_AtAs, d_L, d_Atbs, d_xs, d_mapping, n);
        cudaEventRecord(eFinish, 0);
        cudaEventSynchronize(eFinish);
        cudaEventElapsedTime(&ms, eStart, eFinish);
        printf("runtime = %f (ms)\n", ms);
        cuda::assert_if_error("checking 'kernel_fnnls' kernel");
        kernel_permute<T, 32><<<(n*10+320-1)/320, 320>>>(d_xs, d_mapping, n);
        cuda::assert_if_error("chacking permutation");
    }

    {
        std::cout << "running...\n";
        for (unsigned int i=0; i<10; i++) {
            dim3 nthreadsMult{nrows, nrows};
            dim3 blocksMult{n};
            cudaEventRecord(eStart, 0);
            kernel_mults<T, nrows><<<blocksMult, nthreadsMult>>>(
                d_As, d_bs, d_AtAs, d_Atbs, d_xs, d_mapping);
            cudaEventRecord(eFinish, 0);
            cudaEventSynchronize(eFinish);
            float ms;
            cudaEventElapsedTime(&ms, eStart, eFinish);
            printf("matrix mults runtime = %f (ms)\n", ms);
            cuda::assert_if_error("checking 'kernel_mults' kernel");

            unsigned int threadsFnnls{nthreads};
            unsigned int blocksFnnls{(n+threadsFnnls-1)/threadsFnnls};
            cudaEventRecord(eStart, 0);
            kernel_fnnls<T><<<blocksFnnls, threadsFnnls>>>(
                d_AtAs, d_L, d_Atbs, d_xs, d_mapping, n);
            cudaEventRecord(eFinish, 0);
            cudaEventSynchronize(eFinish);
            cudaEventElapsedTime(&ms, eStart, eFinish);
            printf("fnnls runtime = %f (ms)\n", ms);
            cuda::assert_if_error("checking 'kernel_fnnls' kernel");
            kernel_permute<T, 32><<<(n*10+320-1)/320, 320>>>(d_xs, d_mapping, n);
            cuda::assert_if_error("chacking permutation");
        }
    }
    
    cuda::copy_to_host(results, d_xs);
    cuda::assert_if_error("\tcuda memcpy back to host");

    cudaEventDestroy(eStart);
    cudaEventDestroy(eFinish);
    cudaFree(d_As);
    cudaFree(d_AtAs);
    cudaFree(d_L);
    cudaFree(d_bs);
    cudaFree(d_Atbs);
    cudaFree(d_mapping);
    cudaFree(d_xs);

    return results;
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
        kernel_inplace_fnnls<T><<<blocks, threads>>>(d_As, d_bs, 
            d_result, n);

        // measure
        std::cout << "*** running ****\n";
        for (unsigned int i=0; i<10; i++) {
            cudaEventRecord(startE, 0);
            kernel_inplace_fnnls<T><<<blocks, threads>>>(d_As, d_bs, 
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
        std::cout << "run with './main <number of channels> <option>'\n";
        exit(0);
    }

    using DataType = float;
    std::vector<vector_t<DataType>> results_gpu;

    // input number of channels
    unsigned int n = std::atoi(argv[1]);

    // create the input matrices
    std::vector<matrix_t<DataType>> As(n);
    std::vector<vector_t<DataType>> bs(n);

    // randomize
    for (auto& m : As)
        m = matrix_t<DataType>::Random();
    for (auto& v: bs)
        v = vector_t<DataType>::Random();

    // run cpu version
    auto results_cpu = run_cpu(As, bs);

    int option = std::atoi(argv[2]);
    switch (option) {
    case -1:
    case 0:
        std::cout << "run with './exec <number of channels> <option> <...>'\n";
        exit(0);
        break;
    case 1: 
    {
        unsigned int nthreads = std::atoi(argv[3]);
        results_gpu = run_gpu_cpubased(As, bs, nthreads);
    }
        break;
    case 2:
    {
        // use version 2 - matrix mult + fnnls
        unsigned int nthreads = std::atoi(argv[3]);
        results_gpu = v2::run(As, bs, nthreads);
    }
        break;
    default:
        std::cout << "run with './exec <number of channels> <option> <...>'\n";
        exit(0);
    }
    
    auto cpu_vs_gpu_valid = 
        validation::validate_eigen_vectors(results_cpu, results_gpu);
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
                << results_cpu[i]
                << "\n****** vs ******\n"
                << results_gpu[i] << std::endl
                << "***** diff *****\n"
                << results_cpu[i] - results_gpu[i]
                << std::endl;
    }

    return 0;
}
