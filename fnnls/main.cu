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
        vector_t<T> * __restrict__ xs) {
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
    }
}

template<typename T>
__global__
void kernel_fnnls(
        matrix_t<T> * __restrict__ AtAs,
        vector_t<T> * __restrict__ Atbs,
        vector_t<T> * __restrict__ xs,
        unsigned int n) {
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) return;

    constexpr double eps = 1e-11;
    constexpr unsigned int max_iterations = 1000;
    auto& AtA = AtAs[tid];
    auto& Atb = Atbs[tid];
    auto& x = xs[tid];
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
            auto const atb = Atb(i);
            #pragma unroll
            for (unsigned int k=0; k<VECTOR_SIZE; k++)
                sum_per_row += AtA(i, k) * x(k);

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

        // swap AtA to avoid copy
        AtA.col(nPassive).swap(AtA.col(w_max_idx));
        AtA.row(nPassive).swap(AtA.row(w_max_idx));
        // swap Atb to match with AtA
        Eigen::numext::swap(Atb.coeffRef(nPassive), Atb.coeffRef(w_max_idx));
        Eigen::numext::swap(x.coeffRef(nPassive), x.coeffRef(w_max_idx));

        ++nPassive;

        // inner loop
        vector_t<data_type> s;
        while (nPassive > 0) {
          s.head(nPassive) =
              AtA.topLeftCorner(nPassive, nPassive).llt().solve(Atb.head(nPassive));

          if (s.head(nPassive).minCoeff() > 0.) {
            x.head(nPassive) = s.head(nPassive);
            break;
          }

          auto alpha = std::numeric_limits<double>::max();
          Index alpha_idx = 0;

          for (auto i = 0; i < nPassive; ++i) {
            if (s[i] <= 0.) {
              auto const ratio = x[i] / (x[i] - s[i]);
              if (ratio < alpha) {
                alpha = ratio;
                alpha_idx = i;
              }
            }
          }

          if (std::numeric_limits<double>::max() == alpha) {
            x.head(nPassive) = s.head(nPassive);
            break;
          }

          x.head(nPassive) += alpha * (s.head(nPassive) - x.head(nPassive));
          x[alpha_idx] = 0;
          --nPassive;

          AtA.col(nPassive).swap(AtA.col(alpha_idx));
          AtA.row(nPassive).swap(AtA.row(alpha_idx));
          // swap Atb to match with AtA
          Eigen::numext::swap(Atb.coeffRef(nPassive), Atb.coeffRef(alpha_idx));
          Eigen::numext::swap(x.coeffRef(nPassive), x.coeffRef(alpha_idx));

    }
  }
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
    matrix_t<T> *d_As, *d_AtAs;
    vector_t<T> *d_bs, *d_Atbs, *d_xs;
    unsigned int n = As.size();

    // allocate on device
    cuda::cuda_malloc(d_As, n);
    cuda::cuda_malloc(d_AtAs, n);
    cuda::cuda_malloc(d_bs, n);
    cuda::cuda_malloc(d_Atbs, n);
    cuda::cuda_malloc(d_xs, n);
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
            d_As, d_bs, d_AtAs, d_Atbs, d_xs);
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
            d_AtAs, d_Atbs, d_xs, n);
        cudaEventRecord(eFinish, 0);
        cudaEventSynchronize(eFinish);
        cudaEventElapsedTime(&ms, eStart, eFinish);
        printf("runtime = %f (ms)\n", ms);
        cuda::assert_if_error("checking 'kernel_fnnls' kernel");
    }

    {
        std::cout << "running...\n";
        for (unsigned int i=0; i<10; i++) {
            dim3 nthreadsMult{nrows, nrows};
            dim3 blocksMult{n};
            cudaEventRecord(eStart, 0);
            kernel_mults<T, nrows><<<blocksMult, nthreadsMult>>>(
                d_As, d_bs, d_AtAs, d_Atbs, d_xs);
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
                d_AtAs, d_Atbs, d_xs, n);
            cudaEventRecord(eFinish, 0);
            cudaEventSynchronize(eFinish);
            cudaEventElapsedTime(&ms, eStart, eFinish);
            printf("fnnls runtime = %f (ms)\n", ms);
            cuda::assert_if_error("checking 'kernel_fnnls' kernel");
        }
    }
    
    cuda::copy_to_host(results, d_xs);
    cuda::assert_if_error("\tcuda memcpy back to host");

    cudaEventDestroy(eStart);
    cudaEventDestroy(eFinish);
    cudaFree(d_As);
    cudaFree(d_AtAs);
    cudaFree(d_bs);
    cudaFree(d_Atbs);
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
        std::cout << "run with './main <number of channels> <n'\n";
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
        std::cout << "run with './main <number of channels> <nthreads per block>'\n";
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
        std::cout << "run with './exec <option> <...>'\n";
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
