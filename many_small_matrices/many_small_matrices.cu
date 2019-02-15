#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include "../common/utils.h"

#include "Eigen/Dense"

constexpr int dim = 16;

template<typename T>
using matrix_t = Eigen::Matrix<T, dim, dim>;

template<typename T, int BLOCK_SIZE>
__global__
void kernel_mat_mult_eigen(matrix_t<T> const* A, matrix_t<T> const* B,
                           matrix_t<T> *C, unsigned int size) {
    int bid = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int imat = bid * blockDim.x * blockDim.y;
    int ielem = ty * blockDim.x + tx;

    __shared__ matrix_t<T> As;
    __shared__ matrix_t<T> Bs;

    // would like to just do As
    As(ty, tx) = A[bid](ty, tx);
    Bs(ty, tx) = B[bid](ty, tx);
}

template<typename T>
__global__
void kernel_mat_mult_eigen(matrix_t<T> const* A, matrix_t<T> const* B,
                           matrix_t<T> *C, unsigned int size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < size) {
        C[tid] = A[tid] * B[tid];
    }
}

/*
 * multiplication of N small matrices, where a single matrix can be 
 * considered as a single block
 */
template<typename T, int BLOCK_SIZE>
__global__
void kernel_mat_mult(T const* A, T const* B, 
                     T *C, unsigned int size) {
    int bid = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int imat = bid * blockDim.x * blockDim.y;
    int ielem = ty * blockDim.x + tx;
    
    // shared memory - dynamically allocated at runtime
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

    // load into shared mem
    As[ty][tx] = A[imat + ielem];
    Bs[ty][tx] = B[imat + ielem];

    // sync to make sure all elements are loaded
    __syncthreads();

    // compute the value for (ty, tx)
    T result{static_cast<T>(0.0)};
#pragma unroll
    for (unsigned int i=0; i<BLOCK_SIZE; i++) {
        result += As[ty][i] * Bs[i][tx];
    }

    // store the result back to the global memory
    C[imat + ielem] = result;
}

template<typename T>
std::vector<T> compute_values(std::vector<T> const& A,
                              std::vector<T> const& B, int n, int dim) {
    std::vector<T> result(n * dim * dim, 0);
    for (unsigned int imat=0; imat<n; imat++) {
        int mat_offset = imat * dim * dim;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++) {
                T tmp = 0;
                for (int k=0; k<dim; k++) {
                    tmp += A[mat_offset + i*dim + k] * B[mat_offset + k*dim + j];
                }
                result[mat_offset + i*dim + j] = tmp;
            }
    }

    return result;
}

template<typename T>
void test_mult(unsigned int n) {
    std::vector<T> h_A(n * dim * dim);
    std::vector<T> h_B(n * dim * dim);
    std::vector<T> h_C(n * dim * dim, 0);
    std::vector<T> h_test(n * dim * dim, 0);
    T *d_A{nullptr}, *d_B{nullptr}, *d_C{nullptr};
    std::vector<matrix_t<T>> h_eA(n), h_eB(n), h_eC(n);
    matrix_t<T> *d_eA, *d_eB, *d_eC;

    cudaEvent_t startEvent, stopEvent;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // fill in the host vectors
    for (auto &value : h_A)
        value = std::rand() % 10;
    for (auto &value : h_B)
        value = std::rand() % 10;
    for (unsigned int imat=0; imat<n; imat++)
        for (unsigned int i=0; i<dim; i++)
            for (unsigned int j=0; j<dim; j++) {
                h_eA[imat](i, j) = h_A[imat*dim*dim + i*dim + j];
                h_eB[imat](i, j) = h_B[imat*dim*dim + i*dim + j];
            }

    // allocate
    cuda::cuda_malloc(d_A, n * dim * dim);
    cuda::cuda_malloc(d_B, n * dim * dim);
    cuda::cuda_malloc(d_C, n * dim * dim);
    cuda::cuda_malloc(d_eA, n);
    cuda::cuda_malloc(d_eB, n);
    cuda::cuda_malloc(d_eC, n);
    cuda::assert_if_error("checking cuda mallocs");

    // copy to device
    cuda::copy_to_dev(d_A, h_A);
    cuda::copy_to_dev(d_B, h_B);
    cuda::copy_to_dev(d_eA, h_eA);
    cuda::copy_to_dev(d_eB, h_eB);
    cuda::assert_if_error("checking copying host to device");

    {
        dim3 threads_per_block{dim, dim};
        dim3 blocks{n};
        cudaEventRecord(startEvent, 0);
        kernel_mat_mult<T, dim><<<blocks, threads_per_block>>>(
            d_A, d_B, d_C, n);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float ms;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        printf("my impl kernel runtime (ms) = %f\n", ms);
        cuda::assert_if_error("checking vsum kernel");
    }
    
    {
        dim3 threads_per_block{256};
        dim3 blocks{(n + threads_per_block.x - 1) / threads_per_block.x};
        cudaEventRecord(startEvent, 0);
        kernel_mat_mult_eigen<T><<<blocks, threads_per_block>>>(
            d_eA, d_eB, d_eC, n);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float ms;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        printf("eigen impl kernel runtime (ms) = %f\n", ms);
        cuda::assert_if_error("checking vsum kernel");
    }

    // copy back
    cuda::copy_to_host(h_C, d_C);
    cuda::copy_to_host(h_eC, d_eC);
    cuda::assert_if_error("checking copying device to host");

    // validate
    auto start = std::chrono::high_resolution_clock::now();
    auto test_values = compute_values(h_A, h_B, n, dim);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "cpu runtime (ms) = " << elapsed.count() << std::endl;
    auto is_valid = validation::validate_vectors(test_values, h_C);
    if (is_valid)
        std::cout << "results are valid" << std::endl;
    else 
        std::cout << "results are invalid" << std::endl;
    std::vector<T> eigen_test; eigen_test.reserve(n*dim*dim);
    for (auto const& m : h_eC)
        for (unsigned int i=0; i<dim; i++)
            for (unsigned int j=0; j<dim; j++)
                eigen_test.push_back(m(i, j));
    is_valid = validation::validate_vectors(eigen_test, h_C);
    if (is_valid)
        std::cout << "eigen results are valid" << std::endl;
    else 
        std::cout << "eigen results are invalid" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_eA);
    cudaFree(d_eB);
    cudaFree(d_eC);
}

int main(int argc, char** argv) {
    std::srand(0);
    int nmatrices = std::atoi(argv[1]);
    test_mult<float>(nmatrices);
    return 0;
}
