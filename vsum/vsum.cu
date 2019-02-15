#include <iostream>
#include <vector>
#include <cassert>
#include "../common/utils.h"

template<typename T>
__global__
void kernel_vsum(T* values, unsigned int size) {
    auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    auto local_tid = threadIdx.x;
    if (tid < size) {
        // size of the shared memory will be configured at runtime
        extern __shared__ T s_data[];

        // load data into the shared memory
        s_data[local_tid] = values[tid];

        // sync all threads for this block
        __syncthreads();

        // sum up per block
        int i = blockDim.x / 2;
        while (i != 0) {
            if (local_tid < i) 
                s_data[local_tid] += s_data[local_tid + i];
            __syncthreads();
            i /= 2;
        }

        // store into the global memory
        values[blockIdx.x] = s_data[0];
    }
}

template<typename T>
void test_sum(unsigned int size) {
    std::vector<T> h_values(size);
    T* d_values{nullptr};

    // allocate
    cuda::cuda_malloc(d_values, size);
    cuda::assert_if_error("checking cuda mallocs");

    // fill the input vectors
    long long sum = 0;
    for (unsigned int i=0; i<size; i++) {
        h_values[i] = i;
        sum += i;
    }

    // copy to device
    cuda::copy_to_dev(d_values, h_values);
    cuda::assert_if_error("checking copying host to device");

    int nthreads_per_block = 256;
    int nblocks = (size + nthreads_per_block - 1) / nthreads_per_block;
    kernel_vsum<T><<<nblocks, nthreads_per_block, nthreads_per_block * sizeof(T)>>>(
        d_values, size);
    cuda::assert_if_error("checking vsum kernel");

    // copy back
    cuda::copy_to_host(h_values, d_values);
    cuda::assert_if_error("checking copying device to host");

    // compute the result
    long long test_sum = 0;
    for (unsigned int i=0; i<nblocks; i++) {
        test_sum += h_values[i];
    }

    // validate
    bool is_valid = (test_sum == sum);
    if (!is_valid)
        std::cout << "invalid results: " << test_sum << " vs " << sum << std::endl;
    else
        std::cout << "results valid " << test_sum << " vs " << sum << std::endl;

    cudaFree(d_values);
}

int main(int argc, char** argv) {
    int size = std::atoi(argv[1]);
    test_sum<int>(size);
    return 0;
}
