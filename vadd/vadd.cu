#include <iostream>
#include <vector>
#include <cassert>
#include "../common/utils.h"

template<typename T>
__global__
void kernel_vadd(T const* as, T const* bs, T *cs, unsigned int size) {
    auto idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < size) {
        cs[idx] = as[idx] + bs[idx];
    }
}

template<typename T>
void test_add(unsigned int size) {
    std::vector<T> h_as(size), h_bs(size), h_cs(size, 0), result(size);
    T *d_as{nullptr}, *d_bs{nullptr}, *d_cs{nullptr};

    // allocate
    cuda::cuda_malloc(d_as, size);
    cuda::cuda_malloc(d_bs, size);
    cuda::cuda_malloc(d_cs, size);
    cuda::assert_if_error("checking cuda mallocs");

    // fill the input vectors
    for (unsigned int i=0; i<size; i++) {
        h_as[i] = T{static_cast<int>(i)};
        h_bs[i] = T{static_cast<int>(i)};
        result[i] = h_as[i] + h_bs[i];
    }

    // copy to device
    cuda::copy_to_dev(d_as, h_as);
    cuda::copy_to_dev(d_bs, h_bs);
    cuda::assert_if_error("checking copying host to device");

    int nthreads_per_block = 256;
    int nblocks = (size + nthreads_per_block - 1) / nthreads_per_block;
    kernel_vadd<T><<<nblocks, nthreads_per_block>>>(d_as, d_bs, d_cs, size);
    cuda::assert_if_error("checking vadd kernel");

    // copy back
    cuda::copy_to_host(h_cs, d_cs);
    cuda::assert_if_error("checking copying device to host");

    // validate
    auto is_valid = validation::validate_vectors(result, h_cs);
    assert(is_valid && "incorrect output from vadd kernel");
    std::cout << "results valid" << std::endl;
}

int main(int argc, char** argv) {
    int size = std::atoi(argv[1]);
    test_add<int>(size);
    return 0;
}
