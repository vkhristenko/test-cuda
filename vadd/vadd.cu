#include <iostream>
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
    std::vector<T> h_as, h_bs, h_cs;
    T *d_as, *d_bs, *d_cs;

    // allocate
    cuda::cuda_malloc(d_as);
    cuda::cuda_malloc(d_bs);
    cuda::cuda_malloc(d_cs);

    // copy to device
    cuda::copy_to_dev(d_as, h_as, size);
    cuda::copy_to_dev(d_bs, h_bs, size);

    int nthreads_per_block = 256;
    int nblocks = (size + nthreads_per_block - 1) / nthreads_per_block;
    kernel_vadd<T><<<nblocks, nthreads_per_block>>>(d_as, d_bs, d_cs, size);

    // copy back
    cuda::copy_to_host(h_cs, d_cs, size);
}

int main(int argc, char** argv) {
    int size = std::stoi(argv[1]);
    test_add<int>(size);
    return 0;
}
