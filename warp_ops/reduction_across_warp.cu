#include <stdio.h>

__global__
void warpReduct() {
    int laneId = threadIdx.x & 0x1f;
    int value = 31 - laneId;

    // use xor mode to perfrom butterfly reduction
    for (int i=16; i>=1; i/=2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);

    printf("thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    warpReduct<<<1, 32>>>();
    cudaDeviceSynchronize();

    return 0;
}
