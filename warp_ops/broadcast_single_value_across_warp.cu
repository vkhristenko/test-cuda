#include <stdio.h>

__global__ 
void bcast(int arg) {
    int laneId = threadIdx.x & 0x1f;
    int value = threadIdx.x;

    /*
    if (laneId == 0)
        value = arg;
        */

    {
        value = __shfl_sync(0xffffffff, value, 1);
        printf("thread %d value %d\n", threadIdx.x, value);
    }

    {
        auto value = __shfl_up_sync(0xffffffff, threadIdx.x, 1);
        printf("thread %d value %d\n", threadIdx.x, value);
    }

    {
        auto value = __shfl_down_sync(0xffffffff, threadIdx.x, 1);
        printf("thread %d value %d\n", threadIdx.x, value);
    }
}

int main() {
    bcast<<<1, 100>>>(1234);
    cudaDeviceSynchronize();

    return 0;
}
