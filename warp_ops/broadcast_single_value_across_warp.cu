#include <stdio.h>

__global__ 
void bcast(int arg) {
    int laneId = threadIdx.x & 0x1f;
    int value; 

    if (laneId == 0)
        value = arg;
    value = __shfl_sync(0xffffffff, value, 0);

    if (value != arg)
        printf("thread %d failed.\n", threadIdx.x);
    printf("thread %d value %d\n", threadIdx.x, value);
}

int main() {
    bcast<<<1, 100>>>(1234);
    cudaDeviceSynchronize();

    return 0;
}
