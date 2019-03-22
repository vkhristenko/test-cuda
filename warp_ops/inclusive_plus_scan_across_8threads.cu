#include <stdio.h>

__global__
void scan4() {
    int laneId = threadIdx.x & 0x1f;
    // seed sample starting value (inverse of laneid)
    int value = 31 - laneId;

    for (int i=1; i<=4; i*=2) {
        int n = __shfl_up_sync(0xffffffff, value, i, 8);
        if ((laneId & 7) >= i)
            value += n;
    }

    printf("thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    scan4<<<1, 32>>>();
    cudaDeviceSynchronize();

    return 0;
}
