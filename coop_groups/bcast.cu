#include <stdio.h>

#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__
void bcast(int arg) {
    thread_block g = this_thread_block();

    g.sync();
    printf("thread %d thread block size %d and thread rank %d\n", 
        threadIdx.x, g.size(), g.thread_rank()
    );

    printf("block idx %d %d %d\n", 
        g.group_index().x, g.group_index().y, g.group_index().z);
    printf("thread idx %d %d %d\n",
        g.thread_index().x, g.thread_index().y, g.thread_index().z);

    thread_group tile32 = tiled_partition(g, 32);
    printf("thread %d thread block size %d and thread rank %d\n", 
        threadIdx.x, tile32.size(), tile32.thread_rank()
    );

    thread_group tile4 = tiled_partition(tile32, 4);
    printf("thread %d thread block size %d and thread rank %d\n", 
        threadIdx.x, tile4.size(), tile4.thread_rank()
    );

    {
        thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
        auto value = tile32.shfl(threadIdx.x, 1);
        printf("tbt thread %d value = %d\n", threadIdx.x, value);
        printf("thread %d thread block size %d and thread rank %d\n", 
            threadIdx.x, tile32.size(), tile32.thread_rank()
        );
    }
}

int main() {
    bcast<<<1, 100>>>(1234);
    cudaDeviceSynchronize();

    return 0;
}
