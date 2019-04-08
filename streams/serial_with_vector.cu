#include <vector>

const int N = 1 << 20;

__global__ void kernel(float const* a, float const* b, float* c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int num_streams = 8;

    cudaStream_t streams[num_streams];
//    std::vector<float> h_a[num_streams],
//                       h_b[num_streams],
//                       h_c[num_streams];
    float *h_a[num_streams], *h_b[num_streams], *h_c[num_streams];
//    std::vector<float> h_a[num_streams], h_b[num_streams], h_c[num_streams];
    float *d_a[num_streams],
          *d_b[num_streams],
          *d_c[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_a[i], N * sizeof(float));
        cudaMalloc(&d_b[i], N * sizeof(float));
        cudaMalloc(&d_c[i], N * sizeof(float));

/*        h_a[i].resize(N);
        h_b[i].resize(N);
        h_c[i].resize(N);
        */

        cudaMallocHost(&h_a[i], N*sizeof(float));
        cudaMallocHost(&h_c[i], N*sizeof(float));
        cudaMallocHost(&h_b[i], N*sizeof(float));

        for (unsigned int iii=0; iii<N; iii++) {
            h_a[i][iii] = static_cast<float>(iii);
            h_b[i][iii] = static_cast<float>(iii);
        }

    }

    for (int i=0; i<num_streams; i++) {

        /*
        cudaMemcpyAsync(d_a[i], h_a[i].data(),
                        N*sizeof(float),
                        cudaMemcpyHostToDevice,
                        streams[i]);
        cudaMemcpyAsync(d_b[i], h_b[i].data(),
                        N*sizeof(float),
                        cudaMemcpyHostToDevice,
                        streams[i]);
        */
        cudaMemcpyAsync(d_a[i], h_a[i],
                        N*sizeof(float),
                        cudaMemcpyHostToDevice,
                        streams[i]);
        cudaMemcpyAsync(d_b[i], h_b[i],
                        N*sizeof(float),
                        cudaMemcpyHostToDevice,
                        streams[i]);
    }

    for (int i=0; i<num_streams; i++) {
                                                
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], N);
    }

    for (int i=0; i<num_streams; i++) {
        /*
        cudaMemcpyAsync(h_c[i].data(), d_c[i],
                        N*sizeof(float),
                        cudaMemcpyDeviceToHost,
                        streams[i]);
        */
        cudaMemcpyAsync(h_c[i], d_c[i],
                        N*sizeof(float),
                        cudaMemcpyDeviceToHost,
                        streams[i]);
    }

    cudaDeviceReset();
    return 0;
}

