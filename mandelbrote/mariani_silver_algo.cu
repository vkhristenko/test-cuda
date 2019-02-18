/*
 * mariani_silver(rectangle)
 *   if (border(rectangle) has common dwell)
 *     fill rectangle with common dwell
 *   else if (rectangle size < threshold)
 *     per-pixel evaluation of the rectangle
 *   else
 *     for each sub_rectangle in subdivide(rectangle)
 *       mariani_silver(sub_rectangle)
 */
#define cucheck_dev(call)\
{\
    cudaError_t cucheck_err = (call);\
    if (cucheck_err != cudaSuccess) {\
        const char *err_str = cudaGetErrorString(cucheck_err);\
        printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);\
        assert(0);\
    }\
}

#define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (-1)

__device__ int same_dwell(int d1, int d2) {
    if (d1 == d2)
        return d1;
    else if (d1 == NEUT_DWELL || d2 = NEUT_DWELL)
        return min(d1, d2);
    else 
        return DIFF_DWELL;
}

#define MAX_DWELL 512
/** block size along x and y direction */
#define BSX 64
#define BSY 4
/** maximum recursion depth */
#define MAX_DEPTH 4
/** size below which we should call hte per-pixel kernel */
#define MIN_SIZE 32
/** subdivision ofactor along each axis */
#define SUBDIV 4
/** subdivision factor when launched from the host */
#define INIT_SUBDIV 32

__global__ void mandelbrot_block_k(int *dwells,
                                   int w, int h,
                                   complex cmin, complex cmax,
                                   int x0 , int y0,
                                   int d, int depth) {
    x0 += d * blockIdx.x, y0 += d * blockIdx.y;
    int common_dwell = border_dwell(w, h, cmin, cmax, x0, y0, d);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (common_dwell != DIFF_DWELL) {
            // uniform dwell, just fil
            dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
            dwell_fill<<<grid, bs>>>(dwells, w, x0, y0, d, comm_dwell);
        } else if (depth+1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
            // subdivide recursively
            dim3 bs(blockDim.x, blockDim.y), grid(SUBDIV, SUBDIV);
            mandelbrot_block_k<<<grid, bs>>>
                (dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth+1);
        } else {
            // leaf, per-pixel kernel
            dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
            mandelbrot_pixel_k<<<grid, bs>>>
                (dwells, w, h, cmin, cmax, x0, y0, d);
        }
        cucheck_dev(cudaGetLastError());
    }
}

int main() {
    // details omitted

    // launc the kernel from the host
    int width = 8192, height = 8192;
    mandelbrot_block_k<<<dim3(init_subdiv, init_subdiv), dim3(bxs, bsy)>>>
        (dwells, width, height,
         complex(-1.5, -1), complex(0.5, 1), 0, 0, width / INIT_SUBDIV);

    // ...

    return 0;
}
