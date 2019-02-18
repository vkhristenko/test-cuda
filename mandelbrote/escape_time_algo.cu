#define MAX_DWELL 512

// w,h ---- width and hight of the image in pixels
// cmin, cmax --- coordinates of bottom-left and top-right image
// x, y --- coordinates of the pixel
__host__ __device__ 
int pixel_dwell(int w, int h,
                complex cmin, complex cmax) {
    complex dc = cmax - cmin;
    float fx = (float)x / w, fy = (float)y/h;
    complex c = cmin + complex(fx * dc.re, fy*dc.im);
    complex z = c;
    int dwell = 0;

    while (dwell < MAX_DWELL && abs2(z) < 2*2) {
        z = z*z + c;
        dwell++;
    }

    return dwell;
}

// kernel
__global__ void mandelbrot_k(int* dwells, int w, int h,
                             complex cmin, complex cmax) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < w && y < h)
        dwells[y*w + x] = pixel_dwell(w, h, cmin, cmax, x, y);
}

int main(void) {
    // details omitted

    // kernel launch
    int w = 4096, h = 4096;
    dim3 bs(64, 4), grid(divup(w, s.x), divup(h, bs.y));
    mandelbrot_k<<<grid, bs>>>(d_dwells, w, h,
                               complex(-1.5, 1), complex(0.5, 1));

    return 0;
}
