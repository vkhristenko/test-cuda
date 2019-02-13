#ifndef common_h
#define common_h

namespace cuda {

template<typename T>
void cuda_malloc(T *ptr, size_t size) {
    cudaMalloc((void**)&ptr, sizeof(T) * size);
}

template<typename T>
void copy_to_dev(T *dest, T const*src, size_t size) {
    cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyHostToDevice);
}

template<typename T>
void copy_to_dev(T *dest, std::vector<T> const& src, size_t size) {
    copy_to_dev<T>(dest, src.data(), size);
}

template<typename T>
void copy_to_host(T *dest, T const* src, size_t size) {
    cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyDeviceToHost);
}

template<typename T>
void copy_to_host(std::vector<T> &dest, T const* src, size_t size) {
    copy_to_host<T>(dest.data(), src, size);
}

}

#endif
