#ifndef common_h
#define common_h

#include <string>

#include <cuda.h>

namespace cuda {

inline
void assert_if_error(std::string const& msg) {
    auto check = [](auto code) {
        if (code != cudaSuccess) {
            std::cout << "\t\t" << cudaGetErrorString(code) << std::endl;
            //assert(false);
        }   
    };  

    std::cout << msg << std::endl;
    check(cudaGetLastError());
}

template<typename T>
void cuda_malloc(T *&ptr, size_t size) {
    cudaMalloc((void**)&ptr, sizeof(T) * size);
}

template<typename T>
void copy_to_dev(T *dest, T const*src, size_t size) {
    cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyHostToDevice);
}

template<typename T>
void copy_to_dev(T *dest, std::vector<T> const& src) {
    copy_to_dev<T>(dest, src.data(), src.size());
}

template<typename T>
void copy_to_host(T *dest, T const* src, size_t size) {
    cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyDeviceToHost);
}

template<typename T>
void copy_to_host(std::vector<T> &dest, T const* src) {
    copy_to_host<T>(dest.data(), src, dest.size());
}

}

namespace validation {

template<typename T>
bool validate_vectors(std::vector<T> const& a, std::vector<T> const& b) {
    if (a.size() != b.size())
        return false;
    bool result = true;
    for (unsigned int i=0; i<a.size(); i++) {
        result &= a[i] == b[i];
    }
    return result;
}

template<typename M>
std::vector<unsigned int> 
validate_eigen_vectors(std::vector<M> const& a, std::vector<M> const& b) {
    if (a.size() != b.size())
        return {};
    std::vector<unsigned int> wrongs;
    for (unsigned int i=0; i<a.size(); i++) {
        bool result =  a[i].isApprox(b[i]);
        if (!result) wrongs.push_back(i);
    }

    return wrongs;
}

}

#endif
