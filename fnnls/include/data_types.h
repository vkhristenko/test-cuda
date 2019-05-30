#ifndef NNLS_DATA_TYPES
#define NNLS_DATA_TYPES

#include <Eigen/Dense>

constexpr unsigned long MATRIX_SIZE = 10;
constexpr unsigned long VECTOR_SIZE = 10;

template<typename T>
struct my_vector {
    T *data;
    __forceinline__ __device__ 
    my_vector(T* data) : data{data} {}

    __forceinline__ __device__ 
    T& operator()(int i) { return data[i]; }
    __forceinline__ __device__ 
    T const& operator()(int i) const { return data[i]; }
};

template<int E = Eigen::RowMajor>
struct EigenToMine {
    static constexpr int value = 0;
};

template<>
struct EigenToMine<Eigen::ColMajor> {
    static constexpr int value = 1;
};

template
<
    typename T, int Stride,
    int order = EigenToMine<Eigen::RowMajor>::value
>
struct my_matrix {
    T *data;
    __forceinline__ __device__ 
    my_matrix(T* data) : data{data} {}

    __forceinline__ __device__ 
    T& operator()(int row, int col) { return data[row*Stride + col]; }
    
    __forceinline__ __device__ 
    T const& operator()(int row, int col) const { return data[row*Stride + col]; }
};

template<typename T, int Stride>
struct my_matrix<T, Stride, EigenToMine<Eigen::ColMajor>::value> {
    T *data;
    __forceinline__ __device__ 
    my_matrix(T *data) : data{data} {}

    __forceinline__ __device__ 
    T& operator()(int row, int col) { return data[col*Stride + row]; }

    __forceinline__ __device__ 
    T const& operator()(int row, int col) const { return data[col*Stride + row]; }
};

template<typename T>
using matrix_t = Eigen::Matrix<T, MATRIX_SIZE, MATRIX_SIZE, Eigen::ColMajor>;

template<typename T>
using my_matrix_t = my_matrix<T, MATRIX_SIZE, EigenToMine<Eigen::ColMajor>::value>;

template<typename T>
using vector_t = Eigen::Matrix<T, VECTOR_SIZE, 1>;

template<typename T>
using my_vector_t = my_vector<T>;

template<typename T>
using permutation_matrix_t = Eigen::PermutationMatrix<VECTOR_SIZE>;

#endif
