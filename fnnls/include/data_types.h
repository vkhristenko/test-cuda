#ifndef NNLS_DATA_TYPES
#define NNLS_DATA_TYPES

#include <Eigen/Dense>

constexpr unsigned long MATRIX_SIZE = 10;
constexpr unsigned long VECTOR_SIZE = 10;

template<typename T>
using matrix_t = Eigen::Matrix<T, MATRIX_SIZE, MATRIX_SIZE, Eigen::ColMajor>;

template<typename T>
using vector_t = Eigen::Matrix<T, VECTOR_SIZE, 1>;

template<typename T>
using permutation_matrix_t = Eigen::PermutationMatrix<VECTOR_SIZE>;

#endif
