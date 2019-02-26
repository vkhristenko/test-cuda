#ifndef inplace_fnnls_option1_h
#define inplace_fnnls_option1_h

#include "data_types.h"

#define NNLS_DEBUG
#undef NNLS_DEBUG

namespace gpu { namespace option1 {

template<typename T>
__device__
void
inplace_fnnls(matrix_t<T> const& A,
                  vector_t<T> const& b,
                  vector_t<T>& x,
                  int tx,
                  int tpb,
                  const double eps = 1e-11,
                  const unsigned int max_iterations = 1000);

using namespace Eigen;

template<typename T>
__host__ __device__
inline matrix_t<T> transpose_multiply(matrix_t<T> const& A){
  matrix_t<T> result;
  for(auto i = 0; i < MATRIX_SIZE; ++i){    
    for(auto j = i; j < MATRIX_SIZE; ++j){
      result.data()[j*MATRIX_SIZE + i] = 0;
      for(auto k = 0; k < MATRIX_SIZE; ++k)
        result.data()[j*MATRIX_SIZE + i] += A.data()[i*MATRIX_SIZE+k]*A.data()[j*MATRIX_SIZE+k];
      result.data()[i*MATRIX_SIZE + j] = result.data()[j*MATRIX_SIZE + i];
    }
    // result = result.selfadjointView<Eigen::Upper>();
  }
  return result;
}

template<typename T>
__device__
void inplace_fnnls(matrix_t<T> const& A,
                       vector_t<T> const& b,
                       vector_t<T>& result_x,
                       int tx,
                       int tpb,
                       const double eps,
                       const unsigned int max_iterations) {
  using data_type = T;
  // Fast NNLS (fnnls) algorithm as per
  // http://users.wfu.edu/plemmons/papers/Chennnonneg.pdf
  // page 8

  // FNNLS memorizes the A^T * A and A^T * b to reduce the computation.
  // The pseudo-inverse obtained has the same numerical problems so
  // I keep the same decomposition utilized for NNLS.

  // pseudoinverse (A^T * A)^-1 * A^T
  // this pseudo-inverse has numerical issues
  // in order to avoid that I substituted the pseudoinverse whit the QR
  // decomposition

  // I'm substituting the vectors P and R that represents active and passive set
  // with a boolean vector: if active_set[i] the i belogns to R else to P

  // bool active_set[VECTOR_SIZE];
  // memset(active_set, true, VECTOR_SIZE * sizeof(bool));

  auto nPassive = 0;
  
  /*
  #ifdef __CUDA_ARCH__
  FixedMatrix AtA;
  matrixMultiplication(A, AtA);
  __syncthreads();
  cudaDeviceSynchronize();
  for(auto i = 0; i < MATRIX_SIZE; ++i){
    for(auto j = 0; j < MATRIX_SIZE; ++j)
      printf("%f ", AtA.data()[j* MATRIX_SIZE + i]);
    printf("\n");
  }
  #else
  FixedMatrix AtA = transpose_multiply(A);
  #endif
  assert(AtA == A.transpose() * A);
  */
//  matrix_t<data_type> AtA = transpose_multiply(A);
#ifdef NNLS_DEBUG
  std::cout << "A = \n" << A << std::endl;
#endif
//  matrix_t<data_type> AtA = transpose_multiply(A);
  extern __shared__ T sdata[];
  T* start_sAtA = sdata;
  T* __sAtA = start_sAtA + tx*MATRIX_SIZE*MATRIX_SIZE;

  T* start_sAtb = start_sAtA + tpb*MATRIX_SIZE*MATRIX_SIZE;
  T* __sAtb = start_sAtb + tx*MATRIX_SIZE;

  T* start_sx = start_sAtb + tpb*MATRIX_SIZE;
  T* __sx = start_sx + tx*MATRIX_SIZE;

  T* start_ss = start_sx + tpb*MATRIX_SIZE;
  T* __ss = start_ss + tx*MATRIX_SIZE;

  T* start_sw = start_ss + tpb*MATRIX_SIZE;
  T* __sw = start_sw + tx*MATRIX_SIZE;

  Eigen::Map<matrix_t<T>> AtA{__sAtA};
  AtA = A.transpose() * A;
  Eigen::Map<vector_t<T>> Atb{__sAtb};
  Atb = A.transpose() *b;
  Eigen::Map<vector_t<T>> x{__sx};
  x = vector_t<T>::Zero();
  Eigen::Map<vector_t<T>> s{__ss};
  Eigen::Map<vector_t<T>> w{__sw};

#ifndef __CUDA_ARCH__
#ifdef FNNLS_DEBUG_CPU
    static int __counter__ = 0;
    if (__counter__ == 113) {
    std::cout << "*** AtA ***" << std::endl;
    std::cout << AtA << std::endl;
    std::cout << "*** Atb ***" << std::endl;
    std::cout << Atb << std::endl;
    }
#endif
#endif

//  matrix_t<data_type> AtA = A.transpose() * A;
  // FixedMatrix AtA = A.transpose() * A;
//  vector_t<data_type> s;
//  vector_t<data_type> w;

  Eigen::PermutationMatrix<VECTOR_SIZE> permutation;
  permutation.setIdentity();

#ifdef NNLS_DEBUG
  std::cout << "AtA = \n" << AtA << std::endl;
  std::cout << "Atb = \n" << Atb << std::endl;
#endif

// main loop
  for (auto iter = 0; iter < max_iterations; ++iter) {
    const auto nActive = VECTOR_SIZE - nPassive;

#ifdef NNLS_DEBUG
    std::cout << "***************\n"
        << "iteration = " << iter << std::endl
        << "nactive = " << nActive << std::endl;
    std::cout << "x = \n" << x << std::endl;
#endif

#ifdef DEBUG_FNNLS_CPU
    cout << "iter " << iter << endl;
#endif
    
    if(!nActive)
      break;

#ifdef NNLS_DEBUG
    std::cout << "AtA * x = \n" << AtA*x << std::endl;
#endif

    w.tail(nActive) = Atb.tail(nActive) - (AtA * x).tail(nActive);

#ifdef DEBUG_FNNLS_CPU
    cout << "w" << endl << w.tail(nActive) << endl;
#endif
    // get the index of w that gives the maximum gain
    Index w_max_idx;
    const auto max_w = w.tail(nActive).maxCoeff(&w_max_idx);

#ifdef NNLS_DEBUG
    std::cout << "w = \n" << w << std::endl;
    std::cout << "max_w = " << max_w << std::endl;
    std::cout << "w_max_idx = " << w_max_idx << std::endl;
#endif

    // check for convergence
    if (max_w < eps)
      break;

    // cout << "n active " << nActive << endl;
    // cout << "w max idx " << w_max_idx << endl;

    // need to translate the index into the right part of the vector
    w_max_idx += nPassive;

    // swap AtA to avoid copy
    AtA.col(nPassive).swap(AtA.col(w_max_idx));
    AtA.row(nPassive).swap(AtA.row(w_max_idx));
    // swap Atb to match with AtA
    Eigen::numext::swap(Atb.coeffRef(nPassive), Atb.coeffRef(w_max_idx));
    Eigen::numext::swap(x.coeffRef(nPassive), x.coeffRef(w_max_idx));
    // swap the permutation matrix to reorder the solution in the end
    Eigen::numext::swap(permutation.indices()[nPassive],
                        permutation.indices()[w_max_idx]);

#ifdef NNLS_DEBUG
    std::cout << "permutation = \n" << permutation.indices() << std::endl;
#endif

    ++nPassive;

#ifdef DEBUG_FNNLS_CPU
    cout << "max index " << w_max_idx << endl;
    std::cout << "n_active " << nActive << std::endl;
#endif

// inner loop
    while (nPassive > 0) {
      s.head(nPassive) =
          AtA.topLeftCorner(nPassive, nPassive).llt().solve(Atb.head(nPassive));

#ifdef NNLS_DEBUG
      std::cout << "s = \n" << s << std::endl;
#endif

      if (s.head(nPassive).minCoeff() > 0.) {
        x.head(nPassive) = s.head(nPassive);
        break;
      }

#ifdef DEBUG_FNNLS_CPU
      cout << "s" << endl << s.head(nPassive) << endl;
#endif

      auto alpha = std::numeric_limits<double>::max();
      Index alpha_idx = 0;

#pragma unroll VECTOR_SIZE
      for (auto i = 0; i < nPassive; ++i) {
        if (s[i] <= 0.) {
          auto const ratio = x[i] / (x[i] - s[i]);
          if (ratio < alpha) {
            alpha = ratio;
            alpha_idx = i;
          }
        }
      }

      if (std::numeric_limits<double>::max() == alpha) {
        x.head(nPassive) = s.head(nPassive);
        break;
      }

#ifdef DEBUG_FNNLS_CPU

      cout << "alpha " << alpha << endl;

      cout << "x before" << endl << x << endl;

#endif

      x.head(nPassive) += alpha * (s.head(nPassive) - x.head(nPassive));
      x[alpha_idx] = 0;
      --nPassive;

#ifdef DEBUG_FNNLS_CPU
      cout << "x after" << endl << x << endl;
#endif
      AtA.col(nPassive).swap(AtA.col(alpha_idx));
      AtA.row(nPassive).swap(AtA.row(alpha_idx));
      // swap Atb to match with AtA
      Eigen::numext::swap(Atb.coeffRef(nPassive), Atb.coeffRef(alpha_idx));
      Eigen::numext::swap(x.coeffRef(nPassive), x.coeffRef(alpha_idx));
      // swap the permutation matrix to reorder the solution in the end
      Eigen::numext::swap(permutation.indices()[nPassive],
                          permutation.indices()[alpha_idx]);

    }
  }
  
  x = x.transpose() * permutation.transpose();  
  result_x = x;
}

}}

#endif