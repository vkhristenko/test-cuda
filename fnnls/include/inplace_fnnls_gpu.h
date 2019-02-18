#ifndef ONE_PASS_FNNLS_H
#define ONE_PASS_FNNLS_H

#include "data_types.h"

#define NNLS_DEBUG
#undef NNLS_DEBUG

template<typename T>
__host__ __device__
void
gpu_inplace_fnnls(matrix_t<T> const& A,
                  vector_t<T> const& b,
                  vector_t<T>& x,
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
__host__ __device__
void gpu_transpose_multiply(matrix_t<T>& AtA, matrix<T> const& A) {

}

template<typename T>
__host__ __device__
void gpu_transpose_multiply(matrix_t<T>& Atb, matrix_t<T> const& A, 
                            vector_t<T> const& b) {

}

template<typename T, unsigned int NCHANNELS>
__host__ __device__
void gpu_inplace_fnnls(matrix_t<T> const& A,
                       vector_t<T> const& b,
                       vector_t<T>& x,
                       const double eps,
                       const unsigned int max_iterations) {
//  matrix_t<data_type> AtA = transpose_multiply(A);
#ifdef NNLS_DEBUG
  std::cout << "A = \n" << A << std::endl;
#endif
//  matrix_t<data_type> AtA = transpose_multiply(A);
//  matrix_t<data_type> AtA = A.transpose() * A;
//  FixedMatrix AtA = A.transpose() * A;
//  vector_t<data_type> Atb = A.transpose() *b;

  Eigen::PermutationMatrix<VECTOR_SIZE> permutation;
  permutation.setIdentity();

  // 
  int imatrix = ...;
  int ix = ...;
  int iy = ...;
  __shared__ matrix_t<T> sAtA[NCHANNELS];
  __shared__ vector_t<T> sAtb[NCHANNELS];
  __shared__ vector_t<T> ss[NCHANNELS];
  __shared__ vector_t<T> sw[NCHANNELS];
  __shared__ vector_t<T> sx[NCHANNELS];
  __shared__ unsigned int iterations[NCHANNELS];
  __shared__ unsigned int nsatisfied[NCHANNELS];

  gpu_transpose_multiply(AtA, A);
  gpu_transpose_multiply(Atb, A, b);

  while (iterations[imatrix] < max_iterations) {
      // compute the update
      const auto nsatisfied_per_matrix = nsatisfied[imatrix];
      const auto nunsatisfied_per_matrix = MATRIX_SIZE - nunsatisfied_per_matix;
      // need to select threads to use
      //if (...)
      if (ix==0) {
          T AtAx_vector_element = 0.0;
#pragma unroll
          for (unsigned int i=0; i<MATRIX_SIZE; i++)
              AtAx_vector_element += AtA(iy, i) * sx(i);
          sw(iy) = Atb(iy) - AtAx_vector_element;
      }
      __syncthreads();

      // find the direction of max update -> w coeff
      Index w_max_idx;
      T w_max;
      if (ix==0 && iy==0) {
          w_max = sw.tail(nActive)
      }
      __syncthreads();

      while (npassive[imatrix] > 0) {

          // decrement the #passive elements in the set
          // sync threads right after
          // TODO: use cooperative groups to sync threads per matrix
          if (ix==0 && iy==0)
              npassive[imatrix]--;
          __syncthreads();
      }

      // increment the iterations only for a single thread per matrix
      // and then synchronize threads not to get data races;
      // TODO: try to use cooperative thread groups to sync 
      //       only threads per matrix, not for the whole block
      if (ix==0 && iy==0)
          iterations[imatrix]++;
      __syncthreads();
  }

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
}

#endif
