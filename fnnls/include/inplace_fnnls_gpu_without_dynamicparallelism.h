#ifndef ONE_PASS_FNNLS_H
#define ONE_PASS_FNNLS_H

#include "data_types.h"

#define NNLS_DEBUG
#undef NNLS_DEBUG

namespace nodynpar {

template<typename T>
__host__ __device__
void
gpu_inplace_fnnls(matrix_t<T> const& A,
                  vector_t<T> const& b,
                  vector_t<T>& x,
                  const double eps = 1e-11,
                  const unsigned int max_iterations = 1000);

template<typename T>
__host__ __device__
void precompute(matrix_t<T>& sAtA, vector_t<T>& sAtb,
                matrix_t<T> const& A, vector_t<T> const& b) {
    // load A into sAtA
    // load b into sAtb
    sAtA(iy, ix) = A(iy, ix);
    if (ix == 0)
        sAtb(iy) = b(iy);
    __syncthreads();

    // compute Atb, use threads with ix==0
    T result{0.0};
    if (ix==0) {
        for (unsigned int i=0; i<MATRIX_SIZE; i++)
            result += sAtA(iy, i) * sAtb(i);
    }
    __syncthreads();
    if (ix==0)
        sAtb(iy) == result;

    // compute AtA 
    T result1{0.0};
    for (unsigned int i=0; i<MATRIX_SIZE; i++)
        result1+= sAtA(iy, i) * sAtA(i, ix);
    __syncthreads();
    sAtA(iy ,ix) = result1;
}


template<typename T, unsigned int NCHANNELS>
__host__ __device__
void gpu_inplace_fnnls(matrix_t<T> const& A,
                       vector_t<T> const& b,
                       vector_t<T>& x,
                       const double eps,
                       const unsigned int max_iterations) {
  // permutation
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

  precompute(sAtA, sAtb, A, b);

  while (iterations[imatrix] < max_iterations) {
      // compute the update
      auto npassive = nsatisfied[imatrix];
      auto nactive = MATRIX_SIZE - nunsatisfied;
      // need to select threads to use
      //if (...)
      if (ix==0 && iy>=npassive) {
          T result = 0.0;
#pragma unroll
          for (unsigned int i=0; i<MATRIX_SIZE; i++)
              result += sAtA(iy, i) * sx(i);
          sw(iy) = sAtb(iy) - result;
      }
      __syncthreads();

      // find the direction of max update -> w coeff
      // TODO: does it work using __shared__ guys below???
      __shared__ Index w_max_idx[NCHANNELS];
      __shared__ T w_max[NCHANNELS];
      if (ix==0 && iy==0) {
          w_max[imatrix] = sw.tail(nactive).maxCoeff(&w_max_idx[imatrix]);
          w_max_idx[imatrix] += npassive;
      }
      __syncthreads();

      // check for convergence
      // TODO: note max_w lives in shared mem, and is used for control
      // branching here will lead to all the threads for that matrix to exit the loop
      if (max_w[imatrix] < eps)
          break;

      // AtA swaps
      if (ix==npassive && iy==npassive) {
          T tmp = sAtA(npassive, npassive);
          sAtA(npassive, npassive) = sAtA(w_max_idx, w_max_idx); 
          sAtA(w_max_idx, w_max_idx) = tmp;
      } else if (ix==npassive && iy!=w_max_idx) {
          T tmp = sAtA(iy, npassive);
          sAtA(iy, npassive) = sAtA(iy, w_max_idx);
          sAtA(iy, w_max_idx) = tmp;
          sAtA(npassive, iy) = sAtA(w_max_idx, iy);
          sAtA(w_max_idx, iy) = tmp;
      }

      // Atb swap
      if (ix==npassive && iy==w_max_idx) {
          T tmp = sAtb(ix);
          sAtb(ix) = sAtb(iy);
          sAtb(iy) = tmp;
      }

      // x swap
      if (ix == npassive & iy==w_max_idx) {
          T tmp = sx(ix);
          sx(ix) = sx(iy);
          sx(iy) = tmp;
      }

      if (ix==0 && iy==0) {
          npassive = (++nsatisfied[imatrix]);
      }

      // TODO: anything else missing???
      __syncthreads();

      while (npassive > 0) {
          if (ix==0 && iy==0)
              sx.head(npassive) = sAtA
                  .topLeftCorner(npassive, npassive)
                  .llt().solve(sAtb.head(npassive));

          // decrement the #passive elements in the set
          // sync threads right after
          // TODO: use cooperative groups to sync threads per matrix
          if (ix==0 && iy==0)
              npassive = (--satisfied[imatrix]);
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
}

}

#endif
