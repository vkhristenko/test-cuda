#ifndef inplace_fnnls_option0_h
#define inplace_fnnls_option0_h

#include "data_types.h"

namespace gpu { namespace option0 {

template<typename T>
__device__
void load_and_precompute(matrix_t<T>& sAtA, vector_t<T>& sAtb,
                matrix_t<T> const& A, vector_t<T> const& b,
                unsigned int ix, unsigned int iy) {
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


template<typename T>
__global__
void kernel_inplace_fnnls(matrix_t<T> const* As,
                       vector_t<T> const* bs,
                       vector_t<T>* xs) {
  constexpr double eps = 1e-11;
  constexpr unsigned int max_iterations = 1000;

  // 1 block(tx, ty) -> 1 matrix (x, y)
  int imatrix = blockIdx.x;
  int ix = threadIdx.x;
  int iy = threadIdx.y;

  // shared mem
//  __shared__ matrix_t<T> sAtA;
  __shared__ T __sAtA[MATRIX_SIZE * MATRIX_SIZE];
  Eigen::Map<matrix_t<T>> sAtA{__sAtA};
  __shared__ T __sAtb[MATRIX_SIZE];
  Eigen::Map<vector_t<T>> sAtb{__sAtb};
  __shared__ T __ss[MATRIX_SIZE];
  Eigen::Map<vector_t<T>> ss{__ss};
  __shared__ T __sw[MATRIX_SIZE];
  Eigen::Map<vector_t<T>> sw{__sw};
  __shared__ T __sx[MATRIX_SIZE];
  Eigen::Map<vector_t<T>> sx{__sx};
  __shared__ unsigned int iterations;
  __shared__ unsigned int npassive; // how many coefficients satisfied the constr
  __shared__ uint8_t permutation[MATRIX_SIZE];
  // permutation.setIdentity();
  if (ix==0)
    permutation[iy] = iy;

  if (ix==0 && iy==0)
      npassive = 0;
  // no explicit sync as it will be done in precompute

  // compute At * A and At * b 
  //load_and_precompute(sAtA, sAtb, As[imatrix], b, ix, iy);
  // load A into sAtA
  // load b into sAtb
  sAtA(iy, ix) = As[imatrix](iy, ix);
  if (ix == 0)
    sAtb(iy) = bs[imatrix](iy);
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

  //
  // main loop
  //
  __syncthreads();
  while (iterations < max_iterations) {
      auto nactive = MATRIX_SIZE - npassive; // # coeff not fixed

      // compute the update
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
      __shared__ Index w_max_idx;
      __shared__ T w_max;
      if (ix==0 && iy==0) {
          w_max = sw.tail(nactive).maxCoeff(&w_max_idx);
      }
      __syncthreads();

      // upon satisfying this condition, all threads of the block should jump out
      // TODO: does my logic hold?
      if (w_max < eps)
          break;

      if (ix==0 && iy==0)
        w_max_idx += npassive;
      __syncthreads();

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
      if (ix==npassive && iy==w_max_idx)
          Eigen::numext::swap(sAtb.coeffRef(npassive), sAtb.coeffRef(w_max_idx));

      // x swap
      if (ix == npassive && iy==w_max_idx)
          Eigen::numext::swap(sx.coeffRef(npassive), sx.coeffRef(w_max_idx));
      // permutation swap
      if (ix == npassive && iy==w_max_idx)
          Eigen::numext::swap(permutation[npassive],
                              permutation[w_max_idx]);

      if (ix==0 && iy==0)
          npassive++;

      // TODO: anything else missing???
      __syncthreads();

      while (npassive > 0) {
          // TODO: can we do better?
          if (ix==0 && iy==0) {
              ss.head(npassive) = sAtA
                  .topLeftCorner(npassive, npassive)
                  .llt().solve(sAtb.head(npassive));
          }
          __syncthreads();

          // 
          __shared__ bool all_pos;
          if (ix==0 && iy==0) {
              all_pos = false;
              if (ss.head(npassive).minCoeff() > 0.)
                  all_pos = true;
          }
          __syncthreads();
          if (all_pos) { // all threads per block must enter or not enter at all!
              if (ix==0 && iy<npassive)
                  sx(iy) = ss(iy);
              __syncthreads();
              break;
          }

          __shared__ T alpha;
          __shared__ Index alpha_idx;
          
          if (ix==0 && iy==0) {
            alpha = std::numeric_limits<T>::max();
            alpha_idx = 0;

            for (unsigned int i=0; i<npassive; i++) {
                if (ss[i] < 0.) {
                    const auto ratio = sx[i] / (sx[i] - ss[i]);
                    if (ratio < alpha) {
                        alpha = ratio;
                        alpha_idx = i;
                    }
                }
            }
          }
          __syncthreads();
          if (std::numeric_limits<T>::max() == alpha) {
              // by construction, 
              // if 1 thread enters, all threads must enter this basic block
              if (ix==0 && iy<npassive)
                  sx(iy) = ss(iy);
              __syncthreads();
              break;
          }
          
          if (ix==0 && iy<npassive) {
              sx(iy) += alpha * (ss(iy) - sx(iy));
              if (iy==alpha_idx) {
                  sx(alpha_idx) = 0;
                  --npassive;
              }
          }
          __syncthreads();

          // AtA swaps
          if (ix==npassive && iy==npassive) {
              T tmp = sAtA(npassive, npassive);
              sAtA(npassive, npassive) = sAtA(alpha_idx, alpha_idx); 
              sAtA(alpha_idx, alpha_idx) = tmp;
          } else if (ix==npassive && iy!=alpha_idx) {
              T tmp = sAtA(iy, npassive);
              sAtA(iy, npassive) = sAtA(iy, alpha_idx);
              sAtA(iy, alpha_idx) = tmp;
              sAtA(npassive, iy) = sAtA(alpha_idx, iy);
              sAtA(alpha_idx, iy) = tmp;
          }

          // Atb swap
          if (ix==npassive && iy==alpha_idx)
              Eigen::numext::swap(sAtb.coeffRef(npassive), sAtb.coeffRef(alpha_idx));

          // x swap
          if (ix == npassive && iy==alpha_idx)
              Eigen::numext::swap(sx.coeffRef(npassive), sx.coeffRef(alpha_idx));
          // permutation swap
          if (ix == npassive && iy==w_max_idx)
              Eigen::numext::swap(permutation[npassive],
                                  permutation[alpha_idx]); 
      }

      // increment the iterations only for a single thread per matrix
      // and then synchronize threads not to get data races;
      // TODO: try to use cooperative thread groups to sync 
      //       only threads per matrix, not for the whole block
      if (ix==0 && iy==0)
          iterations++;
      __syncthreads();
  }

  // permute and store the result in global mem
  if (ix==0) {
      auto new_idx = permutation[iy];
      xs[imatrix](new_idx) = sx(iy);
  }
  
  // make sure all the guys get here before exiting
  __syncthreads();
}

}}

#endif // inplace_fnnls_option0_h
