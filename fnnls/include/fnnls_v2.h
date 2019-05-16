#ifndef fnnls_v2_h
#define fnnls_v2_h

#include "data_types.h"

#define NNLS_DEBUG
#undef NNLS_DEBUG

namespace v2 {

template<typename T>
__device__
void
fnnls(matrix_t<T> const& A,
                  vector_t<T> const& b,
                  vector_t<T>& x,
                  const double eps = 1e-11,
                  const unsigned int max_iterations = 1000);

using namespace Eigen;

template<typename T>
__global__
void kernel_compute_update_vector() {

    w.tail(nActive) = Atb.tail(nActive) - (AtA * x).tail(nActive);
    Index w_max_idx; 
    const auto max_w = w.tail(nActive).maxCoeff(&w_max_idx);
         
    // check for convergence
    if (max_w < eps)
    break;
    
    // need to translate the index into the right part of the vector
    w_max_idx += nPassive;
        62     
             63     // swap AtA to avoid copy
              64     AtA.col(nPassive).swap(AtA.col(w_max_idx));
         65     AtA.row(nPassive).swap(AtA.row(w_max_idx));
          66     // swap Atb to match with AtA
               67     Eigen::numext::swap(Atb.coeffRef(nPassive), Atb.coeffRef(w_max_idx));
           68     Eigen::numext::swap(x.coeffRef(nPassive), x.coeffRef(w_max_idx));

}

template<typename T>
__device__
void fnnls(matrix_t<T>& AtA,
           vector_t<T>& Atb,
           vector_t<T>& x,
           vector_t<T>& s,
           vector_t<T>& w,
           const double eps,
           const unsigned int max_iterations) {
  using data_type = T;

  auto nPassive = 0;
  
// main loop
  for (auto iter = 0; iter < max_iterations; ++iter) {

    const auto nActive = VECTOR_SIZE - nPassive;
    if(!nActive)
      break;

    // launch 1 kernel per each thread
    kernel_compute_update_vector<<<...>>>();
    cudaDeviceSynchronize();

    auto const max_w = g_max_w[idx];
    auto const max_w_idx = g_max_w_idx[idx];

    // check for convergence
    if (max_w < eps)
      break;

    // need to translate the index into the right part of the vector
    w_max_idx += nPassive;

    // swap AtA to avoid copy
    AtA.col(nPassive).swap(AtA.col(w_max_idx));
    AtA.row(nPassive).swap(AtA.row(w_max_idx));
    // swap Atb to match with AtA
    Eigen::numext::swap(Atb.coeffRef(nPassive), Atb.coeffRef(w_max_idx));
    Eigen::numext::swap(x.coeffRef(nPassive), x.coeffRef(w_max_idx));

    ++nPassive;

// inner loop
    while (nPassive > 0) {
      s.head(nPassive) =
          AtA.topLeftCorner(nPassive, nPassive).llt().solve(Atb.head(nPassive));

      if (s.head(nPassive).minCoeff() > 0.) {
        x.head(nPassive) = s.head(nPassive);
        break;
      }

      auto alpha = std::numeric_limits<double>::max();
      Index alpha_idx = 0;

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

      x.head(nPassive) += alpha * (s.head(nPassive) - x.head(nPassive));
      x[alpha_idx] = 0;
      --nPassive;

      AtA.col(nPassive).swap(AtA.col(alpha_idx));
      AtA.row(nPassive).swap(AtA.row(alpha_idx));
      // swap Atb to match with AtA
      Eigen::numext::swap(Atb.coeffRef(nPassive), Atb.coeffRef(alpha_idx));
      Eigen::numext::swap(x.coeffRef(nPassive), x.coeffRef(alpha_idx));
    }
  }
}

}

#endif
