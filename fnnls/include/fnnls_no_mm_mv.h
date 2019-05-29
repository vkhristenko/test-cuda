#ifndef fnnls_no_mm_mv_h
#define fnnls_no_mm_mv_h

#include "data_types.h"

#define NNLS_DEBUG
#undef NNLS_DEBUG

namespace v2 {

using namespace Eigen;

template<typename T>
__host__ __device__
void fnnls(
        matrix_t<T>& AtA,
        vector_t<T>& Atb,
        vector_t<T>& x,
        const double const eps = 1e-11,
        const unsigned int const max_iterations = 1000) {
  using data_type = T;

  auto nPassive = 0;
  // main loop
  for (auto iter = 0; iter < max_iterations; ++iter) {
    const auto nActive = VECTOR_SIZE - nPassive;

    if(!nActive)
      break;

    //  
    unsigned int w_max_idx = -1;
    auto max_w {static_cast<T>(-1)};
    for (unsigned int i=VECTOR_SIZE-nActive; i<VECTOR_SIZE; i++) {
        auto sum_per_row{static_cast<T>(0)};
        auto atb = Atb(i);
        #pragma unroll
        for (unsigned int k=0; k<VECTOR_SIZE; k++)
            sum_per_row += AtA(i, k) * x(k);

        // compute gradient value and check if it is greater than max
        auto const wvalue = atb - sum_per_row;
        if (max_w < wvalue) {
            max_w = wvalue;
            w_max_idx = i;
        }
    }

    // check for convergence
    if (max_w < eps)
      break;

    // swap AtA to avoid copy
    AtA.col(nPassive).swap(AtA.col(w_max_idx));
    AtA.row(nPassive).swap(AtA.row(w_max_idx));
    // swap Atb to match with AtA
    Eigen::numext::swap(Atb.coeffRef(nPassive), Atb.coeffRef(w_max_idx));
    Eigen::numext::swap(x.coeffRef(nPassive), x.coeffRef(w_max_idx));

    ++nPassive;

// inner loop
    vector_t<data_type> s;
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
