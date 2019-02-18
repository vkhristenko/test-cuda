#include <iostream>

#include "include/inplace_fnnls.h"

template<typename T>
std::vector<vector_t<T>> run_cpu(unsigned int n) {
    // create the input matrices
    std::vector<vector_t<T>> result(n);
    std::vector<matrix_t<T>> As(n);
    std::vector<vector_t<T>> bs(n);

    // randomize
    for (auto& m : As)
        m = matrix_t<T>::Random();
    for (auto& v: bs)
        v = vector_t<T>::Random();
    for (auto& v: result)
        v = v.SetZero();

    // compute
    for (unsigned int i=0; i<n; i++) {
        inplace_fnnls(As[i], bs[i], result[i])
    }

    return result;
}

int main(int argc, char** argv) {
    if (argc<=1) {
        std::cout << "run with './main <number of matrices>'";
        exit(0);
    }
    unsigned int n = std::atoi(argv[1]);
    auto results = run_cpu(n);

    return 0;
}
