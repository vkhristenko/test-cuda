#include <iostream>

#include "parse_matrices.hpp"

int main() {
    std::string name {"matrices.in"};

    auto ms = parse_matrices<float>(name);

    return 0;
}
