#include <iostream>

#include "Eigen/Dense"

void test0() {
    using type = Eigen::PermutationMatrix<10>;
    std::cout << "size = " << sizeof(type) << std::endl;
}

int main() {
    test0();
    return 0;
}
