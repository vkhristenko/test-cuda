#include <vector>
#include <fstream>

#include "Eigen/Dense"

constexpr unsigned long MATRIX_SIZE = 10;

template<typename T>
using matrix_t = Eigen::Matrix<T, MATRIX_SIZE, MATRIX_SIZE, Eigen::ColMajor>;

template<typename T>
std::vector<matrix_t<T>> parse_matrices(std::string filePathName) {
    std::vector<matrix_t<T>> ms;
    ms.reserve(100); 

    std::ifstream f{filePathName, std::ifstream::in};

    int row = 0, col = 0;
    int currentMatrix = -1;
    int counter = 0;
    while (true) {
        T value;
        f >> value;
        if (f.eof()) break;
        counter++;

        if (row==0 && col==0) {
            ms.push_back(matrix_t<T>{});
            currentMatrix++;
        }
        ms[currentMatrix](row, col) = value;
        col++;

        if (col == 10) {
            row++;
            col = 0;
        }

        if (row == 10) {
            row = 0;
            col = 0;
        }
    }

    return ms;
}
