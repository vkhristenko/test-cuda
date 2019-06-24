#!/bin/bash

nvcc -I/data/user/vkhriste/eigen/eigen -arch=sm_61 --expt-extended-lambda --expt-relaxed-constexpr -lineinfo -Xptxas -v -std=c++14 -O2 -o test_f test_f.cu
