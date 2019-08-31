#!/bin/bash

nvcc -I/data/patatrack/vkhriste/eigen -arch=sm_61 --expt-relaxed-constexpr -DFNNLS_DEBUG_MAIN -lineinfo -Xptxas -v -std=c++14 -O2 -o main main.cu
