#!/bin/bash

nvcc -I/data/patatrack/vkhriste/eigen --expt-relaxed-constexpr -DFNNLS_DEBUG_MAIN -lineinfo -std=c++14 -o main main.cu
