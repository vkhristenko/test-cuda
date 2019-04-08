#!/bin/bash

src=$1
options=$2

nvcc -v -arch=sm_61 ${options} --expt-relaxed-constexpr -lineinfo -Xptxas -v -std=c++14 -O2 -o $src ${src}.cu
