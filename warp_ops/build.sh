#!/bin/bash

TARGET=$1

nvcc -I/data/patatrack/vkhriste/eigen -arch=sm_61 --expt-relaxed-constexpr -lineinfo -Xptxas -v -std=c++14 -O2 -o ${TARGET} ${TARGET}.cu
