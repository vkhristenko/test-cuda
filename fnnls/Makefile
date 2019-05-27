.PHONY: all, clean

all: run_v1

EIGEN_HOME=/data/user/vkhriste/eigen/eigen

run_v1: main.cu
	nvcc -I${EIGEN_HOME} -arch=sm_61 --expt-relaxed-constexpr -DUSE_V1  -lineinfo -Xptxas -v -std=c++14 -O2 -o $@ main.cu

clean:
	rm run_v1 run_v2

