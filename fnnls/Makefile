.PHONY: all, clean

all: run

EIGEN_HOME=/data/user/vkhriste/eigen/eigen

run: main.cu
	nvcc -I${EIGEN_HOME} -arch=sm_61 --expt-relaxed-constexpr -DUSE_V1  -lineinfo -Xptxas -v -std=c++14 -O2 -o $@ $<

clean:
	rm run
