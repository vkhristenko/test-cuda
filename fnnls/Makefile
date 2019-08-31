.PHONY: all, clean

all: main

HEADERS=fnnls_no_mm_mv.h
EIGEN_HOME=/data/patatrack/vkhriste/eigen
#EIGEN_HOME=/data/user/vkhriste/eigen/eigen

main: main.cu
	nvcc -I${EIGEN_HOME} -arch=sm_61 --expt-relaxed-constexpr -DUSE_V1  -lineinfo -Xptxas -v -std=c++14 -O2 -o $@ $<

#
#main_dyn_par_foreach: main_dyn_par_foreach.cu
#	nvcc -I${EIGEN_HOME} -arch=sm_61 --expt-relaxed-constexpr -DUSE_V1  -lineinfo -Xptxas -v -std=c++14 -O2 -rdc=true -o $@ $< -lcudadevrt
#
#main_dyn_par_forall: main_dyn_par_forall.cu
#	nvcc -I${EIGEN_HOME} -arch=sm_61 --expt-relaxed-constexpr -DUSE_V1  -lineinfo -Xptxas -v -std=c++14 -O2 -rdc=true -o $@ $< -lcudadevrt
#
#main_dyn_par_single_fnnls_iteration: main_dyn_par_single_fnnls_iteration.cu
#	nvcc -I${EIGEN_HOME} -arch=sm_61 --expt-relaxed-constexpr -DUSE_V1  -lineinfo -Xptxas -v -std=c++14 -O2 -rdc=true -o $@ $< -lcudadevrt
#

clean:
	rm main
