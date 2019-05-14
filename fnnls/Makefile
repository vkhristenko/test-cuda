.PHONY: all, clean

all: run_v1  run_v2

run_v1: 
	nvcc -I/data/patatrack/vkhriste/eigen -arch=sm_61 --expt-relaxed-constexpr -DUSE_V1  -lineinfo -Xptxas -v -std=c++14 -O2 -o $@ main.cu

run_v2:
	nvcc -I/data/patatrack/vkhriste/eigen -arch=sm_61 --expt-relaxed-constexpr -DUSE_V2  -lineinfo -Xptxas -v -std=c++14 -O2 -o $@ main.cu

clean:
	rm run_v1 run_v2

