# Simple Makefile for FEM CG solver

CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -march=native -fopenmp -DDISABLE_CUDA
NVCCFLAGS = -O3 -arch=sm_75

all: app.host app.cuda

cpu: fem_application.cpp vector.hpp sparse_matrix.hpp conjugate_gradient.hpp
	$(CXX) $(CXXFLAGS) fem_application.cpp -o app.host

gpu: fem_application.cu vector.hpp sparse_matrix.hpp conjugate_gradient.hpp
	$(NVCC) $(NVCCFLAGS) fem_application.cu -o app.cuda

clean:
	rm -f app.host app.cuda
