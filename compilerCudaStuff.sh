#!/bin/bash
if [ "$1" == "gpu.lo" ]; then  
nvcc -arch sm_52 -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -c gpu.cu  -Xcompiler -fPIC -DPIC -o .libs/gpu.o
nvcc -arch sm_52 -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -c gpu.cu  -o gpu.o
cp gpu.lo.bak gpu.lo
else 
nvcc -arch sm_52 -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -c partitioner_gpu.cu  -Xcompiler -fPIC -DPIC -o .libs/partitioner_gpu.o
nvcc -arch sm_52 -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -c partitioner_gpu.cu  -o partitioner_gpu.o
cp partitioner_gpu.lo.bak partitioner_gpu.lo
fi
