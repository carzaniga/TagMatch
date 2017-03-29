#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstring>

#include "partitioner_gpu.hh"

using std::endl;
using std::cout;

// This does only reset the counters and the frequency buffers on the gpu 
__global__ void gzero(unsigned int *freqs, uint32_t * rcounter, uint32_t * lcounter) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= FILTER_WIDTH)
		return;
	freqs[tid] = 0;
	if (tid == 0) {
		*lcounter = 0;
		*rcounter = 0;
	}
}

// Computes the number of times every bit position in the range [0 - FILTER_WIDTH)
// is set to 1 in the fib, analyzing filters in the range [start - start + fib_size)
// Using __shared__ memory to store per block frequency arrays does not help in this
// application
__global__ void getfreqs(const uint32_t * __restrict__ fib,
								unsigned int fib_size,
								unsigned int start,
								uint32_t *freqs) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= fib_size)
		return;
	
	unsigned int mask = 1;
	for (int i = 0; i < FILTER_WIDTH; i++) {
		if (fib[(start + tid)*GPU_WORDS_PER_FILTER + (i / GPU_BITS_PER_WORD)] & (mask << (i % GPU_BITS_PER_WORD)))
			atomicAdd(&freqs[i], 1);
	}
}

// Does the same as std::stable_partition, but it is not guaranteed to be stable.
// Must be called before stable_partition_update (since a global barrier is needed)
__global__ void stable_partition(const uint32_t * __restrict__ fib,
		uint32_t begin, uint32_t size, uint32_t pivot,
		uint32_t * lcounter, uint32_t * rcounter, uint32_t * buffer) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= size)
		return;
	int idx;
	unsigned int mask = 1;
	if ((fib[(begin + tid) * GPU_WORDS_PER_FILTER + (pivot / GPU_BITS_PER_WORD)] & (mask << (pivot % GPU_BITS_PER_WORD))) == 0) {
			idx = atomicAdd(lcounter, 1);
			#pragma unroll
			for (int i = 0 ; i < GPU_WORDS_PER_FILTER; i++) 
				buffer[(begin + idx) * GPU_WORDS_PER_FILTER + i] = fib[(begin + tid) * GPU_WORDS_PER_FILTER + i];
	}
	else {
			idx = atomicAdd(rcounter, 1);
			#pragma unroll
			for (int i = 0 ; i < GPU_WORDS_PER_FILTER; i++) 
				buffer[(begin + size - 1 - idx) * GPU_WORDS_PER_FILTER + i] = fib[(begin + tid) * GPU_WORDS_PER_FILTER + i];
	}
}

// Does the same as std::stable_partition, but it is not guaranteed to be stable.
// Must be called after stable_partition_update (since a global barrier is needed)
__global__ void stable_partition_update(uint32_t * fib,
		uint32_t begin, uint32_t size, const uint32_t * __restrict__ buffer) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= size)
		return;
	#pragma unroll
	for (int i = 0 ; i < GPU_WORDS_PER_FILTER; i++) 
		fib[(begin + tid) * GPU_WORDS_PER_FILTER + i] = buffer[(begin + tid) * GPU_WORDS_PER_FILTER + i];
}


static uint32_t *g_fib, *g_freqs[MAXTHREADS], *g_buffer, *g_rcounter[MAXTHREADS], *g_lcounter[MAXTHREADS];
static uint64_t *h_fib;
static cudaStream_t gstream[MAXTHREADS];


// Compiles the fib on the host memory, so that it can be copied to the device
// global memory
void partitioner_gpu::fibToArray(std::vector<fib_entry *> * fib, uint32_t size) {
	for (uint64_t f = 0; f < size; f++) {
		const uint64_t * ptr = fib->at(f)->filter.begin64();
		for (uint64_t j = 0; j < CPU_WORDS_PER_FILTER; j++)
			h_fib[f*CPU_WORDS_PER_FILTER + j] = *ptr++; 
	}
}

void partitioner_gpu::init(unsigned int part_thread_count, std::vector<fib_entry *> * fib) {
	std::cout << "Setting up gpu... ";
	cudaMallocHost(&h_fib, sizeof(uint64_t) * fib->size() * CPU_WORDS_PER_FILTER);
	fibToArray(fib, fib->size());
	cudaMalloc(&g_fib, sizeof(uint32_t) * fib->size() * GPU_WORDS_PER_FILTER);
	cudaMemcpy(g_fib, h_fib, sizeof(uint32_t) * fib->size() * GPU_WORDS_PER_FILTER, cudaMemcpyHostToDevice);
	
	for (int t=0; t<part_thread_count; t++) {
		cudaMalloc(&(g_rcounter[t]), sizeof(uint32_t));
		cudaMalloc(&(g_lcounter[t]), sizeof(uint32_t));
		cudaMalloc(&(g_freqs[t]), sizeof(uint32_t) * FILTER_WIDTH);
		gzero<<<1, FILTER_WIDTH>>>(g_freqs[t], g_lcounter[t], g_rcounter[t]);
		cudaStreamCreate(&(gstream[t]));
	}
	cudaMalloc(&g_buffer, sizeof(uint32_t) * fib->size() * GPU_WORDS_PER_FILTER);
	cudaDeviceSynchronize();
	cudaFreeHost(h_fib);
	cudaError_t err;
	if ((err = cudaGetLastError()) != 0)
		std::cout << "CUDA error on init: " << err << std::endl;
	std::cout << "\t\t\t" << std::setw(12) << "done!" << endl;
}

void partitioner_gpu::reset_buffers(unsigned int tid) {
	gzero<<<1, FILTER_WIDTH, 0, gstream[tid]>>>(g_freqs[tid], g_lcounter[tid], g_rcounter[tid]);
	cudaError_t err;
	if ((err = cudaGetLastError()) != 0)
		std::cout << "CUDA error on reset buffers: " << err << std::endl;
}

void partitioner_gpu::clear(unsigned int part_thread_count) {
	cudaFree(&g_fib);
	for (int t=0; t<part_thread_count; t++) {
		cudaFree(&(g_rcounter[t]));
		cudaFree(&(g_lcounter[t]));
		cudaFree(&(g_freqs[t]));
		cudaStreamDestroy(gstream[t]);
	}
	cudaFree(&g_buffer);
}

void partitioner_gpu::get_frequencies(unsigned int tid, unsigned int size, unsigned int first, unsigned int * freq, size_t buffer_size) {
	unsigned int gridSize = size / GPU_THREADS + ((size % GPU_THREADS) != 0);
	getfreqs<<<gridSize, GPU_THREADS, 0, gstream[tid]>>>(g_fib, size, first, g_freqs[tid]);
	cudaMemcpyAsync(freq, g_freqs[tid], buffer_size, cudaMemcpyDeviceToHost, gstream[tid]);
	cudaStreamSynchronize(gstream[tid]);
	cudaError_t err;
	if ((err = cudaGetLastError()) != 0)
		std::cout << "CUDA error on get_frequencies: " << err << std::endl;
}

void partitioner_gpu::unstable_partition(unsigned int tid, unsigned int size, unsigned int first, unsigned int pivot) {
	unsigned int gridSize = size / GPU_THREADS + ((size % GPU_THREADS) != 0);
	stable_partition<<<gridSize, GPU_THREADS, 0, gstream[tid]>>>(g_fib, first, size, pivot, g_lcounter[tid], g_rcounter[tid], g_buffer);
	stable_partition_update<<<gridSize, GPU_THREADS, 0, gstream[tid]>>>(g_fib, first, size, g_buffer);
	cudaError_t err;
	if ((err = cudaGetLastError()) != 0)
		std::cout << "CUDA error on unstable_partition: " << err << std::endl;
}
