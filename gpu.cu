// -*- C++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "parameters.hh"
#include "gpu.hh"

#define test 0

#define ABORT_ON_ERROR(f)												\
	do {																\
		cudaError_t status_ = (f);										\
		if (status_ != cudaSuccess) {									\
			fprintf(stderr, "Fatal error: " #f ": %s\n(%s:%d)\nABORTING\n", \
					cudaGetErrorString(status_),						\
					__FILE__, __LINE__);								\
			cudaDeviceReset() ;											\
			exit(0);													\
		}																\
	} while (0)

#define WARNING_ON_ERROR(f)												\
	do {																\
		cudaError_t status_ = (f);										\
		if (status_ != cudaSuccess) {									\
			fprintf(stderr, "Error: " #f ": %s\n(%s:%d)\n",				\
					cudaGetErrorString(status_),						\
					__FILE__, __LINE__);								\
		}																\
	} while (0)

cudaStream_t streams[GPU_STREAMS];

__align__(32) __constant__ 
uint32_t packets[GPU_STREAMS][PACKETS_BATCH_SIZE*GPU_FILTER_WORDS];

#define set_diff(a, b) (((a) & ~(b)))

__global__ void minimal_kernel(uint32_t * fib, unsigned int fib_size, 
							   uint16_t * ti_table, unsigned int * ti_indexes,  
							   uint16_t * query_ti_table ,  unsigned int batch_size, 
							   ifx_result_t * result,  
							   unsigned int stream_id) {

	unsigned int id = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;

	if(id >= fib_size)
		return;

	unsigned int f_id = id*GPU_FILTER_WORDS ;

	uint32_t f[GPU_FILTER_WORDS];

	for(unsigned int i = 0; i < GPU_FILTER_WORDS; ++i)
		f[i] = fib[f_id + i];

	for(unsigned int j = 0; batch_size > 0; --batch_size, j += GPU_FILTER_WORDS, ++query_ti_table) {
		if(set_diff(f[5], packets[stream_id][j+5]) ||
		   set_diff(f[4], packets[stream_id][j+4]) ||
		   set_diff(f[3], packets[stream_id][j+3]) ||
		   set_diff(f[2], packets[stream_id][j+2]) ||
		   set_diff(f[1], packets[stream_id][j+1]) ||
		   set_diff(f[0], packets[stream_id][j]))
			continue;
		
		unsigned int ti_index = ti_indexes[id];
		for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
			// TODO: document these cryptic operations.
			uint16_t xor_tmp = *query_ti_table ^ ti_table[ti_index + i];
			if ((xor_tmp <= 0x1FFF) && (xor_tmp != 0))
				result[(i * INTERFACES) + ((ti_table[ti_index + i]) & 0x1FFF)] = 1;
		}
	}
}

#define a_complement_not_subset_of_b(a,b) (~((a) | (b)))

__global__ void fast_kernel(uint32_t * fib, unsigned int fib_size, 
							   uint16_t * ti_table, unsigned int * ti_indexes,  
							   uint16_t * query_ti_table ,  unsigned int batch_size, 
							   ifx_result_t * result,  
							   unsigned int stream_id)
{

	unsigned int t1 = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) ;
	unsigned int id = t1 + threadIdx.x;

	if(id >= fib_size)
		return;

	unsigned int f_id = GPU_FILTER_WORDS*(t1) + threadIdx.x;
		
	unsigned int f[GPU_FILTER_WORDS];
	f[0]=fib[f_id];
	f[1]=fib[f_id+32];
	f[2]=fib[f_id+64];
	f[3]=fib[f_id+96];
	f[4]=fib[f_id+128];
	f[5]=fib[f_id+160];

	for(unsigned int j = 0; batch_size > 0; --batch_size, j += GPU_FILTER_WORDS, ++query_ti_table) {

		if (a_complement_not_subset_of_b(f[5], packets[stream_id][j+6]))
		    continue;
		if (a_complement_not_subset_of_b(f[4], packets[stream_id][j+4]))
		    continue;
		if (a_complement_not_subset_of_b(f[3], packets[stream_id][j+3]))
		    continue;
		if (a_complement_not_subset_of_b(f[2], packets[stream_id][j+2]))
		    continue;
		if (a_complement_not_subset_of_b(f[1], packets[stream_id][j+1]))
		    continue;
		if (a_complement_not_subset_of_b(f[0], packets[stream_id][j]))
		    continue;

		unsigned int ti_index = ti_indexes[id];
		for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
			// TODO: document these cryptic operations.
			uint16_t xor_tmp = *query_ti_table ^ ti_table[ti_index + i];
			if ((xor_tmp <= 0x1FFF) && (xor_tmp != 0))
				result[(i * INTERFACES) + ((ti_table[ti_index + i]) & 0x1FFF)] = 1;
		}
	}
}

void gpu::initialize() {
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();
	for(unsigned int i = 0; i < GPU_STREAMS; ++i)
		ABORT_ON_ERROR(cudaStreamCreate(streams + i));
}

void gpu::mem_info() {
	size_t free;
	size_t total;
	ABORT_ON_ERROR(cudaMemGetInfo(&free, &total));

	printf("free= %f total_mem=%f\n",(long unsigned)free*1.0/(1024*1024),(long unsigned)total*1.0/(1024*1024));

	WARNING_ON_ERROR(cudaDeviceSynchronize());
}

void gpu::async_copy_packets(unsigned int * host_packets, unsigned int size , unsigned int stream) {
	ABORT_ON_ERROR(cudaMemcpyToSymbolAsync(packets, host_packets, size*GPU_FILTER_WORDS*sizeof(unsigned int), stream*PACKETS_BATCH_SIZE*GPU_FILTER_WORDS*sizeof(unsigned int), cudaMemcpyHostToDevice, streams[stream]));
}

// allocates memory for a table on the device of the given byte size
//
void * gpu::allocate_generic(unsigned int size) {
	void * dev_table = 0;
	ABORT_ON_ERROR(cudaMalloc(&dev_table, size));
	return dev_table; 
}

// allocates memory for a table on the device of the given byte size
// and then copies the content from the host table into the device
// table
//
void * gpu::allocate_and_copy_generic(void * host_table, unsigned int size) {
	void * dev_table = 0;
	ABORT_ON_ERROR(cudaMalloc(&dev_table, size)); 
	ABORT_ON_ERROR(cudaMemcpy(dev_table, host_table, size, cudaMemcpyHostToDevice));
	return dev_table; 
}

void gpu::async_copy(void * host_src, void * dev_dst, unsigned int size, unsigned int stream_id) {
	ABORT_ON_ERROR(cudaMemcpyAsync(dev_dst, host_src, size, cudaMemcpyHostToDevice, streams[stream_id]));
}

 // this is useful for clearing the dev_res (interfaces) to 0 before
 // calling the kernel
void gpu::async_set_zero(void * dev_array, unsigned int size, unsigned int stream_id) {
	ABORT_ON_ERROR(cudaMemsetAsync(dev_array, 0, size, streams[stream_id]));
}


void gpu::async_get_results(ifx_result_t * host_results, ifx_result_t * dev_results, 
							 unsigned int size, unsigned int stream) {
	ABORT_ON_ERROR(cudaMemcpyAsync(host_results, dev_results, size * sizeof(ifx_result_t), cudaMemcpyDeviceToHost, streams[stream]));
}

void gpu::synchronize_stream(unsigned int stream) {
	ABORT_ON_ERROR(cudaStreamSynchronize(streams[stream]));
}

#if WITH_PINNED_HOST_MEMORY
void * gpu::allocate_host_pinned_generic(unsigned int size) {
	void * host_array_pinned;
	ABORT_ON_ERROR(cudaMallocHost(&host_array_pinned, size));
	return host_array_pinned ;
}
#endif

void gpu::run_kernel(uint32_t * fib, unsigned int fib_size, 
					 uint16_t * ti_table, unsigned int * ti_indexes, 
					 uint16_t * query_ti_table, unsigned int batch_size, 
					 ifx_result_t * results, 
					 unsigned int stream) {

	dim3 block(GPU_BLOCK_DIM_X, GPU_BLOCK_DIM_Y);
	unsigned int b = GPU_BLOCK_SIZE;
	unsigned int gridsize = (fib_size % b == 0) ? fib_size/b : fib_size/b + 1;
	dim3 grid(gridsize);

#if GPU_FAST
	fast_kernel<<<gridsize, block,0 , streams[stream] >>> (fib, 
														   fib_size,
														   ti_table, 
														   ti_indexes, 
														   query_ti_table, 
														   batch_size, 
														   results, 
														   stream);
#else
	minimal_kernel<<<gridsize, block,0 , streams[stream] >>> (fib, 
															  fib_size,
															  ti_table, 
															  ti_indexes, 
															  query_ti_table, 
															  batch_size, 
															  results, 
															  stream);
#endif
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr, "Fatal error: run_kernel: %s\n(%s:%d)\nABORTING\n", 
				cudaGetErrorString(status), __FILE__, __LINE__);
		cudaDeviceReset() ; 
		exit(0); 
	} 
}

void gpu::shutdown() {
	cudaDeviceReset();
	// TODO: deallocate 
	return;
}

void gpu::release_memory(void * p) {
	ABORT_ON_ERROR(cudaFree(p)); 
}
