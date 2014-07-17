// -*- C++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "parameters.hh"
#include "gpu.hh"

#define DEBUG

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

__align__(32) __constant__ __device__
uint32_t packets[GPU_STREAMS][PACKETS_BATCH_SIZE*GPU_FILTER_WORDS];

__device__ bool a_complement_not_subset_of_b(uint32_t a, uint32_t b) { 
	return ((a | b) != (~0U));
}

#if 0
#define TI_TREE_OFFSET 13
#define TI_INTERFACE_MASK (((uint16_t)1 << TI_TREE_OFFSET) - 1)
#else
static const unsigned int TI_TREE_OFFSET = 13;
static const uint16_t TI_INTERFACE_MASK = (1 << TI_TREE_OFFSET) - 1;
#endif

__device__ uint16_t ti_tree(uint16_t ti_pair) {
	return ti_pair >> TI_TREE_OFFSET;
}

__device__ uint16_t ti_interface(uint16_t ti_pair) {
	return ti_pair & TI_INTERFACE_MASK;
}

#if WITH_GPU_FAST_KERNEL

__global__ void fast_kernel(uint32_t * fib, unsigned int fib_size, 
							uint16_t * ti_table, unsigned int * ti_indexes,  
							uint16_t * query_ti_table ,  unsigned int batch_size, 
							ifx_result_t * results,  
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

//		printf("matched!\n");

		unsigned int ti_index = ti_indexes[id];
		for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
			// TODO: document these cryptic operations.
			uint16_t xor_tmp = *query_ti_table ^ ti_table[ti_index + i];
			if ((xor_tmp <= tree_interface_pair::IFX_MASK) && (xor_tmp != 0))
				results[(i * INTERFACES) + ((ti_table[ti_index + i]) & 0x1FFF)] = 1;
		}
	}
}

#else // no WITH_GPU_FAST_KERNEL

__global__ void minimal_kernel(uint32_t * fib, unsigned int fib_size, 
							   uint16_t * ti_table, unsigned int * ti_indexes,  
							   uint16_t * query_ti_table ,  unsigned int batch_size, 
							   ifx_result_t * results,
							   unsigned int stream_id) {

	unsigned int id = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;

	if(id >= fib_size)
		return;

	fib += (id * GPU_FILTER_WORDS);
	uint32_t f[GPU_FILTER_WORDS];

	for(unsigned int i = 0; i < GPU_FILTER_WORDS; ++i) 
		f[i] = fib[i];

	uint32_t * p = packets[stream_id] + 5;
	for(unsigned int pi = 0; pi < batch_size; ++pi) {
		if (a_complement_not_subset_of_b(f[5], *p)) {
			p += (0 + GPU_FILTER_WORDS);
			continue;
		}
		--p;
		if (a_complement_not_subset_of_b(f[4], *p)) {
			p += (1 + GPU_FILTER_WORDS);
			continue;
		}
		--p;
		if (a_complement_not_subset_of_b(f[3], *p)) {
			p += (2 + GPU_FILTER_WORDS);
			continue;
		}
		--p;
		if (a_complement_not_subset_of_b(f[2], *p)) {
			p += (3 + GPU_FILTER_WORDS);
			continue;
		}
		--p;
		if (a_complement_not_subset_of_b(f[1], *p)) {
			p += (4 + GPU_FILTER_WORDS);
			continue;
		}
		--p;
		if (a_complement_not_subset_of_b(f[0], *p)) {
			p += (5 + GPU_FILTER_WORDS);
			continue;
		}
		p += (5 + GPU_FILTER_WORDS);

		unsigned int ti_index = ti_indexes[id];
		for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
			// TODO: document these cryptic operations.
			uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
			if ((ti_xor <= 0x1FFF) && (ti_xor != 0)) {
				results[pi*INTERFACES + ((ti_table[ti_index + i]) & 0x1FFF)] = 1;
			}
		}
	}
}
#endif

void gpu::initialize() {
	ABORT_ON_ERROR(cudaSetDevice(0));
	ABORT_ON_ERROR(cudaDeviceSynchronize());
	ABORT_ON_ERROR(cudaThreadSynchronize());
	for(unsigned int i = 0; i < GPU_STREAMS; ++i)
		ABORT_ON_ERROR(cudaStreamCreate(streams + i));
}

void gpu::mem_info(gpu_mem_info * mi) {
	ABORT_ON_ERROR(cudaDeviceSynchronize());
	ABORT_ON_ERROR(cudaMemGetInfo(&(mi->free), &(mi->total)));
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
	ABORT_ON_ERROR(cudaMemcpyAsync(host_results, dev_results, size * INTERFACES * sizeof(ifx_result_t), cudaMemcpyDeviceToHost, streams[stream]));
}

void gpu::get_results(ifx_result_t * host_results, ifx_result_t * dev_results, unsigned int size) {
	ABORT_ON_ERROR(cudaMemcpy(host_results, dev_results, size * INTERFACES * sizeof(ifx_result_t), cudaMemcpyDeviceToHost));
}

void gpu::synchronize_device() {
	ABORT_ON_ERROR(cudaDeviceSynchronize());
	ABORT_ON_ERROR(cudaThreadSynchronize());
}

void gpu::synchronize_stream(unsigned int stream) {
	ABORT_ON_ERROR(cudaStreamSynchronize(streams[stream]));
}

#if WITH_PINNED_HOST_MEMORY
void * gpu::allocate_host_pinned_generic(unsigned int size) {
	void * host_array_pinned;
	ABORT_ON_ERROR(cudaMallocHost(&host_array_pinned, size));
	return host_array_pinned;
}
#endif

static const dim3 BLOCK_DIMS(GPU_BLOCK_DIM_X, GPU_BLOCK_DIM_Y);

void gpu::run_kernel(uint32_t * fib, unsigned int fib_size, 
					 uint16_t * ti_table, unsigned int * ti_indexes, 
					 uint16_t * query_ti_table, unsigned int batch_size, 
					 ifx_result_t * results, 
					 unsigned int stream) {

	unsigned int gridsize = fib_size/GPU_BLOCK_SIZE;
	if ((fib_size % GPU_BLOCK_SIZE) != 0)
		++gridsize;

#if WITH_GPU_FAST_KERNEL
	fast_kernel<<< gridsize, BLOCK_DIMS, 0, streams[stream] >>> (fib, 
																 fib_size,
																 ti_table, 
																 ti_indexes, 
																 query_ti_table, 
																 batch_size, 
																 results, 
																 stream);
#else
	minimal_kernel<<<gridsize, BLOCK_DIMS, 0, streams[stream] >>> (fib, 
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
	// TODO: deallocate 
	for(unsigned int i = 0; i < GPU_STREAMS; ++i)
		ABORT_ON_ERROR(cudaStreamDestroy(streams[i]));
	cudaDeviceReset();
}

void gpu::release_memory(void * p) {
	ABORT_ON_ERROR(cudaFree(p)); 
}

void gpu::release_pinned_memory(void * p) {
	ABORT_ON_ERROR(cudaFreeHost(p)); 
}
