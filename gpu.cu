// -*- C++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "gpu.hh"
#include "parameters.hh"

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

__align__(128) __constant__ __device__
uint32_t packets[GPU_STREAMS][PACKETS_BATCH_SIZE*GPU_FILTER_WORDS];

#define BV_BLOCK_NOT_SUBSET_IS_A_MACRO 0

#if BV_BLOCK_NOT_SUBSET_IS_A_MACRO
#define BV_BLOCK_NOT_SUBSET(x,y) (((x) & (y)) != (x))
#else
__device__ bool BV_BLOCK_NOT_SUBSET(uint32_t x, uint32_t y) { 
	return (x & y) != x;
}
#endif

#define RECORD_MATCHING_FILTER(id,msg)															\
	for(unsigned int i = ti_table[ti_indexes[id]]; i > 0; --i) {								\
		uint16_t ti_xor = query_ti_table[(msg)] ^ ti_table[ti_indexes[id] + i];					\
		if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0))											\
			results[(msg)*INTERFACES + ((ti_table[ti_indexes[id] + i]) & (0xFFFF >> 3))] = 1;	\
	}


template <unsigned int Blocks>
__global__ void three_phase_matching(uint32_t * fib, unsigned int fib_size, 
									 uint16_t * ti_table, unsigned int * ti_indexes,  
									 uint16_t * query_ti_table ,  unsigned int batch_size, 
									 ifx_result_t * results,
									 unsigned int stream_id) {
	// candidate messages are stored in an array in shared memory
	__align__(128) __shared__ uint16_t candidate_messages[PACKETS_BATCH_SIZE];
	__shared__ uint32_t candidate_count; 

	// common prefix of all filters in this thread block
	__shared__ uint32_t prefix_blocks[Blocks];

	// number of full prefix blocks
	__shared__ uint8_t prefix_complete_blocks;

	// position of first non-zero prefix block (==Blocks if no prefix
	// on no non-zero blocks)
	__shared__ uint8_t prefix_first_non_zero; 

	// thread id within a thread block
	uint32_t tid = (blockDim.x * threadIdx.y) + threadIdx.x;

	// thread and filter id within the whole partition 
	uint32_t id  = (blockDim.x * blockDim.y * blockIdx.x) + tid; 

	// local register image of prefix_block_count
	// uint8_t prefix_first_non_zero;

	// it is faster to check if current block is the last block only
	// in tid==0 rather than check it at the begining of every
	// block. I did it that way and it was slower!.
	// 
	if (tid == 0) {
		candidate_count = 0;

		// id of the last thread in this block:
		// (id + GPU_BLOCK_SIZE - 1) or (fib_size - 1)
		// whichever is smaller
		uint32_t last_thread_id
			= (fib_size > id + GPU_BLOCK_SIZE) ? (id + GPU_BLOCK_SIZE - 1) : (fib_size - 1);

		if (id == last_thread_id) {
			// there is only one thread in this block, and therefore
			// only one filter, so there is nothing to do with
			// prefixes...
			prefix_complete_blocks = 0;
			prefix_first_non_zero = Blocks;
		} else {
			prefix_first_non_zero = Blocks;
			for(unsigned int i = 0; i < Blocks; ++i) {
				uint32_t block_of_first_tid = fib[id*Blocks + i];
				uint32_t block_of_last_tid = fib[last_thread_id*Blocks + i];

				// position of the first bit that differs between
				// block_of_first_tid and block_last_tid; positions
				// start from 1, and 0 means no difference
				if (block_of_first_tid == block_of_last_tid) {
					// the two blocks are identical
					prefix_blocks[i] = block_of_first_tid;
					if (prefix_first_non_zero == Blocks && block_of_first_tid != 0)
						prefix_first_non_zero = i;
				} else {
					int pos = __ffs(block_of_first_tid ^ block_of_last_tid);
					if (pos == 1) {
						// the two blocks have no common prefix
						prefix_blocks[i] = 0;
					} else {
						// pos > 1 ==> there are some common bits, so
						// we zero out all remaining bits.  Remeber
						// that the common bits are the least
						// significant ones.
						block_of_first_tid <<= (33 - pos);
						block_of_first_tid >>= (33 - pos);
						prefix_blocks[i] = block_of_first_tid;
						if (prefix_first_non_zero == Blocks && block_of_first_tid != 0)
							prefix_first_non_zero = i;
					}
					prefix_complete_blocks = i;
					break;
				}
			}
		}
	} 
	// end for phase 1: computation of the common prefix by tid==0
	__syncthreads();

	// start of phase 2: candidate selection, performed by all threads
	// in this block.
	//
	// Access to shared memory here (prefix_ and candidate_) is bank
	// conflict free.  Here is the reason (CUDA documentation): "An
	// exception is the case where all threads in a warp address the
	// same shared memory address, resulting in a broadcast"
	// 
	if (prefix_first_non_zero == Blocks) {
		// there is no usable prefix or the common prefix is all-zero,
		// so we fall back to the basic matching procedure, possibly
		// skipping the all-zero prefix blocks.
		if(id < fib_size) {
			uint32_t d[Blocks];
			unsigned int prefix_end = prefix_complete_blocks;

			for(unsigned int i = prefix_end; i < Blocks; ++i)
				d[i] = fib[id*Blocks + i];

			for(unsigned int pi = 0; pi < batch_size; ++pi) {
				for(unsigned int i = prefix_end; i < Blocks; ++i)
					if (BV_BLOCK_NOT_SUBSET(d[i], packets[stream_id][pi*Blocks + i]))
						goto no_match;

				RECORD_MATCHING_FILTER(id,pi);

			no_match: ;
			}
		}
	} else {
		// candidate selection: we consider the prefix blocks starting
		// from the first non-zero block
		// 
		unsigned int prefix_end = prefix_complete_blocks;
		if (prefix_blocks[prefix_end] != 0)
			++prefix_end;

#pragma unroll
		for(unsigned int m = 0; m < batch_size; m += GPU_BLOCK_SIZE) {
			for(unsigned int i = prefix_first_non_zero; i < prefix_end; ++i) 
				if(BV_BLOCK_NOT_SUBSET(prefix_blocks[i], packets[stream_id][(tid + m)*Blocks + i]))
					goto skip_message;

			// add this as a candidate message for phase-three matching
			candidate_messages[atomicAdd(&candidate_count, 1)] = tid + m;

		skip_message: ;
		}
		// end of phase 2 (candidate selection)
		//
		__syncthreads();
		// 
		// phase 3: matching candidate messages.
		// 
		if (candidate_count > 0 && id < fib_size) {
			uint32_t d[Blocks];
			unsigned int prefix_end = prefix_complete_blocks;

			for(unsigned int i = prefix_end; i < Blocks; ++i)
				d[i] = fib[id*Blocks + i];

			for(unsigned int pi = 0; pi < candidate_count; ++pi) {
				uint16_t message = candidate_messages[pi];
				for(unsigned int i = prefix_end; i < Blocks; ++i)
					if(BV_BLOCK_NOT_SUBSET(d[i], packets[stream_id][message*Blocks + i]))
						goto candidate_no_match;

				RECORD_MATCHING_FILTER(id,message);

			candidate_no_match:
				continue;
			}
		}
	}
}

template <unsigned int Blocks>
__global__ void one_phase_matching(uint32_t * fib, unsigned int fib_size, 
								   uint16_t * ti_table, unsigned int * ti_indexes,  
								   uint16_t * query_ti_table ,  unsigned int batch_size, 
								   ifx_result_t * results,
								   unsigned int stream_id) {

	uint32_t id = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;

	if(id >= fib_size)
		return;

	uint32_t d[Blocks];

	for(unsigned int i = 0; i < Blocks; ++i)
		d[i] = fib[id*Blocks + i];

	for(unsigned int pi = 0; pi < batch_size; ++pi) {
#if EXPLICIT_BLOCK_LOOP_UNROLL
		if (BV_BLOCK_NOT_SUBSET(d[0], packets[stream_id][pi*Blocks + 0]))
			continue;
		if (Blocks > 1)
			if (BV_BLOCK_NOT_SUBSET(d[1], packets[stream_id][pi*Blocks + 1]))
				continue;
		if (Blocks > 2)
			if (BV_BLOCK_NOT_SUBSET(d[2], packets[stream_id][pi*Blocks + 2]))
				continue;
		if (Blocks > 3)
			if (BV_BLOCK_NOT_SUBSET(d[3], packets[stream_id][pi*Blocks + 3]))
				continue;
		if (Blocks > 4)
			if (BV_BLOCK_NOT_SUBSET(d[4], packets[stream_id][pi*Blocks + 4]))
				continue;
		if (Blocks > 5)
			if (BV_BLOCK_NOT_SUBSET(d[5], packets[stream_id][pi*Blocks + 5]))
				continue;
#else
#pragma unroll
		for(unsigned int i = 0; i < Blocks; ++i)
			if (BV_BLOCK_NOT_SUBSET(d[i], packets[stream_id][pi*Blocks + i]))
				goto no_match;
#endif

		RECORD_MATCHING_FILTER(id, pi);

#if ! EXPLICIT_BLOCK_LOOP_UNROLL
	no_match:
		;
#endif
	}
}

void gpu::initialize() {
	ABORT_ON_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
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
	ABORT_ON_ERROR(cudaMemcpyToSymbolAsync(packets, host_packets, size*sizeof(unsigned int), stream*PACKETS_BATCH_SIZE*GPU_FILTER_WORDS*sizeof(unsigned int), cudaMemcpyHostToDevice, streams[stream]));
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

void * gpu::allocate_host_pinned_generic(unsigned int size) {
	void * host_array_pinned;
	ABORT_ON_ERROR(cudaMallocHost(&host_array_pinned, size));
	return host_array_pinned;
}

static const dim3 BLOCK_DIMS(GPU_BLOCK_DIM_X, GPU_BLOCK_DIM_Y);

void gpu::run_kernel(uint32_t * fib, unsigned int fib_size, 
					 uint16_t * ti_table, unsigned int * ti_indexes, 
					 uint16_t * query_ti_table, unsigned int batch_size, 
					 ifx_result_t * results, 
					 unsigned int stream,
					 unsigned char skip_blocks) {

	unsigned int gridsize = fib_size/GPU_BLOCK_SIZE;
	if ((fib_size % GPU_BLOCK_SIZE) != 0)
		++gridsize;
	
	switch (skip_blocks) {
	case 0:
		three_phase_matching<6> <<<gridsize, BLOCK_DIMS, 0, streams[stream] >>>
			(fib, fib_size, ti_table, ti_indexes, query_ti_table, batch_size, results, stream);
		break;

	case 1:
		three_phase_matching<5> <<<gridsize, BLOCK_DIMS, 0, streams[stream] >>>
			(fib, fib_size, ti_table, ti_indexes, query_ti_table, batch_size, results, stream);
		break;

	case 2:
		three_phase_matching<4> <<<gridsize, BLOCK_DIMS, 0, streams[stream] >>>
			(fib, fib_size, ti_table, ti_indexes, query_ti_table, batch_size, results, stream);
		break;

	case 3:
		three_phase_matching<3> <<<gridsize, BLOCK_DIMS, 0, streams[stream] >>>
			(fib, fib_size, ti_table, ti_indexes, query_ti_table, batch_size, results, stream);
		break;

	case 4: 	
		one_phase_matching<2> <<<gridsize, BLOCK_DIMS, 0, streams[stream] >>> 
			(fib, fib_size, ti_table, ti_indexes, query_ti_table, batch_size, results, stream);
		break;

	case 5: 	
		one_phase_matching<1> <<<gridsize, BLOCK_DIMS, 0, streams[stream] >>> 
			(fib, fib_size, ti_table, ti_indexes, query_ti_table, batch_size, results, stream);
		break;

	}

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
