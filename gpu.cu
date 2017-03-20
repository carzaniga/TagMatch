// -*- C++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "gpu.hh"
#include "cuda_profiler_api.h"
#include <signal.h>

#define TIME 0
#define PROFILE 0
#define Blocks 6
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

cudaStream_t streams[GPU_NUM][GPU_STREAMS];
cudaEvent_t copiedBack[GPU_NUM][GPU_STREAMS];

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

#define RECORD_MATCHING_FILTER(id,msg)\
rescount = atomicAdd(&(results_count->count), 1);\
if (rescount < MAX_MATCHES) {\
	                  atomicOr(&results_data->pairs[5*(rescount/4)], ((uint32_t)msg << (8 * (rescount % 4))));\
	                  results_data->pairs[5*(rescount/4)+1+(rescount%4)] = id;\
}

__global__ void 
three_phase_matching(const uint32_t * __restrict__ fib, 
									 unsigned int fib_size, 
									 unsigned int batch_size, 
									 result_t * results_count,
									 result_t * results_data,
									 unsigned int stream_id) {
	// candidate messages are stored in an array in shared memory
	__align__(128) __shared__ uint8_t candidate_messages[PACKETS_BATCH_SIZE];
	__shared__ uint32_t candidate_count; 

	// common prefix of all filters in this thread block
	__shared__ uint32_t prefix_blocks[Blocks];

	// number of full prefix blocks
	__shared__ unsigned int prefix_complete_blocks;

	// position of first non-zero prefix block (==Blocks if no prefix
	// on no non-zero blocks)
	__shared__ unsigned int prefix_first_non_zero; 

	// thread id within a thread block
	uint32_t tid = threadIdx.x;
	// thread and filter id within the whole partition 
	uint32_t id  = (blockDim.x * blockIdx.x) + tid; 
	
	// it is faster to check if current block is the last block only
	// in tid==0 rather than check it at the begining of every
	// block. I did it that way and it was slower!.
	// 
	// this is the counter for the output tree-interface pairs. only 1 thread in the current  
	// kernel has to set it
	
	uint32_t rescount;
#if TIME
	uint32_t t1=0,t2=0,t3=0,t4=0 ;
	if(id==0)
		t1= clock() ;
#endif
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
#ifdef COALESCED_READS
				uint32_t block_of_first_tid = fib[id + fib_size * i];
				uint32_t block_of_last_tid = fib[last_thread_id+fib_size * i];	
#else
				uint32_t block_of_first_tid = fib[id*Blocks + i];
				uint32_t block_of_last_tid = fib[last_thread_id*Blocks + i];
#endif

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
	unsigned int prefix_end = prefix_complete_blocks;
	if ((prefix_first_non_zero == Blocks) && (batch_size % 2 == 0))
		goto match_all ;
#if TIME
	if(id==0)
		t2=clock() ;
#endif

	// start of phase 2: candidate selection, performed by all threads
	// in this block.
	//
	// Access to shared memory here (prefix_ and candidate_) is bank
	// conflict free.  Here is the reason (CUDA documentation): "An
	// exception is the case where all threads in a warp address the
	// same shared memory address, resulting in a broadcast"
	// 

	// candidate selection: we consider the prefix blocks starting
	// from the first non-zero block
	prefix_end = prefix_complete_blocks;
	if (prefix_blocks[prefix_end] != 0)
		++prefix_end;

	// "prefix_first_non_zero!=0" is true for 24000 blocks (for 1m queries)
	// in case prefix_first_non_zero == Blocks, the following code still works.
	// it simply puts every message in the candidante_messages. 
	// Removing the atomic increment the condition stated above doesn't improve the code.

#pragma unroll
	for(unsigned int m = tid; m < batch_size; m += GPU_BLOCK_SIZE) {
		for(unsigned int i = prefix_first_non_zero; i < prefix_end; ++i) 
			if(BV_BLOCK_NOT_SUBSET(prefix_blocks[i], packets[stream_id][m*Blocks + i]))
				goto skip_message;

		// add this as a candidate message for phase-three matching
		candidate_messages[atomicAdd(&candidate_count, 1)] =  m;
skip_message: ;
	}
	// end of phase 2 (candidate selection)
	//
	// phase_three:
	__syncthreads();
#if TIME
	if(id==0)
		t3=clock() ;
#endif
	// 
	// phase 3: matching candidate messages.
	// 
	if (candidate_count == PACKETS_BATCH_SIZE){
		goto match_all;
	}
	if(candidate_count==0 || id >=fib_size)
		return;
	else{
		unsigned int prefix_end = prefix_complete_blocks;
		uint32_t d0,d1,d2,d3,d4,d5;
		switch(prefix_end) {
#ifdef COALESCED_READS
			case 0: 
				d0 = fib[id];
			case 1: 
				d1 = fib[id+fib_size * 1];
			case 2: 
				d2 = fib[id+fib_size * 2];
			case 3: 
				d3 = fib[id+fib_size * 3];
			case 4: 
				d4 = fib[id+fib_size * 4];
			case 5: 
				d5 = fib[id+fib_size * 5];
#else
			case 0: 
				d0 = fib[id*Blocks];
			case 1: 
				d1 = fib[id*Blocks + 1];
			case 2: 
				d2 = fib[id*Blocks + 2];
			case 3: 
				d3 = fib[id*Blocks + 3];
			case 4: 
				d4 = fib[id*Blocks + 4];
			case 5: 
				d5 = fib[id*Blocks + 5];
#endif
		}
		uint8_t message; 
		uint8_t message2; 
		uint32_t p0,p1,p2,p3,p4,p5;
		uint32_t p6,p7,p8,p9,p10,p11;

		uint32_t * p ;
		uint8_t c_count = candidate_count; 
		switch(prefix_end) {
			case 0: 
				if (candidate_count % 2 == 1){
						c_count = candidate_count - 1;
						message = candidate_messages[c_count];
						p = &(packets[stream_id][message*Blocks]);
						p0 = *p++;
						p1 = *p++;
						p2 = *p++;
						p3 = *p++;
						p4 = *p++;
						p5 = *p;
						if (BV_BLOCK_NOT_SUBSET(d0, p0)) 
							goto candidate_no_match063;
						if (BV_BLOCK_NOT_SUBSET(d1, p1))
							goto candidate_no_match063;
						if (BV_BLOCK_NOT_SUBSET(d2, p2))
							goto candidate_no_match063;
						if (BV_BLOCK_NOT_SUBSET(d3, p3))
							goto candidate_no_match063;
						if (BV_BLOCK_NOT_SUBSET(d4, p4))
							goto candidate_no_match063;
						if (BV_BLOCK_NOT_SUBSET(d5, p5))
							goto candidate_no_match063;

						RECORD_MATCHING_FILTER(id,message);
				}
candidate_no_match063:

				for(unsigned int pi = 0; pi < c_count; pi+=2) {
					message = candidate_messages[pi];
					p = &(packets[stream_id][message*Blocks]);
					p0 = *p++;
					p1 = *p++;
					p2 = *p++;
					p3 = *p++;
					p4 = *p++;
					p5 = *p;
					message2 = candidate_messages[pi+1];
					p = &(packets[stream_id][message2*Blocks]);
					p6 = *p++;
					p7 = *p++;
					p8 = *p++;
					p9 = *p++;
					p10= *p++;
					p11= *p;

					if (BV_BLOCK_NOT_SUBSET(d0, p0)) 
						goto candidate_no_match062;
					if (BV_BLOCK_NOT_SUBSET(d1, p1))
						goto candidate_no_match062;
					if (BV_BLOCK_NOT_SUBSET(d2, p2))
						goto candidate_no_match062;
					if (BV_BLOCK_NOT_SUBSET(d3, p3))
						goto candidate_no_match062;
					if (BV_BLOCK_NOT_SUBSET(d4, p4))
						goto candidate_no_match062;
					if (BV_BLOCK_NOT_SUBSET(d5, p5))
						goto candidate_no_match062;

						RECORD_MATCHING_FILTER(id,message);
candidate_no_match062:
					if (BV_BLOCK_NOT_SUBSET(d0, p6)) 
						goto candidate_no_match061;
					if (BV_BLOCK_NOT_SUBSET(d1, p7))
						goto candidate_no_match061;
					if (BV_BLOCK_NOT_SUBSET(d2, p8))
						goto candidate_no_match061;
					if (BV_BLOCK_NOT_SUBSET(d3, p9))
						goto candidate_no_match061;
					if (BV_BLOCK_NOT_SUBSET(d4, p10))
						goto candidate_no_match061;
					if (BV_BLOCK_NOT_SUBSET(d5, p11))
						goto candidate_no_match061;

						RECORD_MATCHING_FILTER(id,message2);
candidate_no_match061:
					continue;

				}
				break;
			case 1: 
				for(unsigned int pi = 0; pi < candidate_count; ++pi) {
					message = candidate_messages[pi];
					if (BV_BLOCK_NOT_SUBSET(d1, packets[stream_id][message*Blocks + 1]))
						goto candidate_no_match16;
					if (BV_BLOCK_NOT_SUBSET(d2, packets[stream_id][message*Blocks + 2]))
						goto candidate_no_match16;
					if (BV_BLOCK_NOT_SUBSET(d3, packets[stream_id][message*Blocks + 3]))
						goto candidate_no_match16;
					if (BV_BLOCK_NOT_SUBSET(d4, packets[stream_id][message*Blocks + 4]))
						goto candidate_no_match16;
					if (BV_BLOCK_NOT_SUBSET(d5, packets[stream_id][message*Blocks + 5]))
						goto candidate_no_match16;

						RECORD_MATCHING_FILTER(id,message);
candidate_no_match16:
					continue;
				}
				break;

			case 2: 
				for(unsigned int pi = 0; pi < candidate_count; ++pi) {
					message = candidate_messages[pi];
					if (BV_BLOCK_NOT_SUBSET(d2, packets[stream_id][message*Blocks + 2]))
						goto candidate_no_match26;
					if (BV_BLOCK_NOT_SUBSET(d3, packets[stream_id][message*Blocks + 3]))
						goto candidate_no_match26;
					if (BV_BLOCK_NOT_SUBSET(d4, packets[stream_id][message*Blocks + 4]))
						goto candidate_no_match26;
					if (BV_BLOCK_NOT_SUBSET(d5, packets[stream_id][message*Blocks + 5]))
						goto candidate_no_match26;
						
					RECORD_MATCHING_FILTER(id,message);
candidate_no_match26:
					continue;
				}
				break;

			case 3: 
				for(unsigned int pi = 0; pi < candidate_count; ++pi) {
					message = candidate_messages[pi];
					if (BV_BLOCK_NOT_SUBSET(d3, packets[stream_id][message*Blocks + 3]))
						goto candidate_no_match36;
					if (BV_BLOCK_NOT_SUBSET(d4, packets[stream_id][message*Blocks + 4]))
						goto candidate_no_match36;
					if (BV_BLOCK_NOT_SUBSET(d5, packets[stream_id][message*Blocks + 5]))
						goto candidate_no_match36;
						
					RECORD_MATCHING_FILTER(id,message);
candidate_no_match36:
					continue;
				}
				break;

			case 4: 
				for(unsigned int pi = 0; pi < candidate_count; ++pi) {
					message = candidate_messages[pi];
					if (BV_BLOCK_NOT_SUBSET(d4, packets[stream_id][message*Blocks + 4]))
						goto candidate_no_match46;
					if (BV_BLOCK_NOT_SUBSET(d5, packets[stream_id][message*Blocks + 5]))
						goto candidate_no_match46;

					RECORD_MATCHING_FILTER(id,message);
candidate_no_match46:
					continue;
				}
				break;

			case 5: 
				for(unsigned int pi = 0; pi < candidate_count; ++pi) {
					message = candidate_messages[pi];
					if (BV_BLOCK_NOT_SUBSET(d5, packets[stream_id][message*Blocks + 5]))
						goto candidate_no_match56;
						
					RECORD_MATCHING_FILTER(id,message);
candidate_no_match56:
					continue;
				}
				break;
		}
	}
#if TIME
	if(id==0){
		t4=clock(); 
		printf("%u c= %u %u %u %u\n",fib_size ,candidate_count, (t2-t1), (t3-t2), (t4-t3));
	}
#endif
	return;
match_all:	

	// there is no usable prefix or the common prefix is all-zero,
	// so we fall back to the basic matching procedure, possibly
	// skipping the all-zero prefix blocks.

	//this explicit return improves the performance!
	if(id >= fib_size)
		return ;
	else {
			unsigned int prefix_end = prefix_complete_blocks;
			if(prefix_end!=0){
				uint32_t d1,d2,d3,d4,d5;
				switch(prefix_end) {
#ifdef COALESCED_READS
					case 1: 
						d1 = fib[id+fib_size * 1];
					case 2: 
						d2 = fib[id+fib_size * 2];
					case 3: 
						d3 = fib[id+fib_size * 3];
					case 4: 
						d4 = fib[id+fib_size * 4];
					case 5: 
						d5 = fib[id+fib_size * 5];
#else
					case 1: 
						d1 = fib[id*Blocks + 1];
					case 2: 
						d2 = fib[id*Blocks + 2];
					case 3: 
						d3 = fib[id*Blocks + 3];
					case 4: 
						d4 = fib[id*Blocks + 4];
					case 5: 
						d5 = fib[id*Blocks + 5];
#endif
				}
				switch(prefix_end) {
					case 1: 
						for(unsigned int pi = 0; pi < batch_size; ++pi) {
							if (BV_BLOCK_NOT_SUBSET(d1, packets[stream_id][pi*Blocks + 1]))
								goto candidate_no_match16A;
							if (BV_BLOCK_NOT_SUBSET(d2, packets[stream_id][pi*Blocks + 2]))
								goto candidate_no_match16A;
							if (BV_BLOCK_NOT_SUBSET(d3, packets[stream_id][pi*Blocks + 3]))
								goto candidate_no_match16A;
							if (BV_BLOCK_NOT_SUBSET(d4, packets[stream_id][pi*Blocks + 4]))
								goto candidate_no_match16A;
							if (BV_BLOCK_NOT_SUBSET(d5, packets[stream_id][pi*Blocks + 5]))
								goto candidate_no_match16A;

							RECORD_MATCHING_FILTER(id,pi);
candidate_no_match16A:
							continue;
						}
						break;
					case 2: 
						for(unsigned int pi = 0; pi < batch_size; ++pi) {
							if (BV_BLOCK_NOT_SUBSET(d2, packets[stream_id][pi*Blocks + 2]))
								goto candidate_no_match26A;
							if (BV_BLOCK_NOT_SUBSET(d3, packets[stream_id][pi*Blocks + 3]))
								goto candidate_no_match26A;
							if (BV_BLOCK_NOT_SUBSET(d4, packets[stream_id][pi*Blocks + 4]))
								goto candidate_no_match26A;
							if (BV_BLOCK_NOT_SUBSET(d5, packets[stream_id][pi*Blocks + 5]))
								goto candidate_no_match26A;

							RECORD_MATCHING_FILTER(id,pi);
candidate_no_match26A:
							continue;
						}
						break;
					case 3: 
						for(unsigned int pi = 0; pi < batch_size; ++pi) {
							if (BV_BLOCK_NOT_SUBSET(d3, packets[stream_id][pi*Blocks + 3]))
								goto candidate_no_match36A;
							if (BV_BLOCK_NOT_SUBSET(d4, packets[stream_id][pi*Blocks + 4]))
								goto candidate_no_match36A;
							if (BV_BLOCK_NOT_SUBSET(d5, packets[stream_id][pi*Blocks + 5]))
								goto candidate_no_match36A;
							
							RECORD_MATCHING_FILTER(id,pi);
candidate_no_match36A:
							continue;
						}
						break;
					case 4: 
						for(unsigned int pi = 0; pi < batch_size ; ++pi) {
							if (BV_BLOCK_NOT_SUBSET(d4, packets[stream_id][pi*Blocks + 4]))
								goto candidate_no_match46A;
							if (BV_BLOCK_NOT_SUBSET(d5, packets[stream_id][pi*Blocks + 5]))
								goto candidate_no_match46A;

							RECORD_MATCHING_FILTER(id,pi);
candidate_no_match46A:
							continue;
						}
						break;
					case 5: 
						for(unsigned int pi = 0; pi < batch_size; ++pi) {
							if (BV_BLOCK_NOT_SUBSET(d5, packets[stream_id][pi*Blocks + 5]))
								goto candidate_no_match56A;

							RECORD_MATCHING_FILTER(id,pi);
candidate_no_match56A:
							continue;
						}
						break;
				}
				return ;
			}

		uint32_t d0,d1,d2,d3,d4,d5;
#ifdef COALESCED_READS
		d5 = fib[id+fib_size * 5];
		d4 = fib[id+fib_size * 4];
		d3 = fib[id+fib_size * 3];
		d2 = fib[id+fib_size * 2];
		d1 = fib[id+fib_size * 1];
		d0 = fib[id+fib_size * 0];
#else
		d5 = fib[id*Blocks + 5];
		d4 = fib[id*Blocks + 4];
		d3 = fib[id*Blocks + 3];
		d2 = fib[id*Blocks + 2];
		d1 = fib[id*Blocks + 1];
		d0 = fib[id*Blocks + 0];
#endif
		uint32_t p0,p1,p2,p3,p4,p5;
		uint32_t p6,p7,p8,p9,p10,p11;
		uint32_t * p;

		p = &(packets[stream_id][0]);
		for(unsigned int pi = 0; pi < batch_size; pi+=2) {
			p0 =*p++;
			p1 = *p++;
			p2 = *p++;
			p3 = *p++;
			p4 = *p++;
			p5 = *p++;
			p6 = *p++;
			p7 = *p++;
			p8 = *p++;
			p9 = *p++;
			p10= *p++;
			p11= *p++;

			if (BV_BLOCK_NOT_SUBSET(d0, p0))
				goto next_msg;
			if (BV_BLOCK_NOT_SUBSET(d1, p1))
				goto next_msg;
			if (BV_BLOCK_NOT_SUBSET(d2, p2))
				goto next_msg;
			if (BV_BLOCK_NOT_SUBSET(d3, p3))
				goto next_msg;
			if (BV_BLOCK_NOT_SUBSET(d4, p4))
				goto next_msg;
			if (BV_BLOCK_NOT_SUBSET(d5, p5))
				goto next_msg;

			RECORD_MATCHING_FILTER(id,pi);
next_msg:
			if (BV_BLOCK_NOT_SUBSET(d0, p6))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d1, p7))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d2, p8))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d3, p9))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d4, p10))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d5, p11))
				continue;
		
			RECORD_MATCHING_FILTER(id,pi+1);
		}

#if TIME
//		if(id==0){
//			t4=clock(); 
//			printf("%u c= %u %u %u %u\n",fib_size ,candidate_count, (t2-t1), (t3-t2), (t4-t3));
//		}
#endif

	}
}

#if 0
template <unsigned int Blocks>
__global__ void one_phase_matching(uint32_t * fib, uint32_t * fib_ids, 
								   unsigned int fib_size, 
								   uint16_t * query_ti_table ,  unsigned int batch_size, 
								   result_t * results_count,
								   result_t * results_data,
								   unsigned int stream_id) {

	uint32_t id = (blockDim.x * blockIdx.x) + threadIdx.x;

	uint32_t rescount;
	if(id >= fib_size)
		return;


	uint32_t d0,d1,d2,d3,d4,d5;

	switch(Blocks) {
#ifdef COALESCED_READS
		case 6:	d5 = fib[id+fib_size * 5];
		case 5:	d4 = fib[id+fib_size * 4];
		case 4: d3 = fib[id+fib_size * 3];
		case 3:	d2 = fib[id+fib_size * 2];
		case 2: d1 = fib[id+fib_size * 1];
		case 1:	d0 = fib[id+fib_size * 0];
#else
		case 6:	d5 = fib[id*Blocks + 5];
		case 5:	d4 = fib[id*Blocks + 4];
		case 4: d3 = fib[id*Blocks + 3];
		case 3:	d2 = fib[id*Blocks + 2];
		case 2: d1 = fib[id*Blocks + 1];
		case 1:	d0 = fib[id*Blocks + 0];
#endif
	}
	if(batch_size==256){
		uint32_t p0,p1,p2,p3,p4,p5;
		uint32_t p6,p7,p8,p9,p10,p11;
		uint32_t * p = &(packets[stream_id][0]);
		for(unsigned int pi = 0; pi < batch_size; pi+=2) {
			p0 =*p++;
			if(Blocks==6){
				p1 = *p++;
				p2 = *p++;
				p3 = *p++;
				p4 = *p++;
				p5 = *p++;
				p6 = *p++;
				p7 = *p++;
				p8 = *p++;
				p9 = *p++;
				p10= *p++;
				p11= *p++;
			}
			if(Blocks==5){
				p1 = *p++;
				p2 = *p++;
				p3 = *p++;
				p4 = *p++;
			}
			if(Blocks==4){
				p1 = *p++;
				p2 = *p++;
				p3 = *p++;
			}
			if(Blocks==3){
				p1 = *p++;
				p2 = *p++;
			}
			if(Blocks==2)
				p1 = *p++;

			if (BV_BLOCK_NOT_SUBSET(d0, p0))
				goto next_msg;
			if (Blocks > 1)
				if (BV_BLOCK_NOT_SUBSET(d1, p1))
					goto next_msg;
			if (Blocks > 2)
				if (BV_BLOCK_NOT_SUBSET(d2, p2))
					goto next_msg;
			if (Blocks > 3)
				if (BV_BLOCK_NOT_SUBSET(d3, p3))
					goto next_msg;
			if (Blocks > 4)
				if (BV_BLOCK_NOT_SUBSET(d4, p4))
					goto next_msg;
			if (Blocks > 5)
				if (BV_BLOCK_NOT_SUBSET(d5, p5))
					goto next_msg;

			RECORD_MATCHING_FILTER(id,pi);
next_msg:

			if (BV_BLOCK_NOT_SUBSET(d0, p6))
				continue;
			if (Blocks > 1)
				if (BV_BLOCK_NOT_SUBSET(d1, p7))
					continue;
			if (Blocks > 2)
				if (BV_BLOCK_NOT_SUBSET(d2, p8))
					continue;
			if (Blocks > 3)
				if (BV_BLOCK_NOT_SUBSET(d3, p9))
					continue;
			if (Blocks > 4)
				if (BV_BLOCK_NOT_SUBSET(d4, p10))
					continue;
			if (Blocks > 5)
				if (BV_BLOCK_NOT_SUBSET(d5, p11))
					continue;

			RECORD_MATCHING_FILTER(id,pi+1);
		}
	}
	else {
		uint32_t p0,p1,p2,p3,p4,p5;
		uint32_t * p = &(packets[stream_id][0]);
		for(unsigned int pi = 0; pi < batch_size; ++pi) {
			p0 =*p++;
			if(Blocks==6){
				p1 = *p++;
				p2 = *p++;
				p3 = *p++;
				p4 = *p++;
				p5 = *p++;
			}
			if(Blocks==5){
				p1 = *p++;
				p2 = *p++;
				p3 = *p++;
				p4 = *p++;
			}
			if(Blocks==4){
				p1 = *p++;
				p2 = *p++;
				p3 = *p++;
			}
			if(Blocks==3){
				p1 = *p++;
				p2 = *p++;
			}
			if(Blocks==2)
				p1 = *p++;


			if (BV_BLOCK_NOT_SUBSET(d0, p0))
				continue;
			if (Blocks > 1)
				if (BV_BLOCK_NOT_SUBSET(d1, p1))
					continue;
			if (Blocks > 2)
				if (BV_BLOCK_NOT_SUBSET(d2, p2))
					continue;
			if (Blocks > 3)
				if (BV_BLOCK_NOT_SUBSET(d3, p3))
					continue;
			if (Blocks > 4)
				if (BV_BLOCK_NOT_SUBSET(d4, p4))
					continue;
			if (Blocks > 5)
				if (BV_BLOCK_NOT_SUBSET(d5, p5))
					continue;

			RECORD_MATCHING_FILTER(id,pi);
		}
	}
}
#endif

void gpu::initialize() {
	for (int g = 0; g < GPU_NUM; g++) {
		cudaSetDevice(g);
		cudaDeviceReset() ; 
		ABORT_ON_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		ABORT_ON_ERROR(cudaDeviceSynchronize());
		ABORT_ON_ERROR(cudaThreadSynchronize());
		for(unsigned int i = 0; i < GPU_STREAMS; ++i) {
			ABORT_ON_ERROR(cudaStreamCreate(streams[g] + i));
			ABORT_ON_ERROR(cudaEventCreateWithFlags(&copiedBack[g][i], cudaEventDisableTiming));
		}
	}
}

void gpu::set_device(unsigned int dev) {
	ABORT_ON_ERROR(cudaSetDevice(dev));
}

void gpu::mem_info(gpu_mem_info * mi) {
	gpu_mem_info temp;
	mi->free = 0;
	mi->total = 0;
	for (int i = 0; i < GPU_NUM; i++) {
		ABORT_ON_ERROR(cudaSetDevice(i));
		ABORT_ON_ERROR(cudaDeviceSynchronize());
		ABORT_ON_ERROR(cudaMemGetInfo(&(temp.free), &(temp.total)));
		mi->free += temp.free;
		mi->total += temp.total;
	}
}

void gpu::async_copy_packets(unsigned int * host_packets, unsigned int size , unsigned int stream, unsigned int gpu) {
	ABORT_ON_ERROR(cudaMemcpyToSymbolAsync(packets, host_packets, size*sizeof(unsigned int), stream*PACKETS_BATCH_SIZE*GPU_FILTER_WORDS*sizeof(unsigned int), cudaMemcpyHostToDevice, streams[gpu][stream]));
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

void gpu::async_copy(void * host_src, void * dev_dst, unsigned int size, unsigned int stream_id, unsigned int gpu) {
	ABORT_ON_ERROR(cudaMemcpyAsync(dev_dst, host_src, size, cudaMemcpyHostToDevice, streams[gpu][stream_id]));
}

 // this is useful for clearing the dev_res (interfaces) to 0 before
 // calling the kernel
void gpu::async_set_zero(void * dev_array, unsigned int size, unsigned int stream_id, unsigned int gpu) {
	ABORT_ON_ERROR(cudaMemsetAsync(dev_array, 0, size, streams[gpu][stream_id]));
}


void gpu::async_get_results(result_t * host_results, result_t * dev_results, 
							unsigned int size, unsigned int stream, unsigned int gpu) {
	// count + size * uint32 ids + floor(size/4) uint32 for msg ids
		ABORT_ON_ERROR(cudaMemcpyAsync(host_results, dev_results, sizeof(uint32_t)*(1+size+(size+3)/4), cudaMemcpyDeviceToHost, streams[gpu][stream]));
	ABORT_ON_ERROR(cudaEventRecord(copiedBack[gpu][stream], streams[gpu][stream]));
}

void gpu::syncOnResults(unsigned int stream, unsigned int gpu) {
	ABORT_ON_ERROR(cudaEventSynchronize(copiedBack[gpu][stream]));
}

void gpu::get_results(ifx_result_t * host_results, ifx_result_t * dev_results, unsigned int size) {
	ABORT_ON_ERROR(cudaMemcpy(host_results, dev_results, size * sizeof(ifx_result_t), cudaMemcpyDeviceToHost));
}

void gpu::synchronize_device() {
	ABORT_ON_ERROR(cudaDeviceSynchronize());
}

void gpu::synchronize_stream(unsigned int stream, unsigned int gpu) {
	ABORT_ON_ERROR(cudaStreamSynchronize(streams[gpu][stream]));
}

void * gpu::allocate_host_pinned_generic(unsigned int size) {
	void * host_array_pinned;
	ABORT_ON_ERROR(cudaMallocHost(&host_array_pinned, size));
	return host_array_pinned;
}


#if PROFILE
static const int stopCounter=55 ; 
sig_atomic_t kernelCounter=0 ;
#endif

void gpu::run_kernel(uint32_t * fib, 
					 unsigned int fib_size, 
					 unsigned int batch_size, 
					 result_t * results_count, 
					 result_t * results_data,
					 unsigned int stream, 
					 unsigned int gpu,
					 unsigned char skip_blocks){

	unsigned int gridsize = fib_size/GPU_BLOCK_SIZE;
	if ((fib_size % GPU_BLOCK_SIZE) != 0)
		++gridsize;
	
#if PROFILE
	kernelCounter++;
	if(kernelCounter==1)
	      cudaProfilerStart();
	if (kernelCounter > stopCounter)
	      return;
#endif
	three_phase_matching <<<gridsize, GPU_BLOCK_SIZE, 0, streams[gpu][stream] >>>
			(fib, fib_size, batch_size, results_count, results_data, stream);


	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr, "Fatal error: run_kernel: %s\n(%s:%d)\nABORTING\n", 
				cudaGetErrorString(status), __FILE__, __LINE__);
		cudaDeviceReset() ; 
		exit(0); 
	} 
#if PROFILE
	if (kernelCounter==stopCounter)
		cudaProfilerStop(); 
#endif
}

void gpu::shutdown() {
	for (int i = 0; i < GPU_NUM; i++) {
		ABORT_ON_ERROR(cudaSetDevice(i));
		for(unsigned int j = 0; j < GPU_STREAMS; ++j)
			ABORT_ON_ERROR(cudaStreamDestroy(streams[i][j]));
		cudaDeviceSynchronize();
		cudaDeviceReset();
	}
}

void gpu::release_memory(void * p) {
	ABORT_ON_ERROR(cudaFree(p)); 
}

void gpu::release_pinned_memory(void * p) {
	ABORT_ON_ERROR(cudaFreeHost(p)); 
}
