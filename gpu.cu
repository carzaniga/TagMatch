// -*- C++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "gpu.hh"

#include "parameters.hh"
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

__align__(128) __constant__ __device__
uint32_t packets[GPU_STREAMS][PACKETS_BATCH_SIZE*GPU_FILTER_WORDS];

//__device__ bool BV_BLOCK_NOT_SUBSET(uint32_t a, uint32_t b) { 
//	return ((a | b) != (~0U));
//}

#define BV_BLOCK_NOT_SUBSET(x,y) (((x) & (y)) != (x))
// Here is the logic:
// if prefix_exists == 0, I do not check common_prefix
// common_blocks[] holds the full 32 blocks of the prefix.
// if len % 32 !=0 then the rest will be stored in shm_common_prefix

// prefix_exists == 0 shm_common_counter == -1 --> nothing
// prefix_exists == 1 shm_common_counter ==  0 --> 0 < len < 32
// prefix_exists == 0 shm_common_counter ==  0 --> len = 32
// prefix_exists == 1 shm_common_counter ==  1 --> 32 < len < 64
// prefix_exists == 0 shm_common_counter ==  1 --> len = 64
// prefix_exists == 1 shm_common_counter ==  2 --> 64 < len < 96
// prefix_exists == 0 shm_common_counter ==  2 --> len = 96


__global__ void k6(uint32_t * fib, unsigned int fib_size, 
		uint16_t * ti_table, unsigned int * ti_indexes,  
		uint16_t * query_ti_table ,  unsigned int batch_size, 
		ifx_result_t * results,
		unsigned int stream_id) {

	__shared__ uint32_t shm_common_prefix;
	//access to shared memory here is bank conflict free!! here is the reason (from documentation):
	//"An exception is the case where all threads in a warp address the same shared memory address, resulting in a broadcast"
	__align__(128) 	__shared__ uint16_t shortened_msg[PACKETS_BATCH_SIZE];
	__shared__ uint32_t msg_counter; 
	__shared__ bool prefix_exists;
	__shared__ uint32_t common_blocks[5]; 
	__shared__ int32_t shm_common_block_counter; 

	uint32_t tid= (blockDim.x * threadIdx.y) + threadIdx.x;
	uint32_t id = (blockDim.x * blockDim.y * blockIdx.x) + tid; 

	uint32_t last_thread_id;
	unsigned char blocks =6;

	//it is faster to check if current block is the last block only in tid==0 rather than
	// check it at the begining of every block. I did it that way and it was slower!.
	if (tid==0){
		msg_counter = 0;
		shm_common_prefix = 0;
		prefix_exists = true;
		if((fib_size-1) - id < GPU_BLOCK_SIZE)
			last_thread_id = fib_size-1;
		else
			last_thread_id = id + GPU_BLOCK_SIZE-1;

		uint32_t next_block_of_first_tid ; 

		shm_common_block_counter = 0 ;// -1;

		if (id == last_thread_id){
			shm_common_block_counter-- ;
			goto LBL16;
		}

		//		printf("id= %d fb= %u " ,id, fib[ id * (GPU_FILTER_WORDS - full_blocks)]);
		while(true){
			next_block_of_first_tid = fib[ id * (blocks) +  shm_common_block_counter]; 
			uint32_t next_block_last_tid = fib[ last_thread_id * blocks + shm_common_block_counter]; 
			// this is the position of the first 1
			int pos = __ffs(next_block_of_first_tid ^ next_block_last_tid)-1;
			if(pos!=-1){// pos == -1 when both blocks are identical 
				if (pos == 0){ //nothing in common! len(prefix) = 32x 
					prefix_exists=false; 
					shm_common_block_counter--;
				}
				else{
					next_block_of_first_tid<<=(32-pos);
					next_block_of_first_tid>>=(32-pos);
					if (next_block_of_first_tid == 0 ) 
					{
						prefix_exists=false; 
						shm_common_block_counter--;
					}
				}
				break ;
			}
			common_blocks[shm_common_block_counter] = next_block_of_first_tid ;
			shm_common_block_counter++ ;
		}
		shm_common_prefix = next_block_of_first_tid; // if len=32x, shm_common_prefix is meaningless.
LBL16:

	} // end for if(tid==0)
	__syncthreads();
	int32_t common_block_counter = shm_common_block_counter ;
	uint32_t d[6];//the actual size is dynamic = noBlocks
	uint32_t old;
	//there are cases were there is no common prefix! so we have to check this.
	if(common_block_counter!=-1){
		bool have_to_check_prefix = prefix_exists;
		uint32_t common_prefix = shm_common_prefix ;

		//		you dont need to seriously focus on this code :p 
		//		if(have_to_check_prefix && BV_BLOCK_NOT_SUBSET(common_prefix, packets[stream_id][(tid + m)* GPU_MSG_BLOCKS  + common_block_counter+1]))
		//				goto bla;	
		//		for(int i=0; i < common_block_counter+2 - have_to_check_prefix; i++) 
		//			if(BV_BLOCK_NOT_SUBSET(common_blocks[i], packets[stream_id][(tid + m)* GPU_MSG_BLOCKS + i]))
		//				goto bla;	

		if(have_to_check_prefix){
#pragma unroll
			for(int m=0; m < batch_size; m+=GPU_BLOCK_SIZE){
				if(BV_BLOCK_NOT_SUBSET(common_prefix, packets[stream_id][(tid + m) * blocks + common_block_counter]))
					goto bla6;	
				for(int i=0; i < common_block_counter; i++) 
					if(BV_BLOCK_NOT_SUBSET(common_blocks[i], packets[stream_id][(tid + m) * blocks + i]))
						goto bla6;	
				old=atomicAdd(&msg_counter, 1);
				shortened_msg[old]= tid + m;
bla6:
			}
		}
		else {
#pragma unroll
			for(int m=0; m < batch_size; m+=GPU_BLOCK_SIZE){
				for(int i=0; i <= common_block_counter; i++) 
					if(BV_BLOCK_NOT_SUBSET(common_blocks[i], packets[stream_id][(tid + m) * blocks + i]))
						goto bla26;	

				old=atomicAdd(&msg_counter, 1);
				shortened_msg[old]= tid + m;
bla26:
			}
		}

		__syncthreads();
		//here we know for sure that we already have matching candidates in the Shared memory. 
		if(id >= fib_size || msg_counter==0)
			return;

		if(have_to_check_prefix){
			for(int i= common_block_counter; i < blocks; i++)
				d[i]=fib[ id * blocks + i];

			for(uint32_t mi = 0; mi < msg_counter; ++mi) {  
				uint16_t msg_index= shortened_msg[mi] ;
				for(uint32_t i = common_block_counter; i < blocks; ++i) 
					if(BV_BLOCK_NOT_SUBSET(d[i], packets[stream_id][msg_index * blocks + i]))
						goto next_message6_1;

				unsigned int ti_index = ti_indexes[id];
				for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
					// TODO: document these cryptic operations.
					uint16_t ti_xor = query_ti_table[msg_index] ^ ti_table[ti_index + i];
					if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) {
						results[msg_index*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
					}
				}

next_message6_1: 
			}
		}
		else{
			for(int i= common_block_counter+1; i < blocks; i++)
				d[i]=fib[ id * blocks + i];

			for(uint32_t mi = 0; mi < msg_counter; ++mi) {  
				uint16_t msg_index= shortened_msg[mi] ;
				for(int32_t i = common_block_counter+1; i < blocks; ++i) 
					if(BV_BLOCK_NOT_SUBSET(d[i], packets[stream_id][msg_index * blocks + i]))
						goto next_message6_2;

				unsigned int ti_index = ti_indexes[id];
				for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
					// TODO: document these cryptic operations.
					uint16_t ti_xor = query_ti_table[msg_index] ^ ti_table[ti_index + i];
					if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) {
						results[msg_index*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
					}
				}
next_message6_2: 
			}
		}
	}
	else{
		uint32_t d[6];
		if(id >= fib_size)
			return;

		for(int i=0; i < 6; i++)
			d[i]=fib[ id * 6 + i];

		for(unsigned int pi = 0; pi < batch_size; ++pi) {
			if (BV_BLOCK_NOT_SUBSET(d[0], packets[stream_id][pi*6 + 0]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[1], packets[stream_id][pi*6 + 1]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[2], packets[stream_id][pi*6 + 2]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[3], packets[stream_id][pi*6 + 3]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[4], packets[stream_id][pi*6 + 4]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[5], packets[stream_id][pi*6 + 5]))
				continue;

			unsigned int ti_index = ti_indexes[id];
			for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
				uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
				if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) 
					results[pi*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
			}
		}
	}
}

__global__ void k5(uint32_t * fib, unsigned int fib_size, 
		uint16_t * ti_table, unsigned int * ti_indexes,  
		uint16_t * query_ti_table ,  unsigned int batch_size, 
		ifx_result_t * results,
		unsigned int stream_id) {

	__shared__ uint32_t shm_common_prefix;
	__align__(128) 	__shared__ uint16_t shortened_msg[PACKETS_BATCH_SIZE];
	__shared__ uint32_t msg_counter; 
	__shared__ bool prefix_exists;
	__shared__ uint32_t common_blocks[4]; 
	__shared__ int32_t shm_common_block_counter; 

	uint32_t tid= (blockDim.x * threadIdx.y) + threadIdx.x;
	uint32_t id = (blockDim.x * blockDim.y * blockIdx.x) + tid; 

	uint32_t last_thread_id;
	unsigned char blocks =5;

	if (tid==0){
		msg_counter= 0;
		shm_common_prefix= 0;
		prefix_exists=true;
		if((fib_size-1) - id < GPU_BLOCK_SIZE)
			last_thread_id = fib_size-1;
		else
			last_thread_id = id + GPU_BLOCK_SIZE-1;

		uint32_t next_block_of_first_tid ; 

		shm_common_block_counter = 0;

		if (id == last_thread_id){
			shm_common_block_counter-- ;
			goto LBL15;
		}

		while(true){
			next_block_of_first_tid = fib[ id * (blocks) +  shm_common_block_counter]; 
			uint32_t next_block_last_tid = fib[ last_thread_id * blocks + shm_common_block_counter]; 
			// this is the position of the first 1
			int pos = __ffs(next_block_of_first_tid ^ next_block_last_tid)-1;
			if(pos!=-1){// pos == -1 when both blocks are identical 
				if (pos == 0){ //nothing in common! len(prefix) = 32x 
					prefix_exists=false; 
					shm_common_block_counter--;
				}
				else{
					next_block_of_first_tid<<=(32-pos);
					next_block_of_first_tid>>=(32-pos);
					if (next_block_of_first_tid == 0 ) 
					{
						prefix_exists=false; 
						shm_common_block_counter--;
					}
				}
				break ;
			}
			common_blocks[shm_common_block_counter] = next_block_of_first_tid ;
			shm_common_block_counter++ ;
		}
		shm_common_prefix = next_block_of_first_tid; // if len=32x, shm_common_prefix is meaningless.
LBL15:

	}// end for if(tid==0)
	__syncthreads();
	int32_t common_block_counter = shm_common_block_counter ;
	uint32_t d[5];
	uint32_t old;
	if(common_block_counter!=-1){
		bool have_to_check_prefix = prefix_exists;
		uint32_t common_prefix = shm_common_prefix ;

		if(have_to_check_prefix){
#pragma unroll
			for(int m=0; m < batch_size; m+=GPU_BLOCK_SIZE){
				if(BV_BLOCK_NOT_SUBSET(common_prefix, packets[stream_id][(tid + m) * blocks + common_block_counter]))
					goto bla5;	
				for(int i=0; i < common_block_counter; i++) 
					if(BV_BLOCK_NOT_SUBSET(common_blocks[i], packets[stream_id][(tid + m) * blocks + i]))
						goto bla5;	
				old=atomicAdd(&msg_counter, 1);
				shortened_msg[old]= tid + m;
bla5:
			}
		}
		else {
#pragma unroll
			for(int m=0; m < batch_size; m+=GPU_BLOCK_SIZE){
				for(int i=0; i <= common_block_counter; i++) 
					if(BV_BLOCK_NOT_SUBSET(common_blocks[i], packets[stream_id][(tid + m) * blocks + i]))
						goto bla25;	

				old=atomicAdd(&msg_counter, 1);
				shortened_msg[old]= tid + m;
bla25:
			}
		}

		__syncthreads();

		if(id >= fib_size || msg_counter==0)
			return;

		if(have_to_check_prefix){
			for(int i= common_block_counter; i < blocks; i++)
				d[i]=fib[ id * blocks + i];

			for(uint32_t mi = 0; mi < msg_counter; ++mi) {  
				uint16_t msg_index= shortened_msg[mi] ;
				for(uint32_t i = common_block_counter; i < blocks; ++i) 
					if(BV_BLOCK_NOT_SUBSET(d[i], packets[stream_id][msg_index* blocks+i]))
						goto next_message5_1;

				unsigned int ti_index = ti_indexes[id];
				for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
					// TODO: document these cryptic operations.
					uint16_t ti_xor = query_ti_table[msg_index] ^ ti_table[ti_index + i];
					if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) {
						results[msg_index*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
					}
				}

next_message5_1: 
			}
		}
		else{
			for(int i= common_block_counter+1; i < blocks; i++)
				d[i]=fib[ id * blocks + i];

			for(uint32_t mi = 0; mi < msg_counter; ++mi) {  
				uint16_t msg_index= shortened_msg[mi] ;
				for(int32_t i = common_block_counter+1; i < blocks; ++i) 
					if(BV_BLOCK_NOT_SUBSET(d[i],packets[stream_id][msg_index* blocks+i]))
						goto next_message5_2;

				unsigned int ti_index = ti_indexes[id];
				for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
					// TODO: document these cryptic operations.
					uint16_t ti_xor = query_ti_table[msg_index] ^ ti_table[ti_index + i];
					if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) {
						results[msg_index*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
					}
				}
next_message5_2: 
			}
		}
	}
	else{
		uint32_t d[5];
		if(id >= fib_size)
			return;

		for(int i=0; i < 5; i++)
			d[i]=fib[ id * 5 + i];

		for(unsigned int pi = 0; pi < batch_size; ++pi) {
			if (BV_BLOCK_NOT_SUBSET(d[0], packets[stream_id][pi*5 + 0]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[1], packets[stream_id][pi*5 + 1]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[2], packets[stream_id][pi*5 + 2]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[3], packets[stream_id][pi*5 + 3]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[4], packets[stream_id][pi*5 + 4]))
				continue;

			unsigned int ti_index = ti_indexes[id];
			for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
				uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
				if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) 
					results[pi*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
			}
		}
	}
}

__global__ void k4(uint32_t * fib, unsigned int fib_size, 
		uint16_t * ti_table, unsigned int * ti_indexes,  
		uint16_t * query_ti_table ,  unsigned int batch_size, 
		ifx_result_t * results,
		unsigned int stream_id) {

	__shared__ uint32_t shm_common_prefix;
	__align__(128) 	__shared__ uint16_t shortened_msg[PACKETS_BATCH_SIZE];
	__shared__ uint32_t msg_counter; 
	__shared__ bool prefix_exists;
	__shared__ uint32_t common_blocks[3]; 
	__shared__ int32_t shm_common_block_counter; 

	uint32_t tid= (blockDim.x * threadIdx.y) + threadIdx.x;
	uint32_t id = (blockDim.x * blockDim.y * blockIdx.x) + tid; 

	uint32_t last_thread_id;
	unsigned char blocks =4;

	if (tid==0){
		msg_counter= 0;
		shm_common_prefix= 0;
		prefix_exists=true;
		if((fib_size-1) - id < GPU_BLOCK_SIZE)
			last_thread_id = fib_size-1;
		else
			last_thread_id = id + GPU_BLOCK_SIZE-1;

		uint32_t next_block_of_first_tid ; 

		shm_common_block_counter = 0;

		if (id == last_thread_id){
			shm_common_block_counter-- ;
			goto LBL14;
		}

		while(true){
			next_block_of_first_tid = fib[ id * (blocks) +  shm_common_block_counter]; 
			uint32_t next_block_last_tid = fib[ last_thread_id * blocks + shm_common_block_counter]; 
			// this is the position of the first 1
			int pos = __ffs(next_block_of_first_tid ^ next_block_last_tid)-1;
			if(pos!=-1){// pos == -1 when both blocks are identical 
				if (pos == 0){ //nothing in common! len(prefix) = 32x 
					prefix_exists=false; 
					shm_common_block_counter--;
				}
				else{
					next_block_of_first_tid<<=(32-pos);
					next_block_of_first_tid>>=(32-pos);
					if (next_block_of_first_tid == 0 ) 
					{
						prefix_exists=false; 
						shm_common_block_counter--;
					}
				}
				break ;
			}
			common_blocks[shm_common_block_counter] = next_block_of_first_tid ;
			shm_common_block_counter++ ;
		}
		shm_common_prefix = next_block_of_first_tid; // if len=32x, shm_common_prefix is meaningless.
LBL14:

	} // end for if(tid==0)
	__syncthreads();
	int32_t common_block_counter = shm_common_block_counter ;
	uint32_t d[4];
	uint32_t old;
	if(common_block_counter!=-1){
		bool have_to_check_prefix = prefix_exists;
		uint32_t common_prefix = shm_common_prefix ;
		if(have_to_check_prefix){
#pragma unroll
			for(int m=0; m < batch_size; m+=GPU_BLOCK_SIZE){
				if(BV_BLOCK_NOT_SUBSET(common_prefix, packets[stream_id][(tid + m) * blocks + common_block_counter]))
					goto bla4;	
				for(int i=0; i < common_block_counter; i++) 
					if(BV_BLOCK_NOT_SUBSET(common_blocks[i], packets[stream_id][(tid + m) * blocks + i]))
						goto bla4;	
				old=atomicAdd(&msg_counter, 1);
				shortened_msg[old]= tid + m;
bla4:
			}
		}
		else {
#pragma unroll
			for(int m=0; m < batch_size; m+=GPU_BLOCK_SIZE){
				for(int i=0; i <= common_block_counter; i++) 
					if(BV_BLOCK_NOT_SUBSET(common_blocks[i], packets[stream_id][(tid + m) * blocks + i]))
						goto bla24;	
				old=atomicAdd(&msg_counter, 1);
				shortened_msg[old]= tid + m;
bla24:
			}
		}

		__syncthreads();

		if(id >= fib_size || msg_counter==0)
			return;

		if(have_to_check_prefix){
			for(int i= common_block_counter; i < blocks; i++)
				d[i]=fib[ id * blocks + i];

			for(uint32_t mi = 0; mi < msg_counter; ++mi) {  
				uint16_t msg_index= shortened_msg[mi] ;
				for(uint32_t i = common_block_counter; i < blocks; ++i) 
					if(BV_BLOCK_NOT_SUBSET(d[i], packets[stream_id][msg_index * blocks + i]))
						goto next_message4_1;

				unsigned int ti_index = ti_indexes[id];
				for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
					// TODO: document these cryptic operations.
					uint16_t ti_xor = query_ti_table[msg_index] ^ ti_table[ti_index + i];
					if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) {
						results[msg_index*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
					}
				}

next_message4_1: 
			}
		}
		else{
			for(int i= common_block_counter+1; i < blocks; i++)
				d[i]=fib[ id * blocks + i];

			for(uint32_t mi = 0; mi < msg_counter; ++mi) {  
				uint16_t msg_index= shortened_msg[mi] ;
				for(int32_t i = common_block_counter+1; i < blocks; ++i) 
					if(BV_BLOCK_NOT_SUBSET(d[i],packets[stream_id][msg_index * blocks + i]))
						goto next_message4_2;

				unsigned int ti_index = ti_indexes[id];
				for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
					// TODO: document these cryptic operations.
					uint16_t ti_xor = query_ti_table[msg_index] ^ ti_table[ti_index + i];
					if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) {
						results[msg_index*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
					}
				}
next_message4_2: 
			}
		}
	}
	else{
		uint32_t d[4];
		if(id >= fib_size)
			return;

		for(int i=0; i < 4; i++)
			d[i]=fib[ id * 4 + i];

		for(unsigned int pi = 0; pi < batch_size; ++pi) {
			if (BV_BLOCK_NOT_SUBSET(d[0], packets[stream_id][pi*4 + 0]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[1], packets[stream_id][pi*4 + 1]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[2], packets[stream_id][pi*4 + 2]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[3], packets[stream_id][pi*4 + 3]))
				continue;

			unsigned int ti_index = ti_indexes[id];
			for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
				uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
				if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) 
					results[pi*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
			}
		}
	}
}

__global__ void k3(uint32_t * fib, unsigned int fib_size, 
		uint16_t * ti_table, unsigned int * ti_indexes,  
		uint16_t * query_ti_table ,  unsigned int batch_size, 
		ifx_result_t * results,
		unsigned int stream_id) {


	__shared__ uint32_t shm_common_prefix;
	__align__(128) 	__shared__ uint16_t shortened_msg[PACKETS_BATCH_SIZE];
	__shared__ uint32_t msg_counter; 
	__shared__ bool prefix_exists;
	__shared__ uint32_t common_blocks[2]; 
	__shared__ int32_t shm_common_block_counter; 

	uint32_t tid= (blockDim.x * threadIdx.y) + threadIdx.x;
	uint32_t id = (blockDim.x * blockDim.y * blockIdx.x) + tid; 

	uint32_t last_thread_id;
	unsigned char blocks =3;

	if (tid==0){
		msg_counter= 0;
		shm_common_prefix= 0;
		prefix_exists=true;
		if((fib_size-1) - id < GPU_BLOCK_SIZE)
			last_thread_id = fib_size-1;
		else
			last_thread_id = id + GPU_BLOCK_SIZE-1;

		uint32_t next_block_of_first_tid ; 

		shm_common_block_counter = 0;

		if (id == last_thread_id){
			shm_common_block_counter-- ;
			goto LBL13;
		}

		while(true){
			next_block_of_first_tid = fib[ id * (blocks) +  shm_common_block_counter]; 
			uint32_t next_block_last_tid = fib[ last_thread_id * blocks + shm_common_block_counter]; 
			// this is the position of the first 1
			int pos = __ffs(next_block_of_first_tid ^ next_block_last_tid)-1;
			if(pos!=-1){// pos == -1 when both blocks are identical 
				if (pos == 0){ //nothing in common! len(prefix) = 32x 
					prefix_exists=false; 
					shm_common_block_counter--;
				}
				else{
					next_block_of_first_tid<<=(32-pos);
					next_block_of_first_tid>>=(32-pos);
					if (next_block_of_first_tid == 0 ) 
					{
						prefix_exists=false; 
						shm_common_block_counter--;
					}
				}
				break ;
			}
			common_blocks[shm_common_block_counter] = next_block_of_first_tid ;
			shm_common_block_counter++ ;
		}
		shm_common_prefix = next_block_of_first_tid; // if len=32x, shm_common_prefix is meaningless.
LBL13:

	} // end for if(tid==0)
	__syncthreads();
	int32_t common_block_counter = shm_common_block_counter ;
	uint32_t d[3];
	uint32_t old;
	if(common_block_counter!=-1){
		bool have_to_check_prefix = prefix_exists;
		uint32_t common_prefix = shm_common_prefix ;
		if(have_to_check_prefix){
#pragma unroll
			for(int m=0; m < batch_size; m+=GPU_BLOCK_SIZE){
				if(BV_BLOCK_NOT_SUBSET(common_prefix, packets[stream_id][(tid + m) * blocks + common_block_counter]))
					goto bla3;	
				for(int i=0; i < common_block_counter; i++) 
					if(BV_BLOCK_NOT_SUBSET(common_blocks[i], packets[stream_id][(tid + m) * blocks + i]))
						goto bla3;	
				old=atomicAdd(&msg_counter, 1);
				shortened_msg[old]= tid + m;
bla3:
			}
		}
		else {
#pragma unroll
			for(int m=0; m < batch_size; m+=GPU_BLOCK_SIZE){
				for(int i=0; i <= common_block_counter; i++) 
					if(BV_BLOCK_NOT_SUBSET(common_blocks[i], packets[stream_id][(tid + m) * blocks + i]))
						goto bla23;	
				old=atomicAdd(&msg_counter, 1);
				shortened_msg[old]= tid + m;
bla23:
			}
		}

		__syncthreads();

		if(id >= fib_size || msg_counter==0)
			return;

		if(have_to_check_prefix){
			for(int i= common_block_counter; i < blocks; i++)
				d[i]=fib[ id * blocks + i];

			for(uint32_t mi = 0; mi < msg_counter; ++mi) {  
				uint16_t msg_index= shortened_msg[mi] ;
				for(uint32_t i = common_block_counter; i < blocks; ++i) 
					if(BV_BLOCK_NOT_SUBSET(d[i], packets[stream_id][msg_index * blocks + i]))
						goto next_message3_1;

				unsigned int ti_index = ti_indexes[id];
				for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
					// TODO: document these cryptic operations.
					uint16_t ti_xor = query_ti_table[msg_index] ^ ti_table[ti_index + i];
					if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) {
						results[msg_index*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
					}
				}

next_message3_1: 
			}
		}
		else{
			for(int i= common_block_counter+1; i < blocks; i++)
				d[i]=fib[ id * blocks + i];

			for(uint32_t mi = 0; mi < msg_counter; ++mi) {  
				uint16_t msg_index= shortened_msg[mi] ;
				for(int32_t i = common_block_counter+1; i < blocks; ++i) 
					if(BV_BLOCK_NOT_SUBSET(d[i],packets[stream_id][msg_index * blocks + i]))
						goto next_message3_2;

				unsigned int ti_index = ti_indexes[id];
				for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
					// TODO: document these cryptic operations.
					uint16_t ti_xor = query_ti_table[msg_index] ^ ti_table[ti_index + i];
					if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) {
						results[msg_index*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
					}
				}
next_message3_2: 
			}
		}
	}
	else{
		uint32_t d[3];
		if(id >= fib_size)
			return;

		for(int i=0; i < 3; i++)
			d[i]=fib[ id * 3 + i];

		for(unsigned int pi = 0; pi < batch_size; ++pi) {
			if (BV_BLOCK_NOT_SUBSET(d[0], packets[stream_id][pi*3 + 0]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[1], packets[stream_id][pi*3 + 1]))
				continue;
			if (BV_BLOCK_NOT_SUBSET(d[2], packets[stream_id][pi*3 + 2]))
				continue;

			unsigned int ti_index = ti_indexes[id];
			for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
				uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
				if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) 
					results[pi*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
			}
		}
	}
}

__global__ void k6_old(uint32_t * fib, unsigned int fib_size, 
		uint16_t * ti_table, unsigned int * ti_indexes,  
		uint16_t * query_ti_table ,  unsigned int batch_size, 
		ifx_result_t * results,
		unsigned int stream_id) {

	uint32_t tid= (blockDim.x * threadIdx.y) + threadIdx.x;
	uint32_t id = (blockDim.x * blockDim.y * blockIdx.x) + tid; 

	if(id >= fib_size)
		return;
	uint32_t d[6];//the actual size is dynamic = noBlocks

	for(int i=0; i < 6; i++)
		d[i]=fib[ id * 6 + i];

//	if(id==0)
//		printf("%u %u %u %u %u %u \n", d[0], d[1], d[2], d[3], d[4], d[5]);

	for(unsigned int pi = 0; pi < batch_size; ++pi) {
		if (BV_BLOCK_NOT_SUBSET(d[0], packets[stream_id][pi*6 + 0]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[1], packets[stream_id][pi*6 + 1]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[2], packets[stream_id][pi*6 + 2]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[3], packets[stream_id][pi*6 + 3]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[4], packets[stream_id][pi*6 + 4]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[5], packets[stream_id][pi*6 + 5]))
			continue;

		unsigned int ti_index = ti_indexes[id];
		for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
			uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
			if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) 
				results[pi*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
		}
	}
}


__global__ void k5_old(uint32_t * fib, unsigned int fib_size, 
		uint16_t * ti_table, unsigned int * ti_indexes,  
		uint16_t * query_ti_table ,  unsigned int batch_size, 
		ifx_result_t * results,
		unsigned int stream_id) {

	uint32_t tid= (blockDim.x * threadIdx.y) + threadIdx.x;
	uint32_t id = (blockDim.x * blockDim.y * blockIdx.x) + tid; 

	if(id >= fib_size)
		return;
	uint32_t d[5];//the actual size is dynamic = noBlocks

	for(int i=0; i < 5; i++)
		d[i]=fib[ id * 5 + i];

	for(unsigned int pi = 0; pi < batch_size; ++pi) {
		if (BV_BLOCK_NOT_SUBSET(d[0], packets[stream_id][pi*5 + 0]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[1], packets[stream_id][pi*5 + 1]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[2], packets[stream_id][pi*5 + 2]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[3], packets[stream_id][pi*5 + 3]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[4], packets[stream_id][pi*5 + 4]))
			continue;

		unsigned int ti_index = ti_indexes[id];
		for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
			uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
			if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) 
				results[pi*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
		}
	}
}

__global__ void k4_old(uint32_t * fib, unsigned int fib_size, 
		uint16_t * ti_table, unsigned int * ti_indexes,  
		uint16_t * query_ti_table ,  unsigned int batch_size, 
		ifx_result_t * results,
		unsigned int stream_id) {

	uint32_t tid= (blockDim.x * threadIdx.y) + threadIdx.x;
	uint32_t id = (blockDim.x * blockDim.y * blockIdx.x) + tid; 

	if(id >= fib_size)
		return;
	uint32_t d[4];//the actual size is dynamic = noBlocks

	for(int i=0; i < 4; i++)
		d[i]=fib[ id * 4 + i];

	for(unsigned int pi = 0; pi < batch_size; ++pi) {
		if (BV_BLOCK_NOT_SUBSET(d[0], packets[stream_id][pi*4 + 0]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[1], packets[stream_id][pi*4 + 1]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[2], packets[stream_id][pi*4 + 2]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[3], packets[stream_id][pi*4 + 3]))
			continue;


		unsigned int ti_index = ti_indexes[id];
		for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
			uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
			if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) 
				results[pi*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
		}
	}
}

__global__ void k3_old(uint32_t * fib, unsigned int fib_size, 
		uint16_t * ti_table, unsigned int * ti_indexes,  
		uint16_t * query_ti_table ,  unsigned int batch_size, 
		ifx_result_t * results,
		unsigned int stream_id) {

	uint32_t tid= (blockDim.x * threadIdx.y) + threadIdx.x;
	uint32_t id = (blockDim.x * blockDim.y * blockIdx.x) + tid; 

	if(id >= fib_size)
		return;
	uint32_t d[3];//the actual size is dynamic = noBlocks

	for(int i=0; i < 3; i++)
		d[i]=fib[ id * 3 + i];

	for(unsigned int pi = 0; pi < batch_size; ++pi) {
		if (BV_BLOCK_NOT_SUBSET(d[0], packets[stream_id][pi*3 + 0]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[1], packets[stream_id][pi*3 + 1]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[2], packets[stream_id][pi*3 + 2]))
			continue;
	
		unsigned int ti_index = ti_indexes[id];
		for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
			uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
			if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) 
				results[pi*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
		}
	}
}

__global__ void k2_old(uint32_t * fib, unsigned int fib_size, 
		uint16_t * ti_table, unsigned int * ti_indexes,  
		uint16_t * query_ti_table ,  unsigned int batch_size, 
		ifx_result_t * results,
		unsigned int stream_id) {

	uint32_t tid= (blockDim.x * threadIdx.y) + threadIdx.x;
	uint32_t id = (blockDim.x * blockDim.y * blockIdx.x) + tid; 

	if(id >= fib_size)
		return;
	uint32_t d[2];//the actual size is dynamic = noBlocks

	for(int i=0; i < 2; i++)
		d[i]=fib[ id * 2 + i];

	for(unsigned int pi = 0; pi < batch_size; ++pi) {
		if (BV_BLOCK_NOT_SUBSET(d[0], packets[stream_id][pi*2 + 0]))
			continue;
		if (BV_BLOCK_NOT_SUBSET(d[1], packets[stream_id][pi*2 + 1]))
			continue;

		unsigned int ti_index = ti_indexes[id];
		for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
			uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
			if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) 
				results[pi*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
		}
	}
}

__global__ void k1_old(uint32_t * fib, unsigned int fib_size, 
		uint16_t * ti_table, unsigned int * ti_indexes,  
		uint16_t * query_ti_table ,  unsigned int batch_size, 
		ifx_result_t * results,
		unsigned int stream_id) {

	uint32_t tid= (blockDim.x * threadIdx.y) + threadIdx.x;
	uint32_t id = (blockDim.x * blockDim.y * blockIdx.x) + tid; 

	if(id >= fib_size)
		return;
	uint32_t d[1];//the actual size is dynamic = noBlocks

	for(int i=0; i < 1; i++)
		d[i]=fib[ id * 1 + i];

	for(unsigned int pi = 0; pi < batch_size; ++pi) {
		if (BV_BLOCK_NOT_SUBSET(d[0], packets[stream_id][pi*1 + 0]))
			continue;

		unsigned int ti_index = ti_indexes[id];
		for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
			uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
			if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) 
				results[pi*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
		}
	}
}


/*
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

#if 0
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
#else
	for(unsigned int pi = 0; pi < batch_size; ++pi) {
		if (a_complement_not_subset_of_b(f[0], packets[stream_id][pi*GPU_FILTER_WORDS + 0]))
			continue;
		if (a_complement_not_subset_of_b(f[1], packets[stream_id][pi*GPU_FILTER_WORDS + 1]))
			continue;
		if (a_complement_not_subset_of_b(f[2], packets[stream_id][pi*GPU_FILTER_WORDS + 2]))
			continue;
		if (a_complement_not_subset_of_b(f[3], packets[stream_id][pi*GPU_FILTER_WORDS + 3]))
			continue;
		if (a_complement_not_subset_of_b(f[4], packets[stream_id][pi*GPU_FILTER_WORDS + 4]))
			continue;
		if (a_complement_not_subset_of_b(f[5], packets[stream_id][pi*GPU_FILTER_WORDS + 5]))
			continue;

		unsigned int ti_index = ti_indexes[id];
		for(unsigned int i = ti_table[ti_index]; i > 0; --i) {
			// TODO: document these cryptic operations.
			uint16_t ti_xor = query_ti_table[pi] ^ ti_table[ti_index + i];
			if ((ti_xor < (0x0001 << 13)) && (ti_xor != 0)) {
				results[pi*INTERFACES + ((ti_table[ti_index + i]) & (0xFFFF >> 3))] = 1;
			}
		}
	}
#endif
}
*/

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
//					 unsigned char prefix_length) {
					 unsigned char blocks){

	unsigned int gridsize = fib_size/GPU_BLOCK_SIZE;
	if ((fib_size % GPU_BLOCK_SIZE) != 0)
		++gridsize;
	
//	printf("gridsize=%d, fibsize=%d batch_size=%d \n", gridsize, fib_size, batch_size) ;
//	unsigned int blocks = 6 - (prefix_length >>5) ;
	blocks = 6 - blocks ;
//	printf("=========================================%d \n", blocks); 

	switch (blocks){
		case 1: 	k1_old<<<gridsize, BLOCK_DIMS, 0, streams[stream] >>> (fib, 
																   fib_size,
																   ti_table, 
																   ti_indexes, 
																   query_ti_table,
																   batch_size,
																   results,
																   stream);
					break;
		case 2: 	k2_old<<<gridsize, BLOCK_DIMS, 0, streams[stream] >>> (fib, 
																   fib_size,
																   ti_table, 
																   ti_indexes, 
																   query_ti_table,
																   batch_size,
																   results,
																   stream);
					break;

		case 3: 	k3<<<gridsize, BLOCK_DIMS, 0, streams[stream] >>> (fib, 
																   fib_size,
																   ti_table, 
																   ti_indexes, 
																   query_ti_table,
																   batch_size,
																   results,
																   stream);
					break;

		case 4: 	k4<<<gridsize, BLOCK_DIMS, 0, streams[stream] >>> (fib, 
																   fib_size,
																   ti_table, 
																   ti_indexes, 
																   query_ti_table,
																   batch_size,
																   results,
																   stream);
					break;

		case 5: 	k5<<<gridsize, BLOCK_DIMS, 0, streams[stream] >>> (fib, 
																   fib_size,
																   ti_table, 
																   ti_indexes, 
																   query_ti_table,
																   batch_size,
																   results,
																   stream);
					break;

		case 6: 	k6<<<gridsize, BLOCK_DIMS, 0, streams[stream] >>> (fib, 
																   fib_size,
																   ti_table, 
																   ti_indexes, 
																   query_ti_table,
																   batch_size,
																   results,
																   stream);
					break;
	}

//	minimal_kernel<<<gridsize, BLOCK_DIMS, 0, streams[stream] >>> (fib, 
//																   fib_size,
//																   ti_table, 
//																   ti_indexes, 
//																   query_ti_table,
//																   batch_size,
//																   results,
//																   stream,
//																   prefix_length);
//
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
