#ifndef GPU_HH_INCLUDED
#define GPU_HH_INCLUDED

#include <stdint.h>

#define GPU_BLOCK_DIM_X 32
#define GPU_BLOCK_DIM_Y 32
#define GPU_BLOCK_SIZE GPU_BLOCK_DIM_X * GPU_BLOCK_DIM_Y // Statc block size of 32*32 (1024)0
#define GPU_STREAMS 5 

// we use 32-bit words on the GPU
#define GPU_WORD_SIZE 4

// 192 bits / 32-bit words
#define GPU_FILTER_WORDS 6

typedef unsigned char ifx_result_t;

class gpu {
public:
	static void initialize();
	static void mem_info();

	template<typename T>
	static T * allocate(unsigned int size) {
		return (T *)allocate_generic(size*sizeof(T));
	}
	static void * allocate_generic(unsigned int size);

	template<typename T>
	static T * allocate_and_copy(T * host_table, unsigned int size) {
		return (T *)allocate_and_copy_generic(host_table, size*sizeof(T));
	}
	static void * allocate_and_copy_generic(void * host_table, unsigned int size);

#if WITH_PINNED_HOST_MEMORY
	template<typename T>
	static T * allocate_host_pinned(unsigned int size) {
		return (T *)allocate_host_pinned_generic(size*sizeof(T));
	}
	static void * allocate_host_pinned_generic(unsigned int size);
#endif

	static void release_memory(void * p);

	static void async_copy_packets(uint32_t * pkts, unsigned int size , unsigned int stream);
	static void async_copy(void * host_src, void * dev_dst, unsigned int size, unsigned int stream);
	static void async_set_zero(void * dev_array, unsigned int size, unsigned int stream_id);
	static void async_get_results(ifx_result_t * host_results, ifx_result_t * dev_results, 
								  unsigned int size, unsigned int stream);

	static void run_kernel(uint32_t * fib, unsigned int fib_size, 
						   uint16_t * dev_ti_table, unsigned int * ti_table_indexes, 
						   uint16_t * dev_query_ti_table, unsigned int batch_size, 
						   ifx_result_t * dev_results,
						   unsigned int stream);

	static void synchronize_stream(unsigned int stream);
	static void shutdown();
};


#endif // GPU_HH_INCLUDED