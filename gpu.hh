#ifndef GPU_HH_INCLUDED
#define GPU_HH_INCLUDED

#include <stdint.h>
#include <stdlib.h>
#include "parameters.hh"

#define GPU_MSG_BLOCKS 6
#define GPU_STREAMS 10
#define GPU_NUM 10

#if GPU_STREAMS > 254
#error "too many GPU streams."
#endif

// we use 32-bit words on the GPU
#define GPU_WORD_SIZE 4

// 192 bits / 32-bit words
#define GPU_FILTER_WORDS 6

struct gpu_result {
	uint32_t count;
	uint32_t keys[MAX_MATCHES+MAX_MATCHES/4];
};

struct gpu_mem_info {
	size_t free;
	size_t total;
};

class gpu {
public:
	static void initialize(unsigned int gpu_count);
	static void mem_info(gpu_mem_info *, unsigned int gpu_count);
	static void set_device(unsigned int dev);

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

	template<typename T>
	static T * allocate_host_pinned(unsigned int size) {
		return (T *)allocate_host_pinned_generic(size*sizeof(T));
	}
	static void * allocate_host_pinned_generic(unsigned int size);

	static void release_memory(void * p);
	static void release_pinned_memory(void * p);

	static void async_copy_queries(uint32_t * queries, unsigned int size , unsigned int stream, unsigned int gpu);
	static void async_copy(void * hst_src, void * dev_dst, unsigned int size, unsigned int stream, unsigned int gpu);
	static void async_set_zero(void * dev_array, unsigned int size, unsigned int stream_id, unsigned int gpu);
	static void async_get_results(gpu_result * host_results, gpu_result * dev_results,  unsigned int size, unsigned int stream, unsigned int gpu);
	static void syncOnResults(unsigned int stream, unsigned int gpu);
	static void get_results(gpu_result * host_results, gpu_result * dev_results,
							unsigned int size);

	static void run_kernel(uint32_t * fib, unsigned int fib_size,
						   unsigned int batch_size,
						   gpu_result * dev_results_count,
						   gpu_result * dev_results_data,
						   unsigned int stream,
						   unsigned int gpu);

	static void synchronize_stream(unsigned int stream, unsigned int gpu);
	static void synchronize_device();
	static void shutdown(unsigned int gpu_count);
};

#endif // GPU_HH_INCLUDED
