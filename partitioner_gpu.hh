#ifndef PARTITIONER_GPU_HH_INCLUDED
#define PARTITIONER_GPU_HH_INCLUDED

#include <vector>

#include "fib.hh"

#define GPU_THREADS 256
#define MAXTHREADS 24
#define CPU_MAX_PSIZE 128
#define GPU_WORDS_PER_FILTER 6
#define CPU_WORDS_PER_FILTER 3
#define GPU_BITS_PER_WORD 32
#define CPU_BITS_PER_WORD 64
#define FILTER_WIDTH 192

class partitioner_gpu {
	private:
		static void fibToArray(std::vector<fib_entry *> * fib, uint32_t size);

	public:
		static void init(unsigned int part_thread_count, std::vector<fib_entry *> * fib);
		static void get_frequencies(unsigned int tid, unsigned int size, unsigned int first, unsigned int * freq, size_t buffer_size);
		static void reset_buffers(unsigned int tid);
		static void clear(unsigned int part_thread_count);
		static void unstable_partition(unsigned int tid, unsigned int size, unsigned int first, unsigned int pivot);
};

#endif
