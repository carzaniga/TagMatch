#ifndef PARTITIONER_GPU_HH_INCLUDED
#define PARTITIONER_GPU_HH_INCLUDED

#include <cstddef>
#include <vector>

#include "fib.hh"

class partitioner_gpu {
private:
	static void fibToArray(const std::vector<partition_fib_entry *> & fib, uint32_t size);

public:
	static const unsigned int MAXTHREADS = 24;
	static const unsigned int CPU_MAX_PSIZE = 128;

	static void init(unsigned int part_thread_count, const std::vector<partition_fib_entry *> & fib);
	static void get_frequencies(unsigned int tid, unsigned int size, unsigned int first, unsigned int * freq, size_t buffer_size);
	static void reset_buffers(unsigned int tid);
	static void clear(unsigned int part_thread_count);
	static void unstable_partition(unsigned int tid, unsigned int size, unsigned int first, unsigned int pivot);

};

#endif
