#ifndef PARTITIONER_HH_INCLUDED
#define PARTITIONER_HH_INCLUDED

#include <vector>

#include "tagmatch.hh"
#include "fib.hh"

class partitioner {
private:
	static void initialize();

public:
	static void clear();
	static void add_set(filter_t set, const std::vector<tagmatch_key_t> & keys);
	static void consolidate();
	static void consolidate(unsigned int size, unsigned int thread_count);
	static void get_consolidated_prefixes_and_filters(std::vector<partition_prefix> ** prefixes,
													  std::vector<partition_fib_entry> ** filter);
};

#endif
