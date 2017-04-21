#ifndef BACK_END_HH_INCLUDED
#define BACK_END_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <vector>

#include "filter.hh"
#include "key.hh"
#include "tagmatch_query.hh"
#include "fib.hh"

union batch;

/// add a prefix f of length n to the front end FIB
/// 
class back_end {
public:
	static void add_partition(partition_id_t id, const filter_t & mask);
	static void add_filter(partition_id_t partition, const filter_t & f,
						   std::vector<tagmatch_key_t>::const_iterator begin,
						   std::vector<tagmatch_key_t>::const_iterator end);
	static void start(unsigned int gpu_count);
	static batch * process_batch(partition_id_t part,
								tagmatch_query ** queries, unsigned int batch_size,
								batch * batch_ptr);
	static batch * second_flush_stream();
	static batch * flush_stream();
	static void release_stream_handles();
	static void stop();
	static void clear();

	static unsigned int gpu_count();
	static size_t bytesize();

	static void analyze_fibs();
};

#endif // BACK_END_HH_INCLUDED
