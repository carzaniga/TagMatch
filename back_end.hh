#ifndef BACK_END_HH_INCLUDED
#define BACK_END_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <vector>

#include "tagmatch.hh"
#include "match_handler.hh"
#include "packet.hh"

/// add a prefix f of length n to the front end FIB
/// 
class back_end {
public:
	static void add_partition(unsigned int id, const filter_t & prefix, unsigned int prefix_length);
	static void add_filter(unsigned int partition, const filter_t & f, 
						   std::vector<tagmatch_key_t>::const_iterator begin,
						   std::vector<tagmatch_key_t>::const_iterator end);
	static void start(unsigned int gpu_count);
	static void * process_batch(unsigned int part, match_handler ** batch, unsigned int batch_size, void *batch_ptr);
	static void * second_flush_stream();
	static void * flush_stream();
	static void release_stream_handles(unsigned int gpu_count);
	static void stop(unsigned int gpu_count);
	static void clear(unsigned int gpu_count);
	static filter_t get_cbits(unsigned int id);

	static size_t bytesize();

	static void analyze_fibs();
};

#endif // BACK_END_HH_INCLUDED
