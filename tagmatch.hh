#ifndef TAGMATCH_HH_INCLUDED
#define TAGMATCH_HH_INCLUDED

#include <vector>

#include "filter.hh"
#include "key.hh"
#include "query.hh"
#include "tagmatch_query.hh"

class tagmatch {
public:
	// These methods are used by the partitioner
	//
	static void add_set(filter_t set, const std::vector<tagmatch_key_t> & keys);
	static void add_set(filter_t set, tagmatch_key_t key);
	static void delete_set(filter_t set, tagmatch_key_t key);
	static void consolidate();
	static void consolidate(uint32_t psize, uint32_t threads);

	// These methods are used by the matcher
	//
	static void add_partition(unsigned int id, const filter_t & mask);
	static void add_filter(unsigned int partition_id, const filter_t & f,
						   std::vector<tagmatch_key_t>::const_iterator begin,
						   std::vector<tagmatch_key_t>::const_iterator end);
	static void start();
	static void start(unsigned int threads, unsigned int gpu_count);
	static void stop();
	static void clear();

	static unsigned int get_latency_limit_ms();
	static void set_latency_limit_ms(unsigned int latency_limit);
	static void set_cpu_count(int count);
	static void set_gpu_count(int count);

	// Match operations
	//
	static void match(tagmatch_query * q, match_handler * h) noexcept;
	static void match_unique(tagmatch_query * q, match_handler * h) noexcept;
};

#endif
