#ifndef TAGMATCH_HH_INCLUDED
#define TAGMATCH_HH_INCLUDED

#include <vector>
#include <mutex>
#include <condition_variable>

#include "bitvector.hh"

typedef bitvector<192> filter_t;

typedef uint32_t tagmatch_key_t;

class packet;
class match_handler;
class synchronous_match_handler;;

#include "match_handler.hh"

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
	static void match(packet * query, match_handler * h) noexcept;
	static void match_unique(packet * query, match_handler * h) noexcept;
};

#endif
