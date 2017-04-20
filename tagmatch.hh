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
	static void add(const filter_t & set, const std::vector<tagmatch_key_t> & keys);
	static void add(const filter_t & set, tagmatch_key_t key);
	static void remove(const filter_t & set, tagmatch_key_t key);

	static void consolidate();
	static void consolidate(unsigned int partition_size, unsigned int threads);

	static const char * get_database_filename();
	static void set_database_filename(const char *);

	static void start();
	static void start(unsigned int threads, unsigned int gpu_count);
	static void stop();
	static void clear();

	static unsigned int get_latency_limit_ms();
	static void set_latency_limit_ms(unsigned int latency_limit);

	// Match operations
	//
	static void match(tagmatch_query * q, match_handler * h) noexcept;
	static void match_unique(tagmatch_query * q, match_handler * h) noexcept;
};

#endif
