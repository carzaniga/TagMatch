#ifndef TAGMATCH_HH_INCLUDED
#define TAGMATCH_HH_INCLUDED

#include <condition_variable>
#include "packet.hh"

#include "partitioner.hh"
#include "front_end.hh"
#include "back_end.hh"
#include "match_handler.hh"

class tagmatch {
public:
		// These methods are used by the partitioner
		//
		static void add_set(filter_t set, tk_vector keys);
		static void add_set(filter_t set, tagmatch_key_t key);
		static void delete_set(filter_t set, tagmatch_key_t key);
		static void consolidate(); 
		static void consolidate(uint32_t psize, uint32_t threads);
		// TODO: shall consolidate also pass the data to the matcher???
		// TODO TODO: Or maybe we want to have another method to do that specifically...

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

		// Match handlers, to be used by the tagmatch client
		static void match(match_handler * h) noexcept {
			h->match_unique = false;
			front_end::match(h);
			h->match_hold();
		}
		static void match_unique(match_handler * h) noexcept {
			h->match_unique = true;
			front_end::match(h);
			h->match_hold();
		}
};

#endif
