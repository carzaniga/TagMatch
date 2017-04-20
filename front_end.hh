#ifndef FRONT_END_HH_INCLUDED
#define FRONT_END_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>

#include "tagmatch_query.hh"

class front_end {
public:
	static void add_partition(unsigned int id, const filter_t & mask);
	static void consolidate();
	static void start(unsigned int threads);
	static void match(tagmatch_query * p) noexcept;
	static void stop();
	static void clear();
	static std::ostream & print_statistics(std::ostream &);
	static unsigned int get_latency_limit_ms();
	static void set_latency_limit_ms(unsigned int);
};

#endif // FRONT_END_HH_INCLUDED
