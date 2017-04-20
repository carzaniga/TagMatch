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
	static void start(unsigned int threads);
	static void match(tagmatch_query * p) noexcept;
	static void stop();
	static void clear();
	static std::ostream & print_statistics(std::ostream &);
	static unsigned int get_latency_limit_ms();
	static void set_latency_limit_ms(unsigned int);
	static void set_bit_permutation_pos(unsigned char, unsigned char);
	static void set_identity_permutation() noexcept;
};

#endif // FRONT_END_HH_INCLUDED
