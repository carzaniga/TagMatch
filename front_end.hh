#ifndef FRONT_END_HH_INCLUDED
#define FRONT_END_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>

#include "packet.hh"
#include "gpu.hh"

class front_end {
public:
	static void add_prefix(unsigned int id, const filter_t & f);
	static void start(unsigned int threads);
	static void match(packet * p) noexcept;
	static void stop(unsigned int gpu_count);
	static void clear();
	static std::ostream & print_statistics(std::ostream &);
	static unsigned int get_latency_limit_ms();
	static void set_latency_limit_ms(unsigned int);
	static void set_bit_permutation_pos(unsigned char, unsigned char);
	static void set_identity_permutation() noexcept;
};

#endif // FRONT_END_HH_INCLUDED
