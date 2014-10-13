#ifndef FRONT_END_HH_INCLUDED
#define FRONT_END_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>

#include "packet.hh"

/// add a prefix f of length n to the front end FIB
/// 
class front_end {
public:
	static void add_prefix(unsigned int id, const prefix<192> & f, unsigned int n);
	static void start(unsigned int n);
	static void match(packet * p) noexcept;
	static void stop();
	static void clear();
	static std::ostream & print_statistics(std::ostream &);
	static unsigned int get_latency_limit_ms();
	static void set_latency_limit_ms(unsigned int);
	static void set_bit_permutation_pos(unsigned char, unsigned char);
	static void set_identity_permutation() noexcept;
};

#endif // FRONT_END_HH_INCLUDED
