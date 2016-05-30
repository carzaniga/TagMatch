#ifndef TWITTER_FIB_HH_INCLUDED
#define TWITTER_FIB_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <vector>

#include "io_util.hh"
#include "packet.hh"

typedef uint32_t twitter_id_t;
typedef std::vector<twitter_id_t> twitter_id_vector;

class twitter_fib_entry {
public:
	filter_t filter;
	twitter_id_vector ids;	

	twitter_fib_entry() : filter(), ids() {}

	std::ostream & write_binary(std::ostream & output) const;
	std::istream & read_binary(std::istream & input);
	std::ostream & write_ascii(std::ostream & output) const;
	std::istream & read_ascii(std::istream & input);
};

#endif // include guard
