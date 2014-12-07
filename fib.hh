#ifndef FIB_HH_INCLUDED
#define FIB_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <vector>

#include "io_util.hh"
#include "packet.hh"

class ti_vector : public std::vector<tree_interface_pair> {
public:
	std::ostream & write_binary(std::ostream & output) const;
	std::istream & read_binary(std::istream & input);
};

struct fib_entry {
public:
	filter_t filter;
	ti_vector ti_pairs;

	std::ostream & write_binary(std::ostream & output) const {
		filter.write_binary(output);
		ti_pairs.write_binary(output);
		return output;
	}

	std::istream & read_binary(std::istream & input) {
		if (filter.read_binary(input))
			ti_pairs.read_binary(input);
		return input;
	}

	std::ostream & write_ascii(std::ostream & output) const;
	std::istream & read_ascii(std::istream & input);
};

typedef uint32_t partition_id_t;

struct partition_fib_entry : public fib_entry {
public:
	partition_id_t partition_id;

	std::ostream & write_binary(std::ostream & output) const {
		io_util_write_binary(output, partition_id);
		return fib_entry::write_binary(output);
	}

	std::istream & read_binary(std::istream & input) {
		if (!io_util_read_binary(input, partition_id))
			return input;
		return fib_entry::read_binary(input);
	}
};

#endif // BACK_END_HH_INCLUDED
