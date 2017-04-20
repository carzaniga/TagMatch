#ifndef FIB_HH_INCLUDED
#define FIB_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <vector>

#include "key.hh"
#include "filter.hh"
#include "io_util.hh"

class fib_entry {
public:
	filter_t filter;
	std::vector<tagmatch_key_t> keys;

	fib_entry() : filter(), keys() {}
	fib_entry(const filter_t & f, const std::vector<tagmatch_key_t> & k) : filter(f), keys(k) {}
	fib_entry(const fib_entry & fe) : filter(fe.filter), keys(fe.keys) {}

	std::ostream & write_binary(std::ostream & output) const;
	std::istream & read_binary(std::istream & input);

	std::ostream & write_ascii(std::ostream & output) const;
	std::istream & read_ascii(std::istream & input);
};

typedef uint32_t partition_id_t;
const partition_id_t NULL_PARTITION_ID = 0xffffffff;

class partition_fib_entry : public fib_entry {
public:
	partition_id_t partition;

	partition_fib_entry() : fib_entry(), partition(0) {}
	partition_fib_entry(const fib_entry & fe) : fib_entry(fe), partition(0) {}

	std::ostream & write_binary(std::ostream & output) const {
		io_util_write_binary(output, partition);
		return fib_entry::write_binary(output);
	}

	std::istream & read_binary(std::istream & input) {
		if (!io_util_read_binary(input, partition))
			return input;
		return fib_entry::read_binary(input);
	}

	std::ostream & write_ascii(std::ostream & output) const;
	std::istream & read_ascii(std::istream & input);
};

class partition_prefix {
public:
	filter_t filter;
	uint8_t length;
	partition_id_t partition;
	uint32_t size;

	std::ostream & write_binary(std::ostream & output) const {
		filter.write_binary(output);
		io_util_write_binary(output, length);
		io_util_write_binary(output, partition);
		io_util_write_binary(output, size);
		return output;
	}

	std::istream & read_binary(std::istream & input) {
		if (filter.read_binary(input))
			if (io_util_read_binary(input, length))
				if (io_util_read_binary(input, partition))
					io_util_read_binary(input, size);
		return input;
	}

	std::ostream & write_ascii(std::ostream & output) const;
	std::istream & read_ascii(std::istream & input);
};

#endif // BACK_END_HH_INCLUDED
