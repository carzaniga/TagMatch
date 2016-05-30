#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <cassert>

#include "io_util.hh"
#include "twitter_fib.hh"

static std::ostream & tiv_write_binary(std::ostream & output, const twitter_id_vector & ids)  {
	uint32_t vector_size = ids.size();
	io_util_write_binary(output, vector_size);
	for(twitter_id_vector::const_iterator i = ids.begin(); i != ids.end(); ++i)
		io_util_write_binary(output, *i);
	return output;
}

static std::istream & tiv_read_binary(std::istream & input, twitter_id_vector & ids) {
	uint32_t vector_size;
	if (!io_util_read_binary(input, vector_size))
		return input;

	ids.resize(vector_size);
	for(twitter_id_vector::iterator i = ids.begin(); i != ids.end(); ++i)
		if (!io_util_read_binary(input, *i))
			return input;

	return input;
}

std::ostream & twitter_fib_entry::write_binary(std::ostream & output) const {
	filter.write_binary(output);
	return tiv_write_binary(output, ids);
}

std::istream & twitter_fib_entry::read_binary(std::istream & input) {
	if (filter.read_binary(input))
		return tiv_read_binary(input, ids);
	return input;
}

std::ostream & twitter_fib_entry::write_ascii(std::ostream & output) const {
	assert(false);
	return output;
}

std::istream & twitter_fib_entry::read_ascii(std::istream & input) {
	assert(false);
	return input;
}

std::ostream & twitter_packet::write_binary(std::ostream & output) const {
	filter.write_binary(output);
	return io_util_write_binary(output, id);
}

std::istream & twitter_packet::read_binary(std::istream & input) {
	if (filter.read_binary(input))
		return io_util_read_binary(input, id);
	return input;
}

std::ostream & twitter_packet::write_ascii(std::ostream & output) const {
	assert(false);
	return output;
}

std::istream & twitter_packet::read_ascii(std::istream & input) {
	assert(false);
	return input;
}

