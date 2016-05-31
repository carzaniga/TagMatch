#ifndef TWITTER_FIB_HH_INCLUDED
#define TWITTER_FIB_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>

#include "filter.hh"
#include "io_util.hh"

typedef uint32_t twitter_id_t;
class twitter_id_vector {
public:
	twitter_id_vector() : data(nullptr), length(0) {}
	~twitter_id_vector() { if (data) delete[](data); }

	twitter_id_vector(const twitter_id_vector & other) {
		resize(other.length);
		for (uint32_t i = 0; i < length; ++i)
				data[i] = other.data[i];
	}

	twitter_id_vector(twitter_id_vector && other) : data(other.data), length(other.length) {
		other.data = nullptr;
		other.length = 0;
	}

	twitter_id_vector & operator = (const twitter_id_vector & other) {
		resize(other.length);
		for (uint32_t i = 0; i < length; ++i)
			data[i] = other.data[i];
		return *this;
	}

	twitter_id_vector & operator = (twitter_id_vector && other) {
		data = other.data;
		length = other.length;
		other.data = nullptr;
		other.length = 0;
		return *this;
	}

	void resize(uint32_t new_length) {
		if (data) {
			uint32_t * new_data = new uint32_t[new_length];
			for (uint32_t i = 0; i < new_length && i < length; ++i)
				new_data[i] = data[i];
			delete[](data);
			data = new_data;
		} else {
			data = new uint32_t[new_length];
		}
		length = new_length;
	}

	uint32_t * begin() { return data; }
	uint32_t * end() { return data + length; }
	const uint32_t * begin() const { return data; }
	const uint32_t * end() const { return data + length; }

	typedef uint32_t * iterator;
	typedef const uint32_t * const_iterator;

	uint32_t size() const { return length; }

private:
	uint32_t * data;
	uint32_t length;
};

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

class twitter_packet {
public:
	filter_t filter;
	twitter_id_t id;

	twitter_packet() : filter(), id() {}

	std::ostream & write_binary(std::ostream & output) const;
	std::istream & read_binary(std::istream & input);
	std::ostream & write_ascii(std::ostream & output) const;
	std::istream & read_ascii(std::istream & input);
};

#endif // include guard
