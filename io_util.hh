#ifndef IO_UTIL_HH_INCLUDED
#define IO_UTIL_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>

template <typename IntType> 
std::ostream & io_util_write_binary(std::ostream & output, IntType value) {
#ifdef WORDS_BIGENDIAN
	unsigned char tmp[sizeof(value)];
	unsigned char * cp;
	for(int i = 0; i < sizeof(IntType); ++i) {
		*cp = value & 0xff;
		value >>= CHAR_BIT;
		++cp;
	}
	return output.write(tmp, sizeof(tmp));
#else
	return output.write(reinterpret_cast<const char *>(&value), sizeof(value));
#endif
}

template <typename IntType> 
std::istream & io_util_read_binary(std::istream & input, IntType & value) {
#ifdef WORDS_BIGENDIAN
	unsigned char tmp[sizeof(value)];
	if (input.read(tmp, sizeof(tmp))) {
		const unsigned char * cp = tmp;
		value = 0;
		for(int j = 0; j < sizeof(value); ++j) {
			value = (value << CHAR_BIT) | *cp;
			++cp;
		}
	}
	return input;
#else
	return input.read(reinterpret_cast<char *>(&value), sizeof(value));
#endif
}

#endif // FRONT_END_HH_INCLUDED
