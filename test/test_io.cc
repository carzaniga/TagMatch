// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2014 Antonio Carzaniga
//
//  Siena is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  Siena is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with Siena.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>
#include <sstream>
#include <cstring>

#include "bitvector.hh"

typedef bitvector<192> filter_t;

#define BOOST_TEST_MODULE io
#define BOOST_TEST_DYN_LINK 1

#include <boost/test/unit_test.hpp>

bool check_binary_encoding(const char * input, const char * expected, size_t size) {
	filter_t f;
	std::istringstream input_string_stream(input);
	std::ostringstream output;

	f.read_ascii(input_string_stream);
	f.write_binary(output);

	return (memcmp(expected, output.str().data(), size) == 0);
}

BOOST_AUTO_TEST_SUITE( bitvector_io )

BOOST_AUTO_TEST_CASE( encode_binary_one ) {
	BOOST_CHECK(check_binary_encoding("100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "\1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", 24));
}

BOOST_AUTO_TEST_CASE( encode_binary_zero ) {
	BOOST_CHECK(check_binary_encoding("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", 24));
}

BOOST_AUTO_TEST_CASE( encode_binary_rightmost ) {
	BOOST_CHECK(check_binary_encoding("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001", "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\200", 24));
}

BOOST_AUTO_TEST_CASE( encode_binary_two ) {
	BOOST_CHECK(check_binary_encoding("010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "\2\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", 24));
}

BOOST_AUTO_TEST_CASE( encode_binary_two_byte_two ) {
	BOOST_CHECK(check_binary_encoding("000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "\0\2\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", 24));
}

BOOST_AUTO_TEST_SUITE_END()

