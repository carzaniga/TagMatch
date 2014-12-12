// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2013 Antonio Carzaniga
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

#define BOOST_TEST_MODULE bitvector
#define BOOST_TEST_DYN_LINK 1

#include <boost/test/unit_test.hpp>

#include "bitvector.hh"

typedef bitvector<192> filter_t;

static bool subset_of(const char * x, const char * y) {
	return filter_t(x).subset_of(filter_t(y));
}

static bool prefix_less_than(const char * x, const char * y) {
	return filter_t(x) < filter_t(y);
}

static bool equals(const char * x, const char * y) {
	return filter_t(x) == filter_t(y);
}

BOOST_AUTO_TEST_SUITE( bitvector_assignment )

BOOST_AUTO_TEST_CASE( construct_and_clear ) {
	filter_t x;
	x.clear();
	for(unsigned int i = 0; i < filter_t::WIDTH; ++i)
		BOOST_CHECK(!x[i]);
}

BOOST_AUTO_TEST_CASE( clear ) {
	filter_t x;
	filter_t y("001010000110101011000001000100000000011100001110010001101001001111101000000000001110011010010100100010000001011000000100100000001101000110011001001000000000001000111010000000000110000010000110");

	x.clear();
	BOOST_CHECK(!(x == y));
	y.clear();
	BOOST_CHECK(x == y);
}

BOOST_AUTO_TEST_CASE( direct_assignment ) {
	filter_t x;
	filter_t y("001010000110101011000001000100000000011100001110010001101001001111101000000000001110011010010100100010000001011000000100100000001101000110011001001000000000001000111010000000000110000010000110");

	x.clear();
	BOOST_CHECK(!(x == y));
	x = y;
	BOOST_CHECK(x == y);
}

BOOST_AUTO_TEST_CASE( bit_set_left_block ) {
	filter_t x;
	filter_t y("001000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

	x.clear();
	BOOST_CHECK(!(x == y));
	x.set_bit(2);
	x.set_bit(10);
	BOOST_CHECK(x == y);
}

BOOST_AUTO_TEST_CASE( bit_set_middle_block ) {
	filter_t x;
	filter_t y("000000000000000000000000000000000000000000000000000000000000000000100000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

	x.clear();
	BOOST_CHECK(!(x == y));
	x.set_bit(66);
	x.set_bit(74);
	BOOST_CHECK(x == y);
}

BOOST_AUTO_TEST_CASE( bit_set_right_block ) {
	filter_t x;
	filter_t y("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000100000000000000000000000000000000000000000000000000000");

	x.clear();
	BOOST_CHECK(!(x == y));
	x.set_bit(130);
	x.set_bit(138);
	BOOST_CHECK(x == y);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( bitwise_operations )

BOOST_AUTO_TEST_CASE( bitwise_or ) {
	filter_t x("001000000000000000000000000000110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000110");
	filter_t y("000000000000000000000000000001100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000100000000000000000000000000000000000000000000000000011");

	filter_t z("001000000000000000000000000001110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000100000000000000000000000000000000000000000000000000111");

	x |= y;
	BOOST_CHECK(x == z);
}

BOOST_AUTO_TEST_CASE( bitwise_and ) {
	filter_t x("001000000000000000000000000000110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000110");
	filter_t y("000000000000000000000000000001100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000100000000000000000000000000000000000000000000000000011");

	filter_t z("000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000010");

	x &= y;
	BOOST_CHECK(x == z);
}

BOOST_AUTO_TEST_CASE( bitwise_xor ) {
	filter_t x("001000000000000000000000000000110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000110");
	filter_t y("000000000000000000000000000001100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000100000000000000000000000000000000000000000000000000011");

	filter_t z("001000000000000000000000000001010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000101");

	x ^= y;
	BOOST_CHECK(x == z);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( bit_iteration )

BOOST_AUTO_TEST_CASE( iteration ) {
	filter_t x("001000000000000000000000000001010000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000101");

	BOOST_CHECK_EQUAL(x.next_bit(0),2);
	BOOST_CHECK_EQUAL(x.next_bit(1),2);
	BOOST_CHECK_EQUAL(x.next_bit(1),2);
	BOOST_CHECK_EQUAL(x.next_bit(3),29);
	BOOST_CHECK_EQUAL(x.next_bit(30),31);
	BOOST_CHECK_EQUAL(x.next_bit(31),31);
	BOOST_CHECK_EQUAL(x.next_bit(32),70);
	BOOST_CHECK_EQUAL(x.next_bit(70),70);
	BOOST_CHECK_EQUAL(x.next_bit(71),138);
	BOOST_CHECK_EQUAL(x.next_bit(138),138);
	BOOST_CHECK_EQUAL(x.next_bit(139),189);
	BOOST_CHECK_EQUAL(x.next_bit(189),189);
	BOOST_CHECK_EQUAL(x.next_bit(190),191);
	BOOST_CHECK_EQUAL(x.next_bit(192),192);
}

BOOST_AUTO_TEST_CASE( iteration_on_empty ) {
	filter_t x("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

	BOOST_CHECK_EQUAL(x.next_bit(0),192);
	BOOST_CHECK_EQUAL(x.next_bit(63),192);
	BOOST_CHECK_EQUAL(x.next_bit(64),192);
	BOOST_CHECK_EQUAL(x.next_bit(127),192);
	BOOST_CHECK_EQUAL(x.next_bit(128),192);
	BOOST_CHECK_EQUAL(x.next_bit(130),192);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( popcount )

BOOST_AUTO_TEST_CASE( empty ) {
	filter_t x("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

	BOOST_CHECK_EQUAL(x.popcount(), 0);
}

BOOST_AUTO_TEST_CASE( right_block ) {
	filter_t x("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001011000001000");

	BOOST_CHECK_EQUAL(x.popcount(), 4);
}

BOOST_AUTO_TEST_CASE( middle_block ) {
	filter_t x("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000101100000100000000000000000000000000000000000000000000000000000000000000000000");

	BOOST_CHECK_EQUAL(x.popcount(), 4);
}

BOOST_AUTO_TEST_CASE( left_block ) {
	filter_t x("000000000000000000000000000000000000000000000000001011000001001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

	BOOST_CHECK_EQUAL(x.popcount(), 5);
}

BOOST_AUTO_TEST_CASE( all_blocks ) {
	filter_t x("001100100011001100111000000100100000000010110101000000100000000000000100000000000000100100100000000100100100000000000010000000011000000000001010000000000000000000000000000000110000000000000000");

	BOOST_CHECK_EQUAL(x.popcount(), 32);
}

BOOST_AUTO_TEST_CASE( all_bits ) {
	filter_t x("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111");

	BOOST_CHECK_EQUAL(x.popcount(), 192);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( bitvector_equality )

BOOST_AUTO_TEST_CASE( random_diff ) { 
	BOOST_CHECK(equals("000100100010000000111000000000100000000000110000000000100000000000000000000000000000000000000000000000000100000000000000000000000000000000001010000000000000000000000000000000010000000000000000", 
					   "001100100011001100111000000100100000000010110101000000100000000000000100000000000000100100100000000100100100000000000010000000011000000000001010000000000000000000000000000000110000000000000000") == false); 
}
BOOST_AUTO_TEST_CASE( random_equals ) {
	BOOST_CHECK(equals("001100100011001100111000000100100000000010110101000000100000000000000100000000000000100100100000000100100100000000000010000000011000000000001010000000000000000000000000000000110000000000000000", 
					   "001100100011001100111000000100100000000010110101000000100000000000000100000000000000100100100000000100100100000000000010000000011000000000001010000000000000000000000000000000110000000000000000") == true); 
}
BOOST_AUTO_TEST_CASE( all_zero_equals ) {
	BOOST_CHECK(equals("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 
					   "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000") == true);
}
BOOST_AUTO_TEST_CASE( all_one_equals ) {
	BOOST_CHECK(equals("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", 
					   "111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111") == true);
}
BOOST_AUTO_TEST_CASE( all_one_all_zero ) {
	BOOST_CHECK(equals("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", 
					   "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000") == false);
}
BOOST_AUTO_TEST_CASE( alternate_all_diff ) {
	BOOST_CHECK(equals("101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010", 
					   "010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101") == false);
}

BOOST_AUTO_TEST_CASE( diff_0 ) {
	BOOST_CHECK(equals("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 
					   "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010") == false);
}

BOOST_AUTO_TEST_CASE( diff_0_1 ) {
	BOOST_CHECK(equals("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 
					   "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010") == false);
}

BOOST_AUTO_TEST_CASE( diff_0_1_2 ) {
	BOOST_CHECK(equals("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 
					   "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010") == false);
}

BOOST_AUTO_TEST_CASE( diff_1 ) {
	BOOST_CHECK(equals("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 
					   "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000") == false);
}

BOOST_AUTO_TEST_CASE( diff_1_2 ) {
	BOOST_CHECK(equals("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 
					   "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000") == false);
}

BOOST_AUTO_TEST_CASE( diff_2 ) {
	BOOST_CHECK(equals("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 
					   "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000") == false);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( bitvector_subset )

BOOST_AUTO_TEST_CASE( test_case1 ) { 
	BOOST_CHECK(subset_of("000100100010000000111000000000100000000000110000000000100000000000000000000000000000000000000000000000000100000000000000000000000000000000001010000000000000000000000000000000010000000000000000", 
						  "001100100011001100111000000100100000000010110101000000100000000000000100000000000000100100100000000100100100000000000010000000011000000000001010000000000000000000000000000000110000000000000000") == true); 
}
BOOST_AUTO_TEST_CASE( test_case2 ) {
	BOOST_CHECK(subset_of("001000000001001100000000000100000000000000000000000000000000000000000000000000000000100100100000000100100000000000000010000000001000000000000000000000000000000000000000000000100000000000000000", 
						  "000000001000000010000001101010000100000000001001000010000101001100010000000110100001101000000000000000010000100000000000000000000100000010011000100000000001001000001000000000000000001000010010") == false);
}
BOOST_AUTO_TEST_CASE( test_case3 ) {
	BOOST_CHECK(subset_of("000000000000001000000001001000000000000000000101000010100100000000000100000010000000000000000000000000000000000000000000000000010000000000000000000000000001001000000000000000000000000000010000", 
						  "001010000110101011000001000100000000011100001110010001101001001111101000000000001110011010010100100010000001011000000100100000001101000110011001001000000000001000111010000000000110000010000110") == false);
}
BOOST_AUTO_TEST_CASE( test_case4 ) {
	BOOST_CHECK(subset_of("001010000010101011000001000100000000011100001110010001101001001110101000000000001110011010010100100010000001011000000100100000001101000110011001001000000000001000111010000000000110000010000110", 
						  "001010000110101011000001000100000000011100001110010001101001001111101000000000001110011010000100100010000001011000000100100000001101000110011001001000000000001000111010000000000110000010000110") == false);
}
BOOST_AUTO_TEST_CASE( test_case5 ) {
	BOOST_CHECK(subset_of("100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 
						  "100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000") == true);
}
BOOST_AUTO_TEST_CASE( test_case6 ) {
	BOOST_CHECK(subset_of("100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 
						  "100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001") == true);
}
BOOST_AUTO_TEST_CASE( test_case7 ) {
	BOOST_CHECK(subset_of("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001", 
						  "100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001") == true);
}
BOOST_AUTO_TEST_CASE( test_case8 ) {
	BOOST_CHECK(subset_of("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000", 
						  "100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000") == true);
}
BOOST_AUTO_TEST_CASE( test_case9 ) {
	BOOST_CHECK(subset_of("000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 
						  "100000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000") == true);
}
BOOST_AUTO_TEST_CASE( test_case10 ) {
	BOOST_CHECK(subset_of("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001", 
						  "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001") == true);
}
BOOST_AUTO_TEST_CASE( test_case11 ) {
	BOOST_CHECK(subset_of("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001", 
						  "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001") == true);
}
BOOST_AUTO_TEST_CASE( test_case12 ) {
	BOOST_CHECK(subset_of("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000", 
						  "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000") == true);
}
BOOST_AUTO_TEST_CASE( test_case13 ) {
	BOOST_CHECK(subset_of("000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 
						  "000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000") == true);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( bitvector_less_than )

BOOST_AUTO_TEST_CASE( test_case1 ) { 
	BOOST_CHECK(prefix_less_than("000100100010000000111000000000100000000000110000000000100000000000000000000000000000000000000000000000000100000000000000000000000000000000001010000000000000000000000000000000010000000000000000", 
								 "001100100011001100111000000100100000000010110101000000100000000000000100000000000000100100100000000100100100000000000010000000011000000000001010000000000000000000000000000000110000000000000000") == true); 
}
BOOST_AUTO_TEST_CASE( test_case2 ) { 
	BOOST_CHECK(prefix_less_than("000100100010000000111000000000100000000000110000000000100000000000000000000000000000000000000000000000000100000000000000000000000000000000001010000000000000000000000000000000010000000000000000", 
								 "000100100010000000111000000000100000000000110000000000100000000000000000000000000000000000000000000000000100000000000000000000000000000000001010000000000000000000000000000000010000000000000000") == false); 
}
BOOST_AUTO_TEST_CASE( test_case3 ) { 
	BOOST_CHECK(prefix_less_than("000100100010000000111000000000100000000000110000000000100000000000000000000000000000000000000000000000000100000000000000000000000000000000001010000000000000000000000000000000010000000000000000", 
								 "000010100010000000111000000000100000000000110000000000100000000000000000000000000000000000000000000000000100000000000000000000000000000000001010000000000000000000000000000000010000000000000000") == false); 
}
BOOST_AUTO_TEST_CASE( test_case4 ) { 
	BOOST_CHECK(prefix_less_than("000100100010000000111000000000100000000000110000000000100000000000000000000000000000000000000000000000000100000000000000000000000000000000001010000000000000000000000000000000010000000000000000", 
								 "000100100010000000111000000000100000000000110000000001000000000000000000000000000000000000000000000000000100000000000000000000000000000000001010000000000000000000000000000000010000000000000000") == true); 
}

BOOST_AUTO_TEST_CASE( test_case5 ) { 
	BOOST_CHECK(prefix_less_than("000100100010000000111000000000100000000000110000000001000000000000000000000000000000000000000000000000000100000000000000000000000000000000001010000000000000000000000000000000010000000000000000",
								 "000100100010000000111000000000100000000000110000000000100000000000000000000000000000000000000000000000000100000000000000000000000000000000001010000000000000000000000000000000010000000000000000") == false); 
}

BOOST_AUTO_TEST_SUITE_END()


