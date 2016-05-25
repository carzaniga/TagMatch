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
#include <set>

#include "patricia_predicate.hh"
#include "routing.hh"

#ifndef INTERFACES
#define INTERFACES 256U
#endif

using std::set;
using std::cout;
using std::endl;

class general_matcher : public match_handler {
public:
	general_matcher(set<interface_t> & s): result(s) {};

	virtual bool match(const filter_t & f, tree_t t, interface_t i) {
		result.insert(i);
		return false;
	}

private:
	set<interface_t> & result;
};

class tree_matcher : public match_handler {
public:
	tree_matcher(set<interface_t> & s, tree_t t): result(s), tree(t) {};

	virtual bool match(const filter_t & f, tree_t t, interface_t i) {
		if (t == tree)
			result.insert(i);
		return false;
	}

private:
	set<interface_t> & result;
	tree_t tree;
};

class filter_finder : public match_handler {
public:
	filter_finder(const filter_t & f): filter(f), found(false) {};

	virtual bool match(const filter_t & f, tree_t t, interface_t i) {
		if (f == filter)
			found = true;
		return found;
	}

	bool result() const {
		return found;
    }

private:
	filter_t filter;
	bool found;
};

bool find_filter(predicate & P, const filter_t & f) {
	filter_finder finder(f);
	P.match(f, finder);
	return finder.result();
}

const set<interface_t> & matching_interfaces(predicate & P, const filter_t & f) {
	static set<interface_t> result;
	result.clear();
	general_matcher matcher(result);
	P.match(f, matcher);
	return result;
}

const set<interface_t> & matching_interfaces(predicate & P, const filter_t & f, tree_t t) {
	static set<interface_t> result;
	result.clear();
	tree_matcher matcher(result, t);
	P.match(f, matcher);
	return result;
}

static const filter_t ALL_ONES("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111");
static const filter_t ALL_ZEROS("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

static const filter_t F[] = { 
	filter_t("000000000000100000001000000000000001000000010001000000000000000000000000000000000000000000000000001000000000000000000000001000000000000000000010000000000001000000000000010000000000001010010001"),
	filter_t("000000000000010000000100000000000000100000001000100000000000000000000000000000000000000000000000000100000000000000000000000100000000000000000001000000000000100000000000001000000000000101001000"),
	filter_t("000000000000010000000000000000000000100000000000100000000010000000000010010000000000000100010000000000000000100000000000000000000000000000100000000100000010000101000000000110000010000100000010"),
	filter_t("000000000001000000000001000100000000001000000010000000000000000001000000000000000100000000000000000000000000000010100000010000000011000000010000000010000000000000000010000000001000000000101000"),
	filter_t("000010000100100000000000001000000100000000000000000000000000000000000000000101000000000000010000000000000000000000000000001000000100000000101000000000000010000000000000010000000000000000000000"),
	filter_t("000110000010000100100000000000000000000000010000000000001001000000100000000000000000100010000000000000000000000000000000000000101000000000000011000000001000000000000100000000000001001000000010"),
	filter_t("011000000010000000000000000000000001000000000000000000000000000000000001000010000000000000001000000000000010000000000010000000000000010000000100000000100000000000000000000000000000101000000000"),
	filter_t("000010000000010000000000000010010000000000000001000100000000000000000000000000010000000000000000100000000000000000000000000000000000000000001000000000000000000000000000000000000000111000000000"),
	filter_t("000000000000000000000000000000001100000000000000000100000000000000010000000010001000000000000000000000001000000000000000010000000001100000000000000000000000000000000000000000000100000100100001"),
	filter_t("100000100000000000000010000000000000000000000000000000000010000010000000011010010000000101000000000000000000000000000000001001000000010100000000000000000000000000000000001010001100000000000000"),
	filter_t("000000000000000100000000100000010000000010000000000010100000000000000000100000000000000000010011000101000000000100000000000001000000000001000000010100100000000100000000000000011000000000000000"),
};

static const filter_t Q[] = {
	filter_t("000000000000010000000100000000000000100000001000100000000000000000000000000000000000000000000000000100000000000000000000000100000000000000000001000000000000100000000000000000000000000101001000"),
	filter_t("100000100000010001000010000000000000100000000000100011000010000010000010011010010000000101010000000000000001100000000000001001000000010100100000000100000010000111000000000111000110000100000010"),
	filter_t("000000000000010000000000000000000000100000000000100000000010000000000010010000000000000100010000000000000000100000000000000000000000000000100000000100000010000101000000000110000010000100000010"),
	filter_t("000000000001000000000001000100000000001000000010000000000000000001000000000000000100000000000000000000000000000010100000010000000011000000010000000010000000000000000010000000001000000000101000"),
	filter_t("100010100111110101100010000000000000100000011111010001111101000001100001001101001000100010101000000000000000110000000000001100100100001010111000000001000011100001110000111000001110000100000010"),
	filter_t("000010000100100000000000001000000100000000000000000000000000000000000000000101000000000000010000000000000000000000000000001000000100000000101000000000000010000000000000010000000000000000000000"),
	filter_t("000110000010000100100000000000000000000000010000000000001001000000100000000000000000100010000000000000000000000000000000000000101000000000000011000000001000000000000100000000000001001000000010"),
	filter_t("011000000010000000000000000000000001000000000000000000000000000000000001000010000000000000001000000000000010000000000010000000000000010000000100000000100000000000000000000000000000101000000000"),
	filter_t("000010000000010000000000000010010000000000000001000100000000000000000000000000010000000000000000100000000000000000000000000000000000000000001000000000000000000000000000000000000000111000000000"),
	filter_t("000000000000000000000000000000001100000000000000000100000000000000010000000010001000000000000000000000001000000000000000010000000001100000000000000000000000000000000000000000000100000100100001"),
	filter_t("000000100000000000000010001010011100000000000001000100000010000010010011001110011000000101000011100000001111000000000000011001000001110100001100000000000000000000000000001010001100111100110001"),
	filter_t("000000000000000100000000100000010000000010000000000010100000000000000000100000000000000000010011000101000000000100000000000001000000000001000000010100100000000100000000000000011000000000000000"),
	filter_t("111111111111111111111111111111111111111011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"),
};

#define BOOST_TEST_MODULE predicate
#define BOOST_TEST_DYN_LINK 1

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE( patricia_predicate_basics )

BOOST_AUTO_TEST_CASE( find_in_empty_predicate ) {
	predicate P;
	BOOST_CHECK(!find_filter(P, ALL_ZEROS));
	BOOST_CHECK(!find_filter(P, ALL_ONES));
}

BOOST_AUTO_TEST_CASE( find_one_filter ) {
	predicate P;
	BOOST_CHECK(!find_filter(P, F[1]));
	P.add(F[1], 1, 1);
	BOOST_CHECK(find_filter(P, F[1]));
	BOOST_CHECK(!find_filter(P, F[2]));
	BOOST_CHECK(!find_filter(P, ALL_ZEROS));
	BOOST_CHECK(!find_filter(P, ALL_ONES));
}

BOOST_AUTO_TEST_CASE( clear ) {
	predicate P;
	BOOST_CHECK(!find_filter(P, F[1]));
	P.add(F[1], 1, 1);
	BOOST_CHECK(find_filter(P, F[1]));
	P.clear();
	BOOST_CHECK(!find_filter(P, F[1]));
}

BOOST_AUTO_TEST_CASE( add_tree_interface ) {
	predicate P;
	BOOST_CHECK(matching_interfaces(P, F[1]).empty());
	BOOST_CHECK(matching_interfaces(P, F[1], 1).empty());
	P.add(F[1], 1, 2);
	BOOST_CHECK(matching_interfaces(P, F[1]) == set<interface_t>({ 2 }));
	BOOST_CHECK(matching_interfaces(P, F[1], 1) == set<interface_t>({ 2 }));
	BOOST_CHECK(matching_interfaces(P, F[1], 2).empty());
	P.add(F[1], 1, 3);
	BOOST_CHECK(matching_interfaces(P, F[1]) == set<interface_t>({ 2, 3 }));
	BOOST_CHECK(matching_interfaces(P, F[1], 1) == set<interface_t>({ 2, 3 }));
	BOOST_CHECK(matching_interfaces(P, F[1], 2).empty());
	P.add(F[1], 2, 4);
	BOOST_CHECK(matching_interfaces(P, F[1]) == set<interface_t>({ 2, 3, 4 }));
	BOOST_CHECK(matching_interfaces(P, F[1], 1) == set<interface_t>({ 2, 3 }));
	BOOST_CHECK(matching_interfaces(P, F[1], 2) == set<interface_t>({ 4 }));
}

BOOST_AUTO_TEST_CASE( add_array_of_tips ) {
	predicate P;

	tree_interface_pair tips[3];

	tips[0] = tip_value(1, 2);
	tips[1] = tip_value(1, 3);
	tips[2] = tip_value(2, 4);

	P.add(F[1], tips, tips + 3);
	BOOST_CHECK(matching_interfaces(P, F[1]) == set<interface_t>({ 2, 3, 4 }));
	BOOST_CHECK(matching_interfaces(P, F[1], 1) == set<interface_t>({ 2, 3 }));
	BOOST_CHECK(matching_interfaces(P, F[1], 2) == set<interface_t>({ 4 }));

	tips[0] = tip_value(1, 2);
	tips[1] = tip_value(1, 3);
	tips[2] = tip_value(3, 4);

	P.add(F[1], tips, tips + 3);
	BOOST_CHECK(matching_interfaces(P, F[1]) == set<interface_t>({ 2, 3, 4 }));
	BOOST_CHECK(matching_interfaces(P, F[1], 1) == set<interface_t>({ 2, 3 }));
	BOOST_CHECK(matching_interfaces(P, F[1], 2) == set<interface_t>({ 4 }));
	BOOST_CHECK(matching_interfaces(P, F[1], 3) == set<interface_t>({ 4 }));
}

BOOST_AUTO_TEST_CASE( add_and_find_multi ) {
	predicate P;
	P.add(F[1], 1, 1);
	P.add(F[1], 1, 2);
	P.add(F[2], 1, 3);
	P.add(F[3], 1, 4);

	BOOST_CHECK(find_filter(P, F[1]));
	BOOST_CHECK(find_filter(P, F[2]));
	BOOST_CHECK(find_filter(P, F[3]));
}

BOOST_AUTO_TEST_CASE( add_and_clear ) {
	predicate P;

	BOOST_CHECK(!find_filter(P, F[1]));

	P.add(F[1], 1, 3);
	P.add(F[1], 1, 2);

	BOOST_CHECK(find_filter(P, F[1]));

	P.clear();

	BOOST_CHECK(!find_filter(P, F[1]));
}

BOOST_AUTO_TEST_CASE( add_many_and_find ) {
	predicate P;

	P.add(F[1], 1, 1);
	P.add(F[2], 1, 2);
	P.add(F[3], 1, 3);
	P.add(F[4], 1, 4);
	P.add(F[5], 1, 5);
	P.add(F[6], 1, 6);
	P.add(F[7], 1, 7);
	P.add(F[8], 1, 8);
	P.add(F[9], 1, 9);
	P.add(F[10], 1, 10);

	BOOST_CHECK(find_filter(P, F[1]));
	BOOST_CHECK(find_filter(P, F[2]));
	BOOST_CHECK(find_filter(P, F[3]));
	BOOST_CHECK(find_filter(P, F[4]));
	BOOST_CHECK(find_filter(P, F[5]));
	BOOST_CHECK(find_filter(P, F[6]));
	BOOST_CHECK(find_filter(P, F[7]));
	BOOST_CHECK(find_filter(P, F[8]));
	BOOST_CHECK(find_filter(P, F[9]));
	BOOST_CHECK(find_filter(P, F[10]));
}

BOOST_AUTO_TEST_CASE( add_many_and_match ) {
	predicate P;

	P.add(F[1], 1, 1);
	P.add(F[2], 1, 2);
	P.add(F[3], 1, 3);
	P.add(F[4], 1, 4);
	P.add(F[5], 1, 5);
	P.add(F[6], 1, 6);
	P.add(F[7], 1, 7);
	P.add(F[8], 1, 8);
	P.add(F[9], 1, 9);
	P.add(F[10], 1, 10);

	BOOST_CHECK(matching_interfaces(P, F[1]) == set<interface_t>({ 1 }));
	BOOST_CHECK(matching_interfaces(P, F[2]) == set<interface_t>({ 2 }));
	BOOST_CHECK(matching_interfaces(P, F[3]) == set<interface_t>({ 3 }));
	BOOST_CHECK(matching_interfaces(P, F[4]) == set<interface_t>({ 4 }));
	BOOST_CHECK(matching_interfaces(P, F[5]) == set<interface_t>({ 5 }));
	BOOST_CHECK(matching_interfaces(P, F[6]) == set<interface_t>({ 6 }));
	BOOST_CHECK(matching_interfaces(P, F[7]) == set<interface_t>({ 7 }));
	BOOST_CHECK(matching_interfaces(P, F[8]) == set<interface_t>({ 8 }));
	BOOST_CHECK(matching_interfaces(P, F[9]) == set<interface_t>({ 9 }));
	BOOST_CHECK(matching_interfaces(P, F[10]) == set<interface_t>({ 10 }));
}

BOOST_AUTO_TEST_CASE( subsets ) {
	predicate P;

	static const int N = sizeof(F) / sizeof(const filter_t);

	for(unsigned int i = 0; i < N; ++i)
		P.add(F[i], 1, i);

	static const int M = sizeof(Q) / sizeof(const filter_t);

	for(unsigned int i = 0; i < M; ++i) {
		set<interface_t> expected_results;
		for (int j = 0; j < N; ++j)
			if (F[j].subset_of(Q[i]))
				expected_results.insert(j);

		BOOST_CHECK(matching_interfaces(P, Q[i]) == expected_results);
	}
}

BOOST_AUTO_TEST_CASE( deepest_trie ) {
	predicate P;

	filter_t f;			// all-zero

	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i) {
		f.set_bit(i);
		P.add(f, 1, i);
	}

	set<interface_t> all_interfaces;
	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i)
		all_interfaces.insert(i);

	BOOST_CHECK(matching_interfaces(P, f) == all_interfaces);
}

BOOST_AUTO_TEST_SUITE_END()

