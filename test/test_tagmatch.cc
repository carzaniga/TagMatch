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

#include "key.hh"
#include "filter.hh"
#include "query.hh"
#include "tagmatch.hh"
#include "synchronous_match_handler.hh"

using std::set;
using std::cout;
using std::endl;

class predicate {
public:
	predicate() {
		tagmatch::set_latency_limit_ms(200);
		tagmatch::stop();
		tagmatch::clear();
	}

	~predicate() {
		tagmatch::stop();
		tagmatch::clear();
	}

	void clear() {
		tagmatch::stop();
		tagmatch::clear();
	}

	void stop() {
		tagmatch::stop();
	}

	void start() {
		tagmatch::start();
	}

	void consolidate() {
		tagmatch::consolidate();
		tagmatch::start(4, 1);
	}

	void add(const filter_t & f, tagmatch_key_t k) {
		tagmatch::add(f, k);
	}
};

class general_matcher : public synchronous_match_handler {
public:
	general_matcher(set<tagmatch_key_t> & s): result(s) {};

	virtual void process_results(query * q) {
		std::cerr << "general_matcher::process_results ";
		for (const tagmatch_key_t & k : q->output_keys) {
			std::cerr << ' ' << k;
			result.insert(k);
		}
		std::cerr << std::endl;
	}

private:
	set<tagmatch_key_t> & result;
};

class finder_matcher : public synchronous_match_handler {
public:
	finder_matcher(tagmatch_key_t t): target(t), found(false) {};

	virtual void process_results(query * q) {
		std::cerr << "finder_matcher::process_results ";
		for (const tagmatch_key_t & k : q->output_keys) {
			std::cerr << ' ' << k;
			if (k == target)
				found = true;
		}
		std::cerr << std::endl;
	}

	bool result() const {
		return found;
    }

private:
	tagmatch_key_t target;
	bool found;
};

bool find_filter (const filter_t & f, tagmatch_key_t i) {
	finder_matcher finder(i);
	tagmatch_query q(f);
	tagmatch::match_unique(&q, &finder);
	finder.synchronize_and_process();
	return finder.result();
}

const set<tagmatch_key_t> & matching_results(predicate & P, const filter_t & f) {
	static set<tagmatch_key_t> result;
	result.clear();
	general_matcher matcher(result);
	tagmatch_query q(f);
	tagmatch::match_unique(&q, &matcher);
	matcher.synchronize_and_process();
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

#define BOOST_TEST_MODULE tagmatch
#define BOOST_TEST_DYN_LINK 1

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE( basics )

BOOST_AUTO_TEST_CASE( consolidate_empty ) {
	tagmatch::set_latency_limit_ms(200);
	tagmatch::consolidate();
	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( start_and_stop ) {
	tagmatch::consolidate();
	tagmatch::start();
	tagmatch::stop();

	tagmatch::start(4, 1);
	tagmatch::stop();
	tagmatch::clear();

	tagmatch::consolidate();
	tagmatch::start(4, 1);
	tagmatch::stop();

	tagmatch::start();
	tagmatch::stop();
	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( consolidate_no_match_action ) {
	tagmatch::add(F[0], 0);
	tagmatch::add(F[1], 1);
	tagmatch::add(F[2], 2);
	tagmatch::add(F[3], 3);
	tagmatch::add(F[4], 4);
	tagmatch::consolidate();
	tagmatch::start();
	tagmatch::stop();
	tagmatch::start(4, 1);
	tagmatch::stop();
	tagmatch::start();
	tagmatch::stop();
	tagmatch::clear();
	tagmatch::add(F[0], 0);
	tagmatch::add(F[1], 1);
	tagmatch::add(F[2], 2);
	tagmatch::add(F[3], 3);
	tagmatch::add(F[4], 4);
	tagmatch::consolidate();
	tagmatch::start();
	tagmatch::stop();
	tagmatch::start();
	tagmatch::stop();
	tagmatch::clear();
}

#if 0
BOOST_AUTO_TEST_CASE( double_stop ) {
	tagmatch::start();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::clear();
}
#endif

BOOST_AUTO_TEST_CASE( find_one_filter ) {
	predicate P;
	tagmatch::add(F[1], 1);
	tagmatch::consolidate();
	tagmatch::start();
	BOOST_CHECK(find_filter(F[1], 1));
	BOOST_CHECK(!find_filter(F[2], 1));
	BOOST_CHECK(!find_filter(ALL_ZEROS, 1));
	BOOST_CHECK(find_filter(ALL_ONES, 1));
	tagmatch::stop();
	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( clear ) {
	predicate P;
	P.add(F[1], 1);
	P.consolidate();
	BOOST_CHECK(find_filter(F[1], 1));
	P.clear();
	P.consolidate();
	BOOST_CHECK(!find_filter(F[1], 1));
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( same_filter )

BOOST_AUTO_TEST_CASE( add_same_filter ) {
	predicate P;
	P.add(F[1], 2);
	P.add(F[1], 3);
	P.add(F[1], 4);
	P.consolidate();
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( finding_filters )

BOOST_AUTO_TEST_CASE( add_and_find_multi ) {
	predicate P;
	P.add(F[1], 1);
	P.add(F[1], 2);
	P.add(F[2], 3);
	P.add(F[3], 4);
	P.consolidate();
	BOOST_CHECK(!find_filter(F[1], 1));
	BOOST_CHECK(find_filter(F[1], 2));
	BOOST_CHECK(!find_filter(F[2], 1));
	BOOST_CHECK(!find_filter(F[3], 1));
	BOOST_CHECK(find_filter(F[2], 3));
	BOOST_CHECK(find_filter(F[3], 4));
}

BOOST_AUTO_TEST_CASE( add_and_clear ) {
	predicate P;

	P.consolidate();
	BOOST_CHECK(!find_filter(F[1], 1));
	P.stop();

	P.add(F[1], 3);
	P.consolidate();
	BOOST_CHECK(find_filter(F[1], 3));

	P.stop();
	P.clear();
	P.consolidate();
	BOOST_CHECK(!find_filter(F[1], 3));
}

BOOST_AUTO_TEST_CASE( add_many_and_find ) {
	predicate P;

	P.add(F[1], 1);
	P.add(F[2], 2);
	P.add(F[3], 3);
	P.add(F[4], 4);
	P.add(F[5], 5);
	P.add(F[6], 6);
	P.add(F[7], 7);
	P.add(F[8], 8);
	P.add(F[9], 9);
	P.add(F[10], 10);
	P.consolidate();

	BOOST_CHECK(find_filter(F[1], 1));
	BOOST_CHECK(find_filter(F[2], 2));
	BOOST_CHECK(find_filter(F[3], 3));
	BOOST_CHECK(find_filter(F[4], 4));
	BOOST_CHECK(find_filter(F[5], 5));
	BOOST_CHECK(find_filter(F[6], 6));
	BOOST_CHECK(find_filter(F[7], 7));
	BOOST_CHECK(find_filter(F[8], 8));
	BOOST_CHECK(find_filter(F[9], 9));
	BOOST_CHECK(find_filter(F[10], 10));
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( subset_search )

BOOST_AUTO_TEST_CASE( misc_subsets ) {
	predicate P;

	static const int N = sizeof(F) / sizeof(const filter_t);

	for(unsigned int i = 0; i < N; ++i)
		P.add(F[i], i);

	static const int M = sizeof(Q) / sizeof(const filter_t);

	P.consolidate();

	for(unsigned int i = 0; i < M; ++i) {
		set<tagmatch_key_t> expected_results;
		for (int j = 0; j < N; ++j)
			if (F[j].subset_of(Q[i]))
				expected_results.insert(j);

		BOOST_CHECK(matching_results(P, Q[i]) == expected_results);
	}
}

BOOST_AUTO_TEST_CASE( corner_case_filters ) {
	predicate P;
	filter_t f_first;
	filter_t f_last;
	f_first.clear();
	f_first.set_bit(0);
	f_last.clear();
	f_last.set_bit(filter_t::WIDTH - 1);

	P.add(f_first, 1);
	P.add(f_last, 2);

	filter_t has_first;
	has_first.clear();
	has_first.set_bit(0);
	has_first.set_bit(10);
	has_first.set_bit(100);
	has_first.set_bit(filter_t::WIDTH - 2);

	filter_t has_last;
	has_last.clear();
	has_last.set_bit(1);
	has_last.set_bit(10);
	has_last.set_bit(100);
	has_last.set_bit(filter_t::WIDTH - 1);

	filter_t has_none;
	has_none.clear();
	has_none.set_bit(1);
	has_none.set_bit(10);
	has_none.set_bit(100);
	has_none.set_bit(filter_t::WIDTH - 2);

	filter_t has_both;
	has_both.clear();
	has_both.set_bit(0);
	has_both.set_bit(10);
	has_both.set_bit(100);
	has_both.set_bit(filter_t::WIDTH - 1);

	P.consolidate();
	BOOST_CHECK(matching_results(P, has_none) == set<tagmatch_key_t>({}));
	BOOST_CHECK(matching_results(P, has_first) == set<tagmatch_key_t>({ 1 }));
	BOOST_CHECK(matching_results(P, has_last) == set<tagmatch_key_t>({ 2 }));
	BOOST_CHECK(matching_results(P, has_both) == set<tagmatch_key_t>({ 1, 2 }));
}

BOOST_AUTO_TEST_CASE( single_bit_filters ) {
	predicate P;

	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i) {
		filter_t f;
		f.clear();
		f.set_bit(i);
		P.add(f, i);
	}

	set<tagmatch_key_t> all_interfaces;
	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i)
		all_interfaces.insert(i);

	P.consolidate();
	BOOST_CHECK(matching_results(P, ALL_ONES) == all_interfaces);
}

BOOST_AUTO_TEST_CASE( deepest_trie ) {
	predicate P;

	filter_t f;			// all-zero
	f.clear();

	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i) {
		f.set_bit(i);
		P.add(f, i);
	}

	set<tagmatch_key_t> all_interfaces;
	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i)
		all_interfaces.insert(i);

	P.consolidate();
	BOOST_CHECK(matching_results(P, ALL_ONES) == all_interfaces);
}

BOOST_AUTO_TEST_SUITE_END()

