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
#include <vector>

#include "key.hh"
#include "filter.hh"
#include "query.hh"
#include "tagmatch.hh"
#include "synchronous_match_handler.hh"

using std::set;
using std::cout;
using std::endl;

static void init_test() {
	tagmatch::set_latency_limit_ms(200);
	tagmatch::clear();
}

class general_matcher : public synchronous_match_handler {
public:
	general_matcher(): synchronous_match_handler(), keys() {};

	virtual void process_results(query * q) {
		for (const tagmatch_key_t & k : q->output_keys)
			keys.insert(k);
	}

	const set<tagmatch_key_t> & result() const {
		return keys;
	}

	void reset() {
		keys.clear();
	}

private:
	set<tagmatch_key_t> keys;
};

class finder_matcher : public synchronous_match_handler {
public:
	finder_matcher(tagmatch_key_t t): target(t), found(false) {};

	virtual void process_results(query * q) {
		for (const tagmatch_key_t & k : q->output_keys) {
			if (k == target)
				found = true;
		}
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

const set<tagmatch_key_t> & matching_results(const filter_t & f) {
	static set<tagmatch_key_t> result;
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

class deferred_matching_checks {

	struct check {
		tagmatch_query * q;
		general_matcher * handler;
		set<tagmatch_key_t> expected;

		check(const set<tagmatch_key_t> & s)
			: q(nullptr), handler(nullptr), expected(s) {}
	};

	std::vector<check> checks;

public:
	void run_query(const filter_t & f, const set<tagmatch_key_t> & s) {
		checks.emplace_back(s);
		check & c = checks.back();
		c.handler = new general_matcher();
		c.q = new tagmatch_query(f);
		BOOST_REQUIRE( c.handler != nullptr);
		BOOST_REQUIRE( c.q != nullptr);
		tagmatch::match_unique(c.q, c.handler);
	}

	void check_results() {
		for (unsigned int i = 0; i < checks.size(); ++i) {
			const check & c = checks[i];
			c.handler->synchronize_and_process();
#if 0
			BOOST_CHECK_MESSAGE( (c.handler->result() == c.expected) , "on query " << i << " expecting " << c.expected << " instead of " << c.handler->result() );
#else
			BOOST_CHECK_MESSAGE( (c.handler->result() == c.expected) , "query " << i << ": unexpected results");
#endif			
		}
	}

	~deferred_matching_checks() {
		for (check & c : checks) {
			if (c.handler)
				delete(c.handler);
			if (c.q)
				delete(c.q);
		}
	}
};

std::ostream & operator << (std::ostream & output, const set<tagmatch_key_t> & s) {
	output << '{';
	for (const tagmatch_key_t & k : s)
		output << ' ' << k;
	output << " }";
	return output;
}

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

BOOST_AUTO_TEST_CASE( repeated_stop ) {
	tagmatch::start();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::clear();
	tagmatch::start();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::clear();
	tagmatch::clear();
	tagmatch::start();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::stop();
	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( find_one_filter ) {
	tagmatch::clear();
	tagmatch::add(F[1], 1);
	tagmatch::consolidate();
	tagmatch::start();

	finder_matcher f1(1);
	tagmatch_query q1(F[1]);
	tagmatch::match_unique(&q1, &f1);

	finder_matcher f2(1);
	tagmatch_query q2(F[2]);
	tagmatch::match_unique(&q2, &f2);

	finder_matcher f3(1);
	tagmatch_query q3(ALL_ZEROS);
	tagmatch::match_unique(&q3, &f3);

	finder_matcher f4(1);
	tagmatch_query q4(ALL_ONES);
	tagmatch::match_unique(&q4, &f4);

	tagmatch::stop();

	f1.synchronize_and_process();
	f2.synchronize_and_process();
	f3.synchronize_and_process();
	f4.synchronize_and_process();

	BOOST_CHECK(f1.result());
	BOOST_CHECK(!f2.result());
	BOOST_CHECK(!f3.result());
	BOOST_CHECK(f4.result());

	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( clear ) {
	tagmatch::clear();
	tagmatch::add(F[1], 1);
	tagmatch::consolidate();
	tagmatch::start();

	finder_matcher f1(1);
	tagmatch_query q1(F[1]);
	tagmatch::match_unique(&q1, &f1);

	tagmatch::stop();

	f1.synchronize_and_process();

	BOOST_CHECK(f1.result());

	tagmatch::clear();
	tagmatch::consolidate();
	tagmatch::start();

	finder_matcher f2(1);
	tagmatch_query q2(F[1]);
	tagmatch::match_unique(&q2, &f2);

	tagmatch::stop();

	f2.synchronize_and_process();
	BOOST_CHECK(!f2.result());
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( same_filter )

BOOST_AUTO_TEST_CASE( add_same_filter ) {
	tagmatch::clear();
	tagmatch::add(F[1], 2);
	tagmatch::add(F[1], 3);
	tagmatch::add(F[1], 4);
	tagmatch::consolidate();
	tagmatch::start();

	general_matcher m1;
	tagmatch_query q1(F[1]);
	tagmatch::match_unique(&q1, &m1);

	general_matcher m2;
	tagmatch_query q2(F[2]);
	tagmatch::match_unique(&q2, &m2);

	tagmatch::stop();

	m1.synchronize_and_process();
	m2.synchronize_and_process();

	BOOST_CHECK(m1.result() == set<tagmatch_key_t>({ 2, 3, 4 }));
	BOOST_CHECK(m2.result() == set<tagmatch_key_t>({ }));

	tagmatch::clear();
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( finding_filters )

BOOST_AUTO_TEST_CASE( add_and_find_multi ) {
	tagmatch::set_latency_limit_ms(400);
	tagmatch::clear();

	tagmatch::add(F[1], 1);
	tagmatch::add(F[1], 2);
	tagmatch::add(F[2], 3);
	tagmatch::add(F[3], 4);
	tagmatch::consolidate();

	deferred_matching_checks C;

	C.run_query(F[1], set<tagmatch_key_t>({ 1, 2 }));
	C.run_query(F[2], set<tagmatch_key_t>({ 3 }));
	C.run_query(F[3], set<tagmatch_key_t>({ 4 }));

	tagmatch::start();

	tagmatch::stop();

	C.check_results();

	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( add_and_find_multi2 ) {
	init_test();
	tagmatch::add(F[1], 1);
	tagmatch::add(F[1], 2);
	tagmatch::add(F[2], 3);
	tagmatch::add(F[3], 4);
	tagmatch::consolidate();
	tagmatch::start();

	deferred_matching_checks C;

	C.run_query(F[1], set<tagmatch_key_t>({ 1, 2 }));
	C.run_query(F[2], set<tagmatch_key_t>({ 3 }));
	C.run_query(F[3], set<tagmatch_key_t>({ 4 }));

	tagmatch::stop();

	C.check_results();

	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( add_and_clear ) {
	deferred_matching_checks C;

	init_test();

	tagmatch::consolidate();
	tagmatch::start(4, 1);
	C.run_query(F[1], { });
	tagmatch::stop();

	tagmatch::add(F[1], 3);
	tagmatch::consolidate();
	tagmatch::start(4, 1);
	C.run_query(F[1], { 3 });

	tagmatch::stop();
	tagmatch::clear();
	tagmatch::consolidate();
	tagmatch::start(4, 1);
	C.run_query(F[1], { });

	tagmatch::stop();

	C.check_results();

	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( add_many_and_find ) {
	init_test();
	deferred_matching_checks C;
	
	tagmatch::add(F[1], 1);
	tagmatch::add(F[2], 2);
	tagmatch::add(F[3], 3);
	tagmatch::add(F[4], 4);
	tagmatch::add(F[5], 5);
	tagmatch::add(F[6], 6);
	tagmatch::add(F[7], 7);
	tagmatch::add(F[8], 8);
	tagmatch::add(F[9], 9);
	tagmatch::add(F[10], 10);
	tagmatch::consolidate();
	tagmatch::start(4, 1);

	C.run_query(F[1], { 1 });
	C.run_query(F[2], { 2 });
	C.run_query(F[3], { 3 });
	C.run_query(F[4], { 4 });
	C.run_query(F[5], { 5 });
	C.run_query(F[6], { 6 });
	C.run_query(F[7], { 7 });
	C.run_query(F[8], { 8 });
	C.run_query(F[9], { 9 });
	C.run_query(F[10], { 10 });

	tagmatch::stop();

	C.check_results();

	tagmatch::clear();
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( subset_search )

BOOST_AUTO_TEST_CASE( misc_subsets ) {
	init_test();
	deferred_matching_checks C;

	static const int N = sizeof(F) / sizeof(const filter_t);

	for(unsigned int i = 0; i < N; ++i)
		tagmatch::add(F[i], i);

	static const int M = sizeof(Q) / sizeof(const filter_t);

	tagmatch::consolidate();
	tagmatch::start(4, 1);

	for(unsigned int i = 0; i < M; ++i) {
		set<tagmatch_key_t> expected_results;
		expected_results.clear();
		for (int j = 0; j < N; ++j) 
			if (F[j].subset_of(Q[i]))
				expected_results.insert(j);
		C.run_query(Q[i], expected_results);
	}

	tagmatch::stop();

	C.check_results();

	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( misc_subsets_Q1 ) {
	init_test();

	static const int N = sizeof(F) / sizeof(const filter_t);

	for(unsigned int i = 0; i < N; ++i)
		tagmatch::add(F[i], i);

	tagmatch::consolidate();
	tagmatch::start(4, 1);

	deferred_matching_checks C;
	
	C.run_query(Q[1], set<tagmatch_key_t>({ 2 }));

	tagmatch::stop();

	C.check_results();

	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( corner_case_filters ) {
	init_test();
	filter_t f_first;
	filter_t f_last;
	f_first.clear();
	f_first.set_bit(0);
	f_last.clear();
	f_last.set_bit(filter_t::WIDTH - 1);

	tagmatch::add(f_first, 1);
	tagmatch::add(f_last, 2);

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

	tagmatch::consolidate();
	tagmatch::start(4, 1);

	deferred_matching_checks C;

	C.run_query(has_none, set<tagmatch_key_t>({}));
	C.run_query(has_first, set<tagmatch_key_t>({ 1 }));
	C.run_query(has_last, set<tagmatch_key_t>({ 2 }));
	C.run_query(has_both, set<tagmatch_key_t>({ 1, 2 }));

	tagmatch::stop();
	C.check_results();
	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( single_bit_filters ) {
	init_test();
	deferred_matching_checks C;

	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i) {
		filter_t f;
		f.clear();
		f.set_bit(i);
		tagmatch::add(f, i);
	}

	set<tagmatch_key_t> all_interfaces;
	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i)
		all_interfaces.insert(i);

	tagmatch::consolidate();
	tagmatch::start(4, 1);
	C.run_query(ALL_ONES, all_interfaces);

	tagmatch::stop();
	C.check_results();
	tagmatch::clear();
}

BOOST_AUTO_TEST_CASE( deepest_trie ) {
	init_test();
	deferred_matching_checks C;

	filter_t f;			// all-zero
	f.clear();

	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i) {
		f.set_bit(i);
		tagmatch::add(f, i);
	}

	set<tagmatch_key_t> all_interfaces;
	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i)
		all_interfaces.insert(i);

	tagmatch::consolidate();
	tagmatch::start(4, 1);
	C.run_query(ALL_ONES, all_interfaces);

	tagmatch::stop();
	C.check_results();
	tagmatch::clear();
}

BOOST_AUTO_TEST_SUITE_END()
