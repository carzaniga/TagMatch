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

#include "predicate.hh"

#define BOOST_TEST_MODULE bv192 test
#define BOOST_TEST_DYN_LINK 1

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE( predicate_basics )

BOOST_AUTO_TEST_CASE( add_and_find ) {
	predicate P;

	filter_t f1("001010000110101011000001000100000000011100001110010001101001001111101000000000001110011010010100100010000001011000000100100000001101000110011001001000000000001000111010000000000110000010000110");

	predicate::node * n1 = P.add(f1, 1, 3);
	predicate::node * n2 = P.add(f1, 1, 2);
	
	BOOST_CHECK(n1 == n2);

	n2 = P.add(f1);

	BOOST_CHECK(n1 == n2);

	filter_t f2("100100011010010011111010000000000011100110100101001000100000010001010000110101011000001000100000000011100001100001000111010000000110000001001000000011010001100110010010000000000110000010000110");

	n2 = P.add(f2, 1, 3);
	BOOST_CHECK(n1 != n2);

	predicate::node * nX;

	nX = P.find(f1);
	BOOST_CHECK(nX == n1);

	nX = P.find(f2);
	BOOST_CHECK(nX == n2);

	filter_t fX("100100011010010011111010000000000011100110100101001000100000010001010000110101011000001000100000000011100001100001000111010000000110000001001000000011010001100100010010000000000110000010000110");
	nX = P.find(fX);

	BOOST_CHECK(nX == 0);
}

BOOST_AUTO_TEST_CASE( many_tree_interface_pairs ) {
	predicate P;

	filter_t f1("001010000110101011000001000100000000011100001110010001101001001111101000000000001110011010010100100010000001011000000100100000001101000110011001001000000000001000111010000000000110000010000110");
	filter_t f2("100100011010010011111010000000000011100110100101001000100000010001010000110101011000001000100000000011100001100001000111010000000110000001001000000011010001100110010010000000000110000010000110");

	std::set<tree_interface_pair> S1;
	std::set<tree_interface_pair> S2;

	for(tree_t t = 0; t < 4; ++t) {
		for(interface_t i = 0; i < 10; ++i) {
			P.add(f1, t, i);
			S1.insert(tree_interface_pair(t,i));
			++i;
			P.add(f2, t, i); 
			S2.insert(tree_interface_pair(t,i));
		}
	}

	predicate::node * n = P.find(f1);

	for(const tree_interface_pair * ti = n->ti_begin(); ti != n->ti_end(); ++ti) {
		BOOST_CHECK(S1.count(*ti) == 1);
		S1.erase(*ti);
	}

	BOOST_CHECK(S1.empty());

	n = P.find(f2);

	for(const tree_interface_pair * ti = n->ti_begin(); ti != n->ti_end(); ++ti) {
		BOOST_CHECK(S2.count(*ti) == 1);
		S2.erase(*ti);
	}

	BOOST_CHECK(S2.empty());
}

BOOST_AUTO_TEST_CASE( add_many_and_find ) {
	predicate P;

	const char * filters[] = {
		"000000000000010000000100000000000000100000001000100000000000000000000000000000000000000000000000000100000000000000000000000100000000000000000001000000000000100000000000001000000000000101001000",
		"000000000000010000000000000000000000100000000000100000000010000000000010010000000000000100010000000000000000100000000000000000000000000000100000000100000010000101000000000110000010000100000010",
		"000000000001000000000001000100000000001000000010000000000000000001000000000000000100000000000000000000000000000010100000010000000011000000010000000010000000000000000010000000001000000000101000",
		"000010000100100000000000001000000100000000000000000000000000000000000000000101000000000000010000000000000000000000000000001000000100000000101000000000000010000000000000010000000000000000000000",
		"000110000010000100100000000000000000000000010000000000001001000000100000000000000000100010000000000000000000000000000000000000101000000000000011000000001000000000000100000000000001001000000010",
		"011000000010000000000000000000000001000000000000000000000000000000000001000010000000000000001000000000000010000000000010000000000000010000000100000000100000000000000000000000000000101000000000",
		"000010000000010000000000000010010000000000000001000100000000000000000000000000010000000000000000100000000000000000000000000000000000000000001000000000000000000000000000000000000000111000000000",
		"000000000000000000000000000000001100000000000000000100000000000000010000000010001000000000000000000000001000000000000000010000000001100000000000000000000000000000000000000000000100000100100001",
		"100000100000000000000010000000000000000000000000000000000010000010000000011010010000000101000000000000000000000000000000001001000000010100000000000000000000000000000000001010001100000000000000",
		"000000000000000100000000100000010000000010000000000010100000000000000000100000000000000000010011000101000000000100000000000001000000000001000000010100100000000100000000000000011000000000000000",
	};

	static const int N = sizeof(filters) / sizeof(const char *);

	predicate::node * nodes[N];

	for(int i = 0; i < N; ++i) {
		nodes[i] = P.add(filter_t(filters[i]), 1, i);
		for(int j = 0; j < i; ++j)
			BOOST_CHECK(nodes[i] != nodes[j]);
	}

	for(int i = 0; i < N; ++i) 
		BOOST_CHECK(P.find(filter_t(filters[i])) == nodes[i]);


	filter_t fX("100100011010010011111010000000000011100110100101001000100000010001010000110101011000001000100000000011100001100001000111010000000110000001001000000011010001100100010010000000000110000010000110");
	const predicate::node * nX = P.find(fX);

	BOOST_CHECK(nX == 0);
}

class node_set_filter_const_handler : public filter_const_handler {
public:
	virtual bool handle_filter(const filter_t & filter, const predicate::node & n) {
		nodes.insert(&n);
		return false;
	};

	node_set_filter_const_handler(std::set<const predicate::node *> & s): nodes(s) {};

private:
	std::set<const predicate::node *> & nodes;
};

class node_set_filter_handler : public filter_handler {
public:
	virtual bool handle_filter(const filter_t & filter, predicate::node & n) {
		nodes.insert(&n);
		return false;
	};

	node_set_filter_handler(std::set<const predicate::node *> & s): nodes(s) {};

private:
	std::set<const predicate::node *> & nodes;
};

BOOST_AUTO_TEST_CASE( subsets ) {
	predicate P;

	const char * filters[] = {
		"000000000000010000000100000000000000100000001000100000000000000000000000000000000000000000000000000100000000000000000000000100000000000000000001000000000000100000000000001000000000000101001000",
		"000000000000010000000000000000000000100000000000100000000010000000000010010000000000000100010000000000000000100000000000000000000000000000100000000100000010000101000000000110000010000100000010",
		"000000000001000000000001000100000000001000000010000000000000000001000000000000000100000000000000000000000000000010100000010000000011000000010000000010000000000000000010000000001000000000101000",
		"000010000100100000000000001000000100000000000000000000000000000000000000000101000000000000010000000000000000000000000000001000000100000000101000000000000010000000000000010000000000000000000000",
		"000110000010000100100000000000000000000000010000000000001001000000100000000000000000100010000000000000000000000000000000000000101000000000000011000000001000000000000100000000000001001000000010",
		"011000000010000000000000000000000001000000000000000000000000000000000001000010000000000000001000000000000010000000000010000000000000010000000100000000100000000000000000000000000000101000000000",
		"000010000000010000000000000010010000000000000001000100000000000000000000000000010000000000000000100000000000000000000000000000000000000000001000000000000000000000000000000000000000111000000000",
		"000000000000000000000000000000001100000000000000000100000000000000010000000010001000000000000000000000001000000000000000010000000001100000000000000000000000000000000000000000000100000100100001",
		"100000100000000000000010000000000000000000000000000000000010000010000000011010010000000101000000000000000000000000000000001001000000010100000000000000000000000000000000001010001100000000000000",
		"000000000000000100000000100000010000000010000000000010100000000000000000100000000000000000010011000101000000000100000000000001000000000001000000010100100000000100000000000000011000000000000000",
	};

	static const int N = sizeof(filters) / sizeof(const char *);

	predicate::node * nodes[N];

	for(int i = 0; i < N; ++i) {
		filter_t f(filters[i]);
		nodes[i] = P.add(f, 1, i);
		for(int j = 0; j < i; ++j)
			BOOST_CHECK(nodes[i] != nodes[j]);
	}

	const char * query_filters[] = {
		"000000000000010000000100000000000000100000001000100000000000000000000000000000000000000000000000000100000000000000000000000100000000000000000001000000000000100000000000000000000000000101001000",
		"100000100000010001000010000000000000100000000000100011000010000010000010011010010000000101010000000000000001100000000000001001000000010100100000000100000010000111000000000111000110000100000010",
		"000000000000010000000000000000000000100000000000100000000010000000000010010000000000000100010000000000000000100000000000000000000000000000100000000100000010000101000000000110000010000100000010",
		"000000000001000000000001000100000000001000000010000000000000000001000000000000000100000000000000000000000000000010100000010000000011000000010000000010000000000000000010000000001000000000101000",
		"100010100111110101100010000000000000100000011111010001111101000001100001001101001000100010101000000000000000110000000000001100100100001010111000000001000011100001110000111000001110000100000010",
		"000010000100100000000000001000000100000000000000000000000000000000000000000101000000000000010000000000000000000000000000001000000100000000101000000000000010000000000000010000000000000000000000",
		"000110000010000100100000000000000000000000010000000000001001000000100000000000000000100010000000000000000000000000000000000000101000000000000011000000001000000000000100000000000001001000000010",
		"011000000010000000000000000000000001000000000000000000000000000000000001000010000000000000001000000000000010000000000010000000000000010000000100000000100000000000000000000000000000101000000000",
		"000010000000010000000000000010010000000000000001000100000000000000000000000000010000000000000000100000000000000000000000000000000000000000001000000000000000000000000000000000000000111000000000",
		"000000000000000000000000000000001100000000000000000100000000000000010000000010001000000000000000000000001000000000000000010000000001100000000000000000000000000000000000000000000100000100100001",
		"000000100000000000000010001010011100000000000001000100000010000010010011001110011000000101000011100000001111000000000000011001000001110100001100000000000000000000000000001010001100111100110001",
		"000000000000000100000000100000010000000010000000000010100000000000000000100000000000000000010011000101000000000100000000000001000000000001000000010100100000000100000000000000011000000000000000",
		"111111111111111111111111111111111111111011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
	};

	static const int M = sizeof(query_filters) / sizeof(const char *);

	for(int i = 0; i < M; ++i) {
		filter_t qf(query_filters[i]);
		std::set<const predicate::node *> matching_nodes;
		node_set_filter_const_handler handler(matching_nodes);

		P.find_subsets_of(qf, handler);

		for(int j = 0; j < N; ++j) {
			if (filter_t(filters[j]).subset_of(qf)) {
				BOOST_CHECK(matching_nodes.count(nodes[j]) == 1);
				matching_nodes.erase(nodes[j]);
			}
		}

		BOOST_CHECK(matching_nodes.empty());
	}

	for(int i = 0; i < M; ++i) {
		filter_t qf(query_filters[i]);
		std::set<const predicate::node *> matching_nodes;
		node_set_filter_handler handler(matching_nodes);

		P.find_subsets_of(qf, handler);

		for(int j = 0; j < N; ++j) {
			if (filter_t(filters[j]).subset_of(qf)) {
				BOOST_CHECK(matching_nodes.count(nodes[j]) == 1);
				matching_nodes.erase(nodes[j]);
			}
		}

		BOOST_CHECK(matching_nodes.empty());
	}
}

BOOST_AUTO_TEST_CASE( supersets ) {
	predicate P;

	const char * filters[] = {
		"000000000000010000000100000000000000100000001000100000000000000000000000000000000000000000000000000100000000000000000000000100000000000000000001000000000000100000000000001000000000000101001000",
		"000000000000010000000000000000000000100000000000100000000010000000000010010000000000000100010000000000000000100000000000000000000000000000100000000100000010000101000000000110000010000100000010",
		"000000000001000000000001000100000000001000000010000000000000000001000000000000000100000000000000000000000000000010100000010000000011000000010000000010000000000000000010000000001000000000101000",
		"000010000100100000000000001000000100000000000000000000000000000000000000000101000000000000010000000000000000000000000000001000000100000000101000000000000010000000000000010000000000000000000000",
		"000110000010000100100000000000000000000000010000000000001001000000100000000000000000100010000000000000000000000000000000000000101000000000000011000000001000000000000100000000000001001000000010",
		"011000000010000000000000000000000001000000000000000000000000000000000001000010000000000000001000000000000010000000000010000000000000010000000100000000100000000000000000000000000000101000000000",
		"000010000000010000000000000010010000000000000001000100000000000000000000000000010000000000000000100000000000000000000000000000000000000000001000000000000000000000000000000000000000111000000000",
		"000000000000000000000000000000001100000000000000000100000000000000010000000010001000000000000000000000001000000000000000010000000001100000000000000000000000000000000000000000000100000100100001",
		"100000100000000000000010000000000000000000000000000000000010000010000000011010010000000101000000000000000000000000000000001001000000010100000000000000000000000000000000001010001100000000000000",
		"000000000000000100000000100000010000000010000000000010100000000000000000100000000000000000010011000101000000000100000000000001000000000001000000010100100000000100000000000000011000000000000000",
	};

	static const int N = sizeof(filters) / sizeof(const char *);

	predicate::node * nodes[N];

	for(int i = 0; i < N; ++i) {
		filter_t f(filters[i]);
		nodes[i] = P.add(f, 1, i);
		for(int j = 0; j < i; ++j)
			BOOST_CHECK(nodes[i] != nodes[j]);
	}

	const char * query_filters[] = {
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
		"000000000000010000000100000000000000100000001000100000000000000000000000000000000000000000000000000100000000000000000000000100000000000000000001000000000000100000000000001000000000000101001000",
		"000000000000010000000000000000000000100000000000100000000010000000000010010000000000000100010000000000000000100000000000000000000000000000100000000100000010000101000000000110000010000100000010",
		"000000000001000000000001000100000000001000000010000000000000000001000000000000000100000000000000000000000000000010100000010000000011000000010000000010000000000000000010000000001000000000101000",
		"000010000100100000000000001000000100000000000000000000000000000000000000000101000000000000010000000000000000000000000000001000000100000000101000000000000010000000000000010000000000000000000000",
		"000110000010000100100000000000000000000000010000000000001001000000100000000000000000100010000000000000000000000000000000000000101000000000000011000000001000000000000100000000000001001000000010",
		"011000000010000000000000000000000001000000000000000000000000000000000001000010000000000000001000000000000010000000000010000000000000010000000100000000100000000000000000000000000000101000000000",
		"000010000000010000000000000010010000000000000001000100000000000000000000000000010000000000000000100000000000000000000000000000000000000000001000000000000000000000000000000000000000111000000000",
		"000000000000000000000000000000001100000000000000000100000000000000010000000010001000000000000000000000001000000000000000010000000001100000000000000000000000000000000000000000000100000100100001",
		"100000100000000000000010000000000000000000000000000000000010000010000000011010010000000101000000000000000000000000000000001001000000010100000000000000000000000000000000001010001100000000000000",
		"000000000000000100000000100000010000000010000000000010100000000000000000100000000000000000010011000101000000000100000000000001000000000001000000010100100000000100000000000000011000000000000000",
		"000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
		"000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
		"000000000001000000000001000100000000001000000010000000000000000001000000000000000100000000000000000000000000000010100000010000000011000000010000000010000000000000000010000000001000000000101000",
		"000010000100100000000000001000000100000000000000000000000000000000000000000101000000000000010000000000000000000000000000001000000100000000101000000000000010000000000000010000000000000000000000",
		"111111111111111111111111111111111111111011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
	};

	static const int M = sizeof(query_filters) / sizeof(const char *);

	for(int i = 0; i < M; ++i) {
		filter_t qf(query_filters[i]);
		std::set<const predicate::node *> matching_nodes;
		node_set_filter_const_handler handler(matching_nodes);

		P.find_supersets_of(qf, handler);

		for(int j = 0; j < N; ++j) {
			if (qf.subset_of(filter_t(filters[j]))) {
				BOOST_CHECK(matching_nodes.count(nodes[j]) == 1);
				matching_nodes.erase(nodes[j]);
			}
		}

		BOOST_CHECK(matching_nodes.empty());
	}

	for(int i = 0; i < M; ++i) {
		filter_t qf(query_filters[i]);
		std::set<const predicate::node *> matching_nodes;
		node_set_filter_handler handler(matching_nodes);

		P.find_supersets_of(qf, handler);

		for(int j = 0; j < N; ++j) {
			if (qf.subset_of(filter_t(filters[j]))) {
				BOOST_CHECK(matching_nodes.count(nodes[j]) == 1);
				matching_nodes.erase(nodes[j]);
			}
		}

		BOOST_CHECK(matching_nodes.empty());
	}
}

BOOST_AUTO_TEST_SUITE_END()

