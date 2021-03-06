# -*- Makefile -*-
#
#  This file is part of Siena, a wide-area event notification system.
#  See http://www.inf.usi.ch/carzaniga/siena/
#
#  Author: Antonio Carzaniga
#  See the file AUTHORS for full details. 
#
#  Copyright (C) 2010-2010 Antonio Carzaniga
#
#  Siena is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  Siena is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with Siena.  If not, see <http://www.gnu.org/licenses/>.
#

AM_CPPFLAGS = -I$(top_srcdir)

AM_CFLAGS = @CFLAGS_prof@
AM_CXXFLAGS = @CXXFLAGS_prof@
AM_LDFLAGS = @LDFLAGS_prof@
LIBS =  @LIBS@ @LIBS_prof@ 

check_PROGRAMS = 
UNIT_TESTS = 

if USE_BOOST_UNIT_TEST_FRAMEWORK
check_PROGRAMS += test_io test_bitvector \
	test_patricia_predicate test_multi_patricia_predicate \
	test_compact_patricia_predicate \
	test_partitioning test_tagmatch
UNIT_TESTS += test_io test_bitvector \
	test_patricia_predicate test_multi_patricia_predicate \
	test_compact_patricia_predicate \
	test_partitioning test_tagmatch
endif

test_io_SOURCES = test_io.cc
test_io_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB)

test_bitvector_SOURCES = test_bitvector.cc
test_bitvector_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB)

test_patricia_predicate_SOURCES = test_patricia_predicate.cc
test_patricia_predicate_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB)

test_multi_patricia_predicate_SOURCES = test_multi_patricia_predicate.cc
test_multi_patricia_predicate_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB)

test_compact_patricia_predicate_SOURCES = test_compact_patricia_predicate.cc
test_compact_patricia_predicate_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB)

test_partitioning_SOURCES = test_partitioning.cc
test_partitioning_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB) ../libtagmatch.la

test_tagmatch_SOURCES = test_tagmatch.cc
test_tagmatch_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB) ../libtagmatch.la

TESTS = $(UNIT_TESTS)

TESTS_ENVIRONMENT = BOOST_TEST_LOG_LEVEL=nothing

EXTRA_DIST = 

test: run-unit-tests run-integration-tests 

RUNNER_TEST_ENVIRONMENT = BOOST_TEST_REPORT_LEVEL=nothing
run-unit-tests: $(UNIT_TESTS)
	@list='$(UNIT_TESTS)'; \
	echo "Running unit tests..."; \
        if test -n "$$list"; then \
          for tst in $$list; do \
	    $(RUNNER_TEST_ENVIRONMENT) ./$$tst || exit 1; \
	  done; \
	fi
