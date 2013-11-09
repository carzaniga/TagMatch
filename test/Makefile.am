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
AM_CXXFLAGS = -Wall -std=c++0x @CXXFLAGS_prof@
AM_CFLAGS = -Wall @CFLAGS_prof@
AM_LDFLAGS = @LDFLAGS_prof@
LIBS =  @LIBS@ @LIBS_prof@ 

check_PROGRAMS = 
UNIT_TESTS = 

if USE_BOOST_UNIT_TEST_FRAMEWORK
check_PROGRAMS += test_bv192 test_predicate
UNIT_TESTS += test_bv192 test_predicate
endif

test_bv192_SOURCES = test_bv192.cc
test_bv192_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB)
test_bv192_LDFLAGS = @LIBS_prof@

test_predicate_SOURCES = test_predicate.cc ../predicate.cc
test_predicate_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB)
test_predicate_LDFLAGS = @LIBS_prof@


TESTS = $(UNIT_TESTS)

TESTS_ENVIRONMENT = BOOST_TEST_LOG_LEVEL=nothing BOOST_TEST_REPORT_LEVEL=no 

EXTRA_DIST = 

test: run-unit-tests run-integration-tests 

run-unit-tests: $(UNIT_TESTS)
	@list='$(UNIT_TESTS)'; \
	echo "Running unit tests..."; \
        if test -n "$$list"; then \
          for tst in $$list; do \
	    ./$$tst || exit 1; \
	  done; \
	fi
