ACLOCAL_AMFLAGS = -I m4

bin_PROGRAMS = predicate gen

gen_SOURCES = gen.cc params.h
gen_CXXFLAGS = -std=c++0x

predicate_SOURCES = predicate.cc bv192.h
predicate_CXXFLAGS = -O2 -std=c++11 -Wall @CXXFLAGS_prof@
predicate_LDFLAGS = @LIBS_prof@

AM_CFLAGS = -Wall @CFLAGS_prof@
AM_CXXFLAGS = -Wall @CXXFLAGS_prof@
AM_LDFLAGS = @LDFLAGS_prof@
LIBS =  @LIBS@ @LIBS_prof@ 
