SUBDIRS = . test

ACLOCAL_AMFLAGS = -I m4

bin_PROGRAMS = predicate gen

gen_SOURCES = gen.cc params.h
gen_CXXFLAGS = -std=c++0x

predicate_SOURCES = predicate.cc predicate.hh bv192.hh main.cc
predicate_CXXFLAGS = -O4 -std=c++0x -Wall @CXXFLAGS_prof@
predicate_LDFLAGS = @LIBS_prof@

AM_CFLAGS = -Wall @CFLAGS_prof@
AM_CXXFLAGS = -Wall @CXXFLAGS_prof@
AM_LDFLAGS = @LDFLAGS_prof@
LIBS =  @LIBS@ @LIBS_prof@ 
