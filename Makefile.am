SUBDIRS = . test

ACLOCAL_AMFLAGS = -I m4

if DISABLE_COMPILER_OPTIMIZATIONS
AM_CFLAGS =  -Wall -O0 @CFLAGS_prof@
AM_CXXFLAGS =  -std=c++0x  -Wall -O0 @CXXFLAGS_prof@
else
AM_CFLAGS =  -Wall -O2 @CFLAGS_prof@
AM_CXXFLAGS =  -std=c++0x  -Wall -O2  @CXXFLAGS_prof@
endif

AM_LDFLAGS = @LDFLAGS_prof@
LIBS =  @LIBS@ @LIBS_prof@ 

noinst_LTLIBRARIES = libpredicate.la 
libpredicate_la_SOURCES = predicate.cc predicate.hh bv192.hh 

bin_PROGRAMS = predicate gen

gen_SOURCES = gen.cc params.h

predicate_SOURCES = main.cc predicate.hh
predicate_LDADD = libpredicate.la 


