SUBDIRS = . test

ACLOCAL_AMFLAGS = -I m4

AM_CFLAGS = $(CFLAGS_prof)
AM_CXXFLAGS = $(CXXFLAGS_prof)
AM_LDFLAGS = $(LDFLAGS_prof)

noinst_LTLIBRARIES = libpredicate.la
libpredicate_la_SOURCES = predicate.cc predicate.hh bv192.hh

bin_PROGRAMS = predicate predicate_old gen gen2

gen_SOURCES = gen.cc params.h

gen2_SOURCES = gen2.cc predicate.hh

predicate_SOURCES = main.cc predicate.hh timing.hh timing.cc
predicate_LDADD = $(LIBS_prof) libpredicate.la

predicate_old_SOURCES = predicate.hh predicate_old.cc \
	filter.hh filter.cc \
	timing.hh timing.cc allocator.hh allocator.cc
predicate_old_LDADD = $(LIBS_prof) 
