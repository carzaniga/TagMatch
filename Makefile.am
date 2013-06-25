ACLOCAL_AMFLAGS = -I m4

bin_PROGRAMS = predicate gen
gen_SOURCES = gen.cc params.h
gen_CXXFLAGS = -std=c++0x
predicate_SOURCES = predicate.cc predicate.h filter.h filter.cc timing.h timing.cc allocator.h ft_allocator.cc
predicate_CXXFLAGS = -std=c++0x -O4
AM_CXXFLAGS = $(BOOST_CPPFLAGS)
