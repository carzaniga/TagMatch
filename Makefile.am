SUBDIRS = . test

SUFFIXES = .cu .cc .hh

ACLOCAL_AMFLAGS = -I m4

AM_CFLAGS = $(CFLAGS_prof)
AM_CXXFLAGS = $(CXXFLAGS_prof)
AM_LDFLAGS = $(LDFLAGS_prof)

#
# CUDA build rules and variables
#
if WORKING_NVCC
NVCCFLAGS += -std=c++11 -D_MWAITXINTRIN_H_INCLUDED

NVCC_NONPIC_COMPILE = $(NVCC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	$(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(NVCCFLAGS)

NVCC_PIC_COMPILE = $(NVCC_NONPIC_COMPILE) -Xcompiler -fPIC -DPIC

.cu.o:
	$(AM_V_CXX)$(NVCC_NONPIC_COMPILE) -c -o $@ $<

.cu.lo:
	$(AM_V_CXX)$(NVCC_NONPIC_COMPILE) -c -o $*.o $<
	$(AM_V_at)$(NVCC_PIC_COMPILE) -c -o .libs/$*.o $<
	$(AM_V_at)( echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool (GNU libtool) 2.4.6 Debian-2.4.6-0.1"; \
	  echo "pic_object='.libs/$*.o'"; \
	  echo "non_pic_object='$*.o'" ) > $@

endif

lib_LTLIBRARIES = libtagmatch.la
libtagmatch_la_SOURCES = fib.cc query.cc \
		query.hh io_util.hh fib.hh bitvector.hh gpu.hh parameters.hh \
		front_end.cc front_end.hh back_end.cc back_end.hh partitioning.hh \
		gpu.cu partitioner_gpu.cu partitioning.cc \
		tagmatch.cc tagmatch.hh

libtagmatch_la_LDFLAGS = -lcudart

bin_PROGRAMS = test_partitioning_client tagmatch \
		test_matcher patricia_matcher multi_patricia_matcher compact_patricia_matcher \
		analyze_workload convert_workload \
		sample_workload split_workload gen2

test_partitioning_client_SOURCES = tagmatch.hh test_partitioning_client.cc
test_partitioning_client_LDADD = libtagmatch.la
test_partitioning_client_LDFLAGS = -lcudart

tagmatch_SOURCES = tagmatch.hh tagmatch_main.cc
tagmatch_LDADD = libtagmatch.la
tagmatch_LDFLAGS = -lcudart

analyze_workload_SOURCES = analyze_workload.cc
analyze_workload_LDADD = libtagmatch.la

convert_workload_SOURCES = convert_workload.cc
convert_workload_LDADD = libtagmatch.la

sample_workload_SOURCES = sample_workload.cc
sample_workload_LDADD = libtagmatch.la

split_workload_SOURCES = split_workload.cc
split_workload_LDADD = libtagmatch.la

test_matcher_SOURCES = test_matcher.cc parameters.hh query.hh io_util.hh fib.hh filter.hh
test_matcher_LDADD = libtagmatch.la

patricia_matcher_SOURCES = filter.hh bitvector.hh fib.hh query.hh io_util.hh \
			patricia_predicate.hh patricia_matcher.cc key_array.hh key_array.cc
patricia_matcher_LDADD = libtagmatch.la

compact_patricia_matcher_SOURCES = filter.hh bitvector.hh fib.hh query.hh io_util.hh \
			compact_patricia_predicate.hh key_array.hh \
			key_array.cc compact_patricia_matcher.cc
compact_patricia_matcher_LDADD = libtagmatch.la

multi_patricia_matcher_SOURCES = filter.hh bitvector.hh fib.hh query.hh io_util.hh \
			multi_patricia_predicate.hh key_array.hh \
			key_array.cc multi_patricia_matcher.cc
multi_patricia_matcher_LDADD = libtagmatch.la

gen2_SOURCES = gen2.cc
