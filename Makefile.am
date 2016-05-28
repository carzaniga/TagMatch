SUBDIRS = . test

ACLOCAL_AMFLAGS = -I m4 

AM_CFLAGS = $(CFLAGS_prof) 
AM_CXXFLAGS = -pthread -std=c++11 $(CXXFLAGS_prof)
AM_LDFLAGS = $(LDFLAGS_prof)

#
# CUDA build variables
#
# These things should be checked and set through some autocont
# parameters and tests.
#
if WORKING_NVCC
.cu.$(OBJEXT):
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

noinst_LTLIBRARIES = libpredicate.la
libpredicate_la_SOURCES = predicate.cc predicate.hh bv192.hh bv.hh

bin_PROGRAMS = gen gen2 split_prefixes router \
		cpu_gpu_matcher test_matcher patricia_matcher multi_patricia_matcher \
		compact_patricia_matcher \
		analyze_workload convert_workload sample_workload split_workload

gen_SOURCES = gen.cc

gen2_SOURCES = gen2.cc predicate.hh bv192.hh bv.hh

router_SOURCES = main2.cc predicate.hh timing.hh \
		router.cc router.hh	
router_LDADD = $(LIBS_prof) libpredicate.la

if WORKING_NVCC
cpu_gpu_matcher_SOURCES = packet.hh io_util.hh fib.hh bitvector.hh parameters.hh gpu.hh \
			filter.hh routing.hh \
			cpu_gpu_matcher.cc fib.cc packet.cc \
			front_end.cc front_end.hh back_end.cc back_end.hh \
			gpu.cu 

cpu_gpu_matcher_LDFLAGS = -lcudart
else
##
## this can only work with --disable-backend
##
cpu_gpu_matcher_SOURCES = packet.hh io_util.hh fib.hh bitvector.hh parameters.hh gpu.hh \
			filter.hh routing.hh \
			cpu_gpu_matcher.cc fib.cc packet.cc \
			front_end.cc front_end.hh back_end.cc back_end.hh
endif

test_matcher_SOURCES = test_matcher.cc parameters.hh packet.hh io_util.hh fib.hh \
			filter.hh routing.hh \
			packet.cc fib.cc

patricia_matcher_SOURCES = filter.hh bitvector.hh routing.hh fib.hh packet.hh io_util.hh \
			patricia_predicate.hh tip_array.hh \
			tip_array.cc fib.cc packet.cc \
			patricia_matcher.cc

compact_patricia_matcher_SOURCES = filter.hh bitvector.hh routing.hh fib.hh packet.hh io_util.hh \
			compact_patricia_predicate.hh tip_array.hh \
			tip_array.cc fib.cc packet.cc \
			compact_patricia_matcher.cc

multi_patricia_matcher_SOURCES = filter.hh bitvector.hh routing.hh fib.hh packet.hh io_util.hh \
				multi_patricia_predicate.hh tip_array.hh \
			tip_array.cc fib.cc packet.cc \
			multi_patricia_matcher.cc

split_prefixes_SOURCES = split_prefixes.cc filter.hh routing.hh bitvector.hh 

analyze_workload_SOURCES = analyze_workload.cc packet.cc fib.cc \
			packet.hh fib.hh bitvector.hh filter.hh routing.hh

convert_workload_SOURCES = convert_workload.cc fib.cc packet.cc \
			fib.hh packet.hh bitvector.hh filter.hh routing.hh

sample_workload_SOURCES = sample_workload.cc fib.cc packet.cc \
			fib.hh packet.hh bitvector.hh filter.hh routing.hh

split_workload_SOURCES = split_workload.cc packet.cc fib.cc \
			packet.hh fib.hh bitvector.hh filter.hh routing.hh
