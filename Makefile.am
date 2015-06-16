SUBDIRS = . test

ACLOCAL_AMFLAGS = -I m4 

AM_CFLAGS = $(CFLAGS_prof) 
AM_CXXFLAGS = -pthread -std=c++0x $(CXXFLAGS_prof)
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

bin_PROGRAMS = router gen gen2 split_prefixes \
		cpu_gpu_matcher test_matcher \
		analyze_workload convert_workload sample_workload split_workload \
		subset_counter

gen_SOURCES = gen.cc params.h

gen2_SOURCES = gen2.cc predicate.hh

router_SOURCES = main2.cc predicate.hh timing.hh \
		router.cc router.hh	
router_LDADD = $(LIBS_prof) libpredicate.la

if WORKING_NVCC
cpu_gpu_matcher_SOURCES = packet.hh io_util.hh fib.hh bitvector.hh parameters.hh gpu.hh \
			cpu_gpu_matcher.cc fib.cc packet.cc \
			front_end.cc front_end.hh back_end.cc back_end.hh \
			gpu.cu 

#cpu_gpu_matcher_LDFLAGS = -lcudart
cpu_gpu_matcher_LDFLAGS = -L/usr/local/cuda-7.0/lib64 -lcudart
else
##
## this can only work with --disable-backend
##
cpu_gpu_matcher_SOURCES = packet.hh io_util.hh fib.hh bitvector.hh parameters.hh gpu.hh \
			cpu_gpu_matcher.cc fib.cc packet.cc \
			front_end.cc front_end.hh back_end.cc back_end.hh
endif

test_matcher_SOURCES = test_matcher.cc parameters.hh packet.cc fib.cc io_util.hh

split_prefixes_SOURCES = split_prefixes.cc bv.hh predicates.hh

analyze_workload_SOURCES = analyze_workload.cc packet.cc fib.cc \
			packet.hh fib.hh bitvector.hh

convert_workload_SOURCES = convert_workload.cc fib.cc packet.cc \
			fib.hh packet.hh bitvector.hh

sample_workload_SOURCES = sample_workload.cc fib.cc packet.cc \
			fib.hh packet.hh bitvector.hh

split_workload_SOURCES = split_workload.cc packet.cc fib.cc \
			packet.hh fib.hh bitvector.hh

subset_counter_SOURCES = filter_set_driver.cc trie_filter_set.cc \
			filter_set.hh packet.hh packet.cc bitvector.hh \
			fib.hh fib.cc
