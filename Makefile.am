SUBDIRS = . test

ACLOCAL_AMFLAGS = -I m4 

AM_CFLAGS = $(CFLAGS_prof) 
AM_CXXFLAGS = -pthread -std=c++0x $(CXXFLAGS_prof)
AM_LDFLAGS = $(LDFLAGS_prof)

#
# CUDA build variables
#
if WORKING_NVCC
NVCCFLAGS += -std=c++11 -D_MWAITXINTRIN_H_INCLUDED
#.cu.$(OBJEXT):
#	$(NVCC) $(NVCCFLAGS) -o $@ -c $<    
.cu.lo:
	./compilerCudaStuff.sh $@
endif

#$(NVCC) -Xcompiler -fPIC -DPIC $(NVCCFLAGS) -c $< -o $@
#	$(LIBTOOL) --tag=CC --mode=compile $(NVCC) $(NVCCFLAGS) -o $@  -c $< 

noinst_LTLIBRARIES = libpredicate.la
libpredicate_la_SOURCES = predicate.cc predicate.hh bv192.hh bv.hh

lib_LTLIBRARIES = libtagmatch.la
libtagmatch_la_SOURCES = fib.cc packet.cc \
		packet.hh io_util.hh fib.hh bitvector.hh parameters.hh \
		front_end.cc front_end.hh back_end.cc back_end.hh \
		gpu.cu partitioner_gpu.cu partitioner.cc \
		tagmatch.cc tagmatch.hh

libtagmatch_la_LDFLAGS = -lcudart
# partitioner.cu 

bin_PROGRAMS = test_partitioning_client \
		allinone

gen_SOURCES = gen.cc params.h

gen2_SOURCES = gen2.cc predicate.hh

router_SOURCES = main2.cc predicate.hh timing.hh \
		router.cc router.hh	
router_LDADD = $(LIBS_prof) libpredicate.la

test_partitioning_client_SOURCES = tagmatch.hh test_partitioning_client.cc
test_partitioning_client_LDADD = libtagmatch.la
test_partitioning_client_LDFLAGS = -lcudart

allinone_SOURCES = tagmatch.hh allinone.cc
allinone_LDADD = libtagmatch.la
allinone_LDFLAGS = -lcudart

if WORKING_NVCC
cpu_gpu_matcher_SOURCES = tagmatch.hh cpu_gpu_matcher.cc   
			
cpu_gpu_matcher_LDADD = libtagmatch.la

cpu_gpu_matcher_LDFLAGS = -lcudart
else
##
## this can only work with --disable-backend
##
#cpu_gpu_matcher_SOURCES = packet.hh io_util.hh fib.hh bitvector.hh parameters.hh gpu.hh \
#			cpu_gpu_matcher.cc fib.cc packet.cc tagmatch.cc tagmatch.hh \
#			front_end.cc front_end.hh back_end.cc back_end.hh
endif

#test_matcher_SOURCES = test_matcher.cc parameters.hh packet.cc fib.cc io_util.hh

#split_prefixes_SOURCES = split_prefixes.cc bv.hh predicates.hh

#analyze_workload_SOURCES = analyze_workload.cc packet.cc fib.cc \
#			packet.hh fib.hh bitvector.hh

#convert_workload_SOURCES = convert_workload.cc fib.cc packet.cc \
#			fib.hh packet.hh bitvector.hh

#sample_workload_SOURCES = sample_workload.cc fib.cc packet.cc \
#			fib.hh packet.hh bitvector.hh

#split_workload_SOURCES = split_workload.cc packet.cc fib.cc \
#			packet.hh fib.hh bitvector.hh

#subset_counter_SOURCES = filter_set_driver.cc trie_filter_set.cc \
#			filter_set.hh packet.hh packet.cc bitvector.hh \
#			fib.hh fib.cc

#balanced_partitioning_SOURCES = balanced_partitioning.cc packet.cc fib.cc \
#			packet.hh fib.hh bitvector.hh

#if WORKING_NVCC
#NVCCFLAGS += -std=c++11 -D_MWAITXINTRIN_H_INCLUDED
#balanced_partitioning_cuda_SOURCES = balanced_partitioning_cuda.cu packet.cc fib.cc \
#			packet.hh fib.hh bitvector.hh
#balanced_partitioning_cuda_LDFLAGS = -lcudart
#endif
