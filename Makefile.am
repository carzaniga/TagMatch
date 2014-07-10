SUBDIRS = . test

ACLOCAL_AMFLAGS = -I m4

AM_CFLAGS = $(CFLAGS_prof)
AM_CXXFLAGS = $(CXXFLAGS_prof)
AM_LDFLAGS = $(LDFLAGS_prof)

NVCC=nvcc

noinst_LTLIBRARIES = libpredicate.la
libpredicate_la_SOURCES = predicate.cc predicate.hh bv192.hh bv.hh

bin_PROGRAMS = predicate predicate_old gen gen2 msb_partition split_prefixes \
		cpu_gpu_matcher

gen_SOURCES = gen.cc params.h

gen2_SOURCES = gen2.cc predicate.hh

predicate_SOURCES = main.cc predicate.hh timing.hh timing.cc
predicate_LDADD = $(LIBS_prof) libpredicate.la

predicate_old_SOURCES = predicate.hh predicate_old.cc \
	filter.hh filter.cc \
	timing.hh timing.cc allocator.hh allocator.cc
predicate_old_LDADD = $(LIBS_prof) 

msb_partition_SOURCES = msb_partition.cc main_GPU.cc main_GPU.h
msb_partition_CXXFLAGS = -pthread #-Wl,--no-as-needed
msb_partition_LDADD = GPU_matching.$(OBJEXT)
msb_partition_LDFLAGS = -L/usr/local/cuda/lib -lcudart

CUDA_FLAGS=--ptxas-options="-v" -arch sm_21 -O2

.cu.$(OBJEXT):
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

cpu_gpu_matcher_SOURCES = packet.hh parameters.hh free_list.hh \
			cpu_gpu_matcher.cc \
			front_end.cc front_end.hh back_end.cc back_end.hh 
cpu_gpu_matcher_CXXFLAGS = -pthread #-Wl,--no-as-needed
cpu_gpu_matcher_LDADD = GPU_matching.$(OBJEXT)
cpu_gpu_matcher_LDFLAGS = -L/usr/local/cuda/lib -lcudart

split_prefixes_SOURCES = split_prefixes.cc bv.hh predicates.hh

