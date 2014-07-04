SUBDIRS = . test

ACLOCAL_AMFLAGS = -I m4

AM_CFLAGS = $(CFLAGS_prof)
AM_CXXFLAGS = $(CXXFLAGS_prof)
AM_LDFLAGS = $(LDFLAGS_prof)

NVCC=nvcc

noinst_LTLIBRARIES = libpredicate.la
libpredicate_la_SOURCES = predicate.cc predicate.hh bv192.hh bv.hh

bin_PROGRAMS = predicate predicate_old gen gen2 msb_partition split_prefixes

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

CUDA_FLAGS=--ptxas-options="-v" -arch sm_21

.cu.$(OBJEXT):
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

# GPU_matching.$(OBJEXT): GPU_matching.cu GPU_matching.h
# 	$(NVCC) --ptxas-options="-v" $(CUDA_FLAGS) -c GPU_matching.cu -o $@

#libgpumatching_la_SOURCES = GPU_matching.cc GPU_matching.h
#libgpumatching_la_CXX = $(NVCC)
#libgpumatching_la_CXXFLAGS = --ptxas-options="-v" -arch sm_30

split_prefixes_SOURCES = split_prefixes.cc bv.hh predicates.hh

