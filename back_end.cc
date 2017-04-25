#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef SAFE_FILTER_COPY
#include <cstring>
#endif
#include <cassert>
#include <climits>
#include <vector>
#include <map>
#include <atomic>
#include <cstdint>
#include <algorithm>
#include <time.h>
#include <mutex>

#include "tagmatch.hh"
#include "gpu.hh"
#include "back_end.hh"
#include "fib.hh"
#include "filter.hh"

#define SORT_FILTERS 0

using std::vector;
using std::map;
using std::atomic;

// GENERAL DESIGN:
//
// The back-end consists of a dictionary structure, built once and
// then used repeatedly for matching.  The actual FIB resides mostly
// on the device (GPU) but the host also holds some meta-data used to
// access the proper FIB structures.
//
// The back-end works in two states.  Initially, the back-end is
// configured incrementally through the back_end::add_filter method.
// We store this incrementally recorded data structure using a
// temporary FIB structure on the host (CPU).  Then, once we are done
// adding filters, we can compile the actual FIB and then get rid of
// the temporaty FIB.
//

// We use a fixed numbed of GPUs.  This is the number of GPUs in use
// by the back-end.  The back-end is active when this number is
// greater than 0, and otherwise it is inactive.
//
static unsigned int gpus_in_use = 0;
unsigned int back_end::gpu_count() {
	return gpus_in_use;
}

// we store filters on the gpu using 32-bit blocks.
//
static const unsigned int BACKEND_BLOCK_SIZE = (GPU_WORD_SIZE * CHAR_BIT);

// temporary part of the back-end FIB.  This maps a partition id into
// the vector of the filters in that partition.  We keep this as a
// separate and temporary map because we can get rid of it as soon as
// we copy everything into the GPU.
//
// However, in case we use the CPU-only backend, we do not throw away
// this fib, and instead we use it in a special cpu back-end matcher.
//
typedef map<partition_id_t, vector<filter_t> > tmp_fib_map;
static tmp_fib_map tmp_fib;

// Permanent (non-temporary) part of the back-end FIB
//
// Here we map a partition ID into a vector of vectors of keys.
//
typedef vector<tagmatch_key_t> keys_vector;
typedef vector<keys_vector> f_descr_vector_users;
typedef map<partition_id_t, f_descr_vector_users > tmp_fib_map_users;

static tmp_fib_map_users tmp_fib_users;

void back_end::add_filter(partition_id_t part, const filter_t & f,
						  keys_vector::const_iterator begin,
						  keys_vector::const_iterator end) {
	// we simply add this filter to our temporary table
	//
	tmp_fib[part].emplace_back(f);
	tmp_fib_users[part].emplace_back(begin, end);
}

void back_end::add_partition(partition_id_t id, const filter_t & prefix) { }

// ACTUAL FIB
//
// These are the data structures of the actual FIB, which include
// pointers to the actual FIB on the device, plus meta-data and
// buffers used to communicate with the device.
//
struct part_descr {
	unsigned int size;

	// main FIB map on the device.  This stores a pointer into the
	// device memory that represents the set of filters associated
	// with this partition.
	//
	uint32_t * fib;

	part_descr(): size(0), fib(0) {};

	void clear() {
		size = 0;

		if (fib) {
			gpu::release_memory(fib);
			fib = nullptr;
		}


	}
};

#ifndef BACK_END_IS_VOID
// we have a descriptor for each partition
//
static part_descr * dev_partitions[GPU_NUM];
static unsigned int dev_partitions_size = 0;

// The back end maintains a set of stream handles.  These are
// primarily buffers we use to transfer data to and from the GPU for
// matching.
//
struct stream_handle {
	// we maintain a list of free handles.  This is the pointer to the
	// next free handle.  More specifically, it is an index in the
	// stream_handles array.
	//
	unsigned char next;

	// stream identifier shared with the GPU
	//
	unsigned int stream;
	unsigned int gpu;

private:
	// these are the buffers we use to communicate with the GPU for
	// matching.  For each stream, we store the queries of each
	// batch here, together with their associated tree-interface
	// keys, one for each query, in a separate buffer.  We then
	// store the results that come back from the GPU.
	//
	uint8_t current_buf;
	uint32_t * host_queries[2];
	gpu_result * host_results[2];
	gpu_result * dev_results[2];

	uint32_t last_partition;
	uint32_t second_last_partition;
	tagmatch_query ** last_batch;
	tagmatch_query ** second_last_batch;
	unsigned int last_batch_size, second_last_batch_size;
	batch * last_batch_ptr, * second_last_batch_ptr;

public:
	void flip_buffers() {
		current_buf ^= 1;
	}

	uint32_t * current_host_queries() const { return host_queries[current_buf]; }

	gpu_result * current_dev_results() const {
		return dev_results[current_buf];
	}

	gpu_result * other_dev_results() const {
		return dev_results[current_buf ^ 1];
	}

	gpu_result * current_host_results() const {
		return host_results[current_buf];
	}

	gpu_result * other_host_results() const {
		return host_results[current_buf ^ 1];
	}

	tagmatch_query * current_results_query(unsigned int i) const {
		return second_last_batch[(host_results[current_buf]->keys[5*(i/4)] >> (8*(i % 4))) & 0xFF];
	}

	const keys_vector & current_results_keys(unsigned int i) const {
		unsigned int keys_idx = host_results[current_buf]->keys[5*(i/4) + 1 + (i % 4)];
		return tmp_fib_users[second_last_partition][keys_idx];
	}

	uint32_t current_results_count() const {
		return host_results[current_buf ^ 1]->count;
	}

	void shift_batches () {
		// Update the info about the staged computation; drop the second
		// last iteration and store the info about the current iteration
		// for the next cycles
		//
		second_last_batch = last_batch;
		second_last_batch_size = last_batch_size;
		second_last_batch_ptr = last_batch_ptr;
		second_last_partition = last_partition;
	}

	batch * flush_one_batch();

	void initialize(unsigned int stream_, unsigned int gpu_, unsigned int next_) {
		stream = stream_;
		gpu = gpu_;
		next = next_;

		gpu::set_device(gpu);

		// Here we initialize two buffers for each stream for each gpu
		// for the staged computation that is taking place in
		// process_batch
		//
		current_buf = 0;
		for (unsigned int f = 0; f < 2; f++) {
			host_queries[f] = gpu::allocate_host_pinned<uint32_t>(QUERIES_BATCH_SIZE*GPU_FILTER_WORDS);
			host_results[f] = gpu::allocate_host_pinned<gpu_result>(1);
			host_results[f]->count = 0;
			dev_results[f] = gpu::allocate<gpu_result>(1);
			gpu::async_set_zero(dev_results[f], sizeof(gpu_result),0,gpu);
		}
		last_batch = second_last_batch = nullptr;
		last_batch_ptr = second_last_batch_ptr = nullptr;
		last_batch_size = second_last_batch_size = 0;
	}

	void destroy() {
		gpu::set_device(gpu);
		for (unsigned int f = 0; f < 2; f++) {
			gpu::release_pinned_memory(host_queries[f]);
			gpu::release_pinned_memory(host_results[f]);
			gpu::release_memory(dev_results[f]);
		}
	}

	void async_copy_results_from_gpu() {
		gpu::async_get_results(current_host_results(), current_dev_results(),
							   current_results_count(),
							   stream, gpu);
	}

	batch * process_batch(partition_id_t part, tagmatch_query ** q, unsigned int count, batch * bp);
	batch * process_available_results();
};

batch * stream_handle::process_batch(partition_id_t part, tagmatch_query ** q, unsigned int q_count, batch * batch_ptr) {
	batch * res;
	flip_buffers();
	// We first copy every query (filter plus tree-interface pair)
	// over to the device.  To do that we first assemble the whole
	// thing in buffers here on the host, and then we copy those
	// buffers over to the device.
	//
	uint32_t * hq = current_host_queries();
	for (unsigned int i = 0; i < q_count; ++i)
		for (int j = 0; j < GPU_FILTER_WORDS; ++j)
			*hq++ = q[i]->filter.uint32_value(j);

	gpu::set_device(gpu);
	gpu::async_copy_queries(host_queries[current_buf], q_count * GPU_FILTER_WORDS, stream, gpu);
	gpu::run_kernel(dev_partitions[gpu][part].fib, dev_partitions[gpu][part].size,
					q_count, current_dev_results(), other_dev_results(), stream, gpu);

	res = process_available_results();

	uint32_t count = current_results_count();
	assert(count <= MAX_MATCHES);
	// copy results from current_dev into current_host
	gpu::async_get_results(current_host_results(), current_dev_results(), count, stream, gpu);
	// clear current_dev
	gpu::async_set_zero(current_dev_results(),  sizeof(uint32_t)*(1 + count +(count + 3)/4),
						stream, gpu);

	// Update the info about the staged computation; drop the second
	// last iteration and store the info about the current iteration
	// for the next cycles
	//
	shift_batches();

	last_batch = q;
	last_batch_size = q_count;
	last_batch_ptr = batch_ptr;
	last_partition = part;

	return res;
}

batch * stream_handle::process_available_results() {
	if (second_last_batch_ptr != nullptr) {
		gpu::set_device(gpu);
		gpu::syncOnResults(stream, gpu); // Wait for the data to be copied
		assert(current_results_count() <= MAX_MATCHES);
		for (unsigned int i = 0; i < current_results_count(); ++i) {
			const keys_vector & k = current_results_keys(i);
			current_results_query(i)->add_output(k.begin(), k.end());
		}
		return second_last_batch_ptr;
	} else if (last_batch != nullptr) {
		gpu::syncOnResults(stream, gpu);
	}
	return nullptr;
}

// We maintain a lock-free allocation list for stream handles.
//
// This is a simple singly-linked list where we flip pointers using an
// atomic compare-and-swap operation.  However, instead of using
// pointers for the CAS operations, we use integer indexes in the
// stream_handles array.  The idea is that CAS on an int might be
// faster than on a pointer.  Not sure this is true, but the code
// doesn't change much anyway.
//
static const unsigned char STREAM_HANDLES_NULL = 0xff;
static const unsigned char STREAM_HANDLES_MULTIPLIER = 1;

static stream_handle stream_handles[GPU_NUM * GPU_STREAMS * STREAM_HANDLES_MULTIPLIER];

static atomic<unsigned char> free_stream_handles;

static stream_handle * allocate_stream_handle() {
	unsigned char sp = free_stream_handles;
	do {
		while (sp == STREAM_HANDLES_NULL)
			sp = free_stream_handles.load(std::memory_order_acquire);
	} while (!free_stream_handles.compare_exchange_weak(sp, stream_handles[sp].next));

	return stream_handles + sp;
}

static void recycle_stream_handle(stream_handle * h) {
	unsigned char sp = h - stream_handles;
	h->next = free_stream_handles.load(std::memory_order_acquire);
	do {
	} while(!free_stream_handles.compare_exchange_weak(h->next, sp));
}

static void initialize_stream_handlers() {
	// this is supposed to execute atomically
	//
	free_stream_handles = STREAM_HANDLES_NULL;
	for (unsigned int i = 0; i < gpus_in_use * GPU_STREAMS * STREAM_HANDLES_MULTIPLIER; ++i) {
		// Stream handlers are initialized so that they are assigned
		// evenly to all the gpus available, going round robin
		//
		stream_handles[i].initialize(i / gpus_in_use, i % gpus_in_use, free_stream_handles);
		free_stream_handles = i;
	}
}

static void destroy_stream_handlers() {
	free_stream_handles = STREAM_HANDLES_NULL;
	for (unsigned int i = 0; i < gpus_in_use * GPU_STREAMS * STREAM_HANDLES_MULTIPLIER; ++i) {
		stream_handles[i].destroy();
	}
}

#if SORT_FILTERS
static bool compare_filters_decreasing(const filter_descr & d1, const filter_descr & d2) {
	return d2.filter < d1.filter;
}
#endif

static void compile_fibs() {
	// we first compute the total number of partitions
	// (dev_partitions_size), the size of the global ti_table
	// (total_filters), and the necessary number of entries in the
	// table of tree-interface keys (host_ti_table_size).  We use
	// these to initialize the buffers and global data structures.
	//
	// Since we put partitions in a direct-access table, that is, an
	// array whose index is the partition id, then the size of the
	// array is the largest partition id plus one.
	//
	dev_partitions_size = 0;
	unsigned int total_filters = 0;
	unsigned int max = 0;

	for (auto const & pf : tmp_fib) { // reminder: map<unsigned int, vector<filter_t>> tmp_fib
		unsigned int part_id_plus_one = pf.first + 1;
		const vector<filter_t> & filters = pf.second;

		if(max < filters.size())
			max = filters.size();

		if(part_id_plus_one > dev_partitions_size)
			dev_partitions_size = part_id_plus_one;

		total_filters += filters.size();
	}
	// since we only need to know the number of partitions, instead of
	// dev_partitions_size we could have a simple counter in the
	// previous for loop.
	//
	for (unsigned int g = 0; g < gpus_in_use; g++)
		dev_partitions[g] = new part_descr[dev_partitions_size];

	// local buffers to hold the temporary host-side of the fibs
	//
	uint32_t * host_rep = new uint32_t[max * GPU_FILTER_WORDS];

#if SORT_FILTERS
	for (auto & pf : tmp_fib)
#else
	for (auto const & pf : tmp_fib)
#endif
	{
		unsigned int part = pf.first;
#if SORT_FILTERS
		vector<filter_t> & filters = pf.second;
#else
		const vector<filter_t> & filters = pf.second;
#endif
		for (unsigned int g = 0; g < gpus_in_use; g++)
			dev_partitions[g][part].size = filters.size();

		unsigned int * host_rep_f = host_rep;
		int blocks_in_partition = dev_partitions[0][part].size / (GPU_BLOCK_SIZE) ;

		if (dev_partitions[0][part].size % GPU_BLOCK_SIZE != 0)
			blocks_in_partition++ ;
		int counter=0;

#if SORT_FILTERS
		std::sort(filters.begin(), filters.end(), compare_filters_decreasing) ;
#endif
		for (const filter_t & f : filters) {
			// we now store the index in the global tiff table for this fib entry
			//
			// we then store the *size* of the tiff table for this fib
			// entry in that position in the global table.
			//
			// then we copy the filter in the host table, using the
			// appropriate layout.
			//
			for (unsigned int i = 0; i < GPU_FILTER_WORDS; ++i) {
#ifdef COALESCED_READS
				host_rep_f[dev_partitions[0][part].size * i + counter] = (f.uint32_value(i));
#else
				*host_rep_f++ = (f.uint32_value(i));
#endif
			}
			counter++ ;
		}
		for (unsigned int g = 0; g < gpus_in_use; g++) {
			gpu::set_device(g);
			dev_partitions[g][part].fib = gpu::allocate_and_copy<uint32_t>(host_rep, dev_partitions[g][part].size * GPU_FILTER_WORDS);
		}
	}
	delete[](host_rep);
#ifndef CPU_BACKEND
	tmp_fib.clear();
#endif
}
#endif // BACK_END_IS_VOID

// this is called at the end of the computation, to extract results
// from the second last iterations of the process_batch loop it does
// not release the stream handler, because it forces the front_end to
// loop on all the streams available
//
batch * back_end::flush_stream() {
	stream_handle * sh = allocate_stream_handle();
	batch * res = sh->process_available_results();
	sh->flip_buffers();
	sh->async_copy_results_from_gpu();
	sh->shift_batches();
	return res;
}

// This is used by the front_end to release stream handlers when
// moving from the first stage of the final flushing to the second
//
void back_end::release_stream_handles() {
	for (unsigned char i=0; i < gpus_in_use * GPU_STREAMS; i++) {
		stream_handle * sh = stream_handles + i;
		recycle_stream_handle(sh);
	}
}

// This is called at the end of the computation to extract results
// from the last iterations of the process_batch loop.  It does NOT
// release the stream handler, because it forces the front_end to loop
// on all the streams available
//
batch * back_end::second_flush_stream() {
	stream_handle * sh = allocate_stream_handle();
	batch * res = sh->process_available_results();
	return res;
}

batch * back_end::process_batch(partition_id_t part, tagmatch_query ** q, unsigned int q_count, batch * batch_ptr) {
	batch * res;
#ifndef BACK_END_IS_VOID

// If the partition is small, compute it entirely on the CPU
//
#ifdef CPU_BACKEND
	if (dev_partitions[0][part].size < CPU_BACKEND) {
		tmp_fib_map::const_iterator fi = tmp_fib.find(part);
		assert(fi != tmp_fib.end());
		const vector<filter_t> & filters = fi->second;

		tmp_fib_map_users::const_iterator ki = tmp_fib_users.find(part);
		assert(ki != tmp_fib_map_users.end());
		const vector<keys_vector> & key_vectors = ki->second;

		for (unsigned int j = 0; j < filters.size(); ++j) {
			for (unsigned int i = 0; i < q_count; ++i) {
				if (filters[j].subset_of(q[i]->filter)) {
					for (const tagmatch_key_t & k : key_vectors[j]) {
						q[i]->add_output(k);
					}
				}
			}
		}
		return batch_ptr;
	}
#endif
	stream_handle * sh = allocate_stream_handle();
	res = sh->process_batch(part, q, q_count, batch_ptr);
	recycle_stream_handle(sh);
#else // BACK_END_IS_VOID

#ifndef NO_FRONTEND_OUTPUT_LOOP
	// if the backend is disabled we still loop through the output
	// array.  We do that to get a more accurate performance
	// measurement for the frontend.  More specifically, we account
	// for the extraction of the results.  Here we use the absolute
	// worst case, where we set ALL output interfaces.
	for (unsigned int i = 0; i < batch_size; ++i) {
		batch[i]->partition_done();
	}
#endif
#endif // BACK_END_IS_VOID
	return res;
}

static size_t back_end_memory = 0;

size_t back_end::bytesize() {
	return back_end_memory;
}

void back_end::start(unsigned int gpus) {
#ifndef BACK_END_IS_VOID
	assert(gpus > 0);
	gpu_mem_info mi;
	gpus_in_use = gpus;
	gpu::initialize(gpus_in_use);
	gpu::mem_info(&mi,gpus_in_use);
	back_end_memory = mi.free;
	initialize_stream_handlers();
	compile_fibs();
	for (unsigned int g = 0; g < gpus_in_use; g++) {
		gpu::set_device(g);
		gpu::synchronize_device();
	}
	gpu::mem_info(&mi, gpus_in_use);
	back_end_memory -= mi.free;
#endif
}

void back_end::stop() {
#ifndef BACK_END_IS_VOID
	for (unsigned int g = 0; g < gpus_in_use; g++) {
		gpu::set_device(g);
		gpu::synchronize_device();
	}
#endif
}

void back_end::clear() {
#ifndef BACK_END_IS_VOID
	back_end::stop();
	for (unsigned int g = 0; g < gpus_in_use; g++) {
		if (dev_partitions[g]) {
			for (unsigned int j = 0; j < dev_partitions_size; ++j)
				dev_partitions[g][j].clear();
			delete[](dev_partitions[g]);
			dev_partitions[g] = nullptr;
			dev_partitions_size = 0;
		}
	}
	destroy_stream_handlers();
	gpu::shutdown(gpus_in_use);
	gpus_in_use = 0;
	tmp_fib.clear();
	tmp_fib_users.clear();
#endif
}
