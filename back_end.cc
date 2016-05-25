#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef SAFE_FILTER_COPY
#include <cstring>
#endif
#include <vector>
#include <map>
#include <atomic>

#include "parameters.hh"
#include "gpu.hh"
#include "back_end.hh"

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

// TEMPORARY FIB
// 
typedef vector<tree_interface_pair> tip_vector;

struct filter_descr {
	filter_t filter;
	tip_vector ti_pairs;

	filter_descr(const filter_t & f,
				 tip_vector::const_iterator begin,
				 tip_vector::const_iterator end)
		: filter(f), ti_pairs(begin, end) {};
};

typedef vector<filter_descr> f_descr_vector;
typedef map<unsigned int, f_descr_vector > tmp_fib_map;

static tmp_fib_map tmp_fib;

void back_end::add_filter(unsigned int part, const filter_t & f, 
						  tip_vector::const_iterator begin,
						  tip_vector::const_iterator end) {
	// we simply add this filter to our temporary table
	// 
	tmp_fib[part].emplace_back(f, begin, end);
}

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

    // maps each filter in the partition into the index in
    // dev_ti_table that holds the tree-interface table for that
    // entry.
    // 
	uint32_t * ti_indexes;

	part_descr(): size(0), fib(0), ti_indexes(0) {};

	void clear() {
		size = 0;

		if (fib) {
			gpu::release_memory(fib);
			fib = nullptr;
		}

		if (ti_indexes) {
			gpu::release_memory(ti_indexes);
			ti_indexes = nullptr;
		}
	}
};

#ifndef BACK_END_IS_VOID
// we have a descriptor for each partition
// 
static part_descr * dev_partitions = nullptr;
static unsigned int dev_partitions_size = 0;

// plus a global array of tree-interface pairs.  This is a single
// array for all filters in all partitions.
// 
static uint16_t * dev_ti_table = nullptr; 

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

    // these are the buffers we use to communicate with the GPU for
    // matching.  For each stream, we store the query packets of each
    // batch here, together with their associated tree-interface
    // pairs, one for each packet, in a separate buffer.  We then
    // store the results that come back from the GPU.
	// 
	uint32_t * host_queries;
	uint16_t * host_query_ti_table;
	ifx_result_t * host_results;

    // PACKETS_BATCH_SIZE * INTERFACES
	uint16_t * dev_query_ti_table;
	ifx_result_t * dev_results;

	void initialize(unsigned int s, unsigned int n) {
		stream = s;
		next = n;

		host_query_ti_table = gpu::allocate_host_pinned<uint16_t>(PACKETS_BATCH_SIZE);
		host_queries = gpu::allocate_host_pinned<uint32_t>(PACKETS_BATCH_SIZE*GPU_FILTER_WORDS);
		host_results = gpu::allocate_host_pinned<ifx_result_t>(PACKETS_BATCH_SIZE*INTERFACES);

		dev_query_ti_table = gpu::allocate<uint16_t>(PACKETS_BATCH_SIZE);
		dev_results = gpu::allocate<ifx_result_t>(PACKETS_BATCH_SIZE*INTERFACES);
	}

	void destroy() {
		gpu::release_pinned_memory(host_query_ti_table);
		gpu::release_pinned_memory(host_queries);
		gpu::release_pinned_memory(host_results);
		gpu::release_memory(dev_query_ti_table);
		gpu::release_memory(dev_results);
	}
};

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

static stream_handle stream_handles[GPU_STREAMS];

static atomic<unsigned char> free_stream_handles;

static stream_handle * allocate_stream_handle() {
	unsigned char sp = free_stream_handles;
	do {
		while (sp == STREAM_HANDLES_NULL) 
			sp = free_stream_handles.load(std::memory_order_acquire);
	} while (!free_stream_handles.compare_exchange_weak(sp, stream_handles[sp].next));

	return stream_handles + sp;
}

static void release_stream_handle(stream_handle * h) {
	unsigned char sp = h - stream_handles;
	h->next = free_stream_handles.load(std::memory_order_acquire);
	do {
	} while(!free_stream_handles.compare_exchange_weak(h->next, sp));
}

static void initialize_stream_handlers() {
	// this is supposed to execute atomically
	// 
	free_stream_handles = STREAM_HANDLES_NULL;
	for(unsigned int i = 0; i < GPU_STREAMS; ++i) {
		stream_handles[i].initialize(i,free_stream_handles);
		free_stream_handles = i;
	}
}

static void destroy_stream_handlers() {
	free_stream_handles = STREAM_HANDLES_NULL;
	for(unsigned int i = 0; i < GPU_STREAMS; ++i) {
		stream_handles[i].destroy();
	}
}

static void compile_fibs() {
	// we first compute the total number of partitions
	// (dev_partitions_size), the size of the global ti_table
	// (total_filters), and the necessary number of entries in the
	// table of tree-interface pairs (host_ti_table_size).  We use
	// these to initialize the buffers and global data structures.
	// 
	// Since we put partitions in a direct-access table, that is, an
	// array whose index is the partition id, then the size of the
	// array is the largest partition id plus one.
	// 
	dev_partitions_size = 0;
	unsigned int host_ti_table_size = 0;
	unsigned int total_filters = 0;

	for(auto const & pf : tmp_fib) { // reminder: map<unsigned int, f_descr_vector> tmp_fib
		unsigned int part_id_plus_one = pf.first + 1;
		const f_descr_vector & filters = pf.second;

		if(part_id_plus_one > dev_partitions_size)
			dev_partitions_size = part_id_plus_one;

		total_filters += filters.size();
		for(const filter_descr & fd : filters)
			host_ti_table_size += fd.ti_pairs.size() + 1;
	}
	dev_partitions = new part_descr[dev_partitions_size];

	// local buffers to hold the temporary host-side of the fibs
	// 
	uint16_t * host_ti_table = new uint16_t[host_ti_table_size];
	uint32_t * host_rep = new uint32_t[total_filters * GPU_FILTER_WORDS];
	uint32_t * host_ti_table_indexes = new uint32_t[total_filters];

	unsigned int ti_table_curr_pos = 0;

	for(auto const & pf : tmp_fib) {
		unsigned int part = pf.first;
		const f_descr_vector & filters = pf.second;

		dev_partitions[part].size = filters.size();

		unsigned int * host_rep_f = host_rep;
		unsigned int * ti_index = host_ti_table_indexes;
		for(const filter_descr & fd : filters) {
			// we now store the index in the global tiff table for this fib entry
			// 
			*ti_index++ = ti_table_curr_pos;

			// we then store the *size* of the tiff table for this fib
			// entry in that position in the global table.
			//
			host_ti_table[ti_table_curr_pos++] = fd.ti_pairs.size();
			// and then we copy the table itself starting from the
			// following position
			// 
			for(const tree_interface_pair & tip : fd.ti_pairs)
				host_ti_table[ti_table_curr_pos++] = tip_uint16_value(tip);

			// then we copy the filter in the host table, using the
			// appropriate layout.
			// 
			for(unsigned int i = 0; i < GPU_FILTER_WORDS; ++i)
				*host_rep_f++ = ~(fd.filter.uint32_value(i));
		}
		dev_partitions[part].fib = gpu::allocate_and_copy<uint32_t>(host_rep, dev_partitions[part].size*GPU_FILTER_WORDS);

		dev_partitions[part].ti_indexes = gpu::allocate_and_copy<uint32_t>(host_ti_table_indexes, dev_partitions[part].size);
	}
	dev_ti_table = gpu::allocate_and_copy<uint16_t>(host_ti_table, host_ti_table_size);

	delete[](host_rep);
	delete[](host_ti_table_indexes);
	delete[](host_ti_table);

	tmp_fib.clear();
}
#endif

void back_end::process_batch(unsigned int part, packet ** batch, unsigned int batch_size) {
#ifndef BACK_END_IS_VOID
	stream_handle * sh = allocate_stream_handle();

	// We first copy every packet (filter plus tree-interface pair)
	// over to the device.  To do that we first assemble the whole
	// thing in buffers here on the host, and then we copy those
	// buffers over to the device.
	// 
	uint32_t * curr_p_buf = sh->host_queries;

	for(unsigned int i = 0; i < batch_size; ++i) {
		for(int j = 0; j < GPU_FILTER_WORDS; ++j)
			*curr_p_buf++ = batch[i]->filter.uint32_value(j);
		sh->host_query_ti_table[i] = tip_uint16_value(batch[i]->ti_pair);
	}

	gpu::async_copy_packets(sh->host_queries, batch_size, sh->stream);
	gpu::async_copy(sh->host_query_ti_table, sh->dev_query_ti_table, batch_size*sizeof(uint16_t), sh->stream);
	gpu::async_set_zero(sh->dev_results, batch_size*INTERFACES*sizeof(ifx_result_t), sh->stream);

	gpu::run_kernel(dev_partitions[part].fib, dev_partitions[part].size, 
					dev_ti_table, dev_partitions[part].ti_indexes, 
					sh->dev_query_ti_table, batch_size, 
					sh->dev_results, 
					sh->stream);

	gpu::async_get_results(sh->host_results, sh->dev_results, batch_size, sh->stream);
	gpu::synchronize_stream(sh->stream);

	ifx_result_t * result = sh->host_results;
	for(unsigned int i = 0; i < batch_size; ++i) {
		for(unsigned int j = 0; j < INTERFACES; ++j) 
			if (*result++ == 1)
				batch[i]->set_output(j);
		batch[i]->partition_done();
#if 0
		// this is where we could check whether the processing of
		// message batch[i] is complete, in which case we could
		// release whatever resources are associated with the packet
		//
		if (batch[i]->is_matching_complete())
			deallocate_packet(batch[i]);
#endif
	}
	release_stream_handle(sh);
#else  
#ifndef NO_FRONTEND_OUTPUT_LOOP
    // if the backend is disabled we still loop through the output
	// array.  We do that to get a more accurate performance
	// measurement for the frontend.  More specifically, we account
	// for the extraction of the results.  Here we use the absolute
	// worst case, where we set ALL output interfaces.
	for(unsigned int i = 0; i < batch_size; ++i) {
		for(unsigned int j = 0; j < INTERFACES; ++j) 
			batch[i]->set_output(j);
		batch[i]->partition_done();
#if 0
		if (batch[i]->is_matching_complete())
			deallocate_packet(batch[i]);
#endif
	}
#endif
#endif
}

static size_t back_end_memory = 0;

size_t back_end::bytesize() {
	return back_end_memory;
}

void back_end::start() {
#ifndef BACK_END_IS_VOID
	gpu_mem_info mi;
	gpu::initialize();
	gpu::mem_info(&mi);
	back_end_memory = mi.free;
	initialize_stream_handlers();
	compile_fibs();
	gpu::synchronize_device();
	gpu::mem_info(&mi);
	back_end_memory -= mi.free;
#endif
}

void back_end::stop() {
#ifndef BACK_END_IS_VOID
	gpu::synchronize_device();
#endif
}

void back_end::clear() {
#ifndef BACK_END_IS_VOID
	if (dev_partitions) {
		for(unsigned int i = 0; i < dev_partitions_size; ++i)
			dev_partitions[i].clear();
		delete[](dev_partitions);
		dev_partitions = nullptr;
		dev_partitions_size = 0;
	}
	if (dev_ti_table) {
		gpu::release_memory(dev_ti_table);
		dev_ti_table = nullptr;
	}
	destroy_stream_handlers();
	gpu::shutdown();
#endif
}
