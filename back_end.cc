#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef SAFE_FILTER_COPY
#include <cstring>
#endif
#include <climits>
#include <vector>
#include <map>
#include <atomic>
#include <cstdint>
#include <algorithm>

#include "gpu.hh"
#include "back_end.hh"
#include <time.h> 
#include <mutex>

#define SORT_FILTERS 0
#define COMBO 0
#define COMBO_SIZE 30000000

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

	// we store filters on the gpu using 32-bit blocks.
	// 
	static const unsigned int BACKEND_BLOCK_SIZE = (GPU_WORD_SIZE * CHAR_BIT);
	uint32_t absolute_id = 0;
	// TEMPORARY FIB
	// 
	typedef vector<tagmatch_key_t> keys_vector;

	struct filter_descr {
		filter_t filter;
		uint32_t id;
		filter_descr(const filter_t & f,
					 uint32_t i)
			: filter(f), id(i) {
			};
	};
	
	struct filter_descr_users {
		keys_vector keys; 
		filter_descr_users(keys_vector::const_iterator begin,
					 keys_vector::const_iterator end
					 )
			: keys(begin, end) {
			};
	};

	typedef vector<filter_descr> f_descr_vector;
	typedef map<unsigned int, f_descr_vector > tmp_fib_map;

	static tmp_fib_map tmp_fib;
	
	typedef vector<filter_descr_users> f_descr_vector_users;
	typedef map<unsigned int, f_descr_vector_users > tmp_fib_map_users;

	static tmp_fib_map_users tmp_fib_users;

	void back_end::add_filter(unsigned int part, const filter_t & f, 
							  keys_vector::const_iterator begin,
							  keys_vector::const_iterator end) {
		// we simply add this filter to our temporary table
		//
		tmp_fib[part].emplace_back(f, absolute_id++);
		tmp_fib_users[part].emplace_back(begin, end);
	}

	vector<uint8_t> prefix_block_lengths;

	void back_end::add_partition(unsigned int id, const filter_t & prefix) {
		if (id >= prefix_block_lengths.size())
			prefix_block_lengths.resize(id + 1);
		prefix_block_lengths[id] = 0;
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

		filter_t common_bits ;     // it holds common bits of all filters in this partition
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

		match_handler ** last_batch, ** second_last_batch;
		unsigned int last_batch_size, second_last_batch_size;
		void *last_batch_ptr, *second_last_batch_ptr;
		// these are the buffers we use to communicate with the GPU for
		// matching.  For each stream, we store the query packets of each
		// batch here, together with their associated tree-interface
		// keys, one for each packet, in a separate buffer.  We then
		// store the results that come back from the GPU.
		// 
		bool flip;
		uint32_t * host_queries[2];
		result_t * host_results[2];
		result_t * dev_results[2];
		uint32_t last_partition;
		uint32_t second_last_partition;

		void initialize(unsigned int s, unsigned int n, unsigned int gpu_count) {
			// Stream handlers are initialized so that they are assigned evenly to all the gpus available, going round robin
			//
			stream = s / gpu_count;
			gpu = s % gpu_count;
			next = n;
			
			gpu::set_device(gpu);
			
			// Here we initialize two buffers for each stream for each gpu for the staged computation that is taking place in process_batch
			//
			for (unsigned int f = 0; f < 2; f++) {
				host_queries[f] = gpu::allocate_host_pinned<uint32_t>(PACKETS_BATCH_SIZE*GPU_FILTER_WORDS);
				host_results[f] = gpu::allocate_host_pinned<result_t>(1);
				host_results[f]->count = 0;
				dev_results[f] = gpu::allocate<result_t>(1);
				gpu::async_set_zero(dev_results[f], sizeof(result_t),0,gpu);
			}
			last_batch = second_last_batch = nullptr;
			last_batch_ptr = second_last_batch_ptr = nullptr;
			last_batch_size = second_last_batch_size = 0;
	}

	void destroy(unsigned int gpu_count) {
		for (unsigned int g = 0; g < gpu_count; g++) {
			gpu::release_pinned_memory(host_queries[g]);
			gpu::release_pinned_memory(host_results[g]);
		}
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

static void release_stream_handle(stream_handle * h) {
	unsigned char sp = h - stream_handles;
	h->next = free_stream_handles.load(std::memory_order_acquire);
	do {
	} while(!free_stream_handles.compare_exchange_weak(h->next, sp));
}

static void initialize_stream_handlers(unsigned int gpu_count) {
	// this is supposed to execute atomically
	// 
	free_stream_handles = STREAM_HANDLES_NULL;
	for(unsigned int i = 0; i < gpu_count * GPU_STREAMS * STREAM_HANDLES_MULTIPLIER; ++i) {
		stream_handles[i].initialize(i,free_stream_handles, gpu_count);
		free_stream_handles = i;
	}
}

static void destroy_stream_handlers(unsigned int gpu_count) {
	free_stream_handles = STREAM_HANDLES_NULL;
	for(unsigned int i = 0; i < gpu_count * GPU_STREAMS * STREAM_HANDLES_MULTIPLIER; ++i) {
		stream_handles[i].destroy(gpu_count);
	}
}

filter_t back_end::get_cbits(unsigned int id){
	return dev_partitions[0][id].common_bits;
}

#if SORT_FILTERS
static bool compare_filters_decreasing(const filter_descr & d1, const filter_descr & d2) {
	return d2.filter < d1.filter;
}
#endif

static void compile_fibs(unsigned int gpu_count) {
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

	for(auto const & pf : tmp_fib) { // reminder: map<unsigned int, f_descr_vector> tmp_fib
		unsigned int part_id_plus_one = pf.first + 1;
		const f_descr_vector & filters = pf.second;

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
	for (unsigned int g = 0; g < gpu_count; g++)
		dev_partitions[g] = new part_descr[dev_partitions_size];

	// local buffers to hold the temporary host-side of the fibs
	// 
	uint32_t * host_rep = new uint32_t[max * GPU_FILTER_WORDS];

	filter_t cbits; //common bits

#if SORT_FILTERS
	for(auto & pf : tmp_fib) {
#else
	for(auto const & pf : tmp_fib) {
#endif
		cbits.fill();

		unsigned int part = pf.first;
#if SORT_FILTERS
		f_descr_vector & filters = pf.second;
#else
		const f_descr_vector & filters = pf.second;
#endif
		for (unsigned int g = 0; g < gpu_count; g++)
			dev_partitions[g][part].size = filters.size();

		unsigned int * host_rep_f = host_rep;
		unsigned int full_blocks = prefix_block_lengths[part];
		int blocks_in_partition = dev_partitions[0][part].size / (GPU_BLOCK_SIZE) ;

		if (dev_partitions[0][part].size % GPU_BLOCK_SIZE != 0)
			blocks_in_partition++ ;
		int counter=0; 

#if SORT_FILTERS
		for(filter_descr & fd : filters) {
			fd.filter ^= cbits ;
		}
		std::sort(filters.begin(), filters.end(), compare_filters_decreasing) ; 
#else
		for(const filter_descr & fd : filters) {
			cbits &= fd.filter ;
		}
#endif
			for(const filter_descr & fd : filters) {
			// we now store the index in the global tiff table for this fib entry
			// 
			// we then store the *size* of the tiff table for this fib
			// entry in that position in the global table.
			//
			// then we copy the filter in the host table, using the
			// appropriate layout.
			// 
			for(unsigned int i = full_blocks; i < GPU_FILTER_WORDS; ++i){
#ifdef COALESCED_READS
				host_rep_f[dev_partitions[0][part].size * (i-full_blocks) + counter] = (fd.filter.uint32_value(i));
#else
				*host_rep_f++ = (fd.filter.uint32_value(i));
#endif
			}
			counter++ ;
		}
		for (unsigned int g = 0; g< gpu_count; g++) {
			gpu::set_device(g);
			dev_partitions[g][part].fib = gpu::allocate_and_copy<uint32_t>(host_rep, dev_partitions[g][part].size * (GPU_FILTER_WORDS - full_blocks) );

		}
		// this thing doesn't really need to be in the partition_descriptors!
		//
		dev_partitions[0][part].common_bits = cbits; 
	}

	delete[](host_rep);
#if !COMBO
	tmp_fib.clear();
#endif
}
#endif


// this is called at the end of the computation, to extract results
// from the second last iterations of the process_batch loop it does
// not release the stream handler, because it forces the front_end to
// loop on all the streams available
// 
void* back_end::flush_stream()
{
	void *res = nullptr;
    stream_handle * sh = allocate_stream_handle();
	gpu::set_device(sh->gpu);
	//Wait for the data to be copied
	gpu::syncOnResults(sh->stream, sh->gpu);
	if (sh->second_last_batch != nullptr) {
	  res = sh->second_last_batch_ptr;
	  assert(sh->host_results[!sh->flip]->count <= MAX_MATCHES); 
	  for(unsigned int i = 0; i < sh->host_results[!sh->flip]->count; ++i) {
			unsigned char pkt_idx = (sh->host_results[sh->flip]->keys[5*(i/4)] >> (8*(i%4)))& 0xFF;
			uint32_t id = sh->host_results[sh->flip]->keys[5*(i/4)+1+(i%4)];
			match_handler * h = sh->second_last_batch[pkt_idx];  
			packet * pkt = h->p;
			pkt->lock_mtx();	
#if 1
			const filter_descr_users & filter = tmp_fib_users[sh->second_last_partition][id];
			for(unsigned int i = 0; i < filter.keys.size(); ++i) { 
				pkt->add_output_key(filter.keys[i]);
			}
#else
				// You need to get unique ids from the gpu to use this
				//
				pkt->add_output_key(id);
#endif
			pkt->unlock_mtx();	
	  }
  	}
	gpu::async_get_results(sh->host_results[!sh->flip], sh->dev_results[!sh->flip], sh->host_results[sh->flip]->count, sh->stream, sh->gpu);
	return res;
}


// This is used by the front_end to release stream handlers when
// moving from the first stage of the final flushing to the second
//
void back_end::release_stream_handles(unsigned int gpu_count)
{
	for (unsigned char i=0; i < gpu_count * GPU_STREAMS; i++) {
		stream_handle * sh = stream_handles + i;
		release_stream_handle(sh);
	}
}

// This is called at the end of the computation, to extract results
// from the last iterations of the process_batch loop.  It does NOT
// release the stream handler, because it forces the front_end to loop
// on all the streams available
// 
void* back_end::second_flush_stream()
{
	void *res = nullptr;
    stream_handle * sh = allocate_stream_handle();
	gpu::set_device(sh->gpu);
	if (sh->last_batch != nullptr) {
		res = sh->last_batch_ptr;
		gpu::syncOnResults(sh->stream, sh->gpu);
	 	assert(sh->host_results[sh->flip]->count <= MAX_MATCHES); 
	  	for(unsigned int i = 0; i < sh->host_results[sh->flip]->count; ++i) {
			unsigned char pkt_idx = (sh->host_results[!sh->flip]->keys[5*(i/4)] >> (8*(i%4)))& 0xFF;
			uint32_t id = sh->host_results[!sh->flip]->keys[5*(i/4)+1+i%4];
			match_handler * h = sh->last_batch[pkt_idx];  
			packet * pkt = h->p;
			pkt->lock_mtx();	
#if 1
			const filter_descr_users & filter = tmp_fib_users[sh->last_partition][id];
			for(unsigned int i = 0; i < filter.keys.size(); ++i) 
				pkt->add_output_key(filter.keys[i]);
#else
			pkt->add_output_key(id);
#endif
			pkt->unlock_mtx();	
		}
  	}
	return res;
}


void * back_end::process_batch(unsigned int part, match_handler ** batch, unsigned int batch_size, void *batch_ptr) {
	void *res = nullptr;
#ifndef BACK_END_IS_VOID

// If the partition is small, compute it entirely on the CPU
// 
#if COMBO
	if (dev_partitions[0][part].size < COMBO_SIZE) {
		const f_descr_vector & filters = tmp_fib[part];
		uint32_t fidx = 0;
		for(const filter_descr & fd : filters) { 
			for(unsigned int i = 0; i < batch_size; ++i) {
				if (fd.filter.subset_of(batch[i]->p->filter)) {
					for(const tagmatch_key_t & k : tmp_fib_users[part][fidx].keys) {
						batch[i]->p->add_output_key(k);
					}
				}
			}
			fidx++;
		}

		return batch_ptr; 
	}
#endif
	stream_handle * sh = allocate_stream_handle();
	uint8_t blocks = prefix_block_lengths[part];

	// Flip the buffers for the staged computation to avoid overwriting
	//
	sh->flip = !sh->flip;
	uint32_t * curr_p_buf = sh->host_queries[sh->flip];
	
	// We first copy every packet (filter plus tree-interface pair)
	// over to the device.  To do that we first assemble the whole
	// thing in buffers here on the host, and then we copy those
	// buffers over to the device.
	//
	for(unsigned int i = 0; i < batch_size; ++i) {
		for(int j = blocks; j < GPU_FILTER_WORDS; ++j)
			*curr_p_buf++ = batch[i]->p->filter.uint32_value(j);
	}

	gpu::set_device(sh->gpu);
	gpu::async_copy_packets(sh->host_queries[sh->flip], batch_size * (GPU_FILTER_WORDS-blocks), sh->stream, sh->gpu);
	gpu::run_kernel(dev_partitions[sh->gpu][part].fib,  
					dev_partitions[sh->gpu][part].size, 
					batch_size, 
					sh->dev_results[sh->flip],
					sh->dev_results[!sh->flip],
					sh->stream, 
					sh->gpu, 
					blocks);
	// If the second last batch is set, then we have some results to
	// check from that computation
	//
	if (sh->second_last_batch != nullptr) {
		res = sh->second_last_batch_ptr;
		gpu::syncOnResults(sh->stream, sh->gpu);
		
		// If there are more results than slots available for the results, some results will be
		// ignored, and the following assertion is triggered
		//
	 	assert(sh->host_results[sh->flip]->count <= MAX_MATCHES); 
		for(unsigned int i = 0; i < sh->host_results[sh->flip]->count; ++i) {
			unsigned char pkt_idx = (sh->host_results[!sh->flip]->keys[5*(i/4)] >> (8*(i%4))) & 0xFF;
			uint32_t id = sh->host_results[!sh->flip]->keys[5*(i/4)+1+(i%4)];
			match_handler * h = sh->second_last_batch[pkt_idx];
			packet * pkt = h->p;
			pkt->lock_mtx();
#if 1
			const filter_descr_users & filter = tmp_fib_users[sh->second_last_partition][id];
			for(unsigned int i = 0; i < filter.keys.size(); ++i) 
				pkt->add_output_key(filter.keys[i]);
#else
			pkt->add_output_key(id);
#endif
			pkt->unlock_mtx();	
		}
	}
	// If we are at the 2nd cycle, synchronize on the results from the
	// previous iteration, so that the counter for the results to be
	// copied hereafter is correct
	//
	else if (sh->second_last_batch == nullptr && sh->last_batch != nullptr) {
		gpu::syncOnResults(sh->stream, sh->gpu);
	}
	assert(sh->host_results[!sh->flip]->count <= MAX_MATCHES);
	gpu::async_get_results(sh->host_results[sh->flip], sh->dev_results[sh->flip],sh->host_results[!sh->flip]->count, sh->stream, sh->gpu);
	gpu::async_set_zero(sh->dev_results[sh->flip],  sizeof(uint32_t)*(1+(sh->host_results[!sh->flip]->count)+(sh->host_results[!sh->flip]->count+ 3)/4), sh->stream, sh->gpu);
    
	// Update the info about the staged computation; drop the second
    // last iteration and store the info about the current iteration
    // for the next cycles
	// 
	sh->second_last_batch = sh->last_batch;
	sh->second_last_batch_size = sh->last_batch_size;
	sh->second_last_batch_ptr = sh->last_batch_ptr;

	sh->last_batch = batch;
	sh->last_batch_size = batch_size;
	sh->last_batch_ptr = batch_ptr;

	sh->second_last_partition = sh->last_partition; 
	sh->last_partition = part;

	release_stream_handle(sh);
#else //back_end_is_void 

#ifndef NO_FRONTEND_OUTPUT_LOOP
    // if the backend is disabled we still loop through the output
	// array.  We do that to get a more accurate performance
	// measurement for the frontend.  More specifically, we account
	// for the extraction of the results.  Here we use the absolute
	// worst case, where we set ALL output interfaces.
	for(unsigned int i = 0; i < batch_size; ++i) {
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
#endif
#endif
	return res;
}

static size_t back_end_memory = 0;

size_t back_end::bytesize() {
	return back_end_memory;
}

void back_end::start(unsigned int gpu_count) {
#ifndef BACK_END_IS_VOID
	gpu_mem_info mi;
	gpu::initialize(gpu_count);
	gpu::mem_info(&mi, gpu_count);
	back_end_memory = mi.free;
	initialize_stream_handlers(gpu_count);
	compile_fibs(gpu_count);
	for (unsigned int g = 0; g < gpu_count; g++) {
		gpu::set_device(g);
		gpu::synchronize_device();
	}
	gpu::mem_info(&mi, gpu_count);
	back_end_memory -= mi.free;
#endif
}

void back_end::stop(unsigned int gpu_count) {
#ifndef BACK_END_IS_VOID
	for (unsigned int g = 0; g < gpu_count; g++) {
		gpu::set_device(g);
		gpu::synchronize_device();
	}
#endif
}

void back_end::clear(unsigned int gpu_count) {
#ifndef BACK_END_IS_VOID
	for (unsigned int g = 0; g < gpu_count; g++) {
		if (dev_partitions[g]) {
			for(unsigned int j = 0; j < dev_partitions_size; ++j)
				dev_partitions[g][j].clear();
			delete[](dev_partitions[g]);
			dev_partitions[g] = nullptr;
			dev_partitions_size = 0;
		}
	}
	destroy_stream_handlers(gpu_count);
	gpu::shutdown(gpu_count);
#endif
}
