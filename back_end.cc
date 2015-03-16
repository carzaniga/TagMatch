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

#define COMBO 0

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

// TEMPORARY FIB
// 
typedef vector<tree_interface_pair> ti_vector;

struct filter_descr {
	filter_t filter;
	ti_vector ti_pairs;

	filter_descr(const filter_t & f,
				 ti_vector::const_iterator begin,
				 ti_vector::const_iterator end)
		: filter(f), ti_pairs(begin, end) {};
};

typedef vector<filter_descr> f_descr_vector;
typedef map<unsigned int, f_descr_vector > tmp_fib_map;

static tmp_fib_map tmp_fib;

void back_end::add_filter(unsigned int part, const filter_t & f, 
						  ti_vector::const_iterator begin,
						  ti_vector::const_iterator end) {
	// we simply add this filter to our temporary table
	// 
	tmp_fib[part].emplace_back(f, begin, end);
}

vector<uint8_t> prefix_block_lengths;

void back_end::add_partition(unsigned int id, const filter_t & prefix, unsigned int prefix_length) {
	if (id >= prefix_block_lengths.size())
		prefix_block_lengths.resize(id + 1);
#if NEW_PARTIIONING
	prefix_block_lengths[id] = 0;// prefix_length / BACKEND_BLOCK_SIZE;
#else
	prefix_block_lengths[id] = prefix_length / BACKEND_BLOCK_SIZE;
#endif
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

	uint32_t * intersections ;
	filter_t common_bits ;     // it holds common bits of all filters in this partition
//	part_descr(): size(0), fib(0), ti_indexes(0) {};
	part_descr(): size(0), fib(0), ti_indexes(0), intersections(0) {};

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
//static std::mutex mu[GPU_STREAMS]; 
struct stream_handle {
	// we maintain a list of free handles.  This is the pointer to the
	// next free handle.  More specifically, it is an index in the
	// stream_handles array.
	//
	unsigned char next;
//	std::mutex * mutex;
	
	// stream identifier shared with the GPU
	//
	unsigned int stream;

	packet ** last_batch;
	unsigned int last_batch_size;
	void *last_batch_ptr;
    // these are the buffers we use to communicate with the GPU for
    // matching.  For each stream, we store the query packets of each
    // batch here, together with their associated tree-interface
    // pairs, one for each packet, in a separate buffer.  We then
    // store the results that come back from the GPU.
	// 
	bool flip;
	uint32_t * host_queries[2];
	uint16_t * host_query_ti_table;
	result_t * host_results;

    // PACKETS_BATCH_SIZE * INTERFACES
	uint16_t * dev_query_ti_table;
	result_t * dev_results;

	void initialize(unsigned int s, unsigned int n) {
		stream = s;
//		mutex = mu+ s;
		next = n;

		host_query_ti_table = gpu::allocate_host_pinned<uint16_t>(PACKETS_BATCH_SIZE);
		host_queries[0] = gpu::allocate_host_pinned<uint32_t>(PACKETS_BATCH_SIZE*GPU_FILTER_WORDS);
		host_queries[1] = gpu::allocate_host_pinned<uint32_t>(PACKETS_BATCH_SIZE*GPU_FILTER_WORDS);
				
		//host_results = gpu::allocate_host_pinned<ifx_result_t>(PACKETS_BATCH_SIZE*INTERFACES + 1);
		host_results = gpu::allocate_host_pinned<result_t>(1);
#if 1
		host_results->done = false;
#endif
		dev_query_ti_table = gpu::allocate<uint16_t>(PACKETS_BATCH_SIZE);
		//dev_results = gpu::allocate<ifx_result_t>(PACKETS_BATCH_SIZE*INTERFACES);
		dev_results = gpu::allocate<result_t>(1);

		last_batch = NULL;
		last_batch_ptr = NULL;
		last_batch_size = 0;
	}

	void destroy() {
		gpu::release_pinned_memory(host_query_ti_table);
		gpu::release_pinned_memory(host_queries[0]);
		gpu::release_pinned_memory(host_queries[1]);
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
static const unsigned char STREAM_HANDLES_MULTIPLIER = 1;

static stream_handle stream_handles[GPU_STREAMS * STREAM_HANDLES_MULTIPLIER];

static atomic<unsigned char> free_stream_handles;

static stream_handle * allocate_stream_handle() {
	unsigned char sp = free_stream_handles;
	do {
		while (sp == STREAM_HANDLES_NULL) 
			sp = free_stream_handles.load(std::memory_order_acquire);
	} while (!free_stream_handles.compare_exchange_weak(sp, stream_handles[sp].next));

	return stream_handles + sp;
}

static stream_handle * allocate_stream_handle(unsigned char s) {
	unsigned char sp = s;
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
	for(unsigned int i = 0; i < GPU_STREAMS * STREAM_HANDLES_MULTIPLIER; ++i) {
		stream_handles[i].initialize(i % GPU_STREAMS,free_stream_handles);
		free_stream_handles = i;
	}
}

static void destroy_stream_handlers() {
	free_stream_handles = STREAM_HANDLES_NULL;
	for(unsigned int i = 0; i < GPU_STREAMS * STREAM_HANDLES_MULTIPLIER; ++i) {
		stream_handles[i].destroy();
	}
}

filter_t back_end::get_cbits(unsigned int id){
	return dev_partitions[id].common_bits;
}

static bool compare_filters_decreasing(const filter_descr & d1, const filter_descr & d2) {
	return d2.filter < d1.filter;
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
	unsigned int max = 0;

	for(auto const & pf : tmp_fib) { // reminder: map<unsigned int, f_descr_vector> tmp_fib
		unsigned int part_id_plus_one = pf.first + 1;
		const f_descr_vector & filters = pf.second;

		if(max < filters.size())
			max = filters.size();
		
		if(part_id_plus_one > dev_partitions_size)
			dev_partitions_size = part_id_plus_one;

		total_filters += filters.size();
		for(const filter_descr & fd : filters)
			host_ti_table_size += fd.ti_pairs.size() + 1;
	}
	// since we only need to know the number of partitions, instead of
	// dev_partitions_size we could have a simple counter in the
	// previous for loop.
	dev_partitions = new part_descr[dev_partitions_size];

	// local buffers to hold the temporary host-side of the fibs
	// 
	uint16_t * host_ti_table = new uint16_t[host_ti_table_size];
	uint32_t * host_rep = new uint32_t[max * GPU_FILTER_WORDS];
	uint32_t * host_ti_table_indexes = new uint32_t[total_filters];

	unsigned int ti_table_curr_pos = 0;

	int intersection_max_size = (max / (GPU_BLOCK_SIZE)); 
	if (max % (GPU_BLOCK_SIZE) !=0)
		intersection_max_size++ ;	
	uint32_t * block_intersections = new uint32_t [intersection_max_size * GPU_FILTER_WORDS];
	uint32_t intersections [GPU_FILTER_WORDS] ;
//	uint32_t partition_common_bits [GPU_FILTER_WORDS] ;
	filter_t cbits; //common bits

	//for(auto const & pf : tmp_fib) {
	for(auto & pf : tmp_fib) {
		for(int j=0; j < GPU_FILTER_WORDS; j++){
			intersections[j]= 0xFFFFFFFF ;
//			partition_common_bits[j]=0xFFFFFFFF ;			
		}
		cbits.fill();

		unsigned int part = pf.first;
#if 1
		f_descr_vector & filters = pf.second;
#else
		const f_descr_vector & filters = pf.second;
#endif
		dev_partitions[part].size = filters.size();

		unsigned int * host_rep_f = host_rep;
		unsigned int * ti_index = host_ti_table_indexes;
		unsigned int * block_intersections_f  = block_intersections;
		unsigned int full_blocks = prefix_block_lengths[part];
		int blocks_in_partition = dev_partitions[part].size / (GPU_BLOCK_SIZE) ;

		if (dev_partitions[part].size % GPU_BLOCK_SIZE != 0)
			blocks_in_partition++ ;
//		blocks_in_partition*= (GPU_FILTER_WORDS - full_blocks) ;
		int counter=0; 

#if 1
		for(const filter_descr & fd : filters) {
			cbits &= fd.filter ;
		}
#endif
		for(filter_descr & fd : filters) {
			fd.filter ^= cbits ;
		}
//		std::sort(filters.begin(), filters.end(), compare_filters_decreasing) ; 
		//for(const filter_descr & fd : filters) {
		for(const filter_descr & fd : filters) {
//			cbits &= fd.filter ;
			// we now store the index in the global tiff table for this fib entry
			// 
#if 0
			filter_t ff = fd.filter;
			ff ^= cbits ;
#endif
			*ti_index++ = ti_table_curr_pos;

			// we then store the *size* of the tiff table for this fib
			// entry in that position in the global table.
			//
			host_ti_table[ti_table_curr_pos++] = fd.ti_pairs.size();
			// and then we copy the table itself starting from the
			// following position
			// 
			for(const tree_interface_pair & tip : fd.ti_pairs)
				host_ti_table[ti_table_curr_pos++] = tip.get_uint16_value();

			// then we copy the filter in the host table, using the
			// appropriate layout.
			// 
			for(unsigned int i = full_blocks; i < GPU_FILTER_WORDS; ++i){
				*host_rep_f++ = (fd.filter.uint32_value(i));
				intersections[i- full_blocks] = (intersections[i- full_blocks] & fd.filter.uint32_value(i)) ;
//				*host_rep_f++ = (ff.uint32_value(i));
//				intersections[i- full_blocks] = (intersections[i- full_blocks] & ff.uint32_value(i)) ;
				//partition_common_bits[i- full_blocks] = (partition_common_bits[i- full_blocks] & fd.filter.uint32_value(i)) ;
			}
#if 0
			for(unsigned int i = 0; i < GPU_FILTER_WORDS; ++i)
				partition_common_bits[i] = (partition_common_bits[i] & fd.filter.uint32_value(i)) ;
#endif
			counter++ ;
			if(counter % GPU_BLOCK_SIZE == 0){
//				printf("p_size=%u, ints= ", dev_partitions[part].size) ;
				for(unsigned int i = full_blocks ; i < GPU_FILTER_WORDS; ++i){
					*block_intersections_f++ = intersections[i- full_blocks]; 
//					printf("%u ",intersections[i- full_blocks]) ;	
				}
//				printf("\n") ;
				for(int j=0; j < GPU_FILTER_WORDS; j++)
					intersections[j]= 0xFFFFFFFF ;
			}
		}
#if 0
		std::cout << part << " " ; 
		for(unsigned int i = 0 ; i < GPU_FILTER_WORDS; ++i)
			std::cout << partition_common_bits[i] << " " ; 
		std::cout << std::endl;
		for(const uint64_t * i = cbits.begin64(); i< cbits.end64(); i++) 
			std::cout << *i << " " ; 
		std::cout << std::endl;
#endif
		if(counter % GPU_BLOCK_SIZE != 0)
			for(unsigned int i = full_blocks ; i < GPU_FILTER_WORDS; ++i)
				*block_intersections_f++ = intersections[i- full_blocks]; 		

		dev_partitions[part].fib = gpu::allocate_and_copy<uint32_t>(host_rep, dev_partitions[part].size * (GPU_FILTER_WORDS - full_blocks) );

		dev_partitions[part].ti_indexes = gpu::allocate_and_copy<uint32_t>(host_ti_table_indexes, dev_partitions[part].size);
		
		dev_partitions[part].intersections = gpu::allocate_and_copy<uint32_t>(block_intersections, blocks_in_partition * (GPU_FILTER_WORDS - full_blocks));

		dev_partitions[part].common_bits = cbits; 

//		std::cout<<"p "<<(int) part << " " ;
//		cbits.write_ascii(std::cout);
//		std::cout<< " " <<dev_partitions[part].size << std::endl ;
	}

	dev_ti_table = gpu::allocate_and_copy<uint16_t>(host_ti_table, host_ti_table_size);
//	std::cout<<std::endl<<  "host_ti_table_size " << host_ti_table_size << std::endl;

	delete[](host_rep);
	delete[](host_ti_table_indexes);
	delete[](host_ti_table);
	delete[](block_intersections) ;
#if COMBO
#else 
	tmp_fib.clear();
#endif 
}
#endif

void* back_end::flush_stream()
{
    void *res = NULL;
    //Wrong, but may approximate the real time needed
    //stream_handle * sh = allocate_stream_handle(streamId);//This now fails! stream_handles + streamId;
    stream_handle * sh = allocate_stream_handle();//This now fails! stream_handles + streamId;
if (sh->last_batch != NULL) {
	  gpu::async_get_results(sh->host_results, sh->dev_results, sh->stream);
	  res = sh->last_batch_ptr;
	  gpu::synchronize_stream(sh->stream);
//	while(!sh->host_results->done){
//	}
	sh->host_results->done = false;

	  for(unsigned int i = 0; i < sh->host_results->count; ++i) {
		sh->last_batch[sh->host_results->pairs[i]>>8]->set_output(sh->host_results->pairs[i] & 0xff);
	  }
	  for(unsigned int i=0; i< sh->last_batch_size; i++)
		sh->last_batch[i]->partition_done();
  }
  	  return res;
}

void * back_end::process_batch(unsigned int part, packet ** batch, unsigned int batch_size, void *batch_ptr) {
void *res = NULL;

#ifndef BACK_END_IS_VOID
//	if(dev_partitions[part].size>=20000){
//		for(unsigned int i=0; i< batch_size; i++)
//			batch[i]->partition_done();
//		return;
//	}
#if COMBO

	if (dev_partitions[part].size <1000){
		const f_descr_vector & filters = tmp_fib[part];
//		std::cout <<" " << filters.size() << std::endl; 
		for(const filter_descr & fd : filters){ 
		//	fd.filter.write_ascii(std::cout);
			for(unsigned int i = 0; i < batch_size; ++i) 
				if (fd.filter.subset_of(batch[i]->filter)){
					for(const tree_interface_pair & tip : fd.ti_pairs)
						if (batch[i]->ti_pair.tree() == tip.tree() && batch[i]->ti_pair.interface() != tip.interface())
							batch[i]->set_output(tip.interface());
				}
		}

		for(unsigned int i=0; i< batch_size; i++)
			batch[i]->partition_done();
		return batch_ptr; 
	}
#endif
//	cbits.write_ascii(std::cout);
	
	stream_handle * sh = allocate_stream_handle();
	uint8_t blocks = prefix_block_lengths[part];

	// We first copy every packet (filter plus tree-interface pair)
	// over to the device.  To do that we first assemble the whole
	// thing in buffers here on the host, and then we copy those
	// buffers over to the device.
	// 
//	sh->mutex->lock();

	//This can be removed with a double buffer
	sh->flip = !sh->flip;
	//if (sh->last_batch != NULL)
	  //gpu::synchronize_stream(sh->stream);
	uint32_t * curr_p_buf = sh->host_queries[sh->flip];
//	std::cout << part <<" "<< (int)blocks << std::endl;
#if 1
#if 0
	for(unsigned int i = 0; i < batch_size; ++i) 
		sh->host_query_ti_table[i] = batch[i]->ti_pair.get_uint16_value();
	gpu::async_copy(sh->host_query_ti_table, sh->dev_query_ti_table, batch_size*sizeof(uint16_t), sh->stream);

	for(unsigned int i = 0; i < batch_size; ++i) 
		for(int j = blocks; j < GPU_FILTER_WORDS; ++j)
			*curr_p_buf++ = batch[i]->filter.uint32_value(j);

	gpu::async_copy_packets(sh->host_queries[sh->flip], batch_size * (GPU_FILTER_WORDS-blocks), sh->stream);
#else
	for(unsigned int i = 0; i < batch_size; ++i) {
//		batch[i]->filter.copy_into_uint32_array(curr_p_buf);
		for(int j = blocks; j < GPU_FILTER_WORDS; ++j)
			*curr_p_buf++ = batch[i]->filter.uint32_value(j);
//			*curr_p_buf++ = batch[i]->filter.unsafe_uint32_value(j);
		sh->host_query_ti_table[i] = batch[i]->ti_pair.get_uint16_value();
	}
	gpu::async_copy_packets(sh->host_queries[sh->flip], batch_size * (GPU_FILTER_WORDS-blocks), sh->stream);
	gpu::async_copy(sh->host_query_ti_table, sh->dev_query_ti_table, batch_size*sizeof(uint16_t), sh->stream);


#endif
#else 
	int min = 0 ;
	for(unsigned int i = 0; i < batch_size; ++i) {
		if (batch[i]->filter > batch[min]->filter)
			min=i ;
		for(int j = blocks; j < GPU_FILTER_WORDS; ++j)
			*curr_p_buf++ = batch[min]->filter.uint32_value(j);
		sh->host_query_ti_table[i] = batch[i]->ti_pair.get_uint16_value();
	}
#endif 
	gpu::run_kernel(dev_partitions[part].fib, dev_partitions[part].size, 
					dev_ti_table, dev_partitions[part].ti_indexes, 
					sh->dev_query_ti_table, batch_size, 
					sh->dev_results, 
					sh->stream, blocks,
					dev_partitions[part].intersections);
//	sh->mutex->unlock();
#if 0
	gpu::synchronize_stream(sh->stream);
#else
//	while(!sh->host_results->done){
//	}
//	sh->host_results->done = false;
	// this is to get an ack from kernel when transfer of the results to cpu is done.
	// The code is not finalized yet.
#endif
//	std::cout << "batch_size= "<< batch_size << " pid= " << part << " ,result count is: " << sh->host_results->count << std::endl ;

	if (sh->last_batch != NULL) {
		res = sh->last_batch_ptr;
//	  	gpu::synchronize_stream(sh->stream);
		while(!sh->host_results->done){
		}
		sh->host_results->done = false;
		
//		std::cout<<(int)  sh->host_results->count << std::endl;
		for(unsigned int i = 0; i < sh->host_results->count; ++i) {

//			std::cout << "mid= "<<(int)(sh->host_results->pairs[i]>>8) << " tid= " << (int)(sh->host_results->pairs[i] & 0xff) << std::endl ;
	//		std::cout << "mid= "<<(int)(sh->host_results->pairs[i]>>8) << " tid= " << (int)(sh->host_results->pairs[i] & 0xff) << std::endl ;
			sh->last_batch[sh->host_results->pairs[i]>>8]->set_output(sh->host_results->pairs[i] & 0xff);
	#if 0
			// this is where we could check whether the processing of
			// message batch[i] is complete, in which case we could
			// release whatever resources are associated with the packet
			//
			if (batch[i]->is_matching_complete())
				deallocate_packet(batch[i]);
	#endif
		}
		for(unsigned int i=0; i< sh->last_batch_size; i++)
			sh->last_batch[i]->partition_done();
	}
	
	gpu::async_get_results(sh->host_results, sh->dev_results, sh->stream);

	sh->last_batch = batch;
	sh->last_batch_size = batch_size;
	sh->last_batch_ptr = batch_ptr;

	release_stream_handle(sh);
#else //back_end_is_void 
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
return res;
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
