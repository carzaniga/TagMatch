#include "tagmatch.hh"

#include <map>
#include <set>
#include <fstream>

#include "tagmatch.hh"
#include "fib.hh"
#include "partitioner.hh"
#include "front_end.hh"
#include "back_end.hh"

static bool already_consolidated = false;

// TODO: do we want the library to be thread safe? We need to
// add some locks in that case
//
enum operation_type {
	ADD = 0,
	DELETE = 1,
};

struct fib_update_operation {
	fib_update_operation(operation_type t, filter_t f, tagmatch_key_t k) {
		type = t;
		filter = f;
		key = k;
	};
	operation_type type;
	filter_t filter;
	tagmatch_key_t key;
};

std::map<filter_t, std::set<tagmatch_key_t>> fib_map;
std::vector<fib_update_operation> updates;

void tagmatch::delete_set(filter_t set, tagmatch_key_t key) {
	// I should already have a map in the disk, that will
	// be loaded on the next consolidate() call... For
	// now, just enqueue the request for later
	struct fib_update_operation op(DELETE, set, key);
	updates.push_back(op);
}

void tagmatch::add_set(filter_t set, const std::vector<tagmatch_key_t> & keys) {
	for (tagmatch_key_t k : keys)
		add_set(set, k);
}

void tagmatch::add_set(filter_t set, tagmatch_key_t key) {
	// I should already have a map in the disk, that will
	// be loaded on the next consolidate() call... For
	// now, just enqueue the request for later
	struct fib_update_operation op(ADD, set, key);
	updates.push_back(op);
}

static uint32_t partition_size = 1000;
static uint32_t partitioning_threads = 1; 

void tagmatch::consolidate(uint32_t psize, uint32_t threads) {
	partition_size = psize;
	partitioning_threads = threads;
	consolidate();
}

void tagmatch::consolidate() {
	if (already_consolidated) {
		std::cerr << std::endl << "\tReading previous fib from disk...";
		// Here I should read the map from a file
		std::ifstream cache_file_in(".map.tmp");
		fib_entry f;
		while(f.read_binary(cache_file_in)) {
			for (tagmatch_key_t k : f.keys) {
				fib_map[f.filter].insert(k);
			}
		}
		cache_file_in.close();
		std::cerr << " done";
	}

	// Apply changes!
	//
	std::cerr << std::endl << "\tApplying changes...";
	for (struct fib_update_operation fuo : updates) {
		if (fuo.type == ADD) {
			fib_map[fuo.filter].insert(fuo.key);
		}
		else if (fuo.type == DELETE) {
			std::set<tagmatch_key_t> keys = fib_map.at(fuo.filter);
			if (keys.size() == 1) {
				fib_map.erase(fuo.filter);
			}
			else {
				fib_map.at(fuo.filter).erase(fuo.key);
			}
		}
	}

	updates.clear();
	std::cerr << " done" << std::endl;
	
	std::cerr << "\tUpdating on disk cache...";
	std::ofstream cache_file_out("map.tmp");
	// Flush the set-key map
	for (std::pair<filter_t, std::set<tagmatch_key_t>> fk : fib_map) {
		std::vector<tagmatch_key_t> keys;
		for (tagmatch_key_t k : fk.second) {
			keys.push_back(k);
		}
		fib_entry fe(fk.first, keys);
		// Write this fib_entry to a file (I'm filling the on disk map)
		fe.write_binary(cache_file_out);
		partitioner::add_set(fk.first, keys);
	}
	cache_file_out.close();
	// clear the map and destroy its elements
	fib_map.clear();
	std::cerr << " done" << std::endl;

	std::cerr << "\tConsolidating...";
	partitioner::consolidate(partition_size, partitioning_threads);
	std::cerr << " done";

	// Pass these things to the matcher!
	//
	std::vector<partition_prefix> * prefixes;
	std::vector<partition_fib_entry> * filters;
	partitioner::get_consolidated_prefixes_and_filters(&prefixes, &filters);	
	for (partition_prefix pp : *prefixes)
		add_partition(pp.partition, pp.filter);

	for (partition_fib_entry pfe : *filters) {
		add_filter(pfe.partition, pfe.filter, pfe.keys.begin(), pfe.keys.end());
	}
	already_consolidated = true;
	delete prefixes;
	delete filters;
	partitioner::clear();
}

void tagmatch::add_partition(unsigned int id, const filter_t & mask) {
	front_end::add_prefix(id, mask);
	back_end::add_partition(id, mask);
}

void tagmatch::add_filter(unsigned int partition_id, const filter_t & f, 
				   std::vector<tagmatch_key_t>::const_iterator begin,
				   std::vector<tagmatch_key_t>::const_iterator end) {
	back_end::add_filter(partition_id, f, begin, end);
}

void tagmatch::set_latency_limit_ms(unsigned int latency_limit) {
	front_end::set_latency_limit_ms(latency_limit);
}

int thread_count;
int gpu_count;

void tagmatch::start() {
	back_end::start(gpu_count);
	front_end::start(thread_count);
}

void tagmatch::start(unsigned int ccount, unsigned int gcount) {
	// This calls initialize on both the frontend and the backend
	thread_count = ccount;
	gpu_count = gcount;
	
	back_end::start(gpu_count);
	front_end::start(thread_count);
}

void tagmatch::stop() {
	front_end::stop(gpu_count);
	back_end::stop(gpu_count);
}

void tagmatch::clear() {
	front_end::clear();
	back_end::clear(gpu_count);
}

void tagmatch::match(packet * p, match_handler * h) noexcept {
	p->configure_match(false, h);
	front_end::match(p);
	h->match_hold();
}

void tagmatch::match_unique(packet * p, match_handler * h) noexcept {
	p->configure_match(true, h);
	front_end::match(p);
	h->match_hold();
}
