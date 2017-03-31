#include "tagmatch.hh"

#include <map>
#include <set>
#include <fstream>

static bool already_consolidated = false;

std::map<filter_t, std::set<tagmatch_key_t>> fib_map;
std::map<filter_t, std::set<tagmatch_key_t>> additions;
std::map<filter_t, std::set<tagmatch_key_t>> deletions;

void tagmatch::delete_set(filter_t set, tagmatch_key_t key) {
	// I should already have a map in the disk, that will
	// be loaded on the next consolidate() call... For
	// now, just enqueue the request for later
	deletions[set].insert(key);
}

void tagmatch::add_set(filter_t set, tk_vector keys) {
	for (tagmatch_key_t k : keys)
		add_set(set, k);
}

void tagmatch::add_set(filter_t set, tagmatch_key_t key) {
	// I should already have a map in the disk, that will
	// be loaded on the next consolidate() call... For
	// now, just enqueue the request for later
	additions[set].insert(key);
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
		std::ifstream cache_file_in("map.tmp");
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
	// TODO: the changes should have an order, so this thing needs to be refactored into
	// some kind of log with timestamps, or with an order preserving container...
	//
	std::cerr << std::endl << "\tApplying changes...";
	for (std::pair<filter_t, std::set<tagmatch_key_t>> a : additions) {
		fib_map[a.first].insert(a.second.begin(), a.second.end());
	}
	additions.clear();
	for (std::pair<filter_t, std::set<tagmatch_key_t>> a : deletions) {
		std::set<tagmatch_key_t> keys = fib_map.at(a.first);
		keys.erase(a.second.begin(), a.second.end());
		if (keys.empty())
			fib_map.erase(a.first);
	}
	deletions.clear();
	std::cerr << " done" << std::endl;
	
	std::cerr << "\tUpdating on disk cache...";
	std::ofstream cache_file_out("map.tmp");
	// Flush the set-key map
	// TODO: This does NOT check for duplicated sets, that may be added
	// with the *add_set(filter_t set, tagmatch_key_t key)* api
	//
	for (std::pair<filter_t, std::set<tagmatch_key_t>> fk : fib_map) {
		tk_vector keys;
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
	delete prefixes;
	delete filters;
	partitioner::clear();
	already_consolidated = true;
}

void tagmatch::add_partition(unsigned int id, const filter_t & mask) {
		// TODO: the last parameter is unused... remove it?
		front_end::add_prefix(id, mask, 0);
		back_end::add_partition(id, mask, 0);
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

