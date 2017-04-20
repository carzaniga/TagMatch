#include "tagmatch.hh"

#include <map>
#include <set>
#include <fstream>
#include <string>

#include "tagmatch.hh"
#include "fib.hh"
#include "partitioning.hh"
#include "front_end.hh"
#include "back_end.hh"

using std::vector;
using std::set;

static bool already_consolidated = false;

// TODO: do we want the library to be thread safe? We need to
// add some locks in that case
//
class keyset_change {
public:
	keyset_change(): to_add(), to_remove() {};

	void add(tagmatch_key_t k) { to_add.insert(k); }
	void remove(tagmatch_key_t k) { to_remove.insert(k); }

	void apply(vector<tagmatch_key_t> & keys);

private:
	set<tagmatch_key_t> to_add;
	set<tagmatch_key_t> to_remove;
};

void keyset_change::apply(vector<tagmatch_key_t> & keys) {

	vector<tagmatch_key_t>::iterator j = keys.begin();
	vector<tagmatch_key_t>::iterator i = keys.begin();

	for(i = keys.begin(); i != keys.end(); ++i) {
		set<tagmatch_key_t>::iterator to_remove_itr;
		if ((to_remove_itr = to_remove.find(*i)) == to_remove.end()) {
			if (i != j)
				*j = *i;
			++j;
		} else {
			to_remove.erase(to_remove_itr);
		}
		to_add.erase(*i);
		if (to_add.empty() && to_remove.empty())
			break;
	}
	set<tagmatch_key_t>::iterator to_add_itr = to_add.begin();
	while (j != i && to_add_itr != to_add.end()) 
		*j++ = *to_add_itr++;

	while (to_add_itr != to_add.end()) 
		keys.emplace_back(*to_add_itr++);
}

typedef std::map<filter_t, keyset_change> changes_map;
static changes_map updates;

void tagmatch::remove(const filter_t & s, tagmatch_key_t k) {
	updates[s].remove(k);
}

void tagmatch::add(const filter_t & s, const vector<tagmatch_key_t> & keys) {
	keyset_change & c = updates[s];
	for (tagmatch_key_t k : keys)
		c.add(k);
}

void tagmatch::add(const filter_t & s, tagmatch_key_t key) {
	updates[s].add(key);
}

static std::string db_filename("tagmatch.db");

const char * tagmatch::get_database_filename() {
	return db_filename.c_str();
}

void set_database_filename(const char * name) {
	db_filename = name;
}

static uint32_t partition_size = 1000;
static uint32_t partitioning_threads = 1; 

void tagmatch::consolidate(uint32_t psize, uint32_t threads) {
	partition_size = psize;
	partitioning_threads = threads;
	consolidate();
}

void tagmatch::consolidate() {
	vector<partition_fib_entry *> fib;

	if (already_consolidated) {
		std::ifstream cache_file_in(db_filename);
		if (!cache_file_in) {
			std::cerr << "could not open database file: " << db_filename << std::endl;
		} else {
			fib_entry f;
			while(f.read_binary(cache_file_in)) {
				changes_map::iterator ci = updates.find(f.filter);
				if (ci != updates.end()) {
					ci->second.apply(f.keys);
					updates.erase(ci);
				}

				if (!f.keys.empty()) 
					fib.push_back(new partition_fib_entry(f));
			}
		}
		cache_file_in.close();
	}

	for (changes_map::iterator ci = updates.begin(); ci != updates.end(); ++ci) {
		fib_entry f;
		f.filter = ci->first;
		ci->second.apply(f.keys);

		if (!f.keys.empty()) 
			fib.push_back(new partition_fib_entry(f));
	}

	updates.clear();
	
	std::ofstream cache_file_out(db_filename);
	if (!cache_file_out) {
		std::cerr << "could not open database file: " << db_filename << std::endl;
	} else {
		// Flush the set-key map
		for (partition_fib_entry * fe : fib) {
			// Write this fib_entry to a file (I'm filling the on disk map)
			fe->write_binary(cache_file_out);
		}
		cache_file_out.close();
	}

	// Compute the partitioning and passes the results to the matcher!
	//
	vector<partition_prefix> masks;
	partitioning::set_maxp(partition_size);
	partitioning::set_cpu_threads(partitioning_threads);
	partitioning::balanced_partitioning(fib, masks);

	for (partition_prefix pp : masks) {
		front_end::add_partition(pp.partition, pp.filter);
		back_end::add_partition(pp.partition, pp.filter);
	}

	for (partition_fib_entry * pfe : fib) {
		back_end::add_filter(pfe->partition, pfe->filter, pfe->keys.begin(), pfe->keys.end());
		delete(pfe);
	}
	already_consolidated = true;
}

void tagmatch::set_latency_limit_ms(unsigned int latency_limit) {
	front_end::set_latency_limit_ms(latency_limit);
}

static int gpu_count;
static int thread_count;

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
	front_end::stop();
	back_end::stop();
}

void tagmatch::clear() {
	already_consolidated = false;
	front_end::clear();
	back_end::clear();
}

void tagmatch::match(tagmatch_query * q, match_handler * h) noexcept {
	q->match_unique = false;
	q->set_match_handler(h);
	front_end::match(q);
}

void tagmatch::match_unique(tagmatch_query * q, match_handler * h) noexcept {
	q->match_unique = true;
	q->set_match_handler(h);
	front_end::match(q);
}
