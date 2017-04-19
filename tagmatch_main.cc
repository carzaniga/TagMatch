//
// This program is part of TagMatch, a subset-matching engine based on
// a hybrid CPU/GPU system.  TagMatch is also related to TagNet, a
// tag-based information and network system.
//
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <atomic>

#include "filter.hh"
#include "key.hh"
#include "tagmatch.hh"
#include "tagmatch_query.hh"
#include "match_handler.hh"
#include "fib.hh"

using std::vector;
using std::ifstream;
using std::string;
using std::istringstream;
using std::getline;
using std::cout;
using std::cerr;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;
using std::chrono::duration_cast;
using std::vector;
using std::endl;
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

#define CHUNK_SIZE 5000000

class match_counter : public match_handler {
private:
	std::atomic<unsigned int> tot_matches;
	std::atomic<unsigned int> tot_keys;

public:
	match_counter() : tot_matches(0), tot_keys(0) { };

	virtual void process_results(query * q) {
		tot_matches++;
		tot_keys += q->output_keys.size();
	}

	unsigned int get_matches() {
		return tot_matches;
	}

	unsigned int get_keys() {
		return tot_keys;
	}
};

vector<tagmatch_query> queries;

static void print_usage(const char* progname) {
	std::cerr << "usage: " << progname
			  << " [<params>...]\n"
		"\n  params: any combination of the following:\n"
		"     [m=<N>]         :: maximum size for each partition (default=100)\n"
		"     [t=<N>]         :: size of the thread pool for the matcher (default=4)\n"
		"     [u=<N>]         :: size of the thread pool for the partitioner (default=4)\n"
		"     [g=<N>]         :: number of GPUs for the matcher (default=1)\n"
		"     [in=<filename>]  :: input for filters (default=stdin)\n"
		"     [-a]  :: ascii input\n"
		"     [-b]  :: binary input\n"
			  << std::endl;
}

static unsigned int read_queries(vector<tagmatch_query> & queries, string fname, bool binary_format, uint32_t limit) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	uint32_t res = 0;
	basic_query bq;
	if (binary_format) {
		while(bq.read_binary(is) && res < limit) {
			queries.emplace_back(bq.filter);
			++res;
		}
	} else {
		while(bq.read_ascii(is)) {
			queries.emplace_back(bq.filter);
			++res;
		}
	}
	is.close();
	return res;
}

static uint32_t read_filters(std::istream & input, bool binary_format, uint32_t howmany) {
	fib_entry f;
	uint32_t cnt = 0;
	if (binary_format) {
		while(f.read_binary(input) && ++cnt < howmany) {
			for (tagmatch_key_t k : f.keys)
				tagmatch::add_set(f.filter, k);
		}
	} else {
		while(f.read_ascii(input)) {
			tagmatch::add_set(f.filter, f.keys);
		}
	}
	return cnt;
}


int main(int argc, const char* argv[]) {
	const char* input_fname = nullptr;
	const char * queries_fname = nullptr;

	unsigned int max_size = 200000;
	unsigned int thread_count = 4;
	unsigned int thread_count_part = 4;
	unsigned int gpu_count = 1;
	bool binary_format = false;


	for (int i = 1; i < argc; ++i) {
		if (sscanf(argv[i], "m=%u", &max_size) || sscanf(argv[i], "N=%u", &max_size))
			continue;
		if (sscanf(argv[i], "t=%u", &thread_count))
			continue;
		if (sscanf(argv[i], "u=%u", &thread_count_part))
			continue;
		if (sscanf(argv[i], "gg=%u", &gpu_count))
			continue;
		if (strcmp(argv[i], "-a") == 0) {
			binary_format = false;
			continue;
		}
		if (strcmp(argv[i], "-b") == 0) {
			binary_format = true;
			continue;
		}
		if (strncmp(argv[i], "in=", 3) == 0) {
			input_fname = argv[i] + 3;
			continue;
		}
		if (strncmp(argv[i],"Q=",2)==0) {
			queries_fname = argv[i] + 2;
			continue;
		} else
			print_usage(argv[0]);
		return 1;
	}
	if (!input_fname) {
		print_usage(argv[0]);
		return 1;
	}

	std::ofstream partitions_file;
	std::ofstream filters_file;
	std::ifstream input_file;

	if (input_fname != nullptr) {
		std::ifstream input_file(input_fname);
		if (!input_file) {
			std::cerr << "could not open input file " << input_fname << std::endl;
			return 1;
		}
		uint32_t filters_read;
		std::cerr << "Reading filters..." << std::flush;
		do {
			filters_read = read_filters(input_file, binary_format, CHUNK_SIZE);
			std::cerr << " done!" << std::endl;
			std::cerr << "Balanced partitioning..." << std::flush;
			high_resolution_clock::time_point start = high_resolution_clock::now();
			// Partition size max_size, thread_count_part threads
			tagmatch::consolidate(max_size, thread_count_part);
			high_resolution_clock::time_point stop = high_resolution_clock::now();
			std::cerr << "\t\t" << std::setw(12)
					  << duration_cast<milliseconds>(stop - start).count() << " ms." << std::endl;

			std::cerr << "Reading queries...";
			read_queries(queries, queries_fname, true, 10000);
			std::cerr << " done!" << std::endl;

			std::cerr << "Setting up TagMatch...";
			tagmatch::set_latency_limit_ms(10000);
			// Use thread_count threads, gpu_count gpus
			tagmatch::start(thread_count, gpu_count);
			std::cerr << " done!" << std::endl;
			std::cerr << "Matching " << queries.size() << " queries..." << std::flush;
			match_counter counter;
			start = high_resolution_clock::now();
			for(tagmatch_query & q : queries) {
				tagmatch::match(&q, &counter);
			}
			stop = high_resolution_clock::now();

			std::cerr << "\t\t" << std::setw(12)
					  << duration_cast<milliseconds>(stop - start).count() << " ms." << std::endl;

			std::cerr << counter.get_matches() << "/" << counter.get_keys() << " matches so far..." << std::endl;

			std::cerr << "Clearing...";
			tagmatch::stop();
			tagmatch::clear();
			queries.clear();

			std::cerr << " done!" << std::endl;
			std::cerr << std::endl;
			std::cerr << "Reading filters..." << std::flush;
		} while (filters_read == CHUNK_SIZE);
		input_file.close();
	}
}
