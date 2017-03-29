//
// This program is part of TagMatch, a subset-matching engine based on
// a hybrid CPU/GPU system.  TagMatch is also related to TagNet, a
// tag-based information and network system.
//
// This program implememnts the off-line partitioning algorithm of
// TagMatch.  The input is a set of Bloom filters, each representing a
// set of tags, and associated with a set of keys.  So:
//
// INPUT:
//
//    BF_1, k1, k2, ..
//    BF_2, k3, k4, ...
//    ...
//
// The program partitions the set of filters (and associated keys)
// into partitions so that all the filters in a partition share a
// common "mask" (a non-empty set of one-bits).  So, the output consists of two files:
//
// OUTPUT:
//
// Filters: This is the same as the input where each filter is also
// assigned a partition id:
//
//    Filters:
//    BF_1, partition-id_1, k1, k2, ..
//    BF_2, partition-id_2, k3, k4, ...
//    ...
//
// Partitions: This is the set of partitions, characterized by
// partition id, mask, size, etc.
//
//    Partitions:
//    partition-id_1, mask_1, size_1
//    partition-id_2, mask_2, size_2
//    ...
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

#include "partitioner.hh"

using std::vector;
using std::endl;
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

static void print_usage(const char* progname) {
	std::cerr << "usage: " << progname
	<< " [<params>...]\n"
	"\n  params: any combination of the following:\n"
	"     [m=<N>]         :: maximum size for each partition (default=100)\n"
	"     [t=<N>]         :: size of the thread pool (default=4)\n"
	"     [p=<filename>]  :: output for partitions, '-' means stdout (default=OFF)\n"
	"     [f=<filename>]  :: output for filters, '-' means stdout (default=OFF)\n"
	"     [in=<filename>]  :: input for filters (default=stdin)\n"
	"     [-a]  :: ascii input\n"
	"     [-b]  :: binary input\n"
	<< std::endl;
}


static void read_filters(std::istream & input, bool binary_format) {
	fib_entry f;
	if (binary_format) {
		while(f.read_binary(input)) {
			partitioner::add_set(f.filter, f.keys);
		}
	} else {
		while(f.read_ascii(input)) {
			partitioner::add_set(f.filter, f.keys);
		}
	}
}


int main(int argc, const char* argv[]) {
	const char* partitions_fname = nullptr;
	const char* filters_fname = nullptr;
	const char* input_fname = nullptr;

	unsigned int max_size = 200000;
	unsigned int thread_count = 4;
	bool binary_format = false;

	
	for (int i = 1; i < argc; ++i) {
		if (sscanf(argv[i], "m=%u", &max_size) || sscanf(argv[i], "N=%u", &max_size))
			continue;
		if (sscanf(argv[i], "t=%u", &thread_count))
			continue;
		if (strcmp(argv[i], "-a") == 0) {
			binary_format = false;
			continue;
		}
		if (strncmp(argv[i], "p=", 2) == 0) {
			partitions_fname = argv[i] + 2;
			continue;
		}
		if (strncmp(argv[i], "f=", 2) == 0) {
			filters_fname = argv[i] + 2;
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
		print_usage(argv[0]);
		return 1;
	}
	if (!partitions_fname 
			|| !filters_fname
			|| !input_fname) {
		print_usage(argv[0]);
		return 1;
	}
	
	std::ofstream partitions_file;
	std::ofstream filters_file;
	std::ifstream input_file;

	std::ostream* partitions_output = nullptr;
	std::ostream* filters_output = nullptr;
	
	if (input_fname != nullptr) {
		std::ifstream input_file(input_fname);
		if (!input_file) {
			std::cerr << "could not open input file " << input_fname << std::endl;
			return 1;
		}
		std::cerr << "Reading filters..." << std::flush;
		read_filters(input_file, binary_format);
		input_file.close();
	} else {
		std::cerr << "Reading filters..." << std::flush;
		read_filters(std::cin, binary_format);
	}
	std::cerr << "\t\t\t" << std::setw(12) << "fib.size()" << " filters." << endl;
	
	if (partitions_fname) {
		if (strcmp(partitions_fname, "-") == 0) {
			partitions_output = &std::cout;
		} else {
			partitions_file.open(partitions_fname);
			if (!partitions_file) {
				std::cerr << "error opening partitions file " << partitions_fname
				<< std::endl;
				return -1;
			}
			partitions_output = &partitions_file;
		}
	}
	
	if (filters_fname) {
		if (strcmp(filters_fname, "-") == 0) {
			filters_output = &std::cout;
		} else {
			filters_file.open(filters_fname);
			if (!filters_file) {
				std::cerr << "error opening filters file " << filters_fname
				<< std::endl;
				return -1;
			}
			filters_output = &filters_file;
		}
	}
	
	std::cerr << "Balanced partitioning..." << std::flush;

	high_resolution_clock::time_point start = high_resolution_clock::now();

	partitioner::consolidate(max_size, thread_count);

	high_resolution_clock::time_point stop = high_resolution_clock::now();

	std::cerr << "\t\t" << std::setw(12)
			  << duration_cast<milliseconds>(stop - start).count() << " ms." << std::endl;

	std::vector<partition_prefix> * partitions = partitioner::get_consolidated_prefixes();
	std::vector<partition_fib_entry> * partitioned_filters = partitioner::get_consolidated_filters();

	// get partitions
	std::cerr << "Number of partitions:\t\t\t" << std::setw(12) << partitions->size() << " partitions." << endl;
	std::cerr << "Number of filters:\t\t\t" << std::setw(12) << partitioned_filters->size() << " partitions." << endl;

	for (partition_prefix pp : *partitions) {
		if (partitions_output) {
			if (binary_format)
				pp.write_binary(*partitions_output);
			else
				pp.write_ascii(*partitions_output);
		}
	}
			
	for (partition_fib_entry pfe : *partitioned_filters) {	
		if (filters_output) {
			if (binary_format)
				pfe.write_binary(*filters_output);
			else
				pfe.write_ascii(*filters_output);
		}
	}
	partitioner::clear();
	if (partitions_output == &partitions_file) partitions_file.close();
	if (filters_output == &filters_file) filters_file.close();
}
