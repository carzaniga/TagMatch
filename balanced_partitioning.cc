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
#include <algorithm>
#include <chrono>

#include "packet.hh"
#include "fib.hh"

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
	"     [p=<filename>]  :: output for partitions, '-' means stdout (default=OFF)\n"
	"     [f=<filename>]  :: output for filters, '-' means stdout (default=OFF)\n"
	"     [in=<filename>]  :: input for filters (default=stdin)\n"
	"     [-a]  :: ascii input\n"
	"     [-b]  :: binary input\n"
	<< std::endl;
}

// We read and store fib_entry objects (see fib.hh).  We allocate each
// individual object, and then store the pointers to the objects in a
// vector that serves as an index.
//
typedef vector<fib_entry *> fib_t;

static bool binary_format = false;

static std::ostream* partitions_output = nullptr;
static std::ostream* filters_output = nullptr;

static void read_filters(fib_t & fib, std::istream & input, bool binary_format) {
	fib_entry * f = new fib_entry();
	if (binary_format) {
		while(f->read_binary(input)) {
			fib.push_back(f);
			f = new fib_entry();
		}
		delete(f);
	} else {
		while(f->read_ascii(input)) {
			fib.push_back(f);
			f = new fib_entry();
		}
		delete(f);
	}
}

// We work with a list of partition candidates, which we then
// recursively partition.  Each partition candidate consists of a
// sequence of contiguous filters in the index (fib).
//
struct partition_candidate {
	fib_t::iterator begin;		// beginning of contiguous sequence in fib
	fib_t::iterator end;		// end of contiguous sequence in fib
	filter_t mask;				// mask of pivot bits that identify this partition
	filter_t used_bits;			// set of ALL pivot bits used so far
	unsigned int freq[filter_t::WIDTH]; // frequencies of one bits in the filters
	struct partition_candidate * next;

	partition_candidate(fib_t::iterator b, fib_t::iterator e, partition_candidate * n = nullptr)
		: begin(b), end(e), mask(), used_bits(), next(n) {};

	partition_candidate(fib_t::iterator b, fib_t::iterator e,
						const filter_t & m, const filter_t & ub)
		: begin(b), end(e), mask(m), used_bits(ub) {};

	void compute_frequencies() {
		std::memset(freq, 0, sizeof(freq));
		for(fib_t::const_iterator i = begin; i != end; ++i)
			for(unsigned int b = (*i)->filter.next_bit(0); b < filter_t::WIDTH; b = (*i)->filter.next_bit(b + 1))
				freq[b] += 1;
	}

	void subtract_frequencies(const partition_candidate & x) {
		for(unsigned int b = 0; b < filter_t::WIDTH; ++b)
			freq[b] -= x.freq[b];
	}

	size_t size() const {
		return end - begin;
	}
};

static unsigned int distance(unsigned int a, unsigned int b) {
	return (a > b) ? (a - b) : (b - a);
}

// Predicate object used in stable partitioning.
//
struct has_zero_bit {
	unsigned int b;

	has_zero_bit(unsigned int bitpos): b(bitpos) {};
	bool operator()(const fib_entry * f) { return ! f->filter[b]; }
};

// Processes a partition candidate P0 that is already smaller than the
// maximum partition size, but that is defined by an all-zero mask.
// Looks for bit positions that are common to all filters in P0, and
// if necessary, further partition P0 until such bit positions can be
// found.
//
static partition_candidate * nonzero_mask_partitioning(partition_candidate * P0) {
	for(;;) {
		P0->compute_frequencies();
		unsigned int max_freq = P0->freq[0];
		unsigned int pivot = 0;
		for(unsigned int b = 1; b < filter_t::WIDTH; ++b) {
			if (P0->freq[b] > max_freq) {
				pivot = b;
				max_freq = P0->freq[b];
			}
			if (P0->freq[b] == P0->size())
				P0->mask.set_bit(b);
		}
		if (max_freq == P0->size())
			return P0;

		vector<fib_entry *>::iterator middle = std::stable_partition(P0->begin, P0->end,
																	 has_zero_bit(pivot));
		partition_candidate * P1 = P0;
		P0 = new partition_candidate(P0->begin, middle, P0);
		P1->begin = middle;
		P1->mask.set_bit(pivot);
	}
}

partition_candidate * balanced_partitioning(fib_t::iterator begin, fib_t::iterator end,
											size_t max_p) {
	partition_candidate * Q = new partition_candidate(begin, end);
	if (Q->size() <= max_p)
		return Q;

	partition_candidate * PT = nullptr;
	Q->compute_frequencies();
	while (Q) {
		unsigned int pivot;
		for(pivot = 0; pivot < filter_t::WIDTH; ++pivot)
			if (! Q->used_bits[pivot])
				break;
		unsigned int min_dist = distance(Q->size() / 2, Q->freq[pivot]);
		for(unsigned int b = pivot + 1; b < filter_t::WIDTH; ++b) {
			unsigned d = distance(Q->size() / 2, Q->freq[b]); 
			if (d < min_dist) {
				pivot = b;
				if (d == 0) break;
				min_dist = d;
			}
		}
		vector<fib_entry *>::iterator middle = std::stable_partition(Q->begin, Q->end,
																	 has_zero_bit(pivot));
		partition_candidate * P0;
		partition_candidate * P1 = Q;
		Q = Q->next;
		P1->used_bits.set_bit(pivot);
		P0 = new partition_candidate(P1->begin, middle, P1->mask, P1->used_bits);
		P1->begin = middle;
		P1->mask.set_bit(pivot);
		if (P0->size() > max_p && P1->size() > max_p) {
			P0->compute_frequencies();
			P1->subtract_frequencies(*P0);
			P1->next = Q;
			P0->next = P1;
			Q = P0;
		} else if (P1->size() > max_p) {
			P1->compute_frequencies();
			P1->next = Q;
			Q = P1;
			P0->next = PT;
			if (P0->mask.is_empty())
				P0 = nonzero_mask_partitioning(P0);
			PT = P0;
		} else if (P0->size() > max_p) {
			P0->compute_frequencies();
			P0->next = Q;
			Q = P0;
			P1->next = PT;
			PT = P1;
		} else {
			P1->next = PT;
			P0->next = P1;
			if (P0->mask.is_empty())
				P0 = nonzero_mask_partitioning(P0);
			PT = P0;
		}
	}
	return PT;
}

int main(int argc, const char* argv[]) {
	const char* partitions_fname = nullptr;
	const char* filters_fname = nullptr;
	const char* input_fname = nullptr;
	static unsigned int max_size = 200000;
	
	for (int i = 1; i < argc; ++i) {
		if (sscanf(argv[i], "m=%u", &max_size) || sscanf(argv[i], "N=%u", &max_size))
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
	
	std::ofstream partitions_file;
	std::ofstream filters_file;
	std::ifstream input_file;

	static fib_t fib;
	
	if (input_fname != nullptr) {
		std::ifstream input_file(input_fname);
		if (!input_file) {
			std::cerr << "could not open input file " << input_fname << std::endl;
			return 1;
		}
		std::cerr << "Reading filters..." << std::flush;
		read_filters(fib, input_file, binary_format);
		input_file.close();
	} else {
		std::cerr << "Reading filters..." << std::flush;
		read_filters(fib, std::cin, binary_format);
	}
	std::cerr << "\t\t\t" << std::setw(12) << fib.size() << " filters." << endl;
	
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

	partition_candidate * PT = balanced_partitioning(fib.begin(), fib.end(), max_size);

	high_resolution_clock::time_point stop = high_resolution_clock::now();

	std::cerr << "\t\t" << std::setw(12)
			  << duration_cast<milliseconds>(stop - start).count() << " ms." << std::endl;

	partition_id_t pid = 0;

	while(PT) {
		partition_prefix partition;

		partition.filter.fill();
		partition.length = filter_t::WIDTH;
		partition.partition = pid++;
		partition.size = PT->end - PT->begin;

		for(fib_t::iterator i = PT->begin; i != PT->end; ++i) {
			partition_fib_entry f;
			f.filter = (*i)->filter;
			partition.filter &= f.filter;
			f.ti_pairs = std::move((*i)->ti_pairs);
			f.partition = partition.partition;
			if (filters_output) {
				if (binary_format)
					f.write_binary(*filters_output);
				else
					f.write_ascii(*filters_output);
			}
		}
		if (partitions_output) {
			if (binary_format)
				partition.write_binary(*partitions_output);
			else
				partition.write_ascii(*partitions_output);
		}
		partition_candidate * tmp = PT;
		PT = PT->next;
		delete(tmp);
	}

	std::cerr << "Number of partitions:\t\t\t" << std::setw(12) << pid << " partitions." << endl;

	for(fib_t::iterator i = fib.begin(); i != fib.end(); ++i)
		delete(*i);
	fib.clear();
	
	if (partitions_output == &partitions_file) partitions_file.close();
	if (filters_output == &filters_file) filters_file.close();
}
