#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <atomic>
#include <climits>
#include <algorithm>

#include "packet.hh"
#include "fib.hh"

using std::vector;
using std::endl;
using std::cout;

static void print_usage(const char* progname) {
	std::cerr << "usage: " << progname
	<< " [<params>...]\n"
	"\n  params: any combination of the following:\n"
	"     [m=<N>]         :: maximum size for each partition (default=100)\n"
	"     [p=<filename>]  :: output for prefixes, '-' means stdout (default=OFF)\n"
	"     [f=<filename>]  :: output for filters, '-' means stdout (default=OFF)\n"
	"     [in=<filename>]  :: input for filters (default=stdin)\n"
	"     [-a]  :: ascii input\n"
	"     [-b]  :: binary input\n"
	<< std::endl;
}

typedef vector<fib_entry *> fib_t;
static fib_t fib;

static int partition_counter = 0;
static bool binary_format = false;

static std::ostream* prefixes_output = nullptr;
static std::ostream* filters_output = nullptr;

static void read_filters(std::istream & input, bool binary_format) {
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

struct partition_candidate {
	fib_t::iterator begin;
	fib_t::iterator end;
	filter_t mask;
	filter_t used_bits;
	struct partition_candidate * next;

	partition_candidate(fib_t::iterator b, fib_t::iterator e, partition_candidate * n)
		: begin(b), end(e), mask(), used_bits(), next(n) {};

	partition_candidate(fib_t::iterator b, fib_t::iterator e,
						const filter_t & m, const filter_t & ub, partition_candidate * n)
		: begin(b), end(e), mask(m), used_bits(ub), next(n) {};
};

static unsigned int distance(unsigned int a, unsigned int b) {
	return (a > b) ? (a - b) : (b - a);
}

struct has_bit {
	unsigned int b;

	has_bit(unsigned int bitpos): b(bitpos) {};
	bool operator()(const fib_entry * f) { return ! f->filter[b]; }
};

partition_candidate * balanced_partitioning(size_t max_p) {
	partition_candidate * PT = nullptr;
	partition_candidate * Q = new partition_candidate(fib.begin(), fib.end(), nullptr);
	while (Q) {
		size_t P_size = Q->end - Q->begin; 
		if (P_size <= max_p) {
			partition_candidate * tmp = Q;
			Q = Q->next;
			tmp->next = PT;
			PT = tmp;
			partition_counter += 1;
		} else {
			unsigned int freq[filter_t::WIDTH] = {0};
			for(fib_t::const_iterator i = Q->begin; i < Q->end; ++i) 
				for(unsigned int b = (*i)->filter.next_bit(0);
					b < filter_t::WIDTH; b = (*i)->filter.next_bit(b + 1))
					freq[b] += 1;
			
			unsigned int pivot;
			for(pivot = 0; pivot < filter_t::WIDTH; ++pivot)
				if (! Q->used_bits[pivot])
					break;
			unsigned int min_dist = distance(P_size / 2, freq[pivot]);
			for(unsigned int b = pivot + 1; b < filter_t::WIDTH; ++b) {
				unsigned d = distance(P_size / 2, freq[b]); 
				if (d < min_dist) {
					pivot = b;
					if (d == 0) break;
					min_dist = d;
				}
			}
			vector<fib_entry *>::iterator middle;
			middle = std::stable_partition(Q->begin, Q->end, has_bit(pivot));
			Q->used_bits.set_bit(pivot);
			partition_candidate * P0;
			P0 = new partition_candidate(Q->begin, middle, Q->mask, Q->used_bits, Q);
			Q->begin = middle;
			Q->mask.set_bit(pivot);
			Q = P0;
		}
	}
	return PT;
}
	
int main(int argc, const char* argv[]) {
	const char* prefixes_fname = nullptr;
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
			prefixes_fname = argv[i] + 2;
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
	
	std::ofstream prefixes_file;
	std::ofstream filters_file;
	std::ifstream input_file;
	
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
	std::cerr << "\t\t\t" << std::setw(12) << fib.size() << " filters." << endl;
	
	if (prefixes_fname) {
		if (strcmp(prefixes_fname, "-") == 0) {
			prefixes_output = &std::cout;
		} else {
			prefixes_file.open(prefixes_fname);
			if (!prefixes_file) {
				std::cerr << "error opening prefixes file " << prefixes_fname
				<< std::endl;
				return -1;
			}
			prefixes_output = &prefixes_file;
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
	partition_candidate * PT = balanced_partitioning(max_size);
	std::cerr << "\t\t" << std::setw(12) << partition_counter << " partitions." << endl;

	while(PT) {
		partition_candidate * tmp = PT;
		PT = PT->next;
		delete(tmp);
	}
	for(fib_t::iterator i = fib.begin(); i != fib.end(); ++i)
		delete(*i);
	
	if (prefixes_output == &prefixes_file) prefixes_file.close();
	if (filters_output == &filters_file) filters_file.close();
}
