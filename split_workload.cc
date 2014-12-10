#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <algorithm>

#include "bitvector.hh"
#include "fib.hh"

static bool sorted_fib_entries(const fib_entry & a, const fib_entry & b) {
	return b.filter < a.filter;
}

static unsigned int MIN_K = 2;
static unsigned int max_size = 100;
static bool binary_format = true;

static unsigned int kth_leftmost_bit(const filter_t & f, unsigned int k) {
	for(unsigned int i = f.next_bit(0); i < filter_t::WIDTH; i = f.next_bit(i+1)) {
		if (k == 0)
			return i;
		else
			--k;
	}
	return filter_t::WIDTH;
}

void split_on_prefix(std::istream * input, 
					 std::ostream * prefixes_output, 
					 std::ostream * filters_output) {

	bool sorted = true;
	std::vector<partition_fib_entry> fib;

	for(;;) {
		fib.emplace_back();
		if (binary_format)
			fib.back().fib_entry::read_binary(*input);
		else
			fib.back().fib_entry::read_ascii(*input);

		if (!(*input)) {
			fib.pop_back();
			break;
		}
		if (sorted && fib.size() > 1)
			sorted = sorted_fib_entries(fib[fib.size() - 1], fib[fib.size() - 2]);
	}

	if (!sorted)
		std::sort(fib.begin(), fib.end(), sorted_fib_entries);

	unsigned int pid = 0;

	std::vector<partition_fib_entry>::iterator f = fib.begin();
	while (f != fib.end()) {
		partition_prefix prefix;

		prefix.filter = f->filter;
		prefix.partition = pid;
		prefix.length = filter_t::WIDTH;

		std::vector<partition_fib_entry>::iterator g = f + 1;
		std::vector<partition_fib_entry>::iterator next_f = g; 

		unsigned int kth_msb_pos = kth_leftmost_bit(f->filter, MIN_K);
		
		while(g != fib.end() && (g - f) < max_size) {
			unsigned int msd = f->filter.leftmost_diff(g->filter);

			if (msd < prefix.length) {
				if (kth_msb_pos >= msd) {
					next_f = g;
					++prefix.length;
					break;
				}
				prefix.length = msd;
				next_f = g;
			}
			++g;
		}
		if (g == fib.end()) {
			next_f = g;
			--prefix.length;
		}

		prefix.size = next_f - f;

		if (prefixes_output) {
			if (binary_format) {
				prefix.write_binary(*prefixes_output);
			} else {
				prefix.write_ascii(*prefixes_output);
			}
		}
		if (filters_output) {
			while(f != next_f) {
				f->partition = pid;
				if (binary_format)
					f->write_binary(*filters_output);
				else
					f->write_ascii(*filters_output);
				++f;
			}
		} else {
			f = next_f;
		}
		++pid;
	}
}

int main(int argc, const char * argv[]) {

	const char * prefixes_fname = nullptr;
	const char * filters_fname = nullptr;
	const char * input_fname = nullptr;

	for(int i = 1; i < argc; ++i) {
		if (sscanf(argv[i],"m=%u", &max_size) || sscanf(argv[i],"N=%u", &max_size))
			continue;

		if (sscanf(argv[i],"k=%u", &MIN_K) || sscanf(argv[i],"K=%u", &MIN_K))
			continue;

		if (strncmp(argv[i],"in=",3) == 0) {
			input_fname = argv[i] + 3;
			continue;
		}
		if (strncmp(argv[i],"p=",2) == 0) {
			prefixes_fname = argv[i] + 2;
			continue;
		}
		if (strncmp(argv[i],"f=",2) == 0) {
			filters_fname = argv[i] + 2;
			continue;
		}
		if (strcmp(argv[i], "-a") == 0) {
			binary_format = false;
			continue;
		} 
		if (strcmp(argv[i], "-b") == 0) {
			binary_format = true;
			continue;
		}

		std::cerr << "usage: " << argv[0] << " [<params>...]\n"
			"\n  params: any combination of the following:\n"
			"     [m=<N>]         :: maximum size for each partition (default=100)\n"
			"     [k=<N>]         :: minimum hamming weight for each partition (default=2)\n"
			"     [p=<filename>]  :: output for prefixes, '-' means stdout (default=OFF)\n"
			"     [f=<filename>]  :: output for filters, '-' means stdout (default=OFF)\n"
			"     [in=<filename>]  :: input for filters (default=stdin)\n"
				  << std::endl;
		return 1;
	}

	std::ostream * prefixes_output = nullptr;
	std::ostream * filters_output = nullptr;
	std::istream * input = nullptr;

	std::ofstream prefixes_file;
	std::ofstream filters_file;
	std::ifstream input_file;

	if (filters_fname) {
		if (strcmp(filters_fname, "-")==0) {
			filters_output = &std::cout;
		} else {
			filters_file.open(filters_fname);
			if (!filters_file) {
				std::cerr << "error opening filters file "  << filters_fname << std::endl;
				return -1;
			}
			filters_output = &filters_file;
		}
	}

	if (prefixes_fname) {
		if (strcmp(prefixes_fname, "-")==0) {
			prefixes_output = &std::cout;
		} else {
			prefixes_file.open(prefixes_fname);
			if (!prefixes_file) {
				std::cerr << "error opening prefixes file "  << prefixes_fname << std::endl;
				return -1;
			}
			prefixes_output = &prefixes_file;
		}
	}

	if (!input_fname) {
		input = &std::cin;
	} else {
		input_file.open(input_fname);
		if (!input_file) {
			std::cerr << "error opening input file "  << input_fname << std::endl;
			return -1;
		}
		input = &input_file;
	}

	split_on_prefix(input, prefixes_output, filters_output);

	if (prefixes_output == &prefixes_file)
		prefixes_file.close();

	if (filters_output == &filters_file)
		filters_file.close();

	if (input == &input_file)
		input_file.close();

	return 0;
}
