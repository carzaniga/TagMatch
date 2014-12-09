#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>
#include <map>
#include <algorithm>

#include "packet.hh"
#include "fib.hh"

using std::vector;
using std::map;
using std::ifstream;
using std::string;
using std::istringstream;
using std::getline;
using std::cout;
using std::cerr;
using std::endl;


static void print_usage(const char * progname) {
	cout << "usage: " << progname 
		 << " [-b] [in=<filename>] (freq | count)"
		" | map=<permutation-filename>"
		 << endl
		 << "options:" << endl
		 << "\t-b\t: read input in binary format (default is ASCII)" << endl;
}

static unsigned int permutation[filter_t::WIDTH];

static int read_permutation(const char * permutation_fname) {
	ifstream is(permutation_fname);
	string line;
	if (!is)
		return -1;

	unsigned int new_bit_pos = 0;
	while(getline(is, line) && new_bit_pos < filter_t::WIDTH) {
		istringstream line_s(line);
		string command;
		line_s >> command;
		if (command != "p")
			continue;

		unsigned int old_bit_pos;

		line_s >> old_bit_pos;

		permutation[old_bit_pos] = new_bit_pos;
		++new_bit_pos;
	}
	is.close();
	return new_bit_pos;
}

static void apply_permutation(filter_t & f) {
	filter_t f_tmp;
	f_tmp.clear();

	for(unsigned int p = f.next_bit(0); p < f.WIDTH; p = f.next_bit(p+1))
		f_tmp.set_bit(permutation[p]);

	f.assign(f_tmp.begin());
}

static int map_filters(std::istream & input, std::ostream & output, bool binary_format) {
	fib_entry f;
	if (binary_format) {
		while(f.read_binary(input)) {
			apply_permutation(f.filter);
			f.write_binary(output);
		}
	} else {
		while(f.read_ascii(input)) {
			apply_permutation(f.filter);
			f.write_ascii(output);
		}
	}
	return 0;
}

static int compute_freqs(std::istream & input, std::ostream & output, bool binary_format) {

	unsigned int freqs[filter_t::WIDTH] = { 0 };
	unsigned int count = 0;
	
	fib_entry f;

	if (binary_format) {
		while(f.read_binary(input)) {
			for(unsigned int p = f.filter.next_bit(0); p < filter_t::WIDTH; p = f.filter.next_bit(p+1))
				freqs[p] += 1;

			count += 1;
		}
	} else {
		while(f.read_ascii(input)) {
			for(unsigned int p = f.filter.next_bit(0); p < filter_t::WIDTH; p = f.filter.next_bit(p+1))
				freqs[p] += 1;

			count += 1;
		}
	}

	for(unsigned int i = 0; i < filter_t::WIDTH; ++i)
		output << "p " << i << ' ' << freqs[i] << ' ' << (100.0 * freqs[i] / count) << endl;

	return 0;
}

static int count_filters(std::istream & input, std::ostream & output, bool binary_format) {
	unsigned int count = 0;
	
	fib_entry f;
	if (binary_format) {
		while(f.read_binary(input))
			++count;
	} else {
		while(f.read_ascii(input))
			++count;
	}
	output << count << endl;

	return 0;
}

int main(int argc, const char * argv[]) {
	bool binary_format = false;
	const char * input_fname = nullptr;
	const char * output_fname = nullptr;
	int (*analysis)(std::istream &, std::ostream &, bool) = compute_freqs;

	for(int i = 1; i < argc; ++i) {
		if (strcmp(argv[i],"-b")==0) {
			binary_format = true;
		} else 
		if (strncmp(argv[i],"in=",3)==0) {
			input_fname = argv[i] + 3;
		} else 
		if (strncmp(argv[i],"out=",4)==0) {
			output_fname = argv[i] + 4;
		} else 
		if (strncmp(argv[i],"map=",4)==0) {
			if (read_permutation(argv[i] + 4) < 0) 
				return -1;
			analysis = map_filters;
		} else 
		if (strcmp(argv[i],"freq")==0) {
			analysis = compute_freqs;
		} else 
		if (strcmp(argv[i],"count")==0) {
			analysis = count_filters;
		} else {
			print_usage(argv[0]);
			return 1;
		}
	}

	int res = 0;
	if (input_fname != nullptr) {
		std::ifstream input_file(input_fname);
		if (!input_file) {
			std::cerr << "could not open input file " << input_fname << std::endl;
			return 1;
		}
		if (output_fname != nullptr) {
			std::ofstream output_file(output_fname);
			if (!output_file) {
				std::cerr << "could not open output file " << output_fname << std::endl;
				return 1;
			}
			res = analysis(input_file, output_file, binary_format);
			output_file.close();
		} else {
			res = analysis(input_file, std::cout, binary_format);
		}
		input_file.close();
	} else {
		if (output_fname != nullptr) {
			std::ofstream output_file(output_fname);
			if (!output_file) {
				std::cerr << "could not open output file " << output_fname << std::endl;
				return 1;
			}
			res = analysis(std::cin, output_file, binary_format);
			output_file.close();
		} else {
			res = analysis(std::cin, std::cout, binary_format);
		}
	}
	return res;
}
