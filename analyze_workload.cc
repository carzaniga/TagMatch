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


typedef vector<fib_entry> fib_entry_vector;
typedef map<unsigned int, fib_entry_vector > tmp_fib_map;

static tmp_fib_map tmp_fib;

static map<unsigned int, unsigned int> prefix_len;

static int read_filters(string fname, bool binary_format) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	partition_fib_entry f;
	if (binary_format) {
		while(f.read_binary(is)) {
			tmp_fib[f.partition].emplace_back(f);
			++res;
		}
	} else {
		while(f.read_ascii(is)) {
			tmp_fib[f.partition].emplace_back(f);
			++res;
		}
	}
	is.close();
	return res;
}

static int read_prefixes(const char * fname, bool binary_format) {
	ifstream is(fname);
	string line;
	if (!is)
		return -1;

	partition_prefix p;
	int res = 0;
	if (binary_format) {
		while(p.read_binary(is)) {
			prefix_len[p.partition] = p.length;
			++res;
		}
	} else {
		while(p.read_ascii(is)) {
			prefix_len[p.partition] = p.length;
			++res;
		}
	}
	is.close();
	return res;
}

static void analyze_fibs(unsigned int threshold) {
	for(auto const & pf : tmp_fib) { // reminder: map<unsigned int, fib_entry_vector> tmp_fib
		unsigned int part_id = pf.first;
		const fib_entry_vector & filters = pf.second;
		std::array<unsigned int, filter_t::WIDTH> freq;
		unsigned int prefix_pos = prefix_len[part_id];

		freq.fill(0);
		for(const fib_entry & fd : filters) {
			unsigned int offset = 0;			
			for(const block_t * b = fd.filter.begin(); b != fd.filter.end(); ++b) {
				block_t curr_block = *b;
				while (curr_block != 0) {
					int m = leftmost_bit(curr_block);
					
					if (m + offset >= prefix_pos)
						freq[m + offset] += 1;
					curr_block ^= (BLOCK_ONE << m);
				}
				offset += 64;
			}
		}
		sort(freq.begin(), freq.end(), std::greater<unsigned int>());
		std::cout << "p " << part_id << " (" << filters.size() << "):";
		for(unsigned int f : freq) {
			unsigned int p = f * 100 / filters.size();
			if (p < threshold)
				break;
			std::cout << ' ' << p;
		}
		std::cout << std::endl;
	} 
}

static void print_usage(const char * progname) {
	cout << "usage: " << progname 
		 << " freq"
		" | map=<permutation-filename>"
		" | p=<prefix-file-name> f=<filters-file-name> [options]"
		 << endl
		 << "options:" << endl
		 << "\t-Q\t: disable output of progress steps" << endl;
}

static void apply_permutation(filter_t & f, const unsigned char * bit_permutation) {
	filter_t f_tmp;
	f_tmp.clear();
	unsigned int offset = 0;			
	for(const block_t * b = f.begin(); b != f.end(); ++b) {
		block_t curr_block = *b;
		while (curr_block != 0) {
			int m = leftmost_bit(curr_block);
			f_tmp.set_bit(bit_permutation[offset + m]);
			curr_block ^= (BLOCK_ONE << m);
		}
		offset += 64;
	}
	f.assign(f_tmp.begin());
}

static std::ostream & operator << (std::ostream & os, filter_t & f) {
	for(const block_t * b = f.begin(); b != f.end(); ++b) 
		for(block_t mask = BLOCK_ONE; mask != 0; mask <<= 1)
			os << ((*b & mask) ? '1' : '0');
    return os;
}

static int map_filters(const char * permutation_fname) {
	ifstream is(permutation_fname);
	string line;
	if (!is)
		return -1;

	unsigned char permutation[filter_t::WIDTH];

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

	while(std::getline(std::cin, line)) {
		std::istringstream line_s(line);
		
		std::string command;

		line_s >> command;

		if (command != "+")
			continue;

		unsigned int iface, tree;
		std::string filter;

		line_s >> tree >> iface >> filter;

		filter_t f(filter);
		apply_permutation(f, permutation);
		cout << "+ " << tree << ' ' << iface << ' ' << f;

		while (line_s >> tree >> iface) {
			cout << ' ' << tree << ' ' << iface;
		}
		cout << endl;
	}
	return 0;
}

static int compute_freqs(bool binary_format) {
	string line;
	unsigned int freqs[filter_t::WIDTH] = { 0 };
	unsigned int count = 0;
	
	fib_entry f;

	while(binary_format ? f.read_binary(std::cin) : f.read_ascii(std::cin)) {
		for(unsigned int p = f.filter.next_bit(0); p < filter_t::WIDTH; p = f.filter.next_bit(p+1))
			freqs[p] += 1;

		count += 1;
	}

	for(unsigned int i = 0; i < filter_t::WIDTH; ++i)
		cout << "p " << i << ' ' << freqs[i] << ' ' << (100.0 * freqs[i] / count) << endl;

	return 0;
}

int main(int argc, const char * argv[]) {
	bool binary_format = false;
	unsigned int threshold = 0;
	bool print_progress_steps = true;
	const char * filters_fname = nullptr;
	const char * prefixes_fname = nullptr;

	for(int i = 1; i < argc; ++i) {
		if (strncmp(argv[i],"map=",4)==0) {
			return map_filters(argv[i] + 4);
		} else 
		if (strcmp(argv[i],"-b")==0) {
			binary_format = true;
		} else 
		if (strncmp(argv[i],"freq",4)==0) {
			return compute_freqs(binary_format);
		} else 
		if (strncmp(argv[i],"p=",2)==0) {
			prefixes_fname = argv[i] + 2;
		} else 
		if (strncmp(argv[i],"f=",2)==0) {
			filters_fname = argv[i] + 2;
		} else
		if (sscanf(argv[i],"t=%u", &threshold) > 0) {
			;
		} else
		if (strncmp(argv[i],"-Q",2)==0) {
			print_progress_steps = false;
			continue;
		} else {
			print_usage(argv[0]);
			return 1;
		}
	}

	if (filters_fname == nullptr || prefixes_fname == nullptr) {
		print_usage(argv[0]);
		return 1;
	}		

	int res;

	if (print_progress_steps)
		cout << "Reading prefixes..." << std::flush;
	if ((res = read_prefixes(prefixes_fname, binary_format)) < 0) {
		cerr << endl << "couldn't read prefix file: " << prefixes_fname << endl;
		return 1;
	};
	if (print_progress_steps)
		cout << "\t" << std::setw(12) << res << " prefixes." << endl;

	if (print_progress_steps)
		cout << "Reading filters..." << std::flush;
	if ((res = read_filters(filters_fname, binary_format)) < 0) {
		cerr << endl << "couldn't read filters file: " << filters_fname << endl;
		return 1;
	};
	if (print_progress_steps)
		cout << "\t" << std::setw(12) << res << " filters." << endl;
	
	analyze_fibs(threshold);

	return 0;
}
