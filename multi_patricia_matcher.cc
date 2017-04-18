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

#include "filter.hh"
#include "multi_patricia_predicate.hh"
#include "key_array.hh"
#include "fib.hh"
#include "packet.hh"

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

#ifndef INTERFACES
#define INTERFACES 256U
#endif

typedef multi_patricia_predicate<key_array> predicate;
predicate P;

static int read_filters(string fname, bool binary_format) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	fib_entry f;

	while((binary_format) ? f.read_binary(is) : f.read_ascii(is)) {
		key_array & keys = P.add(f.filter);
		for(const tagmatch_key_t * p = f.keys.data();
			p != f.keys.data() + f.keys.size(); ++p)
			keys.add(*p);
		++res;
	}

	is.close();
	return res;
}

static unsigned int read_queries(vector<network_packet> & packets, string fname, bool binary_format) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	network_packet p;
	if (binary_format) {
		while(p.read_binary(is)) {
			packets.emplace_back(p.filter);
			++res;
		}
	} else {
		while(p.read_ascii(is)) {
			packets.emplace_back(p.filter);
			++res;
		}
	}
	is.close();
	return res;
}


static bool use_identity_permutation = true;
static unsigned char bit_permutation[filter_t::WIDTH] = { 0 };

static void set_identity_permutation() noexcept {
	use_identity_permutation = true;
}

static void set_bit_permutation_pos(unsigned char old_pos, unsigned char new_pos) {
	use_identity_permutation = false;
	bit_permutation[old_pos] = new_pos;
}

static const filter_t & apply_permutation(const filter_t & f) {
	static filter_t f_tmp;
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
	return f_tmp;
}

static int read_bit_permutation(const char * fname) {
	ifstream is(fname);
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

		set_bit_permutation_pos(old_bit_pos, new_bit_pos);
		++new_bit_pos;
	}
	is.close();
	for(;new_bit_pos < filter_t::WIDTH; ++new_bit_pos)
		set_bit_permutation_pos(new_bit_pos, new_bit_pos);

	return new_bit_pos;
}

static void print_usage(const char * progname) {
	cout << "usage: " << progname 
		 << " [options] " 
		"(f|F)=<filters-file-name> (q|Q)=<queries-file-name>"
		 << endl
		 << "(lower case means ASCII input; upper case means binary input)"
		 << endl
		 << "options:" << endl
		 << "\tmap=<permutation-file-name>" << endl
		 << "\t-q\t: disable output of matching results" << endl
		 << "\t-Q\t: disable output of progress steps" << endl;
}

class match_vector : public predicate::match_handler {
public:
	match_vector(): output{0} {};

	virtual bool match(key_array & keys) {
		for (const tagmatch_key_t * k = keys.begin(); k != keys.end(); ++k)
			output.push_back(*k);
		return false;
	}

	void print_results() {
		std::sort(output.begin(), output.end());
		output.erase(std::unique(output.begin(), output.end()), output.end());
		for(tagmatch_key_t k : output)
			cout << ' ' << k;
		cout << endl;
    }

private:
	vector<tagmatch_key_t> output;
};

int main(int argc, const char * argv[]) {
	bool print_matching_results = true;
	bool print_progress_steps = true;
	bool print_matching_time_only = false;
	const char * filters_fname = nullptr;
	bool filters_binary_format = false;
	const char * queries_fname = nullptr; 
	bool queries_binary_format = false;
	const char * permutation_fname = nullptr; 

	for(int i = 1; i < argc; ++i) {
		if (strncmp(argv[i],"f=",2)==0) {
			filters_fname = argv[i] + 2;
			continue;
		} else
		if (strncmp(argv[i],"q=",2)==0) {
			queries_fname = argv[i] + 2;
			continue;
		} else
		if (strncmp(argv[i],"F=",2)==0) {
			filters_binary_format = true;
			filters_fname = argv[i] + 2;
			continue;
		} else
		if (strncmp(argv[i],"Q=",2)==0) {
			queries_binary_format = true;
			queries_fname = argv[i] + 2;
			continue;
		} else
		if (strncmp(argv[i],"map=",4)==0) {
			permutation_fname = argv[i] + 4;
			continue;
		} else
		if (strncmp(argv[i],"-q",2)==0) {
			print_matching_results = false;
			continue;
		} else
		if (strncmp(argv[i],"-Q",2)==0) {
			print_progress_steps = false;
			if (strncmp(argv[i],"-Qt",3)==0)
				print_matching_time_only = true;
			continue;
		} else {
			print_usage(argv[0]);
			return 1;
		}
	}

	if (!filters_fname || !queries_fname) {
		print_usage(argv[0]);
		return 1;
	}		

	int res;

	if (permutation_fname) {
		if (print_progress_steps)
			cout << "Reading bit permutation..." << std::flush;
		if ((res = read_bit_permutation(permutation_fname)) < 0) {
			cerr << endl << "couldn't read permutation file: " << permutation_fname << endl;
			return 1;
		};
		if (print_progress_steps)
			cout << "\t\t" << std::setw(12) << res << " bits." << endl;
	} else {
		set_identity_permutation();
	}
	
	if (print_progress_steps)
		cout << "Reading filters..." << std::flush;
	if ((res = read_filters(filters_fname, filters_binary_format)) < 0) {
		cerr << endl << "couldn't read filters file: " << filters_fname << endl;
		return 1;
	};
	if (print_progress_steps)
		cout << "\t\t\t" << std::setw(12) << res << " filters." << endl;
	
	vector<network_packet> packets;
	
	if (print_progress_steps)
		cout << "Reading packets..." << std::flush;
	if ((res = read_queries(packets, queries_fname, queries_binary_format)) < 0) {
		cerr << endl << "couldn't read queries file: " << queries_fname << endl;
		return 1;
	};
	if (print_progress_steps) 
		cout << "\t\t\t" << std::setw(12) << res << " packets." << endl;
	if (res == 0) {
		cerr << "No packets to process.  Bailing out." << endl;

		P.clear();
		return 0;
	};

	vector<match_vector> match_results(packets.size());

	if (print_progress_steps) {
		cout << "Matching packets... " << std::flush;
	}

	high_resolution_clock::time_point start = high_resolution_clock::now();

	if (use_identity_permutation) {
		unsigned int i = 0;
		for(network_packet & p : packets) 
			P.find_all_subsets(p.filter, match_results[i++]);
	} else {
		unsigned int i = 0;
		for(network_packet & p : packets)
			P.find_all_subsets(apply_permutation(p.filter), match_results[i++]);
	}

	high_resolution_clock::time_point stop = high_resolution_clock::now();

	if (print_progress_steps) {
		cout << "\t\t\t" << std::setw(10)
			 << duration_cast<nanoseconds>(stop - start).count()/packets.size() 
			 << "ns average matching time." << endl;
	} else if (print_matching_time_only) {
		cout << duration_cast<nanoseconds>(stop - start).count()/packets.size() << endl;
	}

	P.clear();

	if (print_matching_results) 
		for(match_vector & m : match_results)
			m.print_results();

	return 0;
}
