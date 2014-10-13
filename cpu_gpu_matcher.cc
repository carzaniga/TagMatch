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

#include "front_end.hh"
#include "back_end.hh"

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

static int read_prefixes(const char * fname) {
	ifstream is(fname);
	string line;
	if (!is)
		return -1;

	int res = 0;
	while(getline(is, line)) {
		istringstream line_s(line);
		string command;
		line_s >> command;
		if (command != "p")
			continue;

		unsigned int prefix_id, prefix_size;
		string prefix_string;

		line_s >> prefix_id >> prefix_string >> prefix_size;

		filter_t f(prefix_string);

		front_end::add_prefix(prefix_id, f, prefix_string.size());
		++res;
	}
	is.close();
	return res;
}

#ifndef BACK_END_IS_VOID
static int read_filters(string fname) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	while(getline(is, line)) {
		istringstream line_s(line);
		string command;
		line_s >> command;
		if (command != "f") 
			continue;

		unsigned int partition_id;
		interface_t iface;
		tree_t tree;
		string filter_string;

		line_s >> partition_id >> filter_string;

		filter_t f(filter_string);

		vector<tree_interface_pair> ti_pairs;

		while (line_s >> tree >> iface)
			ti_pairs.push_back(tree_interface_pair(tree, iface));

		back_end::add_filter(partition_id, f, ti_pairs.begin(), ti_pairs.end());
		++res;
	}
	return res;
}
#endif
static unsigned int read_queries(vector<packet> & packets, string fname) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	while(getline(is, line)) {
		istringstream line_s(line);
		string command;
		line_s >> command;
		if (command != "!") 
			continue;

		tree_t tree;
		interface_t iface;
		string filter_string;

		line_s >> tree >> iface >> filter_string;

		packets.emplace_back(filter_string, tree, iface);
		++res;
	}
	for(packet & p : packets)
		p.reset_output();

	return res;
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

		front_end::set_bit_permutation_pos(old_bit_pos, new_bit_pos);
		++new_bit_pos;
	}
	is.close();
	for(;new_bit_pos < filter_t::WIDTH; ++new_bit_pos)
		front_end::set_bit_permutation_pos(new_bit_pos, new_bit_pos);

	return new_bit_pos;
}

static void print_usage(const char * progname) {
	cout << "usage: " << progname 
		 << " [options] " 
		"p=<prefix-file-name> f=<filters-file-name> q=<queries-file-name>"
		 << endl
		 << "options:" << endl
		 << "\tmap=<permutation-file-name>" << endl
		 << "\t-q\t: disable output of matching results" << endl
		 << "\t-Q\t: disable output of progress steps" << endl
		 << "\t-s\t: enable output of front-end statistics" << endl
		 << "\tt=<N>\t: runs front-end with N threads (default=" << DEFAULT_THREAD_COUNT << ")" << endl
		 << "\tl=<L>\t: runs front-end with a latency limit of L milliseconds (default=no limit)" << endl;
}

int main(int argc, const char * argv[]) {
	bool print_statistics = false;
	bool print_matching_results = true;
	bool print_progress_steps = true;
	const char * prefixes_fname = nullptr;
#ifndef BACK_END_IS_VOID
	const char * filters_fname = nullptr;
#endif
	const char * queries_fname = nullptr; 
	const char * permutation_fname = nullptr; 
	unsigned int thread_count = DEFAULT_THREAD_COUNT;

	for(int i = 1; i < argc; ++i) {
		if (strncmp(argv[i],"p=",2)==0) {
			prefixes_fname = argv[i] + 2;
			continue;
		} else 
		if (strncmp(argv[i],"f=",2)==0) {
#ifndef BACK_END_IS_VOID
			filters_fname = argv[i] + 2;
#endif
			continue;
		} else
		if (strncmp(argv[i],"q=",2)==0) {
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
			continue;
		} else
		if (strncmp(argv[i],"-s",2)==0) {
			print_statistics = true;
			continue;
		} else
		if (strncmp(argv[i],"t=",2)==0) {
			thread_count = atoi(argv[i] + 2);
			continue;
		} else
		if (strncmp(argv[i],"l=",2)==0) {
			unsigned int latency_limit = atoi(argv[i] + 2);
			front_end::set_latency_limit_ms(latency_limit);
			continue;
		} else {
			print_usage(argv[0]);
			return 1;
		}
	}

	if (!prefixes_fname 
#ifndef BACK_END_IS_VOID
		|| !filters_fname 
#endif
		|| !queries_fname) {
		print_usage(argv[0]);
		return 1;
	}		

	int res;

	if (permutation_fname) {
		if (print_progress_steps)
			cout << "Reading bit permutation..." << std::flush;
		if ((res = read_bit_permutation(permutation_fname)) < 0) {
			cerr << endl << "couldn't read prefix file: " << permutation_fname << endl;
			return 1;
		};
		if (print_progress_steps)
			cout << "\t\t" << std::setw(12) << res << " bits." << endl;
	} else {
		front_end::set_identity_permutation();
	}
	
	if (print_progress_steps)
		cout << "Reading prefixes..." << std::flush;
	if ((res = read_prefixes(prefixes_fname)) < 0) {
		cerr << endl << "couldn't read prefix file: " << prefixes_fname << endl;
		return 1;
	};
	if (print_progress_steps)
		cout << "\t\t\t" << std::setw(12) << res << " prefixes." << endl;
	
#ifndef BACK_END_IS_VOID
	if (print_progress_steps)
		cout << "Reading filters..." << std::flush;
	if ((res = read_filters(filters_fname)) < 0) {
		cerr << endl << "couldn't read filters file: " << filters_fname << endl;
		return 1;
	};
	if (print_progress_steps)
		cout << "\t\t\t" << std::setw(12) << res << " filters." << endl;
#endif
	
	vector<packet> packets;
	
	if (print_progress_steps)
		cout << "Reading packets..." << std::flush;
	if ((res = read_queries(packets, queries_fname)) < 0) {
		cerr << endl << "couldn't read queries file: " << queries_fname << endl;
		return 1;
	};
	if (print_progress_steps) 
		cout << "\t\t\t" << std::setw(12) << res << " packets." << endl;
	if (res == 0) {
		cerr << "No packets to process.  Bailing out." << endl;

		front_end::clear();
		back_end::clear();
		return 0;
	};


	if (print_progress_steps) 
		cout << "Back-end FIB compilation..." << std::flush;

	back_end::start();

	if (print_progress_steps) {
		cout << "\t\t" << std::setw(10) 
			 << back_end::bytesize()/(1024*1024) << "MB back-end FIB" << endl;

		cout << "Matching packets with " << thread_count << " threads..." << std::flush;
	}

	front_end::start(thread_count);

	high_resolution_clock::time_point start = high_resolution_clock::now();

	for(packet & p : packets)
		front_end::match(&p);

	front_end::stop();
	back_end::stop();

	high_resolution_clock::time_point stop = high_resolution_clock::now();

	if (print_progress_steps) 
		cout << "\t" << std::setw(10)
			 << duration_cast<nanoseconds>(stop - start).count()/packets.size() 
			 << "ns average matching time." << endl;

	if (print_statistics) {
		cout << "Front-end Statistics:" << endl;
		front_end::print_statistics(cout);
	}

	back_end::clear();
	front_end::clear();

	if (print_matching_results) {
		for(const packet & p : packets) {
			if (p.is_matching_complete()) {
				for(unsigned i = 0; i < INTERFACES; ++i) 
					if (p.get_output(i))
						cout << ' ' << i;
				cout << endl;
			} else {
				cout << "incomplete" << endl;
            }
        }
    }
	return 0;
}
