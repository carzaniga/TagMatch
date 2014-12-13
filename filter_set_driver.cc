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

#include "filter_set.hh"
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

static int read_filters(string fname, bool binary_format) {
	ifstream input (fname) ;
	string line;

	if (!input)
		return -1;

	filter_set::clear();
	int res = 0;
	fib_entry f;
	if (binary_format) {
		while(f.read_binary(input)) {
			filter_set::add(f.filter);
			++res;
		}
	} else {
		while(f.read_ascii(input)) {
			filter_set::add(f.filter);
			++res;
		}
	}
	input.close();
	return res;
}

static int read_queries(vector<packet> & packets, string fname, bool binary_format) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	network_packet p;
	if (binary_format) {
		while(p.read_binary(is)) {
			packets.emplace_back(p.filter, p.ti_pair.tree(), p.ti_pair.interface());
			++res;
		}
	} else {
		while(p.read_ascii(is)) {
			packets.emplace_back(p.filter, p.ti_pair.tree(), p.ti_pair.interface());
			++res;
		}
	}
	is.close();
	return res;
}

static void print_usage(const char * progname) {
	cout << "usage: " << progname 
		 << " [options] " 
		"(f|F)=<filters-file-name> (q|Q)=<queries-file-name>"
		 << endl
		 << "(lower case means ASCII input; upper case means binary input)"
		 << endl
		 << "options:" << endl
		 << "\t-q\t: disable output of matching results" << endl
		 << "\t-Q\t: disable output of progress steps" << endl;
}

int main(int argc, const char * argv[]) {
	bool print_progress_steps = true;
	bool print_matching_time_only = false;
#ifndef BACK_END_IS_VOID
	const char * filters_fname = nullptr;
	bool filters_binary_format = false;
#endif
	const char * queries_fname = nullptr; 
	bool queries_binary_format = false;

	for(int i = 1; i < argc; ++i) {
		if (strncmp(argv[i],"f=",2)==0) {
			filters_binary_format = false;
			filters_fname = argv[i] + 2;
			continue;
		} 
		if (strncmp(argv[i],"q=",2)==0) {
			queries_binary_format = false;
			queries_fname = argv[i] + 2;
			continue;
		} 
		if (strncmp(argv[i],"F=",2)==0) {
			filters_binary_format = true;
			filters_fname = argv[i] + 2;
			continue;
		} 
		if (strncmp(argv[i],"Q=",2)==0) {
			queries_binary_format = true;
			queries_fname = argv[i] + 2;
			continue;
		} 
		if (strncmp(argv[i],"-Q",2)==0) {
			print_progress_steps = false;
			if (strncmp(argv[i],"-Qt",3)==0)
				print_matching_time_only = true;
			continue;
		} 
		print_usage(argv[0]);
		return 1;
	}

	if (!filters_fname || !queries_fname) {
		print_usage(argv[0]);
		return 1;
	}		

	int res;

	filter_set::clear();

	if (print_progress_steps)
		cout << "Reading filters..." << std::flush;
	if ((res = read_filters(filters_fname, filters_binary_format)) < 0) {
		cerr << endl << "couldn't read filters file: " << filters_fname << endl;
		return 1;
	};
	if (print_progress_steps)
		cout << "\t\t\t" << std::setw(12) << res << " filters." << endl;
	
	vector<packet> packets;
	
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

		filter_set::clear();
		return 0;
	};

	filter_set::consolidate();

	if (print_progress_steps) {
		cout << "Matching packets... " << std::flush;
	}

	high_resolution_clock::time_point start = high_resolution_clock::now();

	unsigned int matches = 0;
	for(packet & p : packets) 
		matches += filter_set::count_subsets_of(p.filter);

	high_resolution_clock::time_point stop = high_resolution_clock::now();

	if (print_progress_steps) {
		cout << "\t" << std::setw(10)
			 << duration_cast<nanoseconds>(stop - start).count()/packets.size() 
			 << "ns average matching time." << endl;
	} else if (print_matching_time_only) {
		cout << duration_cast<nanoseconds>(stop - start).count()/packets.size() << endl;
	}

	cout << matches << endl;
	return 0;
}
