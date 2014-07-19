#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
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
		return 1;

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
	}
	is.close();
	return 0;
}

static int read_filters(string fname) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return 1;

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
	}
	return 0;
}

static int read_queries(vector<packet> & packets, string fname) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return 1;

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
		packets.back().reset_output();
	}
	return 0;
}

static void print_usage(const char * progname) {
	cout << "usage: " << progname 
		 << " [options] " 
		"p=<prefix-file-name> f=<filters-file-name> q=<queries-file-name>"
		 << endl
		 << "options:" << endl
		 << "\t-q\t: disable output of matching results" << endl
		 << "\t-Q\t: disable output of progress steps" << endl
		 << "\tt=<N>\t: runs front-end with N threads (default=" << DEFAULT_THREAD_COUNT << ")" << endl
#ifdef WITH_FRONTEND_STATISTICS
		 << "\t-s\t: enable output of front-end statistics" << endl
#endif
		;
}

int main(int argc, const char * argv[]) {
#ifdef WITH_FRONTEND_STATISTICS
	bool print_statistics = false;
#endif
	bool print_matching_results = true;
	bool print_progress_steps = true;
	const char * prefixes_fname = 0;
	const char * filters_fname = 0;
	const char * queries_fname = 0; 
	unsigned int thread_count = DEFAULT_THREAD_COUNT;

	for(int i = 1; i < argc; ++i) {
		if (strncmp(argv[i],"p=",2)==0) {
			prefixes_fname = argv[i] + 2;
			continue;
		}
		if (strncmp(argv[i],"f=",2)==0) {
			filters_fname = argv[i] + 2;
			continue;
		}
		if (strncmp(argv[i],"q=",2)==0) {
			queries_fname = argv[i] + 2;
			continue;
		}
		if (strncmp(argv[i],"-q",2)==0) {
			print_matching_results = false;
			continue;
		}
		if (strncmp(argv[i],"-Q",2)==0) {
			print_progress_steps = false;
			continue;
		}
#ifdef WITH_FRONTEND_STATISTICS
		if (strncmp(argv[i],"-s",2)==0) {
			print_statistics = true;
			continue;
		}
#endif
		if (strncmp(argv[i],"t=",2)==0) {
			thread_count = atoi(argv[i] + 2);
			continue;
		}

		if (strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0) {
			print_usage(argv[0]);
			return 1;
		}
	}

	if (!prefixes_fname || !filters_fname || !queries_fname) {
		print_usage(argv[0]);
		return 1;
	}		

	if (print_progress_steps)
		cout << "Reading prefixes..." << std::flush;
	if (read_prefixes(prefixes_fname) != 0) {
		cerr << endl << "couldn't read prefix file: " << prefixes_fname << endl;
		return 1;
	};
	if (print_progress_steps)
		cout << "done." << endl;
	
	if (print_progress_steps)
		cout << "Reading filters..." << std::flush;
	if (read_filters(filters_fname) != 0) {
		cerr << endl << "couldn't read filters file: " << filters_fname << endl;
		return 1;
	};
	if (print_progress_steps)
		cout << "done." << endl;
	
	vector<packet> packets;
	
	if (print_progress_steps)
		cout << "Reading packets..." << std::flush;
	if (read_queries(packets, queries_fname) != 0) {
		cerr << endl << "couldn't read queries file: " << queries_fname << endl;
		return 1;
	};
	if (print_progress_steps) {
		cout << "done." << endl << "Packets: " << packets.size() << endl;

		cout << "Back-end FIB compilation..." << std::flush;
	}
	back_end::start();
	if (print_progress_steps) {
		cout << "done." << endl
			 << "Back-end memory in use: " << back_end::bytesize()/(1024*1024) << "MB" << endl;

		cout << "Front-end starts matching with " << thread_count << " threads..." << std::flush;
	}

	front_end::start(thread_count);

	high_resolution_clock::time_point start = high_resolution_clock::now();

	for(vector<packet>::iterator p = packets.begin(); p != packets.end(); ++p)
		front_end::match(&(*p));

	front_end::stop();
	back_end::stop();

	high_resolution_clock::time_point stop = high_resolution_clock::now();

	if (print_progress_steps)
		cout << "done." << endl;

#ifdef WITH_FRONTEND_STATISTICS
	if (print_statistics) {
		cout << "Front-end Statistics:" << endl;
		front_end::print_statistics(cout);
	}
#endif

	if (print_progress_steps)
		cout << "Clearing back-end." << endl;
	back_end::clear();

	if (print_progress_steps)
		cout << "Clearing front-end." << endl;
	front_end::clear();

    nanoseconds ns = duration_cast<nanoseconds>(stop - start);
	if (print_progress_steps)
		cout << "Average matching time: " << ns.count()/packets.size() << "ns" << endl;

	if (print_matching_results) {
		for(vector<packet>::const_iterator p = packets.begin(); p != packets.end(); ++p) {
			if (p->is_matching_complete()) {
				for(unsigned i = 0; i < INTERFACES; ++i) 
					cout << ' ' << ((p->get_output(i)) ? '1' : '0');
				cout << endl;
			} else {
				cout << "incomplete" << endl;
            }
        }
    }
	return 0;
}
