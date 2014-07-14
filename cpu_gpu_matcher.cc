#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <string>

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

int read_prefixes(const char * fname) {
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


int read_filters(string fname) {
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

int read_queries(vector<packet> & packets, string fname) {
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

		packets.push_back(packet(filter_string, tree, iface));
	}
	return 0;
}

void print_usage(const char * progname) {
	cout << "usage: " << progname 
		 << " [-q] p=prefix_file_name f=filters_file_name q=queries_file_name"
		 << endl;
}
int main(int argc, const char * argv[]) {
	bool print_output = true;
	const char * prefixes_fname = 0;
	const char * filters_fname = 0;
	const char * queries_fname = 0; 

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
			print_output = false;
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

	if (read_prefixes(prefixes_fname) != 0) {
		cerr << "couldn't read prefix file: " << prefixes_fname << endl;
		return 1;
	};
	
	if (read_filters(filters_fname) != 0) {
		cerr << "couldn't read filters file: " << filters_fname << endl;
		return 1;
	};
	
	vector<packet> packets;
	
	if (read_queries(packets, queries_fname) != 0) {
		cerr << "couldn't read queries file: " << queries_fname << endl;
		return 1;
	};

	back_end::start();
	front_end::start(THREAD_COUNT);

	high_resolution_clock::time_point start = high_resolution_clock::now();

	for(vector<packet>::iterator p = packets.begin(); p != packets.end(); ++p)
		front_end::match(&(*p));

	front_end::stop();

	high_resolution_clock::time_point stop = high_resolution_clock::now();

	back_end::clear();
	front_end::clear();

    nanoseconds ns = duration_cast<nanoseconds>(stop - start);
	cout << "Packets: " << packets.size() << endl
			  << "Average matching time: " << ns.count()/packets.size() << "ns" << endl;

	if (print_output) {
		for(vector<packet>::iterator p = packets.begin(); p != packets.end(); ++p) {
			if (p->is_matching_complete()) {
				for(unsigned i = 0; i < INTERFACES; ++i) 
					cout << ' ' << ((p->output[i]) ? '1' : '0');
				cout << endl;
			} else {
				cout << "incomplete" << endl;
            }
        }
    }
	
	return 0;
}