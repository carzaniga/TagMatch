#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include <string>

#include "front_end.hh"
#include "back_end.hh"

using std::ifstream;
using std::string;
using std::istringstream;
using namespace std::chrono;

int read_prefixes(const char * fname) {
	ifstream is(fname);
	string line;
	if (!is)
		return 1;

	while(std::getline(is, line)) {
		istringstream line_s(line);
		string command;
		line_s >> command;
		if (command != "p")
			continue;

		unsigned int prefix_id, prefix_size;
		std::string prefix_string;

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

	while(std::getline(is, line)) {
		istringstream line_s(line);
		string command;
		line_s >> command;
		if (command != "f") 
			continue;

		unsigned int partition_id;
		interface_t iface;
		tree_t tree;
		std::string filter_string;

		line_s >> partition_id >> filter_string;

		filter_t f(filter_string);

		std::vector<tree_interface_pair> ti_pairs;

		while (line_s >> tree >> iface) 
			ti_pairs.push_back(tree_interface_pair(tree, iface));

		back_end::add_filter(partition_id, f, ti_pairs.begin(), ti_pairs.end());
	}
	return 0;
}

int read_queries(std::vector<packet> & packets, string fname) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return 1;

	while(std::getline(is, line)) {
		istringstream line_s(line);
		string command;
		line_s >> command;
		if (command != "!") 
			continue;

		tree_t tree;
		interface_t iface;
		std::string filter_string;

		line_s >> tree >> iface >> filter_string;

		packets.push_back(packet(filter_string, tree, iface));
	}
	return 0;
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
			std::cout << "usage: " << argv[0] 
					  << " p=prefix_file_name f=filters_file_name q=queries_file_name"
					  << std::endl;
			return 1;
		}
	}

	if (!read_prefixes(prefixes_fname)) {
		std::cerr << "couldn't read prefix file: " << prefixes_fname << std::endl;
		return 1;
	};
	
	if (!read_filters(filters_fname)) {
		std::cerr << "couldn't read filters file: " << filters_fname << std::endl;
		return 1;
	};
	
	std::vector<packet> packets;
	
	if (!read_queries(packets, queries_fname)) {
		std::cerr << "couldn't read queries file: " << queries_fname << std::endl;
		return 1;
	};

	back_end::start();
	front_end::start(THREAD_COUNT);

	high_resolution_clock::time_point start = high_resolution_clock::now();

	for(std::vector<packet>::iterator p = packets.begin(); p != packets.end(); ++p)
		front_end::match(&(*p));

	front_end::shutdown();

	high_resolution_clock::time_point stop = high_resolution_clock::now();

    nanoseconds ns = duration_cast<nanoseconds>(stop - start);
	std::cout << "Packets: " << packets.size() << std::endl
			  << "Average matching time: " << ns.count()/packets.size() << "ns" << std::endl;

	if (print_output) {
		for(std::vector<packet>::iterator p = packets.begin(); p != packets.end(); ++p) {
			if (p->is_matching_complete()) {
				for(unsigned i = 0; i < INTERFACES; ++i) 
					std::cout << ' ' << ((p->output[i]) ? '1' : '0');
				std::cout << std::endl;
			} else {
				std::cout << "incomplete" << std::endl;
            }
        }
    }
	
	return 0;
}
