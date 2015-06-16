#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "packet.hh"
#include "fib.hh"

using std::vector;
using std::ifstream;
using std::string;
using std::istringstream;
using std::getline;
using std::cout;
using std::cerr;
using std::endl;

static vector<partition_fib_entry> fib;

#if 0
static int read_prefixes(const char * fname, bool binary_format) {
	ifstream is(fname);
	string line;
	if (!is)
		return -1;

	partition_prefix p;
	int res = 0;
	if (binary_format) {
		while(p.read_binary(is)) {
			front_end::add_prefix(p.partition, p.filter, p.length);
			++res;
		}
	} else {
		while(p.read_ascii(is)) {
			front_end::add_prefix(p.partition, p.filter, p.length);
			++res;
		}
	}
	is.close();
	return res;
}
#endif

static int read_filters(string fname, bool binary_format) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	partition_fib_entry f;
	if (binary_format) {
		while(f.read_binary(is)) {
			fib.emplace_back(f);
			++res;
		}
	} else {
		while(f.read_ascii(is)) {
			fib.emplace_back(f);
			++res;
		}
	}
	is.close();
	return res;
}

static void match(const network_packet & p) {
	bool results[INTERFACES] = { false };

    const tree_t tree = p.ti_pair.tree();
	const interface_t iface = p.ti_pair.interface();

	for(const fib_entry & entry : fib) 
		if (entry.filter.subset_of(p.filter)){ 
			entry.write_ascii(std::cout);
			std::cout<<std::endl;
			for(const tree_interface_pair & tip : entry.ti_pairs)
				if (tree == tip.tree() && iface != tip.interface())
					results[tip.interface()] = true;
		}
	for(unsigned i = 0; i < INTERFACES; ++i) 
		if (results[i])
			cout << ' ' << i;
	cout << endl;
}

static unsigned int read_and_match_queries(string fname, bool binary_format) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	network_packet p;
	if (binary_format) {
		while(p.read_binary(is)) {
			match(p);
			++res;
		}
	} else {
		while(p.read_ascii(is)) {
			match(p);
			++res;
		}
	}
	is.close();
	return res;
}

void print_usage(const char * progname) {
	cout << "usage: " << progname 
		 << " f=filters_file_name q=queries_file_name"
		 << endl;
}

int main(int argc, const char * argv[]) {
#if 0
	const char * prefixes_fname = nullptr;
	bool prefixes_binary_format = false;
#endif
	const char * filters_fname = nullptr;
	bool filters_binary_format = false;
	const char * queries_fname = nullptr; 
	bool queries_binary_format = false;

	for(int i = 1; i < argc; ++i) {
#if 0
		if (strncmp(argv[i],"P=",2)==0) {
			prefixes_binary_format = true;
			prefixes_fname = argv[i] + 2;
			continue;
		} else 
		if (strncmp(argv[i],"p=",2)==0) {
			prefixes_fname = argv[i] + 2;
			continue;
		} else 
#endif
		if (strncmp(argv[i],"f=",2)==0) {
			filters_fname = argv[i] + 2;
			continue;
		} else
		if (strncmp(argv[i],"F=",2)==0) {
			filters_binary_format = true;
			filters_fname = argv[i] + 2;
			continue;
		} else
		if (strncmp(argv[i],"q=",2)==0) {
			queries_fname = argv[i] + 2;
			continue;
		} else
		if (strncmp(argv[i],"Q=",2)==0) {
			queries_binary_format = true;
			queries_fname = argv[i] + 2;
			continue;
		} else
		if (strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0) {
			print_usage(argv[0]);
			return 1;
		}
	}

	if (!filters_fname || !queries_fname) {
		print_usage(argv[0]);
		return 1;
	}		

	if (read_filters(filters_fname, filters_binary_format) < 0) {
		cerr << endl << "couldn't read filters file: " << filters_fname << endl;
		return 1;
	};
	
	if (read_and_match_queries(queries_fname, queries_binary_format) < 0) {
		cerr << endl << "couldn't read queries file: " << queries_fname << endl;
		return 1;
	};

	fib.clear();
	return 0;
}
