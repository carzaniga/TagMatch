#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "packet.hh"

using std::vector;
using std::ifstream;
using std::string;
using std::istringstream;
using std::getline;
using std::cout;
using std::cerr;
using std::endl;

class f_descr {
public:
	filter_t f;
	vector<tree_interface_pair> ti_pairs;
};

static vector<f_descr> fib;

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

		fib.emplace_back();
		fib.back().f = f;

		while (line_s >> tree >> iface) 
			fib.back().ti_pairs.push_back(tree_interface_pair(tree, iface));
	}
	return 0;
}

static bool covers(const filter_t & f1, const filter_t & f2) {
	const block_t * b1 = f1.begin();
	const block_t * b2 = f2.begin();

	do {
		if ((~(*b1) & *b2) != 0)
			return false;
		++b1;
		++b2;
	} while(b1 != f1.end());
	return true;
}

static void match(const string & f_string, tree_t tree, interface_t iface) {
	bool results[INTERFACES] = { false };

	filter_t f(f_string);

	for(vector<f_descr>::const_iterator di = fib.begin(); di != fib.end(); ++di) {
		if (covers(f, di->f)) {
			for(vector<tree_interface_pair>::const_iterator tip = di->ti_pairs.begin();
				tip != di->ti_pairs.end(); ++tip) {
				if (tree == tip->tree() && iface != tip->interface())
					results[tip->interface()] = true;
			}
		}
	}

	for(unsigned i = 0; i < INTERFACES; ++i) 
		if (results[i])
		cout << ' ' << i;
	cout << endl;
}

int read_and_match_queries(string fname) {
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

		match(filter_string, tree, iface);
	}
	return 0;
}

void print_usage(const char * progname) {
	cout << "usage: " << progname 
		 << " f=filters_file_name q=queries_file_name"
		 << endl;
}
int main(int argc, const char * argv[]) {
	const char * filters_fname = 0;
	const char * queries_fname = 0; 

	for(int i = 1; i < argc; ++i) {
		if (strncmp(argv[i],"f=",2)==0) {
			filters_fname = argv[i] + 2;
			continue;
		}
		if (strncmp(argv[i],"q=",2)==0) {
			queries_fname = argv[i] + 2;
			continue;
		}
		if (strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0) {
			print_usage(argv[0]);
			return 1;
		}
	}

	if (!filters_fname || !queries_fname) {
		print_usage(argv[0]);
		return 1;
	}		

	if (read_filters(filters_fname) != 0) {
		cerr << endl << "couldn't read filters file: " << filters_fname << endl;
		return 1;
	};
	
	if (read_and_match_queries(queries_fname) != 0) {
		cerr << endl << "couldn't read queries file: " << queries_fname << endl;
		return 1;
	};

	fib.clear();
	return 0;
}
