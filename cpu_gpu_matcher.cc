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
			back_end::add_partition(p.partition, p.filter, p.length);
			++res;
//			cout<< (int)p.partition << " " << (int)p.length << endl;
		}
	} else {
		while(p.read_ascii(is)) {
//			std::cout<<(int) p.length << std::endl;
			front_end::add_prefix(p.partition, p.filter, p.length);
			back_end::add_partition(p.partition, p.filter, p.length);
			++res;
		}
	}
	is.close();
	return res;
}

#ifndef BACK_END_IS_VOID
static int read_filters(string fname, bool binary_format) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	partition_fib_entry f;
	if (binary_format) {
		while(f.read_binary(is)) {
			back_end::add_filter(f.partition, f.filter, f.ti_pairs.begin(), f.ti_pairs.end());
			++res;
		}
	} else {
		while(f.read_ascii(is)) {
			back_end::add_filter(f.partition, f.filter, f.ti_pairs.begin(), f.ti_pairs.end());
			++res;
		}
	}
	is.close();
	return res;
}
#endif
static unsigned int read_queries(vector<packet> & packets, string fname, bool binary_format) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	network_packet p;
	if (binary_format) {
		while(p.read_binary(is)) {
#ifndef TWITTER
			packets.emplace_back(p.filter, p.ti_pair.tree(), p.ti_pair.interface());
#else
			packets.emplace_back(p.filter, p.ti_pair.interface());
#endif
			++res;
		}
	} else {
		while(p.read_ascii(is)) {
#ifndef TWITTER
			packets.emplace_back(p.filter, p.ti_pair.tree(), p.ti_pair.interface());
#else
			packets.emplace_back(p.filter, p.ti_pair.interface());
#endif
			++res;
		}
	}
	is.close();
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
		"(p|P)=<prefix-file-name> (f|F)=<filters-file-name> (q|Q)=<queries-file-name>"
		 << endl
		 << "(lower case means ASCII input; upper case means binary input)"
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
	bool print_matching_time_only = false;
	const char * prefixes_fname = nullptr;
	bool prefixes_binary_format = false;
#ifndef BACK_END_IS_VOID
	const char * filters_fname = nullptr;
	bool filters_binary_format = false;
#endif
	const char * queries_fname = nullptr; 
	bool queries_binary_format = false;
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
		if (strncmp(argv[i],"P=",2)==0) {
			prefixes_binary_format = true;
			prefixes_fname = argv[i] + 2;
			continue;
		} else 
		if (strncmp(argv[i],"F=",2)==0) {
#ifndef BACK_END_IS_VOID
			filters_binary_format = true;
			filters_fname = argv[i] + 2;
#endif
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
	if ((res = read_prefixes(prefixes_fname, prefixes_binary_format)) < 0) {
		cerr << endl << "couldn't read prefix file: " << prefixes_fname << endl;
		return 1;
	};
	if (print_progress_steps)
		cout << "\t\t\t" << std::setw(12) << res << " prefixes." << endl;
	
#ifndef BACK_END_IS_VOID
	if (print_progress_steps)
		cout << "Reading filters..." << std::flush;
	if ((res = read_filters(filters_fname, filters_binary_format)) < 0) {
		cerr << endl << "couldn't read filters file: " << filters_fname << endl;
		return 1;
	};
	if (print_progress_steps)
		cout << "\t\t\t" << std::setw(12) << res << " filters." << endl;
#endif
	
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

		front_end::clear();
		back_end::clear();
		return 0;
	};


#ifndef BACK_END_IS_VOID
	if (print_progress_steps) 
		cout << "Back-end FIB compilation..." << std::flush;
#endif

	back_end::start();

	if (print_progress_steps) {
#ifndef BACK_END_IS_VOID
		cout << "\t\t" << std::setw(10) 
			 << back_end::bytesize()/(1024*1024) << "MB back-end FIB" << endl;
#endif
		cout << "Matching packets with " << thread_count << " threads..." << std::flush;
	}

	front_end::start(thread_count);

	high_resolution_clock::time_point start = high_resolution_clock::now();

	for(packet & p : packets)
		front_end::match(&p);

	front_end::stop();
	back_end::stop();

	high_resolution_clock::time_point stop = high_resolution_clock::now();

	if (print_progress_steps) {
		cout << "\t" << std::setw(10)
			 << duration_cast<nanoseconds>(stop - start).count()/packets.size() 
			 << "ns average matching time." << endl;
	} else if (print_matching_time_only) {
		cout << duration_cast<nanoseconds>(stop - start).count()/packets.size() << endl;
	}
	if (print_statistics) {
		cout << "Front-end Statistics:" << endl;
		front_end::print_statistics(cout);
	}

	back_end::clear();
	front_end::clear();

	if (print_matching_results) {
		for(unsigned int pid = 0; pid < packets.size(); ++pid) {
			bool this_pid_printed = false;
			if (packets[pid].is_matching_complete()) {
				for(unsigned i = 0; i < INTERFACES; ++i) { 
					if (packets[pid].get_output(i)) {
						if (!this_pid_printed) {
							cout << "packet=" << pid; 
							this_pid_printed = true;
						}
						cout << ' ' << i;
					}
				}
				if (this_pid_printed)
					cout << endl;
			} else {
				cout << "packet=" << pid << " incomplete" << endl;
            }
        }
    }
	return 0;
}
