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

#include "tagmatch.hh"
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
			tagmatch::add_partition(p.partition, p.filter);
			++res;
		}
	} else {
		while(p.read_ascii(is)) {
			tagmatch::add_partition(p.partition, p.filter);
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
			tagmatch::add_filter(f.partition, f.filter, f.keys.begin(), f.keys.end());
			++res;
		}
	} else {
		while(f.read_ascii(is)) {
			tagmatch::add_filter(f.partition, f.filter, f.keys.begin(), f.keys.end());
			++res;
		}
	}
	is.close();
	return res;
}
#endif

static unsigned int read_queries(vector<synchronous_match_handler> & packets, string fname, bool binary_format) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	network_packet np;
	if (binary_format) {
		while(np.read_binary(is)) {
			packet * p = new packet(np.filter, np.key);
			packets.emplace_back(p);
			++res;
		}
	} else {
		while(np.read_ascii(is)) {
			packet * p = new packet(np.filter, np.key);
			packets.emplace_back(p);
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

		//front_end::set_bit_permutation_pos(old_bit_pos, new_bit_pos);
		++new_bit_pos;
	}
	is.close();
	//for(;new_bit_pos < filter_t::WIDTH; ++new_bit_pos)
		//front_end::set_bit_permutation_pos(new_bit_pos, new_bit_pos);

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
		 << "\tg=<N>\t: runs back-end with N gpus (default=" << DEFAULT_GPU_COUNT << ")" << endl
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
	unsigned int gpu_count = DEFAULT_GPU_COUNT;

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
		if (strncmp(argv[i],"g=",2)==0) {
			gpu_count = atoi(argv[i] + 2);
			continue;
		} else
		if (strncmp(argv[i],"l=",2)==0) {
			unsigned int latency_limit = atoi(argv[i] + 2);
			tagmatch::set_latency_limit_ms(latency_limit);
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
		//front_end::set_identity_permutation();
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
	
	vector<synchronous_match_handler> mhandlers;
	
	if (print_progress_steps)
		cout << "Reading packets..." << std::flush;
	if ((res = read_queries(mhandlers, queries_fname, queries_binary_format)) < 0) {
		cerr << endl << "couldn't read queries file: " << queries_fname << endl;
		return 1;
	};
	if (print_progress_steps) 
		cout << "\t\t\t" << std::setw(12) << res << " packets." << endl;
	if (res == 0) {
		cerr << "No packets to process.  Bailing out." << endl;

		tagmatch::clear();
		return 0;
	};
	if (print_progress_steps)
		cout << "Clearing packet output..." << std::flush;
	for(packet & p : packets)
		p.reset();
	if (print_progress_steps)
		cout << "\t\t\t" << std::setw(12) << res << " packets." << endl;

#ifndef BACK_END_IS_VOID
	if (print_progress_steps) 
		cout << "Back-end FIB compilation..." << std::flush;
#endif

	tagmatch::start(thread_count, gpu_count);

	if (print_progress_steps) {
#ifndef BACK_END_IS_VOID
		cout << "\t\t" << std::setw(10) 
			 << back_end::bytesize()/(1024*1024) << "MB back-end FIB" << endl;
#endif
		cout << "Matching packets with " << thread_count << " threads and " << gpu_count << " gpus..." << std::flush;
	}

	high_resolution_clock::time_point start = high_resolution_clock::now();

	int cidx = 0;
	for(match_handler & m : mhandlers) {
		cout << "Starting a new match! " << cidx++ << endl;
		tagmatch::match(&m);
	}

	tagmatch::stop();

	high_resolution_clock::time_point stop = high_resolution_clock::now();

	if (print_progress_steps) {
		cout << "\t" << std::setw(10)
			 << duration_cast<nanoseconds>(stop - start).count()/mhandlers.size() 
			 << "ns average matching time." << endl;
	} else if (print_matching_time_only) {
		cout << duration_cast<nanoseconds>(stop - start).count()/mhandlers.size() << endl;
	}
	if (print_statistics) {
		cout << "Front-end Statistics: TODO!" << endl;
		//front_end::print_statistics(cout);
	}

	tagmatch::clear();
	

	if (print_matching_results) {
		for(unsigned int pid = 0; pid < mhandlers.size(); ++pid) {
			bool this_pid_printed = false;
			if (mhandlers[pid].p->is_matching_complete()) {
				std::vector<uint32_t> users = mhandlers[pid].p->get_output_users();
					if (users.size()) {
						cout << "packet=" << pid; 
						this_pid_printed = true;
						for (uint32_t user: users) {
							std::cout << " " << user;
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
