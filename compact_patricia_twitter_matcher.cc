#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <mutex>
#include <limits>

#include "filter.hh"
#include "compact_patricia_predicate.hh"
#include "twitter_fib.hh"

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

#define EXPERIMENTAL_PROGRESS_BAR 1

#if EXPERIMENTAL_PROGRESS_BAR
template <unsigned int W>
class progress_bar {
public:
	static const unsigned int WIDTH = W;

	progress_bar(unsigned int n, std::ostream & os = std::cout)
		: total(n), count(0), output(os) {
		mask = 0;
		while (mask < n/WIDTH)
			mask = (mask << 1) | 1U;
	}

	void tick() {
		if ((count & mask) == 0)
			print_progress_bar();
		++count;
	}

	void clear() {
		output << "\e[s";
		for (unsigned int i = 0; i < WIDTH + 2; ++i)
			output << ' ';
		output << "\e[u";
		output.flush();
		count = 0;
	}

private:
	void print_progress_bar() const {
		output << "\e[s";
		output << '|';
		unsigned int limit = (total > std::numeric_limits<unsigned int>::max()/WIDTH)
			? count / (total / WIDTH) : (count * WIDTH) / total;
		for (unsigned int i = 0; i < WIDTH; ++i) {
			if (i < limit) output << '=';
			else if (i == limit) output << '>';
			else output << '-';
		}
		output << '|';
		output << "\e[u";
		output.flush();
	}

	const unsigned int total;
	unsigned int count;
	unsigned int mask;
	std::ostream & output;
};
#endif

typedef compact_patricia_predicate<twitter_id_vector> predicate;

predicate P;

static int read_filters(string fname, bool binary_format, unsigned int pre_sorted,
						bool print_progress) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	unsigned int res = 0;
	twitter_fib_entry e;

	if (pre_sorted) {
		P.use_pre_sorted_filters(pre_sorted);

#if EXPERIMENTAL_PROGRESS_BAR
		progress_bar<50> pb(pre_sorted);
#endif
		while((binary_format) ? e.read_binary(is) : e.read_ascii(is)) {
			twitter_id_vector & tids = P.add(e.filter);
			tids = e.ids;
			++res;
#if EXPERIMENTAL_PROGRESS_BAR
			if (print_progress)
				pb.tick();
#endif
			if (res == pre_sorted)
				break;
		}
#if EXPERIMENTAL_PROGRESS_BAR
		if (print_progress)
			pb.clear();
#endif
	} else {
		while((binary_format) ? e.read_binary(is) : e.read_ascii(is)) {
			twitter_id_vector & tids = P.add(e.filter);
			tids = e.ids;
			++res;
		}
	}
	is.close();
	return res;
}

vector<twitter_packet> packets;

static unsigned int read_queries(string fname, bool binary_format) {
	ifstream is (fname) ;
	string line;

	if (!is)
		return -1;

	int res = 0;
	twitter_packet p;
	if (binary_format) {
		while(p.read_binary(is)) {
			packets.emplace_back(p);
			++res;
		}
	} else {
		while(p.read_ascii(is)) {
			packets.emplace_back(p);
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
		 << "\tT=<number-of-threads>" << endl
		 << "\tSF=<number-of-sorted-filters>" << endl
		 << "\tmap=<permutation-file-name>" << endl
		 << "\t-q\t: disable output of matching results" << endl
		 << "\t--no-merge\t: disable merge phase" << endl
		 << "\t-Q\t: disable output of progress steps" << endl
		 << "\t-Qt\t: disable output of progress steps, print timing results" << endl
		 << "\t-Qm\t: disable output of progress steps, print per-query totals" << endl;
}

static std::mutex output_mtx;

class match_vector : public predicate::match_handler {
public:
	match_vector(): output() {};

	virtual bool match(twitter_id_vector & tids) {
		std::copy(tids.begin(), tids.end(), std::back_inserter(output));
		return false;
	}

	void print_results() {
		std::sort(output.begin(), output.end());
		std::vector<twitter_id_t>::const_iterator begin = output.begin();
		std::vector<twitter_id_t>::const_iterator end = std::unique(output.begin(), output.end());
		while (begin != end)
			cout << *begin++ << ' ';
		cout << endl;
    }

	unsigned int merge_results() {
		std::sort(output.begin(), output.end());
		return std::unique(output.begin(), output.end()) - output.begin();
    }

	void clear() {
		output.clear();
	}

	void clear_and_shrink() {
		output.clear();
		output.shrink_to_fit();
	}

private:
	std::vector<twitter_id_t> output;
};

class null_handler : public predicate::match_handler {
public:
	virtual bool match(twitter_id_vector &) {
		return false;
	}
};

static std::atomic_uint packet_idx;
static std::atomic_bool matchers_hold;

void match_and_merge() {
	match_vector handler;
	unsigned int i;

	while (matchers_hold.load())
		;

	while ((i = packet_idx.fetch_add(1)) < packets.size()) {
		P.find_all_subsets(packets[i].filter, handler);
		handler.merge_results();
		handler.clear();
	}
}

void match_and_merge_output() {
	match_vector handler;
	unsigned int i;

	while (matchers_hold.load())
		;

	while ((i = packet_idx.fetch_add(1)) < packets.size()) {
		P.find_all_subsets(packets[i].filter, handler);
		unsigned int res = handler.merge_results();
		output_mtx.lock();
		std::cout << res << std::endl;
		output_mtx.unlock();
		handler.clear();
	}
}

void match_only() {
	null_handler handler;
	unsigned int i;

	while (matchers_hold.load())
		;

	while ((i = packet_idx.fetch_add(1)) < packets.size())
		P.find_all_subsets(packets[i].filter, handler);
}

int main(int argc, const char * argv[]) {
	bool print_matching_results = true;
	bool print_progress_steps = true;
	bool print_matching_time_only = false;
	bool print_per_query_totals_only = false;
	bool merge_results = true;
	const char * filters_fname = nullptr;
	bool filters_binary_format = false;
	const char * queries_fname = nullptr; 
	bool queries_binary_format = false;
	const char * permutation_fname = nullptr; 
	unsigned int pre_sorted_filters = 0;
	unsigned int thread_count = 0;

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
		if (sscanf(argv[i],"SF=%u", &pre_sorted_filters)==1) {
			continue;
		} else
		if (sscanf(argv[i],"T=%u", &thread_count)==1) {
			continue;
		} else
		if (strcmp(argv[i],"-q")==0) {
			print_matching_results = false;
			continue;
		} else
		if (strcmp(argv[i],"--no-merge")==0) {
			merge_results = false;
			continue;
		} else
		if (strncmp(argv[i],"-Q",2)==0) {
			print_progress_steps = false;
			if (strncmp(argv[i],"-Qt",3)==0)
				print_matching_time_only = true;
			else if (strncmp(argv[i],"-Qm",3)==0)
				print_per_query_totals_only = true;
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

	if ((res = read_filters(filters_fname, filters_binary_format, pre_sorted_filters, print_progress_steps)) < 0) {
		cerr << endl << "couldn't read filters file: " << filters_fname << endl;
		return 1;
	};
	if (print_progress_steps)
		cout << "\t\t\t" << std::setw(12) << res << " filters." << endl;
	
	if (print_progress_steps)
		cout << "Reading packets..." << std::flush;
	if ((res = read_queries(queries_fname, queries_binary_format)) < 0) {
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

	if (print_progress_steps)
		cout << "Consolidating FIB... " << std::flush;

	P.consolidate();
	
	if (print_progress_steps) 
		cout << endl;

	if (print_progress_steps) 
		cout << "Matching packets... " << std::flush;
	
	high_resolution_clock::time_point start;
	high_resolution_clock::time_point stop;

	if (print_matching_results) {
		match_vector handler;

		start = high_resolution_clock::now();

		for(twitter_packet & p : packets) {
			if (use_identity_permutation)
				P.find_all_subsets(p.filter, handler);
			else
				P.find_all_subsets(apply_permutation(p.filter), handler);

			handler.print_results();
			handler.clear();
		}

		stop = high_resolution_clock::now();

	} else {
		packet_idx = 0;
		void (*matcher_function)();

		if (print_per_query_totals_only)
			matcher_function = match_and_merge_output;
		else if (merge_results)
			matcher_function =  match_and_merge;
		else
			matcher_function = match_only;

		if (thread_count > 1) {
			std::thread * T[thread_count];
			matchers_hold = true;

			for(unsigned int i = 0; i < thread_count; ++i)
				T[i] = new std::thread(matcher_function);

			start = high_resolution_clock::now();

			matchers_hold = false;
			for(unsigned int i = 0; i < thread_count; ++i)
				T[i]->join();

			stop = high_resolution_clock::now();

			for(unsigned int i = 0; i < thread_count; ++i)
				delete(T[i]);
		}
		else {
			start = high_resolution_clock::now();
			matcher_function();
			stop = high_resolution_clock::now();
		}
	}

	if (print_progress_steps) {
		cout << "\t\t\t" << std::setw(10)
			 << duration_cast<nanoseconds>(stop - start).count()/packets.size() 
			 << "ns average matching time." << endl;
	} else if (print_matching_time_only) {
		cout << duration_cast<nanoseconds>(stop - start).count()/packets.size() << endl;
	}

	P.clear();

	return 0;
}
