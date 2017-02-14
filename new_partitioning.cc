
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <atomic>
#include <climits>

#include "packet.hh"
#include "fib.hh"

using std::vector;
using std::endl;
using std::cout;

static void print_usage(const char* progname) {
	std::cerr << "usage: " << progname
	<< " [<params>...]\n"
	"\n  params: any combination of the following:\n"
	"     [m=<N>]         :: maximum size for each partition "
	"(default=100)\n"
	"     [p=<filename>]  :: output for prefixes, '-' means stdout "
	"(default=OFF)\n"
	"     [f=<filename>]  :: output for filters, '-' means stdout "
	"(default=OFF)\n"
	"     [in=<filename>]  :: input for filters (default=stdin)\n"
	"     [-a]  :: ascii input\n"
	"     [-b]  :: binary input\n"
	<< std::endl;
}
static vector<fib_entry> fibs;
static vector<int> global_fib_index;
static unsigned int max_size = 300000;
static int partition_counter(0);
static bool binary_format = false;
static std::ostream* prefixes_output = nullptr;
static std::ostream* filters_output = nullptr;
static int progress = 0;
static int totalSize = 0;

void split(filter_t partition, filter_t checked, vector<int>& fib_index) {
	if (fib_index.size() == 0) return;
	
	//If this condition holds, we output the partition
	if ((partition.popcount() <= 7 && fib_index.size() < 50000) ||
		(partition.popcount() > 7 && fib_index.size() <= max_size &&
		 partition.popcount() != 0)) {

			progress += fib_index.size();
			std::cerr << "\r                                  \rprogress: "
			<< ((progress)*100.0) / totalSize << "%";
			
			unsigned int freqs[filter_t::WIDTH] = {0};
			unsigned int fib_index_size = fib_index.size();
			
			for (unsigned int i = 0; i < fib_index_size; i++) {
				for (unsigned int p = fibs[fib_index[i]].filter.next_bit(0);
					 p < filter_t::WIDTH; p = fibs[fib_index[i]].filter.next_bit(p + 1))
					freqs[p] += 1;
			}
			// here we add bits that are common among all the filters in the partition
			// to the partition representative.
			for (unsigned int p = 0; p < filter_t::WIDTH; p++)
				if (freqs[p] == fib_index.size()) partition.set_bit(p);
			
			partition_prefix prefix;
			int pid = partition_counter;
			prefix.filter = partition;
			prefix.partition = pid;
			prefix.length = filter_t::WIDTH;
			prefix.size = fib_index.size();
			
			partition_counter++;
			
			if (prefixes_output) {
				if (binary_format) {
					prefix.write_binary(*prefixes_output);
				} else {
					prefix.write_ascii(*prefixes_output);
				}
			}
			partition_fib_entry f;
			if (filters_output) {
				if (binary_format)
					for (unsigned int i = 0; i < fib_index.size(); i++) {
						f.filter = fibs[fib_index[i]].filter;
						f.ti_pairs = fibs[fib_index[i]].ti_pairs;
						f.filter = fibs[fib_index[i]].filter;
						f.partition = pid;
						f.write_binary(*filters_output);
					}
				else
					for (unsigned int i = 0; i < fib_index.size(); i++) {
						f.filter = fibs[fib_index[i]].filter;
						f.ti_pairs = fibs[fib_index[i]].ti_pairs;
						f.filter = fibs[fib_index[i]].filter;
						f.partition = pid;
						f.write_ascii(*filters_output);
					}
			}
			fib_index.clear();
			return;
		}
	unsigned int freqs[filter_t::WIDTH] = {0};
	unsigned int fib_index_size = fib_index.size();
	
	for (unsigned int i = 0; i < fib_index_size; i++) {
		for (unsigned int p = fibs[fib_index[i]].filter.next_bit(0);
			 p < filter_t::WIDTH; p = fibs[fib_index[i]].filter.next_bit(p + 1))
			freqs[p] += 1;
	}
	
	int cut_pos = -1;
	int min_distance = INT_MAX;

#if 1 
	// This is a micro optimization designed to exploit some properties of our workload.
	// For some workloads, this was not used.
	
	if (partition.popcount() == 0) {
		int cut_pos_ar[7] = {-1};
		vector<int> left, right;
		filter_t l(partition), r(partition);
		filter_t c(checked);
		
		for (int k = 0; k < 7; k++) {
			unsigned int max = 0;
			for (unsigned int i = 0; i < filter_t::WIDTH; i++)
				if (freqs[i] > max) {
					max = freqs[i];
					cut_pos_ar[k] = i;
				}
			if (cut_pos_ar[k] == -1) {
				cout << "error occured." << endl;
				exit(-1);
			}
			c.set_bit(cut_pos_ar[k]);
			r.set_bit(cut_pos_ar[k]);
			freqs[cut_pos_ar[k]] = 0;
		}
		
		for (unsigned int i = 0; i < fib_index.size(); i++) {
			if (fibs[fib_index[i]].filter[cut_pos_ar[0]] == 1 &&
				fibs[fib_index[i]].filter[cut_pos_ar[1]] == 1 &&
				fibs[fib_index[i]].filter[cut_pos_ar[2]] == 1 &&
				fibs[fib_index[i]].filter[cut_pos_ar[3]] == 1 &&
				fibs[fib_index[i]].filter[cut_pos_ar[4]] == 1 &&
				fibs[fib_index[i]].filter[cut_pos_ar[5]] == 1 &&
				fibs[fib_index[i]].filter[cut_pos_ar[6]] == 1)
				right.push_back(fib_index[i]);
			else
				left.push_back(fib_index[i]);
		}
		
		split(l, c, left);
		split(r, c, right);
		
		fib_index.clear();
		return;
	} else {
		for (unsigned int p = 0; p < filter_t::WIDTH; p++) {
			int dis = abs(fib_index.size() / 2 - freqs[p]);
			if ((freqs[p] > 0) && (checked[p] == false) && (dis < min_distance)) {
				min_distance = dis;
				cut_pos = p;
			}
		}
#else
		if (cut_pos == -1)
			for (unsigned int p = 0; p < filter_t::WIDTH; p++) {
				int dis = abs(fib_index_size / 2 - freqs[p]);
				if ((freqs[p] > 0) && (checked[p] == false) && (dis < min_distance)) {
					min_distance = dis;
					cut_pos = p;
				}
			}
#endif
		
		if (cut_pos == -1) {
			std::cout << "something is seriously wrong." << std::endl;
			exit(-1);
		}
		vector<int> left, right;
		filter_t l(partition), r(partition);
		filter_t c(checked);
		c.set_bit(cut_pos);
		
		for (unsigned int i = 0; i < fib_index.size(); i++) {
			if (fibs[fib_index[i]].filter[cut_pos] == 1)
				right.push_back(fib_index[i]);
			else
				left.push_back(fib_index[i]);
		}
		split(l, c, left);
		r.set_bit(cut_pos);
		split(r, c, right);
		
		fib_index.clear();
	}
}

static void read_filters(std::istream& input, bool binary_format) {
	unsigned int count = 0;
	
	fib_entry f;
	
	if (binary_format) {
		while (f.read_binary(input)) {
			fibs.push_back(f);
			global_fib_index.push_back(count);
			count += 1;
		}
	} else {
		while (f.read_ascii(input)) {
			fibs.push_back(f);
			global_fib_index.push_back(count);
			count += 1;
		}
	}
	std::cerr << "reading filters is done" << std::endl;
}
int main(int argc, const char* argv[]) {
	const char* prefixes_fname = nullptr;
	const char* filters_fname = nullptr;
	const char* input_fname = nullptr;
	
	for (int i = 1; i < argc; ++i) {
		if (sscanf(argv[i], "m=%u", &max_size) ||
			sscanf(argv[i], "N=%u", &max_size))
			continue;
		if (strcmp(argv[i], "-a") == 0) {
			binary_format = false;
			continue;
		}
		if (strncmp(argv[i], "p=", 2) == 0) {
			prefixes_fname = argv[i] + 2;
			continue;
		}
		if (strncmp(argv[i], "f=", 2) == 0) {
			filters_fname = argv[i] + 2;
			continue;
		}
		
		if (strcmp(argv[i], "-b") == 0) {
			binary_format = true;
			continue;
		}
		if (strncmp(argv[i], "in=", 3) == 0) {
			input_fname = argv[i] + 3;
			continue;
		}
		print_usage(argv[0]);
		return 1;
	}
	
	std::ofstream prefixes_file;
	std::ofstream filters_file;
	std::ifstream input_file;
	
	if (input_fname != nullptr) {
		std::ifstream input_file(input_fname);
		if (!input_file) {
			std::cerr << "could not open input file " << input_fname << std::endl;
			return 1;
		}
		read_filters(input_file, binary_format);
		input_file.close();
	} else {
		read_filters(std::cin, binary_format);
	}
	
	if (prefixes_fname) {
		if (strcmp(prefixes_fname, "-") == 0) {
			prefixes_output = &std::cout;
		} else {
			prefixes_file.open(prefixes_fname);
			if (!prefixes_file) {
				std::cerr << "error opening prefixes file " << prefixes_fname
				<< std::endl;
				return -1;
			}
			prefixes_output = &prefixes_file;
		}
	}
	
	if (filters_fname) {
		if (strcmp(filters_fname, "-") == 0) {
			filters_output = &std::cout;
		} else {
			filters_file.open(filters_fname);
			if (!filters_file) {
				std::cerr << "error opening filters file " << filters_fname
				<< std::endl;
				return -1;
			}
			filters_output = &filters_file;
		}
	}
	
	filter_t partition, checked;
	partition.clear();
	checked.clear();
	
	unsigned int freqs[filter_t::WIDTH] = {0};
	for (unsigned int i = 0; i < global_fib_index.size(); i++) {
		for (unsigned int p = fibs[i].filter.next_bit(0); p < filter_t::WIDTH;
			 p = fibs[i].filter.next_bit(p + 1))
			freqs[p] += 1;
	}
	
	totalSize = global_fib_index.size();
	
	split(partition, checked, global_fib_index);
	
	if (prefixes_output == &prefixes_file) prefixes_file.close();
	if (filters_output == &filters_file) filters_file.close();
}
