#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

#include "fib.hh"
#include "query.hh"

//
// DOCUMENTATION for the sample_worload program:
//
// Read a workload of N filters (without partitions) and outputs a
// workload of K filters chosen at random.  This program can be used
// to obtain smaller workloads from a base (larger) workload.
//
// More specifically, sample K filters from the first N input filters.
// So, this program can also be use simply to extract the first m
// filters by setting both N=m and K=m.
//
class sampler {
private:
    std::default_random_engine generator;

	unsigned long n;
	unsigned long k;

public:
	sampler(unsigned int seed, unsigned long n_, unsigned long k_)
		: generator(seed), n(n_), k(k_) {};

	bool sample() {
		if (k == 0)
			return false;

		if (k < n) {
			--n;
			std::uniform_int_distribution<unsigned long> random_choice(0,n);
			if (random_choice(generator) < k) {
				--k;
				return true;
			} else {
				return false;
			}
		} else {
			--k;
			return true;
		}
	}

	bool needs_more() {
		return (k > 0);
	}
};

static int sample_filters(std::istream & input, std::ostream & output, const bool binary_format,
						  sampler & s) {
	fib_entry f;

	if (binary_format) {
		while(s.needs_more() && f.read_binary(input))
			if (s.sample())
				f.write_binary(output);
	} else {
		while(s.needs_more() && f.read_ascii(input))
			if (s.sample())
				f.write_ascii(output);
	}
	return 0;
}

static int sample_packets(std::istream & input, std::ostream & output, const bool binary_format,
						  sampler & s) {
	basic_query q;

	if (binary_format) {
		while(q.read_binary(input))
			if (s.sample())
				q.write_binary(output);
	} else {
		while(q.read_ascii(input))
			if (s.sample())
				q.write_ascii(output);
	}
	return 0;
}

static void print_usage(const char * progname) {
	std::cout << "usage: " << progname
			  << " [in=<filename>] [out=<filename>] [-a] N=<total-read> K=<choices>"
			  << std::endl
			  << "options:" << std::endl
			  << "\t-a\t: uses ASCII format for I/O (default is binary)" << std::endl
			  << "\t-p\t: reads and outputs packets (default is filters)" << std::endl;
}

int main(int argc, const char * argv[]) {
	bool binary_format = true;

	const char * input_fname = nullptr;
	const char * output_fname = nullptr;

	int (*sample_workload)(std::istream &, std::ostream &, const bool, sampler &);
	sample_workload = sample_filters;

	unsigned long N = 0;
	unsigned long K = 0;

	std::random_device rd;
	unsigned int seed = rd();

	for(int i = 1; i < argc; ++i) {
		if (strncmp(argv[i],"in=",3)==0) {
			input_fname = argv[i] + 3;
		} else
		if (strncmp(argv[i],"out=",4)==0) {
			output_fname = argv[i] + 4;
		} else
		if (strcmp(argv[i],"-a")==0) {
			binary_format = false;
		} else
		if (strcmp(argv[i],"-p")==0) {
			sample_workload = sample_packets;
		} else
		if (sscanf(argv[i],"N=%lu", &N) > 0) {
			continue;
		} else
		if (sscanf(argv[i],"K=%lu", &K) > 0) {
			continue;
		} else {
		if (sscanf(argv[i],"seed=%u", &seed) > 0) {
			continue;
		} else
			print_usage(argv[0]);
			return 1;
		}
	}

	if (N == 0 || K == 0) {
		print_usage(argv[0]);
		return 1;
	}

	int res;

	sampler s(seed, N, K);

	if (input_fname != nullptr) {
		std::ifstream input_file(input_fname);
		if (!input_file) {
			std::cerr << "could not open input file " << input_fname << std::endl;
			return 1;
		}
		if (output_fname != nullptr) {
			std::ofstream output_file(output_fname);
			if (!output_file) {
				std::cerr << "could not open output file " << output_fname << std::endl;
				return 1;
			}
			res = sample_workload(input_file, output_file, binary_format, s);
			output_file.close();
		} else {
			res = sample_workload(input_file, std::cout, binary_format,s);
		}
		input_file.close();
	} else {
		if (output_fname != nullptr) {
			std::ofstream output_file(output_fname);
			if (!output_file) {
				std::cerr << "could not open output file " << output_fname << std::endl;
				return 1;
			}
			res = sample_workload(std::cin, output_file, binary_format, s);
			output_file.close();
		} else {
			res = sample_workload(std::cin, std::cout, binary_format, s);
		}
	}

	return res;
}
