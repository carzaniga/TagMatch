#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <bitset>
#include <vector>
#include <sstream>
#include <fstream>

#include "predicate.hh"

const unsigned int M = filter_t::WIDTH;

static std::vector< std::bitset<M> > filters_dictionary;

static void read_filters_dictionary(std::istream & is) {
	filters_dictionary.clear();
	std::string line;

	while(std::getline(is, line)) {
		std::istringstream line_s(line);
		
		std::string command;

		line_s >> command;

		if (command != "+")
			continue;

		unsigned int iface;
		std::string filter;

		line_s >> iface >> filter;

		filters_dictionary.push_back(std::bitset<M>(filter));
	}
}

class workload_spec {
public:
	const char * command;
	unsigned int seed;
	unsigned int n;
	unsigned int k;
	unsigned int a_min;
	unsigned int a_max;
	unsigned int i_min;
	unsigned int i_max;
	unsigned int t_min;
	unsigned int t_max;
	const char * filters_file;
	unsigned int filters_count;

	void generate() const;

	workload_spec():
		command("!"),
		seed(0),
		n(1000),
		k(7),
		a_min(1),
		a_max(10),
		i_min(0),
		i_max(255),
		t_min(0),
		t_max(7),
		filters_file(nullptr),
		filters_count(0)
		{}
};

void workload_spec::generate() const {

	if (filters_file != nullptr) {
		if (strcmp(filters_file, "-")==0) {
			read_filters_dictionary(std::cin);
		} else {
			std::ifstream f(filters_file);
			if (!f) {
				std::cerr << "could not open filters file " << filters_file << std::endl;
				return;
			}
			read_filters_dictionary(f);
			f.close();
		}
	}

    std::default_random_engine generator(seed);
    std::uniform_int_distribution<unsigned int> random_position(0,M-1);
    std::uniform_int_distribution<unsigned int> random_fsize(a_min,a_max);
    std::uniform_int_distribution<unsigned int> random_ifx(i_min,i_max);
    std::uniform_int_distribution<unsigned int> random_tree(t_min,t_max);

	std::uniform_int_distribution<unsigned int> random_filter_pos(0, filters_dictionary.size() - 1);

    std::bitset<M> f;

    unsigned int bsize;
    for(unsigned int i = 0; i < n; ++i) {
		f.reset();

		if (filters_dictionary.size() > 0)
			for(unsigned int i = 0; i < filters_count; ++i)
				f |= filters_dictionary[random_filter_pos(generator)];

		for(bsize = ((a_min == a_max) ? a_min : random_fsize(generator)) * k; bsize > 0; --bsize)
			f.set(random_position(generator));

		std::cout << command << ' ' 
				  << ((i_min == i_max) ? i_min : random_ifx(generator)) << ' '
				  << f << std::endl;
    }
}

int main(int argc, char * argv[]) {

	std::ios_base::sync_with_stdio(false);

	bool done = false;
	workload_spec ws;

	for(int i = 1; i < argc; ++i) {
		switch (sscanf(argv[i],"a=%u,%u", &ws.a_min, &ws.a_max)) {
		case 1: ws.a_max = ws.a_min;
		case 2: continue;
		default: break;
		}

		switch (sscanf(argv[i],"i=%u,%u", &ws.i_min, &ws.i_max)) {
		case 1: ws.i_max = ws.i_min;
		case 2: continue;
		default: break;
		}

		switch (sscanf(argv[i],"t=%u,%u", &ws.t_min, &ws.t_max)) {
		case 1: ws.t_max = ws.t_min;
		case 2: continue;
		default: break;
		}

		if (sscanf(argv[i],"k=%u", &ws.k))
			continue;

		if (sscanf(argv[i], "n=%u", &ws.n) || sscanf(argv[i], "N=%u", &ws.n))
			continue;

		if (sscanf(argv[i], "s=%u", &ws.seed) || sscanf(argv[i], "seed=%u", &ws.seed))
			continue;

		if (strncmp(argv[i],"F=",2)==0) {
			ws.filters_file = argv[i] + 2;
			continue;
		}

		if (sscanf(argv[i], "Fc=%u", &ws.filters_count))
			continue;

		if (strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0) {
			std::cout << "usage: " << argv[0] << " <params> <command> [<params> <command>...]\n"
				"\n  params: any combination of the following:\n"
				"     [s=<seed>]      :: seed for random generator (default=0)\n"
				"     [k=<N>]         :: number of hash functions in Bloom filters  (default=7)\n"
				"     [n=<N>]         :: number of generated filters (default=1000)\n"
				"     [a=<N1>[,<N2>]] :: number or range of tags per filters (default=1,10)\n"
				"     [i=<N1>[,<N2>]] :: number or range of interface ids (default=0,0)\n"
				"     [t=<N1>[,<N2>]] :: number or range of tree ids (default=0,0)\n"
				"     [F=<filename>]  :: file containing filters ('-'=stdin, default=none)\n"
				"     [Fc=<N>]        :: number of random filters to combine from F (default=0)\n"
				"\n  command: +|!|?|sub|sup (commands understood by the driver program)\n" 
				"\nExamples:\n"
				"gen2 n=10 +\n"
				"  generates two subscription filters with between 1 and 10 attributes\n"
				"gen2 n=1000 a=5 !\n"
				"  generates 1000 message filters each with exactly 5 tags\n"
				"gen2 n=5000 a=2,6 + n=10000 a=8,10 !\n"
				"  generates 5000 subscriptions with between 2 and 6 attributes, followed\n"
				"  by 10000 messages with between 8 and 10 attributes\n"
				"gen2 n=5000 a=2,6 + n=10000 a=8 ! n=10000 a=10 ! n=10000 a=12 !\n"
				"  generates 5000 subscriptions with between 2 and 6 attributes, followed by\n"
				"  three series of 1000 messages with 8, 10, and 12 attributes, respectively.\n"
				"gen2 n=5000 a=2,6 i=1,16 t=0,7 + n=1000000 a=8 i=0 !\n"
				"  generates 5000 subscriptions with between 2 and 6 attributes distributed\n"
				"  uniformly over 8 trees and 16 interfacesbetween, followed by 1000000\n"
				"  messages with 8 attributes.\n"
				"gen2 n=1000 F=- Fc=3 a=0,2 !\n"
				"  generates 1000 messages by combining 3 filters from those read from\n"
				"  standard input (F=-) and by adding between 0 and 2 attributes.\n"
				"  the filters are read in the same format output by this tool, that is,\n"
				"  each line start with a '+' command followed by tree id, an interface id,\n"
				"  and a bitvector pattern.\n"
					  << std::endl;
			return 1;
		}

		ws.command = argv[i];
		ws.generate();
		done = true;
	}

	if (!done)
		ws.generate();

	return 0;
}
