#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <bitset>

#include "predicate.hh"

const unsigned int M = filter_t::WIDTH;

void do_gen(unsigned int seed, 
			const char * command,
			unsigned int if_id,
			unsigned int tree_id,
			unsigned int N,
			unsigned int k,
			unsigned int Amin,
			unsigned int Amax) {
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<unsigned int> random_position(0,M-1);
    std::uniform_int_distribution<unsigned int> random_fsize(Amin,Amax);

    std::bitset<M> f;
    unsigned int bsize;
    for(unsigned int i = 0; i < N; ++i) {
		f.reset();
		bsize = random_fsize(generator) * k;
		for(unsigned int j = 0; j < bsize; ++j)
			f.set(random_position(generator));

		std::cout << command << ' ' << tree_id << ' ' << if_id << ' ';
		for(unsigned int j = 0; j < M; ++j)
			std::cout << (f[j] ? '1' : '0');
		std::cout << std::endl;
    }
}

int main(int argc, char * argv[]) {

    unsigned int N = 1000;
    unsigned int k = 7;
    unsigned int Amin = 1;
    unsigned int Amax = 10;

    unsigned int if_id = 0;
    unsigned int tree_id = 0;
    unsigned int seed = 0;
    
	bool done = false;

	for(int i = 1; i < argc; ++i) {
		if (sscanf(argv[i],"k=%u", &k))
			continue;

		switch (sscanf(argv[i],"a=%u,%u", &Amin, &Amax)) {
		case 1: Amax=Amin;
		case 2: continue;
		default: break;
		}

		if (sscanf(argv[i], "n=%u", &N) || sscanf(argv[i], "N=%u", &N))
			continue;

		if (sscanf(argv[i], "s=%u", &seed) || sscanf(argv[i], "seed=%u", &seed))
			continue;

		if (sscanf(argv[i], "i=%u", &if_id) || sscanf(argv[i], "interface=%u", &if_id))
			continue;

		if (sscanf(argv[i], "t=%u", &tree_id) || sscanf(argv[i], "tree=%u", &tree_id))
			continue;

		if (strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0) {
			std::cout << "usage: " << argv[0] << " <params> <command> [<params> <command>...]\n"
				"\n  params: any combination of the following:\n"
				"     [s|seed=<seed>] :: seed for random generator (default=0)\n"
				"     [k=<N>] :: number of hash functions in Bloom filters  (default=7)\n"
				"     [n|N=<N>] :: number of generated filters (default=1000)\n"
				"     [a=<A>[,<A2>]] :: number or range of tags per filters (default=1,10)\n"
				"     [i|interface=<interface>] :: interface number (default=0)\n"
				"     [t|tree=<tree>] :: tree number (default=0)\n"
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
					  << std::endl;
			return 1;
		}

		do_gen(seed, argv[i], if_id, tree_id, N, k, Amin, Amax);
		done = true;
	}

	if (!done)
		do_gen(seed, "+", if_id, tree_id, N, k, Amin, Amax);

	return 0;
}
