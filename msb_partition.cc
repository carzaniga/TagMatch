#ifdef HAVE_CONFIG_H
#include "config.h"
#else
#define HAVE_BUILTIN_CTZL
#endif
#include <iostream>
#include <climits>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cassert>

using namespace std;
using namespace std::chrono;

// ***DESIGN OF THE FORWARDING TABLE***
//
// The FIB maps a filter prefix to a queue.
// 
// FIB: prefix -> queue
//
// Filters are 192-bit wide.  Thus a prefix is at most 192-bit wide.
//

//
// MAIN IDEA: we partition the set of prefixes by the position X of
// their leftmost 1-bit.  This is the leftmost bit equal to 1.  Thus
// we have a map Prefixes: {0...191} -> (prefix -> queue).
// 
// Since we represent filters using 64-bit blocks, we further split
// the map into three vectors of 64 positions each.  The first vector
// pp1 holds all the prefixes whose leftmost 1-bit is in position
// 0<=X<64; the second vector pp2 holds all the prefixes whose
// leftmost 1-bit is in position 64<=X<128; and the third vector pp3
// holds all the prefixes whose leftmost 1-bit is in position X>=128.
// We then further split each vector as follows: pp1.p64 contains
// prefixes of total lenght <= 64; pp1.p128 contains prefixes of total
// lenght between 64 and 128; pp1.p192 contains prefixes of total
// lenght > 128; similarly, pp2.p64 and pp3.p64 contains prefixes of
// total lenght <= 64; and pp2.p128 contains prefixes of total length
// between 128 and 192.
//
// This way we can use specialized subset checks for each combination
// of block-length.

//
// IMPORTANT:
// BIT LAYOUT: we represent a prefix with the bit pattern in reverse
// order.  That is, the first bit is the least significant bit, and
// the pattern goes from left-to-right from the least significant bit
// towards the most significant bit.
//

// 
// We use blocks of 64 bits...
//
typedef uint64_t block_t;
static_assert(sizeof(block_t)*CHAR_BIT == 64, "uint64_t must be a 64-bit word");
static const int BLOCK_SIZE = 64;
static const block_t BLOCK_ONE = 0x1;

//
// leftmost 1-bit position 
//
#ifdef HAVE_BUILTIN_CTZL
static inline int leftmost_bit(const uint64_t x) {
    // Since we represent the leftmost bit in the least-significant
    // position, the leftmost bit corresponds to the count of trailing
    // zeroes (see the layout specification above).
    return __builtin_ctzl(x);
} 
#else
static inline int leftmost_bit(uint64_t x) {
    int n = 0;
	if ((x & 0xFFFFFFFF) == 0) {
		n += 32;
		x >>= 32;
	}
	if ((x & 0xFFFF) == 0) {
		n += 16;
		x >>= 16;
	}
	if ((x & 0xFF) == 0) {
		n += 8;
		x >>= 8;
	}
	if ((x & 0xF) == 0) {
		n += 4;
		x >>= 4;
	}
	if ((x & 0x3) == 0) {
		n += 2;
		x >>= 2;
	}
	if ((x & 0x1) == 0) {
		n += 1;
	}
    return n;
}
#endif

//
// Main representation of a prefix.  Essentially we will instantiate
// this template with Size=64, Size=128, and Size=192.
//
template <unsigned int Size>
class prefix {
    static_assert((Size % 64) == 0, "prefix width must be a multiple of 64");
    static const int BLOCK_COUNT = Size / BLOCK_SIZE;
    // 
    // BIT LAYOUT: a prefix is represented with the bit pattern in
    // reverse order.  That is, the first bit is the least significant
    // bit, and the pattern goes from left-to-right from the least
    // significant bit towards the most significant bit.  Notice that
    // we do not store the length of a prefix.  So, a prefix of one
    // bit will still be stored as a 64-bit quantity with all the
    // trailing bits set to 0.
    // 
    // EXAMPLE:
    // prefix "000101" is represented by the three blocks:
    // b[0] = (101000)binary, b[1] = 0, b[2] = 0
    //
    block_t b[BLOCK_COUNT];
    
public:
    const block_t * begin() const {
		return b;
    }

    const block_t * end() const {
		return b + BLOCK_COUNT;
    }

    bool subset_of(const block_t * p) const {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			if ((b[i] & ~p[i]) != 0)
				return false;

		return true;
    }

    prefix(const string & p) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = 0;

		assert(p.size() <= Size);

		// see the layout specification above
		//
		block_t mask = BLOCK_ONE;
		int i = 0;
		for(string::const_iterator c = p.begin(); c != p.end(); ++c) {
			if (*c == '1')
				b[i] |= mask;

			mask <<= 1;
			if (mask == 0) {
				mask = BLOCK_ONE;
				if (++i == BLOCK_COUNT)
					return;
			}
		}
    }

    prefix(const block_t * p) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = p[i];
    }
};

typedef prefix<192> filter_t;

//
// place-holder class for the message queue on the GPU
//
class queue {};

//
// association between a prefix of up to 64 bits, and a queue
//
struct queue64 {
    const prefix<64> p;
    queue * q;

    queue64(const block_t * pb, queue * q_): p(pb), q(q_) {};
};

//
// association between a prefix of up to 128 bits, and a queue
//
struct queue128 {
    const prefix<128> p;
    queue * q;

    queue128(const block_t * pb, queue * q_): p(pb), q(q_) {};
};

//
// association between a prefix of up to 192 bits, and a queue
//
struct queue192 {
    const prefix<192> p;
    queue * q;

    queue192(const block_t * pb, queue * q_): p(pb), q(q_)  {};
};

// 
// container of prefixes whose leftmost bit is the 3rd 64-bit block.
// 
class p3_container {
public:
    // Since this is the third of three blocks, it may only contain
    // prefixes of up to 64 bits.
    //
    vector<queue64> p64;

    void add64(const block_t * p, queue * q) {
		p64.push_back(queue64(p,q));
    }
};

// 
// container of prefixes whose leftmost bit is the 2nd 64-bit block.
// 
class p2_container : public p3_container {
public:
    // Since this is the second of three blocks, it may contain
    // prefixes of up to 64 bits (inherited from p3_container) and
    // prefixes of up to 128 bits.
    //
    vector<queue128> p128;

    void add128(const block_t * p, queue * q) {
		p128.push_back(queue128(p,q));
    }
};

// 
// container of prefixes whose leftmost bit is the 1st 64-bit block.
// 
class p1_container : public p2_container {
public:
    // Since this is the second of three blocks, it may contain
    // prefixes of up to 64 and 128 bits (inherited from p3_container
    // and p2_container) plus prefixes of up to 192 bits.
    //
    vector<queue192> p192;

    void add192(const block_t * p, queue * q) {
		p192.push_back(queue192(p,q));
    }
};

static p1_container pp1[64];
static p2_container pp2[64];
static p3_container pp3[64];

void fib_add_prefix(const filter_t & f, unsigned int n, queue * q) {
    const block_t * b = f.begin();

    if (*b) {
		if (n <= 64) {
			pp1[leftmost_bit(*b)].add64(b,q);
		} else if (n <= 128) {
			pp1[leftmost_bit(*b)].add128(b,q);
		} else {
			pp1[leftmost_bit(*b)].add192(b,q);
		}
    } else if (*(++b)) {
		if (n <= 64) {
			pp2[leftmost_bit(*b)].add64(b,q);
		} else {
			pp2[leftmost_bit(*b)].add128(b,q);
		}
    } else if (*(++b)) {
		pp3[leftmost_bit(*b)].add64(b,q);
    }
}

unsigned int fib_match(const filter_t & q) {
	unsigned count = 0;
    const block_t * b = q.begin();

    if (*b) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p1_container & c = pp1[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b))
					++count;

			for(vector<queue128>::const_iterator i = c.p128.begin(); i != c.p128.end(); ++i) 
				if (i->p.subset_of(b))
					++count;

			for(vector<queue192>::const_iterator i = c.p192.begin(); i != c.p192.end(); ++i) 
				if (i->p.subset_of(b))
					++count;

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);
			
    } else if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p2_container & c = pp2[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b))
					++count;

			for(vector<queue128>::const_iterator i = c.p128.begin(); i != c.p128.end(); ++i) 
				if (i->p.subset_of(b))
					++count;

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);

    } else if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p3_container & c = pp3[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b))
					++count;

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);
    }

	return count;
}

queue Q;

int main () {
    int N = 1000;				// how many cycles throug the queries?
    vector<filter_t> queries;	// we store all the queries here

    string command, filter_string;

    while(std::cin >> command >> filter_string) {
		if (command == "+") {
			filter_t f(filter_string);
			unsigned int n = filter_string.size();
			fib_add_prefix(f,n,&Q);
		} else if (command=="!") {
			filter_t f(filter_string);
			queries.push_back(f);
		} else if (command == "match") {
			break;
		}
    }

    high_resolution_clock::time_point start = high_resolution_clock::now();


    for(int j=0; j<N; j++) {
		for(size_t i = 0; i < queries.size(); ++i) {
			fib_match(queries[i].begin());
		}
    }

    high_resolution_clock::time_point stop = high_resolution_clock::now();

	for(size_t i = 0; i < queries.size(); ++i) {
		cout << fib_match(queries[i].begin()) << endl;
	}
		
    nanoseconds ns = duration_cast<nanoseconds>(stop - start);
    cout << "Average matching time: " << ns.count()/queries.size()/N 
		 << "ns" << endl
		 << "queries: " << queries.size() << endl;

    return 0;
}
