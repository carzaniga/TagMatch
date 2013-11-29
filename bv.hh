#ifndef BV_HH_INCLUDED
#define BV_HH_INCLUDED

#ifdef BV_USE_MEMSET
#include <cstring>
#endif

#include <cstdint>
#include <string>
#include <iostream>

// Implementation of a generic bitvector.
//
template <unsigned int Size>
class bv {
	typedef uint64_t block_t;
	static const int BLOCK_SIZE = 64;
	static const int BLOCK_COUNT = Size / BLOCK_SIZE;
	static const block_t BLOCK_ONE = 0x1;
#ifdef HAVE_STATIC_ASSERT
	static_assert((Size % 64) == 0);
#endif

#ifdef ALIGN_BV_TO_16BYTES
	block_t b[BLOCK_COUNT] __attribute__ ((aligned(16)));
#else
	block_t b[BLOCK_COUNT];
#endif

#ifdef WITH_VECTORIZATION
	static const block_t * PrefixMask[Size];

	static bool init_prefix_masks() {
		for(int i = 0; i < Size; ++i) {
			block_t b = new block_t[BLOCK_COUNT];
			for(int j = 0; j < BLOCK_COUNT; ++j) {
				if (j > (i / BLOCK_SIZE)) {
					res[j] = ~0UL;
				} else if (j < (i / BLOCK_SIZE)) {
					res[j] = 0;
				} else {
					res[j] = ~0 <<  (i % BLOCK_SIZE);
				}
			}
			PrefixMask[i] = b;
		}
	}

	static const is_prefix_initialized = init_prefix_masks();
#endif

public:
	typedef uint16_t pos_t;
	static const pos_t NULL_POSITION = Size;
	static const pos_t WIDTH = Size;

	bv() { reset(); }

	bv(const bv &x) { 
		for(int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = x.b[i];
	}

	bv(const std::string &s) {
		reset();
		std::string::const_iterator si = s.begin();
		for(int i = BLOCK_COUNT - 1; i >= 0; --i) {
			for(block_t mask = (BLOCK_ONE << (BLOCK_SIZE - 1)); mask != 0; mask >>= 1) {
				if (si != s.end()) {
					if (*si == '1')
						b[i] |= mask;
					++si;
				} else {
					return;
				}
			}
		}
	}

	std::ostream & print(std::ostream &os) const {
		for(int i = BLOCK_COUNT - 1; i >= 0; --i) {
			for(block_t mask = BLOCK_ONE << (BLOCK_SIZE - 1); mask != 0; mask >>= 1) {
				os << ((b[i] & mask) ? '1' : '0');
			}
		}
		return os;
	}

	void reset() {
#ifdef BV_USE_MEMSET
		memset(b,0,sizeof(b)); 
#else
		for(int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = 0;
#endif
	}

	void add(const bv & x) {
		for(int i = 0; i < BLOCK_COUNT; ++i)
			b[i] |= x.b[i];
	}

	bv & operator=(const bv & x) {
		for(int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = x.b[i];
		return *this;
	}

	bool operator == (const bv & x) const {
#ifdef AVOID_BRANCHING
		block_t B = 0;
		for(int i = 0; i < BLOCK_COUNT; ++i)
			B |= (b[i] ^ x.b[i]);

		return (B == 0);
#else
		for(int i = 0; i < BLOCK_COUNT; ++i) {
			if (b[i] != x.b[i])
				return false;
		}
		return true;
#endif
	}

	bool operator < (const bv & x) const {
		for(int i = BLOCK_COUNT - 1; i > 0; --i)
			if (b[i] < x.b[i])
				return true;
			else if (b[i] > x.b[i])
				return false;

		return (b[0] < x.b[0]);
	}

	pos_t popcount() const {
		pos_t c = 0;
		for(int i = 0; i < BLOCK_COUNT; ++i)
			c += popcount(b[i]);
		return c;
	}

	bool prefix_subset_of(const bv & x, pos_t pp, pos_t p) const {
		//
		// Check that *this is a subset of x only up to position pos,
		// starting from the left (most significant) down to position
		// pos, including position pos, as illustrated below:
		//
		//   prefix checked      rest of the bits are ignored
		// |######----------#####################################|
		//  ^191  ^pp      ^p                                  0^
		// 
		for(int i = BLOCK_COUNT - 1; i >= 0; --i) {
			pos_t right = BLOCK_SIZE*i; // right boundary of the ith block
			pos_t left = right + BLOCK_SIZE;  // one-past left boundary of the ith block
			
			if (p >= left)
				return true;

		    if (pp < right) 
				continue;

			if (p > right) {
				return (((b[i] & ~x.b[i]) >> (p - right)) == 0);
			} else if ((b[i] & ~x.b[i]) != 0) {
				return false;
			}
		}
		return true;
	}

	bool prefix_subset_of(const bv & x, pos_t p) const {
		//
		// Check that *this is a subset of x only up to position pos,
		// starting from the left (most significant) down to position
		// pos, including position pos, as illustrated below:
		//
		//   prefix checked      rest of the bits are ignored
		// |----------------#####################################|
		//  ^191           ^pos                                0^
		// 
		for(int i = BLOCK_COUNT - 1; i >= 0; --i) {
			pos_t right = BLOCK_SIZE*i; // right boundary of the ith block
			if (p >= right + BLOCK_SIZE)
				return true;
			if (p > right) {
				return (((b[i] & ~x.b[i]) >> (p - right)) == 0);
			} else if ((b[i] & ~x.b[i]) != 0) {
				return false;
			}
		}
		return true;
	}

	bool subset_of(const bv & x) const {
		// return true iff *this is a subset of x
#ifdef AVOID_BRANCHING
		block_t B = 0;
		for(int i = 0; i < BLOCK_COUNT; ++i)
			B |= (b[i] & ~x.b[i]);

		return (B == 0);
#else
		for(int i = 0; i < BLOCK_COUNT; ++i) {
			if ((b[i] & ~x.b[i]) != 0)
				return false;
		}
		return true;
#endif
	}

	bool suffix_subset_of(const bv & x, pos_t p) const {
		// this is essentially a subset_of check with an advice that
		// all bits to the left of position p (i.e., positions > p)
		// can be skipped, because they have been already checked
		// earlier.  So, this is not really a suffix-subset match in
		// the sense analogous to prefix_subset_of above.  In other
		// words, if p < 64, we only need to check the rightmost block
		// b[0], if p < 128 we only need to check b[1] and b[0],
		// otherwise we need to check all three blocks b[2], b[1],
		// and b[0].
#ifdef AVOID_BRANCHING
		block_t B = 0;
		for(int i = 0; i < BLOCK_COUNT; ++i) {
			B |= (b[i] & ~x.b[i]);
		}
		return (B == 0);
#else
		pos_t left = BLOCK_SIZE;
		for(int i = 0; i < BLOCK_COUNT; ++i) {
			if ((b[i] & ~x.b[i]))
				return false;
			if (p < left)
				return true;
			left += BLOCK_SIZE;
		}
		return true;
#endif
	}

	void set(pos_t pos) {
		b[pos/BLOCK_SIZE] |= (BLOCK_ONE << (pos % BLOCK_SIZE));
	}

	bool at(pos_t pos) const {
		return b[pos/BLOCK_SIZE] & (BLOCK_ONE << (pos % BLOCK_SIZE));
	}

	bool operator[](pos_t pos) const {
		return b[pos/BLOCK_SIZE] & (BLOCK_ONE << (pos % BLOCK_SIZE));
	}

	static pos_t log2(uint64_t v) {
#ifdef HAVE_BUILTIN_CLZL
		return 63 - __builtin_clzl(v);
#else
		// code taken from http://graphics.stanford.edu/~seander/bithacks.html
		uint64_t r = 0;

		if (v & 0xFFFFFFFF00000000) {
			v >>= 32;
			r |= 32;
		}
		if (v & 0xFFFF0000) {
			v >>= 16;
			r |= 16;
		}
		if (v & 0xFF00) {
			v >>= 8;
			r |= 8;
		}
		if (v & 0xF0) {
			v >>= 4;
			r |= 4;
		}
		if (v & 0xC) {
			v >>= 2;
			r |= 2;
		}
		if (v & 0x2) {
			v >>= 1;
			r |= 1;
		}
		return r;
#endif
	}

	static pos_t popcount(uint64_t v) {
#ifdef HAVE_BUILTIN_POPCOUNT
		return __builtin_popcount(v);
#else
		// taken from http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
		v = v - ((v >> 1) & 0x5555555555555555);
		v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333);
		v = (v + (v >> 4)) & 0x0f0f0f0f0f0f0f0f;
		v = (v * 0x0101010101010101) >> 56;
		return v;
#endif
	}

	static pos_t most_significant_diff_pos(const bv &x, const bv &y) {
		//
		// returns the index of the most significant bit that is
		// different in x and y.  For example, x="101100",
		// y="101110..." => result=188
		//
		for(int i = BLOCK_COUNT - 1; i > 0; --i) {
			block_t d = x.b[i] ^ y.b[i];
			if (d)
				return i*BLOCK_SIZE + log2(d);
		}
		return log2(x.b[0] ^ y.b[0]);
	}

	pos_t most_significant_one_pos() const {
		//
		// returns the index of the most significant 1-bit.  For
		// example, x="001110..." => result=189
		//
		for(int i = BLOCK_COUNT - 1; i > 0; --i)
			if (b[i])
				return i*BLOCK_SIZE + log2(b[i]);

		return log2(b[0]);
	}
};

template <unsigned int Size>
inline std::ostream & operator << (std::ostream & os, const bv<Size> & x) {
	return x.print(os);
}

#endif
