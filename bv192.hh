#ifndef BV192_HH_INCLUDED
#define BV192_HH_INCLUDED

#ifdef BV192_USE_MEMSET
#include <cstring>
#endif

#include <cstdint>
#include <string>
#include <iostream>

// Implementation of a 192-bit bitvector.
//
class bv192 {
	typedef uint64_t block_t;
	block_t bv[3];

public:
	typedef uint8_t pos_t;
	static const pos_t NULL_POSITION = 192;
	static const pos_t WIDTH = 192;

	bv192() { reset(); }

	bv192(const bv192 &x) { 
		bv[0] = x.bv[0];
		bv[1] = x.bv[1];
		bv[2] = x.bv[2];
	}

	bv192(const std::string &s) {
		std::string::const_iterator si = s.begin();
		block_t * B = &(bv[2]);
		for(int i = 0; i < 3; ++i) {
			for(int j = 0; j < 64; ++j) {
				if (si != s.end()) {
					*B = (*B << 1) | ((*si == '1') ? 1UL : 0);
					++si;
				} else {
					return;
				}
			}
			--B;
		}
	}

	std::ostream & print(std::ostream &os) const {
		for(int i = 2; i >= 0; --i) {
			block_t B = bv[i];
			for(int j = 0; j < 64; ++j) {
				os << ((B & (1UL << 63)) ? '1' : '0');
				B <<= 1;
			}
		}
		return os << std::endl;
	}

	void reset() {
#ifdef BV192_USE_MEMSET
		memset(bv,0,sizeof(bv)); 
#else
		bv[0] = 0; 
		bv[1] = 0; 
		bv[2] = 0;
#endif
	}

	void add(const bv192 & x) {
		bv[0] |= x.bv[0];
		bv[1] |= x.bv[1];
		bv[2] |= x.bv[2];
	}

	bv192 & operator=(const bv192 &rhs) {
		bv[0] = rhs.bv[0];
		bv[1] = rhs.bv[1];
		bv[2] = rhs.bv[2];
		return *this;
	}

	bool operator == (const bv192 &rhs) const {
		return (bv[0] == rhs.bv[0] && bv[1] == rhs.bv[1] &&	bv[2] == rhs.bv[2]) ;
	}

	bool operator < (const bv192 &rhs) const {
		return bv[2] < rhs.bv[2] 
				|| (bv[2] == rhs.bv[2] && bv[1] < rhs.bv[1])
				|| (bv[1] == rhs.bv[1] && bv[0] < rhs.bv[0]);
	}

	bool prefix_subset_of(const bv192 & x, pos_t p) const {
		//
		// Check that *this is a subset of x only up to position pos,
		// starting from the left (most significant) down to position
		// pos, including position pos, as illustrated below:
		//
		//   prefix checked      rest of the bits are ignored
		// |----------------#####################################|
		//  ^191           ^pos                                0^
		// 
		if (p > 127)
			return (((bv[2] & x.bv[2]) ^ bv[2]) >> (p - 128)) == 0;

		if ((bv[2] & x.bv[2]) != bv[2])
			return false;

		if (p > 63)
			return (((bv[1] & x.bv[1]) ^ bv[1]) >> (p - 64)) == 0;

		if ((bv[1] & x.bv[1]) != bv[1])
			return false;

		return (((bv[0] & x.bv[0]) ^ bv[0]) >> p) == 0;
	}

	bool range_subset_of(pos_t p, const bv192 & x) const {
		if (p <= 64)
			return (((bv[2] & x.bv[2]) ^ bv[2]) >> (64 - p)) == 0;

		p -= 64;

		if (p <= 64)
			return (((bv[1] & x.bv[1]) ^ bv[1]) >> (64 - p)) == 0;

		p -= 64;

		return (((bv[0] & x.bv[0]) ^ bv[0]) >> (64 - p)) == 0;
	}

	bool subset_of(const bv192 & x) const {
		return 
			(bv[0] & x.bv[0]) == bv[0]
			&& (bv[1] & x.bv[1]) == bv[1]
			&& (bv[2] & x.bv[2]) == bv[2];
	}

	void set(pos_t pos) {
		bv[pos/64] |= (1UL << (pos % 63));
	}

	bool at(pos_t pos) const {
		return bv[pos/64] & (1UL << (pos % 64));
	}

	bool operator[](pos_t pos) const {
		return bv[pos/64] & (1UL << (pos % 64));
	}

#ifndef HAVE_BUILTIN_CLZL
	// code taken from http://graphics.stanford.edu/~seander/bithacks.html
	static pos_t log2(uint64_t v) {
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
	}
#endif

	static pos_t most_significant_diff_pos(const bv192 &x, const bv192 &y) {
		//
		// returns the index of the most significant bit that is
		// different in x and y.  For example, x="101100",
		// y="101110..." => result=188
		//
#ifdef HAVE_BUILTIN_CLZL
		block_t d = x.bv[2] ^ y.bv[2];
		if (d)
			return 191 - __builtin_clzl(d);

		d = x.bv[1] ^ y.bv[1];
		if (d)
			return 127 - __builtin_clzl(d);

		return 63 - __builtin_clzl(x.bv[0] ^ y.bv[0]);
#else
		block_t d = x.bv[2] ^ y.bv[2];
		if (d)
			return 128 + log2(d);

		d = x.bv[1] ^ y.bv[1];
		if (d)
			return 64 + log2(d);

		return log2(x.bv[0] ^ y.bv[0]);
#endif
	}
};


#endif
