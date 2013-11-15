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
		return os;
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
		return bv[0] == rhs.bv[0] && bv[1] == rhs.bv[1] && bv[2] == rhs.bv[2];
	}

	bool operator < (const bv192 &rhs) const {
		return bv[2] < rhs.bv[2] 
				|| (bv[2] == rhs.bv[2] && bv[1] < rhs.bv[1])
				|| (bv[1] == rhs.bv[1] && bv[0] < rhs.bv[0]);
	}

	bool prefix_subset_of(const bv192 & x, pos_t pp, pos_t p) const {
		//
		// Check that *this is a subset of x only up to position pos,
		// starting from the left (most significant) down to position
		// pos, including position pos, as illustrated below:
		//
		//   prefix checked      rest of the bits are ignored
		// |######----------#####################################|
		//  ^191  ^pp      ^p                                  0^
		// 
		if(pp-p<3){
			while (pp>p){
				if(x[pp]==false && (*this)[pp])
					return false;
				pp--;
			}
			return true;
		}

		if (p > 191)			// p > 191
			return true;

		if (pp > 127) {
			if (p & 128)			// 192 > p > 127
				return (((bv[2] & x.bv[2]) ^ bv[2]) >> (p & 63)) == 0;

			if ((bv[2] & x.bv[2]) != bv[2])
				return false;
		}

		if (pp > 63) {
			if (p & 64) 			// 128 > p > 63
				return (((bv[1] & x.bv[1]) ^ bv[1]) >> (p & 63)) == 0;

			// 64 > p
			if ((bv[1] & x.bv[1]) != bv[1])
				return false;
		}

		return (((bv[0] & x.bv[0]) ^ bv[0]) >> p) == 0;
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
		if (p > 191)			// p > 191
			return true;

		if (p & 128)			// 192 > p > 127
			return (((bv[2] & x.bv[2]) ^ bv[2]) >> (p & 63)) == 0;

		if ((bv[2] & x.bv[2]) != bv[2])
			return false;

		if (p & 64) 			// 128 > p > 63
			return (((bv[1] & x.bv[1]) ^ bv[1]) >> (p & 63)) == 0;

								// 64 > p
		if ((bv[1] & x.bv[1]) != bv[1])
			return false;

		return (((bv[0] & x.bv[0]) ^ bv[0]) >> p) == 0;
	}

	bool subset_of(const bv192 & x) const {
		// return true iff *this is a subset of x
		return (bv[0] & x.bv[0]) == bv[0]
			&& (bv[1] & x.bv[1]) == bv[1]
			&& (bv[2] & x.bv[2]) == bv[2];
	}

	bool suffix_subset_of(const bv192 & x, pos_t p) const {
		// this is essentially a subset_of check with an advice that
		// all bits to the left of position p (i.e., positions > p)
		// can be skipped, because they have been already checked
		// earlier.  So, this is not really a suffix-subset match in
		// the sense analogous to prefix_subset_of above.  In other
		// words, if p < 64, we only need to check the rightmost block
		// bv[0], if p < 128 we only need to check bv[1] and bv[0],
		// otherwise we need to check all three blocks bv[2], bv[1],
		// and bv[0].
		return (bv[0] & x.bv[0]) == bv[0]
			&& (p < 64 || ((bv[1] & x.bv[1]) == bv[1] 
						   && (p < 128 || (bv[2] & x.bv[2]) == bv[2])));
	}
#if 0
	// Antonio's preferred implementation of suffix_subset_of
	// 
	bool suffix_subset_of(const bv192 & x, pos_t p) const {
		if ((bv[0] & x.bv[0]) != bv[0])
			return false;

		if (p < 64)
			return true;

		if ((bv[1] & x.bv[1]) != bv[1])
			return false;

		if (p < 128)
			return true;

		return (bv[2] & x.bv[2]) == bv[2];
	}
#endif
#if 0
	// we'll keep this code around just because Koorosh believes,
	// probably under the influence of some psychotropic substance,
	// that this is a better variant of suffix_subset_of, where it is
	// in fact functionally identical but more confusing in terms of
	// code structure.
	bool suffix_subset_of(const bv192 & x, pos_t p) const {
		if ((bv[0] & x.bv[0]) != bv[0])
			return false;

		if (p > 63) {
			if ((bv[1] & x.bv[1]) != bv[1])
				return false;
			if (p > 127) {
				if ((bv[2] & x.bv[2]) != bv[2])
					return false;
			}
		}
		return true;
	}
#endif

	void set(pos_t pos) {
		bv[pos/64] |= (1UL << (pos % 64));
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

	pos_t most_significant_one_pos() const {
		//
		// returns the index of the most significant 1-bit.  For
		// example, x="001110..." => result=189
		//
#ifdef HAVE_BUILTIN_CLZL
		if (bv[2])
			return 191 - __builtin_clzl(bv[2]);

		if (bv[1])
			return 127 - __builtin_clzl(bv[1]);

		return 63 - __builtin_clzl(bv[0]);
#else
		if (bv[2])
			return 128 + log2(bv[2]);

		if (bv[1])
			return 64 + log2(bv[1]);

		return log2(bv[0]);
#endif
	}
};

inline std::ostream & operator << (std::ostream & os, const bv192 & x) {
	return x.print(os);
}

#endif
