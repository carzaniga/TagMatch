#ifndef BITVECTOR_HH_INCLUDED
#define BITVECTOR_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <climits>
#include <cstdint>
#include <cctype>
#include <cstddef>				// size_t
#include <iostream>
#include <cassert>
#include <cstring>

//
// We represent bit vectors of sizes that are multiples of 64.  Thus
// we use 64-bit aligned strings of bits that we can represent and
// access as arrays of uint64_t or arrays of uint32_t values.
//
// BIT LAYOUT: See the documentation in the bitvector template below.
//

// 
// We use blocks of 64 bits to represent and process filters and their
// prefixes.  So, we are in trouble if the platform does not support
// 64-bit integers.
//
static_assert(sizeof(uint64_t)*CHAR_BIT == 64, "uint64_t must be a 64-bit word");

typedef uint64_t block_t;

//
// leftmost 1-bit position in a block
//
#ifdef HAVE_BUILTIN_CTZL
static inline int leftmost_bit(const block_t x) noexcept {
    // Since we represent the leftmost bit in the least-significant
    // position, the leftmost bit corresponds to the count of trailing
    // zeroes (see the layout specification below).
    return __builtin_ctzl(x);
} 
#else
static inline unsigned int leftmost_bit(block_t x) noexcept {
    unsigned int n = 0;
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
// leftmost 1-bit position in a block
//
#ifdef HAVE_BUILTIN_POPCOUNT
static inline int block_popcount(block_t v) noexcept {
	return __builtin_popcountl(v);
}
#else
static inline unsigned int block_popcount(block_t v) noexcept {
	// taken from http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
	v = v - ((v >> 1) & 0x5555555555555555);
	v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333);
	v = (v + (v >> 4)) & 0x0f0f0f0f0f0f0f0f;
	v = (v * 0x0101010101010101) >> 56;
	return v;
}
#endif

static const unsigned int BLOCK_SIZE = 64;
static const block_t BLOCK_ONE = 0x1;

//
// Main representation of a filter and a prefix.  Essentially we will
// instantiate this template with Size=64, Size=128, and Size=192.
//
template <unsigned int Size>
class bitvector {
    static_assert((Size % 64) == 0, "filter width must be a multiple of 64");

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
	static const unsigned int WIDTH = Size;

    bitvector(const std::string & p) {
		clear();
		// see the layout specification above
		//
		block_t mask = BLOCK_ONE;
		int i = 0;
		for(std::string::const_iterator c = p.begin(); c != p.end(); ++c) {
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

	bitvector() {};

    bitvector(const block_t * p) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = p[i];
    }

    bitvector(const bitvector & p) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = p.b[i];
    }

    const block_t * begin() const noexcept {
		return b;
    }

    const block_t * end() const noexcept {
		return b + BLOCK_COUNT;
    }

    const uint64_t * begin64() const noexcept {
		return b;
    }

    const uint64_t * end64() const noexcept {
		return b + BLOCK_COUNT;
    }

	// WARNING: we want to be able to access the bit pattern of the
	// prefix/filter as an array of 32-bit integer.  In this case we
	// simply look at the array of 64-bit unsigned integer as an array
	// of 32-bit integer.  This, as far as I understand, is undefined
	// behavior?
	// 
	const uint32_t * unsafe_begin32() const {
		return reinterpret_cast<const uint32_t *>(b);
	}

	const uint32_t * unsafe_end32() const {
		return reinterpret_cast<const uint32_t *>(b) + (Size / 32);
	}

	uint32_t uint32_value(unsigned int i) const noexcept {
		return (i % 2 == 1) ? (b[i/2] >> 32) : (b[i/2] & 0xffffffffULL);
	}

	void copy_into_uint32_array(uint32_t * x) const noexcept {
		for(int i = 0; i < BLOCK_SIZE; ++i) {
			*x++ = b[i] & 0xffffffff;
			*x++ = b[i] >> 32;
		}
	}

    void assign(const block_t * p) noexcept {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = p[i];
    }

    bitvector & operator = (const bitvector & p) noexcept {
		assign(p.b);
		return *this;
    }

	void clear() noexcept {
		for(int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = 0;
	}

	void set_bit(unsigned int pos) noexcept {
		b[pos/BLOCK_SIZE] |= (BLOCK_ONE << (pos % BLOCK_SIZE));
	}

	bool operator [] (unsigned int pos) const noexcept {
		return (b[pos/BLOCK_SIZE] & (BLOCK_ONE << (pos % BLOCK_SIZE)));
	}

	bitvector & operator |= (const bitvector & x) noexcept {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] |= x.b[i];
		return *this;
	}

	bitvector & operator &= (const bitvector & x) noexcept {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] &= x.b[i];
		return *this;
	}

	bitvector & operator ^= (const bitvector & x) noexcept {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] ^= x.b[i];
		return *this;
	}

    bool operator == (const bitvector & x) const noexcept {
		return memcmp(b, x.b, sizeof(b)) == 0;
	}

    bool operator != (const bitvector & x) const noexcept {
		return memcmp(b, x.b, sizeof(b)) != 0;
	}

    bool subset_of(const block_t * p) const noexcept {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			if ((b[i] & ~p[i]) != 0)
				return false;

		return true;
    }

    bool subset_of(const bitvector & x) const noexcept {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			if ((b[i] & ~x.b[i]) != 0)
				return false;

		return true;
    }

	bool range_subset_of(const bitvector & x,
						 const unsigned int left,
						 const unsigned int right) const noexcept {
		//
		// Check that *this is a subset of x only in the range of
		// positions from left up to right - 1, as illustrated below:
		//
		//      range checked     rest of the bits are ignored
		// |######----------#####################################|
		//  ^0    ^left     ^right                           191^
		// 
		assert(left < right);
		unsigned int i = left / BLOCK_SIZE;
		block_t mask = ~(0x0);
		mask <<= (left % BLOCK_SIZE);
		if (i == right / BLOCK_SIZE) {
			return !((b[i] & ~x.b[i] & mask) << (BLOCK_SIZE - (right % BLOCK_SIZE)));
		} else if (b[i] & ~x.b[i] & mask)
			return false;

		for (++i; i < right/BLOCK_SIZE; ++i)
			if ((b[i] & ~x.b[i]) != 0)
				return false;

		if (right % BLOCK_SIZE)
			return !((b[i] & ~x.b[i]) << (BLOCK_SIZE - (right % BLOCK_SIZE)));
		return true;
	}

	bool prefix_subset_of(const bitvector & x, const unsigned int right) const noexcept {
		//
		// Check that *this is a subset of x only in the prefix up to
		// position right - 1 as illustrated below:
		//
		//   prefix checked      rest of the bits are ignored
		// |----------------#####################################|
		//  ^0              ^right                           191^
		// 
		assert(right <= WIDTH);

		unsigned int i;
		for (i = 0; i < right/BLOCK_SIZE; ++i)
			if ((b[i] & ~x.b[i]) != 0)
				return false;

		if (right % BLOCK_SIZE)
			return !((b[i] & ~x.b[i]) << (BLOCK_SIZE - (right % BLOCK_SIZE)));
		return true;
	}

	bool suffix_subset_of(const bitvector & x, const unsigned int left) const noexcept {
		//
		// Check that *this is a subset of x only in the suffix starting at
		// position left, as illustrated below:
		//
		//   ignored prefix           checked suffix
		// |################-------------------------------------|
		//  ^0              ^left                            191^
		//
		assert(left < WIDTH);
		unsigned int i = left/BLOCK_SIZE;

		if (((b[i] & ~x.b[i]) >> (left % BLOCK_SIZE)) != 0)
			return false;

		while (++i < BLOCK_COUNT)
			if ((b[i] & ~x.b[i]) != 0)
				return false;

		return true;
	}
	
	bool prefix_equal(const bitvector & x, const unsigned int right) const noexcept {
		//
		// Check that *this and x share the same prefix up to position
		// right - 1, as illustrated below:
		//
		//   prefix checked      rest of the bits are ignored
		// |----------------#####################################|
		//  ^0              ^right                           191^
		// 
		assert(right <= WIDTH);
		for(unsigned int i = 0; i*BLOCK_SIZE < right; ++i) {
			if (right <= i*BLOCK_SIZE + BLOCK_SIZE) {
				return (((b[i] ^ x.b[i]) << (i*BLOCK_SIZE + BLOCK_SIZE - right)) == 0);
			} else if (b[i] != x.b[i])
				return false;
		}
		return true;
	}
	
    bool operator < (const bitvector & x) const noexcept {
		for (int i = 0; i < BLOCK_COUNT; ++i) {
			if (b[i] != x.b[i]) {
				unsigned int pos = leftmost_bit(b[i] ^ x.b[i]);
				return ((BLOCK_ONE << pos) & x.b[i]);
			}
		}
		return false;
    }

    bool operator > (const bitvector & x) const noexcept {
		for (int i = 0; i < BLOCK_COUNT; ++i) {
			if (b[i] != x.b[i]) {
				unsigned int pos = leftmost_bit(b[i] ^ x.b[i]);
				return ((BLOCK_ONE << pos) & b[i]);
			}
		}
		return false;
    }

	/** returns the position of the leftmost bit that differs between
	 *  this bitvector and x, or WIDTH if *this == x.
	 */
    unsigned int leftmost_diff (const bitvector & x) const noexcept {
		for (int i = 0; i < BLOCK_COUNT; ++i) {
			if (b[i] != x.b[i]) 
				return i*BLOCK_SIZE + leftmost_bit(b[i] ^ x.b[i]);
		}
		return WIDTH;
    }

    static unsigned int leftmost_diff (const bitvector & x, const bitvector & y) noexcept {
		return x.leftmost_bit(y);
	}

	/** iterate through the bits
	 * 
	 *  return the first bit position set to 1 starting from pos.
	 *  That is, returns the position of the leftmost 1-bit that is
	 *  greater or equal to pos. Return WIDTH when no bits are set to
	 *  1 (from pos).
	 *
	 *  Example:
	 *  <code>
	 *  bitvector<192> bv;
	 *  bv.read_ascii(std::cin);
	 *  
	 *  for(unsigned int i = next_bit(0); i < bv.WIDTH; i = next_bit(i + 1))
	 *      std::cout << ' ' << i;
	 *  std::cout << std::endl;
	 */
	unsigned int next_bit(unsigned int pos) const noexcept {
		unsigned int i = pos / BLOCK_SIZE; 
		if (i < BLOCK_COUNT) {
			block_t B = b[i];
			pos = pos % BLOCK_SIZE;

			B = ((B >> pos) << pos); // clear the first pos bits

			if (B != 0)
				return leftmost_bit(B) + i*BLOCK_SIZE;

			while(++i < BLOCK_COUNT)
				if (b[i] != 0)
					return leftmost_bit(b[i]) + i*BLOCK_SIZE;
		}		 
		return Size;
	}

	unsigned int popcount() const noexcept {
		unsigned int result = 0;
		for (int i = 0; i < BLOCK_COUNT; ++i) 
			result += block_popcount(b[i]);
		return result;
	}

	unsigned int prefix_popcount(const unsigned int right) const noexcept {
		//
		// Computes the popcount up to position right - 1, as
		// illustrated below:
		//
		//   prefix checked      rest of the bits are ignored
		// |----------------#####################################|
		//  ^0              ^right                           191^
		//
		unsigned int result = 0;
		unsigned int i;
		for(i = 0; i < right/BLOCK_SIZE; ++i)
			result += block_popcount(b[i]);

		if (right % BLOCK_SIZE)
			result += block_popcount(b[i] << (BLOCK_SIZE - (right % BLOCK_SIZE)));

		return result;
	}

	unsigned int suffix_popcount(const unsigned int left) const noexcept {
		//
		// Compute the popcunt starting at position left, as
		// illustrated below:
		//
		//   ignored prefix           checked suffix
		// |################-------------------------------------|
		//  ^0              ^left                            191^
		//
		if (left < WIDTH) {
			unsigned int i = left/BLOCK_SIZE;
			unsigned int result = block_popcount(b[i] >> (left % BLOCK_SIZE));

			while (++i < BLOCK_COUNT)
				result += block_popcount(b[i]);

			return result;
		} else
			return 0;
	}

	std::ostream & write_binary(std::ostream & output) const {
#ifdef WORDS_BIGENDIAN
		unsigned char tmp[sizeof(b)];
		unsigned char * cp = tmp;
		for (int i = 0; i < BLOCK_COUNT; ++i) {
			block_t b = b[i];
			for(int j = 0; j < sizeof(block_t); ++j) {
				*cp = (b & 0xff);
				b >>= CHAR_BIT;
				++cp;
			}
		}
		return output.write(tmp, sizeof(tmp));
#else
		return output.write(reinterpret_cast<const char *>(b), sizeof(b));
#endif
	}

	std::istream & read_binary(std::istream & input) {
#ifdef WORDS_BIGENDIAN
		unsigned char tmp[sizeof(b)];
		if (input.read(tmp, sizeof(tmp))) {
			const unsigned char * cp = tmp;
			for (int i = 0; i < BLOCK_COUNT; ++i) {
				b[i] = 0;
				for(int j = 0; j < sizeof(block_t); ++j) {
					b[i] = (b[i] << CHAR_BIT) | *cp;
					++cp;
				}
			}
		}
		return input;
#else
		return input.read(reinterpret_cast<char *>(b), sizeof(b));
#endif
	}

	std::ostream & write_ascii(std::ostream & output) const {
		for (int i = 0; i < BLOCK_COUNT; ++i) 
			for(block_t mask = BLOCK_ONE; mask != 0; mask <<= 1)
				output << ((b[i] & mask) ? '1' : '0');
		return output;
	}

	std::ostream & write_ascii(std::ostream & output, uint8_t prefix_len) const {
		for (int i = 0; i < BLOCK_COUNT; ++i) 
			for(block_t mask = BLOCK_ONE; mask != 0; mask <<= 1) {
				if (prefix_len == 0) {
					return output;
				} else {
					--prefix_len;
					output << ((b[i] & mask) ? '1' : '0');
				}
			}
		return output;
	}

	std::istream & read_ascii(std::istream & input) {
		while(isspace(input.peek()))
			input.get();
		for (int i = 0; i < BLOCK_COUNT; ++i) {
			b[i] = 0;
			for(block_t mask = BLOCK_ONE; mask != 0; mask <<= 1) {
				switch (input.get()) {
				case '1': b[i] |= mask;
				case '0': break;
				default: 
					while(++i < BLOCK_COUNT)
						b[i] = 0;
					return input;
				}
			}
		}
		return input;
	}
};

#endif // BITVECTOR_HH_INCLUDED
