#ifndef BITVECTOR_HH_INCLUDED
#define BITVECTOR_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#include <cstddef>				// size_t

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

static const size_t BLOCK_SIZE = 64;
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
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = 0;

		assert(p.size() <= Size);

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

    const block_t * begin() const {
		return b;
    }

    const block_t * end() const {
		return b + BLOCK_COUNT;
    }

    const uint64_t * begin64() const {
		return b;
    }

    const uint64_t * end64() const {
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

	uint32_t uint32_value(unsigned int i) const {
		return (i % 2 == 1) ? (b[i/2] >> 32) : (b[i/2] & 0xffffffffULL);
	}

	void copy_into_uint32_array(uint32_t * x) const {
		for(int i = 0; i < BLOCK_SIZE; ++i) {
			*x++ = b[i] & 0xffffffff;
			*x++ = b[i] >> 32;
		}
	}

    void assign(const block_t * p) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = p[i];
    }

    bitvector & operator = (const bitvector & p) {
		assign(p.b);
		return *this;
    }

	void clear() {
		for(int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = 0;
	}

	void set_bit(unsigned int pos) {
		b[pos/BLOCK_SIZE] |= (BLOCK_ONE << (pos % BLOCK_SIZE));
	}

	bitvector & operator |= (const bitvector & x) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] |= x.b[i];
		return *this;
	}

	bitvector & operator &= (const bitvector & x) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] &= x.b[i];
		return *this;
	}

	bitvector & operator ^= (const bitvector & x) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] ^= x.b[i];
		return *this;
	}

    bool subset_of(const block_t * p) const {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			if ((b[i] & ~p[i]) != 0)
				return false;

		return true;
    }
};

//
// leftmost 1-bit position in a block
//
#ifdef HAVE_BUILTIN_CTZL
inline size_t leftmost_bit(const block_t x) noexcept {
    // Since we represent the leftmost bit in the least-significant
    // position, the leftmost bit corresponds to the count of trailing
    // zeroes (see the layout specification above).
    return __builtin_ctzl(x);
} 
#else
inline size_t leftmost_bit(block_t x) noexcept {
    size_t n = 0;
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

#endif // BITVECTOR_HH_INCLUDED
